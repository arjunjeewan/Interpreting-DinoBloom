"""Grad-CAM and occlusion maps for embedding positions and for each metric's linear readout.

Companion to linear_cv_weights.py. On one example cell per stage of the
granulocytic series (BLA, PMO, MYB, MMZ, NGB, NGS; a seeded-random labelled cell
each, from the CV portion, never the 20% hold-out), it asks which image patches
drive a scalar target T computed from the DinoBloom embedding x:

    position   T = x_j, for each metric's best and worst position by |r|
               (linear_cv_weights single beta, mean over folds)
    readout    T = w . x, with each metric's all-position OLS weights
               (joint_768, mean over folds)
    random     T = x_j for 10 seeded-random positions, a control

The readout map does not depend on the weight convention: shifting w along the
LayerNorm null direction adds t * sum_j (x_j - shift_j) / scale_j to T, which is
zero for every input, so its gradient is zero too.

Grad-CAM, signed (no ReLU): A holds the 256 patch tokens entering the last block
(the output of blocks[-1].norm1); with g = dT/dA, map_p = sum_c mean_p'(g[p', c])
A[p, c], one value per 14 x 14 patch. Occlusion: each patch in turn is set to 0
in normalized units (the ImageNet mean colour), and the map is
T(original) - T(occluded). Agreement is the Pearson r between the two 256-patch
maps of the same cell and target. Specificity: the |r| between the Grad-CAM
maps of different positions on the same cell.

The model runs in fp32 on the GPU with src/embed.py's preprocessing; each
example cell's recomputed embedding is compared with the stored one (fp16
autocast), as a guard on the preprocessing, not as a result.

No network access.
"""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import pandas as pd
import torch

from src.analysis.linear_cv import SPLITS, seed_mean_labels
from src.analysis.raw_probe import DATASET, RESULTS, load_embeddings
from src.embed import ENCODERS, INTERIM, SIZE, WEIGHTS, CropDataset
from src.env import go_offline
from src.morphometry import DESCRIPTORS

go_offline()

SEED = 0
GRANULOCYTIC = ["BLA", "PMO", "MYB", "MMZ", "NGB", "NGS"]  # blast to segmented neutrophil
N_RANDOM = 10   # control positions
PATCH = 14
MIN_COS = 0.99  # recomputed vs stored embedding: a guard on the preprocessing


def gradcam(A: torch.Tensor, T: torch.Tensor) -> np.ndarray:
    """Signed Grad-CAM of each target in T over the patch tokens of A (1 x 257 x channels)."""
    out = []
    for t in range(len(T)):
        (g,) = torch.autograd.grad(T[t], A, retain_graph=True)
        out.append((A[0, 1:] @ g[0, 1:].mean(dim=0)).detach().cpu().numpy())
    return np.stack(out)  # targets x patches


def occlusion(model, image: torch.Tensor, C: torch.Tensor, grid: int) -> np.ndarray:
    """T(original) - T(patch p set to 0) for every patch p and every target (rows of C)."""
    with torch.no_grad():
        base = model(image) @ C.T  # 1 x targets
        occluded = image.repeat(grid * grid, 1, 1, 1)
        for p in range(grid * grid):
            r, c = divmod(p, grid)
            occluded[p, :, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 0
        out = torch.cat([model(b) for b in occluded.split(64)]) @ C.T  # patches x targets
    return (base - out).T.cpu().numpy()  # targets x patches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    args = ap.parse_args()
    device = torch.device("cuda")

    # Example cells: labelled, embedded, in the CV portion; one seeded-random cell per class.
    meta, X = load_embeddings(args.encoder)
    splits = pd.read_parquet(SPLITS)
    labelled = seed_mean_labels(meta).index
    eligible = meta[meta.cell_id.isin(splits.cell_id[splits.split == "cv"]) & meta.cell_id.isin(labelled)]
    rng = np.random.default_rng(SEED)
    cells = [rng.choice(np.sort(eligible.cell_id[eligible.class_label == c].to_numpy())) for c in GRANULOCYTIC]
    index = pd.read_parquet(INTERIM / DATASET / "index.parquet").set_index("cell_id").loc[cells]
    stored = X[meta.set_index("cell_id").loc[cells, "row"].to_numpy()].astype(np.float64)

    # Targets: rows of C, so that T = x @ C.T.
    w = pd.read_parquet(RESULTS / f"linear_cv_weights_{args.encoder}_{DATASET}.parquet")
    r = w[w.model == "single"].groupby(["descriptor", "dim"]).beta.mean()
    joint = w[w.model == "joint_768"].groupby(["descriptor", "dim"]).w.mean()
    n_dims = X.shape[1]
    targets, C = [], []
    for d in DESCRIPTORS:
        a = r.loc[d].abs()
        for which, j in (("best", int(a.idxmax())), ("worst", int(a.idxmin()))):
            targets.append({"kind": "position", "descriptor": d, "which": which, "dim": j})
            C.append(np.eye(n_dims)[j])
        targets.append({"kind": "readout", "descriptor": d, "which": "", "dim": -1})
        C.append(joint.loc[d].to_numpy())
    for j in rng.choice(n_dims, N_RANDOM, replace=False):
        targets.append({"kind": "random", "descriptor": "", "which": "", "dim": int(j)})
        C.append(np.eye(n_dims)[j])
    C = torch.tensor(np.stack(C), dtype=torch.float32, device=device)

    spec = ENCODERS[args.encoder]
    model = spec.build(WEIGHTS / spec.local_dir).to(device).eval()
    grid = SIZE // PATCH
    acts = {}
    model.blocks[-1].norm1.register_forward_hook(lambda m, i, o: acts.__setitem__("A", o))
    images = CropDataset(index.reset_index(), DATASET, spec.interpolation)

    rows, cos, max_diff, agree, specific = [], [], [], [], []
    for k, cell in enumerate(cells):
        image, ok = images[k]
        if not ok:
            raise RuntimeError(f"{cell}: crop did not decode")
        image = image[None].to(device)
        emb = spec.forward(model, image)
        e = emb[0].detach().double().cpu().numpy()
        cos.append(float(e @ stored[k] / np.linalg.norm(e) / np.linalg.norm(stored[k])))
        max_diff.append(float(np.abs(e - stored[k]).max()))
        maps = {"gradcam": gradcam(acts["A"], emb[0] @ C.T), "occlusion": occlusion(model, image, C, grid)}

        for t, target in enumerate(targets):
            agree.append({"kind": target["kind"],
                          "r": float(np.corrcoef(maps["gradcam"][t], maps["occlusion"][t])[0, 1])})
            for method, m in maps.items():
                rows.append(pd.DataFrame({"cell_id": cell, "class_label": GRANULOCYTIC[k], **target,
                                          "method": method, "patch_row": np.arange(grid * grid) // grid,
                                          "patch_col": np.arange(grid * grid) % grid, "value": m[t]}))
        # one map per distinct position: 161, say, is best for both nuclear area and IOD
        pos = list({target["dim"]: t for t, target in enumerate(targets) if target["kind"] != "readout"}.values())
        g = np.corrcoef(maps["gradcam"][pos])
        specific.append(float(np.median(np.abs(g[np.triu_indices(len(pos), 1)]))))
        print(f"[linear_cv_gradcam] {cell} ({GRANULOCYTIC[k]}): embedding cos {cos[-1]:.6f}, "
              f"median |r| between position maps {specific[-1]:.2f}", flush=True)

    if min(cos) < MIN_COS:
        raise RuntimeError(f"recomputed embedding differs from the stored one (cos {min(cos):.4f})")
    agree = pd.DataFrame(agree).groupby("kind").r.describe()[["count", "min", "50%", "max"]]
    rows = pd.concat(rows, ignore_index=True)
    summary = {
        "analysis": "signed Grad-CAM and occlusion maps of embedding positions and metric readouts",
        "encoder": args.encoder, "dataset": DATASET, "test_read": False,
        "cells": [{"cell_id": c, "class_label": k, "shard": index.loc[c, "shard"], "shard_row": int(index.loc[c, "row"])}
                  for c, k in zip(cells, GRANULOCYTIC)],
        "selection": f"one cell per class in {GRANULOCYTIC}, np.random.default_rng({SEED}).choice over the "
                     f"sorted labelled, embedded cells of the CV portion ({SPLITS.name})",
        "targets": "position: each descriptor's best and worst position by |fold-mean single beta|; "
                   "readout: fold-mean joint_768 weights; random: control positions",
        "gradcam": "signed; activations = blocks[-1].norm1 output, patch tokens only; channel weight = "
                   "mean gradient over patches",
        "occlusion": f"{PATCH}x{PATCH} patch set to 0 in normalized units; map = T(original) - T(occluded)",
        "checks": {"embedding_cos_min": min(cos), "embedding_max_abs_diff": max(max_diff),
                   "min_cos_guard": MIN_COS},
        "agreement_gradcam_vs_occlusion_r": json.loads(agree.to_json(orient="index")),
        "specificity_median_abs_r_between_position_maps": dict(zip(cells, specific)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__},
        "device": torch.cuda.get_device_name(0), "host": platform.node(),
    }
    stem = f"linear_cv_gradcam_{args.encoder}_{DATASET}"
    rows.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"[linear_cv_gradcam] agreement (Pearson r, Grad-CAM vs occlusion):\n{agree.round(3)}")
    print(f"[linear_cv_gradcam] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
