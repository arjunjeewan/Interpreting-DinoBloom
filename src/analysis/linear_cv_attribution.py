"""Which patch attribution agrees with occlusion? Grad-CAM and gradient x activation at every block.

Follow-up to linear_cv_gradcam.py, whose last-block Grad-CAM was uncorrelated
with occlusion (median r about 0). Same targets (each metric's best and worst
position, its all-position readout, 10 random positions; read from that run)
and the same model and preprocessing. Candidates, both signed, at each of the
12 blocks, with A = the patch tokens leaving that block's norm1 and g = dT/dA:

    gradcam   map_p = sum_c mean_p'(g[p', c]) A[p, c]
    gxa       map_p = sum_c g[p, c] A[p, c]

Reference: occlusion, T(original) - T(patch p filled), with two fills: grey (0
in normalized units, the ImageNet mean colour, as before) and the cell's own
background (per channel, the median of the 224 x 224 image's border ring). The
two fills' agreement with each other shows how stable the reference is.
Agreement is the Pearson r between two 256-patch maps of the same cell and target.

Cells: the method is compared on 10 cells per stage of the granulocytic series
(BLA, PMO, MYB, MMZ, NGB, NGS), drawn with seed 1 from the labelled, embedded
cells of the CV portion, excluding the 6 figure cells of linear_cv_gradcam.py,
which are scored too but kept apart (set = "figure").

Checks: every recomputed embedding matches the stored one (cosine guard, as in
linear_cv_gradcam.py), and block-11 Grad-CAM on the figure cells reproduces that
script's maps.

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
from src.analysis.linear_cv_gradcam import GRANULOCYTIC, MIN_COS, PATCH
from src.analysis.raw_probe import DATASET, RESULTS, load_embeddings
from src.embed import ENCODERS, INTERIM, SIZE, WEIGHTS, CropDataset
from src.env import go_offline

go_offline()

SEED = 1          # evaluation cells; the figure cells used seed 0
N_PER_CLASS = 10
RTOL_REPRO = 1e-3  # block-11 Grad-CAM against linear_cv_gradcam's maps, relative to each map's range


def occlusion(model, image: torch.Tensor, C: torch.Tensor, fill: torch.Tensor, grid: int) -> np.ndarray:
    """T(original) - T(patch p set to fill) for every patch p and every target (rows of C)."""
    with torch.no_grad():
        base = model(image) @ C.T
        occluded = image.repeat(grid * grid, 1, 1, 1)
        for p in range(grid * grid):
            r, c = divmod(p, grid)
            occluded[p, :, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = fill[:, None, None]
        out = torch.cat([model(b) for b in occluded.split(64)]) @ C.T
    return (base - out).T.cpu().numpy()  # targets x patches


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between matching rows of a and b (... x patches)."""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    args = ap.parse_args()
    device = torch.device("cuda")

    prev = json.loads((RESULTS / f"linear_cv_gradcam_{args.encoder}_{DATASET}.json").read_text())
    prev_maps = pd.read_parquet(RESULTS / f"linear_cv_gradcam_{args.encoder}_{DATASET}.parquet")
    figure = [c["cell_id"] for c in prev["cells"]]

    meta, X = load_embeddings(args.encoder)
    splits = pd.read_parquet(SPLITS)
    labelled = seed_mean_labels(meta).index
    eligible = meta[meta.cell_id.isin(splits.cell_id[splits.split == "cv"]) & meta.cell_id.isin(labelled)
                    & ~meta.cell_id.isin(figure)]
    rng = np.random.default_rng(SEED)
    evaluation = [c for k in GRANULOCYTIC for c in
                  rng.choice(np.sort(eligible.cell_id[eligible.class_label == k].to_numpy()), N_PER_CLASS,
                             replace=False)]
    cells = figure + evaluation
    index = pd.read_parquet(INTERIM / DATASET / "index.parquet").set_index("cell_id").loc[cells]
    stored = X[meta.set_index("cell_id").loc[cells, "row"].to_numpy()].astype(np.float64)

    # The targets of linear_cv_gradcam.py, in its order.
    keys = ["kind", "descriptor", "which", "dim"]
    targets = prev_maps[(prev_maps.cell_id == figure[0]) & (prev_maps.method == "gradcam")][keys].drop_duplicates()
    w = pd.read_parquet(RESULTS / f"linear_cv_weights_{args.encoder}_{DATASET}.parquet")
    joint = w[w.model == "joint_768"].groupby(["descriptor", "dim"]).w.mean()
    n_dims = X.shape[1]
    C = np.stack([joint.loc[t.descriptor].to_numpy() if t.kind == "readout" else np.eye(n_dims)[t.dim]
                  for t in targets.itertuples()])
    C = torch.tensor(C, dtype=torch.float32, device=device)

    spec = ENCODERS[args.encoder]
    model = spec.build(WEIGHTS / spec.local_dir).to(device).eval()
    n_blocks, grid = len(model.blocks), SIZE // PATCH
    acts = [None] * n_blocks
    for b, block in enumerate(model.blocks):
        block.norm1.register_forward_hook(lambda m, i, o, b=b: acts.__setitem__(b, o))
    images = CropDataset(index.reset_index(), DATASET, spec.interpolation)

    rows, cos, repro = [], [], 0.0
    for k, cell in enumerate(cells):
        image, ok = images[k]
        if not ok:
            raise RuntimeError(f"{cell}: crop did not decode")
        image = image[None].to(device)
        emb = spec.forward(model, image)
        e = emb[0].detach().double().cpu().numpy()
        cos.append(float(e @ stored[k] / np.linalg.norm(e) / np.linalg.norm(stored[k])))
        T = emb[0] @ C.T

        cand = np.empty((2, n_blocks, len(T), grid * grid))  # method x block x target x patch
        for t in range(len(T)):
            grads = torch.autograd.grad(T[t], acts, retain_graph=True)
            for b, (A, g) in enumerate(zip(acts, grads)):
                A, g = A[0, 1:], g[0, 1:]
                cand[0, b, t] = (A @ g.mean(dim=0)).detach().cpu().numpy()
                cand[1, b, t] = (A * g).sum(dim=1).detach().cpu().numpy()

        border = torch.cat([image[0, :, 0], image[0, :, -1], image[0, :, :, 0], image[0, :, :, -1]], dim=1)
        fills = {"grey": torch.zeros(3, device=device), "background": border.median(dim=1).values}
        occ = {name: occlusion(model, image, C, fill, grid) for name, fill in fills.items()}

        part = "figure" if cell in figure else "evaluation"
        for name, o in occ.items():
            r = corr_rows(cand, o[None, None])  # method x block x target
            for m, method in enumerate(("gradcam", "gxa")):
                for b in range(n_blocks):
                    rows.append(pd.DataFrame({"cell_id": cell, "set": part, **targets.reset_index(drop=True),
                                              "method": method, "block": b, "fill": name, "r": r[m, b]}))
        rows.append(pd.DataFrame({"cell_id": cell, "set": part, **targets.reset_index(drop=True),
                                  "method": "occlusion_other_fill", "block": -1, "fill": "grey",
                                  "r": corr_rows(occ["grey"], occ["background"])}))
        if part == "figure":
            old = prev_maps[(prev_maps.cell_id == cell) & (prev_maps.method == "gradcam")]
            old = np.stack([g.sort_values(["patch_row", "patch_col"]).value.to_numpy()
                            for _, g in old.groupby(keys, sort=False)])
            repro = max(repro, float((np.abs(old - cand[0, -1]).max(axis=1) / np.ptp(old, axis=1)).max()))
        print(f"[linear_cv_attribution] {k + 1}/{len(cells)} {cell}: cos {cos[-1]:.6f}", flush=True)

    if min(cos) < MIN_COS:
        raise RuntimeError(f"recomputed embedding differs from the stored one (cos {min(cos):.4f})")
    if repro > RTOL_REPRO:
        raise RuntimeError(f"block-11 Grad-CAM does not reproduce linear_cv_gradcam's maps ({repro:.1e})")

    rows = pd.concat(rows, ignore_index=True)
    ev = rows[rows.set == "evaluation"]
    table = (ev[ev.method != "occlusion_other_fill"].groupby(["method", "block", "fill"]).r.median()
             .unstack("fill"))
    summary = {
        "analysis": "agreement of Grad-CAM and gradient x activation (each block) with occlusion (two fills)",
        "encoder": args.encoder, "dataset": DATASET, "test_read": False,
        "cells": {"figure": figure, "evaluation": evaluation,
                  "selection": f"{N_PER_CLASS} per class in {GRANULOCYTIC}, np.random.default_rng({SEED}) over "
                               f"the sorted labelled, embedded cells of the CV portion, figure cells excluded"},
        "targets": "those of linear_cv_gradcam.py",
        "activations": "patch tokens leaving each block's norm1",
        "occlusion_fills": {"grey": "0 in normalized units", "background": "per-channel median of the border ring"},
        "median_r_by_method_block_fill_evaluation": table.reset_index().to_dict(orient="records"),
        "median_r_between_fills_evaluation": float(ev[ev.method == "occlusion_other_fill"].r.median()),
        "checks": {"embedding_cos_min": min(cos), "block11_gradcam_vs_linear_cv_gradcam_rel": repro},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__},
        "device": torch.cuda.get_device_name(0), "host": platform.node(),
    }
    stem = f"linear_cv_attribution_{args.encoder}_{DATASET}"
    rows.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"[linear_cv_attribution] median r with occlusion, evaluation cells:\n{table.round(3)}")
    print(f"[linear_cv_attribution] median r between the two fills: {summary['median_r_between_fills_evaluation']:.3f}")
    print(f"[linear_cv_attribution] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
