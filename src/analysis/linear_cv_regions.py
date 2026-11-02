"""Region occlusion: does a target depend on the nucleus, the cytoplasm or the rest of the image?

Follow-up to linear_cv_attribution.py, where single-patch attributions disagreed
across methods and occlusion depended on its fill. Same 66 cells (6 figure, 60
evaluation) and the same 43 targets (each metric's best and worst position, its
all-position readout, 10 random positions). Regions come from the morphometry
segmenter (src/segment.py): per pixel, the majority label over its 25 seeds,
predicted at the encoder's 224 x 224 input (ties go to the lower label:
background, then nucleus), keeping the component at the image centre as
segment.predict does:

    nucleus     label 1
    cytoplasm   label 2
    rest        label 0: background, neighbouring cells and red cells

For each cell, region and fill, the effect is T(original) - T(region filled),
with the two fills of linear_cv_attribution.py: grey (0 in normalized units) and
the cell's own background (per channel, the median of the border ring). The
share of a region is |effect| over the sum of the three regions' |effects|.

Stability: per region, the Pearson and Spearman r between the two fills'
effects over all (evaluation cell, target) pairs. These are not comparable to
the patch-level fill agreement (a correlation over 256 patches within one map).

Checks: the cosine guard on the recomputed embeddings, as before, and all 25
segmenter checkpoints present.

No network access.
"""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr

from src.analysis.linear_cv_gradcam import MIN_COS
from src.analysis.raw_probe import DATASET, RESULTS, load_embeddings
from src.embed import ENCODERS, INTERIM, SIZE, WEIGHTS, CropDataset
from src.env import go_offline
from src.segment import _centre_component, load_model, predict

go_offline()

N_SEG_SEEDS = 25
REGIONS = {"nucleus": 1, "cytoplasm": 2, "rest": 0}


def region_masks(images: CropDataset) -> list[np.ndarray]:
    """Per-pixel majority label over the segmenter seeds at the encoder's input size, centre component kept."""
    pil = [Image.open(images._source(k)).convert("RGB") for k in range(len(images))]
    votes = np.zeros((len(REGIONS), len(images), SIZE, SIZE), dtype=np.int16)
    for seed in range(N_SEG_SEEDS):
        for k, label in enumerate(predict(load_model(seed=seed, device="cuda"), pil, out_size=SIZE)):
            for c in range(len(REGIONS)):
                votes[c, k] += label == c
    return [_centre_component(m.astype(np.uint8)) for m in votes.argmax(axis=0)]


def occluded(image: torch.Tensor, mask: np.ndarray) -> tuple[torch.Tensor, list[str], dict]:
    """The image, then each region filled with each fill (fills outer, regions inner); fill names; region masks."""
    border = torch.cat([image[:, 0], image[:, -1], image[:, :, 0], image[:, :, -1]], dim=1)
    fills = {"grey": torch.zeros(3, device=image.device), "background": border.median(dim=1).values}
    region_px = {name: torch.from_numpy(mask == label).to(image.device) for name, label in REGIONS.items()}
    batch = [image]
    for fill in fills.values():
        for m in region_px.values():
            filled = image.clone()
            filled[:, m] = fill[:, None]
            batch.append(filled)
    return torch.stack(batch), list(fills), region_px


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    args = ap.parse_args()
    device = torch.device("cuda")

    prev = json.loads((RESULTS / f"linear_cv_attribution_{args.encoder}_{DATASET}.json").read_text())
    figure, evaluation = prev["cells"]["figure"], prev["cells"]["evaluation"]
    cells = figure + evaluation
    meta, X = load_embeddings(args.encoder)
    index = pd.read_parquet(INTERIM / DATASET / "index.parquet").set_index("cell_id").loc[cells]
    stored = X[meta.set_index("cell_id").loc[cells, "row"].to_numpy()].astype(np.float64)

    # The targets of linear_cv_gradcam.py, in its order (as in linear_cv_attribution.py).
    keys = ["kind", "descriptor", "which", "dim"]
    gradcam = pd.read_parquet(RESULTS / f"linear_cv_gradcam_{args.encoder}_{DATASET}.parquet")
    targets = gradcam[(gradcam.cell_id == figure[0]) & (gradcam.method == "gradcam")][keys].drop_duplicates()
    w = pd.read_parquet(RESULTS / f"linear_cv_weights_{args.encoder}_{DATASET}.parquet")
    joint = w[w.model == "joint_768"].groupby(["descriptor", "dim"]).w.mean()
    n_dims = X.shape[1]
    C = np.stack([joint.loc[t.descriptor].to_numpy() if t.kind == "readout" else np.eye(n_dims)[t.dim]
                  for t in targets.itertuples()])
    C = torch.tensor(C, dtype=torch.float32, device=device)

    images = CropDataset(index.reset_index(), DATASET, ENCODERS[args.encoder].interpolation)
    masks = region_masks(images)

    spec = ENCODERS[args.encoder]
    model = spec.build(WEIGHTS / spec.local_dir).to(device).eval()
    rows, cos, pixels = [], [], []
    for k, cell in enumerate(cells):
        image, ok = images[k]
        if not ok:
            raise RuntimeError(f"{cell}: crop did not decode")
        batch, fills, region_px = occluded(image.to(device), masks[k])
        pixels.append({"cell_id": cell, **{name: int(m.sum()) for name, m in region_px.items()}})
        with torch.no_grad():
            emb = spec.forward(model, batch)
        e = emb[0].double().cpu().numpy()
        cos.append(float(e @ stored[k] / np.linalg.norm(e) / np.linalg.norm(stored[k])))
        T = (emb @ C.T).cpu().numpy()  # (1 + fills x regions) x targets

        part = "figure" if cell in figure else "evaluation"
        for f, fill in enumerate(fills):
            for g, (region, m) in enumerate(region_px.items()):
                effect = T[0] - T[1 + f * len(REGIONS) + g]
                rows.append(pd.DataFrame({"cell_id": cell, "set": part, "class_label": index.loc[cell, "class_label"],
                                          **targets.reset_index(drop=True), "fill": fill, "region": region,
                                          "pixels": int(m.sum()), "t_original": T[0],
                                          "effect": effect if m.any() else np.nan}))
        print(f"[linear_cv_regions] {k + 1}/{len(cells)} {cell}: cos {cos[-1]:.6f}, pixels {pixels[-1]}", flush=True)

    if min(cos) < MIN_COS:
        raise RuntimeError(f"recomputed embedding differs from the stored one (cos {min(cos):.4f})")

    rows = pd.concat(rows, ignore_index=True)
    total = rows.groupby(["cell_id", *keys, "fill"]).effect.transform(lambda s: s.abs().sum())
    rows["share"] = rows.effect.abs() / total
    ev = rows[rows.set == "evaluation"]
    wide = ev.pivot_table(index=["cell_id", *keys, "region"], columns="fill", values="effect").dropna()
    stability = {region: {"pearson": float(np.corrcoef(g.grey, g.background)[0, 1]),
                          "spearman": float(spearmanr(g.grey, g.background)[0]), "n_pairs": len(g)}
                 for region, g in wide.groupby(level="region")}
    shares = (ev.assign(group=ev.kind + ev.which.map(lambda s: f" {s}" if s else ""))
              .groupby(["fill", "group", "region"]).share.median().unstack("region"))
    px = pd.DataFrame(pixels).set_index("cell_id")
    summary = {
        "analysis": "region occlusion (nucleus, cytoplasm, rest) of embedding positions and metric readouts",
        "encoder": args.encoder, "dataset": DATASET, "test_read": False,
        "cells": "the figure and evaluation cells of linear_cv_attribution.py",
        "targets": "those of linear_cv_gradcam.py",
        "masks": f"per-pixel majority label over segmenter seeds 0-{N_SEG_SEEDS - 1} (src/segment.py), "
                 f"{SIZE} x {SIZE}, ties to the lower label, centre component kept",
        "fills": {"grey": "0 in normalized units", "background": "per-channel median of the border ring"},
        "region_pixels": {"min": px.min().to_dict(), "median": px.median().to_dict(),
                          "cells_with_an_empty_region": int((px == 0).any(axis=1).sum())},
        "stability_between_fills_evaluation": stability,
        "median_share_by_group_evaluation": json.loads(shares.reset_index().to_json(orient="records")),
        "checks": {"embedding_cos_min": min(cos), "segmenter_seeds": N_SEG_SEEDS},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__},
        "device": torch.cuda.get_device_name(0), "host": platform.node(),
    }
    stem = f"linear_cv_regions_{args.encoder}_{DATASET}"
    rows.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"[linear_cv_regions] fill stability (evaluation): {json.dumps(stability)}")
    print(f"[linear_cv_regions] median share of |effect| by region:\n{shares.round(3)}")
    print(f"[linear_cv_regions] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
