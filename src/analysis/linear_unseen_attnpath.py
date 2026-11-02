"""linear_cv_attnpath.py's decomposition on PBC, read by linear_unseen.py's readouts (protocol.md §16).

PBC is not in DinoBloom's pretraining. Every labelled cell of each scored fold is
decomposed: val, and the test fold once under --read-test. Each cell is read by
all five linear_unseen readouts, none of which saw it, and every share is the
mean +/- sd over the five. The terms, regions, partitions, statistic and checks
are linear_cv_attnpath.py's own (batch_terms). Crops are resized without padding
(CropDataset pad=False), as the cached embeddings that the readouts were fit on
were computed; the cosine guard against those embeddings checks it, and the
U-Net masks are predicted on the same unpadded geometry.

measured_region_leads lists the readouts whose measured region carries the
largest of the three region shares, as figures.linear_cv_attnpath marks them
with a dot; N:C, measured in both, counts only if nucleus and cytoplasm both beat
the rest. Four settings: pixel split or interior patches only, over all cells or
within cell type. It is counted on the mean shares over the five readouts.

Maps: for one cell per cell type, drawn at random (seed SEED) before any map is
viewed, each patch token's direct-attention write to each readout summed over the
12 blocks and averaged over the five readouts (the map of the mean readout, as
the write is linear in the weights), beside the patch's region fractions, the
same write averaged over all scored cells (write_mean; the variance shares are
built on writes centred this way), the cell's readout and label, and the
readout's sd across the scored cells.

--read-test as in linear_unseen.py. fp32 throughout, no autocast. No network
access.
"""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.analysis.linear_cv import K_FOLDS, seed_mean_labels
from src.analysis.linear_cv_attnpath import (BATCH, GRID, GROUPS, GROUPS_INTERIOR, INTERIOR, KINDS, MASK_CHUNK,
                                             MEASURED_IN, RTOL_SPLIT, RTOL_SUM, SEED, SHOWN_BLOCKS, TOP_NORM,
                                             batch_terms, centre, patch_fractions, shares, table)
from src.analysis.linear_cv_gradcam import MIN_COS
from src.analysis.linear_cv_regions import REGIONS, region_masks
from src.analysis.linear_unseen import EXCLUDED, TEST_FOLD, frozen_commit
from src.analysis.raw_probe import RESULTS, load_embeddings
from src.embed import ENCODERS, INTERIM, WEIGHTS, CropDataset
from src.env import go_offline
from src.morphometry import DESCRIPTORS

go_offline()

DATASET = "pbc"
L = 12
SETTINGS = {"pixel": ("shares_all_blocks", list(REGIONS)),
            "interior": ("shares_all_blocks_interior_split", INTERIOR)}


def readout_result(T: np.ndarray, const: np.ndarray, Y: np.ndarray, cell_type: np.ndarray) -> dict:
    """linear_cv_attnpath.py's shares for one readout: T (cells, blocks, KINDS, R), const and Y (cells, R)."""
    full = np.concatenate([const[:, None], T[:, :, :5].sum(1)], 1)
    side = T[:, :, 5:7].sum(1)
    inner = np.concatenate([const[:, None], T[:, :, [KINDS.index(k) for k in GROUPS_INTERIOR[1:]]].sum(1)], 1)
    shown = T[:, [b - 1 for b in SHOWN_BLOCKS], :5]
    per_block = np.stack([T[:, :, :4].sum(2), T[:, :, 4]], 2)
    scopes = {"all_cells": (full, side, inner, shown, per_block, Y)}
    scopes["within_type"] = tuple(centre(a, cell_type) for a in scopes["all_cells"])
    result = {}
    for scope, (f, s, i, b, p, y) in scopes.items():
        sb, sp = shares(b, y), shares(p, y)
        result[scope] = {
            "shares_all_blocks": table(shares(f, y), shares(s, y)).to_dict(orient="index"),
            "shares_all_blocks_interior_split": pd.DataFrame(shares(i, y).T, index=DESCRIPTORS,
                                                             columns=GROUPS_INTERIOR).to_dict(orient="index"),
            "shares_by_shown_block": {f"block {blk}": pd.DataFrame(sb[n].T, index=DESCRIPTORS, columns=KINDS[:5])
                                      .to_dict(orient="index") for n, blk in enumerate(SHOWN_BLOCKS)},
            "attention_and_mlp_by_block": {kind: pd.DataFrame(sp[:, n].T, index=DESCRIPTORS,
                                                              columns=[f"block {l + 1}" for l in range(L)])
                                           .to_dict(orient="index") for n, kind in enumerate(["attention", "mlp"])}}
    return result


def mean_sd(results: list):
    """Leaf-wise mean and sd (ddof=1) over the readouts' nested result dicts."""
    if isinstance(results[0], dict):
        return {key: mean_sd([r[key] for r in results]) for key in results[0]}
    v = np.asarray(results, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}


def means(agg: dict):
    """The means of a mean_sd tree, in the shape of one readout's result."""
    return agg["mean"] if set(agg) == {"mean", "sd"} else {key: means(v) for key, v in agg.items()}


def leads(row: dict, cols: list[str], measured: str) -> bool:
    """Does the measured region carry the largest of the three region shares? N:C: both beat the rest."""
    v = dict(zip(REGIONS, (row[c] for c in cols)))
    if measured == "both":
        return v["nucleus"] > v["rest"] and v["cytoplasm"] > v["rest"]
    return max(v, key=v.get) == measured


def measured_region_leads(result: dict) -> dict:
    """Per setting, the readouts whose measured region leads; then those that lead in all four."""
    out = {f"{split}_{scope}": [d for d in DESCRIPTORS if leads(result[scope][key][d], cols, MEASURED_IN[d])]
           for split, (key, cols) in SETTINGS.items() for scope in ("all_cells", "within_type")}
    out["all_four"] = [d for d in DESCRIPTORS if all(d in v for v in out.values())]
    return out


def decompose(sample: pd.DataFrame, model, spec, W: np.ndarray, device) -> tuple:
    """batch_terms over every cell of the sample, each read by all readouts in W (C, readouts)."""
    n, R = len(sample), W.shape[1]
    index = pd.read_parquet(INTERIM / DATASET / "index.parquet").set_index("cell_id").loc[sample.cell_id].reset_index()
    frac = np.zeros((n, len(REGIONS), GRID * GRID))
    for s in range(0, n, MASK_CHUNK):
        chunk = CropDataset(index.iloc[s:s + MASK_CHUNK], DATASET, spec.interpolation)
        for i, m in enumerate(region_masks(chunk)):
            frac[s + i] = patch_fractions(m)
        print(f"[linear_unseen_attnpath] masks {min(s + MASK_CHUNK, n)}/{n}", flush=True)
    boundary = frac.max(axis=1) < 1

    Wt = torch.tensor(W, dtype=torch.float32, device=device)
    loader = DataLoader(CropDataset(index, DATASET, spec.interpolation, pad=False),   # as the cached embeddings
                        batch_size=BATCH, num_workers=4)
    T, const, Y, Z = np.zeros((n, L, len(KINDS), R)), np.zeros((n, R)), np.zeros((n, R)), np.zeros((n, W.shape[0]))
    patches, norm_stats, err, i0 = np.zeros((n, GRID * GRID, R), np.float32), np.zeros((n, L, 4)), {}, 0
    with torch.no_grad():
        for img, ok in loader:
            if not ok.all():
                raise RuntimeError("a crop did not decode")
            sl = slice(i0, i0 + len(img))
            out = batch_terms(model, img.to(device), Wt.expand(len(img), *Wt.shape),
                              torch.tensor(frac[sl], dtype=torch.float32, device=device),
                              torch.tensor(boundary[sl], device=device))
            T[sl], const[sl], Y[sl], Z[sl], norm_stats[sl] = (o.double().cpu().numpy() for o in out[:5])
            patches[sl] = out[6].float().cpu().numpy()
            err = {key: max(err.get(key, 0.0), v) for key, v in out[5].items()}
            i0 = sl.stop
            print(f"[linear_unseen_attnpath] {i0}/{n} cells", flush=True)
    return T, const, Y, Z, patches, frac, boundary, norm_stats, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    ap.add_argument("--read-test", action="store_true", help="also decompose the test fold, once")
    args = ap.parse_args()
    stem = f"linear_unseen_attnpath_{args.encoder}_{DATASET}"
    commit = None
    if args.read_test:                                       # before any data is loaded
        commit = frozen_commit()
        prev = RESULTS / f"{stem}.json"
        if prev.exists() and json.loads(prev.read_text())["test_read"]:
            raise RuntimeError(f"{prev.name} already holds a test run; the test fold is read once")
    device, R = torch.device("cuda"), len(DESCRIPTORS)

    meta, X = load_embeddings(args.encoder, DATASET)
    meta = meta[~meta.class_label.isin(EXCLUDED[DATASET])].reset_index(drop=True)
    labels = seed_mean_labels(meta, DATASET)
    wt = pd.read_parquet(RESULTS / f"linear_unseen_weights_{args.encoder}_{DATASET}.parquet")
    wt = wt[wt.model == "joint_768"].set_index(["fold", "dim", "descriptor"]).w.unstack("descriptor")
    W = np.concatenate([wt.loc[k].sort_index()[DESCRIPTORS].to_numpy() for k in range(K_FOLDS)], 1)  # readout-major
    bench = json.loads((RESULTS / f"linear_unseen_{args.encoder}_{DATASET}.json").read_text())["descriptors"]
    spec = ENCODERS[args.encoder]
    model = spec.build(WEIGHTS / spec.local_dir).to(device).eval()

    parts, cell_rows, map_rows = {}, [], []
    for part in ["val"] + ([TEST_FOLD[DATASET]] if args.read_test else []):
        sample = meta[(meta.fold == part) & meta.cell_id.isin(labels.index)].reset_index(drop=True)
        cell_type = sample.class_label.to_numpy()
        T, const, Y, Z, patches, frac, boundary, norm_stats, err = decompose(sample, model, spec, W, device)
        stored = X[sample.row.to_numpy()].astype(np.float64)
        cos = (Z * stored).sum(1) / np.linalg.norm(Z, axis=1) / np.linalg.norm(stored, axis=1)
        if err["split"] > RTOL_SPLIT or max(v for key, v in err.items() if key != "split") > RTOL_SUM:
            raise RuntimeError(f"decomposition check failed: {err}")
        if cos.min() < MIN_COS:
            raise RuntimeError(f"recomputed embedding differs from the stored one (cos {cos.min():.4f})")

        k_cols = [slice(k * R, (k + 1) * R) for k in range(K_FOLDS)]
        per = [readout_result(T[..., c], const[:, c], Y[:, c], cell_type) for c in k_cols]
        agg = mean_sd(per)
        y_label = labels.loc[sample.cell_id, DESCRIPTORS].to_numpy(np.float64)
        within = np.mean([(centre(Y[:, c], cell_type) ** 2).sum(0) / ((Y[:, c] - Y[:, c].mean(0)) ** 2).sum(0)
                          for c in k_cols], axis=0)
        r2 = {d: {"sample_r2_with_label": float(np.mean([np.corrcoef(Y[:, c][:, i], y_label[:, i])[0, 1] ** 2
                                                         for c in k_cols])),
                  "linear_unseen_r2_pooled": bench[part][d]["r2_pooled"]["mean"]} for i, d in enumerate(DESCRIPTORS)}
        parts[part] = {
            "cells": len(sample), "class_counts": {c: int(m) for c, m in pd.Series(cell_type).value_counts().items()},
            **agg,
            "measured_region_leads": measured_region_leads(means(agg)),
            "all_four_count_per_readout": [len(measured_region_leads(r)["all_four"]) for r in per],
            "within_type_fraction_of_variance": dict(zip(DESCRIPTORS, within.tolist())),
            "mean_area": {**{r: float(frac[:, g].mean()) for g, r in enumerate(REGIONS)},
                          **{f"{r}_interior": float((frac[:, g] == 1).mean()) for g, r in enumerate(REGIONS)},
                          "boundary_patches": float(boundary.mean())},
            "token_norms_by_block_median": pd.DataFrame(
                np.median(norm_stats, 0), index=[f"block {l + 1}" for l in range(L)],
                columns=["max_over_median", f"top{TOP_NORM}th_over_median", f"cls_attention_on_top{TOP_NORM}",
                         "cls_self_attention"]).to_dict(orient="index"),
            "checks": {**{f"max_rel_err_{key}": v for key, v in err.items()},
                       "embedding_cos_min": float(cos.min()), "readout_r2": r2},
        }

        totals = T.sum(1)                                    # (cells, KINDS, readouts), over the 12 blocks
        for k, c in enumerate(k_cols):
            cell_rows.append(pd.DataFrame({
                "split": part, "readout": k, "cell_id": np.repeat(sample.cell_id.to_numpy(), R),
                "class_label": np.repeat(cell_type, R), "descriptor": np.tile(DESCRIPTORS, len(sample)),
                "y": Y[:, c].ravel(), "const": const[:, c].ravel(),
                **{kind: totals[:, g, c].ravel() for g, kind in enumerate(KINDS)}}))
        mean_map = patches.mean(0).reshape(GRID * GRID, K_FOLDS, R).mean(1)          # (patches, R), over all cells
        y_sd = np.mean([Y[:, c].std(0, ddof=1) for c in k_cols], axis=0)
        for i in sample.groupby("class_label").sample(1, random_state=SEED).index:   # before any map is viewed
            pm = patches[i].reshape(GRID * GRID, K_FOLDS, R).mean(1)                  # (patches, R)
            map_rows.append(pd.DataFrame({
                "split": part, "cell_id": sample.cell_id[i], "class_label": cell_type[i],
                "descriptor": np.repeat(DESCRIPTORS, GRID * GRID), "patch": np.tile(np.arange(GRID * GRID), R),
                "write": pm.T.ravel(), "write_mean": mean_map.T.ravel(),
                "y": np.repeat(Y[i].reshape(K_FOLDS, R).mean(0), GRID * GRID),
                "y_sd": np.repeat(y_sd, GRID * GRID), "label": np.repeat(y_label[i], GRID * GRID),
                **{f"frac_{r}": np.tile(frac[i, g], R) for g, r in enumerate(REGIONS)}}))

        print(f"\n[linear_unseen_attnpath] {part}: {len(sample):,} cells; mean shares over {K_FOLDS} readouts, "
              f"all cells, pixel split")
        print(pd.DataFrame(means(agg)["all_cells"]["shares_all_blocks"]).T
              .assign(measured_in=pd.Series(MEASURED_IN)).round(3).to_string())
        print(f"[linear_unseen_attnpath] {part}: measured region leads, all four settings: "
              f"{len(parts[part]['measured_region_leads']['all_four'])}/{R} "
              f"{parts[part]['measured_region_leads']['all_four']}; per readout "
              f"{parts[part]['all_four_count_per_readout']}")
        print(f"[linear_unseen_attnpath] {part} checks: "
              f"{json.dumps({k: v for k, v in parts[part]['checks'].items() if k != 'readout_r2'})}")

    summary = {
        "analysis": "linear_cv_attnpath's decomposition on PBC, read by the five linear_unseen readouts",
        "protocol": "protocol.md §16", "encoder": args.encoder, "dataset": DATASET,
        "test_read": args.read_test, "commit": commit, "excluded_classes": EXCLUDED[DATASET],
        "cells": "every labelled cell of each scored fold, each read by all five linear_unseen readouts",
        "regions": "linear_cv_regions.region_masks; patch writes split by pixel fraction, and as a check "
                   "by interior patches only, with boundary patches as their own group",
        "measured_in": MEASURED_IN,
        "measured_region_leads": "the measured region's share is the largest of nucleus, cytoplasm and rest; "
                                 "N:C: nucleus and cytoplasm both exceed rest; counted on the mean shares",
        "statistic": "Cov(T_g, Y) / Var(Y) over cells, Y = w'(z - beta), as linear_cv_attnpath.py",
        "aggregation": "mean and sd (ddof=1) over the 5 readouts",
        "maps": f"one cell per cell type and scored fold, pandas sample(1, random_state={SEED}); per patch the "
                "direct-attention write summed over the 12 blocks, averaged over the 5 readouts",
        "parts": parts,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__},
        "device": torch.cuda.get_device_name(0), "host": platform.node(),
    }
    pd.concat(cell_rows, ignore_index=True).to_parquet(RESULTS / f"{stem}.parquet", index=False)
    pd.concat(map_rows, ignore_index=True).to_parquet(RESULTS / f"linear_unseen_maps_{args.encoder}_{DATASET}.parquet",
                                                      index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"[linear_unseen_attnpath] wrote {RESULTS / stem}.parquet / .json and the maps")


if __name__ == "__main__":
    main()
