"""Within-cell-type readout of each metric: what the readout gets right beyond the cell type.

The base readouts (linear_cv.py: OLS on all 768 positions; weights from
linear_cv_weights.py's joint_768) are scored on each validation fold after
centring both the target and the prediction on that fold's cell-type means:

    R^2_within = 1 - sum (y~ - yhat~)^2 / sum y~^2,    ~ = minus the cell type's mean

A readout that knew only the cell type would score exactly 0, and the readout's
intercept cancels. Beside it, on the same folds:

    r2_pooled       1 - MSE / MSE(training-mean prediction), as in linear_cv.py
    r2_type_only    the same for predicting each cell's training-fold type mean:
                    how much of the metric the cell type alone carries
    ceiling_within  Spearman-Brown reliability, inside cell types, of the label
                    (the mean over a cell's QC-passing segmenter seeds): the mean
                    pairwise seed-to-seed Pearson r of type-centred single-seed
                    labels, stepped up to the median number of seeds per cell. It
                    bounds R^2_within against segmenter noise only.
    rho (by type)   Spearman rho of prediction and target inside each cell type
                    with at least MIN_TYPE_N validation cells in the fold, with n

Each validation fold is scored by the model trained without it. Only the five
CV folds are read; the 20% hold-out is spent and is not read.

Checks: the readout's validation MSE against linear_cv.py's; the seed-mean labels
rebuilt from the per-seed labels; a type-only prediction must score R^2_within = 0.

No network access.
"""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr

from src.analysis.linear_cv import K_FOLDS, SEG_SEEDS, seed_mean_labels, splits_for
from src.analysis.raw_probe import DATASET, RESULTS, labels_for, load_embeddings
from src.env import go_offline
from src.morphometry import DESCRIPTORS

go_offline()

MIN_TYPE_N = 3      # fewest cells for which a within-type Spearman rho means anything
RTOL_MSE = 1e-8     # validation MSE vs linear_cv.py's, relative
ATOL = 1e-10        # type-only R^2_within vs 0; rebuilt vs stored seed-mean labels


def centred(v: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """v minus its group's mean."""
    return v - pd.Series(v).groupby(groups).transform("mean").to_numpy()


def r2_within(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> float:
    """R^2 of p for y after centring both on the group means."""
    yc, pc = centred(y, groups), centred(p, groups)
    return float(1 - ((yc - pc) ** 2).sum() / (yc ** 2).sum())


def ceiling_within(per_seed: np.ndarray, groups: np.ndarray, m: float) -> float:
    """Spearman-Brown reliability of an m-seed mean label inside groups.

    per_seed: seeds x cells, NaN where a seed failed QC. Each seed's label is
    centred on its group means over the cells it kept; each pair of seeds is
    correlated on the cells both kept.
    """
    C = np.full_like(per_seed, np.nan)
    for s, row in enumerate(per_seed):
        ok = ~np.isnan(row)
        C[s, ok] = centred(row[ok], groups[ok])
    rs = []
    for a in range(len(C)):
        for b in range(a + 1, len(C)):
            ok = ~np.isnan(C[a]) & ~np.isnan(C[b])
            rs.append(np.corrcoef(C[a, ok], C[b, ok])[0, 1])
    r = float(np.mean(rs))
    return m * r / (1 + (m - 1) * r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    args = ap.parse_args()

    meta, X = load_embeddings(args.encoder)
    meta = meta.reset_index(drop=True)
    E = X[meta.row.to_numpy()].astype(np.float64)
    splits = splits_for(meta)
    fold, cell_type = splits.cv_fold.to_numpy(), splits.class_label.to_numpy()
    labels = seed_mean_labels(meta).reindex(meta.cell_id.to_numpy())
    has_label = labels.n_seeds.notna().to_numpy()
    Y, n_seeds = labels[DESCRIPTORS].to_numpy(), labels.n_seeds.to_numpy()
    per_seed = np.full((len(SEG_SEEDS), len(meta), len(DESCRIPTORS)), np.nan)
    row_of = pd.Series(np.arange(len(meta)), index=meta.cell_id)
    for s in SEG_SEEDS:
        lab = labels_for(s, meta)
        per_seed[s, row_of.loc[lab.cell_id].to_numpy()] = lab[DESCRIPTORS].to_numpy()
    rebuilt = np.abs(np.nanmean(per_seed[:, has_label], axis=0) - Y[has_label]).max()

    wt = pd.read_parquet(RESULTS / f"linear_cv_weights_{args.encoder}_{DATASET}.parquet")
    wt = wt[wt.model == "joint_768"].set_index(["fold", "dim", "descriptor"]).w.unstack("descriptor")
    ref = pd.read_parquet(RESULTS / f"linear_cv_{args.encoder}_{DATASET}.parquet")
    ref = ref[(ref.split == "val") & (ref.model == "linear")].set_index(["target", "fold"]).mse

    rows, type_rows, worst_mse, worst_zero = [], [], 0.0, 0.0
    for k in range(K_FOLDS):
        tr, va = (fold >= 0) & (fold != k) & has_label, (fold == k) & has_label
        W = wt.loc[k].sort_index()[DESCRIPTORS].to_numpy()                       # (positions, metrics)
        P = E[va] @ W + (Y[tr].mean(0) - E[tr].mean(0) @ W)                       # with the OLS intercept
        g = cell_type[va]
        T = pd.DataFrame(Y[tr]).groupby(cell_type[tr]).mean().loc[g].to_numpy()  # type-only prediction
        base = ((Y[va] - Y[tr].mean(0)) ** 2).mean(0)                             # MSE of the training mean
        m = float(np.median(n_seeds[va]))
        for i, d in enumerate(DESCRIPTORS):
            y, p = Y[va, i], P[:, i]
            mse = float(((y - p) ** 2).mean())
            worst_mse = max(worst_mse, abs(mse - ref.loc[(d, k)]) / ref.loc[(d, k)])
            worst_zero = max(worst_zero, abs(r2_within(y, T[:, i], g)))
            rows.append({"descriptor": d, "fold": k, "n": int(va.sum()), "n_seeds_median": m,
                         "r2_pooled": 1 - mse / base[i],
                         "r2_type_only": float(1 - ((y - T[:, i]) ** 2).mean() / base[i]),
                         "r2_within": r2_within(y, p, g),
                         "ceiling_within": ceiling_within(per_seed[:, va, i], g, m)})
            for c in np.unique(g):
                sel = g == c
                if sel.sum() >= MIN_TYPE_N:
                    type_rows.append({"descriptor": d, "fold": k, "class_label": c, "n": int(sel.sum()),
                                      "rho": float(spearmanr(p[sel], y[sel]).statistic)})
        print(f"[linear_cv_within] fold {k}: {va.sum():,} validation cells", flush=True)

    if worst_mse > RTOL_MSE or worst_zero > ATOL or rebuilt > ATOL:
        raise RuntimeError(f"check failed: val MSE {worst_mse:.1e}, type-only R2_within {worst_zero:.1e}, "
                           f"rebuilt labels {rebuilt:.1e}")

    out, types = pd.DataFrame(rows), pd.DataFrame(type_rows)
    stats = ["r2_pooled", "r2_type_only", "r2_within", "ceiling_within"]
    agg = out.groupby("descriptor", sort=False)[stats].agg(["mean", "std"])
    by_type = types.groupby(["descriptor", "class_label"]).agg(
        rho_mean=("rho", "mean"), rho_sd=("rho", "std"), n_total=("n", "sum"), folds=("fold", "nunique"))
    summary = {
        "analysis": "within-cell-type validation R^2 of the joint_768 readouts, beside the pooled and "
                    "type-only R^2 and a label-reliability ceiling",
        "encoder": args.encoder, "dataset": DATASET, "test_read": False,
        "splits": "linear_cv.py's five CV folds; each scored by the model trained without it",
        "centring": "each validation fold's own cell-type means, for target and prediction",
        "ceiling": "Spearman-Brown reliability of the seed-mean label inside cell types, stepped up to the "
                   "median number of QC-passing seeds per validation cell",
        "per_type": f"Spearman rho inside each cell type with at least {MIN_TYPE_N} validation cells in a fold",
        "descriptors": {d: {s: {"mean": float(agg.loc[d, (s, "mean")]), "sd": float(agg.loc[d, (s, "std")])}
                            for s in stats} for d in DESCRIPTORS},
        "rho_by_type": {d: by_type.loc[d].reset_index().to_dict(orient="records") for d in DESCRIPTORS},
        "checks": {"max_rel_val_mse_vs_linear_cv": worst_mse, "max_abs_r2_within_of_type_only": worst_zero,
                   "max_abs_rebuilt_seed_mean_label": float(rebuilt)},
        "aggregation": "mean and sd (ddof=1) over the 5 folds",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "host": platform.node(),
    }
    stem = f"linear_cv_within_{args.encoder}_{DATASET}"
    out.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    types.to_parquet(RESULTS / f"{stem}_by_type.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    tab = pd.DataFrame({s: [f"{agg.loc[d, (s, 'mean')]:.3f} ± {agg.loc[d, (s, 'std')]:.3f}" for d in DESCRIPTORS]
                        for s in stats}, index=DESCRIPTORS)
    print(tab.to_string())
    print(f"[linear_cv_within] checks: {json.dumps(summary['checks'])}")
    print(f"[linear_cv_within] wrote {RESULTS / stem}.parquet / _by_type.parquet / .json")


if __name__ == "__main__":
    main()
