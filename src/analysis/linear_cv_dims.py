"""One embedding position at a time: validation MAE and R^2 of a single-position OLS fit.

Companion to linear_cv.py, which fits one OLS model per descriptor on all
embedding positions at once. This fits one per position instead: for each
position (coordinate) j of the embedding E and each morphometric descriptor y,

    y ~ a + b * E[:, j]

fitted on four training folds and scored on the fifth by MAE, MSE and
R^2 = 1 - MSE / MSE(training mean), over the same five folds and the same
seed-mean labels as linear_cv.py. Read across j, the validation error is a
profile of each descriptor over the embedding: dips in MAE (peaks in R^2) mark
the positions that predict it on their own, and linear_cv's all-position model
shows how far any one position is from the full readout. Mean and sd are over
the 5 folds. Cell type has no MAE and is left out.

Only the validation folds are scored; the 20% hold-out was read by
linear_cv.py and is not touched here. Every "best" is chosen on train and
reported on val, as in raw_dims.py: in each fold, the best position is the one
with the lowest MAE on that fold's training cells.

Checks: the closed form b = cov(x, y) / var(x), a = mean(y) - b mean(x) is
compared with sklearn's LinearRegression at sampled positions in every fold,
on MAE and MSE; and the training-mean prediction (slope zero) must reproduce
linear_cv.py's DummyRegressor baseline MAE and MSE fold by fold, which pins the
cells and labels to linear_cv's.

Figure: python -m src.figures --only dims. No network access.
"""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.analysis.linear_cv import K_FOLDS, SPLITS, check_splits, seed_mean_labels, splits_for
from src.analysis.raw_probe import DATASET, RESULTS, load_embeddings
from src.env import ROOT, go_offline
from src.morphometry import DESCRIPTORS

go_offline()

SEED = 0
CHUNK = 128   # positions per block of the n x positions residual matrix
N_SPOT = 10   # random (descriptor, position) pairs per fold checked against sklearn
RTOL = 1e-9


def position_errors(Xc: np.ndarray, yc: np.ndarray, slope: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MAE and MSE of the one-position fit at every position, from X and y centred on train."""
    mae, mse = np.empty(Xc.shape[1]), np.empty(Xc.shape[1])
    for s in range(0, Xc.shape[1], CHUNK):
        e = yc[:, None] - Xc[:, s:s + CHUNK] * slope[s:s + CHUNK]
        mae[s:s + CHUNK], mse[s:s + CHUNK] = np.abs(e).mean(axis=0), (e ** 2).mean(axis=0)
    return mae, mse


def summarize(rows: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Per descriptor: baseline, best position (chosen on train) and all-position MAE, on val."""
    out = {}
    for d in DESCRIPTORS:
        tr = rows[(rows.descriptor == d) & (rows.split == "train")]
        va = rows[(rows.descriptor == d) & (rows.split == "val")].set_index(["fold", "dim"]).mae
        best = tr.loc[tr.groupby("fold").mae.idxmin()].dim.to_numpy()  # fold order 0..4
        best_val = np.array([va.loc[(k, j)] for k, j in enumerate(best)])
        base, full = ref.xs(("baseline", d)).mae, ref.xs(("linear", d)).mae
        r2_full = 1 - ref.xs(("linear", d)).mse / ref.xs(("baseline", d)).mse
        mode = int(pd.Series(best).mode().iloc[0])
        out[d] = {
            "n_val_median": int(rows[(rows.descriptor == d) & (rows.split == "val")].n.median()),
            "mae_val_baseline": {"mean": float(base.mean()), "sd": float(base.std(ddof=1))},
            "mae_val_all_dims": {"mean": float(full.mean()), "sd": float(full.std(ddof=1))},
            "r2_val_all_dims": {"mean": float(r2_full.mean()), "sd": float(r2_full.std(ddof=1))},
            "best_dim_by_fold": best.tolist(),
            "best_dim": mode, "best_dim_n_folds": int((best == mode).sum()),
            "mae_val_best_dim": {"mean": float(best_val.mean()), "sd": float(best_val.std(ddof=1))},
            "gap_closed_by_best_dim": float((base.mean() - best_val.mean()) / (base.mean() - full.mean())),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    args = ap.parse_args()

    meta, X = load_embeddings(args.encoder)
    meta = meta.reset_index(drop=True)
    E = X[meta.row.to_numpy()].astype(np.float64)  # N x positions
    splits = splits_for(meta)
    if (splits.cell_id.to_numpy() != meta.cell_id.to_numpy()).any():
        raise RuntimeError("split rows are not aligned with the embedding rows")
    check_splits(splits)
    Y = seed_mean_labels(meta).reindex(meta.cell_id.to_numpy())
    has_label = Y.n_seeds.notna().to_numpy()
    Y = Y[DESCRIPTORS].to_numpy()
    fold = splits.cv_fold.to_numpy()  # -1 marks the hold-out test, which is never used here
    n_dims = E.shape[1]
    print(f"[linear_cv_dims] {args.encoder}: {E.shape[0]:,} x {n_dims}, "
          f"{len(DESCRIPTORS)} descriptors, {K_FOLDS} folds", flush=True)

    ref = pd.read_parquet(RESULTS / f"linear_cv_{args.encoder}_{DATASET}.parquet")
    ref = ref[ref.split == "val"].set_index(["model", "target", "fold"]).sort_index()
    rng = np.random.default_rng(SEED)
    rows, worst, n_checks = [], 0.0, 0
    for k in range(K_FOLDS):
        tr = (fold >= 0) & (fold != k) & has_label
        va = (fold == k) & has_label
        mx, my = E[tr].mean(axis=0), Y[tr].mean(axis=0)
        Xc, Yc = E[tr] - mx, Y[tr] - my
        Xv, Yv = E[va] - mx, Y[va] - my
        ss = (Xc ** 2).sum(axis=0)
        slope = Xc.T @ Yc / ss[:, None]  # positions x descriptors

        for i, d in enumerate(DESCRIPTORS):
            base = ref.loc[("baseline", d, k)]
            if (base.n != va.sum() or not np.isclose(np.abs(Yv[:, i]).mean(), base.mae, rtol=1e-12, atol=0)
                    or not np.isclose((Yv[:, i] ** 2).mean(), base.mse, rtol=1e-12, atol=0)):
                raise RuntimeError(f"fold {k}, {d}: the training-mean MAE or MSE does not reproduce "
                                   "linear_cv's baseline, so the cells or labels differ")
        err = {part: [position_errors(Xp, Yp[:, i], slope[:, i]) for i in range(len(DESCRIPTORS))]
               for part, Xp, Yp in (("train", Xc, Yc), ("val", Xv, Yv))}
        mae = {part: np.stack([e[0] for e in v]) for part, v in err.items()}  # descriptors x positions
        mse = {part: np.stack([e[1] for e in v]) for part, v in err.items()}
        base_mse = {"train": (Yc ** 2).mean(axis=0), "val": (Yv ** 2).mean(axis=0)}  # training-mean prediction

        # Against sklearn: random pairs, plus the position with the smallest spread.
        pairs = [(int(rng.integers(len(DESCRIPTORS))), int(rng.integers(n_dims))) for _ in range(N_SPOT)]
        pairs.append((int(rng.integers(len(DESCRIPTORS))), int(np.argmin(ss))))
        for i, j in pairs:
            lr = LinearRegression().fit(E[tr, j:j + 1], Y[tr, i])
            for part, m in (("train", tr), ("val", va)):
                p = lr.predict(E[m, j:j + 1])
                worst = max(worst, abs(mae[part][i, j] / mean_absolute_error(Y[m, i], p) - 1),
                            abs(mse[part][i, j] / mean_squared_error(Y[m, i], p) - 1))
                n_checks += 2

        for part, n in (("train", tr.sum()), ("val", va.sum())):
            for i, d in enumerate(DESCRIPTORS):
                rows.append(pd.DataFrame({"descriptor": d, "dim": np.arange(n_dims), "fold": k,
                                          "split": part, "n": int(n), "mae": mae[part][i],
                                          "mse": mse[part][i], "r2": 1 - mse[part][i] / base_mse[part][i]}))
        print(f"[linear_cv_dims] fold {k}: train {tr.sum():,}, val {va.sum():,}", flush=True)
    if worst > RTOL:
        raise RuntimeError(f"closed form differs from sklearn by {worst:.2e} (relative)")

    rows = pd.concat(rows, ignore_index=True)
    summary = {
        "analysis": "OLS of each descriptor on one embedding position at a time; MAE, MSE and R^2 per position",
        "r2": "1 - MSE / MSE(training-mean prediction), on the same split",
        "encoder": args.encoder, "dataset": DATASET, "n_dims": n_dims, "test_read": False,
        "splits": {"file": str(SPLITS.relative_to(ROOT)), "k_folds": K_FOLDS,
                   "scored": "the 5 validation folds; the 20% hold-out is not read"},
        "labels": "as linear_cv.py: per cell, the mean of each descriptor over segmenter seeds "
                  "0-24 in which the cell passed QC",
        "model": "y ~ a + b x_j by OLS on the training folds, closed form; embedding as stored, not scaled",
        "selection": "best position per fold = lowest MAE on that fold's training cells, "
                     "reported on its validation fold; best_dim is the most frequent of the 5",
        "references": f"baseline (training mean) and all-dims OLS: validation MAE from "
                      f"linear_cv_{args.encoder}_{DATASET}.parquet",
        "checks": {"baseline_matches_linear_cv": True, "n_sklearn_comparisons": n_checks,
                   "max_relative_error_vs_sklearn": worst},
        "aggregation": "mean and sd (ddof=1) over the 5 folds",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "host": platform.node(),
        "descriptors": summarize(rows, ref),
    }
    stem = f"linear_cv_dims_{args.encoder}_{DATASET}"
    rows.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))

    print(f"\nOne-position OLS on raw {args.encoder}, Matek BM, validation MAE, "
          f"mean ± sd over {K_FOLDS} folds ({n_checks} sklearn comparisons, "
          f"max relative error {worst:.1e}):\n")
    print(f"  {'descriptor':27s} {'baseline':>9s} {'best dim':>12s} {'MAE at best':>20s} "
          f"{'all dims':>9s} {'gap closed':>10s}")
    for d, s in summary["descriptors"].items():
        b = s["mae_val_best_dim"]
        print(f"  {d:27s} {s['mae_val_baseline']['mean']:>9.4g} "
              f"{s['best_dim']:>5d} ({s['best_dim_n_folds']}/{K_FOLDS}) "
              f"{b['mean']:>11.4g} ± {b['sd']:<6.2g} {s['mae_val_all_dims']['mean']:>9.4g} "
              f"{s['gap_closed_by_best_dim']:>10.1%}")
    print(f"\n[linear_cv_dims] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
