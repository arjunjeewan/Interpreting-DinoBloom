"""Baseline encoders (post hoc): the same linear models on another encoder's CLS embedding, Matek.

Added 2026-11-03 to show how much of each descriptor an encoder pretrained on natural images
carries. The procedure was fixed before any baseline result was seen, and every encoder run is
reported:

    for each encoder, on Matek's five CV folds (cv_splits.parquet, the folds of linear_cv.py),
    one OLS model per descriptor is fit on the four training folds and scored on the fifth:

    pooled R^2 = 1 - MSE / MSE(training-mean prediction)                as linear_cv.py
    within R^2 = 1 - sum (y~ - p~)^2 / sum y~^2, ~ = minus the validation
                 fold's cell-type mean, for label and prediction         as linear_cv_within.py

Mean and sd over the 5 folds. Labels are linear_cv.py's seed-mean descriptors, inputs the
stored CLS embedding (src/embed.py: 224 px, bicubic, ImageNet normalization), unscaled. The 20%
hold-out is not read.

Encoders: dinov2_b (DinoBloom-B's architecture, ViT-B/14, 768-d, pretrained on natural images)
is the baseline; dinov2_l (ViT-L/14, 1024-d) is reported beside it. dinobloom runs through the
same code as a check: its per-fold pooled R^2 must equal linear_cv.py's stored results.

On PBC, which neither encoder saw in pretraining, the baseline runs through linear_unseen.py
instead (--encoder dinov2_b, without --read-test: five fits, scored on the validation split only).

No network access.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

from src.analysis.linear_cv import K_FOLDS, SPLITS, seed_mean_labels
from src.analysis.linear_cv_within import r2_within
from src.analysis.raw_probe import DATASET, RESULTS, load_embeddings
from src.env import go_offline
from src.morphometry import DESCRIPTORS

go_offline()

ATOL = 1e-8   # dinobloom's pooled R^2 vs linear_cv.py's stored per-fold MSEs


def run(encoder: str, splits: pd.DataFrame) -> tuple[list[dict], dict]:
    """Per fold and descriptor: pooled and within-type validation R^2."""
    meta, X = load_embeddings(encoder)
    row_of = pd.Series(meta.row.to_numpy(), index=meta.cell_id.to_numpy())
    if not splits.cell_id.isin(row_of.index).all():
        raise RuntimeError(f"{encoder}: some split cells have no embedding")
    E = X[row_of.loc[splits.cell_id].to_numpy()].astype(np.float64)   # aligned to the split rows
    Y = seed_mean_labels(meta).reindex(splits.cell_id.to_numpy())
    has = Y.n_seeds.notna().to_numpy()
    Yd, cls = Y[DESCRIPTORS].to_numpy(), splits.class_label.to_numpy()
    split, fold = splits.split.to_numpy(), splits.cv_fold.to_numpy()
    rows = []
    for k in range(K_FOLDS):
        t = time.time()
        tr = (split == "cv") & (fold != k) & has
        va = (split == "cv") & (fold == k) & has
        P = LinearRegression().fit(E[tr], Yd[tr]).predict(E[va])
        mu = Yd[tr].mean(axis=0)
        for j, d in enumerate(DESCRIPTORS):
            y, p = Yd[va, j], P[:, j]
            rows.append({"encoder": encoder, "fold": k, "descriptor": d,
                         "n_train": int(tr.sum()), "n_val": int(va.sum()),
                         "r2_pooled": float(1 - ((y - p) ** 2).mean() / ((y - mu[j]) ** 2).mean()),
                         "r2_within": r2_within(y, p, cls[va])})
        print(f"[linear_baseline] {encoder} fold {k}: train {tr.sum():,}, val {va.sum():,} "
              f"({time.time() - t:.0f}s)", flush=True)
    return rows, {"n_cells": int(len(E)), "dim": int(E.shape[1]), "n_labelled": int(has.sum())}


def check_dinobloom(df: pd.DataFrame) -> float:
    """Largest |pooled R^2 - linear_cv.py's| over folds and descriptors, for dinobloom."""
    cv = pd.read_parquet(RESULTS / f"linear_cv_dinobloom_{DATASET}.parquet")
    cv = cv[(cv.split == "val") & cv.target.isin(DESCRIPTORS)]
    mse = cv.pivot_table(index=["target", "fold"], columns="model", values="mse")
    ref = (1 - mse.linear / mse.baseline).rename("ref")
    ours = df[df.encoder == "dinobloom"].set_index(["descriptor", "fold"]).r2_pooled
    ours.index.names = ["target", "fold"]
    return float((ours - ref.reindex(ours.index)).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", nargs="+", default=["dinobloom", "dinov2_b", "dinov2_l"])
    args = ap.parse_args()

    splits = pd.read_parquet(SPLITS)
    rows, shapes = [], {}
    for enc in args.encoder:
        r, shapes[enc] = run(enc, splits)
        rows += r
    df = pd.DataFrame(rows)
    checks = {}
    if "dinobloom" in args.encoder:
        checks["max_abs_r2_pooled_diff_vs_linear_cv"] = check_dinobloom(df)
        if checks["max_abs_r2_pooled_diff_vs_linear_cv"] > ATOL:
            raise RuntimeError(f"dinobloom does not reproduce linear_cv.py: {checks}")

    agg = df.groupby(["encoder", "descriptor"], sort=False)[["r2_pooled", "r2_within"]].agg(["mean", "std"])
    summary = {
        "analysis": "baseline encoders (post hoc): OLS per descriptor on the stored CLS embedding, "
                    "Matek validation folds only (hold-out not read)",
        "encoder_roles": {"dinobloom": "check (reproduces linear_cv.py)", "dinov2_b": "baseline",
                          "dinov2_l": "reported beside the baseline"},
        "encoders": shapes, "dataset": DATASET, "test_read": False,
        "statistics": {"r2_pooled": "1 - MSE / MSE(training-mean prediction)",
                       "r2_within": "1 - SSE / SS after centring label and prediction on the "
                                    "validation fold's cell-type means"},
        "aggregation": "mean and sd (ddof=1) over the 5 validation folds",
        "checks": checks,
        "results": {enc: {d: {s: {"mean": float(agg.loc[(enc, d), (s, "mean")]),
                                  "sd": float(agg.loc[(enc, d), (s, "std")])}
                              for s in ["r2_pooled", "r2_within"]} for d in DESCRIPTORS}
                    for enc in args.encoder},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "host": platform.node(),
    }
    stem = f"linear_baseline_{DATASET}"
    df.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))

    for enc in args.encoder:
        m = agg.loc[enc]
        print(f"\n{enc} ({shapes[enc]['dim']}-d): pooled R^2 {m[('r2_pooled', 'mean')].min():.3f}-"
              f"{m[('r2_pooled', 'mean')].max():.3f}, within {m[('r2_within', 'mean')].min():.3f}-"
              f"{m[('r2_within', 'mean')].max():.3f}")
        print(m.round(3).to_string())
    print(f"\n[linear_baseline] checks: {json.dumps(checks)}")
    print(f"[linear_baseline] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
