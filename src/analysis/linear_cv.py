"""Basic linear models on the raw embedding: stratified 5-fold CV plus a 20% hold-out.

The simplest version of the raw-embedding question: can plain linear models
explain the ground truths from the 768-d embedding? One model per ground truth:

    11 morphometric descriptors   LinearRegression (OLS)   MSE, RMSE, MAE
    cell type (21 classes)        LogisticRegression       log loss, accuracy,
                                                           balanced accuracy

Each sits beside a baseline that ignores the embedding, scored the same way:
DummyRegressor predicting the training mean, DummyClassifier predicting the
training class frequencies. Regression metrics are in each descriptor's own
units (pixels for areas, src/morphometry.py), so they are read against that
baseline, not across descriptors.

Splits, stratified by cell type, seed 0: a random 20% hold-out test, then
StratifiedKFold(5) on the remaining 80%. Each fold is validation once and the
other four train. Every fold's model is scored on its validation fold and on
the test split; both are reported as mean +/- sd over the 5 folds. The test sd
therefore reflects the training folds only, not test sampling. The split is
written once to data/interim/matek_bm/cv_splits.parquet and checked against a
fresh draw on every later run. It is separate from the committed three-way
split (splits.parquet), which is not touched.

Continuous labels are, per cell, the mean of each descriptor over the 25
segmenter seeds in which the cell passed QC (qc_pass and all descriptors
finite, as in raw_probe.py). Cells passing in no seed drop out of the
regressions only.

Nothing is tuned: OLS has no hyperparameter, and the logistic regression runs
at sklearn defaults (L2, C=1, lbfgs) with max_iter raised. So the validation
folds are pure evaluation, and val and test should agree. Inputs are the stored
embedding, unscaled: OLS predictions do not depend on feature scale, and
z-scoring moved the ridge readout by < 0.0003 R^2 (raw_probe.py --no-zscore).

Limits: Matek is in DinoBloom's pretraining, so this characterizes the
representation and supports no generalization claim. Matek has no patient IDs
(src/splits.py), so the split is cell-level and cells from one patient can sit
on both sides. The labels are segmenter output, not human annotation.

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
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, log_loss,
                             mean_absolute_error, mean_squared_error)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.analysis.raw_probe import DATASET, RESULTS, labels_for, load_embeddings
from src.env import ROOT, go_offline
from src.morphometry import DESCRIPTORS

go_offline()

SEED = 0
TEST_SIZE = 0.2
K_FOLDS = 5
SEG_SEEDS = range(25)
MAX_ITER = 1000
SPLITS = ROOT / "data" / "interim" / DATASET / "cv_splits.parquet"
REG_METRICS = ["mse", "rmse", "mae"]
CLS_METRICS = ["log_loss", "accuracy", "balanced_accuracy"]


def draw_splits(meta: pd.DataFrame) -> pd.DataFrame:
    """Stratified 20% hold-out, then stratified 5 folds over the remaining 80%."""
    s = meta[["cell_id", "class_label"]].reset_index(drop=True)
    cv_idx, test_idx = train_test_split(np.arange(len(s)), test_size=TEST_SIZE,
                                        stratify=s.class_label, random_state=SEED)
    cv_idx = np.sort(cv_idx)
    s["split"], s["cv_fold"] = "cv", -1
    s.loc[test_idx, "split"] = "test"
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(skf.split(cv_idx, s.class_label.to_numpy()[cv_idx])):
        s.loc[cv_idx[va], "cv_fold"] = k
    return s


def splits_for(meta: pd.DataFrame) -> pd.DataFrame:
    """The fixed split: drawn and written once, then checked against a fresh draw."""
    fresh = draw_splits(meta)
    if not SPLITS.exists():
        fresh.to_parquet(SPLITS, index=False)
        print(f"[linear_cv] wrote {SPLITS.relative_to(ROOT)}", flush=True)
        return fresh
    saved = pd.read_parquet(SPLITS)
    if len(saved) != len(fresh) or any((saved[c].to_numpy() != fresh[c].to_numpy()).any()
                                       for c in fresh.columns):
        raise RuntimeError(f"{SPLITS} differs from a fresh draw; the split must not drift")
    return saved


def check_splits(s: pd.DataFrame) -> dict:
    """Assert the split's integrity and return its summary."""
    if s.cell_id.duplicated().any():
        raise RuntimeError("a cell appears twice in the split")
    is_test = (s.split == "test").to_numpy()
    if (is_test != (s.cv_fold == -1).to_numpy()).any() or not s.cv_fold[~is_test].isin(range(K_FOLDS)).all():
        raise RuntimeError(f"each cell must be in exactly one of test or folds 0-{K_FOLDS - 1}")
    n_class = s.class_label.value_counts()
    n_test = s.class_label[is_test].value_counts().reindex(n_class.index, fill_value=0)
    test_dev = float((n_test - TEST_SIZE * n_class).abs().max())
    per_fold = pd.crosstab(s.class_label[~is_test], s.cv_fold[~is_test])
    fold_spread = int((per_fold.max(axis=1) - per_fold.min(axis=1)).max())
    if test_dev > 1 or fold_spread > 1:
        raise RuntimeError(f"stratification off: test deviates by {test_dev:.2f} cells, "
                           f"folds by {fold_spread}")
    return {"file": str(SPLITS.relative_to(ROOT)), "seed": SEED, "test_size": TEST_SIZE,
            "k_folds": K_FOLDS, "stratified_by": "class_label",
            "n": len(s), "n_test": int(is_test.sum()),
            "n_per_fold": json.loads(s.cv_fold[~is_test].value_counts().sort_index().to_json()),
            "max_test_deviation_cells": test_dev, "max_fold_spread_cells": fold_spread,
            "class_counts": {"all": json.loads(n_class.to_json()),
                             "test": json.loads(n_test.to_json()),
                             "per_fold": json.loads(per_fold.to_json())}}


def seed_mean_labels(meta: pd.DataFrame, dataset: str = DATASET) -> pd.DataFrame:
    """Per cell, each descriptor's mean over the segmenter seeds where it passed QC."""
    long = pd.concat([labels_for(s, meta, dataset)[["cell_id"] + DESCRIPTORS] for s in SEG_SEEDS],
                     ignore_index=True)
    g = long.groupby("cell_id")
    return g[DESCRIPTORS].mean().assign(n_seeds=g.size())


def regression_scores(model, X: np.ndarray, y: np.ndarray) -> dict:
    p = model.predict(X)
    mse = mean_squared_error(y, p)
    return {"n": len(y), "mse": mse, "rmse": float(np.sqrt(mse)), "mae": mean_absolute_error(y, p)}


def classification_scores(model, X: np.ndarray, y: np.ndarray) -> dict:
    pred = model.predict(X)
    return {"n": len(y), "log_loss": log_loss(y, model.predict_proba(X), labels=model.classes_),
            "accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred)}


def run_fold(k: int, E: np.ndarray, cls: np.ndarray, Y: np.ndarray, has_label: np.ndarray,
             split: np.ndarray, fold: np.ndarray) -> tuple[list[dict], int]:
    """Fit every model on the four training folds; score on fold k and on test."""
    tr = (split == "cv") & (fold != k)
    parts = {"val": fold == k, "test": split == "test"}
    rows = []

    # Cell type, all cells: logistic regression and a class-frequency baseline.
    logistic = LogisticRegression(max_iter=MAX_ITER)
    for name, model in [("logistic", logistic), ("baseline", DummyClassifier(strategy="prior"))]:
        model.fit(E[tr], cls[tr])
        for part, m in parts.items():
            rows.append({"target": "class_label", "model": name, "fold": k, "split": part,
                         **classification_scores(model, E[m], cls[m])})

    # Descriptors, cells with a label: one OLS model each and a training-mean baseline.
    trl = tr & has_label
    Etr = E[trl]
    held = {part: m & has_label for part, m in parts.items()}
    Eheld = {part: E[m] for part, m in held.items()}
    for j, d in enumerate(DESCRIPTORS):
        for name, model in [("linear", LinearRegression()), ("baseline", DummyRegressor(strategy="mean"))]:
            model.fit(Etr, Y[trl, j])
            for part, m in held.items():
                rows.append({"target": d, "model": name, "fold": k, "split": part,
                             **regression_scores(model, Eheld[part], Y[m, j])})
    return rows, int(logistic.n_iter_[0])


def summarize(rows: pd.DataFrame) -> dict:
    """Mean +/- sd over folds, per target x model x split."""
    out = {}
    for (target, model, part), g in rows.groupby(["target", "model", "split"], sort=False):
        metrics = CLS_METRICS if target == "class_label" else REG_METRICS
        out.setdefault(target, {}).setdefault(model, {})[part] = {
            "n_median": int(g.n.median()),
            **{m: {"mean": float(g[m].mean()), "sd": float(g[m].std(ddof=1))} for m in metrics}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    args = ap.parse_args()

    meta, X = load_embeddings(args.encoder)
    meta = meta.reset_index(drop=True)
    emb = pd.DataFrame(X[meta.row.to_numpy()].astype(np.float64), index=meta.cell_id.to_numpy())  # N x 768
    splits = splits_for(meta)
    if (splits.cell_id.to_numpy() != emb.index.to_numpy()).any():
        raise RuntimeError("split rows are not aligned with the embedding rows")
    split_summary = check_splits(splits)
    labels = seed_mean_labels(meta)
    Y = labels.reindex(emb.index)
    has_label = Y.n_seeds.notna().to_numpy()
    print(f"[linear_cv] {args.encoder}: input {emb.shape[0]:,} x {emb.shape[1]}, "
          f"test {split_summary['n_test']:,}, folds {list(split_summary['n_per_fold'].values())}, "
          f"descriptor labels for {has_label.sum():,} cells", flush=True)

    E, cls, Yd = emb.to_numpy(), splits.class_label.to_numpy(), Y[DESCRIPTORS].to_numpy()
    split, fold = splits.split.to_numpy(), splits.cv_fold.to_numpy()
    rows, n_iter = [], []
    for k in range(K_FOLDS):
        t = time.time()
        fold_rows, it = run_fold(k, E, cls, Yd, has_label, split, fold)
        rows += fold_rows
        n_iter.append(it)
        print(f"[linear_cv] fold {k}: {time.time() - t:.0f}s, logistic n_iter {it}"
              + ("  NOT CONVERGED" if it >= MAX_ITER else ""), flush=True)

    rows = pd.DataFrame(rows)
    summary = {
        "analysis": "linear models from the raw embedding: OLS per descriptor, logistic for cell type",
        "encoder": args.encoder, "dataset": DATASET, "test_read": True,
        "splits": split_summary,
        "labels": {
            "descriptors": "mean over segmenter seeds 0-24 in which the cell passed QC "
                           "(qc_pass and all descriptors finite); units as in src/morphometry.py",
            "n_labelled": int(has_label.sum()), "n_unlabelled": int((~has_label).sum()),
            "unlabelled_by_class": json.loads(pd.Series(cls[~has_label]).value_counts().to_json()),
            "n_seeds_quantiles": json.loads(labels.n_seeds.quantile([0, 0.01, 0.05, 0.5]).to_json()),
            "class_label": "Matek class, 21 classes",
        },
        "models": {"regression": "LinearRegression (OLS), one per descriptor; "
                                 "baseline DummyRegressor(strategy='mean')",
                   "classification": f"LogisticRegression at sklearn defaults (L2, C=1.0, lbfgs), "
                                     f"max_iter={MAX_ITER}; baseline DummyClassifier(strategy='prior')",
                   "features": "embedding as stored, not scaled"},
        "logistic_n_iter": n_iter, "logistic_converged": [it < MAX_ITER for it in n_iter],
        "aggregation": "mean and sd (ddof=1) over the 5 fold models, on each val fold and on test",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "host": platform.node(),
        "results": summarize(rows),
    }
    stem = f"linear_cv_{args.encoder}_{DATASET}"
    rows.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))

    print(f"\nLinear models on raw {args.encoder} ({E.shape[1]}-d), Matek BM, "
          f"mean ± sd over {K_FOLDS} folds:")
    for target, by_model in summary["results"].items():
        metrics = CLS_METRICS if target == "class_label" else REG_METRICS
        print(f"\n  {target}")
        print(f"    {'model':9s} {'split':5s} " + " ".join(f"{m:>21s}" for m in metrics))
        for name, by_part in by_model.items():
            for part, s in by_part.items():
                print(f"    {name:9s} {part:5s} "
                      + " ".join(f"{s[m]['mean']:>11.4g} ± {s[m]['sd']:<7.2g}" for m in metrics))
    print(f"\n[linear_cv] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
