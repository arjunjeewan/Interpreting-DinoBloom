"""Readouts on unseen data (protocol.md §16): linear_cv.py's readouts, fit per dataset and
scored on cells DinoBloom never saw.

Matek BM, where every readout in the paper was developed, is in DinoBloom's
pretraining (protocol.md §2). Two sets are not:

    pbc      PBC Barcelona, held out of pretraining by DinoBloom's authors. Fit on
             its train fold, scored on val; the test fold is read once. The platelet
             class is excluded outright (protocol.md §14, 2026-07-31).
    raabin   Raabin testB, inferred unseen and acquired on another device. Fit on
             the train fold of Raabin Train, scored on its val fold; testB is read
             once. testA shares 26 of its 28 blood films with Train and is never
             read; nor is Raabin's own test fold.

Readouts and labels exactly as linear_cv.py: OLS per descriptor on the stored
768-d embedding, logistic regression at sklearn defaults for cell type, and the
dummy baselines; a descriptor label is the cell's mean over the segmenter seeds
in which it passed QC. Nothing is tuned, so validation selects nothing.

Five readouts per dataset, each fit on 4/5 of the train fold (StratifiedKFold(5)
by cell type, seed 0); every number is the mean +/- sd over the five. As in
Table 1, the sd reflects training folds, not test sampling.

Per descriptor and scored fold, as linear_cv_within.py: pooled R^2 (against the
training mean), R^2 of the cell type alone, within-type R^2 (centred on the
scored fold's own type means), the Spearman-Brown label ceiling, and Spearman
rho, pooled and inside each cell type. testB is imaged at another scale than
Raabin Train (crops 466-535 vs 575 px), so on testB the R^2 statistics are kept
for the scale-invariant descriptors only; rho is kept for all 11.

The readouts' weights are written in linear_cv_weights.py's format (joint_768),
for the analyses that reuse them on PBC.

--read-test also scores the test fold. It runs only when src/ and protocol.md
are committed, writes the commit hash into the JSON, and refuses to overwrite
an earlier test run.

No network access.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.analysis.linear_cv import (CLS_METRICS, K_FOLDS, MAX_ITER, SEED, SEG_SEEDS,
                                    classification_scores, seed_mean_labels)
from src.analysis.linear_cv_within import MIN_TYPE_N, ceiling_within, r2_within
from src.analysis.raw_probe import RESULTS, labels_for, load_embeddings
from src.env import ROOT, go_offline
from src.morphometry import DESCRIPTORS

go_offline()

TEST_FOLD = {"pbc": "test", "raabin": "testB"}
NEVER_READ = {"pbc": [], "raabin": ["testA", "test"]}
EXCLUDED = {"pbc": ["platelet"], "raabin": []}     # protocol.md §14, 2026-07-31
SCALE_INVARIANT = ["nc_ratio", "nuclear_solidity", "eccentricity"]
R2_STATS = ["r2_pooled", "r2_type_only", "r2_within"]
STATS = R2_STATS + ["ceiling_within", "rho_pooled"]
ATOL = 1e-10        # type-only R^2_within vs 0; rebuilt vs stored seed-mean labels


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def frozen_commit() -> str:
    """HEAD's hash, provided src/ and protocol.md hold no uncommitted change."""
    dirty = git("status", "--porcelain", "--", "src", "protocol.md")
    if dirty:
        raise RuntimeError(f"commit src/ and protocol.md before reading test:\n{dirty}")
    return git("rev-parse", "HEAD")


def training_subfolds(fold: np.ndarray, cell_type: np.ndarray) -> np.ndarray:
    """Per cell, the readout whose training set leaves it out: StratifiedKFold(5) of the train fold by cell
    type, seed 0; readout k is fit on the train fold minus subfold k. -1 off the train fold."""
    train = np.flatnonzero(fold == "train")
    sub = np.full(len(fold), -1)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, out) in enumerate(skf.split(train, cell_type[train])):
        sub[train[out]] = k
    return sub


def num(v: float) -> float | None:
    """A JSON-safe float: NaN (an R^2 not kept on testB) becomes null."""
    return None if np.isnan(v) else float(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TEST_FOLD))
    ap.add_argument("--encoder", default="dinobloom")
    ap.add_argument("--read-test", action="store_true", help="also score the test fold, once")
    args = ap.parse_args()
    ds, test_fold = args.dataset, TEST_FOLD[args.dataset]
    stem = f"linear_unseen_{args.encoder}_{ds}"

    commit = None
    if args.read_test:                                       # before any data is loaded
        commit = frozen_commit()
        prev = RESULTS / f"{stem}.json"
        if prev.exists() and json.loads(prev.read_text())["test_read"]:
            raise RuntimeError(f"{prev.name} already holds a test run; the test fold is read once")

    meta, X = load_embeddings(args.encoder, ds)
    meta = meta[~meta.class_label.isin(EXCLUDED[ds])].reset_index(drop=True)
    E = X[meta.row.to_numpy()].astype(np.float64)
    fold, cell_type = meta.fold.to_numpy(), meta.class_label.to_numpy()
    labels = seed_mean_labels(meta, ds).reindex(meta.cell_id.to_numpy())
    has_label = labels.n_seeds.notna().to_numpy()
    Y, n_seeds = labels[DESCRIPTORS].to_numpy(), labels.n_seeds.to_numpy()
    per_seed = np.full((len(SEG_SEEDS), len(meta), len(DESCRIPTORS)), np.nan)
    row_of = pd.Series(np.arange(len(meta)), index=meta.cell_id)
    for s in SEG_SEEDS:
        lab = labels_for(s, meta, ds)
        per_seed[s, row_of.loc[lab.cell_id].to_numpy()] = lab[DESCRIPTORS].to_numpy()
    rebuilt = float(np.abs(np.nanmean(per_seed[:, has_label], axis=0) - Y[has_label]).max())

    # Five readouts, each fit on 4/5 of the train fold.
    train = np.flatnonzero(fold == "train")
    sub = training_subfolds(fold, cell_type)
    per_sub = pd.crosstab(cell_type[train], sub[train])
    spread = int((per_sub.max(axis=1) - per_sub.min(axis=1)).max())

    parts = {"val": fold == "val"}
    if args.read_test:
        parts[test_fold] = fold == test_fold
    scored = {p: m & has_label for p, m in parts.items()}
    groups = {p: cell_type[m] for p, m in scored.items()}
    n_med = {p: float(np.median(n_seeds[m])) for p, m in scored.items()}
    ceiling = {p: [ceiling_within(per_seed[:, m, i], groups[p], n_med[p]) for i in range(len(DESCRIPTORS))]
               for p, m in scored.items()}

    rows, type_rows, cls_rows, weights, n_iter, worst_zero = [], [], [], [], [], 0.0
    for k in range(K_FOLDS):
        tr = (fold == "train") & (sub != k)
        logistic = LogisticRegression(max_iter=MAX_ITER)
        for name, model in [("logistic", logistic), ("baseline", DummyClassifier(strategy="prior"))]:
            model.fit(E[tr], cell_type[tr])
            for part, m in parts.items():
                cls_rows.append({"model": name, "fold": k, "split": part,
                                 **classification_scores(model, E[m], cell_type[m])})
        n_iter.append(int(logistic.n_iter_[0]))

        trl = tr & has_label
        ols = LinearRegression().fit(E[trl], Y[trl])         # one OLS per descriptor, solved together
        sd_x, sd_y = E[trl].std(0, ddof=1), Y[trl].std(0, ddof=1)
        weights += [pd.DataFrame({"model": "joint_768", "descriptor": d, "dim": np.arange(E.shape[1]),
                                  "fold": k, "w": ols.coef_[i], "beta": ols.coef_[i] * sd_x / sd_y[i]})
                    for i, d in enumerate(DESCRIPTORS)]
        type_mean = pd.DataFrame(Y[trl]).groupby(cell_type[trl]).mean()
        for part, m in scored.items():
            g, P = groups[part], ols.predict(E[m])
            T = type_mean.loc[g].to_numpy()                  # type-only prediction
            base = ((Y[m] - Y[trl].mean(0)) ** 2).mean(0)    # MSE of the training mean
            for i, d in enumerate(DESCRIPTORS):
                y, p = Y[m, i], P[:, i]
                worst_zero = max(worst_zero, abs(r2_within(y, T[:, i], g)))
                row = {"descriptor": d, "fold": k, "split": part, "n": int(m.sum()),
                       "n_seeds_median": n_med[part],
                       "r2_pooled": float(1 - ((y - p) ** 2).mean() / base[i]),
                       "r2_type_only": float(1 - ((y - T[:, i]) ** 2).mean() / base[i]),
                       "r2_within": r2_within(y, p, g), "ceiling_within": ceiling[part][i],
                       "rho_pooled": float(spearmanr(p, y).statistic)}
                if part == "testB" and d not in SCALE_INVARIANT:
                    row.update(dict.fromkeys(R2_STATS, np.nan))   # another imaging scale (§16)
                rows.append(row)
                for c in np.unique(g):
                    sel = g == c
                    if sel.sum() >= MIN_TYPE_N:
                        type_rows.append({"descriptor": d, "fold": k, "split": part, "class_label": c,
                                          "n": int(sel.sum()),
                                          "rho": float(spearmanr(p[sel], y[sel]).statistic)})
        print(f"[linear_unseen] {ds} readout {k}: {trl.sum():,} labelled training cells, logistic n_iter "
              f"{n_iter[-1]}" + ("  NOT CONVERGED" if n_iter[-1] >= MAX_ITER else ""), flush=True)

    if worst_zero > ATOL or rebuilt > ATOL or spread > 1:
        raise RuntimeError(f"check failed: type-only R2_within {worst_zero:.1e}, rebuilt labels {rebuilt:.1e}, "
                           f"class spread over training subfolds {spread}")

    out, types, cls = pd.DataFrame(rows), pd.DataFrame(type_rows), pd.DataFrame(cls_rows)
    agg = out.groupby(["split", "descriptor"], sort=False)[STATS].agg(["mean", "std"])
    by_type = types.groupby(["split", "descriptor", "class_label"]).agg(
        rho_mean=("rho", "mean"), rho_sd=("rho", "std"), n=("n", "first"))
    cls_agg = cls.groupby(["split", "model"], sort=False)[CLS_METRICS].agg(["mean", "std"])
    summary = {
        "analysis": "linear_cv readouts fit per dataset and scored on cells DinoBloom never saw",
        "protocol": "protocol.md §16",
        "encoder": args.encoder, "dataset": ds, "test_fold": test_fold, "test_read": args.read_test,
        "commit": commit, "excluded_classes": EXCLUDED[ds], "never_read": NEVER_READ[ds],
        "cells": {"train": int((fold == "train").sum()), **{p: int(m.sum()) for p, m in parts.items()},
                  "labelled": {p: int(m.sum()) for p, m in scored.items()}},
        "readouts": f"{K_FOLDS}, each fit on 4/5 of the train fold (StratifiedKFold by cell type, seed {SEED})",
        "labels": {"descriptors": "mean over segmenter seeds 0-24 in which the cell passed QC "
                                  "(qc_pass and all descriptors finite); units as in src/morphometry.py",
                   "n_labelled": int(has_label.sum()),
                   "unlabelled_by_class": json.loads(pd.Series(cell_type[~has_label]).value_counts().to_json()),
                   "n_seeds_median": n_med},
        "models": {"regression": "LinearRegression (OLS), one per descriptor; baseline the training mean",
                   "classification": f"LogisticRegression at sklearn defaults (L2, C=1.0, lbfgs), "
                                     f"max_iter={MAX_ITER}; baseline DummyClassifier(strategy='prior')",
                   "features": "embedding as stored, not scaled"},
        "statistics": {"r2_pooled": "1 - MSE / MSE of the training mean",
                       "r2_type_only": "the same for predicting each cell's training type mean",
                       "r2_within": "R^2 after centring target and prediction on the scored fold's type means",
                       "ceiling_within": "Spearman-Brown reliability of the label inside cell types, stepped up "
                                         "to the median number of QC-passing seeds per scored cell",
                       "rho_pooled": "Spearman rho of prediction and label over all scored cells",
                       "rho_by_type": f"Spearman rho inside each cell type with at least {MIN_TYPE_N} cells"},
        "testB": f"R^2 statistics kept only for {', '.join(SCALE_INVARIANT)}; testB is imaged at another scale",
        "aggregation": "mean and sd (ddof=1) over the 5 readouts",
        "descriptors": {p: {d: {s: {"mean": num(agg.loc[(p, d), (s, "mean")]),
                                    "sd": num(agg.loc[(p, d), (s, "std")])} for s in STATS}
                            for d in DESCRIPTORS} for p in parts},
        "rho_by_type": {p: {d: by_type.loc[(p, d)].reset_index().to_dict(orient="records")
                            for d in DESCRIPTORS} for p in parts},
        "class_label": {p: {mod: {mt: {"mean": float(cls_agg.loc[(p, mod), (mt, "mean")]),
                                       "sd": float(cls_agg.loc[(p, mod), (mt, "std")])} for mt in CLS_METRICS}
                            for mod in ["logistic", "baseline"]} for p in parts},
        "logistic_n_iter": n_iter, "logistic_converged": [it < MAX_ITER for it in n_iter],
        "checks": {"max_abs_r2_within_of_type_only": worst_zero, "max_abs_rebuilt_seed_mean_label": rebuilt,
                   "max_class_spread_over_training_subfolds": spread},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "host": platform.node(),
    }
    out.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    types.to_parquet(RESULTS / f"{stem}_by_type.parquet", index=False)
    cls.to_parquet(RESULTS / f"{stem}_class.parquet", index=False)
    pd.concat(weights, ignore_index=True).to_parquet(
        RESULTS / f"linear_unseen_weights_{args.encoder}_{ds}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))

    for p in parts:
        tab = pd.DataFrame({s: [f"{agg.loc[(p, d), (s, 'mean')]:.3f} ± {agg.loc[(p, d), (s, 'std')]:.3f}"
                                for d in DESCRIPTORS] for s in STATS}, index=DESCRIPTORS)
        acc = summary["class_label"][p]["logistic"]
        print(f"\n{ds} {p}: {summary['cells']['labelled'][p]:,} labelled cells, mean ± sd over {K_FOLDS} readouts")
        print(tab.to_string())
        print(f"cell type: accuracy {acc['accuracy']['mean']:.3f} ± {acc['accuracy']['sd']:.3f}, "
              f"balanced {acc['balanced_accuracy']['mean']:.3f} ± {acc['balanced_accuracy']['sd']:.3f}")
    print(f"\n[linear_unseen] checks: {json.dumps(summary['checks'])}")
    print(f"[linear_unseen] wrote {RESULTS / stem}.parquet / _by_type / _class, the weights, and .json")


if __name__ == "__main__":
    main()
