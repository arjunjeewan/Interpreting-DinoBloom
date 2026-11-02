"""Top-, bottom- and random-K ablation of linear_cv's OLS: validation R^2 against K.

Companion to linear_cv.py, linear_cv_dims.py and linear_cv_weights.py, over the
same five folds and seed-mean labels. In every fold the 768 embedding positions
are put in an order and, for K = 1 .. 768, each descriptor is fitted by OLS on
the first K positions of that order and scored by R^2 on the fold's validation
cells:

    top      positions by |r| with the descriptor, strongest first
    bottom   the same ranking, weakest first
    random   10 seeded random orders, shared by all folds and descriptors

r is the Pearson correlation of a position with the descriptor on the fold's
training cells, the single beta of linear_cv_weights.py. Keeping the top K is
the same model as removing the bottom 768 - K, so each curve read from its
right end is the removal ablation of the other end.

R^2 = 1 - SSE / SSE(training mean), as in linear_cv.py. Fits and scores come
from Gram matrices: G = Xc'Xc and c = Xc'y on the training cells, centred on
their means, and the same products on the validation cells, centred on the
training means, give the validation SSE of every subset without predicting.
K = 768 is linear_cv's own model; its Gram matrix is singular (the LayerNorm
null direction, linear_cv_weights.py), so its R^2 is read from linear_cv's
parquet.

Checks: the training-mean SSE must reproduce linear_cv's baseline; at K = 767
every order has dropped one position, which costs nothing, so all orders must
reproduce linear_cv's validation MSE; random (order, descriptor, K) subsets are
refitted with sklearn; and the ranking's r must equal linear_cv_weights' single
beta. Weights of the top-K models are stored for K = 1, 2, 4, ..., 64.

Only the validation folds are scored; the 20% hold-out is not read.

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
from sklearn.metrics import mean_squared_error

from src.analysis.linear_cv import K_FOLDS, SPLITS, check_splits, seed_mean_labels, splits_for
from src.analysis.raw_probe import DATASET, RESULTS, load_embeddings
from src.env import ROOT, go_offline
from src.morphometry import DESCRIPTORS

go_offline()

SEED = 0
N_RANDOM = 10                         # random orders, shared by all folds and descriptors
WEIGHT_K = [1, 2, 4, 8, 16, 32, 64]   # top-K models whose weights are stored
N_SPOT = 4                            # random (order, descriptor, K) per fold refitted with sklearn
RTOL_BASELINE = 1e-12                 # training-mean SSE against linear_cv's baseline MSE
RTOL_DROP_ONE = 1e-6                  # K = 767 against linear_cv's validation MSE
RTOL_SPOT = 1e-9                      # Gram solution against sklearn, validation MSE
ATOL_R = 1e-12                        # ranking r against linear_cv_weights' single beta


def subset_sse(G, c, H, h, yy, S):
    """OLS on positions S from the training Gram matrices; validation SSE per descriptor, and weights."""
    w = np.linalg.solve(G[np.ix_(S, S)], c[S])  # K x descriptors
    return yy - 2 * (w * h[S]).sum(axis=0) + (w * (H[np.ix_(S, S)] @ w)).sum(axis=0), w


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
    print(f"[linear_cv_ablation] {args.encoder}: {E.shape[0]:,} x {n_dims}, "
          f"{len(DESCRIPTORS)} descriptors, {K_FOLDS} folds, {N_RANDOM} random orders", flush=True)

    ref = pd.read_parquet(RESULTS / f"linear_cv_{args.encoder}_{DATASET}.parquet")
    ref = ref[ref.split == "val"].set_index(["model", "target", "fold"]).mse
    single = pd.read_parquet(RESULTS / f"linear_cv_weights_{args.encoder}_{DATASET}.parquet")
    single = single[single.model == "single"].set_index(["descriptor", "fold", "dim"]).beta
    rng = np.random.default_rng(SEED)
    random_orders = [rng.permutation(n_dims) for _ in range(N_RANDOM)]
    rows, weights = [], []
    worst = {"baseline": 0.0, "drop_one": 0.0, "spot": 0.0, "r": 0.0}
    for k in range(K_FOLDS):
        t = time.time()
        tr = (fold >= 0) & (fold != k) & has_label
        va = (fold == k) & has_label
        mx, my = E[tr].mean(axis=0), Y[tr].mean(axis=0)
        Xc, Yc = E[tr] - mx, Y[tr] - my
        Xv, Yv = E[va] - mx, Y[va] - my
        G, c = Xc.T @ Xc, Xc.T @ Yc  # training: positions x positions, positions x descriptors
        H, h = Xv.T @ Xv, Xv.T @ Yv  # validation, centred on the training means
        yy = (Yv ** 2).sum(axis=0)   # validation SSE of the training-mean prediction
        r = c / np.sqrt(np.outer(np.diag(G), (Yc ** 2).sum(axis=0)))  # Pearson r, positions x descriptors
        sd_x, sd_y = np.sqrt(np.diag(G) / (tr.sum() - 1)), Yc.std(axis=0, ddof=1)

        orders = []  # (name, draw, descriptor index or None for all, positions in order)
        for i, d in enumerate(DESCRIPTORS):
            worst["baseline"] = max(worst["baseline"], abs(yy[i] / va.sum() / ref.loc[("baseline", d, k)] - 1))
            worst["r"] = max(worst["r"], np.abs(r[:, i] - single.loc[(d, k)].sort_index().to_numpy()).max())
            rank = np.argsort(-np.abs(r[:, i]), kind="stable")
            orders += [("top", -1, i, rank), ("bottom", -1, i, rank[::-1])]
        orders += [("random", s, None, o) for s, o in enumerate(random_orders)]

        for name, draw, i, order in orders:
            sse = np.empty((n_dims, len(DESCRIPTORS)))
            for K in range(1, n_dims):
                sse[K - 1], w = subset_sse(G, c, H, h, yy, order[:K])
                if name == "top" and K in WEIGHT_K:
                    S = order[:K]
                    weights.append(pd.DataFrame({"descriptor": DESCRIPTORS[i], "fold": k, "k": K,
                                                 "rank": np.arange(1, K + 1), "dim": S,
                                                 "w": w[:, i], "beta": w[:, i] * sd_x[S] / sd_y[i]}))
            sse[n_dims - 1] = [ref.loc[("linear", d, k)] * va.sum() for d in DESCRIPTORS]
            for j in (range(len(DESCRIPTORS)) if i is None else [i]):
                worst["drop_one"] = max(worst["drop_one"], abs(sse[n_dims - 2, j] / sse[n_dims - 1, j] - 1))
                rows.append(pd.DataFrame({"descriptor": DESCRIPTORS[j], "order": name, "draw": draw,
                                          "fold": k, "k": np.arange(1, n_dims + 1), "dim": order,
                                          "r2": 1 - sse[:, j] / yy[j]}))

        for _ in range(N_SPOT):
            name, draw, i, order = orders[int(rng.integers(len(orders)))]
            j = int(rng.integers(len(DESCRIPTORS))) if i is None else i
            S = order[:int(rng.integers(1, n_dims))]
            lr = LinearRegression().fit(E[tr][:, S], Y[tr, j])
            mse = mean_squared_error(Y[va, j], lr.predict(E[va][:, S]))
            worst["spot"] = max(worst["spot"], abs(subset_sse(G, c, H, h, yy, S)[0][j] / va.sum() / mse - 1))
        print(f"[linear_cv_ablation] fold {k}: {time.time() - t:.0f}s, {len(orders)} orders; "
              f"worst so far {json.dumps({n: f'{v:.1e}' for n, v in worst.items()})}", flush=True)

    for name, tol in (("baseline", RTOL_BASELINE), ("drop_one", RTOL_DROP_ONE),
                      ("spot", RTOL_SPOT), ("r", ATOL_R)):
        if worst[name] > tol:
            raise RuntimeError(f"{name} check failed: {worst[name]:.2e} > {tol:.0e}")

    rows = pd.concat(rows, ignore_index=True)
    weights = pd.concat(weights, ignore_index=True)
    summary = {
        "analysis": "OLS on the first K positions of an order, K = 1..768; validation R^2",
        "encoder": args.encoder, "dataset": DATASET, "n_dims": n_dims, "test_read": False,
        "splits": {"file": str(SPLITS.relative_to(ROOT)), "k_folds": K_FOLDS,
                   "scored": "the 5 validation folds; the 20% hold-out is not read"},
        "labels": "as linear_cv.py: per cell, the mean of each descriptor over segmenter seeds "
                  "0-24 in which the cell passed QC",
        "orders": {"top": "|Pearson r| with the descriptor on the fold's training cells, strongest first",
                   "bottom": "the same ranking, weakest first",
                   "random": f"{N_RANDOM} orders from np.random.default_rng({SEED}), shared by all "
                             f"folds and descriptors"},
        "score": "R^2 = 1 - SSE / SSE(training mean) on the validation fold",
        "k_768": "linear_cv's validation MSE (its Gram matrix is singular)",
        "columns": {"dim": "the position added at step k", "draw": "random order index; -1 for top and bottom"},
        "weights": {"file": f"linear_cv_ablation_weights_{args.encoder}_{DATASET}.parquet",
                    "models": f"top-K, K in {WEIGHT_K}",
                    "beta": "w * sd(x_j) / sd(y), sds (ddof=1) on the fold's training cells"},
        "checks": {"max_rel_baseline_sse_vs_linear_cv": worst["baseline"],
                   "max_rel_k767_mse_vs_linear_cv": worst["drop_one"],
                   "max_rel_val_mse_vs_sklearn": worst["spot"], "n_sklearn_refits": K_FOLDS * N_SPOT,
                   "max_abs_r_vs_linear_cv_weights": worst["r"]},
        "aggregation": "one row per fold model; summarize as mean and sd (ddof=1) over the 5 folds",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "host": platform.node(),
    }
    stem = f"linear_cv_ablation_{args.encoder}_{DATASET}"
    rows.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    weights.to_parquet(RESULTS / summary["weights"]["file"], index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"[linear_cv_ablation] checks: {json.dumps(summary['checks'])}")
    print(f"[linear_cv_ablation] wrote {RESULTS / stem}.parquet / .json and {summary['weights']['file']}")


if __name__ == "__main__":
    main()
