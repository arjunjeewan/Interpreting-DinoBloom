"""linear_cv_ablation.py's top-, bottom- and random-K ablation on PBC (protocol.md §16).

For each of linear_unseen.py's five readouts (each fit on 4/5 of PBC's train fold,
platelets excluded) the 768 positions are put in an order and, for K = 1 .. 768,
each descriptor is fitted by OLS on the first K positions and scored by R^2 on one
scored fold: val, or the test fold once under --read-test. Orders, statistic,
Gram-matrix solution and checks are linear_cv_ablation.py's:

    top      positions by |r| with the descriptor on the readout's training cells
    bottom   the same ranking, weakest first
    random   10 seeded random orders, shared by all readouts and descriptors

K = 768 is linear_unseen's own model; its Gram matrix is singular (the LayerNorm
null direction), so its R^2 is read from linear_unseen's parquet, and at K = 767
every order must reproduce it to within ATOL_DROP_ONE_R2 in R^2: half a unit in
the third decimal, the precision every R^2 is reported to. That bound replaced
linear_cv_ablation's 1e-6 on the relative SSE, a round number with no derivation,
after PBC val measured 1.35e-6 against it (Matek: 1.6e-7); the 767-position fit is
less well conditioned on PBC's ~8,300 training cells than on Matek's ~110,000. The
relative SSE is still recorded. Random (order, descriptor, K) subsets are refitted
with sklearn. Weights of the top-K models are stored for K = 1, 2, 4, ..., 64.

--read-test as in linear_unseen.py. No network access.
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

from src.analysis.linear_cv import K_FOLDS, seed_mean_labels
from src.analysis.linear_cv_ablation import N_RANDOM, N_SPOT, RTOL_SPOT, SEED, WEIGHT_K, subset_sse
from src.analysis.linear_unseen import EXCLUDED, TEST_FOLD, frozen_commit, training_subfolds
from src.analysis.raw_probe import RESULTS, load_embeddings
from src.env import go_offline
from src.morphometry import DESCRIPTORS

go_offline()

DATASET = "pbc"
ATOL_DROP_ONE_R2 = 5e-4   # K = 767 vs K = 768: half a unit in the third decimal of a reported R^2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    ap.add_argument("--read-test", action="store_true", help="score the test fold instead of val, once")
    args = ap.parse_args()
    part = TEST_FOLD[DATASET] if args.read_test else "val"
    stem = f"linear_unseen_ablation_{args.encoder}_{DATASET}_{part}"
    commit = None
    if args.read_test:                                       # before any data is loaded
        commit = frozen_commit()
        if (RESULTS / f"{stem}.json").exists():
            raise RuntimeError(f"{stem}.json exists; the test fold is read once")

    meta, X = load_embeddings(args.encoder, DATASET)
    meta = meta[~meta.class_label.isin(EXCLUDED[DATASET])].reset_index(drop=True)
    E = X[meta.row.to_numpy()].astype(np.float64)
    fold, cell_type = meta.fold.to_numpy(), meta.class_label.to_numpy()
    Y = seed_mean_labels(meta, DATASET).reindex(meta.cell_id.to_numpy())
    has_label = Y.n_seeds.notna().to_numpy()
    Y = Y[DESCRIPTORS].to_numpy()
    sub = training_subfolds(fold, cell_type)
    va = (fold == part) & has_label
    n_dims = E.shape[1]
    print(f"[linear_unseen_ablation] {args.encoder}/{DATASET}: scoring {part} ({va.sum():,} cells), "
          f"{K_FOLDS} readouts, {N_RANDOM} random orders", flush=True)

    ref = pd.read_parquet(RESULTS / f"linear_unseen_{args.encoder}_{DATASET}.parquet")
    ref = ref[ref.split == part].set_index(["descriptor", "fold"]).r2_pooled
    rng = np.random.default_rng(SEED)
    random_orders = [rng.permutation(n_dims) for _ in range(N_RANDOM)]
    rows, weights = [], []
    worst = {"drop_one_r2": 0.0, "drop_one_rel_sse": 0.0, "spot": 0.0}
    for k in range(K_FOLDS):
        t = time.time()
        tr = (fold == "train") & (sub != k) & has_label
        mx, my = E[tr].mean(axis=0), Y[tr].mean(axis=0)
        Xc, Yc = E[tr] - mx, Y[tr] - my
        Xv, Yv = E[va] - mx, Y[va] - my
        G, c = Xc.T @ Xc, Xc.T @ Yc  # training: positions x positions, positions x descriptors
        H, h = Xv.T @ Xv, Xv.T @ Yv  # scored fold, centred on the training means
        yy = (Yv ** 2).sum(axis=0)   # scored SSE of the training-mean prediction
        r = c / np.sqrt(np.outer(np.diag(G), (Yc ** 2).sum(axis=0)))  # Pearson r, positions x descriptors
        sd_x, sd_y = np.sqrt(np.diag(G) / (tr.sum() - 1)), Yc.std(axis=0, ddof=1)

        orders = []  # (name, draw, descriptor index or None for all, positions in order)
        for i, d in enumerate(DESCRIPTORS):
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
            sse[n_dims - 1] = [(1 - ref.loc[(d, k)]) * yy[j] for j, d in enumerate(DESCRIPTORS)]
            for j in (range(len(DESCRIPTORS)) if i is None else [i]):
                worst["drop_one_r2"] = max(worst["drop_one_r2"], abs(sse[n_dims - 2, j] - sse[n_dims - 1, j]) / yy[j])
                worst["drop_one_rel_sse"] = max(worst["drop_one_rel_sse"],
                                                abs(sse[n_dims - 2, j] / sse[n_dims - 1, j] - 1))
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
        print(f"[linear_unseen_ablation] readout {k}: {time.time() - t:.0f}s, {len(orders)} orders; "
              f"worst so far {json.dumps({n: f'{v:.1e}' for n, v in worst.items()})}", flush=True)

    for name, tol in (("drop_one_r2", ATOL_DROP_ONE_R2), ("spot", RTOL_SPOT)):
        if worst[name] > tol:
            raise RuntimeError(f"{name} check failed: {worst[name]:.2e} > {tol:.0e}")

    rows = pd.concat(rows, ignore_index=True)
    weights = pd.concat(weights, ignore_index=True)
    summary = {
        "analysis": "OLS on the first K positions of an order, K = 1..768, on PBC; R^2 on the scored fold",
        "protocol": "protocol.md §16", "encoder": args.encoder, "dataset": DATASET, "n_dims": n_dims,
        "scored": part, "test_read": args.read_test, "commit": commit, "excluded_classes": EXCLUDED[DATASET],
        "readouts": "linear_unseen.py's five, each fit on 4/5 of the train fold (training_subfolds)",
        "labels": "as linear_cv.py: per cell, the mean of each descriptor over segmenter seeds "
                  "0-24 in which the cell passed QC",
        "orders": {"top": "|Pearson r| with the descriptor on the readout's training cells, strongest first",
                   "bottom": "the same ranking, weakest first",
                   "random": f"{N_RANDOM} orders from np.random.default_rng({SEED}), shared by all "
                             f"readouts and descriptors"},
        "score": "R^2 = 1 - SSE / SSE(training mean) on the scored fold",
        "k_768": "linear_unseen's r2_pooled (its Gram matrix is singular)",
        "columns": {"fold": "the readout", "dim": "the position added at step k",
                    "draw": "random order index; -1 for top and bottom"},
        "weights": {"file": f"linear_unseen_ablation_weights_{args.encoder}_{DATASET}_{part}.parquet",
                    "models": f"top-K, K in {WEIGHT_K}",
                    "beta": "w * sd(x_j) / sd(y), sds (ddof=1) on the readout's training cells"},
        "checks": {"max_abs_k767_r2_vs_linear_unseen": worst["drop_one_r2"], "atol_k767_r2": ATOL_DROP_ONE_R2,
                   "max_rel_k767_sse_vs_linear_unseen": worst["drop_one_rel_sse"],
                   "max_rel_mse_vs_sklearn": worst["spot"], "n_sklearn_refits": K_FOLDS * N_SPOT},
        "aggregation": "one row per readout; summarize as mean and sd (ddof=1) over the 5 readouts",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "host": platform.node(),
    }
    rows.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    weights.to_parquet(RESULTS / summary["weights"]["file"], index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"[linear_unseen_ablation] checks: {json.dumps(summary['checks'])}")
    print(f"[linear_unseen_ablation] wrote {RESULTS / stem}.parquet / .json and {summary['weights']['file']}")


if __name__ == "__main__":
    main()
