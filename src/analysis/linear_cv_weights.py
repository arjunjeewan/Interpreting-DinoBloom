"""Per-position weights of linear models of each morphometric descriptor.

Companion to linear_cv.py and linear_cv_dims.py, over the same five folds and
the same seed-mean labels. For each descriptor y it records one weight per
embedding position j, from three fits on the four training folds of each fold:

    joint_768   y ~ a + w . E            linear_cv's OLS on all positions,
                                         weights exactly as sklearn returns them
    joint_767   y ~ a + w . E[:, keep]   the same, refitted without one position
    single      y ~ a + b E[:, j]        one OLS per position (linear_cv_dims)

A joint weight is a position's contribution with every other position held
fixed; a single slope is the position on its own. Each weight is stored raw
(descriptor units per embedding unit) and standardized,
beta = w * sd(x_j) / sd(y), both sds on the fold's training cells. For single,
beta is exactly the Pearson correlation of position j with y.

Why two joint fits. The embedding is the encoder's final LayerNorm output, so
every cell satisfies sum_j (x_j - shift_j) / scale_j = 0. The centred embedding
therefore has one exact null direction, and joint weights are defined only up
to adding a multiple of it: no prediction changes. sklearn's rank cutoff drops
that direction (rank 767) and returns the smallest-norm weights. Refitting
without the position that carries most of the direction makes the weights
unique; the predictions are the same model's. The direction is found from the
data in every fold; where the LayerNorm scale is known (LN_SCALE), it is also
compared with 1/scale, the direction the equation above predicts. Per fold, the
JSON also records the dropped position's spread (training sd) against the
median over positions, and, once, the scale's median magnitude.

Only training cells are used. The validation folds serve one check: both joint
fits must reproduce linear_cv.py's validation MSE, which pins the cells, labels
and model to linear_cv's. The 20% hold-out is not read.

No network access.
"""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from safetensors.numpy import load_file
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.analysis.linear_cv import K_FOLDS, SPLITS, check_splits, seed_mean_labels, splits_for
from src.analysis.raw_probe import DATASET, RESULTS, load_embeddings
from src.env import ROOT, go_offline
from src.morphometry import DESCRIPTORS

go_offline()

SEED = 0
N_SPOT = 10          # random (descriptor, position) pairs per fold checked for single
RTOL_SAME = 1e-12    # joint_768 is linear_cv's model: validation MSE must match
RTOL_REFIT = 1e-6    # joint_767 predictions differ only along the null direction
ATOL_BETA = 1e-12    # single, on the beta scale: closed form vs sklearn, and vs np.corrcoef
# The encoder's final LayerNorm scale (gamma): checkpoint and key, where known.
LN_SCALE = {"dinobloom": ("data/weights/timm_vit_base_patch14_224.dinobloom/model.safetensors",
                          "norm.weight")}


def null_direction(Xc: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit eigenvector of the smallest eigenvalue of Xc'Xc, and that eigenvalue over the largest."""
    lam, V = np.linalg.eigh(Xc.T @ Xc)  # ascending
    return V[:, 0], float(lam[0] / lam[-1])


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
    print(f"[linear_cv_weights] {args.encoder}: {E.shape[0]:,} x {n_dims}, "
          f"{len(DESCRIPTORS)} descriptors, {K_FOLDS} folds", flush=True)

    scale = None
    if args.encoder in LN_SCALE:
        path, key = LN_SCALE[args.encoder]
        scale = load_file(str(ROOT / path))[key].astype(np.float64)

    ref = pd.read_parquet(RESULTS / f"linear_cv_{args.encoder}_{DATASET}.parquet")
    ref = ref[(ref.split == "val") & (ref.model == "linear")].set_index(["target", "fold"]).mse
    rng = np.random.default_rng(SEED)
    rows, null, ranks = [], [], set()
    worst = {"joint_768": 0.0, "joint_767": 0.0, "single": 0.0, "pearson": 0.0}
    for k in range(K_FOLDS):
        tr = (fold >= 0) & (fold != k) & has_label
        va = (fold == k) & has_label
        Etr, Eva, Ytr = E[tr], E[va], Y[tr]
        Xc, Yc = Etr - Etr.mean(axis=0), Ytr - Ytr.mean(axis=0)
        sd_x, sd_y = Xc.std(axis=0, ddof=1), Yc.std(axis=0, ddof=1)

        v, ratio = null_direction(Xc)
        drop = int(np.argmax(v ** 2))
        keep = np.arange(n_dims) != drop
        Etr_keep, Eva_keep = Etr[:, keep], Eva[:, keep]
        null.append({"fold": k, "eigenvalue_ratio": ratio, "dropped_position": drop,
                     "share_on_dropped": float(v[drop] ** 2),
                     "sd_at_dropped": float(sd_x[drop]), "sd_median": float(np.median(sd_x)),
                     "sd_rank_of_dropped": int((sd_x < sd_x[drop]).sum()) + 1})
        if scale is not None:
            null[-1]["ln_scale_at_dropped"] = float(scale[drop])
            null[-1]["abs_cos_with_inverse_ln_scale"] = float(abs(v @ (1 / scale)) / np.linalg.norm(1 / scale))

        W = {"single": (Xc.T @ Yc / (Xc ** 2).sum(axis=0)[:, None]).T,  # descriptors x positions
             "joint_768": np.empty((len(DESCRIPTORS), n_dims)),
             "joint_767": np.full((len(DESCRIPTORS), n_dims), np.nan)}
        for i, d in enumerate(DESCRIPTORS):
            full = LinearRegression().fit(Etr, Ytr[:, i])
            part = LinearRegression().fit(Etr_keep, Ytr[:, i])
            W["joint_768"][i], W["joint_767"][i, keep] = full.coef_, part.coef_
            ranks.add((full.rank_, part.rank_))
            for name, model, Ev in (("joint_768", full, Eva), ("joint_767", part, Eva_keep)):
                mse = mean_squared_error(Y[va, i], model.predict(Ev))
                worst[name] = max(worst[name], abs(mse / ref.loc[(d, k)] - 1))

        # single on the beta scale, against sklearn and np.corrcoef: random pairs,
        # plus the position with the smallest spread.
        pairs = [(int(rng.integers(len(DESCRIPTORS))), int(rng.integers(n_dims))) for _ in range(N_SPOT)]
        pairs.append((int(rng.integers(len(DESCRIPTORS))), int(np.argmin(sd_x))))
        for i, j in pairs:
            beta = W["single"][i, j] * sd_x[j] / sd_y[i]
            b = LinearRegression().fit(Etr[:, j:j + 1], Ytr[:, i]).coef_[0]
            worst["single"] = max(worst["single"], abs(beta - b * sd_x[j] / sd_y[i]))
            worst["pearson"] = max(worst["pearson"], abs(beta - np.corrcoef(Etr[:, j], Ytr[:, i])[0, 1]))

        for name, w in W.items():
            for i, d in enumerate(DESCRIPTORS):
                rows.append(pd.DataFrame({"model": name, "descriptor": d, "dim": np.arange(n_dims),
                                          "fold": k, "w": w[i], "beta": w[i] * sd_x / sd_y[i]}))
        print(f"[linear_cv_weights] fold {k}: train {tr.sum():,}; null direction "
              f"{ratio:.1e} of the largest eigenvalue, {v[drop] ** 2:.1%} on position {drop}", flush=True)

    if ranks != {(n_dims - 1, n_dims - 1)}:
        raise RuntimeError(f"expected rank {n_dims - 1} for both joint fits, got {sorted(ranks)}")
    if len({n["dropped_position"] for n in null}) != 1:
        raise RuntimeError("the null direction's main position differs between folds")
    for name, tol in (("joint_768", RTOL_SAME), ("joint_767", RTOL_REFIT),
                      ("single", ATOL_BETA), ("pearson", ATOL_BETA)):
        if worst[name] > tol:
            raise RuntimeError(f"{name} check failed: {worst[name]:.2e} > {tol:.0e}")

    rows = pd.concat(rows, ignore_index=True)
    summary = {
        "analysis": "per-position weights of linear models of each descriptor",
        "encoder": args.encoder, "dataset": DATASET, "n_dims": n_dims, "test_read": False,
        "splits": {"file": str(SPLITS.relative_to(ROOT)), "k_folds": K_FOLDS,
                   "used": "each fold's training cells; validation folds only for the MSE check; "
                           "the 20% hold-out is not read"},
        "labels": "as linear_cv.py: per cell, the mean of each descriptor over segmenter seeds "
                  "0-24 in which the cell passed QC",
        "models": {"joint_768": "OLS on all positions (sklearn LinearRegression), weights as returned",
                   "joint_767": f"OLS refitted without position {null[0]['dropped_position']}, "
                                f"the main position of the null direction; NaN there",
                   "single": "y ~ a + b x_j, one OLS per position, closed form"},
        "columns": {"w": "raw weight: descriptor units per embedding unit",
                    "beta": "w * sd(x_j) / sd(y), sds (ddof=1) on the fold's training cells; "
                            "for single, the Pearson correlation"},
        "null_direction": null,
        "ln_scale_source": (dict(zip(["file", "key"], LN_SCALE[args.encoder]))
                            if scale is not None else "not known for this encoder; not compared"),
        "ln_scale_summary": ({"median_abs": float(np.median(np.abs(scale))),
                              "position_of_smallest_abs": int(np.argmin(np.abs(scale)))}
                             if scale is not None else None),
        "checks": {"rank_both_joint_fits": n_dims - 1,
                   "max_rel_val_mse_diff_vs_linear_cv": {"joint_768": worst["joint_768"],
                                                         "joint_767": worst["joint_767"]},
                   "max_abs_beta_err_single_vs_sklearn": worst["single"],
                   "max_abs_beta_err_single_vs_pearson": worst["pearson"],
                   "n_single_pairs": K_FOLDS * (N_SPOT + 1)},
        "aggregation": "one row per fold model; summarize as mean and sd (ddof=1) over the 5 folds",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "host": platform.node(),
    }
    stem = f"linear_cv_weights_{args.encoder}_{DATASET}"
    rows.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"[linear_cv_weights] checks: {json.dumps(summary['checks'])}")
    print(f"[linear_cv_weights] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
