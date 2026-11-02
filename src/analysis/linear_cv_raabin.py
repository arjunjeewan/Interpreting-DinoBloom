"""Do Matek's top single positions carry their metrics on Raabin? Pooled and within cell type.

Fig. 1 marks each metric's 8 positions of largest |r| on Matek (linear_cv_weights.py's
single-position fits; the same 8 in every fold). This asks whether those
correlations belong to the positions or to the Matek population, on a second
dataset with ground-truth masks (Raabin-WBC: mature white cells only; 4 types in
the gt arm, which has no basophils, and 5 in the unet arm).

For each (metric, top-8 position) pair: Pearson r of the stored CLS embedding at
that position with the metric, on Raabin's project train fold only, in two arms,

    gt      the train-fold cells with nucleus and cytoplasm ground truth; metrics
            from the dataset authors' masks (morphometry_raabin_gt.parquet)
    unet    every train-fold cell with labels from our segmenter: the mean over its
            QC-passing seeds, the same masks Matek's labels come from

each pooled and within cell type (embedding and metric centred on the arm's
cell-type means). Summaries, with no cutoff: Pearson and Spearman r between the
Matek r and each Raabin r across the 88 pairs, and how many of the 88 keep their
sign. Example: position 161 x nuclear area, with per-type r and type means in the
gt arm.

Raabin's val, test, testA and testB folds are not read, nor is Matek's hold-out.

No network access.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr

from src.analysis.raw_probe import RESULTS
from src.embed import EMB
from src.env import go_offline
from src.morphometry import DESCRIPTORS

go_offline()

TOP_K = 8
SEG_SEEDS = range(25)
FOLD = "train"                     # the only Raabin fold read
EXAMPLE = ("nuclear_area", 161)


def corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r of each column of X with y."""
    Xc, yc = X - X.mean(0), y - y.mean()
    return (Xc * yc[:, None]).sum(0) / np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())


def centred(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean, row-wise; a is 1-D or 2-D."""
    f = pd.DataFrame(a.reshape(len(a), -1))
    return (f - f.groupby(groups).transform("mean")).to_numpy().reshape(a.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    args = ap.parse_args()

    # Matek: each metric's top-8 positions and their fold-mean single-position r.
    abl = pd.read_parquet(RESULTS / f"linear_cv_ablation_{args.encoder}_matek_bm.parquet",
                          columns=["descriptor", "order", "fold", "k", "dim"])
    top = abl[(abl.order == "top") & (abl.k <= TOP_K)]
    if (top.groupby(["descriptor", "fold"]).dim.apply(frozenset).groupby(level=0).nunique() != 1).any():
        raise RuntimeError(f"the top-{TOP_K} sets differ between folds")
    top8 = {d: top[(top.descriptor == d) & (top.fold == 0)].sort_values("k").dim.to_numpy() for d in DESCRIPTORS}
    if EXAMPLE[1] not in top8[EXAMPLE[0]]:
        raise RuntimeError(f"position {EXAMPLE[1]} is not among {EXAMPLE[0]}'s top {TOP_K}")
    w = pd.read_parquet(RESULTS / f"linear_cv_weights_{args.encoder}_matek_bm.parquet")
    r_matek = w[w.model == "single"].groupby(["descriptor", "dim"]).beta.mean()

    # Raabin: stored embeddings, project train fold; the two label arms.
    meta = pd.read_parquet(EMB / args.encoder / "raabin.parquet")
    E = np.load(EMB / args.encoder / "raabin.npy", mmap_mode="r")
    meta = meta.assign(row=np.arange(len(meta)), stem=meta.cell_id.map(lambda c: Path(c).stem))
    train = meta[(meta.fold == FOLD) & meta.ok]
    gt = pd.read_parquet(RESULTS / "morphometry_raabin_gt.parquet")
    gt = gt.assign(stem=gt.cell_id.map(lambda c: Path(c).stem))[["stem", "class_label"] + DESCRIPTORS]
    gt_arm = train.drop(columns="cell_id").merge(gt, on="stem", suffixes=("", "_gt"))
    long = []
    for s in SEG_SEEDS:
        m = pd.read_parquet(RESULTS / f"morphometry_raabin_seed{s}.parquet")
        long.append(m[m.qc_pass & m[DESCRIPTORS].notna().all(axis=1)][["cell_id"] + DESCRIPTORS])
    unet = pd.concat(long, ignore_index=True).groupby("cell_id")[DESCRIPTORS].mean()
    unet_arm = train.merge(unet, left_on="cell_id", right_index=True)

    arms = {}
    for name, arm in [("gt", gt_arm), ("unet", unet_arm)]:
        X = np.asarray(E[arm.row.to_numpy()], np.float64)
        g = arm.class_label.to_numpy()          # the dataset index's labels, in both arms
        arms[name] = {"X": X, "Xw": centred(X, g), "Y": arm[DESCRIPTORS].to_numpy(np.float64), "g": g,
                      "n": len(arm), "types": {c: int(n) for c, n in arm.class_label.value_counts().items()}}

    rows = []
    for i, d in enumerate(DESCRIPTORS):
        dims = top8[d]
        rec = {}
        for name, a in arms.items():
            y = a["Y"][:, i]
            rec[f"r_{name}_pooled"] = corr(a["X"][:, dims], y)
            rec[f"r_{name}_within"] = corr(a["Xw"][:, dims], centred(y, a["g"]))
        for k, j in enumerate(dims):
            rows.append({"descriptor": d, "rank": k + 1, "dim": int(j), "r_matek": float(r_matek[(d, j)]),
                         **{c: float(v[k]) for c, v in rec.items()}})
    t = pd.DataFrame(rows)
    raabin = [c for c in t.columns if c.startswith("r_") and c != "r_matek"]
    agreement = {c: {"pearson_with_matek": float(pearsonr(t.r_matek, t[c])[0]),
                     "spearman_with_matek": float(spearmanr(t.r_matek, t[c])[0]),
                     "same_sign_as_matek": int((np.sign(t[c]) == np.sign(t.r_matek)).sum()),
                     "pairs": len(t)} for c in raabin}

    d, j = EXAMPLE
    a = arms["gt"]
    x, y = a["X"][:, j], a["Y"][:, DESCRIPTORS.index(d)]
    pair = t[(t.descriptor == d) & (t.dim == j)].iloc[0]
    example = {"descriptor": d, "position": j, "r_matek": float(pair.r_matek), **{c: float(pair[c]) for c in raabin},
               "gt_by_type": {c: {"n": int((a["g"] == c).sum()),
                                  "r": float(np.corrcoef(x[a["g"] == c], y[a["g"] == c])[0, 1]),
                                  "mean_position": float(x[a["g"] == c].mean()),
                                  f"mean_{d}": float(y[a["g"] == c].mean())} for c in sorted(set(a["g"]))}}

    summary = {
        "analysis": "Matek's top-8 single-position correlations per metric, re-measured on Raabin, pooled and "
                    "within cell type, with ground-truth and U-Net masks",
        "encoder": args.encoder,
        "matek": f"top-{TOP_K} positions per metric from linear_cv_ablation (identical in all folds); r = the "
                 "fold-mean single-position beta of linear_cv_weights; hold-out not read",
        "raabin_folds_read": [FOLD], "embeddings": "stored CLS (data/embeddings), the project's preprocessing",
        "arms": {name: {"n": a["n"], "cell_types": a["types"]} for name, a in arms.items()},
        "gt_class_label_mismatches_vs_index": int((gt_arm.class_label != gt_arm.class_label_gt).sum()),
        "agreement_with_matek": agreement,
        "example": example,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "host": platform.node(),
    }
    stem = f"linear_cv_raabin_{args.encoder}"
    t.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    print(f"[linear_cv_raabin] arms: {json.dumps(summary['arms'])}")
    print(pd.DataFrame(agreement).T.round(3).to_string())
    print(f"[linear_cv_raabin] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
