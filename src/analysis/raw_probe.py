"""Linear readout of morphometry from the raw embedding, before any SAE.

Question: how much of each morphometric descriptor can a linear head read off
the 768-d DinoBloom CLS embedding directly? If the raw embedding already
carries it, the SAE has to earn its place by *localizing* the signal into
single features, not by finding it.

Kept deliberately minimal:

* **Matek BM only.** One scanner and one 250x250 crop geometry, so every
  descriptor sits on one pixel grid and acquisition is constant by design.
  Matek is in DinoBloom's pretraining (protocol.md section 2), so this
  characterizes the representation and supports no generalization claim.
* **Target** is each descriptor as a percentile of its train distribution.
  Robust to glcm_contrast's heavy tail (kurtosis ~42), and the same
  convention as Stage 5.
* **Head** is ridge regression on z-scored embeddings, alpha chosen per target
  by efficient leave-one-out on train (RidgeCV), scored once on val. Test is
  not read. `--no-zscore` fits it on the stored embedding instead.
* **Seeds.** One run per segmenter seed: the embeddings are fixed, the labels
  differ. QC is applied per seed, as in Stage 5, rather than intersected
  across seeds -- the intersection would drop 8.8% of lymphocytes, likely those
  with the thinnest cytoplasmic rim, i.e. a cut on N:C itself.

Reference points, all on the same val cells:

    class_only    predict the train class mean: how much is just cell type?
    best_dim      the single raw coordinate most correlated on train: is the
                  signal in a few coordinates or spread over many?
    ceiling       mean seed-to-seed correlation of the val label: roughly the
                  most variance any probe could explain in one seed's label.
    shuffled      train labels permuted before fitting: must be ~0.

This is the pooled readout. Whether the embedding carries morphology beyond
cell type is raw_classwise.py, on the same cells, inputs and targets
(`prepare` below).

No network access.
"""

from __future__ import annotations

import argparse
import json
import platform
from typing import NamedTuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from src.env import ROOT, go_offline
from src.morphometry import DESCRIPTORS

go_offline()

DATASET = "matek_bm"
EMB = ROOT / "data" / "embeddings"
RESULTS = ROOT / "results"
ALPHAS = np.logspace(-2, 6, 17)
K = len(DESCRIPTORS)


def load_embeddings(encoder: str, dataset: str = DATASET) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_parquet(EMB / encoder / f"{dataset}.parquet")
    X = np.load(EMB / encoder / f"{dataset}.npy")
    if len(meta) != len(X):
        raise RuntimeError(f"{encoder}/{dataset}: {len(meta)} rows vs {len(X)}")
    # The fold must be the committed split, not whatever the embedding run saw.
    folds = pd.read_parquet(ROOT / "data" / "interim" / dataset / "splits.parquet")
    check = meta.merge(folds, on="cell_id", suffixes=("", "_split"))
    if len(check) != len(meta) or (check.fold != check.fold_split).any():
        raise RuntimeError(f"{encoder}/{dataset}: embedding folds disagree with splits.parquet")
    meta = meta.assign(row=np.arange(len(meta)))
    return meta[meta.ok], X


def labels_for(seed: int, meta: pd.DataFrame, dataset: str = DATASET) -> pd.DataFrame:
    """One segmenter seed's descriptors joined to embedding rows, QC as Stage 5."""
    morph = pd.read_parquet(RESULTS / f"morphometry_{dataset}_seed{seed}.parquet")
    morph = morph[morph.qc_pass & morph[DESCRIPTORS].notna().all(axis=1)]
    df = meta.merge(morph[["cell_id", "class_label"] + DESCRIPTORS],
                    on="cell_id", suffixes=("", "_morph"))
    if (df.class_label != df.class_label_morph).any():
        raise RuntimeError(f"seed {seed}: class labels disagree after the cell_id join")
    return df


def percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of `values` within `reference`, in (0, 1)."""
    ref = np.sort(reference)
    lo = np.searchsorted(ref, values, side="left")
    hi = np.searchsorted(ref, values, side="right")
    return (lo + hi) / (2 * len(ref))


class Design(NamedTuple):
    """One segmenter seed's train and val data, shared with raw_classwise.py."""
    tr: pd.DataFrame
    va: pd.DataFrame
    Xtr: np.ndarray    # embedding z-scored with train statistics
    Xva: np.ndarray
    Ytr: np.ndarray    # descriptors as percentiles of their train distribution
    Yva: np.ndarray
    Ctr: np.ndarray    # each cell's train class mean of Y
    Cva: np.ndarray


def prepare(seed: int, meta: pd.DataFrame, X: np.ndarray) -> Design:
    df = labels_for(seed, meta)
    tr, va = df[df.fold == "train"], df[df.fold == "val"]

    mu, sd = X[tr.row].mean(0), X[tr.row].std(0)
    Xtr = ((X[tr.row] - mu) / sd).astype(np.float64)
    Xva = ((X[va.row] - mu) / sd).astype(np.float64)

    raw_tr, raw_va = tr[DESCRIPTORS].to_numpy(), va[DESCRIPTORS].to_numpy()
    Ytr = np.column_stack([percentile(raw_tr[:, j], raw_tr[:, j]) for j in range(K)])
    Yva = np.column_stack([percentile(raw_tr[:, j], raw_va[:, j]) for j in range(K)])

    # Class means from train only; val is compared with them.
    means = pd.DataFrame(Ytr, index=tr.class_label.to_numpy()).groupby(level=0).mean()
    unseen = set(va.class_label) - set(means.index)
    if unseen:
        raise RuntimeError(f"seed {seed}: val classes absent from train: {unseen}")
    return Design(tr, va, Xtr, Xva, Ytr, Yva,
                  means.loc[tr.class_label].to_numpy(), means.loc[va.class_label].to_numpy())


def ridge_inputs(X: np.ndarray, d: Design, zscore: bool) -> tuple[np.ndarray, np.ndarray]:
    """The ridge's train and val inputs: z-scored, or the stored embedding as is.

    RidgeCV centres its input either way, so the two differ only in
    per-coordinate scaling.
    """
    if zscore:
        return d.Xtr, d.Xva
    return X[d.tr.row].astype(np.float64), X[d.va.row].astype(np.float64)


def run_seed(seed: int, meta: pd.DataFrame, X: np.ndarray,
             zscore: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    design = prepare(seed, meta, X)
    tr, va, Xtr, Xva, Ytr, Yva, _, Cva = design
    Ftr, Fva = ridge_inputs(X, design, zscore)

    # The descriptors and the shuffled control share one design matrix, so one
    # fit covers both.
    shuffled = Ytr[np.random.default_rng(1000 + seed).permutation(len(Ytr))]
    model = RidgeCV(alphas=ALPHAS, alpha_per_target=True).fit(Ftr, np.hstack([Ytr, shuffled]))
    P = model.predict(Fva)

    # Best single coordinate, chosen on train. Xtr is z-scored, so Pearson r is
    # a matrix product and the univariate least-squares fit is closed form.
    Yc = Ytr - Ytr.mean(0)
    corr = (Xtr.T @ Yc) / (len(Xtr) * Ytr.std(0))
    best = np.abs(corr).argmax(0)
    r2_best = [r2_score(Yva[:, j], Ytr[:, j].mean() + corr[d, j] * Ytr[:, j].std() * Xva[:, d])
               for j, d in enumerate(best)]

    rows = pd.DataFrame({
        "encoder": None, "seed": seed, "descriptor": DESCRIPTORS,
        "n_train": len(tr), "n_val": len(va),
        "r2_full": r2_score(Yva, P[:, :K], multioutput="raw_values"),
        "r2_class_only": r2_score(Yva, Cva, multioutput="raw_values"),
        "r2_best_dim": r2_best, "best_dim": best,
        "r2_shuffled": r2_score(Yva, P[:, K:], multioutput="raw_values"),
        "alpha_full": model.alpha_[:K],
    })
    return rows, pd.DataFrame(Yva, index=va.cell_id.to_numpy(), columns=DESCRIPTORS)


def ceiling(per_seed: dict[int, pd.DataFrame], cells=None, method: str = "pearson") -> dict[str, float]:
    """Mean pairwise seed-to-seed correlation of a val label, per descriptor.

    Cells failing QC in a seed are NaN there, so each pair is scored on the
    cells both seeds kept. `cells` restricts it to a subset, e.g. one class.
    """
    out = {}
    for d in DESCRIPTORS:
        wide = pd.DataFrame({s: v[d] for s, v in per_seed.items()})
        if cells is not None:
            wide = wide[wide.index.isin(cells)]
        c = wide.corr(method=method).to_numpy()
        out[d] = float(c[np.triu_indices_from(c, k=1)].mean())
    return out


def summarize(rows: pd.DataFrame, ceil_full: dict) -> dict:
    out = {}
    for d, g in rows.groupby("descriptor", sort=False):
        s = {m: {"mean": float(g[m].mean()), "sd": float(g[m].std(ddof=1)) if len(g) > 1 else None}
             for m in ["r2_full", "r2_class_only", "r2_best_dim", "r2_shuffled"]}
        s["ceiling_full"] = ceil_full.get(d)
        dims = g.best_dim.value_counts()
        s["best_dim_modal"] = int(dims.index[0])
        s["best_dim_modal_seeds"] = int(dims.iloc[0])
        s["alpha_range_full"] = [float(g.alpha_full.min()), float(g.alpha_full.max())]
        s["alpha_at_grid_edge"] = bool(np.isin(g.alpha_full.to_numpy(), [ALPHAS[0], ALPHAS[-1]]).any())
        out[d] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(25)))
    ap.add_argument("--no-zscore", action="store_true",
                    help="fit the ridge on the stored embedding rather than z-scored coordinates")
    args = ap.parse_args()

    meta, X = load_embeddings(args.encoder)
    frames, val_full = [], {}
    for seed in args.seeds:
        rows, yva = run_seed(seed, meta, X, zscore=not args.no_zscore)
        rows["encoder"] = args.encoder
        frames.append(rows)
        val_full[seed] = yva
        r = rows.set_index("descriptor")
        print(f"[raw_probe] seed {seed:2d}: n_train {r.n_train.iloc[0]:,} n_val {r.n_val.iloc[0]:,}  "
              f"median R2 {r.r2_full.median():.3f}  max |shuffled| {r.r2_shuffled.abs().max():.4f}",
              flush=True)

    rows = pd.concat(frames, ignore_index=True)
    ceil_full = ceiling(val_full) if len(args.seeds) > 1 else {}
    summary = {
        "analysis": "ridge readout of morphometric descriptors from the raw embedding, all cells pooled",
        "encoder": args.encoder, "dataset": DATASET,
        "fit_on": "train", "scored_on": "val", "test_read": False,
        "target": "descriptor percentile within the train distribution",
        "features": "embedding as stored, not z-scored" if args.no_zscore else "embedding z-scored on train",
        "alpha_selection": "RidgeCV efficient leave-one-out on train, per target",
        "alphas": ALPHAS.tolist(),
        "qc": "qc_pass and all descriptors finite, per segmenter seed",
        "n_seeds": len(args.seeds), "seeds": args.seeds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "host": platform.node(),
        "descriptors": summarize(rows, ceil_full),
    }
    stem = f"raw_probe_{args.encoder}_{DATASET}" + ("_noz" if args.no_zscore else "")
    rows.to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))

    print(f"\nRidge on raw {args.encoder} ({X.shape[1]}-d), Matek BM val, "
          f"{len(args.seeds)} segmenter seed(s), R2 mean:")
    print(f"  {'descriptor':28s} {'full':>6s} {'class':>6s} {'1 dim':>6s} {'ceiling':>7s}")
    for d, s in summary["descriptors"].items():
        c = f"{s['ceiling_full']:.3f}" if s["ceiling_full"] is not None else "-"
        print(f"  {d:28s} {s['r2_full']['mean']:6.3f} {s['r2_class_only']['mean']:6.3f} "
              f"{s['r2_best_dim']['mean']:6.3f} {c:>7s}")
    print(f"[raw_probe] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
