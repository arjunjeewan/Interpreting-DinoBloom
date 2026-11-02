"""Three-way train/val/test assignment, produced once and committed.

Guardrail 1: selection happens on validation, test is read once at the end.
Splits are therefore fixed here at a single seed and never regenerated per run
-- the 3 seeds in protocol.md vary model initialization, not the partition.

Two kinds of split are emitted:

`data/interim/<dataset>/splits.parquet` -- cell_id -> fold, image-level,
    stratified by class label. No dataset in this project has patient
    identifiers (Matek BM filenames encode class and index only, HiCervix CSVs
    carry no WSI key, Raabin's film_id is a blood-film id), so patient-disjoint
    splitting is impossible and this is a declared limitation, not an oversight.

    Raabin is special: its published Train / TestA / TestB partition IS the
    Stage 7 device axis, so only the Train split is divided three ways. TestA
    and TestB keep their published identity as held-out device sets.

`data/interim/raabin/gt_splits.parquet` -- the 927 segmentation ground-truth
    cells, split **film-disjoint** by film_id.

    Note this deliberately does NOT follow the authors' own train/test division
    of the ground-truth set: 15 of the 54 films have cells on both sides of it,
    so that division leaks blood films and a Dice reported on it would be
    optimistic. Film-disjoint is strictly stronger. Films are very uneven
    (2 to 139 cells), so the realized fractions are approximate; the achieved
    per-fold class balance is reported in the summary rather than assumed.

No network access.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.env import ROOT, go_offline

go_offline()

SEED = 0
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
FOLDS = list(FRACTIONS)


def stratified_three_way(df: pd.DataFrame, seed: int = SEED,
                         stratify_by: str = "class_label") -> pd.Series:
    """Image-level stratified assignment. Returns a fold Series aligned to df."""
    rng = np.random.default_rng(seed)
    fold = pd.Series(index=df.index, dtype=object)
    for _, idx in df.groupby(stratify_by).groups.items():
        idx = rng.permutation(np.asarray(idx))
        n_train = int(round(len(idx) * FRACTIONS["train"]))
        n_val = int(round(len(idx) * FRACTIONS["val"]))
        fold.loc[idx[:n_train]] = "train"
        fold.loc[idx[n_train:n_train + n_val]] = "val"
        fold.loc[idx[n_train + n_val:]] = "test"
    return fold


def grouped_three_way(df: pd.DataFrame, group_col: str, seed: int = SEED,
                      stratify_by: str = "class_label") -> pd.Series:
    """Group-disjoint assignment that also tries to balance class proportions.

    Greedy: place the largest group first, into whichever fold is furthest below
    its target -- measured over the per-class counts, not just the total, so a
    film that is nearly all neutrophils does not skew one fold's class mix.
    """
    rng = np.random.default_rng(seed)
    classes = sorted(df[stratify_by].unique())
    totals = df[stratify_by].value_counts().reindex(classes).fillna(0).to_numpy()
    target = {f: FRACTIONS[f] * totals for f in FOLDS}

    counts = (df.groupby([group_col, stratify_by]).size().unstack(fill_value=0)
              .reindex(columns=classes, fill_value=0))
    sizes = counts.sum(axis=1)

    def greedy(order) -> tuple[dict[str, str], float]:
        current = {f: np.zeros(len(classes)) for f in FOLDS}
        assignment: dict[str, str] = {}
        for group in order:
            vec = counts.loc[group].to_numpy(dtype=float)
            # Score by how full each fold would become *relative to its own
            # target*, summed over classes. An absolute deficit would be
            # minimized by always topping up the smallest fold, starving train.
            cost = {f: float(((current[f] + vec) / np.maximum(target[f], 1e-9)).sum())
                    for f in FOLDS}
            best = min(cost, key=cost.get)
            assignment[group] = best
            current[best] = current[best] + vec
        # Quality = worst per-class, per-fold relative deviation from target.
        worst = max(float(np.abs(current[f] - target[f]).max() / max(totals.max(), 1))
                    for f in FOLDS)
        return assignment, worst

    # 54 films is too few and too uneven for one greedy pass to balance the rare
    # classes; restarts cost milliseconds and materially improve the class mix.
    best_assignment, best_score = None, np.inf
    for _ in range(500):
        jitter = rng.uniform(0.85, 1.15, size=len(sizes))
        order = list(sizes.index[np.argsort(-(sizes.to_numpy() * jitter))])
        assignment, score = greedy(order)
        if score < best_score:
            best_assignment, best_score = assignment, score

    return df[group_col].map(best_assignment)


def _summary(df: pd.DataFrame, fold: pd.Series, group_col: str | None) -> dict:
    out = {
        "n": len(df),
        "fold_counts": fold.value_counts().to_dict(),
        "fold_fractions": (fold.value_counts(normalize=True).round(4)).to_dict(),
        "class_by_fold": pd.crosstab(fold, df["class_label"]).to_dict(),
    }
    if group_col:
        spanning = df.assign(_f=fold).groupby(group_col)._f.nunique()
        out["n_groups"] = int(df[group_col].nunique())
        out["groups_spanning_folds"] = int((spanning > 1).sum())
    return out


def main() -> None:
    interim = ROOT / "data" / "interim"
    summaries = {}

    for name in ["matek_bm", "pbc", "hicervix"]:
        df = pd.read_parquet(interim / name / "index.parquet")
        fold = stratified_three_way(df)
        pd.DataFrame({"cell_id": df.cell_id, "fold": fold.to_numpy()}).to_parquet(
            interim / name / "splits.parquet", index=False)
        summaries[name] = _summary(df, fold, None)

    # Raabin: only the published Train split is divided; TestA/TestB are the
    # Stage 7 device axis and keep their published identity.
    df = pd.read_parquet(interim / "raabin" / "index.parquet")
    fold = df["split"].astype(object).copy()
    is_train = df["split"] == "train"
    fold.loc[is_train] = stratified_three_way(df[is_train]).to_numpy()
    pd.DataFrame({"cell_id": df.cell_id, "fold": fold.to_numpy()}).to_parquet(
        interim / "raabin" / "splits.parquet", index=False)
    summaries["raabin"] = _summary(df, fold, None)

    # Segmentation ground truth: film-disjoint, 927 nucleus-GT cells only.
    gt = pd.read_parquet(interim / "raabin" / "gt_masks.parquet")
    gt = gt[gt.has_nucleus_gt].reset_index(drop=True)
    gt_fold = grouped_three_way(gt, group_col="film_id")
    pd.DataFrame({"cell_id": gt.cell_id, "stem": gt.stem,
                  "fold": gt_fold.to_numpy()}).to_parquet(
        interim / "raabin" / "gt_splits.parquet", index=False)
    summaries["raabin_gt_segmentation"] = _summary(gt, gt_fold, "film_id")
    summaries["raabin_gt_segmentation"]["authors_division_leaks_films"] = int(
        (gt.groupby("film_id").gt_split.nunique() > 1).sum())

    out = ROOT / "results" / "splits_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2, default=str))

    for name, s in summaries.items():
        print(f"[splits] {name:24s} n={s['n']:>7,}  {s['fold_counts']}")
    seg = summaries["raabin_gt_segmentation"]
    print(f"[splits] segmentation GT: {seg['n_groups']} films, "
          f"{seg['groups_spanning_folds']} spanning folds (must be 0)")
    print(f"[splits] wrote {out}")


if __name__ == "__main__":
    main()
