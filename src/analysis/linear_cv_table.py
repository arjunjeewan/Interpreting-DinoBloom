"""linear_cv's regression errors as a CSV: MAE, MSE and RMSE per descriptor.

Reads the summary json that linear_cv.py wrote and recomputes nothing. Two rows
per descriptor: `linear` (OLS on the raw embedding) and `baseline` (predict the
training mean). Each metric is the mean and sd (ddof=1) over the 5 fold models,
scored on each model's own validation fold (val_*) and on the 20% hold-out
(test_*); val_n is the median fold size. rmse_mean is the mean of the per-fold
RMSEs, so it sits at or slightly below sqrt(mse_mean). Errors are in each
descriptor's own units (src/morphometry.py): read `linear` against `baseline`,
not across descriptors.

Also Table 1 of the paper, as tables/linear_cv_table1_<encoder>.csv and a
booktabs tabular (.tex; the caption belongs in the paper). One row per metric,
in the family order of the figures, in three column groups:

    Matek BM      pooled and within-type R^2 on the validation folds
                  (linear_cv_within.py; mean over the 5 fold models)
    PBC           pooled and within-type R^2 on the test fold (linear_unseen.py,
                  protocol.md §16; mean over the 5 readouts)
    Raabin testB  Spearman rho of readout and label, and R^2 for the
                  scale-invariant descriptors only (testB is imaged at another
                  scale)

then each dataset's cell-type readout, accuracy and balanced accuracy, in the
first column of its group. The tex shows means; the CSV also carries the sds
(the caption states the largest), Matek's hold-out R^2 and the cell-type-only
R^2, for the text. --val previews the layout on PBC's and Raabin's validation
folds, before the test folds are read. Nothing is recomputed.

No network access.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.analysis.linear_unseen import SCALE_INVARIANT, TEST_FOLD
from src.analysis.raw_probe import DATASET, RESULTS
from src.env import ROOT, go_offline
from src.figures import DESCRIPTOR_LABEL, HEATMAP_ROWS
from src.morphometry import DESCRIPTORS

go_offline()

METRICS = ["mae", "mse", "rmse"]
TABLES = ROOT / "tables"
ATOL_R2 = 1e-12      # validation R^2 from linear_cv's MSEs vs linear_cv_within.py's
GROUPS = ["matek", "pbc", "raabin"]


def error_table(results: dict) -> pd.DataFrame:
    """One row per descriptor x model; val and test columns side by side."""
    rows = []
    for d in DESCRIPTORS:
        for model in ["linear", "baseline"]:
            row = {"descriptor": d, "model": model}
            for part in ["val", "test"]:
                s = results[d][model][part]
                row[f"{part}_n"] = s["n_median"]
                for m in METRICS:
                    row[f"{part}_{m}_mean"], row[f"{part}_{m}_sd"] = s[m]["mean"], s[m]["sd"]
            rows.append(row)
    return pd.DataFrame(rows)


def table1(encoder: str, unseen: dict[str, str] = TEST_FOLD) -> pd.DataFrame:
    """Table 1 as numbers: one row per metric in figure family order, then the cell-type readouts.

    Columns <group>_1 and <group>_2 are the two numbers each column group shows, with
    their sds alongside.
    """
    within = json.loads((RESULTS / f"linear_cv_within_{encoder}_{DATASET}.json").read_text())["descriptors"]
    runs = pd.read_parquet(RESULTS / f"linear_cv_{encoder}_{DATASET}.parquet")
    mse = runs.pivot_table(index=["split", "target", "fold"], columns="model", values="mse")
    r2 = (1 - mse["linear"] / mse["baseline"]).groupby(["split", "target"]).agg(["mean", "std"])
    worst = max(abs(r2.loc[("val", d), "mean"] - within[d]["r2_pooled"]["mean"]) for d in DESCRIPTORS)
    if worst > ATOL_R2:
        raise RuntimeError(f"validation R^2 differs from linear_cv_within.py's by {worst:.1e}")
    bench = {ds: json.loads((RESULTS / f"linear_unseen_{encoder}_{ds}.json").read_text()) for ds in unseen}
    pbc, rab = (bench[ds]["descriptors"][unseen[ds]] for ds in ("pbc", "raabin"))

    rows = []
    for f, family in enumerate(HEATMAP_ROWS):
        for d in family:
            w, p, r = within[d], pbc[d], rab[d]
            keep = d in SCALE_INVARIANT                  # Raabin testB: R^2 kept for these only
            rows.append({"row": DESCRIPTOR_LABEL[d], "family": f, "statistic": "r2",
                         "matek_1": w["r2_pooled"]["mean"], "matek_1_sd": w["r2_pooled"]["sd"],
                         "matek_2": w["r2_within"]["mean"], "matek_2_sd": w["r2_within"]["sd"],
                         "pbc_1": p["r2_pooled"]["mean"], "pbc_1_sd": p["r2_pooled"]["sd"],
                         "pbc_2": p["r2_within"]["mean"], "pbc_2_sd": p["r2_within"]["sd"],
                         "raabin_1": r["rho_pooled"]["mean"], "raabin_1_sd": r["rho_pooled"]["sd"],
                         "raabin_2": r["r2_pooled"]["mean"] if keep else None,
                         "raabin_2_sd": r["r2_pooled"]["sd"] if keep else None,
                         "matek_holdout": r2.loc[("test", d), "mean"],
                         "matek_type_only": w["r2_type_only"]["mean"], "pbc_type_only": p["r2_type_only"]["mean"]})
    logistic = {"matek": json.loads((RESULTS / f"linear_cv_{encoder}_{DATASET}.json").read_text())
                ["results"]["class_label"]["logistic"]["val"],
                **{ds: bench[ds]["class_label"][unseen[ds]]["logistic"] for ds in unseen}}
    for stat in ["accuracy", "balanced_accuracy"]:
        rows.append({"row": f"cell type, {stat.replace('_', ' ')}", "family": len(HEATMAP_ROWS), "statistic": stat,
                     **{f"{g}_1": logistic[g][stat]["mean"] for g in GROUPS},
                     **{f"{g}_1_sd": logistic[g][stat]["sd"] for g in GROUPS}})
    return pd.DataFrame(rows)


def table1_tex(t: pd.DataFrame, unseen: dict[str, str] = TEST_FOLD) -> str:
    """Booktabs tabular for Table 1: three column groups of two, a little space between metric
    families, a rule before the cell-type rows; '--' where a group shows no number."""
    shown = {"testB": "set B"}  # the paper's name for Raabin's test set B
    names = {"matek": "Matek BM (val)", "pbc": f"PBC ({unseen['pbc']})",
             "raabin": f"Raabin ({shown.get(unseen['raabin'], unseen['raabin'])})"}
    heads = {"matek": ["$R^2$", "within"], "pbc": ["$R^2$", "within"], "raabin": [r"$\rho$", "$R^2$"]}
    lines = [r"\begin{tabular}{l" + "cc" * len(GROUPS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{names[g]}}}" for g in GROUPS) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(GROUPS))),
             "Linear regression & " + " & ".join(h for g in GROUPS for h in heads[g]) + r" \\",
             r"\midrule"]
    for f, group in t.groupby("family", sort=True):
        if f == len(HEATMAP_ROWS):
            lines.append(r"\midrule")
        elif f > 0:
            lines.append(r"\addlinespace")
        for r in group.itertuples():
            cells = [getattr(r, f"{g}_{i}") for g in GROUPS for i in (1, 2)]
            lines.append(" & ".join([r.row, *("--" if pd.isna(v) else f"{v:.3f}".replace("-", "$-$")  # minus, not hyphen
                                              for v in cells)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    ap.add_argument("--val", action="store_true",
                    help="preview the layout on PBC's and Raabin's validation folds")
    args = ap.parse_args()
    unseen = {ds: "val" for ds in TEST_FOLD} if args.val else TEST_FOLD

    src = RESULTS / f"linear_cv_{args.encoder}_{DATASET}.json"
    out = RESULTS / f"linear_cv_errors_{args.encoder}_{DATASET}.csv"
    error_table(json.loads(src.read_text())["results"]).to_csv(out, index=False, float_format="%.8g")
    print(f"[linear_cv_table] wrote {out}")

    TABLES.mkdir(exist_ok=True)
    t = table1(args.encoder, unseen)
    stem = TABLES / f"linear_cv_table1_{args.encoder}"
    t.to_csv(stem.with_suffix(".csv"), index=False, float_format="%.8g")
    stem.with_suffix(".tex").write_text(table1_tex(t, unseen))
    sds = {g: float(t[[f"{g}_1_sd", f"{g}_2_sd"]].max().max()) for g in GROUPS}
    print(f"[linear_cv_table] largest sd per group, for the caption: {json.dumps(sds)}")
    print(f"[linear_cv_table] wrote {stem}.csv / .tex")


if __name__ == "__main__":
    main()
