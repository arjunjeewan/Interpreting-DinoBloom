"""Bootstrap over cells of the measured-region lead of each linear model (post hoc).

A linear model is led by its measured region when that region's share of the
prediction's variance, Cov(T_g, y) / Var(y) with the patch writes split by pixel
fraction, is the largest of nucleus, cytoplasm and rest; N:C, measured in both,
counts only if nucleus and cytoplasm both beat the rest (as in
linear_unseen_attnpath.leads). Matek: the 2,000 cross-validation cells of
linear_cv_attnpath.py, out-of-fold predictions. PBC: the 2,211 test cells of
linear_unseen_attnpath.py; shares per fit, averaged over the five fits.

Cells are resampled with replacement (B resamples, seed SEED) and the shares
recomputed on each resample. Per model: the fraction of resamples in which the
measured region leads, and the 2.5 and 97.5 percentiles of the lead margin
(measured share minus the largest other share; N:C: the smaller of nucleus and
cytoplasm minus rest). Cells are treated as independent, which they are not
(several cells per patient and film), so the spread is a lower bound.

This analysis was added after the test split was read; it reuses the stored
per-cell terms and fits nothing. No network access.

Output: results/linear_region_lead_<encoder>.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.analysis.linear_cv_attnpath import MEASURED_IN
from src.env import ROOT, go_offline
from src.morphometry import DESCRIPTORS

go_offline()

RESULTS = ROOT / "results"
B, SEED = 10_000, 0
REGIONS = ["nucleus", "cytoplasm", "rest"]
GROUPS = ["const", "cls", "nucleus", "cytoplasm", "rest", "mlp"]   # these sum to the prediction
BLOCKS = range(1, 13)


def margin(s: np.ndarray, measured: str) -> np.ndarray:
    """s (..., 3) region shares in REGIONS order -> lead margin; > 0 iff the measured region leads."""
    if measured == "both":
        return np.minimum(s[..., 0], s[..., 1]) - s[..., 2]
    i = REGIONS.index(measured)
    return s[..., i] - np.max(np.delete(s, i, axis=-1), axis=-1)


def shares(W: np.ndarray, T: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cov(T_g, y) / Var(y) under bootstrap count weights W (B, n); T (n, g), y (n) -> (B, g)."""
    n = W.shape[1]
    Ey, Eyy = W @ y / n, W @ (y * y) / n
    ET, ETy = W @ T / n, W @ (T * y[:, None]) / n
    return (ETy - ET * Ey[:, None]) / (Eyy - Ey ** 2)[:, None]


def weights(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=B).astype(np.float64)


def check_sum(terms: np.ndarray, y: np.ndarray, what) -> None:
    if not np.allclose(terms.sum(1), y, rtol=1e-4, atol=1e-6 * np.abs(y).max()):
        raise RuntimeError(f"terms do not sum to the prediction: {what}")


def matek(encoder: str) -> dict:
    d = pd.read_parquet(RESULTS / f"linear_cv_attnpath_{encoder}_matek_bm.parquet")
    cells = d.cell_id.unique()
    W, one = weights(len(cells)), np.ones((1, len(cells)))
    out = {}
    for desc in DESCRIPTORS:
        x = d[d.descriptor == desc].set_index("cell_id").loc[cells]
        summed = {g: x[[f"{g}_{b}" for b in BLOCKS]].to_numpy().sum(1) for g in GROUPS[1:]}
        y = x["y"].to_numpy()
        check_sum(np.stack([x["const"].to_numpy()] + [summed[g] for g in GROUPS[1:]], 1), y, desc)
        T = np.stack([summed[r] for r in REGIONS], 1)
        out[desc] = (shares(one, T, y)[0], shares(W, T, y))
    return out


def pbc(encoder: str) -> dict:
    d = pd.read_parquet(RESULTS / f"linear_unseen_attnpath_{encoder}_pbc.parquet")
    d = d[d.split == "test"]
    cells = np.sort(d.cell_id.unique())
    W, one = weights(len(cells)), np.ones((1, len(cells)))
    out = {}
    for desc in DESCRIPTORS:
        point, boot = [], []
        for fit, x in d[d.descriptor == desc].groupby("readout"):
            x = x.set_index("cell_id").loc[cells]
            y = x["y"].to_numpy()
            check_sum(x[GROUPS].to_numpy(), y, (desc, fit))
            T = x[REGIONS].to_numpy()
            point.append(shares(one, T, y)[0])
            boot.append(shares(W, T, y))
        out[desc] = (np.mean(point, 0), np.mean(boot, 0))   # mean over the five fits
    return out


def summarize(per: dict) -> dict:
    rows = {}
    for desc, (point, boot) in per.items():
        m_point, m_boot = margin(point, MEASURED_IN[desc]), margin(boot, MEASURED_IN[desc])
        rows[desc] = {"measured_in": MEASURED_IN[desc], "shares": dict(zip(REGIONS, point.tolist())),
                      "leads": bool(m_point > 0), "margin": float(m_point),
                      "margin_ci95": np.percentile(m_boot, [2.5, 97.5]).tolist(),
                      "fraction_of_resamples_leading": float((m_boot > 0).mean())}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", default="dinobloom")
    args = ap.parse_args()
    result = {"analysis": "measured-region lead under a bootstrap over cells (post hoc)",
              "resamples": B, "seed": SEED, "regions": REGIONS, "measured_in": MEASURED_IN,
              "matek_cv": summarize(matek(args.encoder)), "pbc_test": summarize(pbc(args.encoder))}
    for part in ("matek_cv", "pbc_test"):
        rows = result[part].values()
        lead = [r["fraction_of_resamples_leading"] for r in rows if r["leads"]]
        rest = [r["fraction_of_resamples_leading"] for r in rows if not r["leads"]]
        print(f"[linear_region_lead] {part}: measured region leads for {len(lead)}/{len(lead) + len(rest)}; "
              f"lowest fraction of resamples among those: {min(lead):.4f}; "
              f"highest among the rest: {max(rest) if rest else float('nan'):.4f}")
    out = RESULTS / f"linear_region_lead_{args.encoder}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"[linear_region_lead] wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
