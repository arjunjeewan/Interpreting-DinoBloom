"""Cytomorphometric descriptors from a nucleus/cytoplasm label map.

The descriptor list is fixed. Eleven descriptors per cell:

    nuclear_area, cytoplasmic_area, nc_ratio, nuclear_solidity, form_factor,
    eccentricity, curvature_variance, glcm_contrast, glcm_homogeneity,
    glcm_entropy, integrated_optical_density

Definitional choices that the brief left open, pinned here and mirrored into
protocol.md:

* **nc_ratio = nuclear area / whole-cell area**, not nucleus/cytoplasm. This is
  the convention behind clinical statements like "blasts have an N:C ratio of
  0.8"; the literal nucleus:cytoplasm quotient is unbounded and blows up when
  the cytoplasmic rim is thin, which is exactly the blast case Gate A tests.
* Granulocyte nuclei are multi-lobed and the mask is deliberately disconnected.
  Every shape descriptor is computed over the union as a single region, so
  form_factor sums lobe perimeters -- lobation lowers form_factor, which is the
  intended behaviour.
* GLCM is restricted to pixel pairs with **both** members inside the nucleus, by
  quantizing to levels 1..NGL inside and 0 outside, then dropping row/column 0
  before normalizing. Masking to a bounding box instead would put a hard
  nucleus/background edge into the chromatin texture.
* Optical density uses a per-image white point I0 taken as the 95th percentile
  of the background (outside the whole cell), i.e. the smear itself, rather than
  a fixed 255. Raabin spans two cameras with different exposure and a fixed
  white point would leak illumination into a "chromatin" descriptor.

Scale: no per-camera micron calibration is published for Raabin, so all areas
are in pixels and `scale_units` records that.

No network access.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import find_contours, regionprops

from src.env import ROOT, go_offline

go_offline()

BG, NUC, CYT = 0, 1, 2

GLCM_LEVELS = 32
GLCM_DISTANCE = 1
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

# Contour resampling for curvature. Fixed point count makes the variance
# comparable across cells of different size; the Gaussian removes pixel
# staircasing, which otherwise dominates the curvature entirely.
CONTOUR_POINTS = 128
CONTOUR_SIGMA = 3.0
MIN_CONTOUR_PX = 20

# Curvature has units of 1/length, so on a pixel grid it cannot meaningfully
# exceed the curvature of a circle a few pixels across. Without this bound a
# single near-degenerate contour point (where the arc-length parameterization
# briefly stalls and the denominator approaches zero) produces astronomical
# curvature, and squaring it for the variance lets that one point dominate the
# whole cell: the unclipped descriptor reached 1.4e6 against a median of 0.017
# and was uncorrelated across segmenter seeds. Clipping to a 3-pixel radius
# bounds the per-point contribution at (1/3)^2 without touching real lobation,
# whose radii of curvature are tens of pixels. (FINDINGS §27 defect 2;
# user-approved 2026-07-31.)
KAPPA_MAX = 1.0 / 3.0

# Minimum nucleus size for a cell to yield valid morphometry. The segmenter
# occasionally emits an ~11-pixel "nucleus" against a Matek median of 13,479;
# these degenerate detections are the source of form_factor values above 1
# (mathematically impossible) and glcm_contrast above 700. Cells below this are
# marked qc_pass=False rather than deleted, so the exclusion stays auditable.
# (FINDINGS §27 defect 1; user-approved 2026-07-31.)
MIN_NUCLEAR_AREA = 200

DESCRIPTORS = [
    "nuclear_area", "cytoplasmic_area", "nc_ratio", "nuclear_solidity",
    "form_factor", "eccentricity", "curvature_variance", "glcm_contrast",
    "glcm_homogeneity", "glcm_entropy", "integrated_optical_density",
]


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def _shape(nucleus: np.ndarray) -> dict:
    """Solidity, form factor and eccentricity over the nucleus as one region."""
    props = regionprops(nucleus.astype(np.uint8))
    if not props:
        return dict(nuclear_solidity=np.nan, form_factor=np.nan, eccentricity=np.nan)
    p = props[0]
    perim = p.perimeter
    return dict(
        nuclear_solidity=float(p.solidity),
        form_factor=float(4 * np.pi * p.area / perim ** 2) if perim > 0 else np.nan,
        eccentricity=float(p.eccentricity),
    )


def _curvature_variance(nucleus: np.ndarray) -> float:
    """Variance of signed curvature along the nucleus boundary.

    Each lobe contributes its own closed contour; contours are resampled to a
    common length and length-weighted so a large lobe counts more than a small
    one.
    """
    kappas, weights = [], []
    for contour in find_contours(nucleus.astype(float), 0.5):
        if len(contour) < MIN_CONTOUR_PX:
            continue
        # Resample to a fixed number of points by arc length, then smooth
        # periodically so the ends of the closed curve join cleanly.
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))]
        if d[-1] <= 0:
            continue
        t = np.linspace(0, d[-1], CONTOUR_POINTS, endpoint=False)
        xy = np.column_stack([np.interp(t, d, contour[:, i]) for i in (0, 1)])
        xy = gaussian_filter1d(xy, CONTOUR_SIGMA, axis=0, mode="wrap")

        dx, dy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        denom = (dx ** 2 + dy ** 2) ** 1.5
        # Guard relative to the actual point spacing rather than at an absolute
        # 1e-12: after resampling, spacing is d[-1]/CONTOUR_POINTS, so a
        # legitimate denominator is on the order of spacing**3 and anything far
        # below that is a stalled parameterization, not a sharp corner.
        floor = (1e-3 * d[-1] / CONTOUR_POINTS) ** 3
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(denom > floor, (dx * ddy - dy * ddx) / denom, np.nan)
        k = k[np.isfinite(k)]
        k = np.clip(k, -KAPPA_MAX, KAPPA_MAX)
        if k.size:
            kappas.append(k)
            weights.append(d[-1])

    if not kappas:
        return np.nan
    weights = np.asarray(weights, dtype=float)
    variances = np.asarray([k.var() for k in kappas])
    return float((variances * weights).sum() / weights.sum())


def _glcm(gray: np.ndarray, nucleus: np.ndarray) -> dict:
    """GLCM texture over nucleus-internal pixel pairs only."""
    nan = dict(glcm_contrast=np.nan, glcm_homogeneity=np.nan, glcm_entropy=np.nan)
    if nucleus.sum() < 2:
        return nan

    rows = np.flatnonzero(nucleus.any(axis=1))
    cols = np.flatnonzero(nucleus.any(axis=0))
    sub_gray = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    sub_nuc = nucleus[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Contrast-stretch within the nucleus so texture is measured independently
    # of the cell's overall staining depth, then reserve level 0 for outside.
    vals = sub_gray[sub_nuc]
    lo, hi = vals.min(), vals.max()
    if hi <= lo:
        return nan
    q = np.zeros(sub_gray.shape, dtype=np.uint8)
    scaled = (sub_gray - lo) / (hi - lo)
    q[sub_nuc] = 1 + np.clip((scaled[sub_nuc] * (GLCM_LEVELS - 1)), 0, GLCM_LEVELS - 1).astype(np.uint8)

    glcm = graycomatrix(q, distances=[GLCM_DISTANCE], angles=list(GLCM_ANGLES),
                        levels=GLCM_LEVELS + 1, symmetric=True, normed=False)
    glcm = glcm[1:, 1:]  # drop pairs touching a non-nucleus pixel
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    if not np.all(totals > 0):
        return nan
    glcm = glcm / totals

    p = glcm[..., 0, :]
    entropy = -(p * np.log2(p, out=np.zeros_like(p), where=p > 0)).sum(axis=(0, 1))
    return dict(
        glcm_contrast=float(graycoprops(glcm, "contrast").mean()),
        glcm_homogeneity=float(graycoprops(glcm, "homogeneity").mean()),
        glcm_entropy=float(entropy.mean()),
    )


def _iod(gray: np.ndarray, nucleus: np.ndarray, wholecell: np.ndarray) -> float:
    """Integrated optical density over the nucleus, per-image white point."""
    background = gray[~wholecell]
    i0 = np.percentile(background, 95) if background.size else 255.0
    if i0 <= 0:
        return np.nan
    ratio = np.clip(gray[nucleus] / i0, 1e-3, 1.0)
    return float(-np.log10(ratio).sum())


def descriptors(rgb: np.ndarray, label: np.ndarray) -> dict:
    """All eleven descriptors for one cell. `label`: 0 bg, 1 nucleus, 2 cytoplasm."""
    nucleus = label == NUC
    cytoplasm = label == CYT
    wholecell = nucleus | cytoplasm
    gray = _luminance(rgb.astype(np.float64))

    n_area, c_area = float(nucleus.sum()), float(cytoplasm.sum())
    out = {
        "nuclear_area": n_area,
        "cytoplasmic_area": c_area,
        "nc_ratio": n_area / (n_area + c_area) if (n_area + c_area) > 0 else np.nan,
    }
    out["qc_pass"] = bool(n_area >= MIN_NUCLEAR_AREA and c_area > 0)
    out.update(_shape(nucleus))
    out["curvature_variance"] = _curvature_variance(nucleus)
    out.update(_glcm(gray, nucleus))
    out["integrated_optical_density"] = _iod(gray, nucleus, wholecell)
    return out


def from_paths(image_path, mask_path) -> dict:
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    label = np.asarray(Image.open(mask_path))
    return descriptors(rgb, label)


KEYS = ["cell_id", "stem", "class_label", "device", "gt_split", "film_id"]


def main() -> None:
    gt = pd.read_parquet(ROOT / "data" / "interim" / "raabin" / "gt_masks.parquet")
    raabin = ROOT / "data" / "interim" / "raabin"

    # The 218 second-microscope cells carry whole-cell annotation only (label 3,
    # see src/masks.py), so no nuclear descriptor is defined for them.
    gt = gt[gt.has_nucleus_gt].reset_index(drop=True)

    rows = []
    for i, rec in enumerate(gt.itertuples(), 1):
        d = from_paths(raabin / rec.image_path, ROOT / rec.mask_path)
        d.update({k: getattr(rec, k) for k in KEYS})
        rows.append(d)
        if i % 250 == 0:
            print(f"[morphometry] {i}/{len(gt)}")

    df = pd.DataFrame(rows)[KEYS + DESCRIPTORS + ["qc_pass"]]
    df["scale_units"] = "pixels"
    out = ROOT / "results" / "morphometry_raabin_gt.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    summary = {
        "n_cells": len(df),
        "scale_units": "pixels",
        "nan_counts": {c: int(df[c].isna().sum()) for c in DESCRIPTORS},
        "describe": json.loads(df[DESCRIPTORS].describe().to_json()),
    }
    (ROOT / "results" / "morphometry_raabin_gt_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(df[DESCRIPTORS].describe().T.to_string())
    print(f"[morphometry] wrote {out} : {len(df):,} cells")


if __name__ == "__main__":
    main()
