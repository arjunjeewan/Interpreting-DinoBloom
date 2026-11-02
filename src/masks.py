"""Build nucleus/cytoplasm label maps for the 1,145 Raabin ground-truth cells.

The BMP masks use an INVERTED palette. From the authors' own NucCytSplitter.py
(data/interim/raabin/_gt/splitter/), **100 = nucleus and 255 = cytoplasm**.
Reading them the intuitive way silently inverts every N:C ratio while still
producing plausible-looking numbers. The rasterized maskRCNN.rar polygons agree
with the 100-region at median IoU 0.89, which is the independent confirmation.

**Only 927 of the 1,145 cells have nucleus ground truth.** The 218 cells from
the second microscope (zeiss_lg_g3) are whole-cell only, and no nucleus
annotation for them exists anywhere in the Raabin distribution -- all three
shipped sources agree:

    GroundTruths_bmpformat.rar   values {0, 255}, no 100-region
    GrTh.rar                     same masks as JPEG, same two levels
    maskRCNN.rar                 one polygon per cell, IoU 0.96 with the
                                 whole-cell mask, i.e. it outlines the cell,
                                 not the nucleus (and never lobed, whereas
                                 first-microscope polygons are)

Those 218 cells are also, without exception, **basophils** -- the rare class was
collected on the second microscope -- so in this ground-truth subset device is
confounded with class and neither may be used to explain the other here. (The
Stage 7 device axis is Train/TestA vs Test-B, not this subset, so it is
unaffected.)

Construction:

    nucleus   = BMP == 100                 first microscope only
    cytoplasm = BMP == 255                 first microscope only
    cell      = BMP > 0                    second microscope, unresolved interior

Output is one uint8 PNG per cell in data/masks/raabin/, encoded 0 = background,
1 = nucleus, 2 = cytoplasm, 3 = cell interior with no nucleus/cytoplasm
annotation. Label 3 carries foreground supervision for the segmenter but must be
excluded from any nucleus/cytoplasm loss and from all morphometry.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.draw import polygon as draw_polygon

from src.env import ROOT, go_offline

go_offline()

RAABIN = ROOT / "data" / "interim" / "raabin"
VIA_JSON = {"train": RAABIN / "_gt/mrcnn/maskRCNN/Train_GT.json",
            "test": RAABIN / "_gt/mrcnn/maskRCNN/Test_GT.json"}
GRTH = RAABIN / "_gt/grth/GrTh/Ground Truth"
OUT = ROOT / "data" / "masks" / "raabin"

BG, NUC, CYT, CELL = 0, 1, 2, 3

# Palette of the shipped BMP masks. Inverted relative to intuition -- see above.
BMP_NUCLEUS, BMP_CYTOPLASM = 100, 255


def load_polygons() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Map filename stem -> list of (rows, cols) polygon vertex arrays."""
    polys: dict[str, list] = {}
    for path in VIA_JSON.values():
        for entry in json.loads(path.read_text()).values():
            stem = Path(entry["filename"]).stem
            regions = [
                (np.asarray(r["shape_attributes"]["all_points_y"]),
                 np.asarray(r["shape_attributes"]["all_points_x"]))
                for r in entry["regions"]
            ]
            polys[stem] = regions
    return polys


def rasterize(regions: list[tuple[np.ndarray, np.ndarray]],
              shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rr, cc in regions:
        r, c = draw_polygon(rr, cc, shape=shape)
        mask[r, c] = True
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def load_classes() -> dict[str, str]:
    """WBC class per stem, from the class-foldered GrTh.rar copy of the same GT."""
    return {p.stem: p.parent.name.lower()
            for p in GRTH.glob("*/*") if p.is_file()}


def build() -> pd.DataFrame:
    manifest = pd.read_parquet(RAABIN / "gt_manifest.parquet")
    polys = load_polygons()
    classes = load_classes()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in manifest.itertuples():
        stem = Path(rec.bmp_mask_path).stem
        bmp = np.asarray(Image.open(RAABIN / rec.bmp_mask_path))
        if bmp.ndim == 3:
            bmp = bmp[..., 0]

        wholecell = bmp > 0
        nucleus = bmp == BMP_NUCLEUS
        cytoplasm = bmp == BMP_CYTOPLASM
        has_nucleus = bool(nucleus.any())

        label = np.full(bmp.shape, BG, dtype=np.uint8)
        if has_nucleus:
            label[cytoplasm] = CYT
            label[nucleus] = NUC
        else:
            # Second microscope: interior is annotated, but not partitioned.
            label[wholecell] = CELL
        Image.fromarray(label).save(OUT / f"{stem}.png", optimize=True)

        # The polygons are only a cross-check now. On the first microscope they
        # should agree with the 100-region (confirming the inverted palette); on
        # the second they agree with the whole cell (confirming they are not
        # nuclei and that these 218 cells carry no nucleus ground truth).
        poly = rasterize(polys[stem], bmp.shape)
        rows.append({
            "cell_id": rec.cell_id,
            "stem": stem,
            "class_label": classes.get(stem),
            "device": rec.device,
            "gt_split": rec.gt_split,
            "film_id": rec.film_id,
            "image_path": rec.image_path,
            "mask_path": str((OUT / f"{stem}.png").relative_to(ROOT)),
            "has_nucleus_gt": has_nucleus,
            "n_polygon_regions": len(polys[stem]),
            "nucleus_px": int(nucleus.sum()),
            "cytoplasm_px": int(cytoplasm.sum()),
            "wholecell_px": int(wholecell.sum()),
            "iou_polygon_vs_bmp_nucleus": iou(poly, nucleus) if has_nucleus else np.nan,
            "iou_polygon_vs_wholecell": iou(poly, wholecell),
        })

    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    out = RAABIN / "gt_masks.parquet"
    df.to_parquet(out, index=False)

    nuc = df[df.has_nucleus_gt]
    agree = nuc.iou_polygon_vs_bmp_nucleus
    summary = {
        "n_cells": len(df),
        "n_with_nucleus_gt": int(df.has_nucleus_gt.sum()),
        "n_wholecell_only": int((~df.has_nucleus_gt).sum()),
        "by_device": df.device.value_counts().to_dict(),
        "class_by_nucleus_gt": {
            k: v.class_label.value_counts().to_dict()
            for k, v in df.groupby("has_nucleus_gt")
        },
        "unlabelled_cells": int(df.class_label.isna().sum()),
        "cells_with_empty_compartment": int(
            ((nuc.nucleus_px == 0) | (nuc.cytoplasm_px == 0)).sum()),
        # First microscope: polygons should match the 100-region (palette check).
        "polygon_vs_bmp_nucleus_iou_median": float(agree.median()),
        "polygon_vs_bmp_nucleus_frac_above_0.8": float((agree > 0.8).mean()),
        # Second microscope: polygons match the whole cell, so they are not nuclei.
        "polygon_vs_wholecell_iou_median_by_device": {
            k: float(v.iou_polygon_vs_wholecell.median())
            for k, v in df.groupby("device")
        },
        "nc_ratio_median_first_microscope": float(
            (nuc.nucleus_px / nuc.wholecell_px).median()),
    }
    (RAABIN / "gt_masks_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[masks] wrote {len(df):,} label maps to {OUT} and {out}")


if __name__ == "__main__":
    main()
