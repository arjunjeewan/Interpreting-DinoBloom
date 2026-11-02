"""Raabin-WBC (Mousavi Kouzehkanan et al., Sci Rep 2022 12(1):1123).

The device-shift dataset: the domain variable for the Stage 7 causal ablation.

Source resolution, 2026-07-30
-----------------------------
raabindata.com was rebuilt on 2026-07-11 (dl.raabindata.com root now serves a
DirectAdmin placeholder dated "Sat Jul 11 11:06:06 2026"). Every granular
archive listed on the old /free-data/ page 404s. Only two aggregate bundles
survive; the WBC one is 51.9 GiB and does contain every sub-archive we need.
Old URLs recovered from a Wayback snapshot and logged to RESOLVED_URLS.txt.

Archive integrity: the bundle passes `unrar t` with "All OK" on every member.
Note the on-disk file is ~166 MB larger than the advertised content-length
because two wget processes briefly raced on it; RAR ignores trailing bytes
after the end-of-archive marker and all CRCs verify, so the data is sound.

Device provenance (must NOT be flattened)
---------------------------------------------------------------
Encoded in the filename convention, which we parse rather than discard:
  95-<film>_<n>_<k>.jpg   -> first microscope,  Olympus CX18 + Samsung Galaxy S5
  YYYYMMDD_HHMMSS*.jpg    -> second microscope, Zeiss + LG G3
  <digits>.jpg            -> Test-B only; acquisition not encoded in the name

Ground truth masks (Stage 2 input) — two complementary sources
--------------------------------------------------------------
1. GroundTruths_bmpformat.rar, 575x575 uint8 masks. Per the authors' own
   NucCytSplitter.py shipped in the bundle, the palette is INVERTED relative to
   the intuitive convention:
        value 100 = NUCLEUS
        value 255 = CYTOPLASM
   Taking 255 for nucleus would silently invert every N:C ratio in the paper.
   Crucially, only the 927 first-microscope masks carry both labels; the 218
   second-microscope masks contain 255 only, and that region covers the WHOLE
   CELL (verified visually and by area: median 56,428 px vs a first-microscope
   whole-cell median of 51,986 px, against a nucleus median of 31,888 px).
2. maskRCNN.rar, VIA-format polygons labelled "nucleus", covering all 1,145
   cells including all 218 second-microscope ones (1,324 polygons, since
   multi-lobed neutrophil nuclei are annotated as several regions).

Source 2 is what makes N:C ratio computable on BOTH devices:
    nucleus   = rasterized maskRCNN polygons
    wholecell = BMP (100|255) first microscope, BMP (255) second microscope
    cytoplasm = wholecell - nucleus
The two sources agree where they overlap: median IoU 0.889 between polygon
nuclei and BMP==100 on the 927 first-microscope cells (97.3% above 0.8), and
second-microscope polygons sit inside the whole-cell mask with median
containment 0.980 (99.5% above 0.95).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from src.acquire.common import ROOT, write_index

DATASET = "raabin"
RAW = ROOT / "data" / "raw" / DATASET
INTERIM = ROOT / "data" / "interim" / DATASET
BUNDLE = RAW / "WBCData.rar"
UNRAR = ROOT / "tools" / "unrar"
URL = "https://dl.raabindata.com/WBCData.rar"
BUNDLE_BYTES = 55781603692  # advertised content-length

CROP = INTERIM / "_crop"
GT = INTERIM / "_gt"

# Splits as distributed. Train is the split DinoBloom pretrained on; TestA and
# Test-B are not. See data/pretraining_status.csv.
SPLIT_DIRS = {"Train": "train", "TestA": "testA", "Test-B": "testB"}

REFERENCE = {
    "n_cropped_total": 16633,
    "n_train": 10175,      # exactly DinoBloom's stated Raabin contribution
    "n_testA": 4339,
    "n_testB": 2119,
    "n_gt_cells": 1145,
    "gt_composition": "242 lymphocyte, 242 monocyte, 242 neutrophil, "
                      "201 eosinophil, 218 basophil",
}

DEVICE_FIRST = "olympus_cx18_samsung_galaxy_s5"
DEVICE_SECOND = "zeiss_lg_g3"
DEVICE_TESTB = "testB_unspecified"

_RE_FIRST = re.compile(r"^95-[\d-]+_")
_RE_SECOND = re.compile(r"^\d{8}_\d{6}")
_RE_NUMERIC = re.compile(r"^\d+\.(jpg|png)$", re.I)


def device_of(filename: str) -> str:
    if _RE_FIRST.match(filename):
        return DEVICE_FIRST
    if _RE_SECOND.match(filename):
        return DEVICE_SECOND
    if _RE_NUMERIC.match(filename):
        return DEVICE_TESTB
    return "unknown"


def film_of(filename: str) -> str | pd.NA:
    """Blood-film identifier, the closest thing Raabin has to a patient key."""
    m = _RE_FIRST.match(filename)
    if m:
        return filename[:m.end() - 1]
    m = _RE_SECOND.match(filename)
    if m:
        return filename[:8]  # acquisition date
    return pd.NA


def download() -> None:
    from src.acquire.common import wget

    if BUNDLE.exists() and BUNDLE.stat().st_size >= BUNDLE_BYTES:
        print(f"[{DATASET}] bundle already present, skipping download")
        return
    wget(URL, BUNDLE, log=RAW / "wget_WBCData.log")


def _unrar(archive: Path, dest: Path, pattern: str | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [str(UNRAR), "e" if pattern else "x", "-y", "-idq", str(archive)]
    if pattern:
        cmd.append(pattern)
    cmd.append(str(dest) + "/")
    subprocess.run(cmd, check=True)


def extract() -> None:
    """Pull the cropped cells and the two ground-truth sources out of the bundle."""
    if not (CROP / "Train").exists():
        _unrar(BUNDLE, CROP, "Index of WBC Data/Index of WBC Cropped_double_labeled/*")
        for name in ["Train.rar", "TestA.rar"]:
            _unrar(CROP / name, CROP)
        subprocess.run(["unzip", "-q", "-o", str(CROP / "TestB.zip"), "-d", str(CROP)],
                       check=True)
    if not (GT / "bmp").exists():
        _unrar(BUNDLE, GT, "Index of WBC Data/Index of WBC nucleus_cytoplasm_GT/*")
        _unrar(GT / "GroundTruths_bmpformat.rar", GT / "bmp")
        _unrar(GT / "maskRCNN.rar", GT / "mrcnn")


def build_index() -> pd.DataFrame:
    rows = []
    for src_dir, split in SPLIT_DIRS.items():
        for p in sorted((CROP / src_dir).rglob("*")):
            if not p.is_file() or p.suffix.lower() not in (".jpg", ".png", ".bmp"):
                continue
            cls = p.parent.name.lower()
            rows.append({
                "cell_id": f"{DATASET}/{split}/{cls}/{p.name}",
                "path": str(p.relative_to(INTERIM)),
                "class_label": cls,
                # No patient key exists. The blood-film id is the nearest
                # grouping variable and is recorded so film-disjoint splits
                # remain possible; it is NOT a patient id.
                "patient_id": pd.NA,
                "center": "raabin_tehran",
                "device": device_of(p.name),
                "split": split,
                "film_id": film_of(p.name),
            })
    return pd.DataFrame(rows)


def gt_manifest() -> pd.DataFrame:
    """One row per ground-truth cell, with both mask sources located."""
    poly_index = {}
    for fname, sub in [("Train_GT.json", "Train"), ("Test_GT.json", "Test")]:
        for entry in json.load(open(GT / "mrcnn" / "maskRCNN" / fname)).values():
            poly_index[entry["filename"].replace(".jpg", ".bmp")] = (sub, fname)

    rows = []
    for sub in ["Train", "Test"]:
        for p in sorted((GT / "bmp" / f"{sub}_GT_bmp").glob("*.bmp")):
            dev = device_of(p.name)
            rows.append({
                "cell_id": f"{DATASET}/gt/{sub}/{p.name}",
                "image_path": str((GT / "bmp" / f"{sub}_bmp" / p.name).relative_to(INTERIM)),
                "bmp_mask_path": str(p.relative_to(INTERIM)),
                "device": dev,
                "gt_split": sub.lower(),
                # first microscope: BMP gives nucleus(100) AND cytoplasm(255)
                # second microscope: BMP gives whole cell(255) only
                "bmp_has_nucleus": dev == DEVICE_FIRST,
                "bmp_has_cytoplasm": dev == DEVICE_FIRST,
                "bmp_is_wholecell_only": dev == DEVICE_SECOND,
                "polygon_nucleus_available": p.name in poly_index,
                "film_id": film_of(p.name),
            })
    return pd.DataFrame(rows)


def main() -> None:
    download()
    extract()

    df = build_index()
    counts = {v: int((df["split"] == v).sum()) for v in SPLIT_DIRS.values()}
    expected = {"train": REFERENCE["n_train"], "testA": REFERENCE["n_testA"],
                "testB": REFERENCE["n_testB"]}
    if counts != expected:
        raise SystemExit(f"[{DATASET}] split counts {counts} != expected {expected}")
    print(f"[{DATASET}] cropped cells verified: {counts}, total {len(df):,}")
    print(f"[{DATASET}] devices: {df['device'].value_counts().to_dict()}")

    gt = gt_manifest()
    if len(gt) != REFERENCE["n_gt_cells"]:
        raise SystemExit(
            f"[{DATASET}] expected {REFERENCE['n_gt_cells']} GT cells, found {len(gt)}"
        )
    if not gt["polygon_nucleus_available"].all():
        n = int((~gt["polygon_nucleus_available"]).sum())
        raise SystemExit(f"[{DATASET}] {n} GT cells lack a nucleus polygon")
    gt_out = INTERIM / "gt_manifest.parquet"
    gt.to_parquet(gt_out, index=False)
    print(f"[{DATASET}] wrote {gt_out}: {len(gt):,} GT cells, "
          f"devices {gt['device'].value_counts().to_dict()}")
    print(f"[{DATASET}] nucleus available for all {len(gt)} via maskRCNN polygons; "
          f"BMP cytoplasm only for the {int(gt['bmp_has_cytoplasm'].sum())} "
          f"first-microscope cells")

    write_index(df, DATASET)


if __name__ == "__main__":
    main()
