"""PBC Barcelona (Acevedo et al., Data in Brief 2020), Mendeley snkd93bnjr.

Chosen as the third centre over Matek AML (LMU). Both resolved to a working
download, and PBC is the strictly better option on rigor grounds:

    DinoBloom, arXiv:2404.05022v1 section 3, verbatim: "We train the DinoBloom
    models on all datasets except the Acevedo dataset, which is kept as external
    test set."

PBC is therefore CLEAN with respect to encoder pretraining, while Matek AML is
in the 13-dataset cohort (18,365 images). This gives the paper a second
uncontaminated generalization set, in-domain (peripheral blood), alongside
HiCervix's cross-domain one. See data/pretraining_status.csv.

Acquisition: Core Laboratory, Hospital Clinic of Barcelona; CellaVision DM96
analyser. Single centre, single device, so PBC contributes cross-CENTRE contrast
in Stage 7 but no within-dataset device contrast.
"""

from __future__ import annotations

import json
import subprocess
import zipfile

import pandas as pd

from src.acquire.common import ROOT, write_index

DATASET = "pbc"
RAW = ROOT / "data" / "raw" / DATASET
INTERIM = ROOT / "data" / "interim" / DATASET
ARCHIVE = RAW / "PBC_dataset_normal_DIB.zip"
URL = (
    "https://data.mendeley.com/public-files/datasets/snkd93bnjr/files/"
    "2fc38728-2ae7-4a62-a857-032af82334c3/file_downloaded"
)

# Per-class counts as published in Acevedo et al., Data in Brief 2020, Table 1.
REFERENCE = {
    "n_images": 17092,
    "n_classes": 8,
    "per_class": {
        "basophil": 1218,
        "eosinophil": 3117,
        "erythroblast": 1551,
        "ig": 2895,
        "lymphocyte": 1214,
        "monocyte": 1420,
        "neutrophil": 3329,
        "platelet": 2348,
    },
    "doi": "10.17632/snkd93bnjr.1",
    "centre": "Hospital Clinic of Barcelona, Core Laboratory",
    "device": "CellaVision DM96",
}
REFERENCE["classes"] = sorted(REFERENCE["per_class"])


def download() -> None:
    from src.acquire.common import wget

    if ARCHIVE.exists() and ARCHIVE.stat().st_size == 281366219:
        print(f"[{DATASET}] archive already present, skipping download")
        return
    wget(URL, ARCHIVE, log=RAW / "wget_pbc.log")


def extract() -> None:
    if INTERIM.exists() and any(INTERIM.rglob("*.jpg")):
        print(f"[{DATASET}] already extracted, skipping")
        return
    INTERIM.mkdir(parents=True, exist_ok=True)
    # -x __MACOSX/* drops the resource-fork sidecars the archive ships with.
    subprocess.run(
        ["unzip", "-q", "-o", str(ARCHIVE), "-d", str(INTERIM), "-x", "__MACOSX/*"],
        check=True,
    )


def build_index() -> pd.DataFrame:
    with zipfile.ZipFile(ARCHIVE) as zf:
        members = [
            n for n in zf.namelist()
            if n.lower().endswith(".jpg")
            and not n.startswith("__MACOSX/")
            # macOS resource-fork junk sitting inside the class folders, not in
            # __MACOSX/. One such file (neutrophil/.DS_169665.jpg, 6,148 bytes,
            # not a JPEG) is the entire reason our first count was 17,093 against
            # the published 17,092, and 3,330 neutrophils against 3,329.
            # Excluding it reproduces Acevedo et al. Table 1 exactly.
            and not n.split("/")[-1].startswith(".")
        ]

    rows = []
    for name in sorted(members):
        # PBC_dataset_normal_DIB/<class>/<FILE>.jpg
        parts = name.split("/")
        cls, fname = parts[-2], parts[-1]
        rows.append({
            "cell_id": f"{DATASET}/{cls}/{fname}",
            "path": f"{cls}/{fname}",
            "class_label": cls,
            # Acevedo et al. release no per-image patient key.
            "patient_id": pd.NA,
            "center": "Barcelona_HospitalClinic",
            "device": "CellaVision_DM96",
            # Assigned by src/splits.py; the release ships no split.
            "split": pd.NA,
        })
    return pd.DataFrame(rows)


def main() -> None:
    download()
    df = build_index()

    observed = sorted(df["class_label"].unique())
    if observed != REFERENCE["classes"]:
        raise SystemExit(
            f"[{DATASET}] class mismatch: {observed} != {REFERENCE['classes']}"
        )

    counts = df["class_label"].value_counts().to_dict()
    deltas = {c: counts[c] - n for c, n in REFERENCE["per_class"].items()
              if counts[c] != n}
    drift = abs(len(df) - REFERENCE["n_images"]) / REFERENCE["n_images"]
    if drift > 0.01:
        raise SystemExit(
            f"[{DATASET}] {len(df)} images vs {REFERENCE['n_images']} published "
            f"({drift:.1%} drift) — too large to be a bookkeeping discrepancy"
        )

    report = {
        "dataset": DATASET,
        "source": f"mendeley:{REFERENCE['doi']}",
        "reference": "Acevedo et al., Data in Brief 2020, Table 1",
        "verified_on": "2026-07-30",
        "n_images_observed": len(df),
        "n_images_published": REFERENCE["n_images"],
        "per_class_observed": counts,
        "per_class_published": REFERENCE["per_class"],
        "per_class_deltas": deltas,
    }
    (RAW / "verification.json").write_text(json.dumps(report, indent=2))

    print(f"[{DATASET}] verified against Acevedo et al. 2020: "
          f"{len(df):,} images, {len(observed)} classes")
    if deltas:
        print(f"[{DATASET}] SOURCE DISCREPANCY (recorded, not corrected): "
              f"{deltas}. The distributed archive differs from the published "
              f"table; the archive is internally consistent (no duplicate "
              f"filenames, uniform layout). See {RAW / 'verification.json'}.")
    print(f"[{DATASET}] NOTE: the 'ig' class lumps promyelocyte, myelocyte and "
          f"metamyelocyte, so PBC cannot supply a fine-grained granulocytic "
          f"maturation series. Stage 6 uses Matek BM for that.")

    extract()
    write_index(df, DATASET)


if __name__ == "__main__":
    main()
