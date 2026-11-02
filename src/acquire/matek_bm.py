"""Matek bone marrow cytomorphology (MLL/Helmholtz/Fraunhofer), Blood 2021.

Source resolution, 2026-07-30
-----------------------------
The official TCIA collection (DOI 10.7937/TCIA.AXH3-T579) distributes this
dataset as a single 6.8 GB IBM Aspera Faspex package. The TCIA landing page
states verbatim that "Download requires IBM-Aspera-Connect plugin"; the Faspex 5
public-package endpoint rejects unauthenticated HTTP (401/404 across the v5 API
surface), so there is no plain-HTTP route to the original archive.

We therefore use the unofficial HuggingFace mirror, and verify it against the TCIA/paper reference before any result may
be reported. See data/raw/matek_bm/MIRROR_VERIFICATION.md for the audit.

Layout note
-----------
The mirror ships 16 parquet shards with embedded JPEG bytes and, importantly,
the ORIGINAL TCIA filenames (e.g. "ABE_00001.jpg"). We deliberately do NOT
explode this into 171k individual files: hundreds of thousands of small files starve the input pipeline and make the
air-gap transfer painful. The index instead carries (shard, row) locators and
the Stage 3 dataloader reads JPEG bytes straight from the shards.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.acquire.common import ROOT, write_index

DATASET = "matek_bm"
RAW = ROOT / "data" / "raw" / DATASET
MIRROR = RAW / "hf_mirror"
HF_REPO = "ekim15/bone_marrow_cell_dataset"

# Reference figures from the TCIA landing page and Matek et al., Blood 2021.
REFERENCE = {
    "n_images_paper": 171374,
    "n_images_tcia_metadata": 171375,  # TCIA "Images" field; off by one vs paper
    "n_subjects": 945,
    "n_classes": 21,
    "license": "CC BY 4.0",
    "stain": "May-Grunwald-Giemsa/Pappenheim",
    "magnification": "40x oil immersion, brightfield",
    "classes": [
        "ABE", "ART", "BAS", "BLA", "EBO", "EOS", "FGC", "HAC", "KSC", "LYI",
        "LYT", "MMZ", "MON", "MYB", "NGB", "NGS", "NIF", "OTH", "PEB", "PLM",
        "PMO",
    ],
}

# The granulocytic maturation series used in Stage 6, in biological order.
MATURATION_SERIES = ["BLA", "PMO", "MYB", "MMZ", "NGB", "NGS"]


def download() -> None:
    """Fetch the mirror. Idempotent: hf skips files already present."""
    import subprocess

    if MIRROR.exists() and list(MIRROR.glob("data/*.parquet")):
        print(f"[{DATASET}] mirror already present, skipping download")
        return
    MIRROR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["hf", "download", HF_REPO, "--repo-type", "dataset",
         "--local-dir", str(MIRROR)],
        check=True,
    )


def build_index() -> pd.DataFrame:
    shards = sorted(glob.glob(str(MIRROR / "data" / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"[{DATASET}] no parquet shards under {MIRROR}")

    names = REFERENCE["classes"]
    rows = []
    for shard in shards:
        table = pq.read_table(shard, columns=["image", "label"])
        shard_rel = str(Path(shard).relative_to(ROOT))
        labels = table.column("label").to_pylist()
        # Take the struct's "path" sub-field in Arrow so the JPEG bytes are
        # never materialized as Python objects.
        fnames = table.column("image").combine_chunks().field("path").to_pylist()
        for i, (lab, fname) in enumerate(zip(labels, fnames)):
            cls = names[lab]
            rows.append({
                "cell_id": f"{DATASET}/{cls}/{fname}",
                # Original TCIA-style relative path, for provenance only. The
                # bytes live in the shard; see the shard/row columns.
                "path": f"{cls}/{fname}",
                "class_label": cls,
                # The TCIA release does not ship per-image patient identifiers
                # (filenames encode only class + running index), so patient-
                # disjoint splitting is impossible for this dataset. Recorded as
                # a limitation rather than silently faked.
                "patient_id": pd.NA,
                "center": "MLL_Munich",
                "device": "MLL_brightfield_40x_oil",
                # Split assigned by src/splits.py, not by the mirror's own
                # random split. Left unassigned here.
                "split": pd.NA,
                "shard": shard_rel,
                "row": i,
            })
    return pd.DataFrame(rows)


def verify(df: pd.DataFrame) -> dict:
    """Audit the mirror against the TCIA/paper reference. Blocking."""
    counts = df["class_label"].value_counts().to_dict()
    checks = {
        "total_images": {
            "observed": len(df),
            "expected_paper": REFERENCE["n_images_paper"],
            "expected_tcia_metadata": REFERENCE["n_images_tcia_metadata"],
            "pass": len(df) == REFERENCE["n_images_paper"],
        },
        "n_classes": {
            "observed": df["class_label"].nunique(),
            "expected": REFERENCE["n_classes"],
            "pass": df["class_label"].nunique() == REFERENCE["n_classes"],
        },
        "class_set": {
            "observed": sorted(df["class_label"].unique()),
            "expected": sorted(REFERENCE["classes"]),
            "pass": sorted(df["class_label"].unique()) == sorted(REFERENCE["classes"]),
        },
        "maturation_series_present": {
            "observed": {c: counts.get(c, 0) for c in MATURATION_SERIES},
            "pass": all(counts.get(c, 0) > 0 for c in MATURATION_SERIES),
        },
    }
    report = {
        "dataset": DATASET,
        "source_used": f"huggingface:{HF_REPO}",
        "official_source": "TCIA 10.7937/TCIA.AXH3-T579 (Aspera Faspex, no HTTP route)",
        "verified_on": "2026-07-30",
        "checks": checks,
        "per_class_counts": counts,
        "known_limitations": [
            "No per-image patient_id in the TCIA release; patient-disjoint "
            "splits are impossible for this dataset.",
            "Per-class counts could not be diffed against a machine-readable "
            "TCIA manifest (Aspera-only); they are recorded here for manual "
            "comparison against Matek et al., Blood 2021 Table 1.",
            "TCIA metadata reports 171,375 images vs 171,374 in the paper and "
            "in this mirror; a one-image discrepancy at source.",
        ],
    }
    (RAW / "MIRROR_VERIFICATION.md").parent.mkdir(parents=True, exist_ok=True)
    (RAW / "mirror_verification.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    download()
    df = build_index()
    report = verify(df)
    failed = [k for k, v in report["checks"].items() if not v["pass"]]
    if failed:
        raise SystemExit(f"[{DATASET}] MIRROR VERIFICATION FAILED: {failed}")
    print(f"[{DATASET}] mirror verification passed against TCIA/paper reference")
    write_index(df, DATASET)


if __name__ == "__main__":
    main()
