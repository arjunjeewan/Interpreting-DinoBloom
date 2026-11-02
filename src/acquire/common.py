"""Shared helpers for dataset acquisition modules.

src/acquire/ is the ONLY package permitted to touch the network, and it only
ever runs locally. Every module here must be idempotent:
re-running must not re-download.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# Normalized index schema, identical for every dataset.
INDEX_COLUMNS = [
    "cell_id",
    "path",
    "class_label",
    "patient_id",
    "center",
    "device",
    "split",
]

USER_AGENT = "cytology-fm-research/0.1 (academic; arjun.jwn@gmail.com)"


def wget(url: str, dest: Path, log: Path | None = None) -> Path:
    """Resumable download with a polite user agent. Idempotent."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget", "-c", "--tries=20", "--timeout=60", "--waitretry=15",
        "--user-agent", USER_AGENT, "-O", str(dest), url,
    ]
    if log:
        cmd += ["-o", str(log)]
    subprocess.run(cmd, check=True)
    return dest


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_index(df: pd.DataFrame, dataset: str) -> Path:
    """Validate against the normalized schema and write index.parquet."""
    missing = set(INDEX_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{dataset}: index is missing columns {sorted(missing)}")
    if df["cell_id"].duplicated().any():
        n = int(df["cell_id"].duplicated().sum())
        raise ValueError(f"{dataset}: {n} duplicate cell_id values")

    # Schema columns first, then any dataset-specific locator columns (e.g. the
    # parquet shard + row a crop lives in, for datasets kept in shards).
    extra = [c for c in df.columns if c not in INDEX_COLUMNS]
    df = df[INDEX_COLUMNS + extra].copy()
    out = ROOT / "data" / "interim" / dataset / "index.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    summary = {
        "dataset": dataset,
        "n_rows": len(df),
        "n_classes": int(df["class_label"].nunique()),
        "splits": df["split"].value_counts().to_dict(),
        "devices": df["device"].value_counts().to_dict(),
        "centers": df["center"].value_counts().to_dict(),
        "n_patients": int(df["patient_id"].nunique(dropna=True)),
        "patient_id_available": bool(df["patient_id"].notna().any()),
    }
    (out.parent / "inventory.json").write_text(json.dumps(summary, indent=2))
    print(f"[{dataset}] wrote {out} : {len(df):,} rows, "
          f"{summary['n_classes']} classes, splits={summary['splits']}")
    return out
