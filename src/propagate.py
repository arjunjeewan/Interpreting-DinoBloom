"""Propagate morphometry to unmasked cells: segment, then measure.

Runs the Raabin-trained U-Net over Matek BM, Raabin (all splits) and PBC, and
computes the eleven descriptors on each predicted label map. HiCervix is
deliberately excluded -- cervical cytology is outside the segmenter's training
domain and propagated descriptors there would not be trustworthy (protocol.md
section 5).

**Units caveat, important for Stage 5.** Descriptors are computed at each
dataset's native crop resolution, and the three datasets have different crop
geometries and fields of view (Raabin 575x575, Matek BM 250x250, PBC 360x363)
with no published micron calibration for any of them. The area-like descriptors
-- nuclear_area, cytoplasmic_area, integrated_optical_density -- are therefore
comparable *within* a dataset but not *across* datasets, and any cross-dataset
analysis must standardize per dataset first. `crop_h`/`crop_w` are recorded on
every row so that normalization stays possible. The other eight descriptors are
scale-invariant by construction.

Segmentation is out-of-domain for every target here (the model is trained on
Raabin alone), so held-out Raabin Dice is an upper bound on propagation quality
rather than an estimate of it. Gate A is the check that actually matters.

fp16 autocast for inference, fp32 elsewhere (Volta: never bfloat16).
No network access.
"""

from __future__ import annotations

import argparse
import io
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.env import ROOT, go_offline, pin_numerics
from src.morphometry import DESCRIPTORS, descriptors
from src.segment import load_model, predict

go_offline()
pin_numerics()

INTERIM = ROOT / "data" / "interim"
CHUNK = 1536          # cells held in memory between GPU pass and CPU pass
GPU_BATCH = 64
N_PROC = 12

# Where a dataset's `path` column is rooted. Matek BM is not on this list: it is
# stored as parquet shards with embedded JPEG bytes (171k loose files would be
# far slower to read and to stage for the HPC).
IMAGE_ROOT = {
    "raabin": INTERIM / "raabin",
    "pbc": INTERIM / "pbc" / "PBC_dataset_normal_DIB",
}


def _safe_open(source) -> Image.Image | None:
    """Decode an image, returning None if it is corrupt.

    Both PBC and the Matek shards contain a few unreadable files. A truncated
    JPEG could be force-loaded, but a partially decoded cell yields garbage
    morphometry, so broken images are dropped and counted instead.
    """
    try:
        return Image.open(source).convert("RGB")
    except (OSError, ValueError):
        return None


def iter_images(dataset: str, index: pd.DataFrame):
    """Yield (cell_id, PIL.Image | None) in index order, chunk-friendly."""
    if dataset == "matek_bm":
        for shard, group in index.groupby("shard", sort=False):
            # The index stores shard paths relative to the project root, so
            # resolve against ROOT rather than the current working directory.
            table = pd.read_parquet(ROOT / shard, columns=["image"])
            for rec in group.itertuples():
                yield rec.cell_id, _safe_open(
                    io.BytesIO(table.image.iloc[rec.row]["bytes"]))
    else:
        root = IMAGE_ROOT[dataset]
        for rec in index.itertuples():
            yield rec.cell_id, _safe_open(root / rec.path)


def _measure(args) -> dict:
    rgb, label = args
    return descriptors(rgb, label)


def run(dataset: str, seed: int, device: str = "cuda", limit: int | None = None) -> pd.DataFrame:
    index = pd.read_parquet(INTERIM / dataset / "index.parquet")
    if limit:
        index = index.head(limit)
    model = load_model(seed=seed, device=device)

    rows: list[dict] = []
    unreadable: list[str] = []
    pool = Pool(N_PROC)
    try:
        buf_ids: list[str] = []
        buf_imgs: list[Image.Image] = []

        def flush():
            if not buf_imgs:
                return
            labels = []
            for i in range(0, len(buf_imgs), GPU_BATCH):
                labels.extend(predict(model, buf_imgs[i:i + GPU_BATCH], device=device))
            payload = [(np.asarray(im), lab) for im, lab in zip(buf_imgs, labels)]
            for cid, im, d in zip(buf_ids, buf_imgs,
                                  pool.map(_measure, payload, chunksize=16)):
                d["cell_id"] = cid
                d["crop_w"], d["crop_h"] = im.size
                rows.append(d)
            buf_ids.clear()
            buf_imgs.clear()

        for i, (cid, img) in enumerate(iter_images(dataset, index), 1):
            if img is None:
                unreadable.append(cid)
                continue
            buf_ids.append(cid)
            buf_imgs.append(img)
            if len(buf_imgs) >= CHUNK:
                flush()
                print(f"[propagate] {dataset} {i:,}/{len(index):,}", flush=True)
        flush()
    finally:
        pool.close()
        pool.join()

    if unreadable:
        print(f"[propagate] {dataset}: {len(unreadable)} unreadable images dropped")
        (ROOT / "results" / f"unreadable_{dataset}.json").write_text(
            json.dumps(unreadable, indent=2))

    df = pd.DataFrame(rows)
    df = index[["cell_id", "class_label", "device", "split"]].merge(df, on="cell_id")
    df["seg_seed"] = seed
    df["scale_units"] = "pixels"
    return df[["cell_id", "class_label", "device", "split", "seg_seed",
               "scale_units", "crop_w", "crop_h", "qc_pass"] + DESCRIPTORS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["raabin", "pbc", "matek_bm"])
    ap.add_argument("--seed", type=int, default=0,
                    help="segmenter seed; propagation uses one, Gate A reports over 3")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for dataset in args.datasets:
        df = run(dataset, seed=args.seed, limit=args.limit)
        out = out_dir / f"morphometry_{dataset}_seed{args.seed}.parquet"
        df.to_parquet(out, index=False)
        failed = int(df.nc_ratio.isna().sum())
        # Platelets are anucleate cell fragments, so a missing nucleus is the
        # correct answer for them, not a segmentation failure. Reported apart.
        by_class = df[df.nc_ratio.isna()].class_label.value_counts().to_dict()
        summary[dataset] = {
            "n": len(df),
            "n_no_nucleus": failed,
            "frac_no_nucleus": round(failed / max(len(df), 1), 5),
            "no_nucleus_by_class": by_class,
            "n_qc_fail": int((~df.qc_pass).sum()),
            "frac_qc_fail": round(float((~df.qc_pass).mean()), 5),
            "nc_ratio_median": float(df.nc_ratio.median()),
            "crop": [int(df.crop_w.iloc[0]), int(df.crop_h.iloc[0])],
        }
        print(f"[propagate] {dataset}: {len(df):,} cells, {failed} failed, "
              f"median N:C {df.nc_ratio.median():.3f} -> {out}")

    path = out_dir / f"propagation_summary_seed{args.seed}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"[propagate] wrote {path}")


if __name__ == "__main__":
    main()
