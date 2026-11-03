"""Stage 3: cache encoder embeddings for every crop in every dataset.

Four encoders (protocol.md section 6) over the four indexed datasets, 245,328
crops each. Inference in fp16 autocast, outputs cast to and stored as fp32
(Volta: no bf16, no FlashAttention 2).

Output per (encoder, dataset):
    data/embeddings/<encoder>/<dataset>.npy       (N, D) float32 memmap
    data/embeddings/<encoder>/<dataset>.parquet   cell_id-aligned metadata
    data/embeddings/<encoder>/<dataset>.json      resume marker + summary

Rows are written strictly in index order, so the resume marker is a single
integer: the number of leading rows already computed. An interrupted pass
restarts from there rather than from zero. The marker only ever lands on a
batch boundary, so a resumed pass reproduces an uninterrupted one bit for bit
provided --batch-size is unchanged (verified; at a boundary the arrays are
identical, and forcing a mid-batch marker perturbs 25 rows of 11,633 by a mean
of 3e-6, which is fp16 batch-composition noise, not misalignment).

**Preprocessing follows each encoder's own model card**, not a single shared
recipe. Resize to 224x224 and ImageNet normalization are common to all four,
but UNI2-h declares a bilinear resize where the other three declare bicubic
(data/weights/manifest.json); stock DINOv2 preprocessing is not assumed. The filter actually used is
recorded in every output summary.

Weights load from an explicit local directory, never a hub identifier. The
timm kwargs for UNI2-h and Virchow2 are duplicated from each repo's model card
rather than imported from src/acquire/encoders.py, so that the compute path
does not import the network-permitted package at all.

No network access.
"""

from __future__ import annotations

import argparse
import bisect
import io
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import timm
import torch
from numpy.lib.format import open_memmap
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.env import ROOT, go_offline, pin_numerics

go_offline()
pin_numerics()

INTERIM = ROOT / "data" / "interim"
WEIGHTS = ROOT / "data" / "weights"
EMB = ROOT / "data" / "embeddings"

SIZE = 224
MEAN = np.float32([0.485, 0.456, 0.406])
STD = np.float32([0.229, 0.224, 0.225])
FLUSH_ROWS = 8192          # rows between memmap flush and resume-marker update

DATASETS = ["raabin", "pbc", "matek_bm", "hicervix"]

# Where a dataset's `path` column is rooted. Matek BM is absent: it is stored as
# parquet shards with embedded JPEG bytes, read through _ShardReader below.
IMAGE_ROOT = {
    "raabin": INTERIM / "raabin",
    "pbc": INTERIM / "pbc" / "PBC_dataset_normal_DIB",
    "hicervix": INTERIM / "hicervix",
}

RESAMPLE = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}


# ------------------------------------------------------------------ encoders --

@dataclass(frozen=True)
class Encoder:
    local_dir: str
    dim: int
    interpolation: str
    pooling: str
    build: Callable[[Path], torch.nn.Module]
    forward: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]


def _load_state(d: Path) -> dict:
    """Read a local checkpoint, safetensors preferred."""
    st = d / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st))
    for name in ("pytorch_model.bin", "pytorch_model_b.bin"):
        if (d / name).exists():
            return torch.load(d / name, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no checkpoint under {d}")


def _timm_builder(arch: str, kwargs: dict) -> Callable[[Path], torch.nn.Module]:
    def build(d: Path) -> torch.nn.Module:
        model = timm.create_model(arch, pretrained=False, **kwargs)
        result = model.load_state_dict(_load_state(d), strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"{d.name}: state dict does not fit {arch} "
                f"({len(result.missing_keys)} missing, "
                f"{len(result.unexpected_keys)} unexpected); "
                f"missing[:5]={result.missing_keys[:5]} "
                f"unexpected[:5]={result.unexpected_keys[:5]}")
        return model
    return build


def _build_dinov2(d: Path) -> torch.nn.Module:
    # HuggingFace format rather than timm. Trained at 518; interpolate_pos_encoding
    # resamples the position embedding to the 224 grid at forward time.
    from transformers import Dinov2Model
    return Dinov2Model.from_pretrained(str(d), local_files_only=True)


ENCODERS: dict[str, Encoder] = {
    # [APPROVED] primary. pos_embed is (1, 257, 768) on disk, i.e. already the
    # 224/14 = 16x16 grid, so nothing is resampled here.
    "dinobloom": Encoder(
        local_dir="timm_vit_base_patch14_224.dinobloom", dim=768,
        interpolation="bicubic", pooling="cls",
        build=_timm_builder("vit_base_patch14_dinov2",
                            dict(img_size=224, num_classes=0)),
        forward=lambda m, x: m(x),
    ),
    # kwargs from the MahmoodLab/UNI2-h model card, not the timm defaults.
    # [APPROVED 2026-08-04] resize is **bicubic**, deviating from this model
    # card's bilinear. All four encoders now share one resize filter so that
    # table T1 compares dictionaries rather than preprocessing; UNI2-h was the
    # only card specifying bilinear, so unifying on bicubic re-embeds one
    # encoder instead of three. The deviation from the card is stated in the
    # paper and recorded in every output summary.
    "uni2_h": Encoder(
        local_dir="uni2_h", dim=1536,
        interpolation="bicubic", pooling="cls",
        build=_timm_builder("vit_giant_patch14_224", dict(
            img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
            embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
            no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)),
        forward=lambda m, x: m(x),
    ),
    # global_pool="" returns tokens [B, 261, 1280] = CLS + 4 registers + 256
    # patches. [APPROVED] CLS only, against the card's CLS + mean(patch) = 2560.
    "virchow2": Encoder(
        local_dir="virchow2", dim=1280,
        interpolation="bicubic", pooling="cls",
        build=_timm_builder("vit_huge_patch14_224", dict(
            img_size=224, init_values=1e-5, num_classes=0, reg_tokens=4,
            mlp_ratio=5.3375, global_pool="", dynamic_img_size=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)),
        forward=lambda m, x: m(x)[:, 0],
    ),
    "dinov2_l": Encoder(
        local_dir="dinov2_large", dim=1024,
        interpolation="bicubic", pooling="cls",
        build=_build_dinov2,
        # last_hidden_state is post-layernorm in transformers' Dinov2Model.
        forward=lambda m, x: m(x, interpolate_pos_encoding=True).last_hidden_state[:, 0],
    ),
    # [APPROVED 2026-11-03] post hoc baseline with DinoBloom-B's architecture (ViT-B/14, 768-d).
    "dinov2_b": Encoder(
        local_dir="dinov2_base", dim=768,
        interpolation="bicubic", pooling="cls",
        build=_build_dinov2,
        forward=lambda m, x: m(x, interpolate_pos_encoding=True).last_hidden_state[:, 0],
    ),
}


# ------------------------------------------------------------------- reading --

class _ShardReader:
    """Row-group reader for the Matek BM parquet shards.

    The shards embed JPEG bytes: ~11k rows each in row groups of 100 (~4 MB).
    Reading a whole shard per worker would cost 450 MB each, so a worker holds
    one row group instead. Opened lazily so the object survives DataLoader fork.
    """

    def __init__(self) -> None:
        self._files: dict[str, tuple[pq.ParquetFile, list[int]]] = {}
        self._cached: tuple[str, int] | None = None
        self._table = None

    def _file(self, shard: str):
        if shard not in self._files:
            handle = pq.ParquetFile(ROOT / shard)
            offsets, total = [], 0
            for i in range(handle.num_row_groups):
                offsets.append(total)
                total += handle.metadata.row_group(i).num_rows
            self._files[shard] = (handle, offsets)
        return self._files[shard]

    def read(self, shard: str, row: int) -> bytes:
        handle, offsets = self._file(shard)
        group = bisect.bisect_right(offsets, row) - 1
        if self._cached != (shard, group):
            self._table = handle.read_row_group(group, columns=["image"])
            self._cached = (shard, group)
        return self._table.column("image")[row - offsets[group]]["bytes"].as_py()


def _pad_square(im: Image.Image) -> Image.Image:
    """Pad to square with the crop's own border median, before any resize.

    Resizing straight to (SIZE, SIZE) is anisotropic, and crop aspect differs
    systematically by dataset -- Raabin 1.00 and PBC 1.01 against a HiCervix
    median of 1.24 and a maximum of 4.39 -- so the distortion was itself a
    domain signal, on the one dataset carrying the cross-domain claim.

    The fill is the median of the crop's own border ring rather than a fixed
    constant. How much padding a crop receives is a function of its aspect, so
    a constant fill would make padded area a second dataset-correlated
    artefact; matching each crop's own background keeps it neutral.
    """
    w, h = im.size
    if w == h:
        return im
    a = np.asarray(im)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill)
    out.paste(im, ((side - w) // 2, (side - h) // 2))
    return out


class CropDataset(Dataset):
    """Decode, pad to square, resize to 224 and normalize, in index order.

    Returns (tensor, ok). A crop that fails to decode yields zeros and ok=False:
    the memmap row must exist for the array to stay aligned with the index, so
    the failure is carried as a column rather than by dropping the row.

    pad=False resizes without padding. The cached embeddings in data/embeddings
    (3-4 Aug) predate _pad_square, so that is how they were computed; it differs
    only for non-square crops (PBC's 360 x 363 moves the CLS to a median cosine
    of 0.97 against the cached one).
    """

    def __init__(self, index: pd.DataFrame, dataset: str, interpolation: str, pad: bool = True):
        self.dataset = dataset
        self.resample = RESAMPLE[interpolation]
        self.pad = pad
        self.root = IMAGE_ROOT.get(dataset)
        if dataset == "matek_bm":
            self.locs = list(zip(index["shard"], index["row"]))
            self.reader: _ShardReader | None = None
        else:
            self.paths = index["path"].tolist()

    def __len__(self) -> int:
        return len(self.locs) if self.dataset == "matek_bm" else len(self.paths)

    def _source(self, i: int):
        if self.dataset == "matek_bm":
            if self.reader is None:
                self.reader = _ShardReader()
            return io.BytesIO(self.reader.read(*self.locs[i]))
        return self.root / self.paths[i]

    def __getitem__(self, i: int):
        try:
            im = Image.open(self._source(i)).convert("RGB")
        except (OSError, ValueError):
            return torch.zeros(3, SIZE, SIZE), False
        a = np.asarray((_pad_square(im) if self.pad else im).resize((SIZE, SIZE), self.resample),
                       np.float32) / 255.0
        return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1).copy()), True


def metadata(dataset: str, unreadable: list[str] | None = None) -> pd.DataFrame:
    """Index columns kept alongside the embeddings, in index order.

    `ok` is false for a crop that failed to decode. Its memmap row still exists,
    so that the array stays aligned with the index, but it holds the encoder's
    response to an all-zero image rather than zeros -- it cannot be found by
    inspecting the array. Everything downstream must filter on this column, as
    Stage 2 filters on `qc_pass`.
    """
    index = pd.read_parquet(INTERIM / dataset / "index.parquet")
    splits = pd.read_parquet(INTERIM / dataset / "splits.parquet")
    cols = ["cell_id", "class_label", "center", "device"]
    out = index[cols].merge(splits[["cell_id", "fold"]], on="cell_id", how="left")
    out.insert(1, "dataset", dataset)
    out["ok"] = ~out.cell_id.isin(unreadable or [])
    return out


# ------------------------------------------------------------------ the pass --

class _UtilSampler:
    """Sample GPU utilization while the loop runs.

    Reads `nvidia-smi -lms` rather than torch.cuda.utilization(), which needs
    pynvml; nothing may be installed at runtime. Local
    subprocess only, no network.
    """

    def __init__(self, period_ms: int = 200) -> None:
        self.period_ms, self.samples = period_ms, []
        self._proc = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        for line in self._proc.stdout:
            line = line.strip()
            if line.isdigit():
                self.samples.append(int(line))

    def __enter__(self) -> "_UtilSampler":
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits", f"-lms={self.period_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._proc.terminate()
        self._thread.join(timeout=2.0)

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")


def _loader(index: pd.DataFrame, dataset: str, spec: Encoder,
            batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        CropDataset(index, dataset, spec.interpolation),
        batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=False,
        prefetch_factor=4 if workers else None)


def run(encoder: str, dataset: str, batch_size: int = 96, workers: int = 8,
        device: str = "cuda") -> dict:
    """Embed one dataset with one encoder, resuming from the row marker."""
    spec = ENCODERS[encoder]
    meta = metadata(dataset)
    n = len(meta)

    out_dir = EMB / encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"{dataset}.npy"
    summary_path = out_dir / f"{dataset}.json"

    state = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    done = int(state.get("rows_done", 0))
    unreadable = list(state.get("unreadable", []))

    if emb_path.exists() and done:
        arr = open_memmap(emb_path, mode="r+")
        if arr.shape != (n, spec.dim):
            raise RuntimeError(f"{emb_path} is {arr.shape}, expected {(n, spec.dim)}")
    else:
        arr = open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n, spec.dim))
        done, unreadable = 0, []

    if done >= n:
        print(f"[embed] {encoder}/{dataset}: already complete ({n:,} rows)")
        return state

    index = pd.read_parquet(INTERIM / dataset / "index.parquet").iloc[done:]
    model = spec.build(WEIGHTS / spec.local_dir).to(device).eval()

    print(f"[embed] {encoder}/{dataset}: {n - done:,} of {n:,} rows to go "
          f"(dim {spec.dim}, {spec.interpolation}, batch {batch_size})")

    start, last_report = time.time(), time.time()
    with torch.inference_mode(), _UtilSampler() as util:
        for x, ok in _loader(index, dataset, spec, batch_size, workers):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = spec.forward(model, x)
            arr[done:done + len(z)] = z.float().cpu().numpy()
            if not bool(ok.all()):
                bad = meta["cell_id"].iloc[done:done + len(z)][~ok.numpy()]
                unreadable.extend(bad.tolist())
            done += len(z)

            if done % FLUSH_ROWS < batch_size:
                arr.flush()
                summary_path.write_text(json.dumps(
                    {"rows_done": done, "n": n, "unreadable": unreadable}, indent=1))
            if time.time() - last_report > 30:
                rate = (done - int(state.get("rows_done", 0))) / (time.time() - start)
                print(f"[embed]   {done:,}/{n:,}  {rate:.0f} img/s  "
                      f"gpu {util.mean:.0f}%  eta {(n - done) / max(rate, 1e-6) / 60:.1f} min",
                      flush=True)
                last_report = time.time()

    arr.flush()
    elapsed = time.time() - start
    summary = {
        "encoder": encoder, "dataset": dataset, "n": n, "dim": spec.dim,
        "rows_done": done, "unreadable": unreadable,
        "pooling": spec.pooling, "interpolation": spec.interpolation,
        "size": SIZE, "mean": MEAN.tolist(), "std": STD.tolist(),
        "autocast_dtype": "float16", "storage_dtype": "float32",
        "batch_size": batch_size, "num_workers": workers,
        "seconds": round(elapsed, 1),
        "images_per_second": round((done - int(state.get("rows_done", 0))) / elapsed, 1),
        "gpu_util_mean": round(util.mean, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=1))
    metadata(dataset, unreadable).to_parquet(out_dir / f"{dataset}.parquet", index=False)

    print(f"[embed] {encoder}/{dataset}: {n:,} rows in {elapsed / 60:.1f} min "
          f"({summary['images_per_second']:.0f} img/s, gpu {util.mean:.0f}%), "
          f"{len(unreadable)} unreadable")
    return summary


def benchmark(encoder: str, dataset: str, crops: int, batch_size: int,
              workers: int, device: str = "cuda") -> dict:
    """Time a short pass and report GPU utilization. Writes no embeddings.

    Run this before the full pass: under 60% GPU
    utilization means the input pipeline is the bottleneck and must be fixed
    (more workers, or repacking the crops) before spending the real time.
    """
    spec = ENCODERS[encoder]
    index = pd.read_parquet(INTERIM / dataset / "index.parquet").head(crops)
    model = spec.build(WEIGHTS / spec.local_dir).to(device).eval()

    loader = _loader(index, dataset, spec, batch_size, workers)
    start, seen = time.time(), 0
    with torch.inference_mode(), _UtilSampler() as util:
        for step, (x, _) in enumerate(loader):
            if step == 1:                      # exclude worker/CUDA warmup
                torch.cuda.synchronize()
                util.samples.clear()
                start, seen = time.time(), 0
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                spec.forward(model, x)
            seen += len(x)
        torch.cuda.synchronize()

    elapsed = time.time() - start
    result = {
        "encoder": encoder, "dataset": dataset, "crops": seen,
        "batch_size": batch_size, "num_workers": workers,
        "seconds": round(elapsed, 2),
        "images_per_second": round(seen / elapsed, 1),
        "gpu_util_mean": round(util.mean, 1),
        "gpu_util_min": float(np.min(util.samples)) if util.samples else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    print(f"[bench] {encoder}/{dataset}: {result['images_per_second']:.0f} img/s, "
          f"gpu {result['gpu_util_mean']:.0f}%, "
          f"{result['peak_vram_gb']:.1f} GB peak")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", nargs="+", default=list(ENCODERS),
                    choices=list(ENCODERS))
    ap.add_argument("--dataset", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--benchmark", type=int, default=0,
                    help="run N crops, report throughput and GPU util, write nothing")
    ap.add_argument("--out", default=None, help="benchmark results json")
    args = ap.parse_args()

    if args.benchmark:
        results = [benchmark(e, d, args.benchmark, args.batch_size, args.workers)
                   for e in args.encoder for d in args.dataset]
        out = ROOT / (args.out or "results/embed_benchmark.json")
        out.write_text(json.dumps(results, indent=1))
        print(f"[bench] wrote {out}")
        return

    for encoder in args.encoder:
        for dataset in args.dataset:
            run(encoder, dataset, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
