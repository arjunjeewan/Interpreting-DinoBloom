"""Nucleus/cytoplasm segmentation: train on Raabin GT, propagate everywhere else.

Trains a small U-Net from scratch on the 927 film-disjoint Raabin ground-truth
cells (3 classes: background / nucleus / cytoplasm) and applies it to the other
hematology datasets so morphometry can be computed on cells that have no mask.

Two properties of the target data drive the design:

* **Apparent cell scale differs a lot between datasets.** Measured median
  cell-area fraction of frame: Raabin 0.157 (ground truth), Matek BM ~0.37,
  PBC ~0.074 -- roughly a 0.7x to 1.5x linear spread. Training only at Raabin's
  scale would not transfer, so scale jitter spans 0.5x to 2.0x, well outside the
  observed range.
* **Matek BM is bone marrow and its 250x250 crops are crowded**, so neighbouring
  cells are usually in frame. The Raabin ground truth annotates only the centred
  cell, which teaches the right bias, but at inference the prediction is
  additionally reduced to the connected component containing the image centre
  (`_centre_component`). Without that, morphometry would silently sum two cells.

Stain differs too (Raabin Giemsa vs Matek May-Grunwald-Giemsa/Pappenheim), hence
the colour jitter.

Selection on validation, test read once (guardrail 1). 3 seeds, mean +/- sd.
No network access.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from src.env import ROOT, go_offline, pin_numerics

go_offline()
pin_numerics()

RAABIN = ROOT / "data" / "interim" / "raabin"
CKPT = ROOT / "data" / "checkpoints" / "segment"

SIZE = 256
N_CLASSES = 3          # 0 background, 1 nucleus, 2 cytoplasm
CLASS_NAMES = ["background", "nucleus", "cytoplasm"]
EPOCHS = 80
BATCH = 8
LR = 3e-4
SCALE_RANGE = (0.5, 2.0)


# ------------------------------------------------------------------- model ---
def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """4-level U-Net, ~2M params. Small on purpose: 645 training images."""

    def __init__(self, base: int = 32, n_classes: int = N_CLASSES):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([_block(3, c[0]), _block(c[0], c[1]),
                                  _block(c[1], c[2]), _block(c[2], c[3])])
        self.bottleneck = _block(c[3], c[3] * 2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1] if i < 3 else c[3] * 2,
                                                    c[i], 2, stride=2)
                                 for i in range(4)])
        self.dec = nn.ModuleList([_block(c[i] * 2, c[i]) for i in range(4)])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for i in range(3, -1, -1):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], dim=1))
        return self.head(x)


# ------------------------------------------------------------------- data ----
class GTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        rec = self.df.iloc[i]
        img = Image.open(RAABIN / rec.image_path).convert("RGB")
        mask = Image.open(ROOT / rec.mask_path)

        if self.train:
            img, mask = self._augment(img, mask)
        img = img.resize((SIZE, SIZE), Image.BILINEAR)
        mask = mask.resize((SIZE, SIZE), Image.NEAREST)

        x = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0)
        y = torch.from_numpy(np.asarray(mask, np.int64))
        return x, y

    def _augment(self, img: Image.Image, mask: Image.Image):
        rng = self.rng
        if rng.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Scale + rotate about the centre. Scale is the important one: it is what
        # lets a Raabin-trained model read a tightly-cropped Matek marrow cell.
        scale = float(rng.uniform(*SCALE_RANGE))
        angle = float(rng.uniform(0, 360))
        w, h = img.size
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
        side = int(round(w / scale))
        if side < w:                                    # zoom in: centre crop
            off = (w - side) // 2
            img = img.crop((off, off, off + side, off + side))
            mask = mask.crop((off, off, off + side, off + side))
        elif side > w:                                  # zoom out: pad
            pad = (side - w) // 2
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(img, (pad, pad))
            mcanvas = Image.new("L", (side, side), 0)
            mcanvas.paste(mask, (pad, pad))
            img, mask = canvas, mcanvas

        # Stain/exposure jitter: Raabin is Giemsa, Matek is MGG/Pappenheim.
        a = np.asarray(img, np.float32)
        a = a * rng.uniform(0.85, 1.15, size=3) + rng.uniform(-18, 18, size=3)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return img, mask


# ------------------------------------------------------------------ train ----
def dice_per_class(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Dice per class over a batch, averaged across images that contain the class."""
    out = np.full(N_CLASSES, np.nan)
    for c in range(N_CLASSES):
        p, t = (pred == c), (target == c)
        inter = (p & t).flatten(1).sum(1).float()
        denom = p.flatten(1).sum(1).float() + t.flatten(1).sum(1).float()
        present = t.flatten(1).any(1)
        if present.any():
            out[c] = ((2 * inter[present] + 1e-6) / (denom[present] + 1e-6)).mean().item()
    return out


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = logits.softmax(1)
    onehot = F.one_hot(target, N_CLASSES).permute(0, 3, 1, 2).float()
    inter = (prob * onehot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + 1e-6) / (denom + 1e-6)).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        scores.append(dice_per_class(pred, y))
    return np.nanmean(np.stack(scores), axis=0)


def assign_folds(gt: pd.DataFrame, mode: str, seed: int) -> pd.Series:
    """Fold assignment under one of three split regimes.

    Only `film` is used for the reported model. The other two exist to measure
    what film leakage is actually worth, rather than asserting it:

    film    the committed split, film-disjoint (src/splits.py).
    authors the Raabin authors' own GT train/test division. 15 of its 54 films
            straddle it. It has no val fold, so one is carved at random from
            their train -- which is itself leaky, matching how it would be used
            in practice.
    random  cell-level random assignment at the SAME fold sizes as `film`. Every
            film straddles, so this is an upper bound on the leakage effect, not
            an estimate of the authors' split. It is the positive control that
            separates "leakage matters" from "fold sizes differ".
    """
    rng = np.random.default_rng(seed)
    if mode == "film":
        folds = pd.read_parquet(RAABIN / "gt_splits.parquet")
        return gt.cell_id.map(folds.set_index("cell_id").fold)

    if mode == "authors":
        fold = gt.gt_split.replace({"test": "test"}).copy()
        train_idx = np.flatnonzero((fold == "train").to_numpy())
        val_n = int(round(len(train_idx) * 0.15))
        val_idx = rng.permutation(train_idx)[:val_n]
        fold.iloc[val_idx] = "val"
        return fold

    if mode == "random":
        sizes = (pd.read_parquet(RAABIN / "gt_splits.parquet")
                 .fold.value_counts().to_dict())
        order = rng.permutation(len(gt))
        fold = pd.Series(index=gt.index, dtype=object)
        start = 0
        for name in ["train", "val", "test"]:
            take = order[start:start + sizes[name]]
            fold.iloc[take] = name
            start += sizes[name]
        return fold

    raise ValueError(f"unknown split mode {mode!r}")


def train_one(seed: int, device: str = "cuda", mode: str = "film") -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    gt = pd.read_parquet(RAABIN / "gt_masks.parquet")
    gt = gt[gt.has_nucleus_gt].reset_index(drop=True)
    gt["fold"] = assign_folds(gt, mode, seed).to_numpy()

    loaders = {}
    for fold in ["train", "val", "test"]:
        sub = gt[gt.fold == fold]
        loaders[fold] = DataLoader(
            GTDataset(sub, train=(fold == "train"), seed=seed),
            batch_size=BATCH, shuffle=(fold == "train"),
            num_workers=4, pin_memory=True, persistent_workers=True,
            drop_last=(fold == "train"))

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    CKPT.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch = -np.inf, -1
    # The reported model keeps the plain name so load_model/propagate are
    # unaffected; the leakage-experiment arms are namespaced.
    suffix = "" if mode == "film" else f"_{mode}"
    ckpt_path = CKPT / f"unet{suffix}_seed{seed}.pt"

    for epoch in range(EPOCHS):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + soft_dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        # Selection on validation only, on the mean of the two cell compartments.
        val = evaluate(model, loaders["val"], device)
        score = float(np.nanmean(val[1:]))
        if score > best_val:
            best_val, best_epoch = score, epoch
            torch.save(model.state_dict(), ckpt_path)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  [{mode}] seed {seed} epoch {epoch:3d} loss {loss.item():.4f} "
                  f"val nuc {val[1]:.4f} cyt {val[2]:.4f}")

    # Test is read exactly once, with the validation-selected checkpoint.
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test = evaluate(model, loaders["test"], device)
    val = evaluate(model, loaders["val"], device)
    print(f"  [{mode}] seed {seed}: selected epoch {best_epoch}, "
          f"test nucleus {test[1]:.4f} cytoplasm {test[2]:.4f}", flush=True)
    return {
        "seed": seed, "mode": mode, "selected_epoch": best_epoch,
        "n_train": int((gt.fold == "train").sum()),
        "n_val": int((gt.fold == "val").sum()),
        "n_test": int((gt.fold == "test").sum()),
        "test_films_also_in_train": int(
            gt[gt.fold == "test"].film_id.isin(gt[gt.fold == "train"].film_id).sum()),
        "val_dice": {n: float(v) for n, v in zip(CLASS_NAMES, val)},
        "test_dice": {n: float(v) for n, v in zip(CLASS_NAMES, test)},
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
    }


# --------------------------------------------------------------- inference ---
def _centre_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the cell touching the image centre.

    Matek BM crops are crowded marrow: without this, a neighbouring cell in
    frame is summed into the centred cell's morphometry.
    """
    cell = mask > 0
    if not cell.any():
        return mask
    labels, n = ndimage.label(cell)
    if n <= 1:
        return mask
    cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    keep = labels[cy, cx]
    if keep == 0:
        # Centre landed on background: fall back to the component nearest it.
        sizes = ndimage.sum(cell, labels, range(1, n + 1))
        coords = ndimage.center_of_mass(cell, labels, range(1, n + 1))
        d = [((y - cy) ** 2 + (x - cx) ** 2) / max(s, 1) for (y, x), s in zip(coords, sizes)]
        keep = int(np.argmin(d)) + 1
    return np.where(labels == keep, mask, 0)


@torch.no_grad()
def predict(model: nn.Module, images: list[Image.Image], device: str = "cuda",
            out_size: int | None = None) -> list[np.ndarray]:
    """Predict label maps. fp16 autocast (Volta: never bfloat16)."""
    model.eval()
    x = torch.stack([
        torch.from_numpy(np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR),
                                    np.float32).transpose(2, 0, 1) / 255.0)
        for im in images]).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
    logits = logits.float()
    out = []
    for i, im in enumerate(images):
        size = (out_size, out_size) if out_size else im.size[::-1]
        up = F.interpolate(logits[i:i + 1], size=size, mode="bilinear", align_corners=False)
        out.append(_centre_component(up.argmax(1)[0].cpu().numpy().astype(np.uint8)))
    return out


def load_model(seed: int = 0, device: str = "cuda") -> nn.Module:
    model = UNet().to(device)
    model.load_state_dict(torch.load(CKPT / f"unet_seed{seed}.pt", weights_only=True))
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--mode", default="film", choices=["film", "authors", "random"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [train_one(s, mode=args.mode) for s in args.seeds]
    agg = {}
    for split in ["val", "test"]:
        for name in CLASS_NAMES:
            vals = [r[f"{split}_dice"][name] for r in runs]
            agg[f"{split}_{name}"] = {"mean": float(np.mean(vals)),
                                      "sd": float(np.std(vals, ddof=1))}
    result = {"n_seeds": len(runs), "mode": args.mode, "runs": runs,
              "aggregate": agg,
              "selection_rule": "max val mean(nucleus, cytoplasm) Dice; test read once"}
    out = ROOT / "results" / (args.out or f"segmentation_dice_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("\nHeld-out Dice, mean +/- sd over "
          f"{len(runs)} seeds (film-disjoint test split):")
    for name in CLASS_NAMES:
        a = agg[f"test_{name}"]
        print(f"  {name:12s} {a['mean']:.4f} +/- {a['sd']:.4f}")
    print(f"[segment] wrote {out}")


if __name__ == "__main__":
    main()
