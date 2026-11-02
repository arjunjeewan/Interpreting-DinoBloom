"""Attention-path decomposition of the linear_cv readouts: which blocks, tokens and image regions carry each readout's variation across cells.

Image information reaches the CLS token only through its own attention, once per
block. At block l the CLS's attention update is a sum over the 257 source tokens,

    sum_k a_lk * W_O W_V LN1(x_k) * ls1      (a_lk: the CLS's attention on token k, per head)

with the value bias carried by the attended token and the output bias counted as
a constant. The CLS's own MLP update per block is a further term, which cannot be
split by source token. With each cell's final-LayerNorm scale sigma held at its
actual value, the final LayerNorm is linear, so every additive term t of the CLS
residual stream maps onto the embedding exactly:

    z - beta = sum over terms t of  gamma * (t - mean_c t) / sigma

Each term satisfies the LayerNorm null constraint on its own (linear_cv_weights.py),
so its readout contribution does not depend on the null direction of the weights.

Readouts: each metric's joint_768 OLS readout (linear_cv_weights.py) of the cell's
out-of-fold model, Y = w'(z - beta). Cells: N_CELLS random labelled CV cells
(seed SEED); the 20% hold-out is not read.

Regions: linear_cv_regions.py's masks (the 25-seed majority vote at 224 px).
Patches tile the mask exactly (16 x 16 patches of 14 x 14 px), and each patch
token's write is split among nucleus / cytoplasm / rest by the fraction of its
196 pixels in each region. Side numbers, overlapping the regions: the writes of
boundary patches (pixels in more than one region) and of the TOP_NORM patches
with the largest residual norm at each block's input. As a check on the pixel
fraction split, a second partition keeps only interior patches (all 196 pixels
in one region) in their region and puts every boundary patch in a group of its
own. The cytoplasm, a thin ring, has few interior patches, so this split is
biased against it.

Statistic: a term group's share of a readout's variance across cells,
Cov(T_g, Y) / Var(Y). The shares of const, cls (the CLS reading itself), nucleus,
cytoplasm, rest and mlp sum to 1, and so do those of const, cls, the three
interior regions, boundary and mlp. Over all cells, and within cell type (every
term and Y centred on its type's mean over the sampled cells); summed over the 12
blocks, and for SHOWN_BLOCKS separately.

Checks: the per-token split against the model's own attention; the terms against
the CLS residual stream, the final LayerNorm and the readout; both region splits
against the patch sums; the cosine guard on the recomputed embeddings; and, as a
report, each readout's r^2 with its label on the sample beside linear_cv's
validation R^2.

fp32 throughout, no autocast. No network access.
"""

from __future__ import annotations

import argparse
import json
import platform

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.analysis.linear_cv import K_FOLDS, seed_mean_labels, splits_for
from src.analysis.linear_cv_gradcam import MIN_COS
from src.analysis.linear_cv_regions import REGIONS, region_masks
from src.analysis.raw_probe import DATASET, RESULTS, load_embeddings
from src.embed import ENCODERS, INTERIM, SIZE, WEIGHTS, CropDataset
from src.env import go_offline
from src.morphometry import DESCRIPTORS

go_offline()

N_CELLS = 2000
SEED = 0
TOP_NORM = 4          # high-norm patch tokens per block, for the side number
BATCH = 50
MASK_CHUNK = 100      # cells per region_masks call, which predicts all its cells in one batch
SHOWN_BLOCKS = [10, 11, 12]
RTOL_SPLIT = 1e-4     # per-token split vs the model's fused attention, relative to the largest write
RTOL_SUM = 1e-5       # terms vs CLS residual, final LN, readout and patch sums, relative to the largest value
PATCH = 14
GRID = SIZE // PATCH
INTERIOR = [f"{r}_interior" for r in REGIONS]
KINDS = ["cls", *REGIONS, "mlp", "boundary", "topnorm", *INTERIOR]   # per block; the first five partition attention + MLP
GROUPS = ["const", "cls", *REGIONS, "mlp"]                 # a complete partition: shares sum to 1
GROUPS_INTERIOR = ["const", "cls", *INTERIOR, "boundary", "mlp"]   # the second complete partition
MEASURED_IN = {d: "nucleus" for d in DESCRIPTORS} | {"cytoplasmic_area": "cytoplasm", "nc_ratio": "both"}


def patch_fractions(mask: np.ndarray) -> np.ndarray:
    """Per patch in token order, the fraction of its pixels in each region (rows in REGIONS order)."""
    return np.stack([(mask == label).reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3)).ravel()
                     for label in REGIONS.values()])


def tau(t: torch.Tensor, sigma: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """A residual-stream term's exact share of z - beta, with the cell's LayerNorm scale fixed."""
    return gamma * (t - t.mean(-1, keepdim=True)) / sigma


def shares(T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cov(T_g, Y) / Var(Y), for terms T (cells x ... x readouts) and readouts Y (cells x readouts)."""
    Tc, Yc = T - T.mean(0), Y - Y.mean(0)
    Yb = Yc.reshape(len(Yc), *[1] * (T.ndim - 2), Yc.shape[1])
    return (Tc * Yb).sum(0) / (Yc ** 2).sum(0)


def centre(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean, row-wise."""
    f = pd.DataFrame(a.reshape(len(a), -1))
    return (f - f.groupby(groups).transform("mean")).to_numpy().reshape(a.shape)


def batch_terms(model, img, Wb, frac, boundary):
    """One batch: readout terms per block and kind, constants, readouts, embeddings, token-norm stats, errors,
    and each patch token's direct-attention write to each readout summed over the blocks (B, patches, R)."""
    gamma, beta, eps = model.norm.weight, model.norm.bias, model.norm.eps
    x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(img))))
    B, N, C = x.shape
    const = x[:, 0].clone()
    toks, mlps, tops, stats, err = [], [], [], [], {"split": 0.0, "regions": 0.0, "interior": 0.0}
    for blk in model.blocks:
        h = blk.norm1(x)
        a_out = blk.attn(h)                                   # the model's own (fused) attention
        g1, g2 = blk.ls1.gamma, blk.ls2.gamma
        x_mid = x + a_out * g1
        m_out = blk.mlp(blk.norm2(x_mid)) * g2
        at = blk.attn                                         # the CLS row again, split by source token
        q, k, v = at.qkv(h).reshape(B, N, 3, at.num_heads, at.head_dim).permute(2, 0, 3, 1, 4)
        a = ((q[:, :, :1] * at.scale) @ k.transpose(-2, -1)).softmax(-1)[:, :, 0]       # (B, heads, N)
        tok = ((a[..., None] * v).permute(0, 2, 1, 3).reshape(B, N, C) @ at.proj.weight.T) * g1
        bias = at.proj.bias * g1
        write = a_out[:, 0] * g1
        err["split"] = max(err["split"], ((tok.sum(1) + bias - write).abs().max() / write.abs().max()).item())
        nrm = x[:, 1:].norm(dim=-1)                           # patch norms at this block's input
        srt, med = nrm.sort(dim=1, descending=True).values, nrm.median(dim=1).values
        top = torch.zeros_like(nrm, dtype=torch.bool).scatter_(1, nrm.topk(TOP_NORM, dim=1).indices, True)
        stats.append(torch.stack([srt[:, 0] / med, srt[:, TOP_NORM - 1] / med,
                                  (a[:, :, 1:] * top[:, None]).sum(-1).mean(1), a[:, :, 0].mean(1)], 1))
        toks.append(tok)
        mlps.append(m_out[:, 0])
        tops.append(top)
        const = const + bias
        x = x_mid + m_out
    xf = x[:, 0]
    recon = const + sum(t.sum(1) for t in toks) + sum(mlps)
    sigma = torch.sqrt(xf.var(-1, unbiased=False) + eps)[:, None]
    z = tau(xf, sigma, gamma) + beta
    z_model = model.norm(x)[:, 0]
    Y = torch.einsum("bc,bcr->br", z - beta, Wb)
    terms, patches = [], 0
    for tok, mlp, top in zip(toks, mlps, tops):
        p = torch.einsum("bnc,bcr->bnr", tau(tok, sigma[:, :, None], gamma), Wb)      # (B, N, R)
        patches = patches + p[:, 1:]
        regions = torch.einsum("bgk,bkr->gbr", frac, p[:, 1:])                        # (regions, B, R)
        inner = torch.einsum("bgk,bkr->gbr", (frac == 1).float(), p[:, 1:])           # interior patches only
        edge = (p[:, 1:] * boundary[..., None]).sum(1)
        total = p[:, 1:].sum(1)
        err["regions"] = max(err["regions"], ((regions.sum(0) - total).abs().max() / total.abs().max()).item())
        err["interior"] = max(err["interior"], ((inner.sum(0) + edge - total).abs().max() / total.abs().max()).item())
        terms.append(torch.stack([p[:, 0], *regions, torch.einsum("bc,bcr->br", tau(mlp, sigma, gamma), Wb),
                                  edge, (p[:, 1:] * top[..., None]).sum(1), *inner], 1))
    T = torch.stack(terms, 1)                                                          # (B, blocks, KINDS, R)
    c = torch.einsum("bc,bcr->br", tau(const, sigma, gamma), Wb)
    err["cls_sum"] = ((recon - xf).abs().max() / xf.abs().max()).item()
    err["final_ln"] = ((z - z_model).abs().max() / z_model.abs().max()).item()
    err["readout"] = ((T[:, :, :5].sum((1, 2)) + c - Y).abs().max() / Y.abs().max()).item()
    return T, c, Y, z, torch.stack(stats, 1), err, patches


def table(full: np.ndarray, side: np.ndarray) -> pd.DataFrame:
    """Readouts x groups: the partition's shares, then the two side numbers."""
    return pd.DataFrame(np.vstack([full, side]).T, index=DESCRIPTORS, columns=GROUPS + ["boundary", "topnorm"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinobloom")
    args = ap.parse_args()
    device = torch.device("cuda")
    R, L = len(DESCRIPTORS), 12

    meta, X = load_embeddings(args.encoder)
    meta = meta.reset_index(drop=True)
    splits = splits_for(meta)
    labels = seed_mean_labels(meta)
    cv = splits[splits.split == "cv"]
    sample = cv[cv.cell_id.isin(labels.index)].sample(N_CELLS, random_state=SEED).reset_index(drop=True)
    fold, cell_type = sample.cv_fold.to_numpy().copy(), sample.class_label.to_numpy()
    y_label = labels.loc[sample.cell_id, DESCRIPTORS].to_numpy(np.float64)
    stored = X[meta.set_index("cell_id").loc[sample.cell_id, "row"].to_numpy()].astype(np.float64)

    wt = pd.read_parquet(RESULTS / f"linear_cv_weights_{args.encoder}_{DATASET}.parquet")
    wt = wt[wt.model == "joint_768"].set_index(["fold", "dim", "descriptor"]).w.unstack("descriptor")
    W = np.stack([wt.loc[k].sort_index()[DESCRIPTORS].to_numpy() for k in range(K_FOLDS)])   # (folds, C, R)

    spec = ENCODERS[args.encoder]
    index = pd.read_parquet(INTERIM / DATASET / "index.parquet").set_index("cell_id").loc[sample.cell_id].reset_index()
    frac = np.zeros((N_CELLS, len(REGIONS), GRID * GRID))
    for s in range(0, N_CELLS, MASK_CHUNK):
        chunk = CropDataset(index.iloc[s:s + MASK_CHUNK], DATASET, spec.interpolation)
        for i, m in enumerate(region_masks(chunk)):
            frac[s + i] = patch_fractions(m)
        print(f"[linear_cv_attnpath] masks {min(s + MASK_CHUNK, N_CELLS)}/{N_CELLS}", flush=True)
    boundary = frac.max(axis=1) < 1

    model = spec.build(WEIGHTS / spec.local_dir).to(device).eval()
    Wt = torch.tensor(W, dtype=torch.float32, device=device)
    loader = DataLoader(CropDataset(index, DATASET, spec.interpolation), batch_size=BATCH, num_workers=4)
    T, const, Y = np.zeros((N_CELLS, L, len(KINDS), R)), np.zeros((N_CELLS, R)), np.zeros((N_CELLS, R))
    norm_stats, cos, err = np.zeros((N_CELLS, L, 4)), np.zeros(N_CELLS), {}
    i0 = 0
    with torch.no_grad():
        for img, ok in loader:
            if not ok.all():
                raise RuntimeError("a crop did not decode")
            sl = slice(i0, i0 + len(img))
            out = batch_terms(model, img.to(device), Wt[torch.tensor(fold[sl], device=device)],
                              torch.tensor(frac[sl], dtype=torch.float32, device=device),
                              torch.tensor(boundary[sl], device=device))
            T[sl], const[sl], Y[sl], z, norm_stats[sl] = (o.double().cpu().numpy() for o in out[:5])
            cos[sl] = (z * stored[sl]).sum(1) / np.linalg.norm(z, axis=1) / np.linalg.norm(stored[sl], axis=1)
            err = {key: max(err.get(key, 0.0), v) for key, v in out[5].items()}
            i0 = sl.stop
            print(f"[linear_cv_attnpath] {i0}/{N_CELLS} cells", flush=True)

    if err["split"] > RTOL_SPLIT or max(v for key, v in err.items() if key != "split") > RTOL_SUM:
        raise RuntimeError(f"decomposition check failed: {err}")
    if cos.min() < MIN_COS:
        raise RuntimeError(f"recomputed embedding differs from the stored one (cos {cos.min():.4f})")

    full = np.concatenate([const[:, None], T[:, :, :5].sum(1)], 1)      # (cells, GROUPS, R)
    side = T[:, :, 5:7].sum(1)                                            # (cells, boundary/topnorm, R)
    inner = np.concatenate([const[:, None], T[:, :, [KINDS.index(k) for k in GROUPS_INTERIOR[1:]]].sum(1)], 1)
    shown = T[:, [b - 1 for b in SHOWN_BLOCKS], :5]                       # (cells, shown blocks, cls..mlp, R)
    per_block = np.stack([T[:, :, :4].sum(2), T[:, :, 4]], 2)             # (cells, blocks, attention/mlp, R)
    scopes = {"all_cells": (full, side, inner, shown, per_block, Y)}
    scopes["within_type"] = tuple(centre(a, cell_type) for a in scopes["all_cells"])
    result = {}
    for scope, (f, s, i, b, p, y) in scopes.items():
        sb, sp = shares(b, y), shares(p, y)
        result[scope] = {
            "shares_all_blocks": table(shares(f, y), shares(s, y)).to_dict(orient="index"),
            "shares_all_blocks_interior_split": pd.DataFrame(shares(i, y).T, index=DESCRIPTORS,
                                                             columns=GROUPS_INTERIOR).to_dict(orient="index"),
            "shares_by_shown_block": {f"block {blk}": pd.DataFrame(sb[n].T, index=DESCRIPTORS, columns=KINDS[:5])
                                      .to_dict(orient="index") for n, blk in enumerate(SHOWN_BLOCKS)},
            "attention_and_mlp_by_block": {kind: pd.DataFrame(sp[:, n].T, index=DESCRIPTORS,
                                                              columns=[f"block {l + 1}" for l in range(L)])
                                           .to_dict(orient="index") for n, kind in enumerate(["attention", "mlp"])}}
    within_frac = (centre(Y, cell_type) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    ref = json.loads((RESULTS / f"linear_cv_dims_{args.encoder}_{DATASET}.json").read_text())["descriptors"]
    r2 = {d: {"sample_r2_with_label": float(np.corrcoef(Y[:, i], y_label[:, i])[0, 1] ** 2),
              "linear_cv_val_r2": ref[d]["r2_val_all_dims"]["mean"]} for i, d in enumerate(DESCRIPTORS)}

    cols = {"cell_id": np.repeat(sample.cell_id.to_numpy(), R), "class_label": np.repeat(cell_type, R),
            "fold": np.repeat(fold, R), "descriptor": np.tile(DESCRIPTORS, N_CELLS),
            "y": Y.ravel(), "const": const.ravel()}
    for l in range(L):
        for g, kind in enumerate(KINDS):
            cols[f"{kind}_{l + 1}"] = T[:, l, g].ravel()
    for g, region in enumerate(REGIONS):
        cols[f"area_{region}"] = np.repeat(frac[:, g].mean(1), R)
        cols[f"area_{region}_interior"] = np.repeat((frac[:, g] == 1).mean(1), R)
    cols["area_boundary"] = np.repeat(boundary.mean(1), R)

    summary = {
        "analysis": "attention-path decomposition of the joint_768 readouts by block, source token and region",
        "encoder": args.encoder, "dataset": DATASET, "test_read": False,
        "cells": f"{N_CELLS} random labelled CV cells (pandas sample, random_state={SEED}); each read by its "
                 "out-of-fold joint_768 readout",
        "class_counts": {c: int(n) for c, n in pd.Series(cell_type).value_counts().items()},
        "regions": "linear_cv_regions.region_masks; patch writes split by pixel fraction, and as a check "
                   "by interior patches only, with boundary patches as their own group",
        "measured_in": MEASURED_IN,
        "statistic": "Cov(T_g, Y) / Var(Y) over cells, Y = w'(z - beta); const, cls, nucleus, cytoplasm, rest "
                     "and mlp sum to 1, as do const, cls, the interior regions, boundary and mlp; in "
                     "shares_all_blocks, boundary and topnorm are side numbers inside the regions",
        "within_type_fraction_of_variance": dict(zip(DESCRIPTORS, within_frac.tolist())),
        "mean_area": {**{r: float(frac[:, g].mean()) for g, r in enumerate(REGIONS)},
                      **{f"{r}_interior": float((frac[:, g] == 1).mean()) for g, r in enumerate(REGIONS)},
                      "boundary_patches": float(boundary.mean())},
        **result,
        "token_norms_by_block_median": pd.DataFrame(
            np.median(norm_stats, 0), index=[f"block {l + 1}" for l in range(L)],
            columns=["max_over_median", f"top{TOP_NORM}th_over_median", f"cls_attention_on_top{TOP_NORM}",
                     "cls_self_attention"]).to_dict(orient="index"),
        "checks": {**{f"max_rel_err_{key}": v for key, v in err.items()},
                   "embedding_cos_min": float(cos.min()), "readout_r2": r2},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__},
        "device": torch.cuda.get_device_name(0), "host": platform.node(),
    }
    stem = f"linear_cv_attnpath_{args.encoder}_{DATASET}"
    pd.DataFrame(cols).to_parquet(RESULTS / f"{stem}.parquet", index=False)
    (RESULTS / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    for scope in scopes:
        print(f"[linear_cv_attnpath] {scope}: variance shares over all blocks")
        print(pd.DataFrame(result[scope]["shares_all_blocks"]).T.assign(measured_in=pd.Series(MEASURED_IN))
              .round(3).to_string())
        print(f"[linear_cv_attnpath] {scope}: interior-only split, boundary patches as their own group")
        print(pd.DataFrame(result[scope]["shares_all_blocks_interior_split"]).T.round(3).to_string())
    print(f"[linear_cv_attnpath] checks: "
          f"{json.dumps({k: v for k, v in summary['checks'].items() if k != 'readout_r2'})}")
    print(f"[linear_cv_attnpath] wrote {RESULTS / stem}.parquet / .json")


if __name__ == "__main__":
    main()
