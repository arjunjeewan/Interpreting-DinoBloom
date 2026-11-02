# Where Does DinoBloom Read Cell Morphology?

Code, results and additional figures for *Where Does DinoBloom Read Cell Morphology? Linear
Readouts and an Exact Attention-Path Decomposition* (Arjun Jeewan, Department of Computer
Science, Indian Institute of Technology Bombay), submitted to ISBI 2027.

The paper fits a linear readout of each of eleven measured cytomorphometric descriptors on the
CLS embedding of DinoBloom-B, tests whether single embedding coordinates carry them, decomposes
each readout value exactly into per-token attention writes and the CLS token's MLP updates, and
tests whether readouts transfer across acquisition set-ups.

## Contents

| Path | Contents |
|---|---|
| `src/acquire/` | downloads and indexes the datasets; the only code that uses the network |
| `src/` | splits, U-Net segmentation, morphometry, embeddings |
| `src/analysis/` | the analyses behind the paper's numbers |
| `src/figures.py`, `figures/overview/` | figure code |
| `results/` | the summaries the paper's numbers are read from |
| `figures/`, `tables/` | the paper's figures and Table 1, and the additional figures below |
| `data/pretraining_status.csv` | which datasets and splits DinoBloom saw in pretraining |

## Setup

Python 3.11 with the versions pinned in `requirements.txt` (torch 2.6.0, CUDA 12.6):

```bash
conda create -n dinobloom-readouts python=3.11
conda activate dinobloom-readouts
pip install -r requirements.txt
```

The figures use the Nimbus Roman font (`fonts-urw-base35` on Debian and Ubuntu). The overview
figure is rendered with Inkscape.

## Data and weights

| | Source | Licence |
|---|---|---|
| Matek bone marrow | TCIA, [doi:10.7937/TCIA.AXH3-T579](https://doi.org/10.7937/TCIA.AXH3-T579), read from a HuggingFace mirror audited against the TCIA original | CC BY 4.0 |
| PBC | Mendeley Data, [doi:10.17632/snkd93bnjr.1](https://doi.org/10.17632/snkd93bnjr.1) | CC BY 4.0 |
| Raabin-WBC | [raabindata.com](https://raabindata.com) | "freely available for all" (dataset paper); no licence stated |
| DinoBloom-B weights | [`1aurent/vit_base_patch14_224.dinobloom`](https://huggingface.co/1aurent/vit_base_patch14_224.dinobloom), a timm port of the [official release](https://zenodo.org/records/10908163) | Apache 2.0 |

```bash
python -m src.acquire.matek_bm
python -m src.acquire.raabin
python -m src.acquire.pbc
python -m src.acquire.pretraining_status        # -> data/pretraining_status.csv
hf download 1aurent/vit_base_patch14_224.dinobloom --local-dir data/weights/timm_vit_base_patch14_224.dinobloom
```

Everything after this step runs offline.

## Reproducing the paper

Run from the repository root, in this order. `--read-test` scores the PBC test split and Raabin
test set B once; the scripts refuse it while `src/` has uncommitted changes.

```bash
# Labels: a U-Net per seed (25 seeds, film-disjoint splits), then the 11 descriptors for every cell
python -m src.splits
python -m src.masks
python -m src.segment --mode film --seeds $(seq 0 24)
for s in $(seq 0 24); do python -m src.propagate --seed $s; done
python -m src.morphometry

# DinoBloom-B CLS embeddings
python -m src.embed --encoder dinobloom --dataset matek_bm pbc raabin

# Matek (Secs. 3.1-3.3)
python -m src.analysis.linear_cv
python -m src.analysis.linear_cv_dims
python -m src.analysis.linear_cv_weights
python -m src.analysis.linear_cv_ablation
python -m src.analysis.linear_cv_within
python -m src.analysis.linear_cv_raabin
python -m src.analysis.linear_cv_gradcam
python -m src.analysis.linear_cv_attribution
python -m src.analysis.linear_cv_attnpath

# PBC and Raabin (Secs. 3.1, 3.3, 3.4)
python -m src.analysis.linear_unseen --dataset pbc --read-test
python -m src.analysis.linear_unseen --dataset raabin --read-test
python -m src.analysis.linear_unseen_ablation --read-test
python -m src.analysis.linear_unseen_attnpath --read-test

# Table 1 and figures
python -m src.analysis.linear_cv_table
python -m src.figures --only attnpath
python -m src.figures --only maps
python -m src.figures --only maps_all
python figures/overview/make_overview.py
inkscape figures/overview/overview.svg --export-type=pdf --export-filename=figures/overview.pdf
```

Comments and result files mention `protocol.md`, the written analysis plan of the working
repository; its section 16 fixed the PBC and Raabin benchmark before any test cell was read. It
is not part of this repository.

## Where each number comes from

All files are in `results/`.

| Paper | File |
|---|---|
| Sec. 2, U-Net Dice | `segmentation_dice_film.json` (seeds 0-2), `segmentation_dice_film_seeds3to24.json` |
| Table 1, Matek | `linear_cv_dinobloom_matek_bm.json`, `linear_cv_within_dinobloom_matek_bm.json` |
| Table 1, PBC and Raabin; label reliability | `linear_unseen_dinobloom_pbc.json`, `linear_unseen_dinobloom_raabin.json` |
| Sec. 3.1, cell type alone and within cell type | `linear_cv_within_dinobloom_matek_bm.json`, `linear_unseen_dinobloom_pbc.json` |
| Sec. 3.2, coordinate subsets | `linear_cv_ablation_dinobloom_matek_bm.json`, `linear_unseen_ablation_dinobloom_pbc_test.json` |
| Sec. 3.2, free direction | `linear_cv_weights_dinobloom_matek_bm.json` |
| Sec. 3.2, coordinates on Raabin | `linear_cv_raabin_dinobloom.json` |
| Sec. 3.3, saliency | `linear_cv_attribution_dinobloom_matek_bm.json` and `.parquet` |
| Sec. 3.3, decomposition | `linear_cv_attnpath_dinobloom_matek_bm.json`, `linear_unseen_attnpath_dinobloom_pbc.json` |
| Sec. 3.4 | `linear_unseen_dinobloom_raabin.json` |

## Additional results

The paper points here for the following.

### 1. The LayerNorm free direction (Sec. 3.2)

The embedding leaves a LayerNorm, z = γ ⊙ (h − h̄1)/σ + β, so every embedding satisfies
Σⱼ (zⱼ − βⱼ)/γⱼ = 0. The covariance of the embeddings is therefore singular along γ⁻¹, and a
readout direction w is defined only up to w + tγ⁻¹: adding any multiple of γ⁻¹ to w changes no
readout value. In `results/linear_cv_weights_dinobloom_matek_bm.json`:

- `null_direction`: in every fold, the null direction of the covariance is γ⁻¹
  (`abs_cos_with_inverse_ln_scale` = 1.0000), and 98.76% of its squared norm lies on
  coordinate 38 (`share_on_dropped`).
- `ln_scale_summary`: coordinate 38 has the smallest scale in the layer, γ₃₈ = −0.00712
  (`ln_scale_at_dropped`), against a median |γ| of 1.707.

Moving w along γ⁻¹ changes mainly the weight on coordinate 38 and leaves every readout value
unchanged, so per-coordinate weights are not identifiable.

### 2. Saliency maps against occlusion (Sec. 3.3)

- **Cells:** 60 Matek evaluation cells, 10 per granulocytic stage (myeloblast, promyelocyte,
  myelocyte, metamyelocyte, band and segmented neutrophil).
- **Maps**, per cell and readout:
  - signed Grad-CAM and gradient × activation on the patch tokens leaving each block's first
    LayerNorm, at each of the 12 blocks;
  - occlusion of each 14 × 14 patch by two fills: grey (0 after normalization) and the crop's
    background (the per-channel median of its border ring).
- **Score:** Pearson r between a map and the occlusion map, median over the 60 cells and the 11
  readouts. These are the rows of `results/linear_cv_attribution_dinobloom_matek_bm.parquet` with
  `kind == "readout"` and `set == "evaluation"`; blocks are 0-indexed there.

| Block | Grad-CAM, background fill | Grad-CAM, grey fill | Gradient × activation, background fill | Gradient × activation, grey fill |
|---:|---:|---:|---:|---:|
| 1 | −0.025 | −0.025 | −0.086 | −0.059 |
| 2 | 0.025 | −0.026 | −0.014 | −0.008 |
| 3 | −0.017 | −0.006 | −0.046 | −0.027 |
| 4 | −0.006 | −0.001 | 0.085 | 0.043 |
| 5 | 0.022 | 0.010 | 0.121 | 0.072 |
| 6 | 0.095 | 0.057 | 0.159 | 0.101 |
| 7 | 0.027 | 0.051 | 0.080 | 0.067 |
| 8 | 0.038 | 0.041 | 0.088 | 0.074 |
| 9 | 0.071 | 0.015 | 0.062 | 0.046 |
| 10 | 0.091 | 0.036 | 0.061 | 0.028 |
| 11 | 0.079 | 0.043 | 0.065 | 0.049 |
| 12 | 0.095 | 0.018 | 0.169 | 0.111 |

The occlusion maps with the two fills correlate at a median r of 0.480.

### 3. Region shares in all four settings (Sec. 3.3)

These are the readouts for which the region where the descriptor is measured carries the largest
share of the readout value's variance. For the N:C ratio, nucleus and cytoplasm must both exceed
the rest of the crop.

| | Pixel split | Interior patches only |
|---|---:|---:|
| Matek, all cells | 11 of 11 | 9 of 11 |
| Matek, within cell type | 9 of 11 | 7 of 11 |
| PBC test split, all cells | 9 of 11 | 1 of 11 |
| PBC test split, within cell type | 10 of 11 | 0 of 11 |

Seven readouts lead in all four Matek settings: cytoplasmic area, eccentricity, curvature
variance, GLCM contrast, homogeneity and entropy, and integrated optical density. On PBC the
nucleus covers 5.5% of the crop and 0.26% of the patches lie wholly inside a nucleus, so the
interior-only split is uninformative there.

`figures/linear_cv_attnpath_dinobloom.pdf` (all cells) and
`figures/linear_cv_attnpath_within_dinobloom.pdf` (within cell type) show the Matek shares:

- **Colour:** the share of each readout value's variance, on 2,000 cross-validation cells.
- **Rows:** readouts. **Columns:** patch-token writes by region, and the CLS token's MLP updates.
- **Left panel:** boundary patches split by pixel fraction. **Right panel:** interior patches
  only, with boundary patches as one column.
- **Dot:** the largest region in each row.
- **Not shown:** the CLS token's write to itself and the constant terms.

### 4. Patch maps for all 11 readouts (Fig. 2)

`figures/linear_unseen_maps_all_dinobloom.pdf` shows the cells of Fig. 2 for all 11 readouts, with
the same nucleus and cell outlines. Its
colour scale is set by this figure's own 99th percentile of |value| (0.09), so its colours are not
directly comparable with Fig. 2.

## Licence

Code: MIT (`LICENSE`). The datasets and weights keep their own licences (table above).
