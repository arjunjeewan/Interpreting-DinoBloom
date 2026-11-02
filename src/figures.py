"""Figures. Vector PDF, colorblind-safe palette.

The raw-embedding linear study (src/analysis/linear_cv*.py): per-position MAE
and R^2 (dims), weights (weights), the top/bottom/random-K ablation (ablation),
the metric x position weight heatmap (heatmap), region occlusion (regions) and
the attention-path region shares (attnpath), and its per-readout patch maps on
PBC (maps, src/analysis/linear_unseen_attnpath.py); plus the morphometry validity
check (morphometry). The SAE-phase figures were
moved to archive/sae/src/figures.py on 2026-10-22.

The linear-study figures carry no text beyond panel names, ticks and axis
labels; their explanation belongs in the caption.

Palette: categorical slots 1 and 2 of the reference palette, validated all-pairs
in light mode (worst CVD deltaE 24.7 protan, normal-vision 33.6, contrast >= 3:1).

No network access.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator, NullLocator, StrMethodFormatter  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402

from src.analysis.linear_cv_weights import LN_SCALE  # noqa: E402
from src.env import ROOT, go_offline  # noqa: E402

go_offline()

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b5b3ad"
SERIES = {"topk": "#2a78d6", "batchtopk": "#eb6834"}
TEXT = "#000000"  # all text in the paper figures

# Nimbus Roman: the Times face of the paper's body text (spconf.sty).
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"]})


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=MUTED, linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)


# ------------------------------------------- figures for the clinical readme ---
# protocol.md section 5, "morphometry validity check". The direction is a
# grounded clinical prediction and is retained; the magnitude bar that used to
# sit alongside it (Gate A) was removed on 2026-08-05, see protocol.md section 15.
VALIDITY_CHECK = [("nc_ratio", "N:C ratio", "higher in blasts"),
                  ("glcm_contrast", "Chromatin contrast (GLCM)", "lower in blasts"),
                  ("glcm_homogeneity", "Chromatin homogeneity (GLCM)",
                   "higher in blasts")]


def morphometry_effect_sizes(seed: int = 0) -> None:
    """Do the measured descriptors separate blasts from segmented neutrophils?

    The question a pathologist would ask first: before any claim about learned
    features, are the descriptors themselves measuring what their names say?
    """
    df = pd.read_parquet(RESULTS / f"morphometry_matek_bm_seed{seed}.parquet")
    df = df[df.qc_pass & df.class_label.isin(["BLA", "NGS"])]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.28)
    for ax, (col, title, expect) in zip(axes, VALIDITY_CHECK):
        _style(ax)
        # Trim the extreme half-percent rather than clipping it: clipping piles
        # the tail onto the end bin and draws a spike that is not in the data.
        lo_q = df[col].quantile(0.005)
        hi_q = df[col].quantile(0.995)
        for label, colour in (("BLA", SERIES["batchtopk"]), ("NGS", SERIES["topk"])):
            values = df.loc[df.class_label == label, col].dropna()
            n_total = len(values)
            values = values[(values >= lo_q) & (values <= hi_q)]
            ax.hist(values, bins=60, range=(lo_q, hi_q), color=colour, alpha=0.55,
                    density=True, label=f"{label} (n={n_total:,})", zorder=3)
        ax.set_title(f"{title}\n{expect}", fontsize=9, color=INK, loc="left", pad=6)
        ax.set_yticks([])
        ax.tick_params(labelsize=7.5)
    axes[0].legend(fontsize=7.5, frameon=True, facecolor=SURFACE,
                   edgecolor=MUTED, labelcolor=INK_2).get_frame().set_linewidth(0.6)

    fig.suptitle("Morphometry validity check — do the measured descriptors "
                 "separate blasts from segmented neutrophils?", fontsize=11.5,
                 color=INK, x=0.045, ha="left", y=0.99)
    fig.text(0.045, 0.90, "Matek bone marrow, propagated segmentation masks, "
             f"QC-passing cells, segmenter seed {seed}", fontsize=8, color=INK_2)
    fig.subplots_adjust(top=0.72)
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "morphometry_blast_vs_neutrophil.pdf"
    fig.savefig(out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ------------------------------------------------------------- descriptors ---
# protocol.md section 4, in the order the descriptor table lists them.
DESCRIPTORS = ["nuclear_area", "cytoplasmic_area", "nc_ratio", "nuclear_solidity",
               "form_factor", "eccentricity", "curvature_variance",
               "glcm_contrast", "glcm_homogeneity", "glcm_entropy",
               "integrated_optical_density"]
DESCRIPTOR_LABEL = {"nuclear_area": "nuclear area",
                    "cytoplasmic_area": "cytoplasmic area",
                    "nc_ratio": "N:C ratio",
                    "nuclear_solidity": "nuclear solidity",
                    "form_factor": "form factor",
                    "eccentricity": "eccentricity",
                    "curvature_variance": "curvature variance",
                    "glcm_contrast": "GLCM contrast",
                    "glcm_homogeneity": "GLCM homogeneity",
                    "glcm_entropy": "GLCM entropy",
                    "integrated_optical_density": "integrated optical density"}

# Diverging: signed values in [-1, 1] (the weight heatmap), so two hues with a
# neutral midpoint. Poles are the two categorical slots already validated for
# this project.
DIVERGING = LinearSegmentedColormap.from_list(
    "assoc", [SERIES["batchtopk"], "#f4b699", "#eeece7", "#9dc0ea", SERIES["topk"]])


# Direction along the series is a polarity, so the two poles of the same
# diverging pair carry it: blue rises with maturation, orange falls.
UP, DOWN = SERIES["topk"], SERIES["batchtopk"]


DEVICE_C, MORPH_C = SERIES["batchtopk"], SERIES["topk"]


# ------------------------------------------- raw-embedding phase (linear_cv) ---
def linear_cv_dims(encoder: str = "dinobloom", score: str = "mae") -> None:
    """Validation MAE or R^2 of y ~ a + b * E[:, j] at every embedding position j.

    One panel per descriptor (src/analysis/linear_cv_dims.py), between two
    references from linear_cv.py: the training-mean baseline (light; R^2 = 0 by
    definition) and OLS on all positions (dark). No text beyond panel names and
    axes; the explanation belongs in the caption.
    """
    stem = f"linear_cv_dims_{encoder}_matek_bm"
    rows = pd.read_parquet(RESULTS / f"{stem}.parquet")
    summary = json.loads((RESULTS / f"{stem}.json").read_text())
    curves = rows[rows.split == "val"].groupby(["descriptor", "dim"])[score].agg(["mean", "std"])
    n_dims = summary["n_dims"]
    colour = SERIES["topk"]

    fig, axes = plt.subplots(3, 4, figsize=(11.5, 7.2), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    for ax, d in zip(axes.flat, DESCRIPTORS):
        s, c = summary["descriptors"][d], curves.loc[d]
        _style(ax)
        x = c.index.to_numpy()
        ax.fill_between(x, c["mean"] - c["std"], c["mean"] + c["std"], color=colour,
                        alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, c["mean"], color=colour, linewidth=0.6, zorder=3)
        base, full = ((s["mae_val_baseline"]["mean"], s["mae_val_all_dims"]["mean"]) if score == "mae"
                      else (0.0, s["r2_val_all_dims"]["mean"]))
        ax.axhline(base, color=MUTED, linewidth=1.2, zorder=1)
        ax.axhline(full, color=INK_2, linewidth=1.2, zorder=1)

        span = abs(base - full)
        ax.set_ylim(min(base, full, c["mean"].min()) - 0.08 * span,
                    max(base, full, c["mean"].max()) + 0.08 * span)
        ax.set_xlim(-8, n_dims + 7)
        ax.set_xticks([0, 256, 512, n_dims - 1] if n_dims == 768 else [0, n_dims - 1])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if base > 100:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        else:
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_title(DESCRIPTOR_LABEL[d], fontsize=9, color=INK, loc="left", pad=6)
    for ax in axes.flat[len(DESCRIPTORS):]:
        ax.axis("off")

    fig.supxlabel("embedding position", fontsize=9, color=INK_2, y=0.04)
    fig.supylabel("MAE" if score == "mae" else "R²", fontsize=9, color=INK_2, x=0.06)
    fig.subplots_adjust(left=0.1)

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / (f"linear_cv_dims_{encoder}.pdf" if score == "mae" else f"linear_cv_dims_r2_{encoder}.pdf")
    fig.savefig(out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, facecolor=SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def linear_cv_weights(encoder: str = "dinobloom") -> None:
    """Per-position weights of the linear models: one figure per model and scale.

    Models (src/analysis/linear_cv_weights.py): joint_768, the all-position OLS
    as sklearn returns it; joint_767, refitted without the null direction's
    position, which shows as a gap; single, the one-position slope. Scales: raw
    w, and standardized w * sd(x_j) / sd(y). Same grid as linear_cv_dims: mean
    over the 5 fold models and a +/- 1 sd band; single also gets a line at zero.
    The joint weights are defined only up to a shift along the null direction,
    which moves every position's weight together, so a joint curve's level is
    arbitrary and zero is not a reference for it. No text beyond panel names
    and axes; the explanation belongs in the caption.
    """
    stem = f"linear_cv_weights_{encoder}_matek_bm"
    rows = pd.read_parquet(RESULTS / f"{stem}.parquet")
    n_dims = json.loads((RESULTS / f"{stem}.json").read_text())["n_dims"]
    colour = SERIES["topk"]

    for model in ["joint_768", "joint_767", "single"]:
        by_dim = rows[rows.model == model].groupby(["descriptor", "dim"])
        for scale, ylabel in (("w", "w"), ("beta", "standardized w")):
            curves = by_dim[scale].agg(["mean", "std"])
            fig, axes = plt.subplots(3, 4, figsize=(11.5, 7.2), facecolor=SURFACE)
            fig.subplots_adjust(hspace=0.4, wspace=0.3)
            for ax, d in zip(axes.flat, DESCRIPTORS):
                c = curves.loc[d]
                _style(ax)
                x = c.index.to_numpy()
                if model == "single":  # a joint curve's level is arbitrary, so zero is no reference
                    ax.axhline(0, color=MUTED, linewidth=1.2, zorder=1)
                ax.fill_between(x, c["mean"] - c["std"], c["mean"] + c["std"], color=colour,
                                alpha=0.18, linewidth=0, zorder=2)
                ax.plot(x, c["mean"], color=colour, linewidth=0.6, zorder=3)

                ax.set_xlim(-8, n_dims + 7)
                ax.set_xticks([0, 256, 512, n_dims - 1] if n_dims == 768 else [0, n_dims - 1])
                ax.yaxis.set_major_locator(MaxNLocator(4))
                if c["mean"].abs().max() > 100:
                    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
                else:
                    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.set_title(DESCRIPTOR_LABEL[d], fontsize=9, color=INK, loc="left", pad=6)
            for ax in axes.flat[len(DESCRIPTORS):]:
                ax.axis("off")

            fig.supxlabel("embedding position", fontsize=9, color=INK_2, y=0.04)
            fig.supylabel(ylabel, fontsize=9, color=INK_2, x=0.03)
            fig.subplots_adjust(left=0.1)

            FIGURES.mkdir(exist_ok=True)
            out = FIGURES / f"linear_cv_weights_{model}_{scale}_{encoder}.pdf"
            fig.savefig(out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
            fig.savefig(out.with_suffix(".png"), dpi=170, facecolor=SURFACE,
                        bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out.relative_to(ROOT)}")


def linear_cv_ablation(encoder: str = "dinobloom") -> None:
    """R^2 against the number of positions kept and removed, as a fraction of the full model.

    From src/analysis/linear_cv_ablation.py. Colour marks which positions are
    acted on: blue the top K by |r|, orange the bottom K, grey K random ones
    (band: min-max over the random orders). Left, only those K are kept; right,
    they are removed, so removing the top K is the bottom-first curve at 768 - K.
    One line per descriptor: its R^2 (mean over the 5 validation folds) divided
    by its all-position R^2, the dark line at 1. Sized for one column. No text
    beyond axis labels; the explanation belongs in the caption.
    """
    rows = pd.read_parquet(RESULTS / f"linear_cv_ablation_{encoder}_matek_bm.parquet")
    n_dims = int(rows.k.max())
    n_draws = rows.loc[rows.order == "random", "draw"].nunique()
    curve = rows.groupby(["descriptor", "order", "draw", "k"]).r2.mean()  # mean over folds

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.12)
    for ax, removed in zip(axes, (False, True)):
        _style(ax)
        x = np.arange(1, n_dims) if removed else np.arange(1, n_dims + 1)
        row = n_dims - 1 - x if removed else x - 1  # index into a k = 1 .. 768 curve
        for d in DESCRIPTORS:
            frac = curve.loc[d] / curve.loc[(d, "top", -1, n_dims)]
            draws = np.stack([frac.loc[("random", s)].to_numpy()[row] for s in range(n_draws)])
            ax.fill_between(x, draws.min(axis=0), draws.max(axis=0), color=MUTED, alpha=0.25,
                            linewidth=0, zorder=1)
            for acted, colour in (("top", SERIES["topk"]), ("bottom", SERIES["batchtopk"])):
                order = {"top": "bottom", "bottom": "top"}[acted] if removed else acted
                ax.plot(x, frac.loc[(order, -1)].to_numpy()[row], color=colour, linewidth=0.6,
                        alpha=0.85, zorder=3)
        ax.axhline(1, color=INK_2, linewidth=0.8, zorder=2)

        ax.set_xscale("log")
        ax.set_xlim(1, n_dims)
        ax.set_xticks([1, 8, 64, 512])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=7)
        ax.set_xlabel("positions removed" if removed else "positions kept", fontsize=7, color=INK_2)
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("fraction of full R²", fontsize=7, color=INK_2)

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / f"linear_cv_ablation_{encoder}.pdf"
    fig.savefig(out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# Heatmap rows: the descriptor families of the raw_dims coordinate map (size,
# cytoplasm, shape, texture), drawn as blocks separated by a thin gap.
HEATMAP_ROWS = [["nuclear_area", "integrated_optical_density"],
                ["cytoplasmic_area", "nc_ratio"],
                ["nuclear_solidity", "form_factor", "eccentricity", "curvature_variance"],
                ["glcm_contrast", "glcm_homogeneity", "glcm_entropy"]]
TOP_K = 8


def _to_unit(v: np.ndarray, scaling: str) -> np.ndarray:
    """Map v to [-1, 1] by its min and max ("plain"), or by its 1st and 99th percentiles, clipped."""
    lo, hi = (v.min(), v.max()) if scaling == "plain" else np.percentile(v, [1, 99])
    return np.clip(2 * (v - lo) / (hi - lo) - 1, -1, 1)


def linear_cv_heatmap(encoder: str = "dinobloom") -> None:
    """Metric x position heatmap of the all-position weights, in two scalings.

    Colour: gamma_j * w_j, the joint_768 weights (src/analysis/linear_cv_weights.py,
    mean over folds) on the LayerNorm-normalized values, scaled per metric to
    [-1, 1] by its min and max (plain) or its 1st and 99th percentiles. Adding a
    constant to every gamma_j * w_j changes no prediction, and both scalings
    remove it, so the picture does not depend on the weight convention. Columns:
    the positions in the order of Ward clustering of their per-metric profiles,
    on the same values centred and divided by their sd per metric, so both
    scalings share one order. Dots: each metric's top-8 positions by |r| (the
    fold-mean single beta), the sets the ablation fitted. Right: R^2 of all
    positions (light) and of the top 8 alone (blue), from linear_cv_ablation.py.
    Bottom: each position's largest one-position R^2 over the metrics
    (linear_cv_dims.py). No text beyond ticks and axis labels; the explanation
    belongs in the caption.
    """
    stem = f"{encoder}_matek_bm"
    mean = (pd.read_parquet(RESULTS / f"linear_cv_weights_{stem}.parquet")
            .groupby(["model", "descriptor", "dim"])[["w", "beta"]].mean())
    path, key = LN_SCALE[encoder]
    gamma = load_file(str(ROOT / path))[key].astype(np.float64)
    rows = [d for family in HEATMAP_ROWS for d in family]
    gw = np.stack([gamma * mean.loc[("joint_768", d)].w.to_numpy() for d in rows])  # metrics x positions
    n_dims = gw.shape[1]
    z = (gw - gw.mean(axis=1, keepdims=True)) / gw.std(axis=1, keepdims=True)
    order = leaves_list(linkage(z.T, method="ward", optimal_ordering=True))
    column = np.empty_like(order)
    column[order] = np.arange(n_dims)  # position -> its column

    top = [mean.loc[("single", d)].beta.abs().sort_values(ascending=False).index[:TOP_K].to_numpy()
           for d in rows]
    abl = pd.read_parquet(RESULTS / f"linear_cv_ablation_{stem}.parquet")
    r2 = abl[abl.order == "top"].groupby(["descriptor", "k"]).r2.mean()
    r2_all = np.array([r2.loc[(d, n_dims)] for d in rows])
    r2_top = np.array([r2.loc[(d, TOP_K)] for d in rows])
    dims = pd.read_parquet(RESULTS / f"linear_cv_dims_{stem}.parquet")
    strip = (dims[dims.split == "val"].groupby(["descriptor", "dim"]).r2.mean()
             .unstack().max(axis=0).to_numpy())  # per position, over the metrics

    for scaling in ("plain", "percentile"):
        unit = np.stack([_to_unit(v, scaling) for v in gw])
        fig = plt.figure(figsize=(7.0, 2.8), facecolor=SURFACE)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.1], height_ratios=[1, 0.25],
                              wspace=0.04, hspace=0.08)
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(unit[:, order], aspect="auto", cmap=DIVERGING, vmin=-1, vmax=1,
                       interpolation="none")
        for i, positions in enumerate(top):
            ax.scatter(column[positions], np.full(TOP_K, i), s=3, color=INK, linewidths=0, zorder=3,
                       clip_on=False)
        for b in np.cumsum([len(f) for f in HEATMAP_ROWS])[:-1]:
            ax.axhline(b - 0.5, color=SURFACE, linewidth=1.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([DESCRIPTOR_LABEL[d] for d in rows], fontsize=6, color=INK_2)
        ax.set_xticks([])
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)

        bar = fig.add_subplot(gs[0, 1], sharey=ax)
        _style(bar)
        bar.barh(range(len(rows)), r2_all, color=MUTED, height=0.8)
        bar.barh(range(len(rows)), r2_top, color=SERIES["topk"], height=0.45)
        bar.set_xlim(0, 1)
        bar.set_xticks([0, 1])
        bar.xaxis.tick_top()  # keeps the space below free for the colour bar
        bar.xaxis.set_label_position("top")
        bar.spines["bottom"].set_visible(False)
        bar.spines["top"].set_visible(True)
        bar.spines["top"].set_color(MUTED)
        bar.tick_params(labelleft=False, left=False, labelsize=6)
        bar.set_xlabel("R²", fontsize=6, color=INK_2)

        low = fig.add_subplot(gs[1, 0], sharex=ax)
        _style(low)
        low.fill_between(np.arange(n_dims), strip[order], color=INK_2, linewidth=0, step="mid")
        low.set_ylim(0, None)
        low.set_xticks([])
        low.yaxis.set_major_locator(MaxNLocator(2))
        low.tick_params(labelsize=6)
        low.set_ylabel("R²", fontsize=6, color=INK_2)
        low.set_xlabel("embedding position (clustered)", fontsize=6, color=INK_2)

        cax = fig.add_subplot(gs[1, 1].subgridspec(2, 1)[1])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_visible(False)

        FIGURES.mkdir(exist_ok=True)
        out = FIGURES / f"linear_cv_heatmap_{scaling}_{encoder}.pdf"
        fig.savefig(out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.relative_to(ROOT)}")


def linear_cv_regions(encoder: str = "dinobloom") -> None:
    """Which cell region each target depends on: median share of the region-occlusion effect.

    From src/analysis/linear_cv_regions.py, evaluation cells only. Rows: each
    metric's all-position readout (HEATMAP_ROWS order), then two references
    pooled over the metrics: the metrics' best single positions and the random
    control positions. Columns: nucleus, cytoplasm, rest. Colour: the median of
    |effect| / sum of the three regions' |effects|, on one scale for both
    panels; a dot marks each row's largest share. Left, regions filled with the
    cell's background colour; right, with grey. Sized for one column. No text
    beyond panel names, ticks and axis labels; the explanation belongs in the
    caption.
    """
    rows = pd.read_parquet(RESULTS / f"linear_cv_regions_{encoder}_matek_bm.parquet")
    ev = rows[rows.set == "evaluation"]
    regions = ["nucleus", "cytoplasm", "rest"]
    metrics = [d for family in HEATMAP_ROWS for d in family]
    readout = ev[ev.kind == "readout"].groupby(["fill", "descriptor", "region"]).share.median()
    best = ev[(ev.kind == "position") & (ev.which == "best")].groupby(["fill", "region"]).share.median()
    random = ev[ev.kind == "random"].groupby(["fill", "region"]).share.median()
    labels = [DESCRIPTOR_LABEL[d] for d in metrics] + ["best positions", "random positions"]
    cmap = LinearSegmentedColormap.from_list("share", ["#eeece7", SERIES["topk"]])

    shares = {fill: np.vstack([[[readout.loc[(fill, d, g)] for g in regions] for d in metrics],
                               [[best.loc[(fill, g)] for g in regions]],
                               [[random.loc[(fill, g)] for g in regions]]])
              for fill in ("background", "grey")}
    vmax = max(s.max() for s in shares.values())
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.9), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.08)
    for ax, (fill, s) in zip(axes, shares.items()):
        y = np.r_[np.arange(len(metrics)), len(metrics) + 0.5 + np.arange(2)]  # gap before the references
        for i, yi in enumerate(y):
            for j in range(len(regions)):
                ax.add_patch(plt.Rectangle((j - 0.5, yi - 0.5), 1, 1, color=cmap(s[i, j] / vmax), linewidth=0))
            ax.scatter(int(np.argmax(s[i])), yi, s=4, color=INK, linewidths=0, zorder=3)
        ax.set_xlim(-0.5, len(regions) - 0.5)
        ax.set_ylim(y[-1] + 0.5, -0.5)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, fontsize=6, color=INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6, color=INK_2)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(fill, fontsize=7, color=INK, pad=4)
    axes[1].tick_params(labelleft=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08,
                      ticks=[0, round(vmax, 1)])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / f"linear_cv_regions_{encoder}.pdf"
    fig.savefig(out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def linear_cv_attnpath(encoder: str = "dinobloom") -> None:
    """Which region's tokens carry each readout's variation: attention-path variance shares.

    From src/analysis/linear_cv_attnpath.py: 2,000 CV cells, each read by its
    out-of-fold readout. Rows: each metric's all-position readout (HEATMAP_ROWS
    order, a gap between families). Left, patch writes split by pixel fraction
    into nucleus, cytoplasm and rest, beside the CLS's own MLP writes; right,
    interior patches only, with boundary patches as their own column. Colour:
    the share of the readout's variance across cells, Cov(T_g, Y) / Var(Y), on
    one scale for both panels and both files; a dot marks each row's largest
    region. The CLS reading itself and the constants (|share| <= .014) are left
    out. Two files: over all cells, and within cell type (_within). Sized for one
    column. No text beyond panel names, ticks and axis labels; the explanation
    belongs in the caption.
    """
    summary = json.loads((RESULTS / f"linear_cv_attnpath_{encoder}_matek_bm.json").read_text())
    regions = ["nucleus", "cytoplasm", "rest"]
    panels = {"pixel split": ("shares_all_blocks", regions + ["mlp"]),
              "interior only": ("shares_all_blocks_interior_split",
                                [f"{r}_interior" for r in regions] + ["boundary", "mlp"])}
    tick = {"mlp": "MLP", "boundary": "boundary", **{r: r for r in regions},
            **{f"{r}_interior": r for r in regions}}
    metrics = [d for family in HEATMAP_ROWS for d in family]
    y, n = [], 0
    for f, family in enumerate(HEATMAP_ROWS):  # half a row of space between families
        y += [n + 0.5 * f + i for i in range(len(family))]
        n += len(family)
    shares = {scope: {name: np.array([[summary[scope][key][d][c] for c in cols] for d in metrics])
                      for name, (key, cols) in panels.items()} for scope in ("all_cells", "within_type")}
    vmax = max(s.max() for scope in shares.values() for s in scope.values())
    cmap = LinearSegmentedColormap.from_list("share", ["#eeece7", SERIES["topk"]])

    for scope, suffix in (("all_cells", ""), ("within_type", "_within")):
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 3.0), sharey=True, facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [4, 5], "wspace": 0.08})
        for ax, (name, (key, cols)) in zip(axes, panels.items()):
            s = shares[scope][name]
            for i, yi in enumerate(y):
                for j in range(len(cols)):
                    ax.add_patch(plt.Rectangle((j - 0.5, yi - 0.5), 1, 1, color=cmap(max(s[i, j], 0) / vmax),
                                               linewidth=0))
                ax.scatter(int(np.argmax(s[i, :len(regions)])), yi, s=4, color=INK, linewidths=0, zorder=3)
            ax.set_xlim(-0.5, len(cols) - 0.5)
            ax.set_ylim(y[-1] + 0.5, -0.5)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([tick[c] for c in cols], fontsize=6, color=TEXT, rotation=90)
            ax.set_yticks(y)
            ax.set_yticklabels([DESCRIPTOR_LABEL[d] for d in metrics], fontsize=6, color=TEXT)
            ax.tick_params(length=0)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(name, fontsize=7, color=TEXT, pad=4)
        axes[1].tick_params(labelleft=False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
        cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.26,
                          ticks=[0, np.floor(vmax * 10) / 10])  # a round tick inside the scale
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_visible(False)

        FIGURES.mkdir(exist_ok=True)
        out = FIGURES / f"linear_cv_attnpath{suffix}_{encoder}.pdf"
        fig.savefig(out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.relative_to(ROOT)}")


# Per-readout patch maps on PBC (src/analysis/linear_unseen_attnpath.py): the cell types shown,
# each the script's seeded random draw, and readouts from the size, cytoplasm, shape and texture
# families.
MAP_TYPES = ["neutrophil", "lymphocyte", "monocyte"]
MAP_READOUTS = ["nuclear_area", "nc_ratio", "form_factor", "eccentricity", "glcm_contrast"]
MAP_TILE = 0.85  # inches per tile


def linear_unseen_maps(encoder: str = "dinobloom", split: str = "test", readouts: list[str] = MAP_READOUTS,
                       suffix: str = "") -> None:
    """Where each readout reads, cell by cell: centred attention-path maps on PBC (protocol.md §16).

    From src/analysis/linear_unseen_attnpath.py: per PBC cell type, one cell drawn at
    random (seed 0) before any map was viewed; MAP_TYPES are shown. Rows: cells. First
    column: the crop as the encoder saw it (resized to 224 px without padding). Then one
    16 x 16 map per readout in `readouts` (the paper's five: MAP_READOUTS): each patch token's direct-attention write to
    the readout, summed over the 12 blocks and averaged over the five readouts, minus that
    patch's mean write over all scored cells, in units of the readout's sd across cells.
    One symmetric scale for every map, clipped at the 99th percentile of |value| over the
    shown maps. Lines: nucleus and cell outlines, the 0.5 level of the per-patch nucleus and
    nucleus + cytoplasm fractions of the 25-seed U-Net majority mask. Full width. No text
    beyond panel names, row labels and colour-bar ticks; the explanation belongs in the caption.
    """
    from PIL import Image

    from src.embed import SIZE, CropDataset

    maps = pd.read_parquet(RESULTS / f"linear_unseen_maps_{encoder}_pbc.parquet")
    maps = maps[maps.split == split].sort_values("patch")
    cells = maps.drop_duplicates("cell_id").set_index("class_label").cell_id.loc[MAP_TYPES]
    index = pd.read_parquet(ROOT / "data" / "interim" / "pbc" / "index.parquet").set_index("cell_id")
    crops = CropDataset(index.loc[cells.to_numpy()].reset_index(), "pbc", "bicubic", pad=False)
    grid = int(np.sqrt(maps.patch.nunique()))
    value = (maps.write - maps.write_mean) / maps.y_sd
    tiles = {(c, d): value[(maps.cell_id == c) & (maps.descriptor == d)].to_numpy().reshape(grid, grid)
             for c in cells for d in readouts}
    lim = float(np.percentile(np.abs(np.stack(list(tiles.values()))), 99))
    patches = maps.drop_duplicates(["cell_id", "patch"]).set_index("cell_id")
    outlines = {c: [patches.loc[c].frac_nucleus.to_numpy().reshape(grid, grid),
                    (patches.loc[c].frac_nucleus + patches.loc[c].frac_cytoplasm).to_numpy().reshape(grid, grid)]
                for c in cells}

    n_cols = 1 + len(readouts)
    fig, axes = plt.subplots(len(MAP_TYPES), n_cols, figsize=(MAP_TILE * n_cols + 0.6, MAP_TILE * len(MAP_TYPES) + 0.3),
                             facecolor=SURFACE, gridspec_kw={"wspace": 0.04, "hspace": 0.04})
    for r, (cell_type, cid) in enumerate(cells.items()):
        crop = Image.open(crops._source(r)).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        axes[r, 0].imshow(crop)
        axes[r, 0].set_ylabel(cell_type, fontsize=6, color=TEXT)
        for j, d in enumerate(readouts):
            im = axes[r, j + 1].imshow(tiles[(cid, d)], cmap=DIVERGING, vmin=-lim, vmax=lim,
                                       interpolation="nearest")
            for field in outlines[cid]:  # nucleus, then cell
                axes[r, j + 1].contour(field, levels=[0.5], colors=[INK], linewidths=0.6)
            if r == 0:
                label = DESCRIPTOR_LABEL[d]
                if len(label) > 16:  # the long names wrap (all-readouts figure); the paper's five fit on one line
                    cut = label.rfind(" ", 0, len(label) // 2 + 1)
                    label = label[:cut] + "\n" + label[cut + 1:]
                axes[r, j + 1].set_title(label, fontsize=6, color=TEXT, pad=3)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    mag = 10 ** np.floor(np.log10(lim))
    tick = np.floor(lim / mag) * mag  # a round tick inside the scale
    cb = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.015, pad=0.01, ticks=[-tick, 0, tick])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / f"linear_unseen_maps{suffix}_{encoder}.pdf"
    fig.savefig(out, format="pdf", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", default="dinobloom")
    ap.add_argument("--only", choices=["morphometry", "dims", "weights", "ablation",
                                       "heatmap", "regions", "attnpath", "maps", "maps_all"])
    args = ap.parse_args()
    if args.only in (None, "morphometry"):
        morphometry_effect_sizes()
    if args.only in (None, "dims"):
        linear_cv_dims(args.encoder)
        linear_cv_dims(args.encoder, score="r2")
    if args.only in (None, "weights"):
        linear_cv_weights(args.encoder)
    if args.only in (None, "ablation"):
        linear_cv_ablation(args.encoder)
    if args.only in (None, "heatmap"):
        linear_cv_heatmap(args.encoder)
    if args.only in (None, "regions"):
        linear_cv_regions(args.encoder)
    if args.only in (None, "attnpath"):
        linear_cv_attnpath(args.encoder)
    if args.only in (None, "maps"):
        linear_unseen_maps(args.encoder)
    if args.only in (None, "maps_all"):   # all 11 readouts, for the repository
        linear_unseen_maps(args.encoder, readouts=[d for family in HEATMAP_ROWS for d in family], suffix="_all")


if __name__ == "__main__":
    main()
