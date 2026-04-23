#!/usr/bin/env python3
"""Create key figures for EXPERIMENT_RESULTS.md.

Figures
-------
Figure A (poster): Accuracy–Latency–Model Size trade-off (batch=1, log-latency)
  - A: 2x2 grid (datasets)
  - Color: model (fixed palette)
  - Size: params (per-dataset normalized, linear)
  - Labels: only Mamba / LeNet / GRU / ViT

Figure B: Params (log) × Accuracy
Figure C: Peak GPU memory × Accuracy

Inputs
------
Uses aggregate CSVs under `result/artifacts/aggregate/` by default.

Outputs
-------
Writes PNG+PDF into `result_figure/EXPERIMENT_RESULTS/` by default.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Configure cache dirs prior to importing Matplotlib so fontconfig does not try
# to write into read-only system locations.
cache_root = Path(".cache")
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
cache_root.mkdir(parents=True, exist_ok=True)
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
(cache_root / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FC_CACHEDIR", str(cache_root / "fontconfig"))

import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MODEL_COLORS: Dict[str, str] = {
    # SSM (main)
    "Mamba": "#d62728",
    # MLP (independent outlier)
    "MLP": "#4d4d4d",
    # RNN-family (blue hues)
    "GRU": "#1f77b4",
    "LSTM": "#003f7f",
    "BiLSTM": "#6baed6",
    "RNN": "#9ecae1",
    # CNN-family (green hues)
    "LeNet": "#a1d99b",
    "CNN+GRU": "#31a354",
    "ResNet18": "#006d2c",
    "ResNet50": "#00441b",
    "ResNet101": "#002d13",
    # Transformer
    "ViT": "#9467bd",
}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def model_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#999999")

@dataclass(frozen=True)
class BubbleSizer:
    mode: str  # "linear" or "log"
    p_max: float
    size_min: float = 25.0
    size_max: float = 950.0

    def size(self, params: float) -> float:
        if params <= 0 or self.p_max <= 0:
            return self.size_min
        if self.mode == "log":
            # Log-normalized (previous behavior): size = 200*(log10(p)-4)
            size = 200.0 * (math.log10(float(params)) - 4.0)
            return float(min(self.size_max, max(self.size_min, size)))
        # Linear/proportional scaling to the max within the plotted group.
        # Area ~ params (after normalization), with a floor to keep small models visible.
        span = self.size_max - self.size_min
        size = self.size_min + (float(params) / float(self.p_max)) * span
        return float(min(self.size_max, max(self.size_min, size)))


def best_variant_per_model(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        model = str(r.get("model", "")).strip()
        acc = _to_float(r.get("acc_mean"))
        if not model or acc is None:
            continue
        if model not in best or acc > (_to_float(best[model].get("acc_mean")) or -1.0):
            best[model] = r
    return list(best.values())


@dataclass(frozen=True)
class Point:
    dataset: str
    model: str
    variant: str
    x: float
    y: float
    color: str
    size: float


def _dataset_styles() -> Dict[str, Dict[str, Any]]:
    return {
        "NTU-Fi-HumanID": {"label": "NTU-Fi HumanID"},
        "NTU-Fi_HAR": {"label": "NTU-Fi HAR"},
        "Widar": {"label": "Widar3.0"},
        "UT_HAR_data": {"label": "UT-HAR"},
    }


def _scatter_points(ax, points: List[Point], alpha: float = 0.7) -> None:
    # Draw larger bubbles first so smaller models remain visible on top.
    points_sorted = sorted(points, key=lambda p: p.size, reverse=True)
    ax.scatter(
        [p.x for p in points_sorted],
        [p.y for p in points_sorted],
        s=[p.size for p in points_sorted],
        c=[p.color for p in points_sorted],
        alpha=alpha,
        edgecolors="white",
        linewidths=0.7,
        zorder=2,
    )


def _color_legend_handles() -> List[Any]:
    order = [
        "Mamba",
        "MLP",
        "GRU",
        "LSTM",
        "BiLSTM",
        "RNN",
        "LeNet",
        "CNN+GRU",
        "ResNet18",
        "ResNet50",
        "ResNet101",
        "ViT",
    ]
    labels = {m: m for m in order}
    labels["Mamba"] = "Mamba (SSM)"
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="none",
            markerfacecolor=model_color(m),
            markeredgecolor="white",
            markersize=7.5,
            label=labels[m],
        )
        for m in order
    ]


def _add_color_legend(ax, anchor=(1.02, 1.0), ncol: int = 3, loc: str = "upper left") -> Any:
    handles = _color_legend_handles()
    return ax.legend(
        handles=handles,
        title="Color (model)",
        loc=loc,
        bbox_to_anchor=anchor,
        fontsize="x-small",
        ncol=ncol,
        frameon=True,
        handletextpad=0.4,
        columnspacing=0.8,
        borderpad=0.6,
        labelspacing=0.3,
        borderaxespad=0.0,
    )


def _legend_marker_sizes_from_areas(areas: List[float], ms_max: float = 14.0) -> List[float]:
    # Convert scatter `s` (area) into legend marker sizes (points) while preserving ratios.
    # Matplotlib's Line2D markersize is roughly proportional to diameter in points.
    if not areas:
        return []
    safe = [max(0.0, float(a)) for a in areas]
    max_a = max(safe) if safe else 1.0
    if max_a <= 0:
        return [6.0 for _ in safe]
    scale = (math.sqrt(max_a) / ms_max) if ms_max > 0 else 1.0
    out = []
    for a in safe:
        ms = math.sqrt(a) / scale
        out.append(float(min(ms_max, max(5.0, ms))))
    return out

def _format_params_short(params: float) -> str:
    if params <= 0:
        return "0"
    if params >= 1e9:
        return f"{params/1e9:.1f}B"
    if params >= 1e6:
        return f"{params/1e6:.1f}M"
    if params >= 1e3:
        return f"{params/1e3:.0f}K"
    return f"{params:.0f}"

def _add_size_legend(
    ax,
    sizer: BubbleSizer,
    ref_params: List[float],
    anchor=(1.02, 0.48),
    loc: str = "upper left",
) -> Any:
    # Use reference points (min/median/max) so the legend reads as a continuous scale,
    # not as discrete bins.
    refs = [p for p in ref_params if p is not None and p > 0]
    refs = sorted(set(float(p) for p in refs))
    areas = [sizer.size(p) for p in refs]
    ms_list = _legend_marker_sizes_from_areas(areas, ms_max=14.0)
    handles = []
    for p, ms in zip(refs, ms_list):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color="none",
                markerfacecolor="#7f7f7f",
                markeredgecolor="white",
                alpha=0.35,
                markersize=ms,
                label=f"~{_format_params_short(p)} params",
            )
        )
    scale_label = "linear" if sizer.mode == "linear" else "log"
    title = f"Bubble area ∝ Params (scaled: {scale_label})"
    return ax.legend(
        handles=handles,
        title=title,
        loc=loc,
        bbox_to_anchor=anchor,
        fontsize="small",
        frameon=True,
        borderaxespad=0.0,
        handletextpad=0.6,
        labelspacing=0.6,
    )


def _annotate_models(ax, points: List[Point], models_to_annotate: Iterable[str]) -> None:
    def _axis_frac_x(x: float) -> float:
        xmin, xmax = ax.get_xlim()
        if ax.get_xscale() == "log":
            xmin = max(float(xmin), 1e-12)
            xmax = max(float(xmax), xmin * 1.0001)
            x = max(float(x), 1e-12)
            return (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
        if xmax == xmin:
            return 0.5
        return (float(x) - float(xmin)) / (float(xmax) - float(xmin))

    def _axis_frac_y(y: float) -> float:
        ymin, ymax = ax.get_ylim()
        if ymax == ymin:
            return 0.5
        return (float(y) - float(ymin)) / (float(ymax) - float(ymin))

    want = set(models_to_annotate)
    for p in points:
        if p.model not in want:
            continue
        x_frac = _axis_frac_x(p.x)
        y_frac = _axis_frac_y(p.y)
        # Keep labels inside the axes by choosing offsets that move inward.
        dx = -5 if x_frac > 0.85 else 5
        dy = -6 if y_frac > 0.88 else 5
        ha = "right" if dx < 0 else "left"
        va = "top" if dy < 0 else "bottom"
        ax.annotate(
            f"{p.model}",
            (p.x, p.y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color=p.color,
            ha=ha,
            va=va,
            clip_on=True,
            annotation_clip=True,
            arrowprops=dict(
                arrowstyle="-",
                color=p.color,
                lw=0.8,
                shrinkA=0,
                shrinkB=2,
                alpha=0.9,
            ),
        )

def _points_for_dataset(summary_rows: List[Dict[str, Any]], dataset: str, sizer: BubbleSizer) -> List[Point]:
    ds_rows = [r for r in summary_rows if str(r.get("dataset", "")).strip() == dataset]
    ds_rows = best_variant_per_model(ds_rows)
    pts: List[Point] = []
    for r in ds_rows:
        model = str(r.get("model", "")).strip()
        variant = str(r.get("variant", "")).strip()
        lat = _to_float(r.get("latency_ms_batch1_mean"))
        acc = _to_float(r.get("acc_mean"))
        params = _to_float(r.get("params_total_mean"))
        if model == "" or lat is None or acc is None or params is None:
            continue
        pts.append(
            Point(
                dataset=dataset,
                model=model,
                variant=variant,
                x=float(lat),
                y=float(acc) * 100.0,
                color=model_color(model),
                size=sizer.size(float(params)),
            )
        )
    return pts


def plot_tradeoff_pareto_two_panel(
    summary_rows: List[Dict[str, Any]],
    left_dataset: str,
    right_dataset: str,
    out_path: Path,
    title: str,
    ylim_left: Tuple[float, float],
    ylim_right: Tuple[float, float],
    bubble_mode: str = "linear",
) -> None:
    styles = _dataset_styles()
    # Use a shared scaling for bubble sizes across the two panels.
    group_rows = [
        r
        for r in summary_rows
        if str(r.get("dataset", "")).strip() in {left_dataset, right_dataset}
    ]
    group_rows = best_variant_per_model(group_rows)
    params_vals = [_to_float(r.get("params_total_mean")) for r in group_rows]
    params_vals = sorted([p for p in params_vals if p is not None and p > 0])
    p_max = max(params_vals) if params_vals else 1.0
    sizer = BubbleSizer(mode=bubble_mode, p_max=float(p_max))

    left_pts = _points_for_dataset(summary_rows, left_dataset, sizer)
    right_pts = _points_for_dataset(summary_rows, right_dataset, sizer)

    all_pts = left_pts + right_pts
    xs = [p.x for p in all_pts if p.x > 0]
    xmin = min(xs) if xs else 1e-3
    xmax = max(xs) if xs else 1.0
    xlim = (xmin * 0.8, xmax * 1.25)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True)
    for ax, ds, pts, ylim in [
        (axes[0], left_dataset, left_pts, ylim_left),
        (axes[1], right_dataset, right_pts, ylim_right),
    ]:
        _scatter_points(ax, pts, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(styles.get(ds, {}).get("label", ds))
        ax.grid(alpha=0.25, which="both")
        ax.set_xlabel("Latency (ms, batch=1) [log scale]")
        ax.set_ylabel("Accuracy (%)")
        _annotate_models(ax, pts, models_to_annotate=["Mamba", "LeNet", "GRU", "ViT"])

    fig.suptitle(title, y=0.98)
    # Single shared color legend centered under both panels.
    color_legend = fig.legend(
        handles=_color_legend_handles(),
        title="Color (model)",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize="x-small",
        ncol=6,  # 12 models -> 2 rows
        frameon=True,
        handletextpad=0.4,
        columnspacing=0.8,
        borderpad=0.6,
        labelspacing=0.3,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.12, 1.0, 0.95))
    fig.savefig(out_path, dpi=250, bbox_inches="tight", bbox_extra_artists=[color_legend], pad_inches=0.2)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=[color_legend])
    plt.close(fig)


def plot_tradeoff_pareto_grid_2x2(
    summary_rows: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    bubble_mode: str = "linear",
) -> None:
    styles = _dataset_styles()
    grid = [
        ["NTU-Fi-HumanID", "NTU-Fi_HAR"],
        ["UT_HAR_data", "Widar"],
    ]
    ylims = {
        "NTU-Fi-HumanID": (60, 100),
        "NTU-Fi_HAR": (60, 100),
        "UT_HAR_data": (40, 100),
        "Widar": (40, 100),
    }

    # Per-dataset normalization (linear scaling within each subplot).
    points_by_ds: Dict[str, List[Point]] = {}
    all_pts: List[Point] = []
    for row in grid:
        for ds in row:
            ds_rows = [r for r in summary_rows if str(r.get("dataset", "")).strip() == ds]
            ds_rows = best_variant_per_model(ds_rows)
            params_vals = [_to_float(r.get("params_total_mean")) for r in ds_rows]
            params_vals = [p for p in params_vals if p is not None and p > 0]
            p_max = max(params_vals) if params_vals else 1.0
            sizer = BubbleSizer(mode="linear", p_max=float(p_max))
            pts = _points_for_dataset(summary_rows, ds, sizer)
            points_by_ds[ds] = pts
            all_pts.extend(pts)

    xs = [p.x for p in all_pts if p.x > 0]
    xmin = min(xs) if xs else 1e-3
    xmax = max(xs) if xs else 1.0
    xlim = (xmin * 0.8, xmax * 1.25)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), sharex=True)
    for r, row in enumerate(grid):
        for c, ds in enumerate(row):
            ax = axes[r, c]
            pts = points_by_ds.get(ds, [])
            _scatter_points(ax, pts, alpha=0.7)
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylims.get(ds, (40, 100)))
            ax.set_title(styles.get(ds, {}).get("label", ds))
            ax.grid(alpha=0.25, which="both")
            if r == 1:
                ax.set_xlabel("Latency (ms, batch=1) [log scale]")
            if c == 0:
                ax.set_ylabel("Accuracy (%)")
            _annotate_models(ax, pts, models_to_annotate=["Mamba", "LeNet", "GRU", "ViT"])

    fig.suptitle(title, y=0.985)
    color_legend = fig.legend(
        handles=_color_legend_handles(),
        title="Color (model)",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        fontsize="x-small",
        ncol=6,
        frameon=True,
        handletextpad=0.4,
        columnspacing=0.8,
        borderpad=0.6,
        labelspacing=0.3,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.08, 1.0, 0.95))
    fig.savefig(out_path, dpi=250, bbox_inches="tight", bbox_extra_artists=[color_legend], pad_inches=0.2)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=[color_legend])
    plt.close(fig)


def plot_params_vs_accuracy(
    summary_rows: List[Dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    datasets = sorted({str(r.get("dataset", "")).strip() for r in summary_rows if str(r.get("dataset", "")).strip()})

    # One point per model per dataset (best-acc variant).
    points: List[Point] = []
    all_models = sorted({str(r.get("model", "")).strip() for r in summary_rows if str(r.get("model", "")).strip()})
    palette = {m: model_color(m) for m in all_models}
    for ds in datasets:
        ds_rows = best_variant_per_model([r for r in summary_rows if str(r.get("dataset", "")).strip() == ds])
        for r in ds_rows:
            model = str(r.get("model", "")).strip()
            variant = str(r.get("variant", "")).strip()
            params = _to_float(r.get("params_total_mean"))
            acc = _to_float(r.get("acc_mean"))
            if model == "" or params is None or acc is None or params <= 0:
                continue
            points.append(
                Point(
                    dataset=ds,
                    model=model,
                    variant=variant,
                    x=float(params),
                    y=float(acc) * 100.0,
                    color=palette.get(model, "#377eb8"),
                    size=55.0,
                )
            )

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for ds in datasets:
        ds_pts = [p for p in points if p.dataset == ds]
        if not ds_pts:
            continue
        ax.scatter(
            [p.x for p in ds_pts],
            [p.y for p in ds_pts],
            s=55,
            c=[p.color for p in ds_pts],
            marker="o",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Params (total) [log scale]")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")

    # Legends: reuse family + bubble legend style for consistency (but size is not meaningful here).
    model_legend = _add_color_legend(ax, anchor=(1.02, 1.0), ncol=3)
    dataset_legend = None

    _annotate_models(ax, points, models_to_annotate=["MLP", "Mamba", "ViT", "GRU"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_path, dpi=250, bbox_inches="tight", bbox_extra_artists=[model_legend], pad_inches=0.2)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=[model_legend])
    plt.close(fig)


def plot_peak_mem_vs_accuracy(
    summary_rows: List[Dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    datasets = sorted({str(r.get("dataset", "")).strip() for r in summary_rows if str(r.get("dataset", "")).strip()})
    all_models = sorted({str(r.get("model", "")).strip() for r in summary_rows if str(r.get("model", "")).strip()})
    palette = {m: model_color(m) for m in all_models}

    points: List[Point] = []
    for ds in datasets:
        ds_rows = best_variant_per_model([r for r in summary_rows if str(r.get("dataset", "")).strip() == ds])
        for r in ds_rows:
            model = str(r.get("model", "")).strip()
            variant = str(r.get("variant", "")).strip()
            mem = _to_float(r.get("peak_gpu_mem_mb_mean"))
            acc = _to_float(r.get("acc_mean"))
            if model == "" or mem is None or acc is None:
                continue
            points.append(
                Point(
                    dataset=ds,
                    model=model,
                    variant=variant,
                    x=float(mem),
                    y=float(acc) * 100.0,
                    color=palette.get(model, "#377eb8"),
                    size=55.0,
                )
            )

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for ds in datasets:
        ds_pts = [p for p in points if p.dataset == ds]
        if not ds_pts:
            continue
        ax.scatter(
            [p.x for p in ds_pts],
            [p.y for p in ds_pts],
            s=55,
            c=[p.color for p in ds_pts],
            marker="o",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
        )

    ax.set_xlabel("Peak GPU memory (MB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    model_legend = _add_color_legend(ax, anchor=(1.02, 1.0), ncol=3)
    dataset_legend = None

    _annotate_models(ax, points, models_to_annotate=["ViT", "MLP", "Mamba", "GRU"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(out_path, dpi=250, bbox_inches="tight", bbox_extra_artists=[model_legend], pad_inches=0.2)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=[model_legend])
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate-dir", type=Path, default=Path("result/artifacts/aggregate"))
    p.add_argument("--out-dir", type=Path, default=Path("result_figure/EXPERIMENT_RESULTS"))
    p.add_argument("--bubble-mode", choices=["linear", "log"], default="linear")
    args = p.parse_args()

    summary_path = args.aggregate_dir / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    summary_rows = _read_csv(summary_path)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_tradeoff_pareto_grid_2x2(
        summary_rows,
        out_path=out_dir / "FigureA_tradeoff_latency_batch1_2x2.png",
        title="Figure A: Accuracy–Latency–Model Size trade-off (batch=1)",
        bubble_mode="linear",
    )
    plot_params_vs_accuracy(
        summary_rows,
        out_path=out_dir / "FigureB_params_log_vs_accuracy.png",
        title="Figure B: Params (log) vs Accuracy",
    )
    plot_peak_mem_vs_accuracy(
        summary_rows,
        out_path=out_dir / "FigureC_peak_gpu_mem_vs_accuracy.png",
        title="Figure C: Peak GPU memory vs Accuracy",
    )

    print(f"Wrote EXPERIMENT_RESULTS figures -> {out_dir}")


if __name__ == "__main__":
    main()
