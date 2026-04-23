#!/usr/bin/env python3
"""Generate unified figures from `runs/` and aggregate CSVs.

This script is meant for local (non-`public/`) workflows and writes all figures
under a single output directory (default: `result_figure/`):

- Accuracy bar chart (mean±std) per dataset from `summary.csv`
- Pareto scatter (accuracy vs latency) per dataset from `pareto.csv`
- Averaged confusion matrices per (dataset, model, variant) from run artifacts
  (`confusion_matrix.npy` stored alongside each `metrics.json`)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

# Configure cache dirs prior to importing Matplotlib so fontconfig does not try
# to write into read-only system locations.
cache_root = pathlib.Path(".cache")
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
cache_root.mkdir(parents=True, exist_ok=True)
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
(cache_root / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FC_CACHEDIR", str(cache_root / "fontconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Use Type 42 fonts in PDF/PS for publication-quality vector output
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MAMBA_COLOR = "#e41a1c"  # strong red to highlight the new model prominently
BASE_COLORS = [
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#f781bf",
    "#a65628",
    "#999999",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]


def _cycle_colors(count: int) -> List[str]:
    if count <= len(BASE_COLORS):
        return BASE_COLORS[:count]
    repeats = (count + len(BASE_COLORS) - 1) // len(BASE_COLORS)
    return (BASE_COLORS * repeats)[:count]


def build_model_palette(models: List[str]) -> Dict[str, str]:
    palette: Dict[str, str] = {}
    if "Mamba" in models:
        palette["Mamba"] = MAMBA_COLOR
    remaining_models = [m for m in sorted(models) if m not in palette]
    for model, color in zip(remaining_models, _cycle_colors(len(remaining_models))):
        palette[model] = color
    return palette


def _read_csv(path: pathlib.Path) -> List[Dict[str, Any]]:
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


def _to_int(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _best_variant_per_model(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for r in summary_rows:
        model = str(r.get("model", "")).strip()
        acc = _to_float(r.get("acc_mean"))
        if not model or acc is None:
            continue
        if model not in best or acc > (_to_float(best[model].get("acc_mean")) or -1.0):
            best[model] = r
    return sorted(best.values(), key=lambda r: _to_float(r.get("acc_mean")) or 0.0, reverse=True)


def _summarize_seed_count(rows: List[Dict[str, Any]]) -> str | None:
    ns = [_to_int(r.get("n")) for r in rows]
    ns = [n for n in ns if n is not None]
    if not ns:
        return None
    if len(set(ns)) == 1:
        return f"Seeds: n={ns[0]}"
    return f"Seeds: n≈{int(round(float(np.mean(ns))))}"


def plot_accuracy_bar_meanstd(
    summary_rows: List[Dict[str, Any]],
    out_path: pathlib.Path,
    dataset: str,
    palette: Dict[str, str],
) -> None:
    best_rows = _best_variant_per_model(summary_rows)
    labels = [str(r.get("model", "")) for r in best_rows]
    means = [(_to_float(r.get("acc_mean")) or 0.0) * 100.0 for r in best_rows]
    stds_raw = [(_to_float(r.get("acc_std")) or 0.0) * 100.0 for r in best_rows]
    stds = [min(std, max(0.0, 100.0 - mean)) for mean, std in zip(means, stds_raw)]
    colors = [palette.get(m, "#377eb8") for m in labels]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.barh(labels, means, xerr=stds, color=colors, alpha=0.9, capsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    subtitle = _summarize_seed_count(best_rows)
    title = f"{dataset} Validation Accuracy (best variant per model)"
    if subtitle:
        title = f"{title}\n{subtitle}"
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    upper = max((m + s for m, s in zip(means, stds)), default=100.0)
    ax.set_xlim(0, min(115, max(100, upper + 8)))

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            mean + std + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.1f}±{std:.1f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    pareto_rows: List[Dict[str, Any]],
    out_path: pathlib.Path,
    dataset: str,
    batch: str,
    palette: Dict[str, str],
    label_points: bool = True,
) -> None:
    points: List[Tuple[float, float, str, str, float]] = []
    for r in pareto_rows:
        acc = _to_float(r.get("acc_mean"))
        lat = _to_float(r.get(f"latency_ms_batch{batch}_mean"))
        params = _to_float(r.get("params_total_mean"))
        model = str(r.get("model", "")).strip()
        variant = str(r.get("variant", "")).strip()
        if not model or acc is None or lat is None:
            continue
        size = max(18.0, (params or 1.0) ** 0.5 / 12.0)
        points.append((lat, acc * 100.0, model, variant, size))

    if not points:
        return

    variants_by_model: Dict[str, set[str]] = {}
    for _, _, model, variant, _ in points:
        variants_by_model.setdefault(model, set()).add(variant)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for lat, acc_pct, model, variant, size in points:
        color = palette.get(model, "#377eb8")
        ax.scatter(lat, acc_pct, s=size, alpha=0.75, color=color, edgecolors="white", linewidths=0.6)
        if label_points:
            label = model
            if variant and len(variants_by_model.get(model, set())) > 1:
                label = f"{model} ({variant})"
            ax.annotate(
                label,
                (lat, acc_pct),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    ax.set_xlabel(f"Latency (ms) batch={batch}")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{dataset} Pareto: Accuracy vs Latency (batch={batch})")
    ax.grid(alpha=0.3)

    # Legend: model -> color
    unique_models = sorted({m for _, _, m, _, _ in points})
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette.get(m, "#377eb8"), markersize=8, label=m)
        for m in unique_models
    ]
    legend = ax.legend(handles=handles, title="Model", loc="lower left", bbox_to_anchor=(1.01, 0.0), fontsize="small")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", bbox_extra_artists=[legend], pad_inches=0.2)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", bbox_extra_artists=[legend])
    plt.close(fig)


def canonicalize_labels(class_names: List[str]) -> Tuple[List[str], List[int]]:
    if class_names and all(name.isdigit() for name in class_names):
        numeric = [int(name) for name in class_names]
        order = sorted(range(len(numeric)), key=lambda i: numeric[i])
        sorted_names = [f"{numeric[i]:03d}" for i in order]
        return sorted_names, order
    return class_names, list(range(len(class_names)))


def reorder_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    sorted_names, order = canonicalize_labels(class_names)
    if order != list(range(len(class_names))):
        cm = cm[np.ix_(order, order)]
    return cm, sorted_names


def _summarize_seeds(seeds: List[int]) -> str:
    if not seeds:
        return "Seeds: none"
    seeds = sorted(set(seeds))
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"Seeds: s{seeds[0]}-s{seeds[-1]} (n={len(seeds)})"
    return f"Seeds: {', '.join(f's{s}' for s in seeds)} (n={len(seeds)})"


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: pathlib.Path,
    subtitle: str | None = None,
) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized (%)")

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=full_title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = float(np.nanmax(cm_norm)) * 0.6 if cm_norm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j]
            color = "white" if value > thresh else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _guess_data_root(metrics: Dict[str, Any]) -> pathlib.Path | None:
    cfg = metrics.get("config") or {}
    data_root = (cfg.get("data_root") or "").strip()
    if not data_root:
        return None
    return pathlib.Path(data_root)


def _infer_class_names_from_data_root(dataset: str, data_root: pathlib.Path, n_classes: int) -> List[str]:
    import glob

    # Prefer the evaluation split path (mirrors the common plotting conventions in this repo).
    candidates = [
        data_root / dataset / "test_amp",
        data_root / dataset / "train_amp",
    ]
    split_dir = next((p for p in candidates if p.is_dir()), None)
    if split_dir is None:
        return [str(i) for i in range(n_classes)]

    folders = glob.glob(str(split_dir) + "/*/")
    # Keep only folders that actually contain samples, to avoid empty-category drift.
    names: List[str] = []
    for f in folders:
        has_samples = bool(glob.glob(os.path.join(f, "*.mat")))
        if has_samples:
            names.append(pathlib.Path(f).name)
    if len(names) < n_classes:
        return [str(i) for i in range(n_classes)]
    return names[:n_classes]


@dataclass(frozen=True)
class RunKey:
    dataset: str
    model: str
    variant: str


def discover_runs(runs_dir: pathlib.Path) -> Dict[RunKey, List[Tuple[int, pathlib.Path, Dict[str, Any]]]]:
    groups: Dict[RunKey, List[Tuple[int, pathlib.Path, Dict[str, Any]]]] = {}
    for metrics_path in runs_dir.rglob("metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            continue
        dataset = str(metrics.get("dataset", "")).strip()
        model = str(metrics.get("model", "")).strip()
        variant = str(metrics.get("variant", "")).strip()
        seed = int(metrics.get("seed", -1))
        if not dataset or not model or not variant or seed < 0:
            continue
        run_dir = metrics_path.parent
        if not (run_dir / "confusion_matrix.npy").is_file():
            continue
        key = RunKey(dataset=dataset, model=model, variant=variant)
        groups.setdefault(key, []).append((seed, run_dir, metrics))
    # Stable order for downstream plotting
    for key in list(groups.keys()):
        groups[key] = sorted(groups[key], key=lambda t: t[0])
    return groups


def _select_default_aggregate_dir() -> pathlib.Path:
    preferred = pathlib.Path("result/artifacts/aggregate")
    if preferred.is_dir():
        return preferred
    fallback = pathlib.Path("artifacts/aggregate")
    return fallback


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=pathlib.Path, default=pathlib.Path("runs"), help="Root containing run folders.")
    p.add_argument(
        "--aggregate-dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing summary.csv/pareto.csv (default: auto-detect).",
    )
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("result_figure"))
    p.add_argument("--dataset", type=str, default=None, help="Optional dataset filter (e.g., NTU-Fi-HumanID).")
    p.add_argument("--model", type=str, default=None, help="Optional model filter for confusion matrices.")
    p.add_argument("--variant", type=str, default=None, help="Optional variant filter for confusion matrices.")
    p.add_argument(
        "--no-point-labels",
        action="store_true",
        help="Disable point labels in Pareto plots (keeps legend only).",
    )
    args = p.parse_args()

    aggregate_dir = args.aggregate_dir if args.aggregate_dir is not None else _select_default_aggregate_dir()

    # 1) Aggregate plots from CSVs
    summary_path = aggregate_dir / "summary.csv"
    pareto_path = aggregate_dir / "pareto.csv"
    if summary_path.is_file() and pareto_path.is_file():
        summary = _read_csv(summary_path)
        pareto = _read_csv(pareto_path)
        if args.dataset:
            summary = [r for r in summary if str(r.get("dataset", "")).strip() == args.dataset]
            pareto = [r for r in pareto if str(r.get("dataset", "")).strip() == args.dataset]

        datasets = sorted({str(r.get("dataset", "")).strip() for r in summary if str(r.get("dataset", "")).strip()})
        for dataset in datasets:
            summary_ds = [r for r in summary if str(r.get("dataset", "")).strip() == dataset]
            pareto_ds = [r for r in pareto if str(r.get("dataset", "")).strip() == dataset]
            models = sorted({str(r.get("model", "")).strip() for r in summary_ds if str(r.get("model", "")).strip()})
            palette = build_model_palette(models)
            plot_accuracy_bar_meanstd(summary_ds, args.out_dir / dataset / "accuracy_bar_meanstd.png", dataset, palette)
            plot_pareto(
                pareto_ds,
                args.out_dir / dataset / "pareto_batch1.png",
                dataset,
                batch="1",
                palette=palette,
                label_points=not args.no_point_labels,
            )
            plot_pareto(
                pareto_ds,
                args.out_dir / dataset / "pareto_batch64.png",
                dataset,
                batch="64",
                palette=palette,
                label_points=not args.no_point_labels,
            )

    # 2) Confusion matrices from run artifacts
    run_groups = discover_runs(args.runs_dir)
    for key, runs in sorted(run_groups.items(), key=lambda kv: (kv[0].dataset, kv[0].model, kv[0].variant)):
        if args.dataset and key.dataset != args.dataset:
            continue
        if args.model and key.model != args.model:
            continue
        if args.variant and key.variant != args.variant:
            continue

        cms: List[np.ndarray] = []
        seeds: List[int] = []
        metrics_ref: Dict[str, Any] | None = None
        for seed, run_dir, metrics in runs:
            path = run_dir / "confusion_matrix.npy"
            try:
                cm = np.load(path)
            except Exception:
                continue
            if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
                continue
            cms.append(cm.astype(np.int64))
            seeds.append(seed)
            if metrics_ref is None:
                metrics_ref = metrics

        if not cms:
            continue
        cm_sum = np.sum(cms, axis=0)
        n_classes = int(cm_sum.shape[0])
        data_root = _guess_data_root(metrics_ref or {})
        class_names = (
            _infer_class_names_from_data_root(key.dataset, data_root, n_classes) if data_root is not None else []
        )
        if not class_names:
            class_names = [str(i) for i in range(n_classes)]
        if len(class_names) != n_classes:
            class_names = [str(i) for i in range(n_classes)]
        cm_sum, class_names = reorder_confusion_matrix(cm_sum, class_names)

        # Prefer reporting samples/seed using the first available matrix.
        samples_per_seed = int(np.sum(cms[0])) if cms else 0
        subtitle = f"{_summarize_seeds(seeds)}, samples/seed: {samples_per_seed}"
        title = f"{key.dataset} ({key.model}, {key.variant})"
        out_path = args.out_dir / key.dataset / "confusion" / key.variant / f"confusion_{key.model}.png"
        plot_confusion_matrix(cm_sum, class_names, title, out_path, subtitle=subtitle)

    print(f"Wrote figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
