#!/usr/bin/env python3
"""Parse SenseFi training logs and visualize NTU-Fi HumanID or HAR results.

This helper collects the per-epoch accuracy/loss curves from the SenseFi
training logs (e.g., `logs/train_all/<dataset>/result/*.log`) and builds:

1. A bar chart comparing validation accuracies across models.
2. Separate learning-curve plots for training accuracy and training loss.

Usage
-----
source .venv/bin/activate
python plot_ntu_fi_results.py --dataset NTU-Fi-HumanID
python plot_ntu_fi_results.py --dataset NTU-Fi_HAR --log-dir NTU-Fi_HAR
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

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


def parse_log(path: pathlib.Path) -> Dict[str, object]:
    """Return training curves + validation metrics for a single log file."""
    epoch_re = re.compile(
        r"Epoch:(?P<epoch>\d+), Accuracy:(?P<acc>[0-9\.]+),Loss:(?P<loss>[-0-9\.eE]+)"
    )
    val_re = re.compile(
        r"validation accuracy:(?P<acc>[0-9\.]+), loss:(?P<loss>[-0-9\.eE]+)", re.IGNORECASE
    )

    epochs: List[int] = []
    accs: List[float] = []
    losses: List[float] = []
    val_acc = None
    val_loss = None

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            match = epoch_re.match(line)
            if match:
                epochs.append(int(match.group("epoch")))
                accs.append(float(match.group("acc")))
                losses.append(float(match.group("loss")))
                continue
            match = val_re.match(line)
            if match:
                val_acc = float(match.group("acc"))
                val_loss = float(match.group("loss"))

    if not epochs:
        raise ValueError(f"No epoch data found in {path}")
    if val_acc is None or val_loss is None:
        raise ValueError(f"No validation metrics found in {path}")

    # Ensure curves are ordered by epoch.
    zipped = sorted(zip(epochs, accs, losses), key=lambda t: t[0])
    epochs, accs, losses = map(list, zip(*zipped))

    return {
        "epochs": epochs,
        "train_acc": accs,
        "train_loss": losses,
        "val_acc": val_acc,
        "val_loss": val_loss,
    }


def _parse_model_and_seed(stem: str) -> Tuple[str, Optional[int]]:
    """Infer base model name and optional seed from a log filename stem.

    Expected patterns:
    - <timestamp>_<Model>
    - <timestamp>_s<seed>_<Model>
    """
    parts = stem.split("_", 1)
    if len(parts) == 1:
        return stem, None
    rest = parts[1]
    if rest.startswith("s") and "_" in rest:
        seed_part, model_part = rest.split("_", 1)
        if seed_part[1:].isdigit():
            return model_part, int(seed_part[1:])
    return rest, None


def collect_logs(
    log_dir: pathlib.Path,
    only_prefix: str | None = None,
    exclude_models: List[str] | None = None,
) -> List[Dict[str, object]]:
    """Parse *.log files inside log_dir with optional filename prefix filter."""
    runs: List[Dict[str, object]] = []
    for log_path in sorted(log_dir.glob("*.log")):
        stem = log_path.stem
        if only_prefix and not stem.startswith(only_prefix.rstrip("_")):
            continue
        model_name, seed = _parse_model_and_seed(stem)
        if exclude_models and model_name in exclude_models:
            continue
        try:
            stats = parse_log(log_path)
        except ValueError as err:
            print(f"[warn] {err}")
            continue
        stats["model"] = model_name
        stats["seed"] = seed
        stats["log_path"] = str(log_path)
        runs.append(stats)
    if not runs:
        prefix_msg = f" with prefix {only_prefix}" if only_prefix else ""
        raise RuntimeError(f"No valid log files found in {log_dir}{prefix_msg}")
    return runs


def aggregate_by_model(runs: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """Group run stats by base model and compute mean/std of validation accuracy."""
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in runs:
        grouped.setdefault(r["model"], []).append(r)
    aggregated: Dict[str, Dict[str, object]] = {}
    for model, entries in grouped.items():
        val_accs = [e["val_acc"] for e in entries]
        val_losses = [e["val_loss"] for e in entries]
        mean_acc = float(np.mean(val_accs))
        std_acc = float(np.std(val_accs))
        mean_loss = float(np.mean(val_losses))
        std_loss = float(np.std(val_losses))
        best_run = max(entries, key=lambda e: e["val_acc"])
        aggregated[model] = {
            "mean_val_acc": mean_acc,
            "std_val_acc": std_acc,
            "mean_val_loss": mean_loss,
            "std_val_loss": std_loss,
            "runs": entries,
            "best_run": best_run,
        }
    return aggregated


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


def build_model_palette(models: List[str]) -> Dict[str, tuple]:
    palette: Dict[str, tuple] = {}
    if "Mamba" in models:
        palette["Mamba"] = MAMBA_COLOR
    remaining_models = [m for m in sorted(models) if m not in palette]
    for model, color in zip(remaining_models, _cycle_colors(len(remaining_models))):
        palette[model] = color
    return palette


def summarize_seed_count(aggregated: Dict[str, Dict[str, object]]) -> str:
    """Return a short seed-count string for figure subtitles."""
    seen: set[int] = set()
    total_runs = 0
    for stats in aggregated.values():
        for run in stats["runs"]:
            total_runs += 1
            seed = run.get("seed")
            if seed is not None:
                seen.add(int(seed))
    if seen:
        return f"Seeds: n={len(seen)}"
    if total_runs:
        return f"Runs: n={total_runs}"
    return ""


@dataclass(frozen=True)
class DatasetProfile:
    label: str
    default_log_dir: pathlib.Path
    default_out_dir: pathlib.Path
    bar_xlim: Tuple[float, float] | None = None


DATASET_PROFILES: Dict[str, DatasetProfile] = {
    "NTU-Fi-HumanID": DatasetProfile(
        label="NTU-Fi HumanID",
        default_log_dir=pathlib.Path("logs/train_all/NTU-Fi-HumanID/result"),
        default_out_dir=pathlib.Path("figures/ntu_fi_results"),
        bar_xlim=(0, 115),
    ),
    "NTU-Fi_HAR": DatasetProfile(
        label="NTU-Fi HAR",
        default_log_dir=pathlib.Path("logs/train_all/NTU-Fi_HAR/result"),
        default_out_dir=pathlib.Path("figures/ntu_fi_har_results"),
        bar_xlim=(0, 115),
    ),
}


def plot_validation_bar(
    results: Dict[str, Dict[str, object]],
    out_path: pathlib.Path,
    palette: Dict[str, tuple],
    dataset_label: str,
    xlim: Tuple[float, float] | None,
    subtitle: str | None = None,
) -> None:
    """Create a horizontal bar chart of validation accuracies."""
    data = sorted(
        [(model, stats["val_acc"]) for model, stats in results.items()],
        key=lambda item: item[1],
    )
    models, accuracies = zip(*data)
    accuracies_pct = [acc * 100 for acc in accuracies]

    colors = [palette[model] for model in models]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(models, accuracies_pct, color=colors, alpha=0.9)
    ax.set_xlabel("Validation Accuracy (%)")
    title = f"{dataset_label} Validation Accuracy"
    if subtitle:
        title = f"{title}\n{subtitle}"
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        upper = max(accuracies_pct) if accuracies_pct else 100
        right = max(upper + 15, 115)
        ax.set_xlim(0, right)
    for bar, acc in zip(bars, accuracies_pct):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            va="center",
        )
    legend = ax.legend(
        bars,
        models,
        title="Model",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        fontsize="small",
        borderaxespad=0,
    )
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_validation_bar_meanstd(
    aggregated: Dict[str, Dict[str, object]],
    out_path: pathlib.Path,
    palette: Dict[str, tuple],
    dataset_label: str,
    xlim: Tuple[float, float] | None,
    subtitle: str | None = None,
) -> None:
    """Create a horizontal bar chart of mean/std validation accuracies."""
    data = sorted(
        [
            (model, stats["mean_val_acc"], stats["std_val_acc"])
            for model, stats in aggregated.items()
        ],
        key=lambda item: item[1],
        reverse=True,  # sort by accuracy descending (top to bottom)
    )
    models, means, stds = zip(*data)
    means_pct = [m * 100 for m in means]
    stds_pct_raw = [s * 100 for s in stds]
    # Clip error bars so mean+std does not exceed 100%
    stds_pct = [min(std, max(0.0, 100.0 - mean)) for mean, std in zip(means_pct, stds_pct_raw)]
    colors = [palette.get(model, "#377eb8") for model in models]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(models, means_pct, xerr=stds_pct, color=colors, alpha=0.9, capsize=6)
    ax.set_xlabel("Validation Accuracy (%)")
    title = f"{dataset_label} Validation Accuracy (mean \u00b1 std)"
    if subtitle:
        title = f"{title}\n{subtitle}"
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()

    for bar, mean, std in zip(bars, means_pct, stds_pct):
        ax.text(
            mean + std + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.1f}\u00b1{std:.1f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    fig.tight_layout(rect=(0, 0, 0.94, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_training_accuracy(
    results: Dict[str, Dict[str, object]],
    out_path: pathlib.Path,
    palette: Dict[str, tuple],
    dataset_label: str,
) -> None:
    """Plot training accuracy for every model in a single figure.

    Legend is placed inside the axes at bottom-right to avoid being cropped.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, stats in results.items():
        epochs = stats["epochs"]
        color = palette[model]
        ax.plot(epochs, [a * 100 for a in stats["train_acc"]], label=model, color=color)
    ax.set_title(f"{dataset_label} Training Accuracy vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize="small", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_training_loss(
    results: Dict[str, Dict[str, object]],
    out_path: pathlib.Path,
    palette: Dict[str, tuple],
    dataset_label: str,
) -> None:
    """Plot training loss for every model in a single figure.

    Legend is placed inside the axes at top-right to avoid being cropped.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, stats in results.items():
        epochs = stats["epochs"]
        color = palette[model]
        ax.plot(epochs, stats["train_loss"], label=model, color=color)
    ax.set_title(f"{dataset_label} Training Loss vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_training_curves_combined(
    results: Dict[str, Dict[str, object]],
    out_path: pathlib.Path,
    palette: Dict[str, tuple],
    dataset_label: str,
) -> None:
    """Plot training accuracy and loss side-by-side in one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_acc, ax_loss = axes
    # Accuracy subplot
    for model, stats in results.items():
        epochs = stats["epochs"]
        color = palette[model]
        ax_acc.plot(epochs, [a * 100 for a in stats["train_acc"]], label=model, color=color)
    ax_acc.set_title(f"{dataset_label} Training Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, alpha=0.3)
    # Loss subplot
    for model, stats in results.items():
        epochs = stats["epochs"]
        color = palette[model]
        ax_loss.plot(epochs, stats["train_loss"], label=model, color=color)
    ax_loss.set_title(f"{dataset_label} Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.grid(True, alpha=0.3)
    # Single shared legend
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize="small")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def save_metrics(
    best_results: Dict[str, Dict[str, object]],
    aggregated: Dict[str, Dict[str, object]],
    out_path: pathlib.Path,
) -> None:
    """Dump parsed metrics (best runs + aggregated stats) to JSON for downstream use."""
    serializable_best = {
        model: {
            "seed": stats.get("seed"),
            "val_acc": stats["val_acc"],
            "val_loss": stats["val_loss"],
            "epochs": stats["epochs"],
            "train_acc": stats["train_acc"],
            "train_loss": stats["train_loss"],
            "log_path": stats.get("log_path"),
        }
        for model, stats in best_results.items()
    }
    serializable_agg = {
        model: {
            "mean_val_acc": stats["mean_val_acc"],
            "std_val_acc": stats["std_val_acc"],
            "mean_val_loss": stats["mean_val_loss"],
            "std_val_loss": stats["std_val_loss"],
            "seeds": [r.get("seed") for r in stats["runs"]],
            "runs": [
                {
                    "seed": r.get("seed"),
                    "val_acc": r["val_acc"],
                    "val_loss": r["val_loss"],
                    "log_path": r.get("log_path"),
                }
                for r in stats["runs"]
            ],
        }
        for model, stats in aggregated.items()
    }
    payload = {"best_runs": serializable_best, "aggregated": serializable_agg}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PROFILES.keys()),
        default="NTU-Fi-HumanID",
        help="Which dataset's defaults to use for plotting metadata.",
    )
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing SenseFi log files. Defaults to the dataset profile.",
    )
    parser.add_argument(
        "--only-prefix",
        type=str,
        default=None,
        help="Only include logs whose basename starts with this prefix (e.g., 20251115-183157_).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Model names to exclude (e.g., --exclude SSM). Can be repeated.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to store the generated figures. Defaults to the dataset profile.",
    )
    args = parser.parse_args()

    profile = DATASET_PROFILES[args.dataset]
    log_dir = args.log_dir if args.log_dir else profile.default_log_dir
    out_dir = args.out_dir if args.out_dir else profile.default_out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_logs(log_dir, only_prefix=args.only_prefix, exclude_models=args.exclude)
    aggregated = aggregate_by_model(runs)
    # Use the best run per model (by val acc) for training curves
    best_results: Dict[str, Dict[str, object]] = {m: stats["best_run"] for m, stats in aggregated.items()}
    # Use mean accuracy (instead of argmax) for the primary bar chart
    mean_results: Dict[str, Dict[str, float]] = {
        m: {"val_acc": stats["mean_val_acc"]} for m, stats in aggregated.items()
    }

    bar_path = out_dir / "validation_accuracy_bar.png"
    bar_meanstd_path = out_dir / "validation_accuracy_bar_meanstd.png"
    acc_path = out_dir / "training_accuracy.png"
    loss_path = out_dir / "training_loss.png"
    metrics_path = out_dir / "parsed_metrics.json"
    combined_path = out_dir / "training_curves.png"

    palette = build_model_palette(list(best_results.keys()))
    subtitle = summarize_seed_count(aggregated)
    plot_validation_bar(mean_results, bar_path, palette, profile.label, profile.bar_xlim, subtitle=subtitle or None)
    plot_validation_bar_meanstd(
        aggregated, bar_meanstd_path, palette, profile.label, profile.bar_xlim, subtitle=subtitle or None
    )
    plot_training_accuracy(best_results, acc_path, palette, profile.label)
    plot_training_loss(best_results, loss_path, palette, profile.label)
    save_metrics(best_results, aggregated, metrics_path)
    plot_training_curves_combined(best_results, combined_path, palette, profile.label)
    print(f"[{profile.label}] Wrote validation chart -> {bar_path}")
    print(f"[{profile.label}] Wrote validation chart (mean±std) -> {bar_meanstd_path}")
    print(f"[{profile.label}] Wrote training accuracy -> {acc_path}")
    print(f"[{profile.label}] Wrote training loss -> {loss_path}")
    print(f"[{profile.label}] Wrote metrics dump -> {metrics_path}")
    print(f"[{profile.label}] Wrote combined training curves -> {combined_path}")


if __name__ == "__main__":
    main()
