#!/usr/bin/env python3
"""Parse SenseFi training logs and visualize NTU-Fi HumanID results.

This helper collects the per-epoch accuracy/loss curves from every
`logs/train_all/NTU-Fi-HumanID/result/*.log` file (i.e., the runs without the
new SSM model), builds:

1. A bar chart comparing validation accuracies across models.
2. Combined learning-curve plots (accuracy + loss) showing every model.

Usage
-----
source .venv/bin/activate
python plot_ntu_fi_results.py \
    --log-dir logs/train_all/NTU-Fi-HumanID/result \
    --out-dir figures/ntu_fi_results
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
from typing import Dict, List

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


def collect_logs(log_dir: pathlib.Path) -> Dict[str, Dict[str, object]]:
    """Parse every *.log file inside log_dir and return keyed by model name."""
    results: Dict[str, Dict[str, object]] = {}
    for log_path in sorted(log_dir.glob("*.log")):
        model_name = log_path.stem.split("_", 1)[1]
        try:
            results[model_name] = parse_log(log_path)
        except ValueError as err:
            print(f"[warn] {err}")
    if not results:
        raise RuntimeError(f"No valid log files found in {log_dir}")
    return results


def plot_validation_bar(results: Dict[str, Dict[str, object]], out_path: pathlib.Path) -> None:
    """Create a horizontal bar chart of validation accuracies."""
    data = sorted(
        [(model, stats["val_acc"]) for model, stats in results.items()],
        key=lambda item: item[1],
    )
    models, accuracies = zip(*data)
    accuracies_pct = [acc * 100 for acc in accuracies]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(models, accuracies_pct, color=colors, alpha=0.85)
    ax.set_xlabel("Validation Accuracy (%)")
    ax.set_title("NTU-Fi HumanID Validation Accuracy (non-SSM models)")
    ax.set_xlim(0, 105)
    for bar, acc in zip(bars, accuracies_pct):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}%",
            va="center",
        )
    # Add a legend mapping colors to models for clarity.
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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_curves(results: Dict[str, Dict[str, object]], out_path: pathlib.Path) -> None:
    """Plot accuracy + loss learning curves for every model."""
    plt.figure(figsize=(14, 6))
    ax_acc = plt.subplot(1, 2, 1)
    ax_loss = plt.subplot(1, 2, 2)

    for model, stats in results.items():
        epochs = stats["epochs"]
        ax_acc.plot(epochs, [a * 100 for a in stats["train_acc"]], label=model)
        ax_loss.plot(epochs, stats["train_loss"], label=model)

    ax_acc.set_title("Training Accuracy vs. Epoch")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, alpha=0.3)

    ax_loss.set_title("Training Loss vs. Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.grid(True, alpha=0.3)

    handles, labels = ax_acc.get_legend_handles_labels()
    plt.figlegend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        fontsize="small",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_metrics(results: Dict[str, Dict[str, object]], out_path: pathlib.Path) -> None:
    """Dump parsed metrics to JSON for downstream use."""
    serializable = {
        model: {
            **{
                "epochs": stats["epochs"],
                "train_acc": stats["train_acc"],
                "train_loss": stats["train_loss"],
            },
            "val_acc": stats["val_acc"],
            "val_loss": stats["val_loss"],
        }
        for model, stats in results.items()
    }
    with out_path.open("w") as f:
        json.dump(serializable, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=pathlib.Path,
        default=pathlib.Path("logs/train_all/NTU-Fi-HumanID/result"),
        help="Directory containing timestamped SenseFi log files.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("figures/ntu_fi_results"),
        help="Directory to store the generated figures.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = collect_logs(args.log_dir)

    bar_path = args.out_dir / "validation_accuracy_bar.png"
    curves_path = args.out_dir / "training_curves.png"
    metrics_path = args.out_dir / "parsed_metrics.json"

    plot_validation_bar(results, bar_path)
    plot_training_curves(results, curves_path)
    save_metrics(results, metrics_path)
    print(f"Wrote validation chart -> {bar_path}")
    print(f"Wrote training curves -> {curves_path}")
    print(f"Wrote metrics dump -> {metrics_path}")


if __name__ == "__main__":
    main()
