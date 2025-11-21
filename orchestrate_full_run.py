#!/usr/bin/env python3
"""End-to-end automation: train models, generate plots, and collect artifacts.

This script runs the full pipeline for NTU-Fi datasets:
1) Train all supported models (or a subset) and save checkpoints.
2) Generate bar charts, training curves, and metrics JSON from logs.
3) Render UMAP plots for selected models.
4) Render confusion matrices for Mamba.

All outputs (logs, checkpoints, figures) are written under a single
`--artifact-root` directory to keep runs organized.

Example
-------
source .venv/bin/activate
export NTU_FI_NORM_MEAN=42.3199
export NTU_FI_NORM_STD=4.9802
python orchestrate_full_run.py \\
  --datasets NTU-Fi-HumanID NTU-Fi_HAR \\
  --artifact-root artifacts/full_run_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import sys
from typing import Dict, Iterable, List

# Default model sets per dataset (aligned with util.support)
DEFAULT_MODELS: Dict[str, List[str]] = {
    "NTU-Fi-HumanID": [
        "MLP",
        "LeNet",
        "ResNet18",
        "ResNet50",
        "ResNet101",
        "RNN",
        "GRU",
        "LSTM",
        "BiLSTM",
        "CNN+GRU",
        "ViT",
        "SSM",
        "Mamba",
    ],
    "NTU-Fi_HAR": [
        "MLP",
        "LeNet",
        "ResNet18",
        "ResNet50",
        "ResNet101",
        "RNN",
        "GRU",
        "LSTM",
        "BiLSTM",
        "CNN+GRU",
        "ViT",
        "Mamba",
    ],
}

# UMAP defaults (fastest useful trio)
DEFAULT_UMAP_MODELS = ["GRU", "ViT", "Mamba"]


def run_cmd(cmd: List[str], cwd: pathlib.Path | None = None, env: Dict[str, str] | None = None) -> None:
    """Run a command and stream output; raise on failure."""
    print(f"\n$ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=cwd, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def train_models(dataset: str, models: Iterable[str], python_bin: str, log_dir: pathlib.Path, ckpt_dir: pathlib.Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    cmd = [
        python_bin,
        "train_all_models.py",
        "--dataset",
        dataset,
        "--saveckpt",
        "--ckptdir",
        str(ckpt_dir),
        "--logdir",
        str(log_dir),
    ]
    # train_all_models takes models via a single --models flag followed by values.
    cmd += ["--models", *models]
    run_cmd(cmd)
    print(f"[{dataset}] training completed. Logs -> {log_dir}/{dataset}/{timestamp}_*.log")


def generate_bar_and_curves(dataset: str, python_bin: str, log_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            python_bin,
            "plot_ntu_fi_results.py",
            "--dataset",
            dataset,
            "--log-dir",
            str(log_dir / dataset),
            "--out-dir",
            str(out_dir),
        ]
    )
    # Also write LaTeX table for convenience
    run_cmd(
        [
            python_bin,
            "export_results_table.py",
            "--log-dir",
            str(log_dir / dataset),
            "--out",
            str(out_dir / "results.tex"),
        ]
    )


def generate_confusion(dataset: str, python_bin: str, ckpt_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    ckpt_path = ckpt_dir / f"{dataset}_Mamba.pt"
    run_cmd(
        [
            python_bin,
            "plot_confusion_matrices.py",
            "--dataset",
            dataset,
            "--model",
            "Mamba",
            "--checkpoint",
            str(ckpt_path),
            "--out-dir",
            str(out_dir),
        ]
    )


def generate_umap(
    dataset: str,
    models: Iterable[str],
    python_bin: str,
    ckpt_dir: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    run_cmd(
        [
            python_bin,
            "visualize_umap_embeddings.py",
            "--dataset",
            dataset,
            "--checkpoint-dir",
            str(ckpt_dir),
            "--out-dir",
            str(out_dir),
            "--models",
            *models,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["NTU-Fi-HumanID", "NTU-Fi_HAR"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override model list (applied to all datasets). Defaults to full supported set per dataset.",
    )
    parser.add_argument(
        "--umap-models",
        nargs="+",
        default=DEFAULT_UMAP_MODELS,
        help="Models to include in UMAP plots.",
    )
    parser.add_argument(
        "--artifact-root",
        type=pathlib.Path,
        default=None,
        help="Root directory for all outputs. Defaults to artifacts/<timestamp>.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for all subprocesses.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (reuse existing checkpoints/logs).",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip UMAP generation.",
    )
    parser.add_argument(
        "--skip-confusion",
        action="store_true",
        help="Skip confusion matrix generation.",
    )
    args = parser.parse_args()

    if args.artifact_root is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_root = pathlib.Path("artifacts") / f"full_run_{ts}"
    else:
        artifact_root = args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)

    # Common dirs
    log_dir = artifact_root / "logs"
    ckpt_dir = artifact_root / "checkpoints"
    figures_root = artifact_root / "figures"

    for dataset in args.datasets:
        print(f"\n=== Processing {dataset} ===")
        models = args.models if args.models else DEFAULT_MODELS.get(dataset, [])
        if not models:
            raise ValueError(f"No default models configured for dataset {dataset}")

        # 1) Train + save logs/ckpts
        if not args.skip_train:
            train_models(dataset, models, args.python, log_dir, ckpt_dir)
        else:
            print(f"[{dataset}] skipping training per --skip-train (expects logs in {log_dir/dataset})")

        # 2) Bar chart + training curves + table
        fig_dir = figures_root / dataset
        generate_bar_and_curves(dataset, args.python, log_dir, fig_dir)

        # 3) Confusion matrices (Mamba only)
        if not args.skip_confusion:
            generate_confusion(dataset, args.python, ckpt_dir, fig_dir)
        else:
            print(f"[{dataset}] skipping confusion matrices per --skip-confusion")

        # 4) UMAP embeddings
        if not args.skip_umap:
            generate_umap(dataset, args.umap_models, args.python, ckpt_dir, fig_dir)
        else:
            print(f"[{dataset}] skipping UMAP per --skip-umap")

    print(f"\nAll artifacts written under: {artifact_root.resolve()}")


if __name__ == "__main__":
    main()
