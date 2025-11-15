#!/usr/bin/env python3
"""Train every SenseFi model sequentially on a chosen dataset.

Usage:
  source .venv/bin/activate
  export NTU_FI_NORM_MEAN=38.8246
  export NTU_FI_NORM_STD=5.9803
  python train_all_models.py --dataset NTU-Fi-HumanID

All stdout/stderr from each `run.py` invocation is tee'd into
`logs/train_all/<dataset>/<timestamp>_<model>.log`.

Optionally save trained checkpoints usable by downstream tools (e.g. UMAP
visualization) with:

  python train_all_models.py --dataset NTU-Fi-HumanID --saveckpt --ckptdir model_pt

This stores files as `<ckptdir>/<dataset>_<model>.pt`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import sys
from typing import Iterable, List

# Full union of models across datasets
ALL_MODELS: List[str] = [
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
]


def supported_models_for(dataset: str) -> List[str]:
    """Return models compatible with the given dataset per util.load_data_n_model."""
    base = [
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
    ]
    if dataset == "NTU-Fi-HumanID":
        return base + ["SSM", "Mamba"]
    if dataset == "NTU-Fi_HAR":
        return base + ["Mamba"]  # SSM not supported
    if dataset == "APPLIED":
        return base + ["Mamba"]  # SSM not supported
    if dataset in ("UT_HAR_data", "Widar"):
        return base  # No SSM/Mamba
    return base


def run_command(cmd: List[str], log_path: pathlib.Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="NTU-Fi-HumanID",
        choices=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar", "APPLIED"],
        help="Dataset argument passed to run.py.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to train. Defaults to all supported models for the dataset.",
    )
    parser.add_argument(
        "--saveckpt",
        action="store_true",
        help="Save trained weights for each model to <ckptdir>/<dataset>_<model>.pt",
    )
    parser.add_argument(
        "--ckptdir",
        default="model_pt",
        help="Directory to store checkpoints when --saveckpt is set.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use when invoking run.py.",
    )
    parser.add_argument(
        "--logdir",
        default="logs/train_all",
        help="Directory to store stdout/stderr logs.",
    )
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Determine model list, filtering out unsupported ones
    default_models = supported_models_for(args.dataset)
    models: List[str] = args.models if args.models else default_models
    unsupported = [m for m in models if m not in supported_models_for(args.dataset)]
    if unsupported:
        print(f"[warn] Skipping unsupported for {args.dataset}: {', '.join(unsupported)}")
        models = [m for m in models if m not in unsupported]
    if not models:
        print("[error] No models to train after filtering.")
        sys.exit(1)
    failures: List[str] = []
    for model in models:
        log_path = pathlib.Path(args.logdir) / args.dataset / f"{timestamp}_{model}.log"
        cmd = [
            args.python,
            "run.py",
            "--dataset",
            args.dataset,
            "--model",
            model,
        ]
        if args.saveckpt:
            ckptdir = pathlib.Path(args.ckptdir)
            ckptdir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckptdir / f"{args.dataset}_{model}.pt"
            cmd += ["--save-ckpt", str(ckpt_path)]
        print(f"[{model}] training started… log -> {log_path}")
        ret = run_command(cmd, log_path)
        if ret == 0:
            print(f"[{model}] finished successfully.")
        else:
            failures.append(model)
            print(f"[{model}] failed with exit code {ret}. See {log_path}")

    if failures:
        print(f"\nCompleted with failures: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
