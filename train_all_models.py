#!/usr/bin/env python3
"""Train every SenseFi model sequentially on a chosen dataset.

Usage:
  source .venv/bin/activate
  export NTU_FI_NORM_MEAN=38.8246
  export NTU_FI_NORM_STD=5.9803
  python train_all_models.py --dataset NTU-Fi-HumanID

All stdout/stderr from each `run.py` invocation is tee'd into
`logs/train_all/<dataset>/<timestamp>_<model>.log`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import sys
from typing import Iterable, List

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
        choices=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar"],
        help="Dataset argument passed to run.py.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        help="Subset of models to train. Defaults to all SenseFi models.",
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
    failures: List[str] = []
    for model in args.models:
        log_path = pathlib.Path(args.logdir) / args.dataset / f"{timestamp}_{model}.log"
        cmd = [
            args.python,
            "run.py",
            "--dataset",
            args.dataset,
            "--model",
            model,
        ]
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
