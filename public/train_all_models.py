#!/usr/bin/env python3
"""Train all supported models for a dataset using `public/train.py`.

Writes per-run metrics to `runs/<dataset>/<model>/<variant>/<seed>/metrics.json`.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import List, Optional


DEFAULT_MODELS = {
    "UT_HAR_data": ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101", "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT"],
    "NTU-Fi-HumanID": ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101", "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT", "Mamba"],
    "NTU-Fi_HAR": ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101", "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT", "Mamba"],
    "Widar": ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101", "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT"],
    "APPLIED": ["MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101", "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT", "Mamba"],
}


def run_cmd(cmd: List[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(DEFAULT_MODELS.keys()))
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--config", type=pathlib.Path, default=None, help="Base JSON config passed to train.py.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--log-dir", type=pathlib.Path, default=None, help="Optional log dir (tee stdout/stderr per run).")
    args = p.parse_args()

    models = args.models if args.models else DEFAULT_MODELS[args.dataset]
    for model in models:
        for seed in args.seeds:
            cmd = [args.python, str(pathlib.Path(__file__).with_name("train.py"))]
            if args.config:
                cmd += ["--config", str(args.config)]
            cmd += ["--dataset", args.dataset, "--model", model, "--seed", str(seed)]
            if args.log_dir:
                args.log_dir.mkdir(parents=True, exist_ok=True)
                log_file = args.log_dir / f"{args.dataset}_{model}_s{seed}.log"
                cmd += ["--log-file", str(log_file)]
            run_cmd(cmd)


if __name__ == "__main__":
    main()

