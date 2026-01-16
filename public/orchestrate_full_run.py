#!/usr/bin/env python3
"""One-command runner: train models, aggregate metrics, and generate plots."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import List


def run_cmd(cmd: List[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["NTU-Fi-HumanID", "NTU-Fi_HAR"],
        help="Datasets to run.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--config", type=pathlib.Path, default=None, help="Base JSON config.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--log-dir", type=pathlib.Path, default=None)
    p.add_argument("--aggregate-out", type=pathlib.Path, default=pathlib.Path("artifacts/aggregate"))
    args = p.parse_args()

    train_all = pathlib.Path(__file__).with_name("train_all_models.py")
    aggregate = pathlib.Path(__file__).with_name("scripts") / "aggregate_results.py"
    plot_agg = pathlib.Path(__file__).with_name("scripts") / "plot_aggregate.py"

    for dataset in args.datasets:
        cmd = [args.python, str(train_all), "--dataset", dataset, "--seeds", *[str(s) for s in args.seeds]]
        if args.config:
            cmd += ["--config", str(args.config)]
        if args.log_dir:
            cmd += ["--log-dir", str(args.log_dir / dataset)]
        run_cmd(cmd)

    run_cmd([args.python, str(aggregate), "--runs-dir", "runs", "--out-dir", str(args.aggregate_out)])
    for dataset in args.datasets:
        run_cmd([args.python, str(plot_agg), "--in-dir", str(args.aggregate_out), "--out-dir", str(args.aggregate_out / "plots" / dataset), "--dataset", dataset])


if __name__ == "__main__":
    main()

