#!/usr/bin/env python3
"""One-command runner: train models, aggregate metrics, and generate plots."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import List


def run_cmd(cmd: List[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def default_config_for(dataset: str) -> pathlib.Path:
    cfg_dir = pathlib.Path(__file__).with_name("configs")
    mapping = {
        "NTU-Fi-HumanID": cfg_dir / "ntu_humanid_all_models.json",
        "NTU-Fi_HAR": cfg_dir / "ntu_humanid_all_models.json",
        "Widar": cfg_dir / "widar_all_models.json",
        "UT_HAR_data": cfg_dir / "ut_har_all_models.json",
        "APPLIED": cfg_dir / "ntu_humanid_all_models.json",
    }
    return mapping.get(dataset, cfg_dir / "ntu_humanid_all_models.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["NTU-Fi-HumanID", "NTU-Fi_HAR"],
        help="Datasets to run.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--config", type=pathlib.Path, default=None, help="Base JSON config (applied to all datasets).")
    p.add_argument(
        "--config-map",
        type=pathlib.Path,
        default=None,
        help="Optional JSON mapping {dataset: config_path} to use per dataset.",
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--log-dir", type=pathlib.Path, default=None)
    p.add_argument("--aggregate-out", type=pathlib.Path, default=pathlib.Path("artifacts/aggregate"))
    args = p.parse_args()

    train_all = pathlib.Path(__file__).with_name("train_all_models.py")
    aggregate = pathlib.Path(__file__).with_name("scripts") / "aggregate_results.py"
    plot_agg = pathlib.Path(__file__).with_name("scripts") / "plot_aggregate.py"

    config_map = {}
    if args.config_map:
        config_map = json.loads(args.config_map.read_text())

    for dataset in args.datasets:
        config = args.config
        if dataset in config_map:
            config = pathlib.Path(config_map[dataset])
        if config is None:
            config = default_config_for(dataset)
        cmd = [args.python, str(train_all), "--dataset", dataset, "--seeds", *[str(s) for s in args.seeds]]
        if config:
            cmd += ["--config", str(config)]
        if args.log_dir:
            cmd += ["--log-dir", str(args.log_dir / dataset)]
        run_cmd(cmd)

    run_cmd([args.python, str(aggregate), "--runs-dir", "runs", "--out-dir", str(args.aggregate_out)])
    for dataset in args.datasets:
        run_cmd([args.python, str(plot_agg), "--in-dir", str(args.aggregate_out), "--out-dir", str(args.aggregate_out / "plots" / dataset), "--dataset", dataset])


if __name__ == "__main__":
    main()
