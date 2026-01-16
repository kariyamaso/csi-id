#!/usr/bin/env python3
"""Aggregate per-run metrics into summary and Pareto CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def _mean(values: Iterable[Any]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _mean_std(values: Iterable[Any]) -> Tuple[float | None, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = float(statistics.mean(vals))
    std = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return mean, std


def _load_metrics(runs_dir: pathlib.Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in runs_dir.rglob("metrics.json"):
        try:
            data = json.loads(path.read_text())
            entries.append(data)
        except Exception:
            continue
    return entries


def _write_csv(path: pathlib.Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=pathlib.Path,
        default=pathlib.Path("runs"),
        help="Directory containing per-run metrics.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/aggregate"),
        help="Output directory for CSV summaries.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset filter.")
    parser.add_argument("--model", type=str, default=None, help="Optional model filter.")
    args = parser.parse_args()

    entries = _load_metrics(args.runs_dir)
    if args.dataset:
        entries = [e for e in entries if e.get("dataset") == args.dataset]
    if args.model:
        entries = [e for e in entries if e.get("model") == args.model]

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = (entry.get("dataset", ""), entry.get("model", ""), entry.get("variant", ""))
        grouped[key].append(entry)

    summary_rows: List[Dict[str, Any]] = []
    pareto_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []

    for (dataset, model, variant), rows in grouped.items():
        acc_mean, acc_std = _mean_std(r.get("acc") for r in rows)
        loss_mean, loss_std = _mean_std(r.get("loss") for r in rows)
        params_total = _mean(r.get("params_total") for r in rows)
        params_trainable = _mean(r.get("params_trainable") for r in rows)
        flops_forward = _mean(r.get("flops_forward") for r in rows)
        latency_b1 = _mean(r.get("latency_ms_batch1") for r in rows)
        latency_b64 = _mean(r.get("latency_ms_batch64") for r in rows)
        peak_mem = _mean(r.get("peak_gpu_mem_mb") for r in rows)
        train_epoch = _mean(r.get("train_time_sec_epoch") for r in rows)
        train_total = _mean(r.get("train_time_total_sec") for r in rows)
        n = len(rows)

        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "variant": variant,
                "n": n,
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "loss_mean": loss_mean,
                "loss_std": loss_std,
                "params_total_mean": params_total,
                "params_trainable_mean": params_trainable,
                "flops_forward_mean": flops_forward,
                "latency_ms_batch1_mean": latency_b1,
                "latency_ms_batch64_mean": latency_b64,
                "peak_gpu_mem_mb_mean": peak_mem,
                "train_time_sec_epoch_mean": train_epoch,
                "train_time_total_sec_mean": train_total,
            }
        )

        pareto_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "variant": variant,
                "acc_mean": acc_mean,
                "latency_ms_batch1_mean": latency_b1,
                "latency_ms_batch64_mean": latency_b64,
                "params_total_mean": params_total,
                "flops_forward_mean": flops_forward,
            }
        )

        example = rows[0] if rows else {}
        ablation_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "variant": variant,
                "n": n,
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "loss_mean": loss_mean,
                "loss_std": loss_std,
                "mamba_selective": example.get("mamba_selective"),
                "pooling": example.get("pooling"),
                "seq_len": example.get("seq_len"),
                "shuffle_subcarriers": example.get("shuffle_subcarriers"),
                "shuffle_antennas": example.get("shuffle_antennas"),
                "train_fraction": example.get("train_fraction"),
                "noise": example.get("noise"),
                "noise_p": example.get("noise_p"),
                "val_noise": example.get("val_noise"),
            }
        )

    summary_rows.sort(key=lambda r: (r["dataset"], r["model"], r["variant"]))
    pareto_rows.sort(key=lambda r: (r["dataset"], r["model"], r["variant"]))
    ablation_rows.sort(key=lambda r: (r["dataset"], r["model"], r["variant"]))

    _write_csv(
        args.out_dir / "summary.csv",
        summary_rows,
        [
            "dataset",
            "model",
            "variant",
            "n",
            "acc_mean",
            "acc_std",
            "loss_mean",
            "loss_std",
            "params_total_mean",
            "params_trainable_mean",
            "flops_forward_mean",
            "latency_ms_batch1_mean",
            "latency_ms_batch64_mean",
            "peak_gpu_mem_mb_mean",
            "train_time_sec_epoch_mean",
            "train_time_total_sec_mean",
        ],
    )
    _write_csv(
        args.out_dir / "pareto.csv",
        pareto_rows,
        [
            "dataset",
            "model",
            "variant",
            "acc_mean",
            "latency_ms_batch1_mean",
            "latency_ms_batch64_mean",
            "params_total_mean",
            "flops_forward_mean",
        ],
    )
    _write_csv(
        args.out_dir / "ablation.csv",
        ablation_rows,
        [
            "dataset",
            "model",
            "variant",
            "n",
            "acc_mean",
            "acc_std",
            "loss_mean",
            "loss_std",
            "mamba_selective",
            "pooling",
            "seq_len",
            "shuffle_subcarriers",
            "shuffle_antennas",
            "train_fraction",
            "noise",
            "noise_p",
            "val_noise",
        ],
    )

    print(f"Wrote {len(summary_rows)} rows -> {args.out_dir}")


if __name__ == "__main__":
    main()

