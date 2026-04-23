#!/usr/bin/env python3
"""Render averaged confusion matrices from per-run artifacts.

Each `public/train.py` run writes `confusion_matrix.npy` next to `metrics.json`.
This script aggregates those matrices across seeds and renders publication-ready
plots (row-normalized %, colorbar legend, readable labels).

Examples
--------
source .venv/bin/activate

# All models/variants for one dataset
python public/scripts/plot_confusion_from_runs.py --runs-dir runs --out-dir artifacts/figures --dataset NTU-Fi-HumanID

# Single model/variant
python public/scripts/plot_confusion_from_runs.py --dataset NTU-Fi_HAR --model Mamba --variant selective_on_pool_mean_seq500

# Provide explicit class names (comma-separated) if you need to match a custom ordering
python public/scripts/plot_confusion_from_runs.py --class-names "box,circle,clean,fall,run,walk"
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
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

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


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
    # Prefer the evaluation split path.
    candidates = [
        data_root / dataset / "test_amp",
        data_root / dataset / "train_amp",
        data_root / dataset,  # Widar/UT-HAR style
    ]
    split_dir = next((p for p in candidates if p.is_dir()), None)
    if split_dir is None:
        return [str(i) for i in range(n_classes)]

    # Keep a stable order (sorted).
    names = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if len(names) < n_classes:
        return [str(i) for i in range(n_classes)]
    return names[:n_classes]


def _parse_class_names_arg(arg: str) -> List[str]:
    # Accept either a path to a file (one label per line) or a comma-separated list.
    p = pathlib.Path(arg)
    if p.is_file():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return [s.strip() for s in arg.split(",") if s.strip()]


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
    for key in list(groups.keys()):
        groups[key] = sorted(groups[key], key=lambda t: t[0])
    return groups


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", type=pathlib.Path, default=pathlib.Path("runs"))
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("artifacts/figures"))
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--variant", type=str, default=None)
    p.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Comma-separated labels or path to a newline-delimited text file.",
    )
    args = p.parse_args()

    run_groups = discover_runs(args.runs_dir)
    if not run_groups:
        raise RuntimeError(f"No runs with confusion_matrix.npy found under {args.runs_dir}")

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
            try:
                cm = np.load(run_dir / "confusion_matrix.npy")
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

        if args.class_names:
            class_names = _parse_class_names_arg(args.class_names)
        else:
            data_root = _guess_data_root(metrics_ref or {}) or pathlib.Path("public/Data")
            class_names = _infer_class_names_from_data_root(key.dataset, data_root, n_classes)

        if len(class_names) != n_classes:
            class_names = [str(i) for i in range(n_classes)]

        cm_sum, class_names = reorder_confusion_matrix(cm_sum, class_names)

        samples_per_seed = int(np.sum(cms[0])) if cms else 0
        subtitle = f"{_summarize_seeds(seeds)}, samples/seed: {samples_per_seed}"
        title = f"{key.dataset} ({key.model}, {key.variant})"
        out_path = args.out_dir / key.dataset / "confusion" / key.variant / f"confusion_{key.model}.png"
        plot_confusion_matrix(cm_sum, class_names, title, out_path, subtitle=subtitle)

    print(f"Wrote confusion figures -> {args.out_dir}")


if __name__ == "__main__":
    main()

