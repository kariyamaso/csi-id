#!/usr/bin/env python3
"""Generate averaged confusion matrices for NTU-Fi datasets using checkpoints.

When a checkpoint directory contains files named `<dataset>_<model>_s<seed>.pt`,
the script aggregates the confusion matrices across seeds and saves a single
figure that includes the seed list and sample count (n). A flag can also be
used to keep the per-seed figures if needed.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
from typing import Dict, Iterable, List, Tuple, Optional

# Route matplotlib/font caches into a writable local directory.
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
import torch

# Use Type 42 fonts in PDF/PS for publication-quality vector output
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from util import load_data_n_model


DEFAULT_CKPT_DIRS: Dict[str, pathlib.Path] = {
    "NTU-Fi-HumanID": pathlib.Path("model_pt"),
    "NTU-Fi_HAR": pathlib.Path("model_pt_HAR"),
}

DEFAULT_OUT_DIRS: Dict[str, pathlib.Path] = {
    "NTU-Fi-HumanID": pathlib.Path("figures/ntu_fi_results"),
    "NTU-Fi_HAR": pathlib.Path("figures/ntu_fi_har_results"),
}

MODEL_CHOICES: List[str] = [
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
]


def canonicalize_labels(class_names: List[str]) -> Tuple[List[str], List[int]]:
    """Return labels ordered and formatted for display plus the index mapping used.

    If all labels are numeric, they are sorted by their integer value and
    zero-padded to three digits (e.g., 1 -> 001). Otherwise, the original order
    is preserved.
    """
    if class_names and all(name.isdigit() for name in class_names):
        numeric = [int(name) for name in class_names]
        order = sorted(range(len(numeric)), key=lambda i: numeric[i])
        sorted_names = [f"{numeric[i]:03d}" for i in order]
        return sorted_names, order
    return class_names, list(range(len(class_names)))


def reorder_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Reorder confusion matrix rows/cols to match the canonical label order."""
    sorted_names, order = canonicalize_labels(class_names)
    if order != list(range(len(class_names))):
        cm = cm[np.ix_(order, order)]
    return cm, sorted_names


def normalize_seed_label(seed_label: str | None, checkpoint: pathlib.Path) -> str:
    """Infer a user-friendly seed label even when not explicitly provided."""
    if seed_label:
        return seed_label
    match = re.search(r"_s(\d+)\.pt$", checkpoint.name)
    if match:
        return f"s{match.group(1)}"
    return "best"


def summarize_seed_labels(seed_labels: List[str]) -> str:
    """Format the seed list for figure subtitles."""
    if not seed_labels:
        return "Seeds: none (n=0)"
    unique: List[str] = []
    for lbl in seed_labels:
        if lbl not in unique:
            unique.append(lbl)
    numeric = [int(lbl[1:]) for lbl in unique if re.fullmatch(r"s\d+", lbl)]
    if len(numeric) == len(unique):
        numeric_sorted = sorted(numeric)
        if numeric_sorted == list(range(min(numeric_sorted), max(numeric_sorted) + 1)):
            return f"Seeds: s{numeric_sorted[0]}-s{numeric_sorted[-1]} (n={len(unique)})"
        unique = [f"s{n}" for n in numeric_sorted]
    return f"Seeds: {', '.join(unique)} (n={len(unique)})"


def invert_category_map(dataset) -> List[str]:
    inv = {idx: os.path.basename(os.path.normpath(folder)) for folder, idx in dataset.category.items()}
    return [inv[i] for i in range(len(inv))]


def collect_predictions(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            target = target.to(device)
            logits = model(inputs)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())
            labels.append(target.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    return y_true, y_pred


def build_confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def save_confusion_matrix(
    cm: np.ndarray, class_names: List[str], title: str, out_path: pathlib.Path, subtitle: str | None = None
) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Per-class accuracy")
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

    thresh = cm_norm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def discover_checkpoints(
    dataset: str,
    model: str,
    ckpt_dir: pathlib.Path,
    seeds: Optional[List[int]],
) -> List[Tuple[str, pathlib.Path]]:
    """Return list of (seed_label, checkpoint_path) for the given model/dataset."""
    ckpt_dir = ckpt_dir.expanduser()
    if seeds:
        paths: List[Tuple[str, pathlib.Path]] = []
        for s in seeds:
            path = ckpt_dir / f"{dataset}_{model}_s{s}.pt"
            if path.is_file():
                paths.append((f"s{s}", path))
        if paths:
            return paths
    # Auto-discover seeds if none provided or none found
    pattern = f"{dataset}_{model}_s*.pt"
    discovered: List[Tuple[str, pathlib.Path]] = []
    for path in sorted(ckpt_dir.glob(pattern)):
        match = re.search(r"_s(\d+)\.pt$", path.name)
        seed_label = f"s{match.group(1)}" if match else "auto"
        discovered.append((seed_label, path))
    if discovered:
        return discovered
    # Fallback to unseeded
    fallback = ckpt_dir / f"{dataset}_{model}.pt"
    if fallback.is_file():
        return [("best", fallback)]
    return []


def evaluate_checkpoint(
    dataset_name: str,
    model_name: str,
    checkpoint: pathlib.Path,
    batch_size: int | None = None,
    seed_label: str | None = None,
) -> Tuple[np.ndarray, List[str], float]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    label = normalize_seed_label(seed_label, checkpoint)
    print(f"[info] Dataset={dataset_name}, model={model_name}, seed={label}, ckpt={checkpoint}")
    train_loader, test_loader, model, _ = load_data_n_model(dataset_name, model_name, "./Data/")
    if batch_size:
        test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    y_true, y_pred = collect_predictions(model, test_loader, device)
    acc = float((y_true == y_pred).mean())
    class_names = invert_category_map(test_loader.dataset)
    cm = build_confusion_matrix(y_true, y_pred, num_classes=len(class_names))

    return cm, class_names, acc


def generate_confusion_for_model(
    dataset_name: str,
    model_name: str,
    checkpoints: List[Tuple[str, pathlib.Path]],
    batch_size: int | None,
    out_dir: pathlib.Path,
    save_per_seed: bool = False,
) -> pathlib.Path:
    """Average confusion matrices across seeds and save a single figure."""
    matrices: List[np.ndarray] = []
    accuracies: List[float] = []
    seed_labels: List[str] = []
    class_names_ref: List[str] | None = None

    for seed_label, checkpoint in checkpoints:
        label = normalize_seed_label(seed_label, checkpoint)
        cm, class_names, acc = evaluate_checkpoint(
            dataset_name, model_name, checkpoint, batch_size=batch_size, seed_label=label
        )
        matrices.append(cm)
        accuracies.append(acc)
        seed_labels.append(label)
        if class_names_ref is None:
            class_names_ref = class_names
        elif len(class_names) != len(class_names_ref):
            raise ValueError("Class count mismatch across checkpoints; cannot average confusion matrices.")

        if save_per_seed:
            cm_seed, names_seed = reorder_confusion_matrix(cm, class_names)
            suffix = f"_{label}" if label else ""
            out_path_seed = out_dir / f"confusion_matrix_{model_name}{suffix}.pdf"
            save_confusion_matrix(
                cm_seed,
                names_seed,
                f"{dataset_name} ({model_name} {label})",
                out_path_seed,
                subtitle=summarize_seed_labels([label]),
            )
            print(f"[info] Accuracy: {acc*100:.2f}% - saved {out_path_seed}")

    if not matrices:
        raise RuntimeError("No checkpoints were evaluated; nothing to plot.")

    mean_cm = np.mean(np.stack(matrices, axis=0), axis=0)
    cm_final, class_names_final = reorder_confusion_matrix(mean_cm, class_names_ref or [])
    subtitle = summarize_seed_labels(seed_labels)
    out_path = out_dir / f"confusion_matrix_{model_name}.pdf"
    save_confusion_matrix(cm_final, class_names_final, f"{dataset_name} ({model_name})", out_path, subtitle=subtitle)

    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies)) if len(accuracies) > 1 else 0.0
    print(f"[info] Accuracy (mean+/-std): {mean_acc*100:.2f}% +/- {std_acc*100:.2f}% - saved {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["NTU-Fi-HumanID", "NTU-Fi_HAR"],
        default="NTU-Fi-HumanID",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=["Mamba"],
        help="Models to evaluate (default: Mamba).",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to model checkpoint. Defaults to pre-trained Mamba weights for each dataset.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing checkpoints named <dataset>_<model>_s<seed>.pt.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Specific seeds to evaluate. If omitted, will auto-discover from checkpoint-dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size (uses loader default if omitted).",
    )
    parser.add_argument(
        "--save-per-seed",
        action="store_true",
        help="Also save per-seed confusion matrices (default: only the averaged matrix).",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=None, help="Directory for output figures.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate confusion matrices for both NTU-Fi datasets using default checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["NTU-Fi-HumanID", "NTU-Fi_HAR"] if args.all else [args.dataset]
    for ds in datasets:
        out_dir = args.out_dir or DEFAULT_OUT_DIRS.get(ds, pathlib.Path("figures"))
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = args.checkpoint_dir or DEFAULT_CKPT_DIRS.get(ds, pathlib.Path("checkpoints"))
        for model in args.models:
            ckpts: List[Tuple[str, pathlib.Path]]
            if args.checkpoint:
                ckpts = [(normalize_seed_label(None, args.checkpoint), args.checkpoint)]
            else:
                ckpts = discover_checkpoints(ds, model, ckpt_dir, seeds=args.seeds)
                if not ckpts:
                    print(f"[warn] No checkpoints found for {ds} {model} in {ckpt_dir}")
                    continue
            generate_confusion_for_model(
                ds,
                model,
                ckpts,
                batch_size=args.batch_size,
                out_dir=out_dir,
                save_per_seed=args.save_per_seed,
            )


if __name__ == "__main__":
    main()
