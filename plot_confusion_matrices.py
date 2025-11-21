#!/usr/bin/env python3
"""Generate confusion matrices for NTU-Fi datasets using saved checkpoints."""

from __future__ import annotations

import argparse
import os
import pathlib
from typing import Dict, Iterable, List, Tuple

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

from util import load_data_n_model


DEFAULT_CKPTS: Dict[str, pathlib.Path] = {
    "NTU-Fi-HumanID": pathlib.Path("model_pt/NTU-Fi-HumanID_Mamba.pt"),
    "NTU-Fi_HAR": pathlib.Path("model_pt_HAR/NTU-Fi_HAR_Mamba.pt"),
}

DEFAULT_OUT_DIRS: Dict[str, pathlib.Path] = {
    "NTU-Fi-HumanID": pathlib.Path("figures/ntu_fi_results"),
    "NTU-Fi_HAR": pathlib.Path("figures/ntu_fi_har_results"),
}


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


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, out_path: pathlib.Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Per-class accuracy")
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
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


def run_single(
    dataset_name: str,
    model_name: str,
    checkpoint: pathlib.Path,
    batch_size: int | None = None,
    out_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"[info] Dataset={dataset_name}, model={model_name}, ckpt={checkpoint}")
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

    out_dir = out_dir or DEFAULT_OUT_DIRS.get(dataset_name, pathlib.Path("figures"))
    out_path = out_dir / f"confusion_matrix_{model_name}.pdf"
    save_confusion_matrix(cm, class_names, f"{dataset_name} ({model_name})", out_path)

    print(f"[info] Accuracy: {acc*100:.2f}% — saved {out_path}")
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
        "--model",
        choices=["Mamba"],
        default="Mamba",
        help="Model to evaluate (default: Mamba with saved checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to model checkpoint. Defaults to pre-trained Mamba weights for each dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size (uses loader default if omitted).",
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
        ckpt = args.checkpoint
        if ckpt is None:
            ckpt = DEFAULT_CKPTS.get(ds)
            if ckpt is None:
                raise ValueError(f"No default checkpoint configured for {ds}; please provide --checkpoint.")
        out_dir = args.out_dir or DEFAULT_OUT_DIRS.get(ds)
        run_single(ds, args.model, ckpt, batch_size=args.batch_size, out_dir=out_dir)


if __name__ == "__main__":
    main()
