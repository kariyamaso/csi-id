#!/usr/bin/env python3
"""Few-shot fine-tuning on APPLIED using a pretrained SenseFi model.

This script loads a pretrained checkpoint (e.g., NTU-Fi-HumanID_<Model>.pt),
adapts the classifier to 3 classes, optionally freezes the backbone, and
fine-tunes on a k-shot subset per class of the APPLIED training split.
Then it evaluates on the APPLIED test split and optionally saves the tuned
weights.

Examples
--------
source .venv/bin/activate

# Head-only few-shot fine-tune (k=5) from HumanID GRU
python finetune_fewshot.py \
  --model GRU \
  --source-ckpt model_pt/NTU-Fi-HumanID_GRU.pt \
  --k-shot 5 \
  --freeze-backbone \
  --epochs 40 \
  --save-ckpt app_model_pt/APPLIED_GRU_fewshot5.pt

# If you used train_all_models.py --dataset NTU-Fi-HumanID --saveckpt
# you can omit --source-ckpt and it will auto-locate it.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import scipy.io as sio

import dataset as csi_dataset
from util import load_data_n_model
from NTU_Fi_model import (
    NTU_Fi_GRU,
    NTU_Fi_ViT,
    NTU_Fi_Mamba,
    NTU_Fi_LeNet,
    NTU_Fi_MLP,
    NTU_Fi_ResNet,
)


def auto_source_ckpt(model: str, ckpt_dir: Path) -> Path | None:
    pattern = str(ckpt_dir / f"NTU-Fi-HumanID_{model}.pt")
    matches = glob.glob(pattern)
    if matches:
        return Path(matches[0])
    return None


def compute_applied_norm_if_needed(data_root: Path) -> None:
    if os.environ.get("NTU_FI_NORM_MEAN") and os.environ.get("NTU_FI_NORM_STD"):
        return
    split_dir = data_root / "APPLIED" / "train_amp"
    files = sorted(glob.glob(str(split_dir / "*" / "*.mat")))
    if not files:
        return
    total = 0
    s = 0.0
    s2 = 0.0
    for path in files:
        mat = sio.loadmat(path)
        if "CSIamp" not in mat:
            continue
        x = np.asarray(mat["CSIamp"], dtype=np.float64)
        # Accept (S,T) or (1,S,T) or (3,S,T); reduce to (S,T)
        if x.ndim == 3:
            try:
                x = x.reshape(-1, 114, 500).mean(axis=0)
            except Exception:
                continue
        elif x.ndim == 2:
            pass
        else:
            continue
        if x.shape != (114, 500):
            if x.shape == (500, 114):
                x = x.T
            else:
                continue
        s += x.sum()
        s2 += np.square(x, dtype=np.float64).sum()
        total += x.size
    if total > 0:
        mean = s / total
        var = max(s2 / total - mean * mean, 0.0)
        std = float(np.sqrt(var)) if var > 0 else 1.0
        try:
            csi_dataset.set_csi_normalization(float(mean), float(std))
        except Exception:
            os.environ["NTU_FI_NORM_MEAN"] = str(float(mean))
            os.environ["NTU_FI_NORM_STD"] = str(float(std))


def build_fewshot_subset(ds, k: int, seed: int = 42) -> Subset:
    # ds is CSI_Ready_Dataset with attributes data_list and category
    # Build mapping from class id -> indices in dataset
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, path in enumerate(ds.data_list):
        cls_name = path.split("/")[-2]
        cls_id = ds.category[cls_name]
        by_class.setdefault(cls_id, []).append(idx)
    chosen: List[int] = []
    for cls_id, indices in by_class.items():
        if len(indices) <= k:
            chosen.extend(indices)
            continue
        chosen.extend(list(rng.choice(indices, size=k, replace=False)))
    chosen.sort()
    return Subset(ds, chosen)


def freeze_backbone_params(model: nn.Module, model_name: str) -> None:
    # Enable grad only for the classifier head; freeze others
    # Identify classifier per architecture
    head_params: List[nn.Parameter] = []
    for p in model.parameters():
        p.requires_grad = False

    if isinstance(model, NTU_Fi_GRU):
        head_params = list(model.fc.parameters())
    elif isinstance(model, NTU_Fi_ViT):
        # model[2] is ClassificationHead
        head_params = list(model[2].parameters())
    elif isinstance(model, NTU_Fi_Mamba):
        head_params = list(model.head.parameters())
    elif hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Module):
        # ResNet variants
        head_params = list(model.fc.parameters())
    elif isinstance(model, NTU_Fi_MLP):
        # last Linear in classifier
        head_params = list(model.classifier.parameters())
    elif isinstance(model, NTU_Fi_LeNet):
        head_params = list(model.fc.parameters())
    else:
        # Fallback: unfreeze all
        for p in model.parameters():
            p.requires_grad = True
        return

    for p in head_params:
        p.requires_grad = True


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device: torch.device) -> None:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_acc += (preds == labels).float().mean().item()
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_acc / len(loader)
        print(f"Epoch {epoch+1:03d}: acc={epoch_acc:.4f} loss={epoch_loss:.6f}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval().to(device)
    criterion = nn.CrossEntropyLoss()
    total = 0
    loss_sum = 0.0
    correct = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_sum += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
    acc = correct / total if total else 0.0
    loss_avg = loss_sum / total if total else 0.0
    return acc, loss_avg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=[
        "MLP","LeNet","ResNet18","ResNet50","ResNet101","RNN","GRU","LSTM","BiLSTM","CNN+GRU","ViT","Mamba"
    ])
    parser.add_argument("--source-ckpt", type=Path, default=None,
                        help="Pretrained checkpoint to initialize from. If omitted, auto-searches model_pt/NTU-Fi-HumanID_<Model>.pt")
    parser.add_argument("--ckptdir", type=Path, default=Path("model_pt"))
    parser.add_argument("--k-shot", type=int, default=5, help="Number of training samples per class in APPLIED.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone and train classifier head only.")
    parser.add_argument("--save-ckpt", type=Path, default=None, help="Where to save the fine-tuned weights.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("./Data")

    # Ensure APPLIED normalization is set (mean/std)
    compute_applied_norm_if_needed(data_root)

    # Build APPLIED loaders and model with correct output classes (3)
    train_loader, test_loader, model, _ = load_data_n_model("APPLIED", args.model, str(data_root) + "/")

    # Locate source checkpoint if not provided
    src_ckpt = args.source_ckpt or auto_source_ckpt(args.model, args.ckptdir)
    if src_ckpt and src_ckpt.exists():
        print(f"Loading source checkpoint: {src_ckpt}")
        state = torch.load(src_ckpt, map_location=device)
        # Load with strict=False to ignore classifier mismatch
        model.load_state_dict(state, strict=False)
    else:
        print("[warn] No source checkpoint found. Proceeding from random init.")

    # Build few-shot subset for training
    base_ds = train_loader.dataset
    few_ds = build_fewshot_subset(base_ds, k=args.k_shot)
    few_loader = DataLoader(few_ds, batch_size=32, shuffle=True)

    # Optionally freeze backbone
    if args.freeze_backbone:
        freeze_backbone_params(model, args.model)

    # Fine-tune
    train(model, few_loader, epochs=args.epochs, lr=args.lr, device=device)
    acc, loss = evaluate(model, test_loader, device=device)
    print(f"Evaluation on APPLIED test: acc={acc:.4f}, loss={loss:.6f}")

    if args.save_ckpt:
        args.save_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_ckpt)
        print(f"Saved fine-tuned weights -> {args.save_ckpt}")


if __name__ == "__main__":
    main()

