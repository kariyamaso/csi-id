"""Retrain a curated set of models on CPU to produce fresh checkpoints matching
the current data pipeline. Previous `model_pt/` and `model_pt_HAR/` checkpoints
were stale (20-30% acc) and do not match reported accuracy.

Saves weights to demo/ckpt/<dataset>/<model>.pt plus a per-model metrics.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "public"))
sys.path.insert(0, str(REPO / "scripts"))

from wisense.util import load_data_n_model  # noqa: E402
from cpu_mamba import CpuGenericMambaClassifier  # noqa: E402

DEFAULT_MODELS = ["MLP", "LeNet", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ResNet18", "ViT"]
MAX_EPOCHS = {
    "MLP": 30, "LeNet": 30, "ResNet18": 30, "ResNet50": 30, "ResNet101": 30,
    "RNN": 40, "GRU": 22, "LSTM": 22, "BiLSTM": 22, "CNN+GRU": 40, "ViT": 40,
    "Mamba": 12,
}
PATIENCE = 5  # early stop


def _build_model(dataset_name: str, model_name: str, root: str):
    if model_name == "Mamba":
        # CPU Mamba needs small batch — the selective scan is a Python loop, so
        # peak memory is O(B * L * d_inner * d_state) ≈ 1 GB per 16-sample batch.
        num_classes = {"NTU-Fi-HumanID": 14, "NTU-Fi_HAR": 6}[dataset_name]
        train_loader, test_loader, _, _ = load_data_n_model(
            dataset_name=dataset_name, model_name="GRU", root=root,
            seq_len=500, batch_train=16, batch_test=16, num_workers=0, seed=0,
        )
        model = CpuGenericMambaClassifier(
            num_classes=num_classes, in_features=342,
            d_model=128, depth=2, d_state=16, d_conv=4, expand=2, dropout=0.1,
        )
        return train_loader, test_loader, model
    train_loader, test_loader, model, _ = load_data_n_model(
        dataset_name=dataset_name, model_name=model_name, root=root,
        seq_len=500, batch_train=64, batch_test=64, num_workers=0, seed=0,
    )
    return train_loader, test_loader, model


def train_one(dataset_name: str, model_name: str, root: str, out_dir: Path, device: torch.device) -> dict:
    torch.manual_seed(0)
    train_loader, test_loader, model = _build_model(dataset_name, model_name, root)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    max_ep = MAX_EPOCHS.get(model_name, 30)
    best_acc, best_loss, best_state, stale = -1.0, float("inf"), None, 0
    history = []
    t0 = time.time()
    for epoch in range(1, max_ep + 1):
        model.train()
        n_tot, n_ok, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(device).contiguous()
            y = y.to(device).long()
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            n_tot += y.numel()
            n_ok += int((logits.argmax(-1) == y).sum())
            loss_sum += float(loss) * y.numel()

        model.eval()
        c, n, vloss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device).contiguous()
                y = y.to(device).long()
                logits = model(x)
                c += int((logits.argmax(-1) == y).sum())
                n += y.numel()
                vloss += float(F.cross_entropy(logits, y)) * y.numel()
        test_acc = c / max(1, n)
        test_loss = vloss / max(1, n)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": loss_sum / n_tot, "train_acc": n_ok / n_tot, "test_acc": test_acc, "test_loss": test_loss})
        print(f"  [{dataset_name}/{model_name}] ep{epoch:2d}  train {n_ok/n_tot*100:5.1f}%  test {test_acc*100:5.1f}%  ({elapsed:.1f}s)")
        if test_loss < best_loss - 1e-4:
            best_loss = test_loss
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE and best_acc > 0:
                print(f"  [{dataset_name}/{model_name}] early stop at epoch {epoch} (best acc {best_acc*100:.2f}%)")
                break

    if best_state is None:
        best_state = model.state_dict()

    ckpt_dir = out_dir / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / f"{model_name}.pt")
    meta = {
        "dataset": dataset_name,
        "model": model_name,
        "best_test_acc": best_acc,
        "best_test_loss": best_loss,
        "epochs_ran": len(history),
        "elapsed_sec": time.time() - t0,
        "history": history,
    }
    (ckpt_dir / f"{model_name}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["NTU-Fi_HAR", "NTU-Fi-HumanID"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--root", default="Data/")
    ap.add_argument("--out", default=str(REPO / "demo" / "ckpt"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    device = torch.device("cpu")
    summary = []
    for ds in args.datasets:
        for m in args.models:
            try:
                meta = train_one(ds, m, args.root, out_dir, device)
                summary.append({k: meta[k] for k in ("dataset", "model", "best_test_acc", "epochs_ran", "elapsed_sec")})
            except Exception as exc:  # noqa: BLE001
                print(f"[{ds}/{m}] FAILED: {exc}")
                summary.append({"dataset": ds, "model": m, "error": str(exc)})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
