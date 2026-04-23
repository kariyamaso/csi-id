"""Build a self-contained JSON + PNG bundle for the demo web UI.

Runs every trained model on a small number of held-out samples from each
dataset, collects top-k softmax predictions, and exports:
    demo/data.json       - metadata, class list, model list, samples, predictions
    demo/heatmaps/*.png  - CSI amplitude heatmaps for each demo sample

Usage:
    python3 scripts/build_demo_assets.py --samples-per-class 2
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "public"))

from wisense.models.ntu_fi_model import (  # noqa: E402
    NTU_Fi_BiLSTM,
    NTU_Fi_CNN_GRU,
    NTU_Fi_GRU,
    NTU_Fi_LSTM,
    NTU_Fi_MLP,
    NTU_Fi_Mamba,
    NTU_Fi_RNN,
    NTU_Fi_ResNet101,
    NTU_Fi_ResNet18,
    NTU_Fi_ResNet50,
    NTU_Fi_LeNet,
    NTU_Fi_ViT,
)

CSI_MEAN = 42.3199
CSI_STD = 4.9802
SEQ_LEN = 500

DATASET_SPECS = {
    "NTU-Fi-HumanID": {
        # SenseFi convention: train the model on test_amp, evaluate on train_amp.
        # Sample demo examples from the held-out split (train_amp).
        "sample_root": REPO / "Data" / "NTU-Fi-HumanID" / "train_amp",
        "class_root": REPO / "Data" / "NTU-Fi-HumanID" / "test_amp",
        "ckpt_dir": REPO / "model_pt",
        "ckpt_prefix": "NTU-Fi-HumanID_",
        "num_classes": 14,
        "task": "Person re-identification (14 subjects)",
    },
    "NTU-Fi_HAR": {
        "sample_root": REPO / "Data" / "NTU-Fi_HAR" / "test_amp",
        "class_root": REPO / "Data" / "NTU-Fi_HAR" / "train_amp",
        "ckpt_dir": REPO / "model_pt_HAR",
        "ckpt_prefix": "NTU-Fi_HAR_",
        "num_classes": 6,
        "task": "Human activity recognition (6 activities)",
    },
}

MAMBA_CFG = dict(d_model=256, depth=4, d_state=64, d_conv=4, expand=2, dropout=0.1)
VIT_CFG = dict(in_channels=1, patch_size_w=9, patch_size_h=25, emb_size=225, img_size=171000, depth=1)


def build_model(name: str, num_classes: int):
    if name == "MLP":
        return NTU_Fi_MLP(num_classes)
    if name == "LeNet":
        return NTU_Fi_LeNet(num_classes)
    if name == "ResNet18":
        return NTU_Fi_ResNet18(num_classes)
    if name == "ResNet50":
        return NTU_Fi_ResNet50(num_classes)
    if name == "ResNet101":
        return NTU_Fi_ResNet101(num_classes)
    if name == "RNN":
        return NTU_Fi_RNN(num_classes)
    if name == "GRU":
        return NTU_Fi_GRU(num_classes)
    if name == "LSTM":
        return NTU_Fi_LSTM(num_classes)
    if name == "BiLSTM":
        return NTU_Fi_BiLSTM(num_classes)
    if name == "CNN+GRU":
        return NTU_Fi_CNN_GRU(num_classes)
    if name == "ViT":
        return NTU_Fi_ViT(num_classes=num_classes, **VIT_CFG)
    if name == "Mamba":
        return NTU_Fi_Mamba(num_classes, pooling="mean", selective=True, **MAMBA_CFG)
    raise ValueError(f"Unknown model {name}")


KNOWN_MODELS = [
    "Mamba",
    "GRU",
    "LeNet",
    "BiLSTM",
    "LSTM",
    "MLP",
    "RNN",
    "CNN+GRU",
    "ResNet18",
    "ResNet50",
    "ResNet101",
    "ViT",
]


def discover_classes(root: Path):
    # Match the exact ordering used by wisense.dataset.CSI_Dataset at training time,
    # which relies on glob.glob's filesystem order (NOT alphabetical).
    folders = glob.glob(str(root) + "/*/")
    return [f.split("/")[-2] for f in folders]


def load_sample(mat_path: Path):
    raw = sio.loadmat(str(mat_path))["CSIamp"]
    raw = np.array(raw, dtype=np.float32)
    x = (raw - CSI_MEAN) / CSI_STD
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    if x.shape[0] != 3 * 114:
        raise ValueError(f"unexpected shape {x.shape} in {mat_path}")
    orig_len = x.shape[1]
    if orig_len < SEQ_LEN:
        raise ValueError(f"seq too short in {mat_path}")
    if orig_len != SEQ_LEN:
        step = max(1, orig_len // SEQ_LEN)
        x = x[:, ::step][:, :SEQ_LEN]
    x = x.reshape(3, 114, SEQ_LEN)
    return x, raw


VIRIDIS = np.array([
    [68, 1, 84], [71, 19, 101], [72, 36, 117], [69, 52, 128], [64, 68, 135],
    [57, 84, 140], [51, 99, 141], [44, 114, 142], [39, 127, 142], [34, 141, 141],
    [30, 155, 138], [32, 168, 133], [44, 182, 124], [68, 193, 112], [101, 203, 94],
    [139, 211, 70], [179, 217, 45], [219, 220, 35], [253, 231, 37],
], dtype=np.uint8)


def to_viridis(arr01: np.ndarray) -> np.ndarray:
    # arr01 in [0,1], shape (H,W)
    idx = np.clip((arr01 * (len(VIRIDIS) - 1)).round().astype(np.int32), 0, len(VIRIDIS) - 1)
    return VIRIDIS[idx]  # (H, W, 3)


def write_png(path: Path, rgb: np.ndarray) -> None:
    # Minimal PNG writer (RGB uint8). Avoids bringing in PIL.
    h, w, _ = rgb.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter: None
        raw.extend(rgb[y].tobytes())
    compressed = zlib.compress(bytes(raw), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png)


def heatmap_png(raw_amp: np.ndarray, target_w: int = 240) -> np.ndarray:
    # raw_amp: (342, T) amplitude in dB-ish; collapse antennas → mean over 3 antennas (114)
    if raw_amp.ndim == 2 and raw_amp.shape[0] == 342:
        x = raw_amp.reshape(3, 114, -1).mean(axis=0)
    else:
        x = raw_amp
    t = x.shape[1]
    if t > target_w:
        step = t // target_w
        x = x[:, ::step][:, :target_w]
    # normalize per-image
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    n = np.clip((x - lo) / (hi - lo), 0, 1)
    return to_viridis(n.astype(np.float32))


def load_summary(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=str(REPO / "demo"))
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "heatmaps").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    summary_rows = load_summary(REPO / "result" / "artifacts" / "aggregate" / "summary.csv")
    pareto_rows = load_summary(REPO / "result" / "artifacts" / "aggregate" / "pareto.csv") \
        if (REPO / "result" / "artifacts" / "aggregate" / "pareto.csv").exists() else []

    device = torch.device("cpu")

    bundle = {
        "summary": [],
        "pareto": [],
        "datasets": {},
    }

    # Performance summary (all datasets)
    for r in summary_rows:
        def f(k):
            try:
                return float(r.get(k, "") or "nan")
            except ValueError:
                return None
        bundle["summary"].append({
            "dataset": r["dataset"],
            "model": r["model"],
            "variant": r.get("variant", ""),
            "acc_mean": f("acc_mean"),
            "acc_std": f("acc_std"),
            "macro_f1_mean": f("macro_f1_mean"),
            "macro_f1_std": f("macro_f1_std"),
            "params_total_mean": f("params_total_mean"),
            "flops_forward_mean": f("flops_forward_mean"),
            "latency_ms_batch1_mean": f("latency_ms_batch1_mean"),
            "latency_ms_batch64_mean": f("latency_ms_batch64_mean"),
            "peak_gpu_mem_mb_mean": f("peak_gpu_mem_mb_mean"),
            "train_time_total_sec_mean": f("train_time_total_sec_mean"),
        })

    for r in pareto_rows:
        bundle["pareto"].append({k: r[k] for k in r.keys()})

    for dataset_name, spec in DATASET_SPECS.items():
        sample_root = spec["sample_root"]
        class_root = spec["class_root"]
        if not sample_root.exists() or not class_root.exists():
            print(f"[skip] {dataset_name}: missing {sample_root} or {class_root}")
            continue
        # Class ordering must match training time (glob-based, not alphabetical).
        classes = discover_classes(class_root)
        class_idx_by_name = {c: i for i, c in enumerate(classes)}
        num_classes = spec["num_classes"]
        assert len(classes) == num_classes, f"{dataset_name} class count mismatch: {len(classes)} vs {num_classes}"

        # Pick samples from held-out split; present them sorted alphabetically for UX.
        present_classes = discover_classes(sample_root)
        samples = []
        for cls in sorted(present_classes):
            if cls not in class_idx_by_name:
                continue
            files = sorted((sample_root / cls).glob("*.mat"))
            if not files:
                continue
            chosen = rng.choice(len(files), size=min(args.samples_per_class, len(files)), replace=False)
            for ci in chosen:
                fp = files[int(ci)]
                sample_id = f"{dataset_name}__{cls}__{fp.stem}"
                samples.append({
                    "id": sample_id,
                    "class_name": cls,
                    "class_idx": class_idx_by_name[cls],
                    "file": str(fp.relative_to(REPO)),
                })

        # Preload sample tensors + heatmap PNGs
        tensors = {}
        for s in samples:
            mat_path = REPO / s["file"]
            x, raw = load_sample(mat_path)
            tensors[s["id"]] = torch.from_numpy(x).float().unsqueeze(0).contiguous()  # (1,3,114,500)
            rgb = heatmap_png(raw)
            png_name = f"{s['id']}.png"
            write_png(out_dir / "heatmaps" / png_name, rgb)
            s["heatmap"] = f"heatmaps/{png_name}"

        # Only use freshly-retrained checkpoints under demo/ckpt/<dataset>/<model>.pt.
        # Legacy checkpoints in model_pt*/ are stale relative to the current data
        # pipeline and produce ~random predictions — they'd make the demo look broken.
        fresh_dir = Path(args.out) / "ckpt" / dataset_name
        available_models = []
        for m in KNOWN_MODELS:
            fresh = fresh_dir / f"{m}.pt"
            if fresh.exists():
                available_models.append((m, fresh, "fresh"))

        # Run inference per model
        predictions = {}  # model_name -> sample_id -> [(class_idx, prob)]
        per_model_meta = {}
        for mname, pt_path, source in available_models:
            print(f"[{dataset_name}] loading {mname} …")
            try:
                model = build_model(mname, num_classes)
                sd = torch.load(pt_path, map_location=device, weights_only=False)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                model.eval().to(device)
            except Exception as exc:  # noqa: BLE001
                print(f"[{dataset_name}] {mname}: skip ({exc})")
                continue

            n_params = sum(p.numel() for p in model.parameters())
            per_model_meta[mname] = {
                "params_total": n_params,
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "ckpt": str(pt_path.relative_to(REPO)),
                "ckpt_source": source,
            }

            with torch.no_grad():
                preds_this = {}
                for s in samples:
                    x = tensors[s["id"]].to(device).contiguous()
                    try:
                        logits = model(x)
                    except RuntimeError:
                        logits = model(x.reshape(x.shape))
                    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
                    top = np.argsort(-probs)[: args.topk]
                    preds_this[s["id"]] = [
                        {"class_idx": int(i), "prob": float(probs[int(i)])} for i in top
                    ]
                predictions[mname] = preds_this
            del model

        # Load per-model seed-0 confusion matrix from runs/runs/<dataset>/<model>/<variant>/0
        confusions = {}
        for mname in sorted(predictions.keys()):
            cand = list((REPO / "runs" / "runs" / dataset_name / mname).glob("*/0/confusion_matrix.json"))
            if cand:
                try:
                    cm = json.loads(cand[0].read_text())
                    confusions[mname] = cm
                except Exception as exc:  # noqa: BLE001
                    print(f"[{dataset_name}] {mname}: failed to read CM ({exc})")

        bundle["datasets"][dataset_name] = {
            "task": spec["task"],
            "num_classes": num_classes,
            "classes": classes,
            "samples": samples,
            "models": sorted(predictions.keys()),
            "model_meta": per_model_meta,
            "predictions": predictions,
            "confusion_matrices": confusions,
        }

    def clean(o):
        if isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                return None
            return o
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    out_file = out_dir / "data.json"
    out_file.write_text(json.dumps(clean(bundle), indent=2, allow_nan=False))
    print(f"wrote {out_file} ({out_file.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
