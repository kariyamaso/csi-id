#!/usr/bin/env python3

import argparse
import datetime as dt
import glob
import json
import os
import random
import sys
import time
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from wisense import dataset as csi_dataset
from wisense.util import load_data_n_model
from wisense.utils.config import deep_update, default_config, load_json_config
from wisense.utils.efficiency import count_params, measure_flops, measure_latency, measure_peak_mem


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.__version__ >= "2.0":
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def _format_tag(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") if text else "0"


def build_variant(cfg: dict) -> str:
    a = cfg["ablations"]
    model = cfg["model"]["name"]
    parts = []
    if model == "Mamba":
        parts.append(f"selective_{a['mamba_selective']}")
        parts.append(f"pool_{a['pooling']}")
    parts.append(f"seq{a['seq_len']}")
    if a["shuffle_subcarriers"]:
        parts.append("shuffle_subcarriers")
    if a["shuffle_antennas"]:
        parts.append("shuffle_antennas")
    if a["train_fraction"] < 1.0:
        parts.append(f"frac{_format_tag(a['train_fraction'])}")
    if a["noise"] != "none":
        parts.append(f"noise_{a['noise']}_p{_format_tag(a['noise_p'])}")
    if a["val_noise"] != "none":
        parts.append(f"valnoise_{a['val_noise']}_p{_format_tag(a['val_noise_p'] or a['noise_p'])}")
    return "_".join(parts) if parts else "base"


def _ensure_batch(inputs: torch.Tensor, batch_size: int) -> torch.Tensor:
    if inputs.size(0) == batch_size:
        return inputs
    if inputs.size(0) > batch_size:
        return inputs[:batch_size]
    repeat = batch_size // inputs.size(0)
    remainder = batch_size % inputs.size(0)
    expanded = inputs.repeat((repeat,) + (1,) * (inputs.ndim - 1))
    if remainder:
        expanded = torch.cat([expanded, inputs[:remainder]], dim=0)
    return expanded


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_times = []
    for epoch in range(num_epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        epoch_loss = 0
        epoch_accuracy = 0
        for inputs, labels in tensor_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - epoch_start)
        print(f"Epoch:{epoch+1}, Accuracy:{float(epoch_accuracy):.4f},Loss:{float(epoch_loss):.9f}")
    total_time = sum(epoch_times)
    mean_epoch = total_time / len(epoch_times) if epoch_times else None
    return {"train_time_total_sec": total_time, "train_time_sec_epoch": mean_epoch}


def test(model, tensor_loader, criterion, device, *, verbose: bool = True):
    model.eval()
    test_acc = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in tensor_loader:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs).float()
            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    if verbose:
        print(f"validation accuracy:{float(test_acc):.4f}, loss:{float(test_loss):.5f}")
    return {"acc": float(test_acc), "loss": float(test_loss)}


def parse_args():
    p = argparse.ArgumentParser("WiSense public runner")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--save-ckpt", type=str, default=None)

    p.add_argument("--measure_efficiency", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--pooling", choices=["mean", "last", "attn"], default=None)
    p.add_argument("--mamba_selective", choices=["on", "off"], default=None)
    p.add_argument("--shuffle_subcarriers", type=int, choices=[0, 1], default=None)
    p.add_argument("--shuffle_antennas", type=int, choices=[0, 1], default=None)
    p.add_argument("--train_fraction", type=float, default=None)
    p.add_argument("--noise", choices=["none", "time_mask", "subcarrier_dropout", "gaussian"], default=None)
    p.add_argument("--noise_p", type=float, default=None)
    p.add_argument("--val_noise", choices=["none", "time_mask", "subcarrier_dropout", "gaussian"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_train", type=int, default=None)
    p.add_argument("--batch_test", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = default_config()
    deep_update(cfg, load_json_config(args.config))

    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.batch_train is not None:
        cfg["dataloader"]["batch_train"] = args.batch_train
    if args.batch_test is not None:
        cfg["dataloader"]["batch_test"] = args.batch_test
    if args.num_workers is not None:
        cfg["dataloader"]["num_workers"] = args.num_workers

    if args.eval_only:
        cfg["training"]["eval_only"] = True
    if args.measure_efficiency is not None:
        cfg["efficiency"]["enabled"] = bool(args.measure_efficiency)
    a = cfg["ablations"]
    if args.seq_len is not None:
        a["seq_len"] = args.seq_len
    if args.pooling is not None:
        a["pooling"] = args.pooling
    if args.mamba_selective is not None:
        a["mamba_selective"] = args.mamba_selective
    if args.shuffle_subcarriers is not None:
        a["shuffle_subcarriers"] = bool(args.shuffle_subcarriers)
    if args.shuffle_antennas is not None:
        a["shuffle_antennas"] = bool(args.shuffle_antennas)
    if args.train_fraction is not None:
        a["train_fraction"] = args.train_fraction
    if args.noise is not None:
        a["noise"] = args.noise
    if args.noise_p is not None:
        a["noise_p"] = args.noise_p
    if args.val_noise is not None:
        a["val_noise"] = args.val_noise

    if args.seed is not None:
        set_seed(args.seed)

    # Optional APPLIED normalization auto-compute (matches root script behavior).
    data_root = cfg["data_root"]
    if cfg["dataset"]["name"] == "APPLIED" and not (os.getenv("NTU_FI_NORM_MEAN") and os.getenv("NTU_FI_NORM_STD")):
        split_dir = os.path.join(data_root, "APPLIED", "train_amp")
        files = sorted(glob.glob(os.path.join(split_dir, "*", "*.mat")))
        if files:
            total = 0
            s = 0.0
            s2 = 0.0
            for path in files:
                mat = sio.loadmat(path)
                if "CSIamp" not in mat:
                    continue
                x = mat["CSIamp"]
                try:
                    x = x.reshape(3, 114, 500)
                except Exception:
                    pass
                x = x.astype(np.float64, copy=False)
                s += x.sum()
                s2 += np.square(x, dtype=np.float64).sum()
                total += x.size
            if total > 0:
                mean = s / total
                var = max(s2 / total - mean * mean, 0.0)
                std = float(np.sqrt(var)) if var > 0 else 1.0
                print(f"[info] APPLIED normalization mean={mean:.4f} std={std:.4f}")
                try:
                    csi_dataset.set_csi_normalization(float(mean), float(std))
                except Exception:
                    os.environ["NTU_FI_NORM_MEAN"] = str(float(mean))
                    os.environ["NTU_FI_NORM_STD"] = str(float(std))

    train_loader, test_loader, model, train_epoch_default = load_data_n_model(
        cfg["dataset"]["name"],
        cfg["model"]["name"],
        data_root,
        seq_len=a["seq_len"],
        pooling=a["pooling"],
        mamba_selective=a["mamba_selective"],
        shuffle_subcarriers=a["shuffle_subcarriers"],
        shuffle_antennas=a["shuffle_antennas"],
        shuffle_seed=(args.seed or 0),
        train_fraction=a["train_fraction"],
        noise=a["noise"],
        noise_p=a["noise_p"],
        val_noise=a["val_noise"],
        val_noise_p=a["val_noise_p"],
        seed=args.seed,
        batch_train=cfg["dataloader"]["batch_train"],
        batch_test=cfg["dataloader"]["batch_test"],
        num_workers=cfg["dataloader"]["num_workers"],
        model_params=cfg.get("model_params", {}),
    )

    epochs_by_model = cfg.get("training", {}).get("epochs_by_model", {}) or {}
    epochs_override = epochs_by_model.get(cfg["model"]["name"])
    if epochs_override is not None:
        epochs = int(epochs_override)
    else:
        epochs = cfg["training"]["epochs"] if cfg["training"]["epochs"] is not None else train_epoch_default
    lr = float(cfg["training"]["lr"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.checkpoint}")

    log_path = None
    if args.log_file:
        log_path = Path(args.log_file)
    elif args.log_dir:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = Path(args.log_dir) / cfg["dataset"]["name"] / f"{timestamp}_{cfg['model']['name']}.log"

    def _run():
        metrics = {}
        metrics.update(count_params(model))

        efficiency = {
            "flops_forward": None,
            "latency_ms_batch1": None,
            "latency_ms_batch64": None,
            "peak_gpu_mem_mb": None,
        }
        if cfg["efficiency"]["enabled"]:
            try:
                sample_batch = next(iter(train_loader))
                sample_inputs = sample_batch[0]
                batch1 = _ensure_batch(sample_inputs, 1)
                batch64 = _ensure_batch(sample_inputs, 64)
                efficiency.update(measure_flops(model, batch1))
                efficiency.update(measure_latency(model, batch1, device, warmup=cfg["efficiency"]["warmup"], iters=cfg["efficiency"]["iters"]))
                efficiency["latency_ms_batch1"] = efficiency.pop("median_ms", None)
                latency64 = measure_latency(model, batch64, device, warmup=cfg["efficiency"]["warmup"], iters=cfg["efficiency"]["iters"])
                efficiency["latency_ms_batch64"] = latency64.get("median_ms")
                efficiency.update(measure_peak_mem(model, batch64, device))
            except Exception as exc:
                print(f"[warn] Efficiency measurement failed: {exc}")

        train_stats = {"train_time_total_sec": None, "train_time_sec_epoch": None}
        test_stats = {"acc": None, "loss": None}

        early = cfg.get("training", {}).get("early_stop", {}) or {}
        early_enabled = bool(early.get("enabled", False))
        patience = int(early.get("patience", 5))
        metric_name = str(early.get("metric", "loss"))
        min_delta = float(early.get("min_delta", 0.0))
        restore_best = bool(early.get("restore_best", True))

        best_epoch = None
        best_value = None
        best_state = None
        bad_epochs = 0
        epochs_ran = 0

        if cfg["training"]["eval_only"]:
            test_stats = test(model, test_loader, criterion, device, verbose=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            epoch_times = []
            for epoch in range(int(epochs)):
                model.train()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                epoch_start = time.perf_counter()

                epoch_loss = 0.0
                epoch_accuracy = 0.0
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device, dtype=torch.long)
                    optimizer.zero_grad()
                    outputs = model(inputs).float()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * inputs.size(0)
                    predict_y = torch.argmax(outputs, dim=1).to(device)
                    epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)

                epoch_loss = epoch_loss / len(train_loader.dataset)
                epoch_accuracy = epoch_accuracy / len(train_loader)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                epoch_times.append(time.perf_counter() - epoch_start)
                epochs_ran = epoch + 1
                print(f"Epoch:{epoch+1}, Accuracy:{float(epoch_accuracy):.4f},Loss:{float(epoch_loss):.9f}")

                # validation (same split used for final eval)
                val_stats = test(model, test_loader, criterion, device, verbose=True)

                if early_enabled:
                    current = val_stats.get(metric_name)
                    if current is None:
                        raise ValueError(f"early_stop.metric={metric_name} not present in metrics")

                    improved = False
                    if best_value is None:
                        improved = True
                    else:
                        if metric_name == "loss":
                            improved = (best_value - current) > min_delta
                        elif metric_name == "acc":
                            improved = (current - best_value) > min_delta
                        else:
                            raise ValueError("early_stop.metric must be 'loss' or 'acc'")

                    if improved:
                        best_value = current
                        best_epoch = epoch + 1
                        bad_epochs = 0
                        if restore_best:
                            best_state = copy.deepcopy(model.state_dict())
                    else:
                        bad_epochs += 1
                        if bad_epochs >= patience:
                            print(f"[info] Early stopping triggered at epoch {epoch+1} (best={best_epoch}, {metric_name}={best_value})")
                            break

            total_time = float(sum(epoch_times))
            train_stats = {
                "train_time_total_sec": total_time,
                "train_time_sec_epoch": (total_time / len(epoch_times)) if epoch_times else None,
                "epochs_ran": epochs_ran,
            }

            if restore_best and best_state is not None:
                model.load_state_dict(best_state)

            test_stats = test(model, test_loader, criterion, device, verbose=True)

        if args.save_ckpt:
            ckpt_path = Path(args.save_ckpt)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        variant = build_variant(cfg)
        seed_dir = str(args.seed) if args.seed is not None else "noseed"
        run_dir = Path(cfg["runs_root"]) / cfg["dataset"]["name"] / cfg["model"]["name"] / variant / seed_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics.update(
            {
                "dataset": cfg["dataset"]["name"],
                "model": cfg["model"]["name"],
                "variant": variant,
                "seed": args.seed,
                "acc": test_stats.get("acc"),
                "loss": test_stats.get("loss"),
                "pooling": a["pooling"],
                "seq_len": a["seq_len"],
                "mamba_selective": a["mamba_selective"],
                "shuffle_subcarriers": bool(a["shuffle_subcarriers"]),
                "shuffle_antennas": bool(a["shuffle_antennas"]),
                "train_fraction": a["train_fraction"],
                "noise": a["noise"],
                "noise_p": a["noise_p"],
                "val_noise": a["val_noise"],
                "val_noise_p": a["val_noise_p"],
                "epochs": epochs,
                "epochs_ran": train_stats.get("epochs_ran", epochs if not cfg["training"]["eval_only"] else 0),
                "lr": lr,
                "early_stop_enabled": early_enabled,
                "early_stop_patience": patience,
                "early_stop_metric": metric_name,
                "early_stop_best_epoch": best_epoch,
                "early_stop_best_value": best_value,
                "train_time_total_sec": train_stats.get("train_time_total_sec"),
                "train_time_sec_epoch": train_stats.get("train_time_sec_epoch"),
            }
        )
        metrics.update(efficiency)
        metrics["config"] = cfg
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics written to {metrics_path}")

    if not log_path:
        _run()
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        log_file = stack.enter_context(log_path.open("w", buffering=1))
        cmdline = f"python {' '.join(sys.argv[1:])}"
        log_file.write(f"$ {cmdline}\n\n")
        tee_out = _Tee(sys.stdout, log_file)
        tee_err = _Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            _run()
    print(f"Logs written to {log_path}")


if __name__ == "__main__":
    main()
