import argparse
import glob
import datetime as dt
import json
import os
import sys
import time
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio

from util import load_data_n_model
import dataset as csi_dataset
from utils.efficiency import count_params, measure_latency, measure_peak_mem, measure_flops


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

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    epoch_times = []
    for epoch in range(num_epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.float()
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - epoch_start)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
    total_time = sum(epoch_times)
    mean_epoch = total_time / len(epoch_times) if epoch_times else None
    return {"train_time_total_sec": total_time, "train_time_sec_epoch": mean_epoch}


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    with torch.no_grad():
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(inputs)
            outputs = outputs.float()
            
            loss = criterion(outputs,labels)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return {"acc": float(test_acc), "loss": float(test_loss)}


def set_seed(seed: int) -> None:
    """Set RNG seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CuBLAS deterministic requirement for CUDA >= 10.2 when using torch.use_deterministic_algorithms(True).
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

    
def parse_args():
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar','APPLIED'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT','SSM','Mamba'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a pretrained state_dict to load before training.')
    parser.add_argument('--eval-only', action='store_true', help='Skip training and only run evaluation.')
    parser.add_argument('--log-dir', type=str, default=None, help='Directory to store logs (mirrors train_all format).')
    parser.add_argument('--log-file', type=str, default=None, help='Explicit log file path. Overrides --log-dir if both are provided.')
    parser.add_argument('--save-ckpt', type=str, default=None, help='File path to save model.state_dict() after training. Ignored with --eval-only.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for reproducibility (applies to python/numpy/torch).')
    parser.add_argument('--mamba_selective', choices=['on', 'off'], default='on', help='Enable/disable Mamba selectivity.')
    parser.add_argument('--pooling', choices=['mean', 'last', 'attn'], default='mean', help='Temporal pooling for sequence models.')
    parser.add_argument('--seq_len', type=int, default=500, help='Sequence length after downsampling (CSI datasets only).')
    parser.add_argument('--shuffle_subcarriers', type=int, choices=[0, 1], default=0, help='Shuffle subcarriers in CSI inputs.')
    parser.add_argument('--shuffle_antennas', type=int, choices=[0, 1], default=0, help='Shuffle antennas in CSI inputs.')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Fraction of training data to use.')
    parser.add_argument('--noise', choices=['none', 'time_mask', 'subcarrier_dropout', 'gaussian'], default='none', help='Noise augmentation for training.')
    parser.add_argument('--noise_p', type=float, default=0.0, help='Noise strength/probability.')
    parser.add_argument('--val_noise', choices=['none', 'time_mask', 'subcarrier_dropout', 'gaussian'], default='none', help='Noise applied at evaluation.')
    parser.add_argument('--measure_efficiency', type=int, default=0, help='Measure params/FLOPs/latency/memory.')
    return parser.parse_args()


def _format_tag(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") if text else "0"


def build_variant(args) -> str:
    parts = []
    if args.model == "Mamba":
        parts.append(f"selective_{args.mamba_selective}")
        parts.append(f"pool_{args.pooling}")
    elif args.model == "SSM" and args.pooling != "mean":
        parts.append(f"pool_{args.pooling}")
    parts.append(f"seq{args.seq_len}")
    if args.shuffle_subcarriers:
        parts.append("shuffle_subcarriers")
    if args.shuffle_antennas:
        parts.append("shuffle_antennas")
    if args.train_fraction < 1.0:
        parts.append(f"frac{_format_tag(args.train_fraction)}")
    if args.noise != "none":
        parts.append(f"noise_{args.noise}_p{_format_tag(args.noise_p)}")
    if args.val_noise != "none":
        parts.append(f"valnoise_{args.val_noise}_p{_format_tag(args.noise_p)}")
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


def run_experiment(args):
    if args.seed is not None:
        set_seed(args.seed)

    if args.model != 'Mamba' and args.mamba_selective != 'on':
        print("[warn] --mamba_selective only applies to Mamba; resetting to 'on'.")
        args.mamba_selective = 'on'
    if args.model not in ('Mamba', 'SSM') and args.pooling != 'mean':
        print("[warn] --pooling only applies to Mamba/SSM; resetting to 'mean'.")
        args.pooling = 'mean'

    if args.dataset in ('UT_HAR_data', 'Widar') and args.seq_len != 500:
        print("[warn] --seq_len applies to CSI datasets only; resetting to 500.")
        args.seq_len = 500
    if args.seq_len != 500:
        seq_ok = {'RNN', 'GRU', 'LSTM', 'BiLSTM', 'CNN+GRU', 'SSM', 'Mamba'}
        if args.model not in seq_ok:
            raise ValueError(
                f"--seq_len only supported for {sorted(seq_ok)}; got model={args.model}"
            )

    root = './Data/' 
    # If using APPLIED and normalization not specified, compute from train split
    if args.dataset == 'APPLIED' and not (os.getenv('NTU_FI_NORM_MEAN') and os.getenv('NTU_FI_NORM_STD')):
        split_dir = os.path.join(root, 'APPLIED', 'train_amp')
        files = sorted(glob.glob(os.path.join(split_dir, '*', '*.mat')))
        if files:
            total = 0
            s = 0.0
            s2 = 0.0
            for path in files:
                mat = sio.loadmat(path)
                if 'CSIamp' not in mat:
                    continue
                x = mat['CSIamp']
                try:
                    x = x.reshape(3,114,500)
                except Exception:
                    pass
                x = x.astype(np.float64, copy=False)
                s += x.sum()
                s2 += np.square(x, dtype=np.float64).sum()
                total += x.size
            if total > 0:
                mean = s/total
                var = max(s2/total - mean*mean, 0.0)
                std = float(np.sqrt(var)) if var>0 else 1.0
                print(f"[info] APPLIED normalization mean={mean:.4f} std={std:.4f}")
                try:
                    csi_dataset.set_csi_normalization(float(mean), float(std))
                except Exception:
                    os.environ['NTU_FI_NORM_MEAN'] = str(float(mean))
                    os.environ['NTU_FI_NORM_STD'] = str(float(std))
    train_loader, test_loader, model, train_epoch = load_data_n_model(
        args.dataset,
        args.model,
        root,
        seq_len=args.seq_len,
        pooling=args.pooling,
        mamba_selective=args.mamba_selective,
        shuffle_subcarriers=bool(args.shuffle_subcarriers),
        shuffle_antennas=bool(args.shuffle_antennas),
        train_fraction=args.train_fraction,
        noise=args.noise,
        noise_p=args.noise_p,
        val_noise=args.val_noise,
        val_noise_p=args.noise_p,
        seed=args.seed,
    )
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics = {}
    metrics.update(count_params(model))

    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.checkpoint}")

    efficiency = {
        "flops_forward": None,
        "latency_ms_batch1": None,
        "latency_ms_batch64": None,
        "peak_gpu_mem_mb": None,
    }
    if args.measure_efficiency:
        try:
            sample_batch = next(iter(train_loader))
            sample_inputs = sample_batch[0]
            batch1 = _ensure_batch(sample_inputs, 1)
            batch64 = _ensure_batch(sample_inputs, 64)
            efficiency.update(measure_flops(model, batch1))
            efficiency.update(measure_latency(model, batch1, device))
            efficiency["latency_ms_batch1"] = efficiency.pop("median_ms", None)
            latency64 = measure_latency(model, batch64, device)
            efficiency["latency_ms_batch64"] = latency64.get("median_ms")
            efficiency.update(measure_peak_mem(model, batch64, device))
        except StopIteration:
            pass
        except Exception as exc:
            print(f"[warn] Efficiency measurement failed: {exc}")

    train_stats = {"train_time_total_sec": None, "train_time_sec_epoch": None}
    if not args.eval_only:
        train_stats = train(
            model=model,
            tensor_loader= train_loader,
            num_epochs= train_epoch,
            learning_rate=1e-3,
            criterion=criterion,
            device=device
        )
    test_stats = test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device
    )

    # Save checkpoint if requested
    if args.save_ckpt:
        ckpt_path = Path(args.save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    variant = build_variant(args)
    seed_dir = str(args.seed) if args.seed is not None else "noseed"
    run_dir = Path("runs") / args.dataset / args.model / variant / seed_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics.update(
        {
            "dataset": args.dataset,
            "model": args.model,
            "variant": variant,
            "seed": args.seed,
            "acc": test_stats.get("acc"),
            "loss": test_stats.get("loss"),
            "mamba_selective": args.mamba_selective,
            "pooling": args.pooling,
            "seq_len": args.seq_len,
            "shuffle_subcarriers": bool(args.shuffle_subcarriers),
            "shuffle_antennas": bool(args.shuffle_antennas),
            "train_fraction": args.train_fraction,
            "noise": args.noise,
            "noise_p": args.noise_p,
            "val_noise": args.val_noise,
            "val_noise_p": args.noise_p,
            "train_time_total_sec": train_stats.get("train_time_total_sec"),
            "train_time_sec_epoch": train_stats.get("train_time_sec_epoch"),
        }
    )
    metrics.update(efficiency)
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {metrics_path}")


def main():
    args = parse_args()
    log_path = None
    if args.log_file:
        log_path = Path(args.log_file)
    elif args.log_dir:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = Path(args.log_dir) / args.dataset / f"{timestamp}_{args.model}.log"

    if not log_path:
        run_experiment(args)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        log_file = stack.enter_context(log_path.open("w", buffering=1))
        cmdline = f"python {' '.join(sys.argv[1:])}"
        log_file.write(f"$ {cmdline}\n\n")
        tee_out = _Tee(sys.stdout, log_file)
        tee_err = _Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            run_experiment(args)
    print(f"Logs written to {log_path}")


if __name__ == "__main__":
    main()
