from __future__ import annotations

import statistics
import time
from typing import Tuple

import torch


def _normalize_inputs(sample_batch, device: torch.device) -> Tuple:
    if isinstance(sample_batch, (tuple, list)):
        inputs = sample_batch[0]
    else:
        inputs = sample_batch
    if torch.is_tensor(inputs):
        return (inputs.to(device),)
    if isinstance(inputs, (tuple, list)):
        return tuple(x.to(device) if torch.is_tensor(x) else x for x in inputs)
    return (inputs,)


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}


def measure_latency(
    model: torch.nn.Module,
    sample_batch,
    device: torch.device,
    warmup: int = 30,
    iters: int = 200,
) -> dict:
    inputs = _normalize_inputs(sample_batch, device)
    was_training = model.training
    model.eval()
    timings = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    if was_training:
        model.train()
    median_ms = statistics.median(timings) if timings else None
    return {"median_ms": median_ms}


def measure_peak_mem(
    model: torch.nn.Module,
    sample_batch,
    device: torch.device,
) -> dict:
    if device.type != "cuda":
        return {"peak_gpu_mem_mb": None}
    inputs = _normalize_inputs(sample_batch, device)
    was_training = model.training
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(*inputs)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if was_training:
        model.train()
    return {"peak_gpu_mem_mb": peak}


def measure_flops(model: torch.nn.Module, sample_batch) -> dict:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return {"flops_forward": None}
    device = next(model.parameters()).device
    inputs = _normalize_inputs(sample_batch, device)
    try:
        with torch.no_grad():
            flops = FlopCountAnalysis(model, inputs)
            total = float(flops.total())
    except Exception:
        return {"flops_forward": None}
    return {"flops_forward": total}
