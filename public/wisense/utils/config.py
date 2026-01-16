from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_json_config(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    return json.loads(p.read_text())


def default_config() -> Dict[str, Any]:
    return {
        "data_root": "public/Data/",
        "runs_root": "runs",
        "dataset": {"name": "NTU-Fi-HumanID"},
        "model": {"name": "Mamba"},
        "dataloader": {"batch_train": 64, "batch_test": 64, "num_workers": 0},
        "training": {
            "epochs": None,
            "lr": 1e-3,
            "eval_only": False,
            "epochs_by_model": {},
            "early_stop": {
                "enabled": False,
                "patience": 6,
                "metric": "loss",
                "min_delta": 0.0001,
                "restore_best": True,
            },
        },
        "ablations": {
            "seq_len": 500,
            "pooling": "mean",
            "mamba_selective": "on",
            "shuffle_subcarriers": False,
            "shuffle_antennas": False,
            "train_fraction": 1.0,
            "noise": "none",
            "noise_p": 0.0,
            "val_noise": "none",
            "val_noise_p": None,
        },
        "efficiency": {"enabled": False, "warmup": 30, "iters": 200},
        "model_params": {
            "mamba": {
                "d_model": 256,
                "depth": 4,
                "d_state": 64,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
            },
            "vit": {
                "in_channels": 1,
                "patch_size_w": 9,
                "patch_size_h": 25,
                "emb_size": 225,
                "img_size": 342 * 500,
                "depth": 1,
            },
        },
    }
