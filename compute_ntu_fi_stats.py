#!/usr/bin/env python3
"""Compute mean/std statistics for NTU-Fi HumanID splits.

The script replicates the preprocessing logic from `dataset.CSI_Dataset` up to,
but *excluding*, the hard-coded normalization constants. It is useful for
verifying whether a local dataset dump matches the statistics assumed by the
pretrained checkpoints.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, Tuple

import numpy as np
import scipy.io as sio


def list_mat_files(split_dir: str) -> Iterable[str]:
    pattern = os.path.join(split_dir, "*", "*.mat")
    return sorted(glob.glob(pattern))


def load_raw_tensor(path: str) -> np.ndarray:
    """Load CSIamp, downsample, and reshape without applying normalization."""
    mat = sio.loadmat(path)
    if "CSIamp" not in mat:
        raise KeyError(f"'CSIamp' not found in {path}")
    x = mat["CSIamp"]
    # Downsample time dimension (2000 -> 500) exactly like dataset.py.
    x = x[:, ::4]
    try:
        x = x.reshape(3, 114, 500)
    except ValueError as exc:
        raise ValueError(f"Unexpected shape {x.shape} for {path}") from exc
    return x.astype(np.float64, copy=False)


def compute_stats(files: Iterable[str]) -> Tuple[float, float]:
    count = 0
    sum_ = 0.0
    sum_sq = 0.0
    for idx, path in enumerate(files, start=1):
        x = load_raw_tensor(path)
        sum_ += x.sum()
        sum_sq += np.square(x, dtype=np.float64).sum()
        count += x.size
        if idx % 100 == 0:
            print(f"Processed {idx} files ...")
    if count == 0:
        raise RuntimeError("No MAT files found.")
    mean = sum_ / count
    var = max(sum_sq / count - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="./Data/NTU-Fi-HumanID",
        help="Path to NTU-Fi HumanID root containing train_amp/test_amp folders.",
    )
    parser.add_argument(
        "--split",
        choices=["train_amp", "test_amp"],
        default="test_amp",
        help="Split to analyze. Note: SenseFi uses test_amp for training.",
    )
    args = parser.parse_args()

    split_dir = os.path.join(args.root, args.split)
    files = list_mat_files(split_dir)
    print(f"Split: {split_dir}")
    print(f"Found {len(files)} MAT files")
    mean, std = compute_stats(files)
    print(f"Raw (pre-normalization) mean = {mean:.4f}, std = {std:.4f}")
    print(
        "If you plug these values into dataset.CSI_Dataset normalization, "
        "the processed tensors will be roughly zero-mean, unit-variance."
    )


if __name__ == "__main__":
    main()
