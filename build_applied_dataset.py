#!/usr/bin/env python3
"""Build a SenseFi-compatible dataset from measured PCAP CSI dumps.

This script scans a directory of nexmon_csi PCAP files (e.g., 応用プロジェクト),
extracts CSI, reshapes it to (3, 114, 500), computes amplitude as `CSIamp`, and
emits MATLAB .mat files under a SenseFi-like folder layout:

  <out_root>/train_amp/<class_id>/*.mat
  <out_root>/test_amp/<class_id>/*.mat

where <class_id> is zero-padded to 3 digits (001..003).

Labeling
--------
- Expects filenames like: `output_(humanID)_(trialID).pcap`.
- The class label is `humanID` (e.g., output_1_7.pcap -> class 001).
- The trial index is `trialID`; you can select test trials explicitly.

After generation, you can train/evaluate with:

  python run.py --dataset APPLIED --model GRU

Notes
-----
- Requires scapy to parse PCAPs: pip install scapy
- The time/subcarrier selection mirrors 応用プロジェクト/csi_heatmap.py
  (center 114 subcarriers, 500 packets per stream, 3 streams by chunking time).
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.io import savemat

try:
    from scapy.all import rdpcap, UDP  # type: ignore
except Exception as e:
    rdpcap = None
    UDP = None


# Accept output_<sid>_<idx>.pcap (primary) and output_<sid>-<idx>.pcap (fallback)
PCAP_RE = re.compile(r"^output_(?P<sid>\d+)[-_](?P<idx>\d+)\.pcap$")


def list_pcaps(pcap_dir: Path) -> List[Path]:
    files = []
    for p in sorted(pcap_dir.glob("*.pcap")):
        if PCAP_RE.search(p.name):
            files.append(p)
    return files


def parse_subject_id(path: Path) -> Tuple[int, int]:
    m = PCAP_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected PCAP name: {path.name}")
    sid = int(m.group("sid"))
    idx = int(m.group("idx"))
    return sid, idx


def extract_csi_from_pcap(pcap_file: Path) -> List[np.ndarray]:
    if rdpcap is None or UDP is None:
        raise RuntimeError("scapy is required. Install with `pip install scapy`. ")
    packets = rdpcap(str(pcap_file))
    NEXMON_HEADER_SIZE = 40
    csi_data_list: List[np.ndarray] = []
    for pkt in packets:
        if UDP in pkt:
            payload = bytes(pkt[UDP].payload)
            if len(payload) <= NEXMON_HEADER_SIZE:
                continue
            magic = int.from_bytes(payload[0:2], "little", signed=False)
            if magic != 0x1111:
                continue
            csi_data = payload[NEXMON_HEADER_SIZE:]
            # Expect pairs of int16 (real, imag). Handle odd-length payloads gracefully.
            buf = np.frombuffer(csi_data, dtype=np.int16)
            if buf.size < 2:
                continue
            if (buf.size % 2) != 0:
                # Drop the last lone int16 to make pairs
                buf = buf[:-1]
            try:
                buf = buf.reshape(-1, 2)
            except ValueError:
                # As a last resort, skip malformed packet
                continue
            complex_data = buf[:, 0].astype(np.float32) + 1j * buf[:, 1].astype(np.float32)
            if complex_data.size > 0:
                csi_data_list.append(complex_data)
    return csi_data_list


def create_csi_matrix(csi_data_list: List[np.ndarray]) -> np.ndarray:
    if not csi_data_list:
        return np.zeros((0, 0), dtype=np.complex64)
    min_len = min(len(x) for x in csi_data_list)
    t = len(csi_data_list)
    out = np.zeros((t, min_len), dtype=np.complex64)
    for i, arr in enumerate(csi_data_list):
        out[i, :] = arr[:min_len]
    return out


def reshape_csi_for_whofi(csi_matrix: np.ndarray, num_streams=1, num_subcarriers=114, num_packets=500) -> np.ndarray:
    T, F = csi_matrix.shape
    # center crop subcarriers
    start_sc = max((F - num_subcarriers) // 2, 0)
    end_sc = start_sc + num_subcarriers
    if end_sc > F:
        # pad subcarriers symmetrically if too few
        pad = end_sc - F
        left = pad // 2
        right = pad - left
        sub = np.pad(csi_matrix, ((0, 0), (left, right)), mode="edge")[:, :num_subcarriers]
    else:
        sub = csi_matrix[:, start_sc:end_sc]

    csi_3d = np.zeros((num_streams, num_subcarriers, num_packets), dtype=np.complex64)
    packets_per_stream = max(T // num_streams, 1)
    for s in range(num_streams):
        start = s * packets_per_stream
        if start + num_packets <= T:
            seg = sub[start:start + num_packets, :]
        else:
            need = num_packets
            seg_list: List[np.ndarray] = []
            pos = start
            while need > 0:
                take = min(need, max(T - pos, 0))
                if take <= 0:
                    # wrap
                    pos = 0
                    continue
                seg_list.append(sub[pos:pos + take, :])
                pos += take
                need -= take
            seg = np.vstack(seg_list)
        csi_3d[s, :, :] = seg.T
    return csi_3d


def save_mat(out_path: Path, csi_amp: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(out_path), {"CSIamp": csi_amp.astype(np.float32, copy=False)}, do_compression=True)


def split_by_ratio(paths: List[Path], ratio: float) -> Tuple[List[Path], List[Path]]:
    k = max(1, int(math.floor(len(paths) * ratio)))
    return paths[:k], paths[k:]


def split_by_trials(
    paths: List[Path],
    test_trials: Optional[List[int]] = None,
    train_trials: Optional[List[int]] = None,
) -> Tuple[List[Path], List[Path]]:
    """Split deterministically using trial IDs.

    If `test_trials` is provided, any path with trialID in that set goes to test,
    others to train. If both `train_trials` and `test_trials` are given, they
    must be disjoint; `test_trials` still wins in case of overlap.
    """
    test_set = set(test_trials or [])
    train_set = set(train_trials or [])
    train_list: List[Path] = []
    test_list: List[Path] = []
    for p in paths:
        _, idx = parse_subject_id(p)
        if idx in test_set:
            test_list.append(p)
        elif train_set and (idx in train_set):
            train_list.append(p)
        elif test_set:
            # not in explicit test set -> train
            train_list.append(p)
        else:
            # fallback to train if only train_trials provided
            train_list.append(p)
    return train_list, test_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcap-dir", type=Path, default=Path("応用プロジェクト"))
    parser.add_argument("--out-root", type=Path, default=Path("Data/APPLIED"))
    parser.add_argument("--streams", type=int, default=1,
                        help="Number of antenna streams to encode (use 1 for single-antenna setups).")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Within each humanID, fraction of trials for train (ignored if --test-trials is set)")
    parser.add_argument("--test-trials", type=int, nargs='*', default=None,
                        help="Explicit trialIDs for test split (e.g., --test-trials 8 9 10). Overrides --train-ratio.")
    parser.add_argument("--train-trials", type=int, nargs='*', default=None,
                        help="Optional trialIDs for train split (used only with --test-trials for clarity).")
    args = parser.parse_args()

    pcaps = list_pcaps(args.pcap_dir)
    if not pcaps:
        print(f"No PCAP files found in {args.pcap_dir}")
        return

    # Group by subject id
    groups: Dict[int, List[Path]] = {}
    for p in pcaps:
        sid, idx = parse_subject_id(p)
        groups.setdefault(sid, []).append(p)
    for sid in groups:
        groups[sid] = sorted(groups[sid], key=lambda p: parse_subject_id(p)[1])

    print(f"Found subjects: {sorted(groups.keys())}")
    for sid, paths in groups.items():
        cls = f"{sid:03d}"
        if args.test_trials:
            train_list, test_list = split_by_trials(paths, args.test_trials, args.train_trials)
        else:
            train_list, test_list = split_by_ratio(paths, args.train_ratio)
        for split_name, split_paths in [("train_amp", train_list), ("test_amp", test_list)]:
            for p in split_paths:
                try:
                    csi_list = extract_csi_from_pcap(p)
                    csi_mat = create_csi_matrix(csi_list)
                    if csi_mat.size == 0:
                        print(f"[warn] No CSI parsed from {p}")
                        continue
                    csi_3d = reshape_csi_for_whofi(csi_mat, args.streams, 114, 500)
                    csi_amp = np.abs(csi_3d)
                    out_path = args.out_root / split_name / cls / (p.stem + ".mat")
                    save_mat(out_path, csi_amp)
                    print(f"[ok] {split_name}/{cls} <- {p.name}")
                except Exception as e:
                    print(f"[err] {p}: {e}")

    print("Done. You can now train with: \n  python run.py --dataset APPLIED --model GRU")


if __name__ == "__main__":
    main()
