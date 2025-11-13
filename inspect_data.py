import argparse
import os
import struct
from typing import List, Optional, Tuple

import numpy as np
import torch
try:
    import scipy.io as sio  # optional; only needed for MAT datasets
except Exception:  # pragma: no cover
    sio = None

# Dataset types are imported lazily inside inspector functions to avoid
# importing SciPy when only parsing PCAPs.


def stats_str(arr: np.ndarray) -> str:
    arr = arr.astype(np.float32)
    return (
        f"shape={tuple(arr.shape)} dtype={arr.dtype} "
        f"min={float(arr.min()):.4f} max={float(arr.max()):.4f} "
        f"mean={float(arr.mean()):.4f} std={float(arr.std()):.4f}"
    )


def inspect_ntu_fi(root: str, split: str):
    from dataset import CSI_Dataset  # lazy import
    split_dir = os.path.join(root, f"{split}_amp")
    ds = CSI_Dataset(split_dir)
    print(f"Split dir: {split_dir}")
    print(f"Total files: {len(ds)}")
    print(f"Classes (folders): {len(ds.category)}")

    # Print class mapping and per-class counts
    idx2name = {v: k for k, v in ds.category.items()}
    names = [idx2name[i] for i in range(len(idx2name))]
    counts = {name: 0 for name in names}
    for p in ds.data_list:
        name = p.split('/')[-2]
        counts[name] += 1
    print("Per-class counts:")
    for name in names:
        print(f"  {name}: {counts[name]}")

    # Inspect first file raw and processed
    if len(ds.data_list) == 0:
        print("No .mat files found.")
        return
    sample_path = ds.data_list[0]
    if sio is None:
        raise ImportError("scipy is required to inspect NTU-Fi datasets but is not installed.")
    raw = sio.loadmat(sample_path)[ds.modal]
    print(f"Raw {ds.modal} from {sample_path} -> {stats_str(raw)}")

    x, y = ds[0]
    x_np = x.numpy()
    print(f"Processed tensor -> shape={tuple(x_np.shape)} mean={x_np.mean():.4f} std={x_np.std():.4f}")
    print(f"Label idx: {y} (folder {idx2name[y]})")


def inspect_ut_har(root: str):
    from dataset import UT_HAR_dataset  # lazy import
    data = UT_HAR_dataset(root)
    for k in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
        if k in data:
            arr = data[k].numpy() if torch.is_tensor(data[k]) else data[k]
            print(f"{k}: {stats_str(arr)}")


def inspect_widar(root: str, split: str):
    from dataset import Widar_Dataset  # lazy import
    split_dir = os.path.join(root, f"Widardata/{split}")
    ds = Widar_Dataset(split_dir)
    print(f"Split dir: {split_dir}")
    print(f"Total files: {len(ds)}")
    print(f"Classes (folders): {len(ds.category)}")
    idx2name = {v: k for k, v in ds.category.items()}
    names = [idx2name[i] for i in range(len(idx2name))]
    counts = {name: 0 for name in names}
    for p in ds.data_list:
        name = p.split('/')[-2]
        counts[name] += 1
    print("Per-class counts:")
    for name in names:
        print(f"  {name}: {counts[name]}")
    if len(ds) > 0:
        x, y = ds[0]
        x_np = x.numpy()
        print(f"Processed tensor -> shape={tuple(x_np.shape)} mean={x_np.mean():.4f} std={x_np.std():.4f}")
        print(f"Label idx: {y} (folder {idx2name[y]})")


# -------------------------
# Nexmon PCAP CSI utilities
# -------------------------

def _iter_pcap_udp_payloads(path: str):
    """Yield UDP payload bytes from a pcap file.

    Supports little/big-endian pcap global header. Parses Ethernet + IPv4 + UDP.
    """
    with open(path, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            return
        magic = gh[:4]
        little = magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1")
        endian = "<" if little else ">"
        # Iterate packets
        while True:
            ph = f.read(16)
            if len(ph) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + "IIII", ph)
            data = f.read(incl_len)
            if len(data) < 42:  # min Ethernet+IPv4+UDP header
                continue
            # Ethernet type (big-endian network order)
            eth_type = int.from_bytes(data[12:14], "big")
            if eth_type != 0x0800:
                continue  # not IPv4
            ihl = (data[14] & 0x0F) * 4
            if 14 + ihl + 8 > len(data):
                continue
            proto = data[23]
            if proto != 17:
                continue  # not UDP
            udpo = 14 + ihl
            ulen = int.from_bytes(data[udpo + 4 : udpo + 6], "big")
            payload = data[udpo + 8 : udpo + ulen]
            if not payload:
                continue
            yield payload


def _extract_nexmon_csi_from_payload(payload: bytes,
                                     offsets: Tuple[int, ...] = (18, 30, 22, 42),
                                     expected: Tuple[int, ...] = (256, 128, 64)) -> Optional[np.ndarray]:
    """Try to extract complex CSI samples from a Nexmon UDP payload.

    - Tries a set of header offsets; at each offset, interprets the remaining bytes
      as little-endian int16 I/Q pairs and returns the first N pairs if N is one of
      expected subcarrier counts (default: 256, 128, 64).
    - Returns a float32 array of shape (N, 2) for I/Q, or None if not a Nexmon payload.
    """
    # Nexmon CSI payloads commonly start with 0x1111 signature
    if len(payload) < 32 or not payload.startswith(b"\x11\x11"):
        return None

    for off in offsets:
        if off >= len(payload):
            continue
        rem = len(payload) - off
        if rem < 4 or (rem % 2) != 0:
            continue
        # Interpret remaining as little-endian int16
        arr = np.frombuffer(payload[off:], dtype="<i2")
        # Ensure even count for I/Q pairing
        pairs = (arr.shape[0] // 2)
        if pairs <= 0:
            continue
        # Prefer an exact expected size
        n = None
        for e in expected:
            if pairs == e:
                n = e
                break
        # If not exact, accept the largest expected <= pairs
        if n is None:
            cand = [e for e in expected if e <= pairs]
            if not cand:
                continue
            n = max(cand)
        # Slice to n pairs and reshape (n, 2) -> I/Q
        iq = arr[: n * 2].astype(np.float32).reshape(-1, 2)
        return iq
    return None


def load_csi_from_pcap(path: str,
                       max_frames: Optional[int] = None,
                       expect_subcarriers: Tuple[int, ...] = (256, 128, 64)) -> np.ndarray:
    """Load CSI amplitudes from a Nexmon PCAP file.

    Returns an array of shape (frames, subcarriers) with amplitude values.
    """
    frames: List[np.ndarray] = []
    for payload in _iter_pcap_udp_payloads(path):
        iq = _extract_nexmon_csi_from_payload(payload, expected=expect_subcarriers)
        if iq is None:
            continue
        amp = np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2)
        frames.append(amp)
        if max_frames is not None and len(frames) >= max_frames:
            break
    if not frames:
        raise RuntimeError("No CSI frames parsed from PCAP. Check file/format.")
    # Ensure consistent subcarrier count
    nsc = min(f.shape[0] for f in frames)
    frames = [f[:nsc] for f in frames]
    return np.stack(frames, axis=0)


def plot_csi_heatmap(csi_amp: np.ndarray,
                     save: Optional[str] = None,
                     show: bool = False,
                     log_amp: bool = True,
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     title: Optional[str] = None) -> None:
    """Plot heatmap: x=frames, y=subcarriers for amplitude matrix."""
    import matplotlib
    # Force a non-interactive backend for headless environments
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    M = np.asarray(csi_amp, dtype=np.float32)
    if log_amp:
        M_plot = 20.0 * np.log10(np.maximum(M, 1.0))
    else:
        M_plot = M

    plt.figure(figsize=(9, 4))
    im = plt.imshow(
        M_plot.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("Frame")
    plt.ylabel("Subcarrier index")
    if title:
        plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("Amplitude (dB)" if log_amp else "Amplitude")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200)
        print(f"Saved heatmap to: {save}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser("Inspect WiFi CSI datasets")
    # Dataset inspection
    parser.add_argument("--dataset", choices=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar"], help="Built-in dataset to inspect")
    parser.add_argument("--root", default="./Data/", help="Path to Data root directory")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Split for NTU-Fi/Widar datasets")

    # PCAP CSI options
    parser.add_argument("--pcap", type=str, default=None, help="Path to Nexmon CSI PCAP (e.g., output2.pcap)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to parse from PCAP")
    parser.add_argument("--no-log", action="store_true", help="Plot linear amplitude instead of dB scale")
    parser.add_argument("--save", type=str, default=None, help="Path to save heatmap image (e.g., csi_heatmap.png)")
    parser.add_argument("--show", action="store_true", help="Display heatmap interactively")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # If PCAP path is given, parse and visualize CSI
    if args.pcap:
        print(f"Parsing PCAP: {args.pcap}")
        csi_amp = load_csi_from_pcap(args.pcap, max_frames=args.max_frames)
        print(f"Parsed CSI amplitude matrix: {stats_str(csi_amp)}")
        title = os.path.basename(args.pcap)
        plot_csi_heatmap(
            csi_amp,
            save=args.save,
            show=args.show,
            log_amp=not args.no_log,
            title=title,
        )
        return

    if args.dataset == "UT_HAR_data":
        print("Inspecting UT_HAR_data ...")
        inspect_ut_har(args.root)
    elif args.dataset in ("NTU-Fi_HAR", "NTU-Fi-HumanID"):
        print(f"Inspecting {args.dataset} [{args.split}] ...")
        dataset_root = os.path.join(args.root, args.dataset)
        inspect_ntu_fi(dataset_root, "train" if args.split == "train" else "test")
    elif args.dataset == "Widar":
        print(f"Inspecting Widar [{args.split}] ...")
        inspect_widar(args.root, args.split)
    else:
        print("No action specified. Use --pcap to parse a PCAP or --dataset to inspect a dataset.")


if __name__ == "__main__":
    main()
