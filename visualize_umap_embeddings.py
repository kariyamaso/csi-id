#!/usr/bin/env python3
"""UMAP visualization of representations for GRU, ViT, and Mamba.

This script extracts intermediate embeddings from selected models on a chosen
dataset split and runs UMAP to project them into 2D for visualization.

Defaults are tuned for NTU-Fi HumanID and the SenseFi-style model zoo in this
repo. It can automatically load pretrained checkpoints from `model_pt/` when
available, falling back to randomly initialized weights otherwise.

Usage examples
--------------
source .venv/bin/activate

# NTU-Fi HumanID, compare GRU/ViT/Mamba on the evaluation split
python visualize_umap_embeddings.py \
  --dataset NTU-Fi-HumanID \
  --models GRU ViT Mamba \
  --out-dir figures/ntu_fi_results

# If you have custom checkpoints (optional)
python visualize_umap_embeddings.py \
  --dataset NTU-Fi-HumanID \
  --models GRU ViT \
  --checkpoint-dir model_pt

Notes
-----
- Requires `umap-learn` and `scikit-learn` (install via pip).
- For NTU-Fi HumanID, `util.load_data_n_model` returns the evaluation set as
  `test_loader` (mapped to `train_amp`), which matches the paper's protocol.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Dict, Iterable, List, Tuple, Optional
import glob

import numpy as np
import torch
import torch.nn as nn

# Configure cache dirs prior to importing Matplotlib so fontconfig does not try
# to write into read-only system locations.
cache_root = pathlib.Path(".cache")
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
cache_root.mkdir(parents=True, exist_ok=True)
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
(cache_root / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FC_CACHEDIR", str(cache_root / "fontconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio  # for APPLIED normalization
from matplotlib.lines import Line2D

try:
    from umap import UMAP
except Exception as e:  # pragma: no cover - optional dependency
    print(
        f"[error] UMAP import failed: {e!r}.\n"
        "Try installing compatible versions, e.g.\n"
        "  pip install 'umap-learn>=0.5.4' 'scikit-learn==1.0.2' 'pandas==1.3.5'\n",
        file=sys.stderr,
    )
    raise

from util import load_data_n_model
import dataset as csi_dataset
from NTU_Fi_model import NTU_Fi_GRU, NTU_Fi_ViT, NTU_Fi_Mamba


def _auto_checkpoint_path(dataset: str, model_name: str, ckpt_dir: pathlib.Path) -> pathlib.Path:
    # Standard naming used in this repo's `model_pt/` directory
    return ckpt_dir / f"{dataset}_{model_name}.pt"


@torch.no_grad()
def extract_features(
    model: nn.Module, inputs: torch.Tensor
) -> torch.Tensor:
    """Return a batch of embeddings before the final classifier.

    Handles GRU, ViT, and Mamba as defined in NTU_Fi_model.py.
    """
    model.eval()
    if isinstance(model, NTU_Fi_GRU):
        # Follow NTU_Fi_GRU.forward up to ht[-1]
        x = inputs.view(-1, 342, 500)
        x = x.permute(2, 0, 1)  # 500 x batch x 342
        _, ht = model.gru(x)
        feats = ht[-1]  # (batch, hidden)
        return feats

    if isinstance(model, NTU_Fi_ViT):
        # model = nn.Sequential(PatchEmbedding, TransformerEncoder, ClassificationHead)
        # We take the output after Reduce + LayerNorm inside ClassificationHead, before final Linear.
        tokens = model[0](inputs)        # PatchEmbedding
        tokens = model[1](tokens)        # TransformerEncoder
        reduced = model[2][0](tokens)    # Reduce('b n e -> b e')
        feats = model[2][1](reduced)     # LayerNorm
        return feats

    if isinstance(model, NTU_Fi_Mamba):
        # Follow NTU_Fi_Mamba.forward up to pooled, normalized sequence
        b = inputs.size(0)
        seq = inputs.view(b, 3 * 114, 500).permute(0, 2, 1)  # (b, 500, 342)
        seq = model.input_proj(seq)
        for block in model.blocks:
            seq = block(seq)
        feats = model.norm(seq).mean(dim=1)
        return feats

    raise TypeError(
        f"Unsupported model type for feature extraction: {type(model).__name__}"
    )


def collect_embeddings(
    model: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    limit: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    model.to(device)
    n = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            feats = extract_features(model, inputs).detach().cpu().numpy()
            xs.append(feats)
            ys.append(labels.numpy())
            n += inputs.size(0)
            if limit is not None and n >= limit:
                break
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if limit is not None and len(y) > limit:
        X = X[:limit]
        y = y[:limit]
    return X, y


def fit_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> np.ndarray:
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(X)


def _compute_class_order(class_ids: Iterable[int], class_names: Optional[Dict[int, str]] = None) -> List[int]:
    """Return class ids ordered by their display name (e.g., '001'..'015')."""
    def key_fn(cid: int):
        name = None
        if class_names and int(cid) in class_names:
            name = class_names[int(cid)]
        if name is None:
            # fallback to numeric id
            return (0, int(cid))
        # Try numeric sort based on the display name if it's digits like '001'
        return (0, int(name)) if name.isdigit() else (1, str(name))

    return [int(c) for c in sorted(set(int(i) for i in class_ids), key=key_fn)]


def plot_umap(
    Z: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: pathlib.Path,
    color_map: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    class_names: Optional[Dict[int, str]] = None,
    class_order: Optional[List[int]] = None,
) -> None:
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    classes_present = list(np.unique(y))
    if class_order is None:
        classes = _compute_class_order(classes_present, class_names)
    else:
        # keep only those present, preserve provided order
        classes = [cid for cid in class_order if cid in set(int(c) for c in classes_present)]
    cmap = plt.get_cmap("tab20")
    for cls in classes:
        idx = (y == cls)
        color = (
            color_map.get(int(cls)) if (color_map is not None and int(cls) in color_map) else cmap(int(cls) % cmap.N)
        )
        ax.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=8,
            alpha=0.8,
            label=(class_names[int(cls)] if class_names and int(cls) in class_names else str(int(cls))),
            color=color,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.2)
    # Place legend outside to avoid overlapping the scatter
    ax.legend(
        title="Class",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        fontsize="small",
        borderaxespad=0,
    )
    plt.tight_layout(rect=(0, 0, 0.86, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def plot_comparison_grid(
    panels: List[Tuple[str, np.ndarray, np.ndarray]],
    out_path: pathlib.Path,
    color_map: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    class_names: Optional[Dict[int, str]] = None,
    class_order: Optional[List[int]] = None,
) -> None:
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5))
    if cols == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab20")
    for ax, (title, Z, y) in zip(axes, panels):
        classes_present = list(np.unique(y))
        if class_order is None:
            classes = _compute_class_order(classes_present, class_names)
        else:
            classes = [cid for cid in class_order if cid in set(int(c) for c in classes_present)]
        for cls in classes:
            idx = (y == cls)
            color = (
                color_map.get(int(cls)) if (color_map is not None and int(cls) in color_map) else cmap(int(cls) % cmap.N)
            )
            ax.scatter(
                Z[idx, 0],
                Z[idx, 1],
                s=8,
                alpha=0.8,
                label=(class_names[int(cls)] if class_names and int(cls) in class_names else str(int(cls))),
                color=color,
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.2)
    # Build a unified legend in bottom-right with the requested order
    if class_order is None:
        # derive from all panels
        all_ids: List[int] = []
        for _, _, y in panels:
            all_ids.extend(list(np.unique(y)))
        class_order = _compute_class_order(all_ids, class_names)

    handles: List[Line2D] = []
    labels: List[str] = []
    for cid in class_order:
        color = (
            color_map.get(int(cid)) if (color_map is not None and int(cid) in color_map) else cmap(int(cid) % cmap.N)
        )
        label = class_names[int(cid)] if class_names and int(cid) in class_names else str(int(cid)).zfill(3)
        handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=color, markersize=6, label=label))
        labels.append(label)

    fig.legend(
        handles,
        labels,
        title="Class",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        fontsize="small",
        borderaxespad=0,
    )
    plt.tight_layout(rect=(0, 0, 0.88, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _extract_class_names_from_loader(loader) -> Optional[Dict[int, str]]:
    """Try to get stable class names from the dataset.

    For CSI_Dataset we can use the `category` dict (name->id). We invert it to
    id->name for legend labels. If not available, return None.
    """
    ds = getattr(loader, "dataset", None)
    # Handle ConcatDataset by peeking the first child with `category`
    if ds is not None and hasattr(ds, "datasets"):
        for child in ds.datasets:
            if hasattr(child, "category") and isinstance(child.category, dict):
                inv = {int(v): str(k) for (k, v) in child.category.items()}
                return inv
    if ds is not None and hasattr(ds, "category") and isinstance(ds.category, dict):
        inv = {int(v): str(k) for (k, v) in ds.category.items()}
        return inv
    return None


def _build_color_map(class_ids: Iterable[int], class_names: Optional[Dict[int, str]] = None) -> Dict[int, Tuple[float, float, float, float]]:
    """Stable color assignment for each class id using tab20/tab20b/tab20c.

    The assignment follows the provided class order (when possible via class_names),
    ensuring colors map consistently to '001'..'015'.
    """
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    colors: Dict[int, Tuple[float, float, float, float]] = {}
    classes = _compute_class_order(class_ids, class_names)
    idx = 0
    for cls in classes:
        cmap = cmaps[(idx // 20) % len(cmaps)]
        colors[cls] = cmap(idx % 20)
        idx += 1
    return colors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="NTU-Fi-HumanID",
        choices=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar", "APPLIED"],
        help="Dataset name passed to util.load_data_n_model.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["GRU", "ViT", "Mamba"],
        help="Models to visualize (subset of repo-supported names).",
    )
    parser.add_argument(
        "--use-train",
        action="store_true",
        help="Use the training loader instead of the evaluation loader.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap samples for speed/memory. Default=0 uses all samples.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=pathlib.Path,
        default=pathlib.Path("model_pt"),
        help="Directory containing <dataset>_<model>.pt checkpoints.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("figures/ntu_fi_results"),
        help="Directory to save UMAP figures.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="UMAP distance metric.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure normalization for APPLIED dataset, mirroring run.py behavior
    data_root = "./Data/"
    if (
        args.dataset == "APPLIED"
        and not (os.environ.get("NTU_FI_NORM_MEAN") and os.environ.get("NTU_FI_NORM_STD"))
    ):
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
                x = np.asarray(mat["CSIamp"], dtype=np.float64)
                # Accept (S,T) or (1,S,T) or (3,S,T)
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

    # Load a representative loader (we don't train here). Choose the first
    # model that can be initialized for this dataset to obtain the loaders.
    train_loader = test_loader = None
    for name in args.models + ["GRU", "ViT", "LeNet", "ResNet18"]:  # fallbacks
        try:
            tl, vl, _, _ = load_data_n_model(args.dataset, name, "./Data/")
            train_loader, test_loader = tl, vl
            break
        except Exception:
            continue
    if train_loader is None or test_loader is None:
        raise RuntimeError(
            f"Could not obtain data loaders for dataset {args.dataset}."
        )
    loader = train_loader if args.use_train else test_loader

    # Prepare outputs
    panels: List[Tuple[str, np.ndarray, np.ndarray]] = []
    collected_class_ids: List[int] = []
    for model_name in args.models:
        # Re-initialize model via util to ensure correct architecture
        try:
            _, _, model, _ = load_data_n_model(args.dataset, model_name, "./Data/")
        except Exception as e:
            print(f"[warn] Skipping {model_name}: failed to initialize model ({e})")
            continue

        # Try to load checkpoint if available
        ckpt_path = _auto_checkpoint_path(args.dataset, model_name, args.checkpoint_dir)
        if ckpt_path.exists():
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print(f"[info] Loaded checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"[warn] Failed to load {ckpt_path}: {e}")
        else:
            print(f"[warn] No checkpoint found for {model_name} at {ckpt_path} (using random init)")

        limit = None if args.limit == 0 else args.limit
        X, y = collect_embeddings(model, loader, device=device, limit=limit)
        Z = fit_umap(
            X,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed,
        )

        panels.append((model_name, Z, y))
        collected_class_ids.extend(list(np.unique(y)))

    # Combined comparison figure
    # Build stable color map and (optional) class names
    class_names = _extract_class_names_from_loader(loader)
    class_order = _compute_class_order(collected_class_ids, class_names)
    color_map = _build_color_map(class_order, class_names)

    # Write individual panels using the same color map/names
    for model_name, Z, y in panels:
        title = f"{args.dataset} — {model_name}"
        out_path = args.out_dir / f"umap_{args.dataset}_{model_name}.png"
        plot_umap(Z, y, title, out_path, color_map=color_map, class_names=class_names, class_order=class_order)
        print(f"[ok] Wrote {out_path}")

    # Combined comparison figure
    comp_path = args.out_dir / f"umap_{args.dataset}_comparison.png"
    plot_comparison_grid(panels, comp_path, color_map=color_map, class_names=class_names, class_order=class_order)
    print(f"[ok] Wrote {comp_path}")


if __name__ == "__main__":
    main()
