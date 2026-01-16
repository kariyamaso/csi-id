#!/usr/bin/env python3
"""Prepare datasets under `public/Data/` (symlink/copy/extract).

This is meant for distribution: keep `public/` self-contained while allowing
datasets to live outside version control.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import zipfile


def _link_or_copy(src: pathlib.Path, dst: pathlib.Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
        return
    if mode == "copy":
        shutil.copytree(src, dst)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def _extract_zip(zip_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--public-root",
        type=pathlib.Path,
        default=pathlib.Path("public"),
        help="Path to public folder.",
    )
    p.add_argument(
        "--src-root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Repo root (where Data/, UT_HAR/, Widardata/ and/or zips live).",
    )
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument(
        "--extract-missing",
        action="store_true",
        help="If a source directory is missing, try extracting from zip archives in src-root.",
    )
    args = p.parse_args()

    public_data = args.public_root / "Data"
    public_data.mkdir(parents=True, exist_ok=True)

    # Source locations in this repo
    src_ntu_humanid = args.src_root / "Data" / "NTU-Fi-HumanID"
    src_ntu_har = args.src_root / "Data" / "NTU-Fi_HAR"
    src_applied = args.src_root / "Data" / "APPLIED"
    src_ut_har = args.src_root / "UT_HAR"
    src_widar = args.src_root / "Widardata"

    if args.extract_missing:
        # Optional extraction from archives (if present)
        if not src_ut_har.exists():
            zip_path = args.src_root / "UT_HAR.zip"
            if zip_path.exists():
                _extract_zip(zip_path, args.src_root)
        if not src_widar.exists():
            zip_path = args.src_root / "Widardata.zip"
            if zip_path.exists():
                _extract_zip(zip_path, args.src_root)
        if not src_ntu_humanid.exists():
            zip_path = args.src_root / "NTU-Fi-HumanID.zip"
            if zip_path.exists():
                _extract_zip(zip_path, args.src_root)
        # NTU-Fi_HAR is already present as a directory in this repo; no zip handling here.

    mappings = [
        (src_ntu_humanid, public_data / "NTU-Fi-HumanID"),
        (src_ntu_har, public_data / "NTU-Fi_HAR"),
        (src_applied, public_data / "APPLIED"),
        (src_ut_har, public_data / "UT_HAR"),
        (src_widar, public_data / "Widardata"),
    ]

    missing = []
    for src, dst in mappings:
        if dst.exists() or dst.is_symlink():
            # Destination already prepared; don't require the source path.
            continue
        if not src.exists():
            missing.append(str(src))
            continue
        _link_or_copy(src, dst, args.mode)

    if missing:
        print("[warn] Missing sources (not linked/copied):")
        for m in missing:
            print(f"  - {m}")
        print("Place/extract datasets there, or re-run with --extract-missing if zips exist.")
    else:
        print(f"Prepared datasets under {public_data}")


if __name__ == "__main__":
    main()
