#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pack LiDAR radar-frame CSVs into NPY (sorted by timestamp) and write a JSON index.

New features:
- --dtype {float16,float32,float64} to control numeric dtype stored in npy
- Works for both:
    (A) object array mode (no --pad): npy dtype=object, each frame is ndarray(dtype)
    (B) padded mode (--pad): npy is regular ndarray(dtype) with shape (T,max_rows,C)

Notes:
- Padding only handles varying ROW counts; columns must be consistent across frames.
- Timestamp sorting supports:
    - YYYYMMDD_HHMMSS_micro  (e.g., 20251105_172554_530685)
    - numeric epoch (s/ms/us/ns; float allowed)
"""

import re
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np


# ----------------- Timestamp parsing -----------------

def normalize_linux_timestamp_to_us(t) -> Optional[int]:
    if t is None:
        return None
    if isinstance(t, float):
        sec = int(t)
        micro = int(round((t - sec) * 1e6))
        return sec * 1_000_000 + micro
    try:
        t = int(t)
    except Exception:
        return None
    if t <= 0:
        return None
    if t > 10**17:      # ns
        return t // 1000
    elif t > 10**14:    # us
        return t
    elif t > 10**11:    # ms
        return t * 1000
    else:               # s
        return t * 1_000_000


def parse_timestamp_key_from_name(name: str) -> Optional[int]:
    s = str(name).strip()
    m = re.search(r"(\d{8}_\d{6}_\d{6})", s)
    if m:
        return int(m.group(1).replace("_", ""))

    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    if m2:
        token = m2.group(1)
        try:
            if "." in token:
                return normalize_linux_timestamp_to_us(float(token))
            return normalize_linux_timestamp_to_us(int(token))
        except Exception:
            return None
    return None


# ----------------- Utilities -----------------

def is_radar_frame_csv(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".csv"


def make_key_from_relpath(rel_dir: Path) -> str:
    parts = list(rel_dir.parts)
    return ("_".join(parts) + "_npy") if parts else "unknown_npy"


def out_dir_for_leaf(in_root: Path, out_root: Path, leaf_dir: Path) -> Path:
    rel = leaf_dir.relative_to(in_root)
    rel_parts = list(rel.parts)
    rel_parts[-1] = rel_parts[-1] + "_npy"
    return out_root.joinpath(*rel_parts)


def parse_dtype(dtype_str: str) -> np.dtype:
    dtype_str = str(dtype_str).strip().lower()
    if dtype_str in ("float16", "fp16", "f16"):
        return np.dtype(np.float16)
    if dtype_str in ("float32", "fp32", "f32"):
        return np.dtype(np.float32)
    if dtype_str in ("float64", "fp64", "f64", "double"):
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported --dtype: {dtype_str}. Use one of: float16, float32, float64")


def read_csv_to_array(csv_path: Path, dtype: np.dtype, delimiter: str = ",") -> np.ndarray:
    """
    Read numeric CSV to numpy array and cast to dtype.
    - Try without header; if fail, retry skipping first row.
    - Ensures 2D.
    """
    try:
        arr = np.loadtxt(csv_path, delimiter=delimiter)
    except Exception:
        arr = np.loadtxt(csv_path, delimiter=delimiter, skiprows=1)

    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.astype(dtype, copy=False)


def pad_frames_to_max_rows(frames: List[np.ndarray], dtype: np.dtype, pad_value: float = 0.0) -> Tuple[np.ndarray, int, int]:
    """
    Pad each frame (Ni, C) to (max_rows, C) using pad_value, returning stacked array (T,max_rows,C).
    Columns must be consistent across frames.
    """
    if not frames:
        raise ValueError("frames is empty")

    num_cols = frames[0].shape[1]
    for i, f in enumerate(frames):
        if f.ndim != 2:
            raise ValueError(f"Frame {i} is not 2D: shape={f.shape}")
        if f.shape[1] != num_cols:
            raise ValueError(
                f"Column count mismatch: frame0 C={num_cols}, frame{i} C={f.shape[1]} "
                f"(padding only handles varying rows; columns must be consistent)"
            )

    max_rows = max(f.shape[0] for f in frames)
    T = len(frames)
    stacked = np.full((T, max_rows, num_cols), pad_value, dtype=dtype)

    for i, f in enumerate(frames):
        n = f.shape[0]
        stacked[i, :n, :] = f.astype(dtype, copy=False)

    return stacked, max_rows, num_cols


def pack_one_leaf_folder(
    in_root: Path,
    out_root: Path,
    leaf_dir: Path,
    min_frames: int,
    pad: bool,
    pad_value: float,
    dtype: np.dtype,
    dry_run: bool = False,
) -> Optional[Tuple[str, Dict[str, Any]]]:

    csv_files = [p for p in leaf_dir.iterdir() if is_radar_frame_csv(p)]
    if len(csv_files) < min_frames:
        return None

    pairs: List[Tuple[int, Path]] = []
    for f in csv_files:
        ts = parse_timestamp_key_from_name(f.stem)
        if ts is None:
            continue
        pairs.append((ts, f))

    if len(pairs) < min_frames:
        return None

    pairs.sort(key=lambda x: x[0])
    sorted_files = [p for _, p in pairs]

    out_dir = out_dir_for_leaf(in_root, out_root, leaf_dir)
    npy_path = out_dir / "lidar_frames.npy"

    rel_dir = leaf_dir.relative_to(in_root)
    key = make_key_from_relpath(rel_dir)

    meta: Dict[str, Any] = {
        "npy_path": str(npy_path),
        "num_frames": len(sorted_files),
        "src_dir": str(leaf_dir),
        "padded": bool(pad),
        "dtype": str(dtype),
    }

    if dry_run:
        return key, meta

    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [read_csv_to_array(f, dtype=dtype) for f in sorted_files]

    if not pad:
        # object array (frames can have varying rows)
        frames_obj = np.empty((len(frames),), dtype=object)
        for i, arr in enumerate(frames):
            frames_obj[i] = arr  # each is dtype
        np.save(npy_path, frames_obj, allow_pickle=True)
        meta["storage"] = "object"
        meta["num_cols"] = int(frames[0].shape[1])
        meta["max_rows"] = None
        meta["pad_value"] = None
    else:
        stacked, max_rows, num_cols = pad_frames_to_max_rows(frames, dtype=dtype, pad_value=pad_value)
        np.save(npy_path, stacked, allow_pickle=False)
        meta["storage"] = "tensor"
        meta["num_cols"] = int(num_cols)
        meta["max_rows"] = int(max_rows)
        meta["pad_value"] = float(pad_value)

    return key, meta


def find_leaf_dirs_with_csv(in_root: Path) -> List[Path]:
    leafs = []
    for d in in_root.rglob("*"):
        if not d.is_dir():
            continue
        try:
            has_csv = any(is_radar_frame_csv(p) for p in d.iterdir())
        except Exception:
            continue
        if has_csv:
            leafs.append(d)
    return leafs


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser("Pack timestamp-named LiDAR frame CSVs into NPY and write JSON index.")
    ap.add_argument("--in-root", type=Path, required=True, help="Input root: .../vlp16")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root: .../vlp16_npy")
    ap.add_argument("--index-name", type=str, default="index.json", help="JSON filename under out-root")
    ap.add_argument("--min-frames", type=int, default=1, help="Min number of valid timestamp CSV frames in a folder to pack")
    ap.add_argument("--pad", action="store_true", help="Pad all frames to max rows and save as (T,max_rows,C) array")
    ap.add_argument("--pad-value", type=float, default=0.0, help="Padding fill value (default 0.0)")
    ap.add_argument("--dtype", type=str, default="float32", help="Data dtype: float16|float32|float64 (default float32)")
    ap.add_argument("--dry-run", action="store_true", help="Only scan and write index without saving npy")
    ap.add_argument("--print-skipped", action="store_true", help="Print skipped directories")
    args = ap.parse_args()

    dtype = parse_dtype(args.dtype)

    in_root = args.in_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    index_path = out_root / args.index_name
    index: Dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}

    leaf_dirs = find_leaf_dirs_with_csv(in_root)
    print(f"Found {len(leaf_dirs)} candidate folders containing CSV frames.")

    packed = 0
    skipped = 0
    failed = 0
    skipped_dirs: List[str] = []

    for leaf in leaf_dirs:
        try:
            res = pack_one_leaf_folder(
                in_root=in_root,
                out_root=out_root,
                leaf_dir=leaf,
                min_frames=args.min_frames,
                pad=args.pad,
                pad_value=args.pad_value,
                dtype=dtype,
                dry_run=args.dry_run,
            )
        except Exception as e:
            failed += 1
            print(f"[FAIL] {leaf}: {e}")
            continue

        if res is None:
            skipped += 1
            if args.print_skipped:
                skipped_dirs.append(str(leaf))
            continue

        key, meta = res
        index[key] = meta
        packed += 1
        print(f"[OK] {key} -> frames={meta['num_frames']} padded={meta['padded']} dtype={meta['dtype']}")

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. packed={packed}, skipped={skipped}, failed={failed}")
    print(f"Index saved: {index_path}")

    if args.print_skipped and skipped_dirs:
        print("\n[SKIPPED DIRECTORIES]")
        for d in skipped_dirs:
            print("  -", d)


if __name__ == "__main__":
    main()
