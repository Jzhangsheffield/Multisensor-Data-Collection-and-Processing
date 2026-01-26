#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pack MindRove clip.csv (left/right) into npy files and write a JSON index.

Assumptions:
- Each clip folder contains at most ONE csv per side: left/clip.csv, right/clip.csv
- No need to control or align row/column counts
- Just convert each CSV to a pure numeric npy array

Input example:
  .../mindrove/J/adjust_slider/run_3_clip_000003/left/clip.csv
  .../mindrove/J/adjust_slider/run_3_clip_000003/right/clip.csv

Output:
  .../mindrove_npy/J/adjust_slider/run_3_clip_000003_npy/left/clip.npy
  .../mindrove_npy/J/adjust_slider/run_3_clip_000003_npy/right/clip.npy

Index JSON:
  key: J_adjust_slider_run_3_clip_000003_npy
  meta:
    {
      "left_npy": "...",
      "right_npy": "...",
      "left_shape": [N, C],
      "right_shape": [N, C],
      "dtype": "float32"
    }
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd


# ----------------- Utils -----------------

def parse_dtype(dtype_str: str) -> np.dtype:
    s = dtype_str.lower()
    if s in ("float16", "fp16"):
        return np.float16
    if s in ("float32", "fp32"):
        return np.float32
    if s in ("float64", "fp64"):
        return np.float64
    raise ValueError("dtype must be one of: float16, float32, float64")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_clip_csv(p: Path) -> bool:
    return p.is_file() and p.name.lower() == "clip.csv"


def find_clip_dirs(root: Path) -> List[Path]:
    """
    Find all clip directories that contain left/clip.csv or right/clip.csv
    """
    clip_dirs = set()
    for csv_path in root.rglob("clip.csv"):
        if not is_clip_csv(csv_path):
            continue
        side = csv_path.parent.name.lower()
        if side in ("left", "right"):
            clip_dirs.add(csv_path.parent.parent)
    return sorted(clip_dirs)


def make_key_from_rel(rel_clip_dir: Path) -> str:
    return "_".join(rel_clip_dir.parts) + "_npy"


def out_clip_dir(in_root: Path, out_root: Path, clip_dir: Path) -> Path:
    rel = clip_dir.relative_to(in_root)
    parts = list(rel.parts)
    parts[-1] = parts[-1] + "_npy"
    return out_root.joinpath(*parts)


def read_csv_to_numeric_npy(csv_path: Path, dtype: np.dtype) -> np.ndarray:
    """
    Read CSV and return pure numeric numpy array.
    Non-numeric columns will be automatically dropped.
    """
    df = pd.read_csv(csv_path)
    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {csv_path}")
    arr = df_num.to_numpy(dtype=dtype)
    return arr


# ----------------- Per clip -----------------

def process_one_clip(in_root: Path, out_root: Path, clip_dir: Path, dtype: np.dtype, dry_run=False):
    rel = clip_dir.relative_to(in_root)
    key = make_key_from_rel(rel)

    out_dir = out_clip_dir(in_root, out_root, clip_dir)
    meta: Dict[str, Any] = {
        "left_npy": None,
        "right_npy": None,
        "left_shape": None,
        "right_shape": None,
        "dtype": str(dtype),
        "src_dir": str(clip_dir)
    }

    for side in ("left", "right"):
        csv_path = clip_dir / side / "clip.csv"
        if not csv_path.exists():
            continue

        npy_path = out_dir / side / "clip.npy"
        meta[f"{side}_npy"] = str(npy_path)

        if dry_run:
            continue

        arr = read_csv_to_numeric_npy(csv_path, dtype)
        ensure_dir(npy_path.parent)
        np.save(npy_path, arr, allow_pickle=False)

        meta[f"{side}_shape"] = list(arr.shape)

    return key, meta


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser("Convert MindRove clip.csv to npy and build index.json")
    ap.add_argument("--in-root", type=Path, required=True,
                    help="Input root: .../Thermal_Crimping_Dataset/mindrove")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Output root: .../Thermal_Crimping_Dataset/mindrove_npy")
    ap.add_argument("--dtype", type=str, default="float32",
                    help="float16|float32|float64")
    ap.add_argument("--index-name", type=str, default="index.json")
    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    dtype = parse_dtype(args.dtype)

    ensure_dir(args.out_root)

    index_path = args.out_root / args.index_name
    index: Dict[str, Any] = {}

    clip_dirs = find_clip_dirs(args.in_root)
    print(f"Found {len(clip_dirs)} clip directories.")

    packed = 0
    failed = 0

    for clip_dir in clip_dirs:
        try:
            key, meta = process_one_clip(
                in_root=args.in_root,
                out_root=args.out_root,
                clip_dir=clip_dir,
                dtype=dtype,
                dry_run=args.dry_run
            )
            index[key] = meta
            packed += 1
            print(f"[OK] {key}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {clip_dir}: {e}")

    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. packed={packed}, failed={failed}")
    print(f"Index saved: {index_path}")


if __name__ == "__main__":
    main()
