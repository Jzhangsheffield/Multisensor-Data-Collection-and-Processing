#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Remove trailing '_npy' from top-level JSON keys.

Example:
  "J_adjust_slider_run_11_clip_000003_left_npy"  ->  "J_adjust_slider_run_11_clip_000003_left"

Rules:
- Only remove when the key ends with exactly "_npy".
- If removing causes duplicate keys, later keys will overwrite earlier ones by default,
  and we will print a warning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def strip_npy_suffix(key: str) -> Tuple[str, bool]:
    """Return (new_key, changed?). Only strip if key endswith '_npy'."""
    if key.endswith("_npy"):
        return key[:-4], True  # remove the 4 chars: '_npy'
    return key, False


def process_one_json(in_path: Path) -> Tuple[Dict[str, Any], int, int, int]:
    """
    Returns:
      new_dict, n_total, n_changed, n_collisions
    """
    data = load_json(in_path)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level JSON must be an object/dict: {in_path}")

    out: Dict[str, Any] = {}
    n_total = 0
    n_changed = 0
    n_collisions = 0

    for k, v in data.items():
        n_total += 1
        new_k, changed = strip_npy_suffix(k)
        if changed:
            n_changed += 1
        if new_k in out and new_k != k:
            n_collisions += 1
        out[new_k] = v

    return out, n_total, n_changed, n_collisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", type=str, default="", help="Input JSON file")
    ap.add_argument("--in_dir", type=str, default="", help="Input directory containing *.json")
    ap.add_argument("--out_json", type=str, default="", help="Output JSON file (when using --in_json)")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory (when using --in_dir)")
    args = ap.parse_args()

    if not args.in_json and not args.in_dir:
        raise SystemExit("You must provide --in_json or --in_dir")

    if args.in_json:
        in_path = Path(args.in_json)
        if not in_path.exists():
            raise SystemExit(f"Input not found: {in_path}")

        if not args.out_json:
            raise SystemExit("When using --in_json, you must provide --out_json")

        out_path = Path(args.out_json)
        new_data, n_total, n_changed, n_collisions = process_one_json(in_path)
        save_json(new_data, out_path)

        print(f"✅ Processed: {in_path}")
        print(f"   Saved to:  {out_path}")
        print(f"   Keys: total={n_total}, changed={n_changed}, collisions={n_collisions}")
        return

    # Directory mode
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    if not args.out_dir:
        raise SystemExit("When using --in_dir, you must provide --out_dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No *.json files found in: {in_dir}")

    total_files = 0
    total_keys = 0
    total_changed = 0
    total_collisions = 0

    for in_path in json_files:
        total_files += 1
        out_path = out_dir / in_path.name

        new_data, n_total, n_changed, n_collisions = process_one_json(in_path)
        save_json(new_data, out_path)

        total_keys += n_total
        total_changed += n_changed
        total_collisions += n_collisions

        if n_collisions > 0:
            print(f"⚠️  Collisions in {in_path.name}: {n_collisions} (later keys overwrite earlier ones)")

    print(f"✅ Done. files={total_files}, keys={total_keys}, changed={total_changed}, collisions={total_collisions}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
