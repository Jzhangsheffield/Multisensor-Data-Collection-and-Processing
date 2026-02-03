#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare action folders across multiple roots.

Assumption:
- Each root directory contains action folders directly under it (1-level).
- We only check action folder names, NOT clip/segment subfolders.

Enhancement:
- Optionally compare only the FIRST token ("action word") of folder names.
  Example: put_long_wire -> put
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Set, List
import csv


def extract_action_word(folder_name: str, sep: str = "_", lower: bool = False) -> str:
    """
    Extract the first token as action word.
    e.g., "put_long_wire" -> "put"
    If no separator is found, returns the whole name.
    """
    name = folder_name.strip()
    if lower:
        name = name.lower()
    if not name:
        return name
    return name.split(sep, 1)[0]


def list_action_folders(
    root: Path,
    ignore_hidden: bool = True,
    compare_action_word: bool = False,
    sep: str = "_",
    lower: bool = False,
) -> Set[str]:
    """
    List first-level subdirectories.

    If compare_action_word=True:
      return set of action words (first token) extracted from folder names.
    else:
      return set of full folder names.
    """
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    actions: Set[str] = set()
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if ignore_hidden and name.startswith("."):
            continue

        key = extract_action_word(name, sep=sep, lower=lower) if compare_action_word else (name.lower() if lower else name)
        if key:  # guard: skip empty
            actions.add(key)

    return actions


def compute_unique(sets_by_root: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Actions that appear only in that root."""
    all_roots = list(sets_by_root.keys())
    unique: Dict[str, Set[str]] = {}
    for r in all_roots:
        others_union = set().union(*(sets_by_root[o] for o in all_roots if o != r)) if len(all_roots) > 1 else set()
        unique[r] = sets_by_root[r] - others_union
    return unique


def main():
    parser = argparse.ArgumentParser(description="Compare action folders across multiple roots (first-level dirs).")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Root paths. Example: D:\\...\\RGB\\J D:\\...\\Depth\\J D:\\...\\EMG\\J D:\\...\\LiDAR\\J"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Optional CSV output path, e.g. D:\\temp\\compare_actions_report.csv"
    )
    parser.add_argument(
        "--ignore_hidden",
        action="store_true",
        help="Ignore folders starting with '.'"
    )

    # NEW
    parser.add_argument(
        "--compare_action_word",
        action="store_true",
        help="Compare only the first token of folder names (e.g., put_long_wire -> put)."
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="_",
        help="Separator used to split action folder name into tokens (default: '_')."
    )
    parser.add_argument(
        "--lower",
        action="store_true",
        help="Convert compared keys to lowercase (avoid Put vs put)."
    )

    args = parser.parse_args()

    roots: List[Path] = [Path(r) for r in args.roots]
    sets_by_root: Dict[str, Set[str]] = {}

    for r in roots:
        actions = list_action_folders(
            r,
            ignore_hidden=args.ignore_hidden,
            compare_action_word=args.compare_action_word,
            sep=args.sep,
            lower=args.lower,
        )
        sets_by_root[str(r)] = actions

    # Union & intersection
    all_sets = list(sets_by_root.values())
    union_all = set().union(*all_sets) if all_sets else set()
    inter_all = set.intersection(*all_sets) if all_sets else set()

    # Missing per root
    missing_by_root: Dict[str, Set[str]] = {root: (union_all - acts) for root, acts in sets_by_root.items()}

    # Unique per root
    unique_by_root = compute_unique(sets_by_root)

    # -------- Print report --------
    print("=" * 80)
    print("Action folder comparison report")
    print("=" * 80)

    mode = "ACTION WORD ONLY" if args.compare_action_word else "FULL FOLDER NAME"
    print(f"\nCompare mode: {mode}")
    if args.compare_action_word:
        print(f"Split separator: '{args.sep}'")
    print(f"Lowercase: {bool(args.lower)}")

    print("\nRoots scanned:")
    for root, acts in sets_by_root.items():
        print(f"- {root}  (keys: {len(acts)})")

    print("\n[Common across ALL roots] (intersection)")
    print(f"Count: {len(inter_all)}")
    for a in sorted(inter_all):
        print(f"  - {a}")

    print("\n[All across ANY root] (union)")
    print(f"Count: {len(union_all)}")

    print("\n[Missing per root] (relative to union)")
    for root in sets_by_root:
        miss = missing_by_root[root]
        print(f"\n- Root: {root}")
        print(f"  Missing count: {len(miss)}")
        for a in sorted(miss):
            print(f"    - {a}")

    print("\n[Unique per root] (only appears in that root)")
    for root in sets_by_root:
        uniq = unique_by_root[root]
        print(f"\n- Root: {root}")
        print(f"  Unique count: {len(uniq)}")
        for a in sorted(uniq):
            print(f"    - {a}")

    # -------- Optional CSV --------
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        root_cols = list(sets_by_root.keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key"] + root_cols)
            for key in sorted(union_all):
                row = [key] + [("1" if key in sets_by_root[root] else "0") for root in root_cols]
                writer.writerow(row)

        print("\n" + "=" * 80)
        print(f"CSV written: {out_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
