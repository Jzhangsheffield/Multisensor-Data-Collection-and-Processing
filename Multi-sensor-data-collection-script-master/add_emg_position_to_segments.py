#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rename segment folders to inject EMG position (mid/elbow) based on run mapping.

Input layout (per subject root):
  <subject_root>/               e.g., N, MR, M, J
    <action_folder>/
      <segment_folder>/         e.g., run_10_clip_000008_left_npy
      <segment_folder>/         e.g., run_8-37_clip_000030_normal
      ...

You provide a CSV mapping: subject, run, emg_pos (mid/elbow)

Renaming rules:
- If segment folder already contains 'mid' or 'elbow' token -> skip
- If name ends with '_npy' (i.e., last token == 'npy'), insert emg_pos before 'npy'
  e.g. run_10_clip_..._left_npy -> run_10_clip_..._left_mid_npy
- Otherwise, append emg_pos at end
  e.g. run_10_clip_..._left -> run_10_clip_..._left_mid

Safety:
- dry-run mode prints planned renames without changing anything.
- collision protection: if target name exists, skip (or optionally auto-suffix).
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


EMG_POS = {"mid", "elbow"}
RUN_RE = re.compile(r"\brun_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class RunKey:
    subject: str
    run: int


def parse_run_id_from_segment(name: str) -> Optional[int]:
    """
    Extract run id from segment folder name.
    Examples:
      run_10_clip_... -> 10
      run_8-37_clip_... -> 8  (takes first integer after run_)
    """
    m = RUN_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def load_mapping_csv(csv_path: Path) -> Dict[RunKey, str]:
    """
    Load mapping: (subject, run) -> emg_pos
    CSV columns: subject, run, emg_pos
    """
    mapping: Dict[RunKey, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"subject", "run", "emg_pos"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must contain columns: {sorted(required)}. Got: {reader.fieldnames}")

        for row in reader:
            subject = (row["subject"] or "").strip()
            run_s = (row["run"] or "").strip()
            emg = (row["emg_pos"] or "").strip().lower()

            if not subject:
                continue
            if not run_s.isdigit():
                raise ValueError(f"Invalid run value: {row['run']} (subject={subject})")
            run = int(run_s)

            if emg not in EMG_POS:
                raise ValueError(f"Invalid emg_pos: {row['emg_pos']} (subject={subject}, run={run})")

            mapping[RunKey(subject=subject, run=run)] = emg

    return mapping


def build_new_name(old_name: str, emg_pos: str) -> Optional[str]:
    """
    Return new folder name after inserting/appending emg_pos.
    If already contains emg_pos token ('mid' or 'elbow'), return None (skip).
    """
    toks = old_name.split("_")

    # if already has any emg token, skip
    lower_toks = [t.lower() for t in toks]
    if "mid" in lower_toks or "elbow" in lower_toks:
        return None

    # insert before trailing 'npy' token if present
    if lower_toks and lower_toks[-1] == "npy":
        new_toks = toks[:-1] + [emg_pos] + toks[-1:]
        return "_".join(new_toks)

    # else append at end
    return old_name + "_" + emg_pos


def rename_segments_in_subject(
    subject_root: Path,
    mapping: Dict[RunKey, str],
    dry_run: bool,
    auto_suffix_on_conflict: bool,
) -> Tuple[int, int, int]:
    """
    Traverse subject_root/action/segment and rename segment folders.

    Returns: (scanned, renamed, skipped)
    """
    subject = subject_root.name
    scanned = renamed = skipped = 0

    if not subject_root.is_dir():
        print(f"[WARN] Not a directory: {subject_root}")
        return scanned, renamed, skipped

    for action_dir in subject_root.iterdir():
        if not action_dir.is_dir():
            continue

        for seg_dir in action_dir.iterdir():
            if not seg_dir.is_dir():
                continue

            scanned += 1
            run_id = parse_run_id_from_segment(seg_dir.name)
            if run_id is None:
                skipped += 1
                continue

            key = RunKey(subject=subject, run=run_id)
            emg_pos = mapping.get(key)
            if emg_pos is None:
                # no mapping for this (subject, run)
                skipped += 1
                continue

            new_name = build_new_name(seg_dir.name, emg_pos)
            if new_name is None or new_name == seg_dir.name:
                skipped += 1
                continue

            dst = seg_dir.parent / new_name
            if dst.exists():
                if not auto_suffix_on_conflict:
                    print(f"[SKIP-CONFLICT] {seg_dir} -> {dst} (target exists)")
                    skipped += 1
                    continue

                # auto suffix: _dup1, _dup2...
                k = 1
                while True:
                    candidate = seg_dir.parent / f"{new_name}_dup{k}"
                    if not candidate.exists():
                        dst = candidate
                        break
                    k += 1

            if dry_run:
                print(f"[DRY] {seg_dir.name}  ->  {dst.name}")
                renamed += 1
            else:
                print(f"[REN] {seg_dir.name}  ->  {dst.name}")
                seg_dir.rename(dst)
                renamed += 1

    return scanned, renamed, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subject_roots",
        nargs="+",
        required=True,
        help="Subject root paths, e.g. D:\\...\\N D:\\...\\MR D:\\...\\M D:\\...\\J"
    )
    ap.add_argument(
        "--run_map_csv",
        type=str,
        required=True,
        help="CSV mapping file with columns: subject,run,emg_pos"
    )
    ap.add_argument("--dry_run", action="store_true", help="Print planned renames, do not rename.")
    ap.add_argument(
        "--auto_suffix_on_conflict",
        action="store_true",
        help="If target name exists, auto append _dup1/_dup2... instead of skipping."
    )
    args = ap.parse_args()

    mapping = load_mapping_csv(Path(args.run_map_csv))
    subject_roots = [Path(p) for p in args.subject_roots]

    total_scanned = total_renamed = total_skipped = 0

    for sroot in subject_roots:
        print("\n" + "=" * 90)
        print(f"[SUBJECT ROOT] {sroot}")
        scanned, renamed, skipped = rename_segments_in_subject(
            subject_root=sroot,
            mapping=mapping,
            dry_run=args.dry_run,
            auto_suffix_on_conflict=args.auto_suffix_on_conflict,
        )
        print(f"[STATS] scanned={scanned}, renamed={renamed}, skipped={skipped}")
        total_scanned += scanned
        total_renamed += renamed
        total_skipped += skipped

    print("\n" + "=" * 90)
    print(f"[TOTAL] scanned={total_scanned}, renamed={total_renamed}, skipped={total_skipped}")
    if args.dry_run:
        print("Dry-run finished (no changes made).")


if __name__ == "__main__":
    main()
