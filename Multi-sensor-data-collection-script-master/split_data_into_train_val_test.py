#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split dataset by run, stratified by (emg_pos, lighting), and copy segments.

Input layout (per subject root):
  subject_root/   (e.g., N, MR, M, J)
    cut_plastic_tape/
      run_10_clip_000008_left_mid_npy/
        ... (may contain subfolders/files)
      run_8-37_clip_000030_normal_elbow_npy/
        ...
    another_action/
      ...

We traverse only to segment folders (action_dir/segment_dir), parse:
- run_id: first integer after "run_"
- lighting: normal|left|right
- emg_pos: mid|elbow

Split rule (recommended main split):
- For each subject, for each (emg_pos, lighting) group:
  expected 6 runs -> 4 train, 1 val, 1 test (balanced)
- Keep all segments from the same run in the same split (no leakage).

Output layout:
  out_root/
    train/<action>/<subject>_<segment_folder>
    val/<action>/<subject>_<segment_folder>
    test/<action>/<subject>_<segment_folder>

Note:
- Segment directories are copied as whole trees (including subfolders).
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


LIGHTINGS = {"normal", "left", "right"}
EMG_POS = {"mid", "elbow"}

RUN_RE = re.compile(r"\brun_(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class SegInfo:
    subject: str
    action: str
    seg_path: Path
    seg_name: str
    run_id: int
    lighting: str
    emg_pos: str


def parse_seg_folder_name(name: str) -> Optional[Tuple[int, str, str]]:
    """
    Parse segment folder name -> (run_id, lighting, emg_pos).

    Examples:
      run_10_clip_000008_left_mid_npy -> (10, "left", "mid")
      run_8-37_clip_000030_normal_elbow_npy -> (8, "normal", "elbow")

    Returns None if any field cannot be found.
    """
    m = RUN_RE.search(name)
    if not m:
        return None
    run_id = int(m.group(1))

    toks = name.lower().split("_")
    lighting = next((t for t in toks if t in LIGHTINGS), None)
    emg_pos = next((t for t in toks if t in EMG_POS), None)

    if lighting is None or emg_pos is None:
        return None
    return run_id, lighting, emg_pos


def safe_copytree(src: Path, dst: Path, overwrite: bool = False) -> None:
    """
    Copy a directory tree (whole segment folder, including subfolders).

    - If overwrite=False and dst exists -> skip
    - If overwrite=True and dst exists -> remove then copy
    """
    if dst.exists():
        if not overwrite:
            return
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def scan_subject_root(subject_root: Path) -> List[SegInfo]:
    """
    Scan: subject_root/action/segment_dir
    """
    subject = subject_root.name
    out: List[SegInfo] = []

    if not subject_root.is_dir():
        return out

    for action_dir in subject_root.iterdir():
        if not action_dir.is_dir():
            continue
        action = action_dir.name

        for seg_dir in action_dir.iterdir():
            if not seg_dir.is_dir():
                continue
            parsed = parse_seg_folder_name(seg_dir.name)
            if parsed is None:
                # ignore non-matching folders silently
                continue
            run_id, lighting, emg_pos = parsed
            out.append(SegInfo(
                subject=subject,
                action=action,
                seg_path=seg_dir,
                seg_name=seg_dir.name,
                run_id=run_id,
                lighting=lighting,
                emg_pos=emg_pos
            ))

    return out


def choose_run_condition(segs: List[SegInfo]) -> Dict[int, Tuple[str, str]]:
    """
    For each run_id, decide its (emg_pos, lighting).
    If a run_id appears with multiple conditions, choose the most common one and warn.
    """
    by_run: Dict[int, Counter] = defaultdict(Counter)
    for s in segs:
        by_run[s.run_id][(s.emg_pos, s.lighting)] += 1

    run2cond: Dict[int, Tuple[str, str]] = {}
    for run_id, cnt in by_run.items():
        (cond, _) = cnt.most_common(1)[0]
        run2cond[run_id] = cond
        if len(cnt) > 1:
            details = ", ".join([f"{k}:{v}" for k, v in cnt.most_common()])
            print(f"[WARN] run {run_id} has multiple conditions -> choose {cond} (counts: {details})")
    return run2cond


def stratified_run_split(
    runs: Set[int],
    run2cond: Dict[int, Tuple[str, str]],
    seed: int,
    per_group: Tuple[int, int, int] = (4, 1, 1),
    shuffle: bool = True,
) -> Dict[int, str]:
    """
    Split run_ids into train/val/test stratified by (emg_pos, lighting).
    Default expects each group has 6 runs and splits as 4/1/1.

    Returns: run_id -> split_name
    """
    train_n, val_n, test_n = per_group
    rng = random.Random(seed)

    cond2runs: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for r in runs:
        if r in run2cond:
            cond2runs[run2cond[r]].append(r)

    run2split: Dict[int, str] = {}

    for cond, rlist in sorted(cond2runs.items()):
        rlist = sorted(rlist)
        if shuffle:
            rng.shuffle(rlist)

        total = len(rlist)
        need = train_n + val_n + test_n

        if total < need:
            raise RuntimeError(
                f"Condition group {cond} has only {total} runs (< {need}). "
                f"Cannot split as {per_group}. Runs: {rlist}"
            )

        chosen = rlist[:need]
        extra = rlist[need:]

        train_runs = chosen[:train_n] + extra  # extras -> train
        val_runs = chosen[train_n:train_n + val_n]
        test_runs = chosen[train_n + val_n:train_n + val_n + test_n]

        for r in train_runs:
            run2split[r] = "train"
        for r in val_runs:
            run2split[r] = "val"
        for r in test_runs:
            run2split[r] = "test"

        print(f"[INFO] group {cond}: total={total} -> train={len(train_runs)}, val={len(val_runs)}, test={len(test_runs)}")

    return run2split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subject_roots",
        nargs="+",
        required=True,
        help="Subject root paths, e.g. ...\\N ...\\MR ...\\M ...\\J (each contains action folders)."
    )
    ap.add_argument("--out_root", type=str, required=True, help="Output root directory for train/val/test.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling runs in each condition group.")
    ap.add_argument("--no_shuffle", action="store_true", help="Do not shuffle runs; split by sorted run order.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing copied segment folders.")
    ap.add_argument("--dry_run", action="store_true", help="Only print stats, do not copy anything.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    subject_roots = [Path(p) for p in args.subject_roots]

    for sroot in subject_roots:
        subject = sroot.name  # expected: N, MR, M, J

        segs = scan_subject_root(sroot)
        if not segs:
            print(f"[WARN] No segment folders found under: {sroot}")
            continue

        print("\n" + "=" * 90)
        print(f"[SUBJECT] {subject}  ({sroot})")
        print(f"[FOUND] segments={len(segs)} | actions={len(set(s.action for s in segs))} | runs={len(set(s.run_id for s in segs))}")
        print("=" * 90)

        run2cond = choose_run_condition(segs)
        runs = set(run2cond.keys())

        cond_count = Counter(run2cond.values())
        for cond, c in sorted(cond_count.items()):
            print(f"[COND COUNT] {cond}: {c} runs")

        run2split = stratified_run_split(
            runs=runs,
            run2cond=run2cond,
            seed=args.seed,
            per_group=(4, 1, 1),
            shuffle=(not args.no_shuffle),
        )

        n_copied = Counter()

        for s in segs:
            split = run2split.get(s.run_id)
            if split is None:
                continue

            # NEW: merge subject into segment folder name
            new_seg_name = f"{subject}_{s.seg_name}"
            dst = out_root / split / s.action / new_seg_name

            if args.dry_run:
                n_copied[split] += 1
                continue

            # copy entire segment folder tree (including subfolders)
            safe_copytree(s.seg_path, dst, overwrite=args.overwrite)
            n_copied[split] += 1

        print(f"[DONE] {subject} -> " + ", ".join([f"{k}={v}" for k, v in n_copied.items()]))

    print("\nAll done.")


if __name__ == "__main__":
    main()
