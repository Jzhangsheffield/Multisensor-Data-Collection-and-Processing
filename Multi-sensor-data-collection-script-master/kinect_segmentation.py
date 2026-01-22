#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build dataset clips from run_* folders based on CSV annotations.

Folder layout (per run):
run_xxx/
  001431512812/
    frames_rgb_blured/   (front/back/side mapping doesn't matter here)
      20251105_172554_530685.jpg ...
    frames_rgb/
      ...
  001484412812/
    frames_rgb/
      20251105_172554_530685.jpg ...
    frames_rgb_blured/
      ...
  001528512812/          (ignored by default)
  N_run_1_annotation.csv  (or *annotation.csv)

CSV rows may be:
- With header: columns include action, object, start, end, start_ts, end_ts (names may vary)
- Without header: columns are:
  idx, action, object, start, end, start_ts, end_ts, comment

Timestamp formats supported for frame filenames and CSV:
- "YYYYMMDD_HHMMSS_micro" (e.g., 20251105_172554_530685)
- Linux epoch timestamps in seconds / ms / us / ns (e.g., 1730816754, 1730816754.530685, 1730816754530685000)

Output dataset:
out_root/
  action_object_class/
    run_xxx_clip_000001/
      cam_001431512812/ frame_000001.jpg ...
      cam_001484412812/ frame_000001.jpg ...
(Optionally --flat-clip puts both cams into same folder with prefixes.)
"""

import re
import csv
import shutil
import argparse
from pathlib import Path
from bisect import bisect_left, bisect_right
from typing import Optional, Tuple, List, Dict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ----------------- Timestamp parsing (unified to microseconds int) -----------------

def normalize_linux_timestamp_to_us(t) -> Optional[int]:
    """
    Normalize Linux epoch timestamps to microseconds (int).
    Accepts int/float/string-like numeric (already converted before calling).
    Heuristics based on magnitude:
      - seconds: ~1e9..1e10  -> *1e6
      - milliseconds: ~1e12..1e13 -> *1e3
      - microseconds: ~1e15..1e16 -> as is
      - nanoseconds: ~1e18.. -> //1e3
    For float seconds with fractional part -> convert precisely to us.
    """
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

    if t > 10**17:      # nanoseconds
        return t // 1000
    elif t > 10**14:    # microseconds
        return t
    elif t > 10**11:    # milliseconds
        return t * 1000
    else:               # seconds
        return t * 1_000_000


def parse_timestamp_to_us(x) -> Optional[int]:
    """
    Convert a timestamp (filename stem / csv field) to microseconds int.

    Supports:
      1) 'YYYYMMDD_HHMMSS_micro' (e.g., 20251105_172554_530685) -> 20251105172554530685 (int)
         NOTE: This is NOT epoch us; it's a sortable compact datetime+micro representation.
         It's still totally fine for range searching as long as both sides use same style.
      2) Linux epoch numeric timestamps in seconds/ms/us/ns (including floats with fractional seconds).
      3) Strings containing the above patterns (e.g., with extensions or prefixes).
    """
    if x is None:
        return None

    # Numeric direct
    if isinstance(x, (int, float)):
        # We treat numeric as Linux epoch timestamps
        return normalize_linux_timestamp_to_us(x)

    s = str(x).strip()
    if not s:
        return None

    # Case A: YYYYMMDD_HHMMSS_micro anywhere in string
    m = re.search(r"(\d{8}_\d{6}_\d{6})", s)
    if m:
        return int(m.group(1).replace("_", ""))

    # Case B: a pure numeric token -> linux epoch
    # Extract first numeric-like token (supports "1730816754.530685")
    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    if m2:
        num_s = m2.group(1)
        try:
            if "." in num_s:
                return normalize_linux_timestamp_to_us(float(num_s))
            else:
                return normalize_linux_timestamp_to_us(int(num_s))
        except Exception:
            return None

    return None


def extract_timestamp_from_filename(fp: Path) -> Optional[int]:
    # Use stem (no extension). Pattern may still include extra bits; parser handles that.
    return parse_timestamp_to_us(fp.stem)


# ----------------- Helpers -----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_run_dirs(root: Path) -> List[Path]:
    runs = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            runs.append(p)
    return sorted(runs)


def find_annotation_csv(run_dir: Path) -> Path:
    p = run_dir / "N_run_1_annotation.csv"
    if p.exists():
        return p
    cands = sorted(run_dir.glob("*annotation.csv"))
    if not cands:
        raise FileNotFoundError(f"Annotation csv not found in: {run_dir}")
    return cands[0]


def safe_class_name(action: str, obj: str) -> str:
    action = (action or "").strip()
    obj = (obj or "").strip()
    s = f"{action}_{obj}" if obj else action
    s = s.strip().replace(" ", "_")
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = re.sub(r"_+", "_", s)
    return s if s else "unknown"


def index_frames(frames_dir: Path) -> Tuple[List[int], List[Path]]:
    """
    Build a sorted index: timestamps(list[int]), files(list[Path]) aligned.
    """
    pairs = []
    for f in frames_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            ts = extract_timestamp_from_filename(f)
            if ts is not None:
                pairs.append((ts, f))
    pairs.sort(key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def select_frames_in_range(ts_list: List[int], fp_list: List[Path], start_us: int, end_us: int) -> List[Path]:
    """
    Select frames in closed interval [start, end].
    """
    if start_us is None or end_us is None:
        return []
    if end_us < start_us:
        start_us, end_us = end_us, start_us
    l = bisect_left(ts_list, start_us)
    r = bisect_right(ts_list, end_us)
    return fp_list[l:r]


def copy_frames(frames: List[Path], out_dir: Path, prefix: str = ""):
    ensure_dir(out_dir)
    for i, src in enumerate(frames, start=1):
        dst = out_dir / src.name
        shutil.copy2(src, dst)


# ----------------- Debug tool -----------------

def debug_ts_mapping(
    run_dir: Path,
    cam_frames_dir: Path,
    ts_list: List[int],
    fp_list: List[Path],
    start_us: int,
    end_us: int,
    side_label: str,
    class_name: str,
    clip_id: int,
    show_n: int = 5
):
    """
    Print helpful info when clip selection returns empty or looks suspicious.
    """
    print("\n====== [TS DEBUG] ======")
    print(f"RUN: {run_dir.name}")
    print(f"SIDE/CAM: {side_label}")
    print(f"CLASS: {class_name}")
    print(f"CLIP: {clip_id}")
    print(f"Frames dir: {cam_frames_dir}")

    if not fp_list:
        print("Frame index is empty (no frames parsed).")
        print("========================\n")
        return

    print(f"CSV start_us: {start_us}")
    print(f"CSV end_us  : {end_us}")

    # Show first and last few indexed frames
    print(f"Indexed frames: {len(fp_list)}")
    for f in fp_list[:show_n]:
        print(f"  FIRST: {f.name} -> {extract_timestamp_from_filename(f)}")
    for f in fp_list[-show_n:]:
        print(f"  LAST : {f.name} -> {extract_timestamp_from_filename(f)}")

    # Show nearest frames around start/end
    i_start = max(0, min(len(ts_list) - 1, bisect_left(ts_list, start_us)))
    i_end = max(0, min(len(ts_list) - 1, bisect_left(ts_list, end_us)))

    def show_near(idx, tag):
        lo = max(0, idx - 2)
        hi = min(len(fp_list), idx + 3)
        print(f"\nNearest frames around {tag} (index ~ {idx}):")
        for j in range(lo, hi):
            print(f"  [{j:6d}] {fp_list[j].name} -> {ts_list[j]}")

    show_near(i_start, "start")
    show_near(i_end, "end")
    print("========================\n")


# ----------------- CSV reading -----------------

def detect_header(sample_text: str) -> bool:
    # light heuristic: must contain both action and object keywords
    low = sample_text.lower()
    return ("action" in low) and ("object" in low) and ("," in low)


def iter_annotations(csv_path: Path):
    """
    Yield dicts: {action, object, start_us, end_us}
    Supports header/no-header.
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = detect_header(sample)

        if has_header:
            reader = csv.DictReader(f)
            colmap = {c.strip().lower(): c for c in (reader.fieldnames or [])}

            def get(row, key):
                kk = key.lower()
                if kk not in colmap:
                    return ""
                return row.get(colmap[kk], "")

            for row in reader:
                action = get(row, "action").strip()
                obj = get(row, "object").strip()

                # Prefer start_ts/end_ts if present
                start_raw = get(row, "start_ts") or get(row, "start")
                end_raw = get(row, "end_ts") or get(row, "end")

                start_us = parse_timestamp_to_us(start_raw)
                end_us = parse_timestamp_to_us(end_raw)

                if not action or start_us is None or end_us is None:
                    continue

                yield {
                    "action": action,
                    "object": obj,
                    "start_us": start_us,
                    "end_us": end_us,
                }
        else:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 7:
                    continue

                # idx, action, object, start, end, start_ts, end_ts, comment
                action = row[1].strip() if len(row) > 1 else ""
                obj = row[2].strip() if len(row) > 2 else ""

                start_raw = row[5] if len(row) > 5 else row[3]
                end_raw = row[6] if len(row) > 6 else row[4]

                start_us = parse_timestamp_to_us(start_raw)
                end_us = parse_timestamp_to_us(end_raw)

                if not action or start_us is None or end_us is None:
                    continue

                yield {
                    "action": action,
                    "object": obj,
                    "start_us": start_us,
                    "end_us": end_us,
                }


# ----------------- Core processing -----------------

def process_one_run(
    run_dir: Path,
    out_root: Path,
    cam1_id: str,
    cam1_sub: str,
    cam2_id: str,
    cam2_sub: str,
    flat_clip: bool,
    debug: bool,
    debug_on_empty_only: bool,
):
    ann_csv = find_annotation_csv(run_dir)

    cam1_frames_dir = run_dir / cam1_id / cam1_sub
    cam2_frames_dir = run_dir / cam2_id / cam2_sub

    if not cam1_frames_dir.exists():
        print(f"[WARN] Missing frames dir: {cam1_frames_dir}")
    if not cam2_frames_dir.exists():
        print(f"[WARN] Missing frames dir: {cam2_frames_dir}")

    cam1_ts, cam1_fp = index_frames(cam1_frames_dir) if cam1_frames_dir.exists() else ([], [])
    cam2_ts, cam2_fp = index_frames(cam2_frames_dir) if cam2_frames_dir.exists() else ([], [])

    clip_counter = 0
    for ann in iter_annotations(ann_csv):
        clip_counter += 1

        action = ann["action"]
        obj = ann["object"]
        start_us = ann["start_us"]
        end_us = ann["end_us"]

        class_name = safe_class_name(action, obj)
        class_dir = out_root / class_name
        ensure_dir(class_dir)

        clip_name = f"{run_dir.name}_clip_{clip_counter:06d}"
        clip_dir = class_dir / clip_name
        ensure_dir(clip_dir)

        cam1_frames = select_frames_in_range(cam1_ts, cam1_fp, start_us, end_us)
        cam2_frames = select_frames_in_range(cam2_ts, cam2_fp, start_us, end_us)

        # Write frames
        if flat_clip:
            copy_frames(cam1_frames, clip_dir, prefix=f"{cam1_id}_")
            copy_frames(cam2_frames, clip_dir, prefix=f"{cam2_id}_")
        else:
            copy_frames(cam1_frames, clip_dir / f"cam_{cam1_id}")
            copy_frames(cam2_frames, clip_dir / f"cam_{cam2_id}")

        # Debug conditions
        cam1_empty = (cam1_frames_dir.exists() and len(cam1_frames) == 0)
        cam2_empty = (cam2_frames_dir.exists() and len(cam2_frames) == 0)

        if debug:
            if (not debug_on_empty_only) or cam1_empty or cam2_empty:
                if cam1_frames_dir.exists():
                    debug_ts_mapping(
                        run_dir, cam1_frames_dir, cam1_ts, cam1_fp,
                        start_us, end_us, f"cam_{cam1_id}", class_name, clip_counter
                    )
                if cam2_frames_dir.exists():
                    debug_ts_mapping(
                        run_dir, cam2_frames_dir, cam2_ts, cam2_fp,
                        start_us, end_us, f"cam_{cam2_id}", class_name, clip_counter
                    )

        # Warn
        if cam1_empty:
            print(f"[WARN] {run_dir.name} {class_name} clip{clip_counter}: cam1 empty")
        if cam2_empty:
            print(f"[WARN] {run_dir.name} {class_name} clip{clip_counter}: cam2 empty")


def main():
    ap = argparse.ArgumentParser(
        "Build dataset: out_root/action_object_class/clip_folder/(cam folders)/frames"
    )
    ap.add_argument("--in-root", type=Path, required=True, help="Directory containing run_* folders")
    ap.add_argument("--out-root", type=Path, required=True, help="Output dataset directory")

    ap.add_argument("--cam1-id", type=str, default="001431512812")
    ap.add_argument("--cam1-sub", type=str, default="frames_rgb_blured")
    ap.add_argument("--cam2-id", type=str, default="001484412812")
    ap.add_argument("--cam2-sub", type=str, default="frames_rgb")

    ap.add_argument("--flat-clip", action="store_true",
                    help="Put both cameras into same clip folder, prefixing filenames with cam id")

    # Debug
    ap.add_argument("--debug-ts", action="store_true",
                    help="Enable timestamp debug prints")
    ap.add_argument("--debug-empty-only", action="store_true",
                    help="Only print debug when selected frames are empty for a camera")

    args = ap.parse_args()

    runs = list_run_dirs(args.in_root)
    if not runs:
        print("No run_* folders found in --in-root")
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(runs)} run folders.")

    for r in runs:
        print(f"[RUN] {r.name}")
        process_one_run(
            run_dir=r,
            out_root=args.out_root,
            cam1_id=args.cam1_id,
            cam1_sub=args.cam1_sub,
            cam2_id=args.cam2_id,
            cam2_sub=args.cam2_sub,
            flat_clip=args.flat_clip,
            debug=args.debug_ts,
            debug_on_empty_only=args.debug_empty_only,
        )

    print("Done.")


if __name__ == "__main__":
    main()
