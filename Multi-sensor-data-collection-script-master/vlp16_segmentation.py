#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build LiDAR clip dataset by matching annotations (folder A) to radar frame CSVs (folder B).

A/
  run_*/ or run_*_*/
    N_run_1_annotation.csv (or *annotation.csv)
    Each row includes: action, object, start, end (timestamps may be YYYYMMDD_HHMMSS_micro or linux epoch)

B/
  run_*/ or run_*_*/
    20251105_172532/               (session folder, time-named)
      20251105_172532_547260_cloud.csv
      20251105_172532_547300_cloud.csv
      ...

Matching requirement (UPDATED):
For each annotation interval [start, end], find a radar frame interval defined by:
  - t_left  = nearest timestamp <= start  (i.e., the closest frame time smaller than or equal to start)
  - t_right = nearest timestamp >= end    (i.e., the closest frame time larger than or equal to end)
Then select ALL radar frame CSVs with timestamps in [t_left, t_right] and copy them to the clip folder.

Output dataset:
out_root/
  action_object_class/
    run_xxx_clip_000001/
        20251105_172532_547260_cloud.csv
        20251105_172532_547300_cloud.csv
        ...
(Each copied file is cleaned by removing rows where x,y,z,intensity are all zero.)

Cleaning:
- For each radar CSV copied, remove rows where (x,y,z,intensity) are all zero (handles header name variants).
- Writes cleaned file to destination (does NOT concatenate files).

Timestamp parsing supports:
- 'YYYYMMDD_HHMMSS_micro' (e.g., 20251105_172554_530685)
- Linux epoch timestamps in seconds / ms / us / ns (including floats)
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import csv
import argparse
from pathlib import Path
from bisect import bisect_left, bisect_right
from typing import Optional, List, Tuple, Dict


# ----------------- Timestamp parsing -----------------

def normalize_linux_timestamp_to_us(t) -> Optional[int]:
    """Normalize Linux epoch timestamps to microseconds int."""
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


def parse_timestamp_to_key(x) -> Optional[int]:
    """
    Convert timestamp to a sortable key (int).
    - If matches 'YYYYMMDD_HHMMSS_micro' => int(YYYYMMDDHHMMSSmicro) (not epoch but sortable)
    - Else parse numeric token as Linux epoch and normalize to microseconds
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return normalize_linux_timestamp_to_us(float(x)) if isinstance(x, float) else normalize_linux_timestamp_to_us(x)

    s = str(x).strip()
    if not s:
        return None

    m = re.search(r"(\d{8}_\d{6}_\d{6})", s)
    if m:
        return int(m.group(1).replace("_", ""))

    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    if m2:
        num_s = m2.group(1)
        try:
            if "." in num_s:
                return normalize_linux_timestamp_to_us(float(num_s))
            return normalize_linux_timestamp_to_us(int(num_s))
        except Exception:
            return None
    return None


# ----------------- Folder helpers -----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_run_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def safe_class_name(action: str, obj: str) -> str:
    action = (action or "").strip()
    obj = (obj or "").strip()
    s = f"{action}_{obj}" if obj else action
    s = s.strip().replace(" ", "_")
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = re.sub(r"_+", "_", s)
    return s if s else "unknown"


# ----------------- STRICT annotation.csv selection -----------------
# 支持：N_run_24_annotation.csv  或  N_run_24-37_annotation.csv
STRICT_ANN_RE = re.compile(r"^MR_run_(\d+)(?:-(\d+))?_annotation\.csv$")

def find_strict_annotation_csv(run_dir: Path) -> Path:
    """
    Strictly select ONLY files like:
      - N_run_24_annotation.csv
      - N_run_24-37_annotation.csv

    Reject any variants, e.g.:
      - N_run_23_annotation.copy.csv
      - N_run_23_annotation.csv.bak
      - N_run_23_annotation (no .csv)
    Strategy:
      - If multiple strict matches exist, pick the one with the "narrowest range" first,
        then by smallest start index. (You can change this policy if you want.)
    """
    matches = []
    for p in run_dir.iterdir():
        if not (p.is_file() and p.suffix.lower() == ".csv"):
            continue
        m = STRICT_ANN_RE.match(p.name)
        if not m:
            continue

        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            a, b = b, a  # 容错：24-37 或 37-24 都支持

        span = b - a
        matches.append((span, a, b, p))

    if not matches:
        raise FileNotFoundError(
            f"No STRICT annotation file found in: {run_dir} "
            f"(expect N_run_<id>_annotation.csv or N_run_<a>-<b>_annotation.csv)"
        )

    # 选择策略：优先范围最小（更具体），再选起始最小
    matches.sort(key=lambda x: (x[0], x[1], x[2]))
    if len(matches) > 1:
        print(f"[WARN] Multiple strict annotation CSVs in {run_dir.name}: "
              f"{[p.name for _, _, _, p in matches]}. Using: {matches[0][3].name}")

    return matches[0][3]

# ----------------- Read annotations -----------------

def detect_header(sample_text: str) -> bool:
    low = sample_text.lower()
    return ("action" in low) and ("object" in low) and ("," in low)


def iter_annotations(csv_path: Path):
    """
    Yield dicts: {action, object, start_key, end_key}
    CSV may be header or no-header.
    If header exists, prefer columns start_ts/end_ts if present.
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
                start_raw = get(row, "start_ts") or get(row, "start")
                end_raw = get(row, "end_ts") or get(row, "end")
                start_key = parse_timestamp_to_key(start_raw)
                end_key = parse_timestamp_to_key(end_raw)
                if not action or start_key is None or end_key is None:
                    continue
                yield {"action": action, "object": obj, "start_key": start_key, "end_key": end_key}

        else:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 5:
                    continue
                # idx, action, object, start, end, (optional start_ts, end_ts, comment)
                action = row[1].strip() if len(row) > 1 else ""
                obj = row[2].strip() if len(row) > 2 else ""
                start_raw = row[5] if len(row) > 5 else row[3]
                end_raw = row[6] if len(row) > 6 else row[4]
                start_key = parse_timestamp_to_key(start_raw)
                end_key = parse_timestamp_to_key(end_raw)
                if not action or start_key is None or end_key is None:
                    continue
                yield {"action": action, "object": obj, "start_key": start_key, "end_key": end_key}


# ----------------- Radar frames indexing -----------------

def find_session_folder(run_dir_b: Path) -> Optional[Path]:
    """
    In B/run_xxx/ there is a time-named folder, e.g. 20251105_172532
    Pick the first directory that matches YYYYMMDD_HHMMSS, else first sorted.
    """
    cands = [p for p in run_dir_b.iterdir() if p.is_dir()]
    if not cands:
        return None
    cands.sort(key=lambda p: (0 if re.match(r"^\d{8}_\d{6}$", p.name) else 1, p.name))
    return cands[0]


def extract_radar_ts_key_from_filename(fp: Path) -> Optional[int]:
    # e.g. 20251105_172532_547260_cloud.csv
    return parse_timestamp_to_key(fp.stem)


def index_radar_frames(session_dir: Path) -> Tuple[List[int], List[Path]]:
    pairs = []
    for f in session_dir.glob("*.csv"):
        if not f.is_file():
            continue
        ts = extract_radar_ts_key_from_filename(f)
        if ts is not None:
            pairs.append((ts, f))
    pairs.sort(key=lambda x: x[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def find_bounds_interval(ts_list: List[int], start_key: int, end_key: int) -> Optional[Tuple[int, int]]:
    """
    t_left  = nearest <= start
    t_right = nearest >= end
    Clamp to endpoints if out of range.
    """
    if not ts_list:
        return None
    if end_key < start_key:
        start_key, end_key = end_key, start_key

    li = bisect_right(ts_list, start_key) - 1
    if li < 0:
        li = 0
    t_left = ts_list[li]

    ri = bisect_left(ts_list, end_key)
    if ri >= len(ts_list):
        ri = len(ts_list) - 1
    t_right = ts_list[ri]

    if t_right < t_left:
        t_left, t_right = t_right, t_left
    return t_left, t_right


def select_radar_files_by_bounds(ts_list: List[int], fp_list: List[Path], t_left: int, t_right: int) -> List[Path]:
    l = bisect_left(ts_list, t_left)
    r = bisect_right(ts_list, t_right)
    return fp_list[l:r]


# ----------------- LiDAR CSV cleaning (copy-per-file) -----------------

def _guess_column_indices(header: List[str]) -> Dict[str, Optional[int]]:
    h = [c.strip().lower() for c in header]

    def find_one(keys):
        for k in keys:
            if k in h:
                return h.index(k)
        return None

    return {
        "x": find_one(["x", "px", "posx"]),
        "y": find_one(["y", "py", "posy"]),
        "z": find_one(["z", "pz", "posz"]),
        "intensity": find_one(["intensity", "i", "reflectivity", "r", "int"]),
    }


def clean_lidar_csv(src_csv: Path, dst_csv: Path) -> Tuple[int, int]:
    """
    Copy src_csv -> dst_csv while removing rows where (x,y,z,intensity) all zero.
    Returns: (kept_rows, removed_rows)
    """
    kept = 0
    removed = 0

    with src_csv.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            dst_csv.write_text("", encoding="utf-8")
            return (0, 0)

        col_idx = _guess_column_indices(header)
        fallback_all_numeric = any(v is None for v in col_idx.values())

        with dst_csv.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(header)

            for row in reader:
                if not row:
                    continue

                def get_float(i):
                    try:
                        return float(row[i])
                    except Exception:
                        return 0.0

                if fallback_all_numeric:
                    nums = []
                    for v in row:
                        try:
                            nums.append(float(v))
                        except Exception:
                            nums.append(0.0)
                    if all(abs(n) < 1e-12 for n in nums):
                        removed += 1
                        continue
                else:
                    x = get_float(col_idx["x"])
                    y = get_float(col_idx["y"])
                    z = get_float(col_idx["z"])
                    inten = get_float(col_idx["intensity"])
                    if abs(x) < 1e-12 and abs(y) < 1e-12 and abs(z) < 1e-12 and abs(inten) < 1e-12:
                        removed += 1
                        continue

                writer.writerow(row)
                kept += 1

    return (kept, removed)


# ----------------- Processing per run -----------------

def process_one_run(a_run: Path, b_run: Path, out_root: Path, debug: bool = False):
    ann_csv = find_strict_annotation_csv(a_run)

    session_dir = find_session_folder(b_run)
    if session_dir is None:
        print(f"[WARN] No session folder in: {b_run}")
        return

    ts_list, fp_list = index_radar_frames(session_dir)
    if not fp_list:
        print(f"[WARN] No radar CSV frames in: {session_dir}")
        return

    clip_counter = 0
    for ann in iter_annotations(ann_csv):
        clip_counter += 1
        action = ann["action"]
        obj = ann["object"]
        start_key = ann["start_key"]
        end_key = ann["end_key"]

        class_name = safe_class_name(action, obj)
        class_dir = out_root / class_name
        ensure_dir(class_dir)

        clip_name = f"{a_run.name}_clip_{clip_counter:06d}"
        clip_dir = class_dir / clip_name
        ensure_dir(clip_dir)

        bounds = find_bounds_interval(ts_list, start_key, end_key)
        if bounds is None:
            print(f"[WARN] {a_run.name} {class_name} clip{clip_counter}: no radar index")
            continue
        t_left, t_right = bounds

        selected = select_radar_files_by_bounds(ts_list, fp_list, t_left, t_right)
        if not selected:
            print(f"[WARN] {a_run.name} {class_name} clip{clip_counter}: no radar frames in bounds")
            continue

        lidar_dir = clip_dir
        ensure_dir(lidar_dir)

        total_kept = 0
        total_removed = 0

        for src in selected:
            dst = lidar_dir / src.name
            kept, removed = clean_lidar_csv(src, dst)
            total_kept += kept
            total_removed += removed

        if debug:
            print("\n[DEBUG]")
            print(f"RUN: {a_run.name}")
            print(f"ANN: {ann_csv.name}")
            print(f"CLASS: {class_name}")
            print(f"CLIP: {clip_counter}")
            print(f"Annotation: start={start_key}, end={end_key}")
            print(f"Bounds: left={t_left}, right={t_right}")
            print(f"Selected files: {len(selected)}")
            if selected:
                print(f"  first: {selected[0].name} -> {extract_radar_ts_key_from_filename(selected[0])}")
                print(f"  last : {selected[-1].name} -> {extract_radar_ts_key_from_filename(selected[-1])}")
            print(f"Rows kept={total_kept}, removed={total_removed}")
            print(f"Output dir: {lidar_dir}")
            print("[/DEBUG]\n")


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser("Match annotations (A) to radar frames (B) and build LiDAR dataset (copy CSVs).")
    ap.add_argument("--a-root", type=Path, required=True, help="Folder A containing run_* with strict annotation CSV")
    ap.add_argument("--b-root", type=Path, required=True, help="Folder B containing run_* with radar frame CSVs")
    ap.add_argument("--out-root", type=Path, required=True, help="Output dataset folder")
    ap.add_argument("--debug", action="store_true", help="Print debug info per clip")

    args = ap.parse_args()

    a_runs = {p.name: p for p in list_run_dirs(args.a_root)}
    b_runs = {p.name: p for p in list_run_dirs(args.b_root)}

    common = sorted(set(a_runs.keys()) & set(b_runs.keys()))
    if not common:
        print("No matching run_* folders between A and B.")
        return

    ensure_dir(args.out_root)
    print(f"Found {len(common)} matching runs.")

    for rn in common:
        print(f"[RUN] {rn}")
        process_one_run(a_runs[rn], b_runs[rn], args.out_root, debug=args.debug)

    print("Done.")


if __name__ == "__main__":
    main()

