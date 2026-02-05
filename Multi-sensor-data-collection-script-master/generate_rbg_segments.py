#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate per-subject CSV annotation files from RGB frame folders.

Root:
  D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\RGB
    N/M/MR/J
      <action>/
        <segment>/
          cam_001431512812/   (or 001431512812)
            20251113_170325_880571.jpg ...

Output:
  <root>\N_rgb_segments_cam_001431512812.csv
  <root>\M_rgb_segments_cam_001431512812.csv
  <root>\MR_rgb_segments_cam_001431512812.csv
  <root>\J_rgb_segments_cam_001431512812.csv

CSV columns:
  subject, action, segment, start_frame, end_frame
"""

import re
import csv
import argparse
from pathlib import Path
from typing import Optional, Tuple, List


TS_RE = re.compile(r"^(\d{8}_\d{6}_\d{6})\.(jpg|jpeg|png)$", re.IGNORECASE)


def extract_ts_stem(filename: str) -> Optional[str]:
    """
    Return timestamp stem like '20251113_170325_880571' if matches; else None.
    """
    m = TS_RE.match(filename)
    if not m:
        return None
    return m.group(1)


def find_camera_dir(segment_dir: Path, cam_name: str, allow_no_prefix: bool = True) -> Optional[Path]:
    """
    Find camera folder inside a segment dir.
    Priority:
      1) exact match cam_name
      2) if allow_no_prefix: also try removing/adding 'cam_' prefix
    """
    cand = segment_dir / cam_name
    if cand.is_dir():
        return cand

    if allow_no_prefix:
        if cam_name.startswith("cam_"):
            alt = segment_dir / cam_name.replace("cam_", "", 1)
            if alt.is_dir():
                return alt
        else:
            alt = segment_dir / ("cam_" + cam_name)
            if alt.is_dir():
                return alt

    return None


def get_start_end_from_cam_dir(cam_dir: Path) -> Optional[Tuple[str, str]]:
    """
    Scan image files in cam_dir, parse timestamps from filenames, return (min_ts, max_ts).
    """
    ts_list: List[str] = []
    # 只扫当前文件夹，不递归（通常帧都在这一层）
    for p in cam_dir.iterdir():
        if not p.is_file():
            continue
        ts = extract_ts_stem(p.name)
        if ts is not None:
            ts_list.append(ts)

    if not ts_list:
        return None

    ts_list.sort()
    return ts_list[0], ts_list[-1]


def generate_csv_for_subject(subject_dir: Path, cam_name: str, out_csv: Path) -> None:
    """
    subject_dir: e.g. ...\RGB\N
    """
    rows = []
    # 动作文件夹
    for action_dir in sorted(subject_dir.iterdir()):
        if not action_dir.is_dir():
            continue

        action_name = action_dir.name

        # 片段文件夹
        for segment_dir in sorted(action_dir.iterdir()):
            if not segment_dir.is_dir():
                continue

            segment_name = segment_dir.name

            cam_dir = find_camera_dir(segment_dir, cam_name, allow_no_prefix=True)
            if cam_dir is None:
                # 没有这个相机文件夹就跳过
                continue

            se = get_start_end_from_cam_dir(cam_dir)
            if se is None:
                # 相机文件夹存在但没有可解析的帧
                continue

            start_ts, end_ts = se
            rows.append([subject_dir.name, action_name, segment_name, start_ts, end_ts])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["subject", "action", "segment", "start_frame", "end_frame"])
        w.writerows(rows)

    print(f"✅ Wrote {len(rows)} rows -> {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\RGB",
        help="RGB root containing N/M/MR/J",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="N,M,MR,J",
        help="Comma-separated subject folders to process",
    )
    parser.add_argument(
        "--cam",
        type=str,
        default="cam_001431512812",
        help="Camera folder name to use (e.g. cam_001431512812 or 001431512812)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory for CSVs (default: same as --root)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    for s in subjects:
        subject_dir = root / s
        if not subject_dir.is_dir():
            print(f"⚠️ Skip missing subject dir: {subject_dir}")
            continue

        out_csv = out_dir / f"{s}_rgb_segments_{args.cam}.csv"
        # 文件名不能含 \ / : * ? " < > |，这里 cam 名字可能含非法字符？一般不会，但保险起见替换一下
        safe_name = args.cam.replace(":", "_").replace("/", "_").replace("\\", "_")
        out_csv = out_dir / f"{s}_rgb_segments_{safe_name}.csv"

        generate_csv_for_subject(subject_dir, args.cam, out_csv)


if __name__ == "__main__":
    main()
