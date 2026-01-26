#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import csv
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple
from zoneinfo import ZoneInfo
from datetime import datetime

import numpy as np
import pandas as pd


"""
A_root/
  run_* 或 run_*_*  (共 36 个)
    - 仅使用以下两类标注文件（忽略任何带 "copy" 的文件）：
        1) N_run_1_annotation.csv
        2) N_run_1_28_annotation.csv
    - 标注 CSV 每行至少包含 4 列：
        action, object, start, end
      其中 start/end 时间格式固定为：
        YYYYMMDD_HHMMSS_micro
      例：20251023_184757_399654

B_root/
  （同样 36 个 run 文件夹，但名字可能很长，例如：sub-N_..._run-001_emg）
    - 每个 run 文件夹下有两个子文件夹：
        left/  right/
      其中各有 1 个 EMG+IMU CSV 文件（文件名任意）
    - EMG+IMU CSV 内有一列：
        board_ts
      它是 Linux epoch 时间戳（秒，float），例如：
        1762363522.9356194

本脚本做的事情：

1) 遍历 A_root 下每个 run 文件夹
2) 读取该 run 的标注 CSV（严格匹配文件名模式 + 忽略 copy）
3) 对标注 CSV 的每一行（动作片段）：
     - action + object 组成 “动作类别文件夹名”
     - 将 start/end（本地时间字符串）按指定时区（默认 Europe/London）转换为 epoch 秒
     - 在 B_root 对应 run 的 left/right CSV 中：
         起始点 = 小于等于 start 且最接近 start 的 board_ts 行（最后一个 <= start）
         结束点 = 大于等于 end   且最接近 end   的 board_ts 行（第一个 >= end）
       然后裁剪 [起始点, 结束点] 之间的所有行（闭区间）
4) 输出数据集结构：
   out_root/
     action_object_class/
       <A_run_name>_clip_000001/
         left/clip.csv
         right/clip.csv

可选参数：
- --tz     : 控制 start/end 解析时区（若你发现切不到数据，尝试 --tz UTC）
- --side   : 只导出 left / right 或 both
- --debug  : 输出每个 clip 的切片索引、起止 board_ts 等调试信息

================================================================================
"""

# ------------------------
# 1) 工具：安全命名
# ------------------------
def safe_name(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = re.sub(r"_+", "_", s)
    return s if s else "unknown"


def action_class_name(action: str, obj: str) -> str:
    action = (action or "").strip()
    obj = (obj or "").strip()
    return safe_name(f"{action}_{obj}" if obj else action)


# ------------------------
# 2) 工具：遍历 run 文件夹
# ------------------------
def list_run_dirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def extract_run_number(name: str) -> Optional[int]:
    """
    从文件夹名中提取 run 编号：
      - run_001
      - run-001
      - ..._run-001_emg
      - ..._run_001_...
    """
    m = re.search(r"(?:^|[^a-zA-Z0-9])run[-_]?(\d+)(?:[^0-9]|$)", name)
    if not m:
        return None
    return int(m.group(1))


def build_b_run_map(b_root: Path) -> Dict[int, Path]:
    """
    B 里每个 run 文件夹名可能不以 run_ 开头，所以我们遍历所有子目录，
    只要能提取 run 编号就纳入映射。
    """
    m = {}
    for p in b_root.iterdir():
        if not p.is_dir():
            continue
        rn = extract_run_number(p.name)
        if rn is None:
            continue
        # 若重复，保留字典里第一个；你也可改成按名字最短/最早排序策略
        m.setdefault(rn, p)
    return m


# ------------------------
# 3) 严格选择 annotation 文件（忽略 copy）
# ------------------------
ANN_RE = re.compile(r"^N_run_(\d+)(?:-(\d+))?_annotationlight\.csv$", re.IGNORECASE)

def find_annotation_csv_strict(run_dir: Path) -> Path:
    """
    只接受：
      - N_run_1_annotation.csv
      - N_run_1_28_annotation.csv
    并忽略任何文件名包含 'copy'（大小写不敏感）
    """
    candidates = []
    for p in run_dir.iterdir():
        if not (p.is_file() and p.suffix.lower() == ".csv"):
            continue
        name_low = p.name.lower()
        if "copy" in name_low:
            continue
        if ANN_RE.match(p.name):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"[A] {run_dir}: 未找到严格匹配的 annotation 文件 "
            f"(N_run_<id>_annotation.csv 或 N_run_<a>_<b>_annotation.csv，且不含 copy)"
        )

    candidates.sort(key=lambda x: x.name.lower())
    if len(candidates) > 1:
        print(f"[WARN] {run_dir.name} 下存在多个合规 annotation："
              f"{[c.name for c in candidates]}，将使用：{candidates[0].name}")
    return candidates[0]


# ------------------------
# 4) 解析 start/end 时间戳 -> epoch seconds
# ------------------------
TS_STR_RE = re.compile(r"^(\d{8})_(\d{6})_(\d{6})$")

def ts_string_to_epoch_seconds(ts: str, tz_name: str) -> float:
    """
    输入：20251023_184757_399654
    解释为 tz_name 时区的本地时间 -> epoch seconds (float)
    """
    ts = str(ts).strip()
    m = TS_STR_RE.match(ts)
    if not m:
        raise ValueError(f"不支持的 start/end 时间格式：{ts}")

    ymd, hms, micro = m.group(1), m.group(2), m.group(3)
    dt = datetime(
        year=int(ymd[0:4]), month=int(ymd[4:6]), day=int(ymd[6:8]),
        hour=int(hms[0:2]), minute=int(hms[2:4]), second=int(hms[4:6]),
        microsecond=int(micro),
        tzinfo=ZoneInfo(tz_name)
    )
    return dt.timestamp()


# ------------------------
# 5) 在 board_ts 中找边界
# ------------------------
def find_slice_indices(board_ts: np.ndarray, start_sec: float, end_sec: float) -> Optional[Tuple[int, int]]:
    """
    board_ts: 已排序的一维 float 数组
    返回 (i0, i1) 为包含端点的闭区间索引：df.iloc[i0:i1+1]
    规则：
      i0 = 最大的 idx 使 board_ts[idx] <= start_sec
      i1 = 最小的 idx 使 board_ts[idx] >= end_sec
    如果 start/end 超出范围则进行 clamp；若仍无法形成有效区间则返回 None
    """
    if board_ts.size == 0:
        return None

    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec

    # i0: last <= start
    i0 = np.searchsorted(board_ts, start_sec, side="right") - 1
    if i0 < 0:
        i0 = 0

    # i1: first >= end
    i1 = np.searchsorted(board_ts, end_sec, side="left")
    if i1 >= board_ts.size:
        i1 = board_ts.size - 1

    if i1 < i0:
        return None
    return i0, i1


# ------------------------
# 6) 找到 B/run 里的 left/right CSV
# ------------------------
def find_left_right_csv(b_run_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    left_dir = b_run_dir / "left"
    right_dir = b_run_dir / "right"

    def find_one(side_dir: Path, side: str) -> Optional[Path]:
        if not side_dir.exists():
            return None
        # 优先找含 side 的 emg_imu 文件
        cands = sorted([p for p in side_dir.glob("*.csv") if p.is_file()])
        if not cands:
            return None
        # 优先：文件名里包含 _left_ 或 _right_
        prefer = [p for p in cands if re.search(fr"_{side}_", p.name, re.IGNORECASE)]
        return prefer[0] if prefer else cands[0]

    return find_one(left_dir, "left"), find_one(right_dir, "right")


# ------------------------
# 7) 主处理：对每个 annotation 行切片并输出
# ------------------------
def iter_annotations_rows(ann_csv: Path):
    """
    支持：
    - 有表头：action, object, start, end
    - 无表头：尝试按列名失败后按固定列读取（不建议无表头，但做容错）
    """
    with ann_csv.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        has_header = ("action" in sample.lower()) and ("object" in sample.lower())

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
                start = get(row, "start").strip()
                end = get(row, "end").strip()
                light = get(row, "light").strip()
                if not action or not start or not end:
                    continue
                yield action, obj, start, end, light
        else:
            # fallback: idx, action, object, start, end, ...
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 5:
                    continue
                action = row[1].strip() if len(row) > 1 else ""
                obj = row[2].strip() if len(row) > 2 else ""
                start = row[3].strip() if len(row) > 3 else ""
                end = row[4].strip() if len(row) > 4 else ""
                light = row[9].strip() if len(row) > 9 else ""

                if not action or not start or not end:
                    continue
                yield action, obj, start, end


def slice_and_save_one_side(
    src_csv: Path,
    dst_csv: Path,
    start_sec: float,
    end_sec: float,
    debug: bool = False
) -> bool:
    """
    读取 src_csv（必须含 board_ts），按边界规则切片并保存到 dst_csv。
    返回 True 表示成功写出（哪怕只有几行），False 表示失败。
    """
    df = pd.read_csv(src_csv)
    if "board_ts" not in df.columns:
        raise KeyError(f"{src_csv} 不包含 board_ts 列。")

    board = df["board_ts"].to_numpy(dtype=np.float64)
    # 若 board_ts 未排序，排序会打乱其它列对应关系，这里默认它已按时间顺序
    # 如不确定，可改成按 board_ts 排序后再切片：df = df.sort_values("board_ts")

    idx = find_slice_indices(board, start_sec, end_sec)
    if idx is None:
        if debug:
            print(f"[DEBUG] {src_csv.name}: 找不到有效区间，start={start_sec}, end={end_sec}")
        return False

    i0, i1 = idx
    clip_df = df.iloc[i0:i1+1].copy()
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    clip_df.to_csv(dst_csv, index=False)

    if debug:
        print(f"[DEBUG] {src_csv.name}: slice [{i0}, {i1}] "
              f"({clip_df.shape[0]} rows) "
              f"ts [{board[i0]}, {board[i1]}] "
              f"target [{start_sec}, {end_sec}] -> {dst_csv}")
    return True


def main():
    ap = argparse.ArgumentParser(
        "Build EMG/IMU dataset by slicing left/right CSVs according to annotations."
    )
    ap.add_argument("--a-root", type=Path, required=True, help="Folder A: run_* with annotation CSV")
    ap.add_argument("--b-root", type=Path, required=True, help="Folder B: folders containing run-xxx and left/right CSVs")
    ap.add_argument("--out-root", type=Path, required=True, help="Output dataset root")
    ap.add_argument("--tz", type=str, default="Europe/London",
                    help="Timezone for parsing start/end strings (default: Europe/London)")
    ap.add_argument("--side", choices=["both", "left", "right"], default="both",
                    help="Which sides to export")
    ap.add_argument("--debug", action="store_true", help="Print debug logs per clip")
    args = ap.parse_args()

    a_runs = list_run_dirs(args.a_root)
    if not a_runs:
        print("[A] 未找到 run_* 子文件夹。")
        return

    b_map = build_b_run_map(args.b_root)
    if not b_map:
        print("[B] 未找到包含 run 编号的子文件夹（如 run-001）。")
        return

    args.out_root.mkdir(parents=True, exist_ok=True)

    total_clips = 0
    total_written = 0

    for a_run in a_runs:
        run_no = extract_run_number(a_run.name)
        if run_no is None:
            print(f"[WARN] A run 文件夹名无法提取 run 编号：{a_run.name}，跳过")
            continue

        if run_no not in b_map:
            print(f"[WARN] B 中找不到对应 run 编号 {run_no}（A: {a_run.name}），跳过")
            continue

        b_run = b_map[run_no]

        try:
            ann_csv = find_annotation_csv_strict(a_run)
        except Exception as e:
            print(f"[WARN] {a_run.name}: {e}")
            continue

        left_csv, right_csv = find_left_right_csv(b_run)
        if args.side in ("both", "left") and left_csv is None:
            print(f"[WARN] {b_run.name}: 未找到 left CSV")
        if args.side in ("both", "right") and right_csv is None:
            print(f"[WARN] {b_run.name}: 未找到 right CSV")

        clip_idx = 0
        for action, obj, start_s, end_s, light in iter_annotations_rows(ann_csv):
            clip_idx += 1
            total_clips += 1

            # start/end string -> epoch seconds
            try:
                start_sec = ts_string_to_epoch_seconds(start_s, args.tz)
                end_sec = ts_string_to_epoch_seconds(end_s, args.tz)
            except Exception as e:
                print(f"[WARN] {a_run.name} clip{clip_idx}: 时间戳解析失败: {start_s}~{end_s} ({e})")
                continue

            cls = action_class_name(action, obj)
            clip_folder = f"{a_run.name}_clip_{clip_idx:06d}_{light}"
            base_out = args.out_root / cls / clip_folder

            wrote_any = False

            if args.side in ("both", "left") and left_csv is not None:
                dst = base_out / "left" / "clip.csv"
                ok = slice_and_save_one_side(left_csv, dst, start_sec, end_sec, debug=args.debug)
                wrote_any = wrote_any or ok

            if args.side in ("both", "right") and right_csv is not None:
                dst = base_out / "right" / "clip.csv"
                ok = slice_and_save_one_side(right_csv, dst, start_sec, end_sec, debug=args.debug)
                wrote_any = wrote_any or ok

            if wrote_any:
                total_written += 1
            else:
                if args.debug:
                    print(f"[DEBUG] {a_run.name} clip{clip_idx}: left/right 都未写出（可能区间找不到）")

        print(f"[RUN] {a_run.name} (run={run_no}) -> processed {clip_idx} clips")

    print(f"Done. total_clips={total_clips}, clips_written={total_written}")


if __name__ == "__main__":
    main()
