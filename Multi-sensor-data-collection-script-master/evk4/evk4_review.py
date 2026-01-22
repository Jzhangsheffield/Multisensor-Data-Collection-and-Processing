#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EVK4 .raw 文件离线可视化质检 + 单行日志记录

使用说明：
  python evk4_review_rawreader.py --root "L:\MULTI_SENSOR_DATA_COLLECTION\evk4\sample_prepare_Nov_03_clean"
  控制键：
    [P] = 通过
    [K] = 跳过
    [S] = 保存截图
    [Q] = 退出
    [SPACE] = 暂停/继续
    [ [ / ] ] = 减小/增大 Δt 聚合窗口
    [C] = 切换渲染模式
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
from metavision_core.event_io.raw_reader import RawReader  # ✅ 只保留 RawReader

# ============================ 可视化与渲染 ============================

def make_frame(events, h, w, mode="gray_bipolar"):
    """
    将事件聚合成一帧图像。
    mode == "gray_bipolar": 灰底，p=1 为白，p=0 为黑
    mode == "bwr": 蓝-白-红（p=0 蓝，p=1 红），空白为黑
    """
    if events.size == 0:
        if mode == "bwr":
            return np.zeros((h, w, 3), dtype=np.uint8)
        else:
            return np.full((h, w, 3), 128, dtype=np.uint8)

    x = events['x'].astype(np.int32)
    y = events['y'].astype(np.int32)
    p = events['p'].astype(np.uint8)

    if mode == "bwr":
        # 蓝红双色
        img = np.zeros((h, w, 3), dtype=np.uint8)
        pos = (p == 1)
        neg = (p == 0)
        img[y[pos], x[pos], 2] = 255  # 红
        img[y[neg], x[neg], 0] = 255  # 蓝
        return img
    else:
        img = np.full((h, w, 3), 128, dtype=np.uint8)
        img[y, x] = (p * 255)[:, None]
        return img


# ============================ 日志（单行/文件） ============================

def log_path(base_dir: Path) -> Path:
    d = base_dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "evk4_review_log.txt"


def parse_existing_log(base_dir: Path):
    """读取已有日志"""
    path = log_path(base_dir)
    table = {}
    if not path.exists():
        return table
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if (not s) or s.startswith("#") or s.startswith("time "):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        tm = parts[0] + " " + parts[1]
        file_rel = parts[2]
        run_tag = parts[3]
        status = parts[4] if len(parts) >= 5 else "SKIP"
        abs_key = str((base_dir / file_rel).resolve())
        table[abs_key] = dict(time=tm, file=file_rel, run=run_tag, status=status)
    return table


def write_log(base_dir: Path, table: dict):
    """写日志文件，列：time file_rel run_tag status"""
    path = log_path(base_dir)
    W_TIME = 19
    W_FILE = max(35, max((len(v["file"]) for v in table.values()), default=35))
    W_RUN = 10
    hdr = f"{'time':<{W_TIME}} {'file':<{W_FILE}} {'run':<{W_RUN}} status"
    sep = "-" * len(hdr)
    lines = ["# EVK4 review log (one line per raw)", hdr, sep]
    for _, v in sorted(table.items(), key=lambda kv: kv[1]["file"]):
        lines.append(f"{v['time']:<{W_TIME}} {v['file']:<{W_FILE}} {v['run']:<{W_RUN}} {v['status']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[LOG] 写入：{path}")


def format_run_tag(raw_path: Path) -> str:
    """生成 run 标签，例如 MR_A/run_19"""
    try:
        run_dir = raw_path.parent.name
        mr_dir = raw_path.parent.parent.name
        return f"{mr_dir}/{run_dir}"
    except Exception:
        return "UNKNOWN/UNKNOWN"


# ============================ 播放器（单文件人工审核） ============================

def review_one_raw(raw_path: Path, args) -> str:
    """单个 raw 文件播放与审核"""
    try:
        reader = RawReader(str(raw_path))
    except Exception as e:
        print(f"[ERROR] 无法打开文件 {raw_path} ：{e}")
        return "SKIP"

    h, w = reader.get_size()
    win = f"EVK4 Review: {raw_path.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # 全屏显示
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass

    paused = False
    dt_us = args.delta_t_us
    mode = args.mode
    delay_ms = max(1, int(1000.0 / args.fps))

    print("Controls: [P]=Pass  [K]=Skip  [S]=Save  [Q]=Quit  [SPACE]=Pause/Play  [ [ / ] ]=Δt- / Δt+  [C]=ColorMode")
    print(f"初始 Δt={dt_us} μs, 显示帧率≈{args.fps} FPS, 模式={mode}")

    status = "SKIP"
    last_frame = None

    while not reader.is_done():
        events = reader.load_delta_t(dt_us)
        frame = make_frame(events, h, w, mode=mode)
        last_frame = frame
        cv2.imshow(win, frame)
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

        if key == ord(' '):       # 暂停/继续
            paused = not paused
        elif key == ord('p'):     # 通过
            status = "PASS"
            break
        elif key == ord('k'):     # 跳过
            status = "SKIP"
            break
        elif key == ord('q'):     # 退出所有
            cv2.destroyWindow(win)
            return None
        elif key == ord('s'):     # 保存截图
            out_dir = raw_path.parent / raw_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            snap = out_dir / f"{raw_path.stem}_snapshot.png"
            if last_frame is not None:
                cv2.imwrite(str(snap), last_frame)
                print(f"[Save] {snap}")
        elif key == ord('['):     # 调整聚合窗口
            dt_us = max(1000, int(dt_us * 0.8))
            print(f"Δt -> {dt_us} μs")
        elif key == ord(']'):
            dt_us = min(2_000_000, int(dt_us * 1.25))
            print(f"Δt -> {dt_us} μs")
        elif key == ord('c'):     # 切换模式
            mode = "bwr" if mode == "gray_bipolar" else "gray_bipolar"
            print(f"Mode -> {mode}")

    cv2.destroyWindow(win)
    return status


# ============================ 扫描与主流程 ============================

def find_raw_files(root: Path):
    """在 root 下递归搜索 *.raw 文件"""
    raws = sorted(root.rglob("*.raw"))
    return raws


def main():
    ap = argparse.ArgumentParser("EVK4 .raw 文件离线可视化质检 + 单行日志")
    ap.add_argument("--root", type=Path, default=r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\MR\evk4", help="含 MR/*/run_*/<file.raw> 的根目录")
    ap.add_argument("--fps", type=float, default=30.0, help="播放帧率")
    ap.add_argument("--delta-t-us", dest="delta_t_us", type=int, default=300_000, help="聚合时间窗（微秒）")
    ap.add_argument("--mode", choices=["gray_bipolar", "bwr"], default="bwr", help="渲染模式")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"[ERROR] 根目录不存在：{root}")
        sys.exit(1)

    base_dir = root.parent
    table = parse_existing_log(base_dir)

    raws = find_raw_files(root)
    if not raws:
        print("[INFO] 未找到 .raw 文件。")
        write_log(base_dir, table)
        return

    for raw_path in raws:
        abs_key = str(raw_path.resolve())
        if abs_key in table and table[abs_key]["status"] == "PASS":
            print(f"[SKIP PASS] {raw_path}")
            continue

        file_rel = str(raw_path.relative_to(base_dir))
        run_tag = format_run_tag(raw_path)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n[REVIEW] {run_tag}  {raw_path.name}")
        status = review_one_raw(raw_path, args)
        if status is None:  # 用户退出
            write_log(base_dir, table)
            print("[QUIT] 用户退出。")
            return

        table[abs_key] = dict(time=now_str, file=file_rel, run=run_tag, status=status)
        write_log(base_dir, table)

    print("\n全部完成。日志位于：", log_path(base_dir))


if __name__ == "__main__":
    main()
