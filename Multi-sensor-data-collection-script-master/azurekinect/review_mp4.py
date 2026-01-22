#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MP4 视频离线可视化质检 + 单行日志记录 (三相机版本)

新增功能：
  ✔ 日志统一写入固定路径：
        D:/Junxi_data/MULTISENSOR_DATA_COLLECTION/_raw_data_structured/N/logs
  ✔ 播放窗口自动变大 (1280×720) 且居中显示

控制键：
  [P] = PASS
  [K] = SKIP
  [S] = 保存截图
  [Q] = 退出程序
  [SPACE] = 暂停/继续
"""

import os
import sys
import cv2
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================
# 摄像头目录映射
# ============================================================

CAM_MAP = {
    "001484412812": "front",
    "001431512812": "back",
    "001528512812": "side"
}


# ============================================================
# 固定日志路径
# ============================================================

FIXED_LOG_DIR = Path(r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\logs")
FIXED_LOG_DIR.mkdir(parents=True, exist_ok=True)
FIXED_LOG_FILE = FIXED_LOG_DIR / "mp4_review_log.txt"


# ============================================================
# 日志函数
# ============================================================

def parse_existing_log():
    """读取已有日志"""
    table = {}

    if not FIXED_LOG_FILE.exists():
        return table

    for line in FIXED_LOG_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("time "):
            continue

        parts = s.split()
        if len(parts) < 4:
            continue

        tm = parts[0] + " " + parts[1]
        file_rel = parts[2]
        run_tag = parts[3]
        status = parts[4] if len(parts) >= 5 else "SKIP"

        abs_key = str(Path(file_rel).resolve())
        table[abs_key] = dict(time=tm, file=file_rel, run=run_tag, status=status)

    return table


def write_log(table: dict):
    """写入日志到固定目录"""
    W_TIME = 19
    W_FILE = max(35, max((len(v["file"]) for v in table.values()), default=35))
    W_RUN = 15

    hdr = f"{'time':<{W_TIME}} {'file':<{W_FILE}} {'run':<{W_RUN}} status"
    sep = "-" * len(hdr)

    lines = ["# MP4 review log (three-camera system)", hdr, sep]

    for _, v in sorted(table.items(), key=lambda kv: kv[1]["file"]):
        lines.append(
            f"{v['time']:<{W_TIME}} {v['file']:<{W_FILE}} {v['run']:<{W_RUN}} {v['status']}"
        )

    FIXED_LOG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[LOG] 写入：{FIXED_LOG_FILE}")


# ============================================================
# run tag
# ============================================================

def format_run_tag(mp4_path: Path) -> str:
    try:
        cam_dir = mp4_path.parent.name
        cam_pos = CAM_MAP.get(cam_dir, "unknown")

        run_dir = mp4_path.parent.parent.name
        mr_dir = mp4_path.parent.parent.parent.name

        return f"{mr_dir}/{run_dir}/{cam_pos}"
    except:
        return "UNKNOWN/UNKNOWN/UNKNOWN"


# ============================================================
# 播放器函数
# ============================================================

def review_one_mp4(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开文件：{path}")
        return "SKIP"

    win = f"MP4 Review: {path.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # ----- Center + resize window -----
    width, height = 1280, 720
    cv2.resizeWindow(win, width, height)
    screen_w = 1920
    screen_h = 1080
    cv2.moveWindow(win, (screen_w - width) // 2, (screen_h - height) // 2)

    paused = False
    status = "SKIP"

    print("\nControls: [P]=Pass  [K]=Skip  [S]=Save  [Q]=Quit  [SPACE]=Pause")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(win, frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):
            paused = not paused
        elif key == ord('p'):
            status = "PASS"
            break
        elif key == ord('k'):
            status = "SKIP"
            break
        elif key == ord('s'):
            snap = path.parent / (path.stem + "_snapshot.png")
            cv2.imwrite(str(snap), frame)
            print(f"[Save] {snap}")
        elif key == ord('q'):
            cv2.destroyWindow(win)
            return None

    cap.release()
    cv2.destroyWindow(win)
    return status


# ============================================================
# 扫描 MP4 文件
# ============================================================

def find_mp4_files(root: Path):
    """只扫描 run_xxx/001xxxxx/*.mp4"""
    files = []
    for run_dir in sorted(root.glob("run_*")):
        for cam_id in CAM_MAP.keys():
            cam_path = run_dir / cam_id
            if cam_path.exists():
                files.extend(sorted(cam_path.glob("*.mp4")))
    return files


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser("MP4 视频质检（三相机）")
    ap.add_argument("--root", type=Path, default=r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\kinect",
                    help="输入根目录，例如: .../MR/kinect/")

    args = ap.parse_args()
    root: Path = args.root

    if not root.exists():
        print(f"[ERROR] 根目录不存在：{root}")
        sys.exit(1)

    table = parse_existing_log()
    mp4_list = find_mp4_files(root)

    if not mp4_list:
        print("[INFO] 未找到 mp4 文件")
        write_log(table)
        return

    for mp4 in mp4_list:
        abs_key = str(mp4.resolve())

        # Skip already reviewed PASS entries
        if abs_key in table and table[abs_key]["status"] == "PASS":
            print(f"[SKIP PASS] {mp4}")
            continue

        file_rel = str(mp4)
        run_tag = format_run_tag(mp4)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n[REVIEW] {run_tag}  {mp4.name}")
        status = review_one_mp4(mp4)

        if status is None:
            write_log(table)
            print("[QUIT] 用户主动退出")
            return

        table[abs_key] = dict(time=now_str, file=file_rel, run=run_tag, status=status)
        write_log(table)

    print("\n全部完成！日志位于：", FIXED_LOG_FILE)


if __name__ == "__main__":
    main()
