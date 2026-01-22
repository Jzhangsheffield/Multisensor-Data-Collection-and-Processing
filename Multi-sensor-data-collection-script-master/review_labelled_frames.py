#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kinect RGB 视频帧离线可视化质检 + 日志记录（CSV）
P/K 后回车输入 comment，U 撤销上一条
"""

import sys
import cv2
import argparse
import csv
from pathlib import Path
from datetime import datetime

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

WIN_NAME = "Kinect RGB Review"
WIN_W, WIN_H = 1600, 900
WIN_X, WIN_Y = 100, 50


# ============================ CSV 日志处理 ============================

CSV_FIELDS = ["time", "action", "folder", "status", "comment"]


def log_path(base_dir: Path) -> Path:
    d = base_dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "kinect_review_log.csv"


def parse_existing_log(base_dir: Path):
    """
    读取已有 CSV 日志
    返回 dict: abs_path -> row dict
    """
    path = log_path(base_dir)
    table = {}

    if not path.exists():
        return table

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abs_key = str((base_dir / row["folder"]).resolve())
            table[abs_key] = row

    return table


def write_log(base_dir: Path, table: dict):
    """
    写 CSV 日志文件
    """
    path = log_path(base_dir)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for _, v in sorted(table.items(), key=lambda kv: kv[1]["folder"]):
            writer.writerow(v)

    print(f"[LOG] CSV 已写入：{path}")


# ============================ 扫描视频帧文件夹 ============================

def find_clip_dirs(root: Path, only_action: str | None = None):
    clips = []
    for p in root.rglob("*_N_run_*"):
        if not p.is_dir():
            continue
        action = p.parent.name
        if only_action and action != only_action:
            continue
        clips.append(p)
    return sorted(clips, key=lambda x: str(x))


# ============================ 单个视频审核 ============================

def review_one_clip(
    clip_dir: Path,
    action: str,
    idx_video: int,
    total_videos: int,
    base_dir: Path,
    comment_init: str = "",
):
    frame_paths = sorted(
        [p for p in clip_dir.iterdir() if p.suffix.lower() in _IMG_EXTS],
        key=lambda p: p.name,
    )

    if not frame_paths:
        return "EMPTY", comment_init

    cur_idx = 0
    status = None
    status_chosen = False
    comment = comment_init
    in_comment_mode = False

    while True:
        img = cv2.imread(str(frame_paths[cur_idx]))
        if img is None:
            cur_idx = min(cur_idx + 1, len(frame_paths) - 1)
            continue

        h, w = img.shape[:2]
        scale = min(WIN_W / w, WIN_H / h, 1.0)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        overlay = img.copy()

        info = [
            f"Action: {action}",
            f"Folder: {clip_dir.name}",
            f"Video: {idx_video}/{total_videos}",
            f"Frame: {cur_idx+1}/{len(frame_paths)}",
        ]
        if status_chosen:
            info.append(f"Status: {status} (Enter 输入 comment)")

        for i, t in enumerate(info):
            cv2.putText(overlay, t, (20, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(
            overlay,
            f"Comment: {comment}",
            (20, overlay.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        hint = (
            "Browse: [A/D] [P/K/S] [Enter] comment [U] undo [Q] quit"
            if not in_comment_mode
            else "Comment: type, Backspace delete, Esc cancel, Enter confirm"
        )

        cv2.putText(
            overlay,
            hint,
            (20, overlay.shape[0] - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WIN_NAME, overlay)
        key = cv2.waitKey(0) & 0xFF

        if not in_comment_mode:
            if key == ord("a") and cur_idx > 0:
                cur_idx -= 1
            elif key == ord("d") and cur_idx < len(frame_paths) - 1:
                cur_idx += 1
            elif key == ord("p"):
                status, status_chosen = "PASS", True
            elif key == ord("k"):
                status, status_chosen = "FAIL", True
            elif key == ord("s"):
                status, status_chosen = "SKIP", True
            elif key == ord("u"):
                return "UNDO", comment
            elif key == ord("q"):
                return None, None
            elif key in (13, 10) and status_chosen:
                in_comment_mode = True
        else:
            if key in (13, 10):
                return status or "SKIP", comment
            elif key == 27:
                status, status_chosen = None, False
                comment = comment_init
                in_comment_mode = False
            elif key in (8, 127):
                comment = comment[:-1]
            elif 32 <= key <= 126:
                comment += chr(key)


# ============================ 主流程 ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--only-action", type=str, default=None)
    args = ap.parse_args()

    root = args.root
    base_dir = root

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)
    cv2.moveWindow(WIN_NAME, WIN_X, WIN_Y)

    table = parse_existing_log(base_dir)
    clips = find_clip_dirs(root, args.only_action)

    i = 0
    while i < len(clips):
        clip = clips[i]
        abs_key = str(clip.resolve())

        if abs_key in table and (table[abs_key]["status"] == "PASS" or table[abs_key]["status"] == "FAIL"):
            i += 1
            continue

        status, comment = review_one_clip(
            clip, clip.parent.name, i + 1, len(clips), base_dir,
            table[abs_key]["comment"] if abs_key in table else "",
        )

        if status is None:
            break

        if status == "UNDO":
            if i > 0:
                table.pop(str(clips[i - 1].resolve()), None)
                write_log(base_dir, table)
                i -= 1
            continue

        table[abs_key] = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": clip.parent.name,
            "folder": str(clip.relative_to(base_dir)),
            "status": status,
            "comment": comment,
        }
        write_log(base_dir, table)
        i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
