#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MP4 视频离线可视化质检脚本（递归扫描 RGB MP4 版本）

功能说明
------------------------------------------------------------
1. 用户提供一个根目录 --root
2. 脚本会递归扫描根目录下所有 .mp4 文件
3. 只保留“文件名中包含 rgb（不区分大小写）”的视频进行检查
4. 日志文件写入根目录下：
       <root>/mp4_review_log.tsv
5. 如果某个视频在日志中已经有记录，则自动跳过
6. 播放窗口中会显示当前视频路径，方便区分
7. 支持设置播放倍速 --speed，范围 (0, 3]
8. 保留以下按键控制：
       [P]     = PASS
       [K]     = SKIP
       [S]     = 保存截图
       [Q]     = 退出程序
       [SPACE] = 暂停/继续

新增功能
------------------------------------------------------------
在“暂停状态”下：
    [A] 按住：连续后退
    [D] 按住：连续前进
松开后自动停止。

注意：
OpenCV 本身不能可靠监听 key-up（按键松开）事件，
所以这里采用“按键自动重复 + 超时停止”的近似实现方式。
在 Windows 上通常能达到接近“按住拖动、松开停止”的效果。

日志格式
------------------------------------------------------------
日志使用 TSV（tab 分隔）格式，字段如下：
    time    status    rel_path    abs_path

使用示例
------------------------------------------------------------
1) 默认 1 倍速：
   python review_rgb_mp4_recursive.py --root "D:/your/root/path"

2) 2 倍速播放：
   python review_rgb_mp4_recursive.py --root "D:/your/root/path" --speed 2.0

3) 3 倍速播放：
   python review_rgb_mp4_recursive.py --root "D:/your/root/path" --speed 3.0
"""

import os
import sys
import cv2
import math
import time
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================
# 可调参数：暂停状态下 A / D 连续前后拖动的速度与停止阈值
# ============================================================

# 暂停时按住 A 或 D，连续前后移动的速度（“伪播放”速度）
SCRUB_FPS = 12.0

# 如果超过这个时间（秒）没有再收到 A/D 的重复按键事件，
# 则认为用户已经松开按键，自动停止移动
HOLD_TIMEOUT_SEC = 0.16


# ============================================================
# 工具函数：获取屏幕尺寸（用于窗口居中）
# ============================================================

def get_screen_size():
    """
    尝试获取当前屏幕分辨率，用于将 OpenCV 窗口居中显示。
    若获取失败，则退回到常见的 1920x1080。
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


# ============================================================
# 工具函数：构造稳定的文件键
# ============================================================

def make_abs_key(path: Path) -> str:
    """
    为日志中的文件建立一个稳定的唯一键。
    在 Windows 下使用 normcase 可以减少大小写差异影响。
    """
    return os.path.normcase(str(path.resolve()))


# ============================================================
# 日志相关函数
# ============================================================

def parse_existing_log(log_file: Path):
    """
    读取已有日志。

    返回：
        table: dict
            key   = 规范化后的绝对路径
            value = {
                "time": ...,
                "status": ...,
                "rel_path": ...,
                "abs_path": ...
            }

    注意：
    只要日志里已有记录，后续就会跳过对应文件。
    """
    table = {}

    if not log_file.exists():
        return table

    lines = log_file.read_text(encoding="utf-8").splitlines()
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith("time\tstatus\trel_path\tabs_path"):
            continue

        parts = s.split("\t")
        if len(parts) < 4:
            continue

        tm, status, rel_path, abs_path = parts[:4]
        key = os.path.normcase(abs_path)

        table[key] = {
            "time": tm,
            "status": status,
            "rel_path": rel_path,
            "abs_path": abs_path,
        }

    return table


def write_log(log_file: Path, table: dict):
    """
    将当前日志表写回到根目录下的 log 文件。

    使用 TSV 格式，避免路径中空格导致解析错误。
    """
    lines = [
        "# RGB MP4 review log",
        "time\tstatus\trel_path\tabs_path"
    ]

    items = sorted(table.items(), key=lambda kv: kv[1]["rel_path"].lower())

    for _, v in items:
        lines.append(
            f"{v['time']}\t{v['status']}\t{v['rel_path']}\t{v['abs_path']}"
        )

    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[LOG] 写入：{log_file}")


# ============================================================
# 文件扫描函数
# ============================================================

def find_rgb_mp4_files(root: Path):
    """
    递归扫描根目录下所有 .mp4 文件，
    仅保留“文件名中包含 rgb（不区分大小写）”的文件。
    """
    results = []
    for p in root.rglob("*.mp4"):
        if ("rgb" in p.name.lower()) and "001484412812" in str(p):
            results.append(p)

    results.sort(key=lambda x: str(x).lower())
    return results


# ============================================================
# 绘制文字辅助函数
# ============================================================

def wrap_text_by_width(text, max_width_px, font, font_scale, thickness):
    """
    根据目标像素宽度对文本进行简单换行。
    """
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        candidate = current + " " + word
        (w, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if w <= max_width_px:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def draw_text_block(
    image,
    lines,
    x=15,
    y=30,
    line_gap=8,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.65,
    thickness=2,
    bg_alpha=0.2,  # 背景透明度，越小越透明；建议 0.2 ~ 0.35
):
    """
    在图像左上角绘制多行文字，并使用半透明背景增强可读性。
    bg_alpha:
        0.0 = 完全透明
        1.0 = 完全不透明
    """
    if image is None:
        return image

    img = image.copy()
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    if not text_sizes:
        return img

    max_w = max(size[0] for size in text_sizes)
    total_h = sum(size[1] for size in text_sizes) + line_gap * (len(lines) - 1)

    pad = 10
    top_left = (max(0, x - pad), max(0, y - text_sizes[0][1] - pad))
    bottom_right = (x + max_w + pad, y + total_h + pad)

    # ---------- 半透明背景 ----------
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), thickness=-1)
    img = cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0)

    # ---------- 白色文字 ----------
    cur_y = y
    for line, (_, th) in zip(lines, text_sizes):
        cv2.putText(
            img,
            line,
            (x, cur_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        cur_y += th + line_gap

    return img


# ============================================================
# 视频帧定位函数
# ============================================================

def get_current_frame_index(cap):
    """
    获取当前已读取到的帧索引（尽量返回“当前显示帧”的索引）。
    OpenCV 在 read() 后 CAP_PROP_POS_FRAMES 往往指向“下一帧位置”，
    因此这里减 1 更接近实际当前帧。
    """
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if pos is None or math.isnan(pos):
        return 0
    return max(0, int(round(pos)) - 1)


def read_frame_at_index(cap, frame_idx, total_frames=None):
    """
    跳转到指定帧并读取该帧。

    参数：
        cap         : cv2.VideoCapture
        frame_idx   : 目标帧索引
        total_frames: 视频总帧数，可选

    返回：
        (ok, frame, actual_idx)
    """
    if total_frames is not None and total_frames > 0:
        frame_idx = max(0, min(frame_idx, total_frames - 1))
    else:
        frame_idx = max(0, frame_idx)

    ok_set = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    if not ok_set:
        return False, None, None

    ret, frame = cap.read()
    if not ret or frame is None:
        return False, None, None

    actual_idx = get_current_frame_index(cap)
    return True, frame, actual_idx


def step_relative(cap, current_idx, step, total_frames=None):
    """
    相对于当前帧前进/后退 step 帧，并读取目标帧。
    step 可以为正（前进）或负（后退）。
    """
    target_idx = current_idx + step
    return read_frame_at_index(cap, target_idx, total_frames=total_frames)


# ============================================================
# 播放器核心函数
# ============================================================

def review_one_mp4(path: Path, rel_path: str, speed: float):
    """
    检查单个 MP4 文件。

    参数：
        path     : 视频完整路径
        rel_path : 相对根目录路径（用于显示和写日志）
        speed    : 播放速度，范围 (0, 3]

    返回：
        "PASS" / "SKIP" / None
        其中：
          - 返回 None 表示用户按 Q 主动退出整个程序
          - 返回 PASS / SKIP 表示本视频已完成判断
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERROR] 无法打开文件：{path}")
        return "SKIP"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6 or math.isnan(fps):
        fps = 30.0

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames is None or math.isnan(total_frames) or total_frames <= 0:
        total_frames = None
    else:
        total_frames = int(total_frames)

    delay_ms = max(1, int(round(1000.0 / fps / speed)))

    win_name = "MP4 Review"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    window_w, window_h = 1280, 720
    cv2.resizeWindow(win_name, window_w, window_h)

    screen_w, screen_h = get_screen_size()
    pos_x = max(0, (screen_w - window_w) // 2)
    pos_y = max(0, (screen_h - window_h) // 2)
    cv2.moveWindow(win_name, pos_x, pos_y)

    paused = False
    status = "SKIP"
    last_frame = None
    current_frame_idx = 0

    # 与“按住 A / D 连续移动”有关的状态
    hold_dir = 0           # -1: backward, +1: forward, 0: idle
    hold_last_event = 0.0  # 上一次接收到 A/D 事件的时间
    hold_last_step = 0.0   # 上一次实际执行前后移动的时间
    scrub_interval = 1.0 / SCRUB_FPS

    print("\nControls:")
    print("  [P]=Pass  [K]=Skip  [S]=Save  [Q]=Quit  [SPACE]=Pause/Resume")
    print("  Pause mode: hold [A]=backward, hold [D]=forward")
    print(f"[PLAY] speed={speed:.2f}x  delay={delay_ms} ms/frame")

    while True:
        # ----------------------------------------------------
        # 正常播放状态
        # ----------------------------------------------------
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame.copy()
            current_frame_idx = get_current_frame_index(cap)

            # 一旦恢复播放，就清空 hold 状态
            hold_dir = 0
            hold_last_event = 0.0
            hold_last_step = 0.0

        # ----------------------------------------------------
        # 暂停状态下：根据 A / D 的“持续按住”近似实现连续前进/后退
        # ----------------------------------------------------
        else:
            now = time.monotonic()

            if hold_dir != 0:
                # 如果距离最近一次 A/D 按键事件没有超时，则继续移动
                if (now - hold_last_event) <= HOLD_TIMEOUT_SEC:
                    if (now - hold_last_step) >= scrub_interval:
                        ok, frame, actual_idx = step_relative(
                            cap,
                            current_frame_idx,
                            hold_dir,
                            total_frames=total_frames
                        )
                        if ok and frame is not None:
                            last_frame = frame
                            current_frame_idx = actual_idx
                        hold_last_step = now
                else:
                    # 超时，认为已经松开按键，停止移动
                    hold_dir = 0

        if last_frame is None:
            break

        disp = last_frame.copy()
        h, w = disp.shape[:2]
        max_text_width = max(200, w - 40)

        header_line = f"File: {rel_path}"
        header_lines = wrap_text_by_width(
            header_line,
            max_text_width,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )

        if total_frames is not None:
            frame_info = f"Frame: {current_frame_idx + 1}/{total_frames}"
        else:
            frame_info = f"Frame: {current_frame_idx + 1}"

        state_text = "PLAYING"
        if paused and hold_dir == 0:
            state_text = "PAUSED"
        elif paused and hold_dir < 0:
            state_text = "SCRUB BACKWARD (hold A)"
        elif paused and hold_dir > 0:
            state_text = "SCRUB FORWARD (hold D)"

        info_lines = [
            f"State: {state_text}",
            f"Speed: {speed:.2f}x   FPS: {fps:.2f}   Delay: {delay_ms} ms   {frame_info}",
            "Controls: P=PASS  K=SKIP  S=SNAPSHOT  SPACE=PAUSE/RESUME",
            "Pause mode: hold A=backward   hold D=forward   release=stop",
        ]

        disp = draw_text_block(
            disp,
            header_lines + info_lines,
            x=15,
            y=30,
            line_gap=8,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.65,
            thickness=2,
        )

        cv2.imshow(win_name, disp)

        # 播放状态按 speed 控制；暂停状态更短轮询，便于及时响应按键
        wait_ms = 10 if paused else delay_ms
        key = cv2.waitKey(wait_ms) & 0xFF

        # -------------------------
        # 通用按键
        # -------------------------
        if key == ord(' '):
            paused = not paused
            if not paused:
                hold_dir = 0

        elif key == ord('p'):
            status = "PASS"
            break

        elif key == ord('k'):
            status = "SKIP"
            break

        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = path.parent / f"{path.stem}_snapshot_{timestamp}.png"
            cv2.imwrite(str(snap_path), last_frame)
            print(f"[SAVE] 截图已保存：{snap_path}")

        elif key == ord('q'):
            cap.release()
            cv2.destroyWindow(win_name)
            return None

        # -------------------------
        # 暂停模式下 A / D 连续前进后退
        # -------------------------
        elif paused and key in (ord('a'), ord('d')):
            now = time.monotonic()
            new_dir = -1 if key == ord('a') else 1

            # 如果方向发生变化，立即切换方向
            hold_dir = new_dir
            hold_last_event = now

            # 首次按下时立即移动一帧，提升手感
            # 若正在持续自动重复，也允许按节奏继续移动
            if (now - hold_last_step) >= min(scrub_interval, 0.05):
                ok, frame, actual_idx = step_relative(
                    cap,
                    current_frame_idx,
                    hold_dir,
                    total_frames=total_frames
                )
                if ok and frame is not None:
                    last_frame = frame
                    current_frame_idx = actual_idx
                hold_last_step = now

    cap.release()
    cv2.destroyWindow(win_name)
    return status


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="递归检查根目录下文件名包含 rgb 的 MP4 视频"
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="根目录，脚本会递归扫描其下所有 .mp4 文件，并筛选文件名中包含 rgb 的视频"
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="播放速度，范围 (0, 3]，例如 1.0 / 2.0 / 3.0"
    )

    args = ap.parse_args()
    root: Path = args.root
    speed: float = args.speed

    if not root.exists():
        print(f"[ERROR] 根目录不存在：{root}")
        sys.exit(1)

    if not root.is_dir():
        print(f"[ERROR] 提供的 root 不是目录：{root}")
        sys.exit(1)

    if not (0.0 < speed <= 3.0):
        print(f"[ERROR] --speed 必须满足 0 < speed <= 3.0，当前值为：{speed}")
        sys.exit(1)

    root = root.resolve()
    log_file = root / "mp4_review_log.tsv"

    print(f"[ROOT ] {root}")
    print(f"[LOG  ] {log_file}")
    print(f"[SPEED] {speed:.2f}x")

    table = parse_existing_log(log_file)
    mp4_list = find_rgb_mp4_files(root)

    if not mp4_list:
        print("[INFO] 未找到文件名中包含 rgb 的 mp4 文件")
        write_log(log_file, table)
        return

    print(f"[FOUND] 共找到 {len(mp4_list)} 个候选视频文件")

    skipped_logged_count = 0
    new_count = 0

    for idx, mp4 in enumerate(mp4_list, start=1):
        abs_key = make_abs_key(mp4)

        try:
            rel_path = str(mp4.relative_to(root))
        except Exception:
            rel_path = str(mp4)

        # 只要日志里已有记录，就直接跳过
        if abs_key in table:
            skipped_logged_count += 1
            print(f"[{idx}/{len(mp4_list)}] [SKIP LOGGED] {rel_path}  -> {table[abs_key]['status']}")
            continue

        print(f"\n[{idx}/{len(mp4_list)}] [REVIEW] {rel_path}")
        status = review_one_mp4(mp4, rel_path, speed)

        if status is None:
            write_log(log_file, table)
            print("[QUIT] 用户主动退出，当前日志已保存")
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        table[abs_key] = {
            "time": now_str,
            "status": status,
            "rel_path": rel_path,
            "abs_path": str(mp4.resolve()),
        }

        write_log(log_file, table)
        new_count += 1

    print("\n全部完成！")
    print(f"  - 总候选文件数: {len(mp4_list)}")
    print(f"  - 已有日志而跳过: {skipped_logged_count}")
    print(f"  - 本次新检查数量: {new_count}")
    print(f"  - 日志文件位置: {log_file}")


if __name__ == "__main__":
    main()