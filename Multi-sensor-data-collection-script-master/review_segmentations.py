#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import pandas as pd


# ====== OpenCV waitKeyEx 常见键值（Windows）======
KEY_ENTER_1 = 13
KEY_ENTER_2 = 10
KEY_ESC = 27

KEY_LEFT = 2424832
KEY_UP = 2490368
KEY_RIGHT = 2555904
KEY_DOWN = 2621440

KEY_BACKSPACE_1 = 8
KEY_BACKSPACE_2 = 127

KEY_SPACE = 32

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


@dataclass(frozen=True)
class ClipItem:
    letter: str
    action: str
    clip: str
    cam_dir: Path


def natural_sort_key(name: str):
    return name.lower()


def get_frames_sorted(cam_dir: Path) -> List[Path]:
    frames = []
    for name in os.listdir(cam_dir):
        p = cam_dir / name
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            frames.append(p)
    frames.sort(key=lambda p: natural_sort_key(p.name))
    return frames


def collect_clips(root: Path, cam_name: str,
                  only_letters: Optional[List[str]] = None,
                  only_actions: Optional[List[str]] = None) -> List[ClipItem]:
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    only_letters_set = set([x.strip() for x in only_letters]) if only_letters else None
    only_actions_set = set([x.strip() for x in only_actions]) if only_actions else None

    items: List[ClipItem] = []

    letter_order = ["N", "M", "MR", "J"]
    letter_dirs = []
    for L in letter_order:
        p = root / L
        if p.is_dir():
            if only_letters_set is None or L in only_letters_set:
                letter_dirs.append(p)

    for letter_dir in letter_dirs:
        letter = letter_dir.name
        action_dirs = sorted([d for d in letter_dir.iterdir() if d.is_dir()],
                             key=lambda x: x.name.lower())

        for action_dir in action_dirs:
            action = action_dir.name
            if only_actions_set is not None and action not in only_actions_set:
                continue

            clip_dirs = sorted([d for d in action_dir.iterdir() if d.is_dir()],
                               key=lambda x: x.name.lower())

            for clip_dir in clip_dirs:
                clip = clip_dir.name
                cam_dir = clip_dir / cam_name
                if cam_dir.is_dir():
                    has_img = any((cam_dir / f).suffix.lower() in IMG_EXTS for f in os.listdir(cam_dir))
                    if has_img:
                        items.append(ClipItem(letter, action, clip, cam_dir))

    return items


def read_existing_csv(csv_path: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    results: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    if not csv_path.exists():
        return results

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    needed = ["字母文件夹", "动作文件夹", "片段文件夹", "是否通过", "不通过理由", "后续处理方式", "清晰度"]
    # 兼容旧 CSV：没有“清晰度”也能读
    if "清晰度" not in df.columns:
        df["清晰度"] = ""

    base_needed = ["字母文件夹", "动作文件夹", "片段文件夹", "是否通过", "不通过理由", "后续处理方式"]
    if any(col not in df.columns for col in base_needed):
        return results

    for _, r in df.iterrows():
        key = (r["字母文件夹"], r["动作文件夹"], r["片段文件夹"])
        results[key] = {
            "是否通过": r.get("是否通过", ""),
            "不通过理由": r.get("不通过理由", ""),
            "后续处理方式": r.get("后续处理方式", ""),
            "清晰度": r.get("清晰度", ""),
        }
    return results


def write_csv(csv_path: Path, items: List[ClipItem],
              results: Dict[Tuple[str, str, str], Dict[str, str]]):
    rows = []
    for it in items:
        key = (it.letter, it.action, it.clip)
        rec = results.get(key, {"是否通过": "", "不通过理由": "", "后续处理方式": "", "清晰度": ""})
        rows.append({
            "字母文件夹": it.letter,
            "动作文件夹": it.action,
            "片段文件夹": it.clip,
            "是否通过": rec.get("是否通过", ""),
            "不通过理由": rec.get("不通过理由", ""),
            "后续处理方式": rec.get("后续处理方式", ""),
            "清晰度": rec.get("清晰度", ""),
        })
    df = pd.DataFrame(rows, columns=[
        "字母文件夹", "动作文件夹", "片段文件夹",
        "是否通过", "不通过理由", "后续处理方式", "清晰度"
    ])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def draw_text_with_outline(img, text, org, font_scale=0.7, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = org
    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def overlay_info(frame, it: ClipItem, idx: int, total: int,
                 status_line: str,
                 frame_line: str,
                 input_mode: bool,
                 paused: bool,
                 fps: float,
                 input_prompt: str = "",
                 input_buffer: str = ""):
    show = frame.copy()
    pad = 12
    y = 30

    play_state = "PAUSED" if paused else "PLAYING"
    help_line = (
        "Keys: Enter=PASS->clarity | Down=FAIL->reason/follow/clarity | "
        "Right=Next | Left=Prev | Space=Pause/Play | A/D=Prev/Next frame | ESC=Save&Exit"
    )

    lines = [
        f"[{idx+1}/{total}] {it.letter} / {it.action} / {it.clip}",
        f"Cam: {it.cam_dir.name}",
        help_line,
        f"State: {play_state} | FPS: {fps}",
        status_line,
        frame_line,
    ]
    for s in lines:
        draw_text_with_outline(show, s, (pad, y), font_scale=0.7)
        y += 28

    if input_mode:
        y += 10
        draw_text_with_outline(show, "INPUT MODE:", (pad, y), font_scale=0.85)
        y += 32
        draw_text_with_outline(show, input_prompt, (pad, y), font_scale=0.8)
        y += 32

        h, w = show.shape[:2]
        box_top = y - 24
        box_bottom = y + 24
        cv2.rectangle(show, (pad, box_top), (w - pad, box_bottom), (255, 255, 255), 2)

        cursor = "|" if (int(cv2.getTickCount() / cv2.getTickFrequency() * 2) % 2 == 0) else " "
        draw_text_with_outline(show, input_buffer + cursor, (pad + 8, y + 10), font_scale=0.8)

        y += 40
        draw_text_with_outline(show, "Enter=Confirm, Backspace=Delete, ESC=Cancel input", (pad, y), font_scale=0.7)

    return show


def key_to_char(k: int) -> Optional[str]:
    if k < 0:
        return None
    if k in (KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN):
        return None
    if k in (KEY_ESC, KEY_ENTER_1, KEY_ENTER_2, KEY_BACKSPACE_1, KEY_BACKSPACE_2):
        return None
    if 32 <= k <= 126:
        return chr(k)
    return None



def clamp_fps(fps: float) -> float:
    fps = float(fps)
    if fps < 1:
        fps = 1.0
    if fps > 240:
        fps = 240.0
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\RGB")
    parser.add_argument("--cam", type=str, default="cam_001431512812")
    parser.add_argument("--out_csv", type=str, default=r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\RGB\review_results.csv")

    parser.add_argument("--letters", nargs="*", default=None, help="Only check letter folders, e.g. --letters N MR")
    parser.add_argument("--actions", nargs="*", default=None, help="Only check actions, e.g. --actions adjust_slider")

    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS (visual speed), e.g. --fps 15")

    args = parser.parse_args()

    root = Path(args.root)
    cam_name = args.cam
    out_csv = Path(args.out_csv)

    fps = clamp_fps(args.fps)
    delay_ms = max(1, int(round(1000.0 / fps)))

    items = collect_clips(root, cam_name, args.letters, args.actions)
    if not items:
        print(f"No clips found. root={root}, cam={cam_name}, letters={args.letters}, actions={args.actions}")
        return

    results = read_existing_csv(out_csv)

    win = "Clip Checker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    i = 0
    total = len(items)

    # 输入状态机：reason -> follow -> clarity
    input_mode = False
    input_stage: Optional[str] = None  # None / "reason" / "follow" / "clarity"
    input_buffer = ""

    # 暂存（用于 FAIL / PASS 流程）
    tmp_passfail = ""   # "PASS" / "FAIL"
    tmp_reason = ""
    tmp_follow = ""

    paused = False

    while 0 <= i < total:
        it = items[i]
        key = (it.letter, it.action, it.clip)

        frames = get_frames_sorted(it.cam_dir)
        if not frames:
            results[key] = {
                "是否通过": "FAIL",
                "不通过理由": "cam folder empty",
                "后续处理方式": "check data export",
                "清晰度": ""
            }
            write_csv(out_csv, items, results)
            i += 1
            continue

        fidx = 0

        while True:
            img_path = frames[fidx]
            frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if frame is None:
                if not paused and not input_mode:
                    fidx = (fidx + 1) % len(frames)
                continue

            rec = results.get(key, {"是否通过": "", "不通过理由": "", "后续处理方式": "", "清晰度": ""})
            status_line = f"Status: {rec.get('是否通过','')}"
            if rec.get("是否通过", ""):
                status_line += f" | Clarity: {rec.get('清晰度','')}"
            if rec.get("是否通过", "") == "FAIL":
                status_line += f" | Reason: {rec.get('不通过理由','')} | Action: {rec.get('后续处理方式','')}"

            frame_line = f"Frame: {fidx+1}/{len(frames)} | {img_path.name}"

            if input_mode:
                if input_stage == "reason":
                    prompt = "请输入不通过理由："
                elif input_stage == "follow":
                    prompt = "请输入后续处理方式："
                else:
                    prompt = "请输入清晰度（例如 1-5 或者 0-100）："
            else:
                prompt = ""

            show = overlay_info(
                frame, it, i, total,
                status_line=status_line,
                frame_line=frame_line,
                input_mode=input_mode,
                paused=paused,
                fps=fps,
                input_prompt=prompt,
                input_buffer=input_buffer
            )
            cv2.imshow(win, show)

            k = cv2.waitKeyEx(delay_ms if (not paused and not input_mode) else 30)

            # 无按键：播放时自动前进；暂停/输入不自动动
            if k == -1 and not input_mode:
                if not paused:
                    fidx = (fidx + 1) % len(frames)
                continue
            if k == -1 and input_mode:
                continue

            # Space：暂停/继续（输入模式不响应）
            if (not input_mode) and (k == KEY_SPACE):
                paused = not paused
                continue

            # ====== 输入模式处理 ======
            if input_mode:
                if k == KEY_ESC:
                    # 取消输入，不改变结果（保持之前的 rec）
                    input_mode = False
                    input_stage = None
                    input_buffer = ""
                    tmp_passfail = ""
                    tmp_reason = ""
                    tmp_follow = ""
                    continue

                if k in (KEY_BACKSPACE_1, KEY_BACKSPACE_2):
                    input_buffer = input_buffer[:-1]
                    continue

                if k in (KEY_ENTER_1, KEY_ENTER_2):
                    text = input_buffer.strip()

                    if input_stage == "reason":
                        tmp_reason = text
                        input_buffer = ""
                        input_stage = "follow"
                        continue

                    if input_stage == "follow":
                        tmp_follow = text
                        input_buffer = ""
                        input_stage = "clarity"
                        continue

                    if input_stage == "clarity":
                        clarity = text

                        # 写入结果（PASS 或 FAIL）
                        if tmp_passfail == "PASS":
                            results[key] = {"是否通过": "PASS", "不通过理由": "", "后续处理方式": "", "清晰度": clarity}
                        else:
                            results[key] = {"是否通过": "FAIL", "不通过理由": tmp_reason, "后续处理方式": tmp_follow, "清晰度": clarity}

                        write_csv(out_csv, items, results)

                        # 清空输入状态 -> 下一段
                        input_mode = False
                        input_stage = None
                        input_buffer = ""
                        tmp_passfail = ""
                        tmp_reason = ""
                        tmp_follow = ""
                        i += 1
                        break

                ch = key_to_char(k)
                if ch is not None:
                    input_buffer += ch
                continue

            # ====== 非输入模式处理 ======
            if k == KEY_ESC:
                write_csv(out_csv, items, results)
                cv2.destroyAllWindows()
                print(f"Saved: {out_csv}")
                return

            # Enter：PASS -> 进入清晰度输入
            if k in (KEY_ENTER_1, KEY_ENTER_2):
                input_mode = True
                input_stage = "clarity"
                input_buffer = ""
                tmp_passfail = "PASS"
                tmp_reason = ""
                tmp_follow = ""
                continue

            # Down：FAIL -> reason -> follow -> clarity
            if k == KEY_DOWN:
                input_mode = True
                input_stage = "reason"
                input_buffer = ""
                tmp_passfail = "FAIL"
                tmp_reason = ""
                tmp_follow = ""
                continue

            if k == KEY_RIGHT:
                i = min(i + 1, total - 1)
                break

            if k == KEY_LEFT:
                i = max(i - 1, 0)
                break

            # A/D：逐帧（暂停时必须用，播放时也可手动跳）
            if k in (ord('a'), ord('A')):
                fidx = (fidx - 1) % len(frames)
                continue
            if k in (ord('d'), ord('D')):
                fidx = (fidx + 1) % len(frames)
                continue

            if not paused:
                fidx = (fidx + 1) % len(frames)

    write_csv(out_csv, items, results)
    cv2.destroyAllWindows()
    print(f"Done. Saved: {out_csv}")


if __name__ == "__main__":
    main()
