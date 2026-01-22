#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import csv
from pathlib import Path
import argparse

# <<< CHANGED: 新增（后台预加载）
import threading
from queue import PriorityQueue
from collections import OrderedDict

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_images(img_dir: Path):
    img_paths = [p for p in img_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in IMG_EXTS]
    img_paths = sorted(img_paths, key=lambda p: p.stem)
    return img_paths


def draw_overlay(img, text_lines, font_scale=0.6, thickness=1):
    """在左上角画一小块黑底文字，显示当前信息和提示。"""
    overlay = img.copy()
    x, y0 = 10, 25

    alpha = 0.6
    bg_top_left = (5, 5)
    bg_bottom_right = (540, y0 + len(text_lines) * 20)
    cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (0, 0, 0), -1)

    for i, line in enumerate(text_lines):
        y = y0 + i * 20
        cv2.putText(overlay, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def get_text_from_window(base_img, base_lines, prompt, init_text=""):
    """
    在 OpenCV 窗口中输入一行文本：
    - 可输入 ASCII 可打印字符
    - Backspace / Delete：删除最后一个字符
    - Enter：确认并返回字符串
    - ESC：取消，返回 None
    """
    text = init_text
    while True:
        lines = base_lines + [prompt, f"> {text}"]
        img_disp = draw_overlay(base_img.copy(), lines)
        cv2.imshow("Annotator", img_disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (13, 10):  # Enter
            return text
        elif key == 27:  # ESC
            return None
        elif key in (8, 127):  # Backspace or Delete
            text = text[:-1]
        else:
            if 32 <= key <= 126:
                text += chr(key)


def ask_yes_no(base_img, base_lines, question):
    """
    在 OpenCV 窗口中询问 y/n，返回 True / False。
    - Y/y：True
    - N/n 或 ESC：False
    """
    while True:
        lines = base_lines + [question, "按 Y 确认，N 或 ESC 取消"]
        img_disp = draw_overlay(base_img.copy(), lines)
        cv2.imshow("Annotator", img_disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('y'), ord('Y')):
            return True
        elif key in (ord('n'), ord('N'), 27):
            return False


# <<< CHANGED: 后台预加载缓存（先可用后补齐）
class AsyncPreloadAllFramesCache:
    """
    - 不阻塞启动：后台线程逐步把所有帧读进内存
    - get(idx)：若未加载，主线程同步读一次（保证立即可用）
    - request_prefetch_around：给当前 idx 附近帧更高优先级，减少翻帧卡顿
    """

    def __init__(self, img_paths, num_workers=2, prefetch_radius=60, max_pending=20000):
        self.img_paths = img_paths
        self.n = len(img_paths)
        self.prefetch_radius = int(prefetch_radius)

        self.lock = threading.Lock()
        self.cache = [None] * self.n              # idx -> image or None
        self.loaded = [False] * self.n            # idx -> bool (是否已尝试加载过)
        self.inflight = set()                     # idx 正在被后台加载

        self.q = PriorityQueue(maxsize=int(max_pending))
        self.stop_event = threading.Event()
        self.workers = []

        # 先把“全量任务”以低优先级塞进队列（不会阻塞主线程）
        # priority 数字越小越优先
        for i in range(self.n):
            # 低优先级：100
            self._try_put_task(priority=100, idx=i)

        # 启动后台 worker
        for _ in range(int(num_workers)):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)

    def _try_put_task(self, priority: int, idx: int):
        if idx < 0 or idx >= self.n:
            return
        with self.lock:
            if self.loaded[idx] or idx in self.inflight:
                return
        try:
            self.q.put_nowait((priority, idx))
        except Exception:
            # 队列满：忽略（主线程 get 会兜底同步读）
            pass

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                priority, idx = self.q.get(timeout=0.2)
            except Exception:
                continue

            with self.lock:
                if self.loaded[idx] or idx in self.inflight:
                    self.q.task_done()
                    continue
                self.inflight.add(idx)

            img = cv2.imread(str(self.img_paths[idx]))

            with self.lock:
                self.cache[idx] = img
                self.loaded[idx] = True
                self.inflight.discard(idx)

            self.q.task_done()

    def request_prefetch_around(self, idx: int):
        # 附近帧高优先级：0~radius（越近越先）
        r = self.prefetch_radius
        for off in range(0, r + 1):
            p = off  # 越小越优先
            self._try_put_task(priority=p, idx=idx + off)
            self._try_put_task(priority=p, idx=idx - off)

    def get(self, idx: int):
        # 先尝试直接返回缓存
        with self.lock:
            if self.loaded[idx]:
                return self.cache[idx]

        # 未加载：主线程同步读一次，保证立即可用
        img = cv2.imread(str(self.img_paths[idx]))

        with self.lock:
            self.cache[idx] = img
            self.loaded[idx] = True
            self.inflight.discard(idx)

        return img

    def close(self):
        self.stop_event.set()
        # 不强制 join 很久（daemon 线程也会随主进程退出）
        for t in self.workers:
            if t.is_alive():
                t.join(timeout=0.2)


def main():
    parser = argparse.ArgumentParser(
        description="逐帧标注动作片段（开始/结束 + 动作标签 + 作用物体 + 可选手别 + 存疑标记，全在 OpenCV 窗口操作）"
    )
    parser.add_argument("--img_dir", type=str, default=r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\MR\kinect\run_10\001431512812\frames_rgb",
                        help="RGB 帧所在的文件夹（文件名为 20251103_171334_002627 这种格式）")
    parser.add_argument("--out_csv", type=str, default=r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\MR\kinect\run_10\MR_run_10_annotation.csv",
                        help="输出的 CSV 路径（默认：segments.csv）")
    parser.add_argument("--use-hand", action="store_true",
                        help="是否标注左右手（L/R/B）。若不设置，则不标注手别，也不会在 CSV 中出现 hand 列。")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")

    img_paths = load_images(img_dir)
    if not img_paths:
        raise RuntimeError(f"在 {img_dir} 下没有找到图像文件")

    print(f"共找到 {len(img_paths)} 张图像。")
    print("操作全部在 OpenCV 窗口中完成，不再使用命令行输入。")
    print("后台预加载已开启：先可用后补齐（不会等待全部帧加载完）。")

    # <<< CHANGED: 启动后台预加载缓存（参数可按需调）
    # 720p/4000帧：num_workers=2 通常足够；prefetch_radius=60 翻帧很顺
    cache = AsyncPreloadAllFramesCache(
        img_paths,
        num_workers=2,
        prefetch_radius=60
    )

    idx = 0
    n = len(img_paths)

    start_idx = None
    end_idx = None
    segments = []

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotator", 1280, 720)

    try:
        while True:
            img_path = img_paths[idx]

            # <<< CHANGED: 优先预取当前附近，减少下一帧卡顿
            cache.request_prefetch_around(idx)

            img = cache.get(idx)
            if img is None:
                print(f"警告：无法读取图像 {img_path}")
                idx = min(n - 1, idx + 1)
                continue

            current_name = img_path.stem

            base_lines = [
                f"Frame {idx+1}/{n}",
                f"File : {img_path.name}",
                f"Start idx : {start_idx+1 if start_idx is not None else 'None'}",
                f"End idx   : {end_idx+1 if end_idx is not None else 'None'}",
                f"Segments  : {len(segments)}",
                "Keys:",
                "  A/← prev | D/→ next",
                "  S set start | E set end",
                "  Z clear start | X clear end",
                "  N new seg | U undo seg | Q quit",
            ]
            if args.use_hand:
                base_lines.append("  输入时: '.' 复制上一条动作/物体/手别")
            else:
                base_lines.append("  输入时: '.' 复制上一条动作/物体")

            img_disp = draw_overlay(img.copy(), base_lines)
            cv2.imshow("Annotator", img_disp)
            key = cv2.waitKey(0) & 0xFF

            # ============== 导航（保持原逻辑：点按一帧一帧走，无播放模式） ==============
            if key in (ord('d'), ord('D'), 83):    # → 方向键
                if idx < n - 1:
                    idx += 1

            elif key in (ord('a'), ord('A'), 81):  # ← 方向键
                if idx > 0:
                    idx -= 1

            # ============== 设置开始帧 ==============
            elif key in (ord('s'), ord('S')):
                if start_idx is None:
                    print(f"设置开始帧： idx={idx+1}, name={current_name}")
                    start_idx = idx
                else:
                    if start_idx == idx:
                        print(f"提示：开始帧已经是当前帧 idx={idx+1}，无需修改。")
                    else:
                        print(f"已存在开始帧 idx={start_idx+1}，当前帧为 idx={idx+1}")
                        want = ask_yes_no(img, base_lines,
                                          f"是否覆盖开始帧为当前帧 idx={idx+1}？")
                        if want:
                            print(f"覆盖开始帧：由 idx={start_idx+1} 改为 idx={idx+1}")
                            start_idx = idx
                        else:
                            print("保持原来的开始帧不变。")

            # ============== 设置结束帧 ==============
            elif key in (ord('e'), ord('E')):
                if end_idx is None:
                    print(f"设置结束帧： idx={idx+1}, name={current_name}")
                    end_idx = idx
                else:
                    if end_idx == idx:
                        print(f"提示：结束帧已经是当前帧 idx={idx+1}，无需修改。")
                    else:
                        print(f"已存在结束帧 idx={end_idx+1}，当前帧为 idx={idx+1}")
                        want = ask_yes_no(img, base_lines,
                                          f"是否覆盖结束帧为当前帧 idx={idx+1}？")
                        if want:
                            print(f"覆盖结束帧：由 idx={end_idx+1} 改为 idx={idx+1}")
                            end_idx = idx
                        else:
                            print("保持原来的结束帧不变。")

            # ============== 清除开始帧 / 结束帧 ==============
            elif key in (ord('z'), ord('Z')):
                if start_idx is not None:
                    print(f"清除开始帧：原 start_idx={start_idx+1}")
                    start_idx = None
                else:
                    print("当前没有开始帧可清除。")

            elif key in (ord('x'), ord('X')):
                if end_idx is not None:
                    print(f"清除结束帧：原 end_idx={end_idx+1}")
                    end_idx = None
                else:
                    print("当前没有结束帧可清除。")

            # ============== 撤销上一条 segment ==============
            elif key in (ord('u'), ord('U')):
                if segments:
                    last = segments.pop()
                    if args.use_hand:
                        print(
                            f"撤销片段: action={last['action']}, object={last['object']}, hand={last.get('hand','')}, "
                            f"start={last['start']} (idx={last['start_idx']+1}), "
                            f"end={last['end']} (idx={last['end_idx']+1}), mark={last['mark']}"
                        )
                    else:
                        print(
                            f"撤销片段: action={last['action']}, object={last['object']}, "
                            f"start={last['start']} (idx={last['start_idx']+1}), "
                            f"end={last['end']} (idx={last['end_idx']+1}), mark={last['mark']}"
                        )
                else:
                    print("当前没有片段可以撤销。")

            # ============== 新建片段 (N) ==============
            elif key in (ord('n'), ord('N')):
                if start_idx is None or end_idx is None:
                    print("请先用 S / E 设置开始帧和结束帧（必要时可用 Z/X 清除后重设）。")
                    continue

                s_idx = min(start_idx, end_idx)
                e_idx = max(start_idx, end_idx)
                start_name = img_paths[s_idx].stem
                end_name = img_paths[e_idx].stem
                print(f"准备新片段： start={start_name}, end={end_name}")

                prev_action = segments[-1]["action"] if segments else None
                prev_object = segments[-1]["object"] if segments else None
                prev_hand   = segments[-1].get("hand") if (segments and args.use_hand) else None

                action = None
                while True:
                    prompt = "Action: 输入动作 (Enter确认, ESC取消, '.'=复制上一条, 为空则跳过片段)"
                    action_text = get_text_from_window(img, base_lines, prompt, init_text="")

                    if action_text is None:
                        print("取消本次片段。")
                        action = None
                        break

                    action_text = action_text.strip()

                    if action_text == "":
                        print("动作为空，本次片段跳过。")
                        action = None
                        break
                    if action_text == ".":
                        if prev_action:
                            action = prev_action.strip()
                            print(f"使用上一条动作: {action}")
                            break
                        else:
                            print("无上一条动作可复制，请重新输入。")
                            continue
                    else:
                        action = action_text
                        break

                if action is None:
                    start_idx = None
                    end_idx = None
                    continue

                prompt_obj = "Object: 输入作用物体 (Enter确认, ESC取消, '.'=复制上一条, 可为空)"
                obj_text = get_text_from_window(img, base_lines, prompt_obj, init_text="")

                if obj_text is None:
                    print("取消本次片段。")
                    start_idx = None
                    end_idx = None
                    continue

                if obj_text == ".":
                    if prev_object:
                        obj = prev_object.strip()
                        print(f"使用上一条物体: {obj}")
                    else:
                        obj = ""
                        print("无上一条物体可复制，将 object 留空。")
                else:
                    obj = obj_text.strip()

                if args.use_hand:
                    prompt_hand = "Hand: 输入手别 (L=left, R=right, B=both, '.'=上一条, 可为空)"
                    hand_text = get_text_from_window(img, base_lines, prompt_hand, init_text="")

                    if hand_text is None:
                        print("取消本次片段。")
                        start_idx = None
                        end_idx = None
                        continue

                    hand_text = hand_text.strip()

                    if hand_text == ".":
                        if prev_hand is not None:
                            hand = prev_hand.strip()
                            print(f"使用上一条手别: {hand}")
                        else:
                            hand = ""
                            print("无上一条手别可复制，将 hand 留空。")
                    else:
                        low = hand_text.lower()
                        if low == "l":
                            hand = "left"
                        elif low == "r":
                            hand = "right"
                        elif low == "b":
                            hand = "both"
                        else:
                            hand = hand_text
                else:
                    hand = None

                prompt_mark = "Mark: 输入备注 (Enter确认, ESC取消, 可为空, 如 '?' / 'unclear' )"
                mark_text = get_text_from_window(img, base_lines, prompt_mark, init_text="")

                if mark_text is None:
                    print("取消本次片段。")
                    start_idx = None
                    end_idx = None
                    continue

                mark = mark_text.strip()

                if (obj == "") and (mark == ""):
                    mark = "object unclear"

                seg_record = {
                    "action": action,
                    "object": obj,
                    "start": start_name,
                    "end": end_name,
                    "mark": mark,
                    "start_idx": s_idx,
                    "end_idx": e_idx,
                }
                if args.use_hand:
                    seg_record["hand"] = hand

                segments.append(seg_record)

                if args.use_hand:
                    print(
                        f"已添加片段：action={action}, object={obj if obj else 'None'}, hand={hand if hand else 'None'}, "
                        f"start={start_name} (idx={s_idx+1}), end={end_name} (idx={e_idx+1}), total frames={e_idx - s_idx}"
                        f"mark={mark if mark else 'None'}"
                    )
                else:
                    print(
                        f"已添加片段：action={action}, object={obj if obj else 'None'}, "
                        f"start={start_name} (idx={s_idx+1}), end={end_name} (idx={e_idx+1}),  total frames={e_idx - s_idx}"
                        f"mark={mark if mark else 'None'}"
                    )

                start_idx = None
                end_idx = None

            # ============== 退出 ==============
            elif key in (ord('q'), ord('Q'), 27):
                print("退出标注。")
                break

            else:
                print(f"未知按键: {key} (忽略)")

    finally:
        cv2.destroyAllWindows()
        cache.close()

    # ============== 保存 CSV ==============
    if segments:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            if args.use_hand:
                fieldnames = ["No", "action", "object", "hand", "start_idx", "end_idx", "start", "end", "mark"]
            else:
                fieldnames = ["No", "action", "object", "start_idx", "end_idx", "start", "end", "mark"]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, seg in enumerate(segments, start=1):
                row = {
                    "No": i,
                    "start_idx": seg["start_idx"],
                    "end_idx": seg["end_idx"],
                    "action": seg["action"],
                    "object": seg["object"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "mark": seg["mark"],
                }
                if args.use_hand:
                    row["hand"] = seg.get("hand", "")
                writer.writerow(row)
        print(f"\n已保存 {len(segments)} 个片段到: {out_path}")
    else:
        print("没有任何片段被标注，不生成 CSV。")


if __name__ == "__main__":
    main()
