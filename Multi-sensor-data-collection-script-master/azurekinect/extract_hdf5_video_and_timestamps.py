#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_extract_hdf5_video_and_timestamps.py

功能概述
--------
将已有的两个脚本功能合并成一个批处理脚本：
1. 从总根目录下，只筛选名字形如 stage_*_clean 的文件夹；
2. 在其下按如下结构查找数据：
       root/
         stage_xxx_clean/
           J/
             run_1/
               20260313_093609/
                 001431512812.hdf5
                 001484412812.hdf5
                 ...
3. 只扫描“时间戳文件夹”这一层中的 .hdf5 文件，不递归进入更深子文件夹；
4. 从每个 hdf5 中提取：
       - rgb 视频，或
       - depth 视频，或
       - 两者都提取
   并同时提取 timestamp 为 csv；
5. 重新组织输出目录为：
       output_root/
         J/
           run_1/
             001431512812/
               20260313_093609_rgb.mp4
               20260313_093609_timestamp.csv
           run_1_stage_2_March_13_clean/
             001431512812/
               ...
6. 如果同一个 participant 下，不同 stage 中出现同名 run，则自动改名为：
       run_x_stage_xxx_clean
   以避免覆盖；
7. 如果 hdf5 损坏、缺少数据集或写视频失败，自动跳过并记录到 log.txt；
8. 支持多进程并行处理多个 hdf5 文件，以加速运行。

依赖：
    pip install h5py opencv-python numpy pandas tqdm

说明：
- depth 会按固定范围做可视化后再写入 mp4。
- timestamp 会转换到 Europe/London 时区，并格式化为：
      YYYYMMDD_HHMMSS_microseconds
- 多进程并不一定总是越多越快。如果磁盘读写是瓶颈，建议把 --num_workers 调小。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------------------------
# 用于匹配目录名字的正则规则
# ------------------------------
STAGE_PATTERN = re.compile(r"^stage_.*_clean$")
RUN_PATTERN = re.compile(r"^run_\d+$")
TIMESTAMP_FOLDER_PATTERN = re.compile(r"^\d{8}_\d{6}$")


@dataclass
class H5Task:
    """表示一个待处理的 HDF5 文件及其上下文信息。"""

    stage_name: str
    participant: str
    run_name: str
    timestamp_folder: str
    h5_path: Path


class SimpleLogger:
    """一个简单的文本日志器，同时输出到终端和 log.txt。"""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, level: str, message: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{now}] [{level}] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def info(self, message: str) -> None:
        self.write("INFO", message)

    def warning(self, message: str) -> None:
        self.write("WARNING", message)

    def error(self, message: str) -> None:
        self.write("ERROR", message)


@dataclass
class WorkerResult:
    """
    单个 worker 的返回结果。

    说明：
    - stats 用于主进程汇总数量；
    - logs 用于把 worker 内产生的日志统一回传给主进程，避免多个进程同时写同一个 log.txt。
    """

    stats: dict[str, int]
    logs: list[tuple[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量从 HDF5 中提取 RGB/Depth 视频和时间戳 CSV，并按 participant/run/h5_id 重组输出目录。支持多进程。"
    )
    parser.add_argument(
        "--input_root",
        required=True,
        type=str,
        help="总根目录。其下包含多个 stage_*_clean 文件夹。",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        type=str,
        help="输出根目录。会在这里生成 participant/run/log.txt 等内容。",
    )
    parser.add_argument(
        "--modality",
        default="rgb",
        choices=["rgb", "depth", "both"],
        help="提取哪种视频：rgb / depth / both。默认 rgb。",
    )
    parser.add_argument(
        "--fps",
        default=30,
        type=int,
        help="输出 mp4 的帧率。默认 30。",
    )
    parser.add_argument(
        "--depth_min",
        default=0,
        type=float,
        help="depth 可视化最小深度值。默认 0。",
    )
    parser.add_argument(
        "--depth_max",
        default=3000,
        type=float,
        help="depth 可视化最大深度值。默认 3000。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出文件已存在，则覆盖重写；默认不覆盖，直接跳过该输出。",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help=(
            "并行进程数。默认 0，表示自动选择。"
            "若设为 1，则退化为单进程串行执行。"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"输入根目录不存在：{input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"输入根目录不是文件夹：{input_root}")

    if args.depth_max <= args.depth_min:
        raise ValueError("--depth_max 必须大于 --depth_min")

    if args.fps <= 0:
        raise ValueError("--fps 必须为正数")

    if args.num_workers < 0:
        raise ValueError("--num_workers 不能为负数")


def resolve_num_workers(requested: int) -> int:
    """
    解析实际使用的进程数。

    经验上：
    - 太多进程可能导致磁盘随机读写竞争，速度反而下降；
    - 因此自动模式不直接取全部 CPU 核心，而是做一个保守上限。
    """
    cpu_count = os.cpu_count() or 1
    if requested == 0:
        return max(1, min(8, cpu_count))
    return max(1, requested)


def list_stage_dirs(input_root: Path) -> List[Path]:
    """只保留名字形如 stage_*_clean 的一级子文件夹。"""
    stage_dirs: List[Path] = []
    for child in sorted(input_root.iterdir()):
        if child.is_dir() and STAGE_PATTERN.match(child.name):
            stage_dirs.append(child)
    return stage_dirs


def scan_tasks(stage_dirs: Iterable[Path], logger: SimpleLogger) -> List[H5Task]:
    """
    扫描所有待处理的 HDF5 文件。

    只识别以下层级：
        stage_dir / participant / run_x / timestamp_folder / *.hdf5

    注意：
    - 只扫描 timestamp_folder 这一层中的 .hdf5 文件；
    - 不递归进入 timestamp_folder 下的更深层目录。
    """
    tasks: List[H5Task] = []

    for stage_dir in stage_dirs:
        stage_name = stage_dir.name

        # 遍历 participant 文件夹，例如 J / M / MR / N
        for participant_dir in sorted(stage_dir.iterdir()):
            if not participant_dir.is_dir():
                continue
            participant = participant_dir.name

            # 遍历 run_x 文件夹
            for run_dir in sorted(participant_dir.iterdir()):
                if not run_dir.is_dir() or not RUN_PATTERN.match(run_dir.name):
                    continue
                run_name = run_dir.name

                # 遍历 run 下的“时间戳文件夹”
                for ts_dir in sorted(run_dir.iterdir()):
                    if not ts_dir.is_dir() or not TIMESTAMP_FOLDER_PATTERN.match(ts_dir.name):
                        continue
                    timestamp_folder = ts_dir.name

                    # 只在时间戳文件夹这一层找 .hdf5，不递归往下走
                    h5_files = sorted(p for p in ts_dir.iterdir() if p.is_file() and p.suffix.lower() == ".hdf5")

                    if not h5_files:
                        logger.warning(f"时间戳文件夹中未找到 hdf5：{ts_dir}")
                        continue

                    for h5_path in h5_files:
                        tasks.append(
                            H5Task(
                                stage_name=stage_name,
                                participant=participant,
                                run_name=run_name,
                                timestamp_folder=timestamp_folder,
                                h5_path=h5_path,
                            )
                        )

    return tasks


def build_run_name_map(tasks: List[H5Task]) -> dict[tuple[str, str, str], str]:
    """
    预先判断同一个 participant 下的 run 名是否跨 stage 重复。

    返回：
        key = (participant, stage_name, run_name)
        value = 输出时使用的 run 目录名

    规则：
    - 若某 participant 的某个 run_name 只来自一个 stage，则输出目录仍用 run_name；
    - 若该 run_name 来自多个 stage，则全部改为：run_name_stage_name
      这样可以保证命名稳定，不会出现“第一个不带后缀、第二个带后缀”的不一致问题。
    """
    participant_run_to_stages: dict[tuple[str, str], set[str]] = defaultdict(set)

    for task in tasks:
        participant_run_to_stages[(task.participant, task.run_name)].add(task.stage_name)

    run_name_map: dict[tuple[str, str, str], str] = {}
    for task in tasks:
        stages = participant_run_to_stages[(task.participant, task.run_name)]
        if len(stages) == 1:
            out_run_name = task.run_name
        else:
            out_run_name = f"{task.run_name}_{task.stage_name}"
        run_name_map[(task.participant, task.stage_name, task.run_name)] = out_run_name

    return run_name_map


def format_timestamps_for_csv(timestamp_array: np.ndarray) -> List[str]:
    """将 unix timestamp 秒数数组转为 Europe/London 时区字符串。"""
    ts = (
        pd.to_datetime(timestamp_array, unit="s", utc=True)
        .tz_convert("Europe/London")
        .round("us")
    )
    return ts.strftime("%Y%m%d_%H%M%S_%f").tolist()


def prepare_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """
    为 RGB 视频写出做最基本的形状整理。

    注意：
    - 为保持和你旧代码的行为一致，这里默认不做 RGB->BGR 转换；
    - 如果源数据本身已经能正常写出，则这里会保持兼容；
    - 如果将来发现颜色通道相反，可在这里自行加 cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)。
    """
    frame = np.asarray(frame)

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        pass
    else:
        raise ValueError(f"不支持的 RGB 帧形状：{frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frame)


def prepare_depth_frame(frame: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    """
    将深度帧可视化成彩色图后用于写视频。

    处理方式基本沿用你旧代码中的逻辑：
    - 先按固定深度范围裁剪
    - 再归一化到 0~255
    - 再用 JET colormap 可视化
    - 深度为 0 的无效区域置黑
    """
    frame = np.asarray(frame)

    # 有些 depth 可能已经是三通道图像；若如此，直接尽量转成 uint8 使用。
    if frame.ndim == 3 and frame.shape[2] == 3:
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    if frame.ndim != 2:
        raise ValueError(f"不支持的 depth 帧形状：{frame.shape}")

    valid_mask = frame > 0
    frame_clipped = np.clip(frame, depth_min, depth_max)
    normalized = ((frame_clipped - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colorized[~valid_mask] = [0, 0, 0]
    return np.ascontiguousarray(colorized)


def write_video_from_dataset(
    ds: h5py.Dataset,
    modality: str,
    out_path: Path,
    fps: int,
    depth_min: float,
    depth_max: float,
) -> int:
    """
    将 hdf5 dataset 写成 mp4。

    返回值：
        实际写入的视频帧数。
    """
    if len(ds) == 0:
        raise ValueError(f"dataset {modality} 为空，无法写视频")

    first_frame_raw = ds[0]
    if modality == "rgb":
        first_frame = prepare_rgb_frame(first_frame_raw)
    elif modality == "depth":
        first_frame = prepare_depth_frame(first_frame_raw, depth_min=depth_min, depth_max=depth_max)
    else:
        raise ValueError(f"不支持的 modality：{modality}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height), True)

    if not writer.isOpened():
        raise RuntimeError(f"无法打开 VideoWriter：{out_path}")

    frame_count = 0
    try:
        writer.write(first_frame)
        frame_count += 1

        for i in range(1, len(ds)):
            frame_raw = ds[i]
            if modality == "rgb":
                frame = prepare_rgb_frame(frame_raw)
            else:
                frame = prepare_depth_frame(frame_raw, depth_min=depth_min, depth_max=depth_max)

            # 如果中间帧尺寸不一致，opencv 写视频会失败，这里提前报错更清楚。
            if frame.shape[0] != height or frame.shape[1] != width:
                raise ValueError(
                    f"{modality} 第 {i} 帧尺寸与首帧不一致："
                    f"首帧=({height}, {width}), 当前帧=({frame.shape[0]}, {frame.shape[1]})"
                )

            writer.write(frame)
            frame_count += 1
    finally:
        writer.release()

    return frame_count


def maybe_skip_output(path: Path, overwrite: bool) -> bool:
    """判断是否应跳过已有输出文件。返回 True 表示跳过。"""
    if path.exists() and not overwrite:
        return True
    return False


def _append_log(logs: list[tuple[str, str]], level: str, message: str) -> None:
    """供 worker 使用：先把日志缓存在内存，最后统一回传给主进程写文件。"""
    logs.append((level, message))


def process_one_h5_worker(
    task: H5Task,
    out_run_dir_str: str,
    modality_mode: str,
    fps: int,
    depth_min: float,
    depth_max: float,
    overwrite: bool,
) -> WorkerResult:
    """
    处理单个 HDF5 的 worker 函数。

    设计说明：
    - 该函数必须是顶层函数，便于 Windows 下被多进程 pickle；
    - 该函数不直接写 log.txt，而是把日志回传给主进程，避免并发写日志文件时互相打断。
    """
    stats = {
        "csv_saved": 0,
        "video_saved": 0,
        "video_skipped": 0,
        "warning": 0,
        "failed": 0,
    }
    logs: list[tuple[str, str]] = []

    out_run_dir = Path(out_run_dir_str)
    base_id = task.h5_path.stem

    # 在目标 run 文件夹下，先按每个 hdf5 的文件名（不含后缀）建立子文件夹。
    # 例如：001431512812.hdf5 -> out_run_dir / "001431512812"
    out_h5_dir = out_run_dir / base_id
    file_prefix = task.timestamp_folder
    out_h5_dir.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(task.h5_path, "r") as f:
            # ------------------------------
            # 1) 先提取 timestamp -> csv
            # ------------------------------
            if "timestamp" not in f:
                raise KeyError("缺少 'timestamp' 数据集")

            timestamp_array = f["timestamp"][:]
            timestamp_formatted = format_timestamps_for_csv(timestamp_array)
            csv_path = out_h5_dir / f"{file_prefix}_timestamp.csv"

            if maybe_skip_output(csv_path, overwrite=overwrite):
                _append_log(logs, "INFO", f"CSV 已存在，跳过：{csv_path}")
            else:
                df = pd.DataFrame({"timestamp": timestamp_formatted})
                df.to_csv(csv_path, index=False)
                stats["csv_saved"] += 1
                _append_log(logs, "INFO", f"时间戳 CSV 已保存：{csv_path}")

            # ------------------------------
            # 2) 再提取视频
            # ------------------------------
            if modality_mode == "both":
                modalities = ["rgb", "depth"]
            else:
                modalities = [modality_mode]

            for modality in modalities:
                if modality not in f:
                    _append_log(logs, "WARNING", f"缺少 '{modality}' 数据集，跳过该视频：{task.h5_path}")
                    stats["warning"] += 1
                    continue

                out_video_path = out_h5_dir / f"{file_prefix}_{modality}.mp4"
                if maybe_skip_output(out_video_path, overwrite=overwrite):
                    _append_log(logs, "INFO", f"视频已存在，跳过：{out_video_path}")
                    stats["video_skipped"] += 1
                    continue

                frame_count = write_video_from_dataset(
                    ds=f[modality],
                    modality=modality,
                    out_path=out_video_path,
                    fps=fps,
                    depth_min=depth_min,
                    depth_max=depth_max,
                )
                stats["video_saved"] += 1
                _append_log(logs, "INFO", f"视频已保存：{out_video_path} | 帧数={frame_count}")

                # 可选一致性提示：timestamp 数量和当前视频帧数不一致时给出提醒
                if len(timestamp_array) != frame_count:
                    _append_log(
                        logs,
                        "WARNING",
                        f"timestamp 数量与 {modality} 帧数不一致："
                        f"h5={task.h5_path} | timestamp={len(timestamp_array)} | {modality}_frames={frame_count}",
                    )
                    stats["warning"] += 1

    except Exception as e:
        stats["failed"] += 1
        err_text = (
            f"处理失败，已跳过：{task.h5_path}\n"
            f"stage={task.stage_name}, participant={task.participant}, run={task.run_name}, timestamp_folder={task.timestamp_folder}\n"
            f"错误类型：{type(e).__name__}\n"
            f"错误信息：{e}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        _append_log(logs, "ERROR", err_text)
        # 失败时不再抛出，直接跳过，让整个批处理继续执行

    return WorkerResult(stats=stats, logs=logs)


def process_tasks_serial(
    tasks: list[H5Task],
    run_name_map: dict[tuple[str, str, str], str],
    output_root: Path,
    args: argparse.Namespace,
    logger: SimpleLogger,
) -> dict[str, int]:
    """单进程串行执行，便于调试，也兼容不希望开启多进程的情况。"""
    totals = {
        "csv_saved": 0,
        "video_saved": 0,
        "video_skipped": 0,
        "warning": 0,
        "failed": 0,
    }

    for task in tqdm(tasks, desc="Processing HDF5 files (serial)"):
        out_run_name = run_name_map[(task.participant, task.stage_name, task.run_name)]
        out_run_dir = output_root / task.participant / out_run_name

        result = process_one_h5_worker(
            task=task,
            out_run_dir_str=str(out_run_dir),
            modality_mode=args.modality,
            fps=args.fps,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            overwrite=args.overwrite,
        )

        for level, message in result.logs:
            if level == "INFO":
                logger.info(message)
            elif level == "WARNING":
                logger.warning(message)
            else:
                logger.error(message)

        for key in totals:
            totals[key] += result.stats.get(key, 0)

    return totals


def process_tasks_parallel(
    tasks: list[H5Task],
    run_name_map: dict[tuple[str, str, str], str],
    output_root: Path,
    args: argparse.Namespace,
    logger: SimpleLogger,
    num_workers: int,
) -> dict[str, int]:
    """
    多进程并行执行。

    并行粒度：
    - 每个 hdf5 文件对应一个独立任务；
    - 这样改动最小，且不会在同一个 hdf5 内再做更细的并行，逻辑更稳定。
    """
    totals = {
        "csv_saved": 0,
        "video_saved": 0,
        "video_skipped": 0,
        "warning": 0,
        "failed": 0,
    }

    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for task in tasks:
            out_run_name = run_name_map[(task.participant, task.stage_name, task.run_name)]
            out_run_dir = output_root / task.participant / out_run_name

            futures.append(
                executor.submit(
                    process_one_h5_worker,
                    task,
                    str(out_run_dir),
                    args.modality,
                    args.fps,
                    args.depth_min,
                    args.depth_max,
                    args.overwrite,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing HDF5 files ({num_workers} workers)"):
            try:
                result = future.result()
            except Exception as e:
                # 理论上 worker 内已经吃掉了大部分异常；
                # 如果这里还能抛出，通常是进程序列化、系统资源或子进程崩溃问题。
                logger.error(
                    "并行 worker 发生未捕获异常：\n"
                    f"错误类型：{type(e).__name__}\n"
                    f"错误信息：{e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                totals["failed"] += 1
                continue

            for level, message in result.logs:
                if level == "INFO":
                    logger.info(message)
                elif level == "WARNING":
                    logger.warning(message)
                else:
                    logger.error(message)

            for key in totals:
                totals[key] += result.stats.get(key, 0)

    return totals


def main() -> int:
    args = parse_args()
    validate_args(args)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    logger = SimpleLogger(output_root / "log.txt")
    logger.info("=" * 80)
    logger.info("开始执行批量提取任务")
    logger.info(f"input_root  = {input_root}")
    logger.info(f"output_root = {output_root}")
    logger.info(f"modality    = {args.modality}")
    logger.info(f"fps         = {args.fps}")
    logger.info(f"depth range = [{args.depth_min}, {args.depth_max}]")
    logger.info(f"overwrite   = {args.overwrite}")

    num_workers = resolve_num_workers(args.num_workers)
    logger.info(f"num_workers = {num_workers} (requested={args.num_workers})")

    # 1) 先找到所有 stage_*_clean 文件夹
    stage_dirs = list_stage_dirs(input_root)
    if not stage_dirs:
        logger.warning("未找到任何 stage_*_clean 文件夹，程序结束。")
        return 0

    logger.info(f"找到 stage 文件夹数量：{len(stage_dirs)}")
    for stage_dir in stage_dirs:
        logger.info(f"  stage: {stage_dir}")

    # 2) 扫描所有 HDF5 任务
    tasks = scan_tasks(stage_dirs, logger)
    logger.info(f"总共找到待处理 hdf5 数量：{len(tasks)}")
    if not tasks:
        logger.warning("未找到任何符合规则的 hdf5 文件，程序结束。")
        return 0

    # 3) 预先判断 run 名冲突，确保输出命名稳定
    run_name_map = build_run_name_map(tasks)

    # 4) 正式处理
    if num_workers == 1:
        totals = process_tasks_serial(
            tasks=tasks,
            run_name_map=run_name_map,
            output_root=output_root,
            args=args,
            logger=logger,
        )
    else:
        totals = process_tasks_parallel(
            tasks=tasks,
            run_name_map=run_name_map,
            output_root=output_root,
            args=args,
            logger=logger,
            num_workers=num_workers,
        )

    logger.info("=" * 80)
    logger.info("任务完成")
    logger.info(f"CSV 保存数量      ：{totals['csv_saved']}")
    logger.info(f"视频保存数量      ：{totals['video_saved']}")
    logger.info(f"视频跳过数量      ：{totals['video_skipped']}")
    logger.info(f"警告数量          ：{totals['warning']}")
    logger.info(f"失败数量          ：{totals['failed']}")
    logger.info(f"日志文件          ：{output_root / 'log.txt'}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断执行。")
        raise SystemExit(130)
