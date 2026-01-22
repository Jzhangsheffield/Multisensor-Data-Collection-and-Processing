# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import argparse
# from pathlib import Path
# from typing import Optional
# import shutil

# import pandas as pd

# # 根据你上传的 csv 和时间戳格式推断：
# # 例如 20251105_172554_530685 -> "%Y%m%d_%H%M%S_%f"
# FMT = "%Y%m%d_%H%M%S_%f"

# _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="根据 CSV 中的 start/end 区间，从 Kinect run_* 目录下的 frames_rgb 拷贝对应帧"
#     )
#     parser.add_argument(
#         "--csv_path", type=str, required=True,
#         help=r"注释 CSV 路径（包含 action, object, start, end, run 列）"
#     )
#     parser.add_argument(
#         "--run_root", type=str, required=True,
#         help=r"run_* 根目录，例如 D:\_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\N\kinect"
#     )
#     parser.add_argument(
#         "--angle_dir", type=str, required=True,
#         help=r"每个 run_* 下面的子目录名，例如 001431512812（该目录下有 rgb_frames）"
#     )
#     parser.add_argument(
#         "--out_root", type=str, required=True,
#         help=r"输出根目录，例如 D:\_data\MECCANO_clips"
#     )
#     parser.add_argument(
#         "--participant", type=str, required=True,
#         help=r"文件夹名中的 N，例如 1 或 001；只需运行时指定一次"
#     )
#     return parser.parse_args()


# def copy_frames_in_window(
#     kinect_dir: Path,
#     out_root: Path,
#     angle: str,
#     action: str,
#     obj: str,
#     run_folder_name: str,
#     participant: str,
#     start_str: str,
#     end_str: str,
# ) -> int:
#     """
#     在 kinect_dir/frames_rgb 中找到时间戳在 [start, end] 内的帧，
#     拷贝到 out_root/kinect_rgb/angle/action/(action_object_N_runFolder) 下。
#     返回拷贝文件数。
#     """

#     rgb_dir = kinect_dir / "frames_rgb"
#     if not rgb_dir.is_dir():
#         print(f"[警告] 找不到 frames_rgb 目录: {rgb_dir}")
#         return 0

#     try:
#         start_ts = pd.to_datetime(start_str, format=FMT)
#         end_ts = pd.to_datetime(end_str, format=FMT)
#     except Exception as e:
#         print(f"[警告] 时间解析失败: start={start_str}, end={end_str}, 错误: {e}")
#         return 0

#     # 目标文件夹名：action_object_N_run
#     folder_name = f"{action}_{obj}_{participant}_{run_folder_name}"
#     out_dir = out_root / "kinect_rgb" / angle / action / folder_name
#     out_dir.mkdir(parents=True, exist_ok=True)

#     n = 0
#     for p in sorted(rgb_dir.iterdir()):
#         if not (p.is_file() and p.suffix.lower() in _IMG_EXTS):
#             continue

#         stem = p.stem
#         try:
#             ts = pd.to_datetime(stem, format=FMT)
#         except Exception:
#             # 不是时间戳命名的文件就跳过
#             continue

#         if start_ts <= ts <= end_ts:
#             shutil.copy2(p, out_dir / p.name)
#             n += 1

#     return n


# def main():
#     args = parse_args()

#     csv_path = Path(args.csv_path)
#     run_root = Path(args.run_root)
#     angle_dir_name = args.angle_dir  # 例如 "001431512812"
#     out_root = Path(args.out_root)
#     particiant = str(args.participant)

#     if not csv_path.is_file():
#         print(f"[错误] 找不到 CSV: {csv_path}")
#         return
#     if not run_root.is_dir():
#         print(f"[错误] 找不到 run 根目录: {run_root}")
#         return

#     df = pd.read_csv(csv_path)

#     required_cols = {"action", "object", "start", "end", "run"}
#     if not required_cols.issubset(df.columns):
#         print(f"[错误] CSV 中缺少必要列: {required_cols - set(df.columns)}")
#         return

#     total_copied = 0

#     for idx, row in df.iterrows():
#         action = str(row["action"])
#         obj = str(row["object"])
#         start_str = str(row["start"])
#         end_str = str(row["end"])
#         run_val = str(row["run"])  # 例如 '1' 或 '8-18' 或 '24-37'

#         # run_* 文件夹名：前面加 'run_'
#         run_folder_name = f"run_{run_val}"
#         kinect_dir = run_root / run_folder_name / angle_dir_name

#         if not kinect_dir.is_dir():
#             print(f"[警告] 对应 run 目录不存在，跳过该行（index={idx}）: {kinect_dir}")
#             continue

#         copied = copy_frames_in_window(
#             kinect_dir=kinect_dir,
#             out_root=out_root,
#             angle=angle_dir_name,
#             action=action,
#             obj=obj,
#             run_folder_name=run_folder_name,
#             participant=particiant,
#             start_str=start_str,
#             end_str=end_str,
#         )
#         print(
#             f"[{idx}] action={action}, object={obj}, run={run_val} -> 拷贝 {copied} 张图像"
#         )
#         total_copied += copied

#     print(f"完成！总共拷贝 {total_copied} 张图像。")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional
import shutil

import pandas as pd

# 时间戳格式，例如：20251105_172554_530685 -> "%Y%m%d_%H%M%S_%f"
FMT = "%Y%m%d_%H%M%S_%f"

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 CSV 中的 start/end 区间，从 Kinect run_* 目录下的 frames_rgb 拷贝对应帧"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help=r"注释 CSV 路径（包含 action, object, start, end, run 列）"
    )
    parser.add_argument(
        "--run_root", type=str, required=True,
        help=r"run_* 根目录，例如 D:\_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\N\kinect"
    )
    parser.add_argument(
        "--angle_dir", type=str, required=True,
        help=r"每个 run_* 下面的子目录名，例如 001431512812（该目录下有 rgb_frames）"
    )
    parser.add_argument(
        "--out_root", type=str, required=True,
        help=r"输出根目录，例如 D:\_data\MECCANO_clips"
    )
    parser.add_argument(
        "--participant", type=str, required=True,
        help=r"文件夹名中的 N，例如 1 或 001；只需运行时指定一次"
    )
    return parser.parse_args()


def copy_frames_in_window(
    kinect_dir: Path,
    out_root: Path,
    angle: str,
    action: str,
    folder_name: str,
    start_str: str,
    end_str: str,
) -> int:
    """
    在 kinect_dir/frames_rgb 中找到时间戳在 [start, end] 内的帧，
    拷贝到 out_root/kinect_rgb/angle/action/folder_name 下。
    返回拷贝文件数。
    """

    rgb_dir = kinect_dir / "frames_rgb"
    if not rgb_dir.is_dir():
        print(f"[警告] 找不到 frames_rgb 目录: {rgb_dir}")
        return 0

    try:
        start_ts = pd.to_datetime(start_str, format=FMT)
        end_ts = pd.to_datetime(end_str, format=FMT)
    except Exception as e:
        print(f"[警告] 时间解析失败: start={start_str}, end={end_str}, 错误: {e}")
        return 0

    # 目标文件夹：out_root/kinect_rgb/angle/action/folder_name
    out_dir = out_root / "kinect_rgb" / angle / action / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for p in sorted(rgb_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in _IMG_EXTS):
            continue

        stem = p.stem
        try:
            ts = pd.to_datetime(stem, format=FMT)
        except Exception:
            # 不是时间戳命名的文件就跳过
            continue

        if start_ts <= ts <= end_ts:
            shutil.copy2(p, out_dir / p.name)
            n += 1

    return n


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    run_root = Path(args.run_root)
    angle_dir_name = args.angle_dir  # 例如 "001431512812"
    out_root = Path(args.out_root)
    participant = str(args.participant)

    if not csv_path.is_file():
        print(f"[错误] 找不到 CSV: {csv_path}")
        return
    if not run_root.is_dir():
        print(f"[错误] 找不到 run 根目录: {run_root}")
        return

    df = pd.read_csv(csv_path)

    required_cols = {"action", "object", "start", "end", "run"}
    if not required_cols.issubset(df.columns):
        print(f"[错误] CSV 中缺少必要列: {required_cols - set(df.columns)}")
        return

    total_copied = 0

    # 用于对同一 (angle, action, object, run) 组合计数
    # key: (angle, action, object, run_folder_name) -> 1, 2, 3, ...
    counters = {}

    for idx, row in df.iterrows():
        action = str(row["action"])
        obj = str(row["object"])
        start_str = str(row["start"])
        end_str = str(row["end"])
        run_val = str(row["run"])  # 例如 '3' 或 '8-18' 或 '24-37'

        # run_* 文件夹名：前面加 'run_'
        run_folder_name = f"run_{run_val}"
        kinect_dir = run_root / run_folder_name / angle_dir_name

        if not kinect_dir.is_dir():
            print(f"[警告] 对应 run 目录不存在，跳过该行（index={idx}）: {kinect_dir}")
            continue

        # 对同一 (angle, action, object, run_folder_name) 组合计数
        key = (angle_dir_name, action, obj, run_folder_name)
        counters[key] = counters.get(key, 0) + 1
        clip_idx = counters[key]

        # 文件夹名：action_object_N_run_xxx_001
        folder_name = f"{action}_{obj}_{participant}_{run_folder_name}_{clip_idx:03d}"

        copied = copy_frames_in_window(
            kinect_dir=kinect_dir,
            out_root=out_root,
            angle=angle_dir_name,
            action=action,
            folder_name=folder_name,
            start_str=start_str,
            end_str=end_str,
        )
        print(
            f"[{idx}] action={action}, object={obj}, run={run_val}, clip_idx={clip_idx:03d} "
            f"-> 拷贝 {copied} 张图像，保存到 {folder_name}"
        )
        total_copied += copied

    print(f"完成！总共拷贝 {total_copied} 张图像。")


if __name__ == "__main__":
    main()
