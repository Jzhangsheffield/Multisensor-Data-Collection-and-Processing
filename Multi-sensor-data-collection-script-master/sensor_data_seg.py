import os
import shutil
import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import h5py

# -------------------- 全局常量 --------------------
FMT = "%Y%m%d_%H%M%S_%f"
DEFAULT_TZ = "Europe/London"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# -------------------- 通用时间/解析工具 --------------------
def normalize_input_time(
    x: Union[str, float, int, np.floating, datetime, pd.Timestamp],
    tz: str = DEFAULT_TZ,
    round_to_us: bool = False,
) -> pd.Timestamp:
    """
    将多种时间表示（UNIX 秒/字符串/FMT/ISO/datetime/pandas Timestamp）统一转为 tz-aware 的 Timestamp。
    - 若 round_to_us=True，则会四舍五入到微秒（保证与 FMT 一致的 6 位微秒）。
    """
    if isinstance(x, (float, int, np.floating)):
        ts = pd.to_datetime(float(x), unit="s", utc=True).tz_convert(tz)
        return ts.round("us") if round_to_us else ts

    if isinstance(x, pd.Timestamp):
        ts = x if x.tzinfo else x.tz_localize(tz)
        return ts.round("us") if round_to_us else ts

    if isinstance(x, datetime):
        ts = pd.Timestamp(x) if x.tzinfo else pd.Timestamp(x, tz=tz)
        ts = ts.tz_convert(tz) if ts.tzinfo else ts
        return ts.round("us") if round_to_us else ts

    if isinstance(x, str):
        # 优先按 FMT 解析；失败则退回通用解析
        try:
            ts = pd.to_datetime(x, format=FMT)
        except ValueError:
            ts = pd.to_datetime(x)
        ts = ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)
        return ts.round("us") if round_to_us else ts

    raise TypeError(f"Unsupported time type: {type(x)}")


def _ensure_sorted_tz_series(s: pd.Series) -> pd.Series:
    """确保是 tz-aware 且已排序的 Series[Timestamp]."""
    if not pd.api.types.is_datetime64tz_dtype(s):
        raise TypeError("Input series must be tz-aware datetime64[ns, tz].")
    return s.sort_values().reset_index(drop=True)


def _parse_emg_time_col(s: pd.Series, tz: str) -> pd.Series:
    """
    统一解析 EMG CSV 中的时间列：
    - 如果是数字类型：按 UNIX 秒解析；
    - 如果是字符串：优先按 FMT 解析，失败再用 pandas 自动推断。
    返回 tz-aware 且已排序的 Series[Timestamp]。
    """
    # 数值型：视为 UNIX 秒
    if np.issubdtype(s.dtype, np.number):
        ts = pd.to_datetime(s.astype(float), unit="s", utc=True).dt.tz_convert(tz)
        return ts.sort_values().reset_index(drop=True)

    # 其他情况：先当字符串处理
    s_str = s.astype(str)

    # 优先尝试 FMT
    try:
        ts = pd.to_datetime(s_str, format=FMT, errors="raise")
    except ValueError:
        # FMT 不行就让 pandas 自己猜
        ts = pd.to_datetime(s_str, errors="coerce")
        if ts.isna().any():
            bad = s_str[ts.isna()].head()
            raise ValueError(
                f"无法解析 EMG 时间列，示例无效值: {list(bad)}; "
                f"请检查是否为 UNIX 秒或 {FMT} 格式字符串。"
            )
    ts = ts.dt.tz_localize(tz)
    return ts.sort_values().reset_index(drop=True)


# -------------------- 载入各传感器时间戳 --------------------
def load_lidar_dir(lidar_dir: str, tz: str = DEFAULT_TZ) -> pd.DataFrame:
    """
    从 LiDAR 目录中文件名 (FMT+"_cloud.csv") 解析时间戳。
    返回 DataFrame{file, datetimes}，按时间排序。
    """
    p = Path(lidar_dir)
    lidar_files = sorted([f.name for f in p.iterdir() if f.is_file() and f.name.endswith("_cloud.csv")])
    stem_series = pd.Series([f.replace("_cloud.csv", "") for f in lidar_files])
    dt = pd.to_datetime(stem_series, format=FMT).dt.tz_localize(tz)
    df = pd.DataFrame({"file": lidar_files, "datetimes": dt})
    return df.sort_values("datetimes").reset_index(drop=True)


# def load_emg_csv(csv_path: str, time_col: str = "board_ts", tz: str = DEFAULT_TZ) -> pd.Series:
#     """
#     从 EMG CSV 的指定时间列（FMT 字符串）解析为 tz-aware Series。
#     返回 Series[Timestamp]（已排序）。
#     """
#     df = pd.read_csv(csv_path)
#     ts = pd.to_datetime(df[time_col], format=FMT, errors="raise").dt.tz_localize(tz)
#     return ts.sort_values().reset_index(drop=True)


def load_emg_csv(csv_path: str, time_col: str = "board_ts", tz: str = DEFAULT_TZ) -> pd.Series:
    """
    从 EMG CSV 的指定时间列解析为 tz-aware Series。
    - 支持数值（UNIX 秒）和字符串（FMT / 自动推断）。
    返回 Series[Timestamp]（已排序）。
    """
    df = pd.read_csv(csv_path)
    ts = _parse_emg_time_col(df[time_col], tz=tz)
    return ts


def load_kinect_from_folder(img_dir: str, tz: str = DEFAULT_TZ) -> pd.Series:
    """
    从 Kinect 图像目录中，按文件名（FMT）解析时间戳。
    返回 Series[Timestamp]（已排序）。
    """
    p = Path(img_dir)
    names = []
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in _IMG_EXTS:
            names.append(f.stem)
    if not names:
        return pd.Series([], dtype="datetime64[ns, UTC]").tz_convert(tz)  # 空
    dt = pd.to_datetime(pd.Series(names), format=FMT).dt.tz_localize(tz)
    return dt.sort_values().reset_index(drop=True)


def load_kinect_hdf5(h5_path: str, tz: str = DEFAULT_TZ) -> pd.Series:
    """
    （可选）从 Kinect HDF5 的 'timestamp'（UNIX 秒，float64/float128）解析为 tz-aware Series（四舍五入到微秒）。
    返回 Series[Timestamp]（已排序）。
    """
    with h5py.File(h5_path, "r") as f:
        timestamp = f["timestamp"][:]  # UNIX 秒
    ts = pd.to_datetime(pd.Series(timestamp), unit="s", utc=True).dt.tz_convert(tz).dt.round("us")
    return ts.sort_values().reset_index(drop=True)


# -------------------- 窗口边界计算 --------------------
def window_edges(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, Optional[pd.Timestamp]]:
    """
    对已排序 tz-aware Series，计算：
    - start_le: 第一个 <= start 的时间点（窗口内起点）
    - end_ge  : 最后一个 >= end   的时间点（窗口内终点）
    若窗口内不存在任何点，则 start_le/end_ge 为 None。
    """
    s = _ensure_sorted_tz_series(s)

    # 最接近 start 且 <= start
    i = s.searchsorted(start, side="right") - 1
    start_le = s.iloc[i] if i >= 0 and s.iloc[i] <= start else None

    # 最接近 end 且 >= end
    j = s.searchsorted(end, side="left")
    end_ge = s.iloc[j] if j < len(s) and s.iloc[j] >= end else None

    return {"start_le": start_le, "end_ge": end_ge}


def collect_window_edges(
    start: Union[str, float, int, np.floating, datetime, pd.Timestamp],
    end:   Union[str, float, int, np.floating, datetime, pd.Timestamp],
    tz: str = DEFAULT_TZ,
    **streams: Mapping[str, Union[pd.Series, pd.DataFrame]]
) -> Dict[str, Dict[str, Optional[pd.Timestamp]]]:
    """
    对传入的各传感器流（命名参数形式，如 lidar=..., emg_left=...）分别计算窗口边界。
    - DataFrame 自动取 'datetimes' 列。
    返回：{stream_name: {'start_le','end_ge','prev','next'}}。
    """
    start_ts = normalize_input_time(start, tz=tz)
    end_ts = normalize_input_time(end, tz=tz)
    if not (start_ts < end_ts):
        raise ValueError("start must be earlier than end.")

    out: Dict[str, Dict[str, Optional[pd.Timestamp]]] = {}
    for name, obj in streams.items():
        ser = obj["datetimes"] if isinstance(obj, pd.DataFrame) else obj
        out[name] = window_edges(ser, start_ts, end_ts)
    return out


# -------------------- 拷贝/切片动作 --------------------
def copy_lidar_window(lidar_dir: str, out_root: str, 
                      action: str, session: str, participant: str, run: str, count: int,
                      start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> int:
    """
    拷贝 LiDAR 窗口内（[start_ts, end_ts]）的帧到 out_root/lidar。
    返回拷贝文件数。
    """
    if start_ts is None or end_ts is None:
        return 0

    lidar_dir = Path(lidar_dir)
    folder = f'{session}_{participant}_{run}_{count}'
    out_dir = Path(out_root) / "lidar" / action / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_lidar_dir(str(lidar_dir), tz=str(start_ts.tz))
    mask = df["datetimes"].between(start_ts, end_ts, inclusive="both")
    files = df.loc[mask, "file"].tolist()

    for fname in files:
        shutil.copy2(lidar_dir / fname, out_dir / fname)
    return len(files)


def slice_emg_window(in_csv: str, out_root: str, time_col: str,
                     action: str, session: str, participant: str, run: str, count: int,
                     start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp],
                     out_name: str = "emg_window.csv") -> int:
    """
    切片 EMG CSV（时间列为 FMT），输出到 out_root/emg/out_name。
    返回写出的行数。
    """
    if start_ts is None or end_ts is None:
        return 0

    tz = str(start_ts.tz)
    df = pd.read_csv(in_csv)
    # ts = pd.to_datetime(df[time_col], format=FMT, errors="raise").dt.tz_localize(tz)
    ts = pd.to_datetime(df[time_col], format=FMT, errors="raise").dt.tz_localize(tz)
    mask = ts.between(start_ts, end_ts, inclusive="both")
    out_df = df.loc[mask].copy()

    folder = f'{session}_{participant}_{run}_{count}'
    out_path = Path(out_root) / "emg" / action / folder / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return len(out_df)


def copy_kinect_window(kinect_dir: str, out_root: str,
                       action: str, session: str, participant: str, run: str, angle: str, count: int,
                       start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> int:
    """
    拷贝 Kinect 图像窗口内（[start_ts, end_ts]）到 out_root/kinect。
    返回拷贝文件数。
    """
    if start_ts is None or end_ts is None:
        return 0

    kinect_dir = Path(kinect_dir)
    folder = f'{session}_{participant}_{run}_{count}'
    
    if 'depth' in str(kinect_dir):
        out_dir = Path(out_root) / "kinect_depth" / angle / action / folder
    else:
        out_dir = Path(out_root) / "kinect_rgb" / angle / action / folder
        
    out_dir.mkdir(parents=True, exist_ok=True)

    tz = str(start_ts.tz)
    n = 0
    for p in kinect_dir.iterdir():
        if not (p.is_file() and p.suffix.lower() in _IMG_EXTS):
            continue
        try:
            dt = pd.to_datetime(p.stem, format=FMT).tz_localize(tz)
        except Exception:
            continue
        if start_ts <= dt <= end_ts:
            shutil.copy2(p, out_dir / p.name)
            n += 1
    return n


# -------------------- 编排器：一把梭 --------------------
def extract_all(
    start: Union[str, float, int, np.floating, datetime, pd.Timestamp],
    end:   Union[str, float, int, np.floating, datetime, pd.Timestamp],
    out_root: str,
    action: str,
    session: str, 
    participant: str, 
    run: str, 
    count: int,
    lidar_dir: Optional[str] = None,
    emg_left_csv: Optional[str] = None,
    emg_right_csv: Optional[str] = None,
    emg_time_col: str = "board_ts",
    kinect_front_dir_rgb: Optional[str] = None,
    kinect_back_dir_rgb: Optional[str] = None,
    kinect_side_dir_rgb: Optional[str] = None,
    kinect_front_dir_depth: Optional[str] = None,
    kinect_back_dir_depth: Optional[str] = None,
    kinect_side_dir_depth: Optional[str] = None,
    tz: str = DEFAULT_TZ,
) -> Dict[str, Dict[str, Optional[pd.Timestamp]]]:
    """
    统一执行：读取 → 计算窗口边界 → 拷贝/切片。
    返回各流的窗口信息（便于日志/检查）。
    """
    streams: Dict[str, Union[pd.Series, pd.DataFrame]] = {}

    if lidar_dir:
        lidar_df = load_lidar_dir(lidar_dir, tz=tz)
        streams["lidar"] = lidar_df["datetimes"]
    if emg_left_csv:
        streams["emg_left"] = load_emg_csv(emg_left_csv, time_col=emg_time_col, tz=tz)
    if emg_right_csv:
        streams["emg_right"] = load_emg_csv(emg_right_csv, time_col=emg_time_col, tz=tz)
    if kinect_front_dir_rgb:
        streams["kinect_front_rgb"] = load_kinect_from_folder(kinect_front_dir_rgb, tz=tz)
    if kinect_back_dir_rgb:
        streams["kinect_back_rgb"] = load_kinect_from_folder(kinect_back_dir_rgb, tz=tz)
    if kinect_side_dir_rgb:
        streams["kinect_side_rgb"] = load_kinect_from_folder(kinect_side_dir_rgb, tz=tz)
    if kinect_front_dir_depth:
        streams["kinect_front_depth"] = load_kinect_from_folder(kinect_front_dir_depth, tz=tz)
    if kinect_back_dir_depth:
        streams["kinect_back_depth"] = load_kinect_from_folder(kinect_back_dir_depth, tz=tz)
    if kinect_side_dir_depth:
        streams["kinect_side_depth"] = load_kinect_from_folder(kinect_side_dir_depth, tz=tz)

    per_sensor = collect_window_edges(start, end, tz=tz, **streams)
    print(per_sensor)

    # 执行动作（窗口内）
    if lidar_dir:
        copy_lidar_window(
            lidar_dir=lidar_dir, out_root=out_root,
            action=action, session=session, participant=participant, run=run, count=count,
            start_ts=per_sensor["lidar"]["start_le"], end_ts=per_sensor["lidar"]["end_ge"]
        )

    if emg_left_csv:
        slice_emg_window(
            in_csv=emg_left_csv, out_root=out_root, time_col=emg_time_col,
            action=action, session=session, participant=participant, run=run, count=count,
            start_ts=per_sensor["emg_left"]["start_le"], end_ts=per_sensor["emg_left"]["end_ge"],
            out_name="emg_left.csv",
        )
    if emg_right_csv:
        slice_emg_window(
            in_csv=emg_right_csv, out_root=out_root, time_col=emg_time_col,
            action=action, session=session, participant=participant, run=run, count=count,
            start_ts=per_sensor["emg_right"]["start_le"], end_ts=per_sensor["emg_right"]["end_ge"],
            out_name="emg_right.csv",
        )

    if kinect_front_dir_rgb:
        copy_kinect_window(
            kinect_dir=kinect_front_dir_rgb, out_root=out_root,
            action=action, session=session, participant=participant, run=run, count=count,
            start_ts=per_sensor["kinect_front_rgb"]["start_le"], end_ts=per_sensor["kinect_front_rgb"]["end_ge"],
            angle = "front"
        )
    if kinect_back_dir_rgb:
        copy_kinect_window(
            kinect_dir=kinect_back_dir_rgb, out_root=out_root,
            action=action, session=session, participant=participant, run=run, count=count,
            start_ts=per_sensor["kinect_back_rgb"]["start_le"], end_ts=per_sensor["kinect_back_rgb"]["end_ge"],
            angle = "back"
        )
    if kinect_side_dir_rgb:
        copy_kinect_window(
            kinect_dir=kinect_side_dir_rgb, out_root=out_root,
            action=action, session=session, participant=participant, run=run, count=count,
            start_ts=per_sensor["kinect_side_rgb"]["start_le"], end_ts=per_sensor["kinect_side_rgb"]["end_ge"],
            angle = "side"
        )
        
    if kinect_front_dir_depth:
        copy_kinect_window(
            kinect_dir=kinect_front_dir_depth, out_root=out_root,
            action=action, session=session, participant=participant, run=run, count=count,
            start_ts=per_sensor["kinect_front_depth"]["start_le"], end_ts=per_sensor["kinect_front_depth"]["end_ge"],
            angle = "front"
        )
    if kinect_back_dir_depth:
        copy_kinect_window(
            kinect_dir=kinect_back_dir_depth, out_root=out_root,
            action=action, session=session, participant=participant, run=run, count=count,
            start_ts=per_sensor["kinect_back_depth"]["start_le"], end_ts=per_sensor["kinect_back_depth"]["end_ge"],
            angle = "back"
        )
    if kinect_side_dir_depth:
        copy_kinect_window(
            kinect_dir=kinect_side_dir_depth, out_root=out_root,
            action=action, session=session, participant=participant, run=run, count=count,
            start_ts=per_sensor["kinect_side_depth"]["start_le"], end_ts=per_sensor["kinect_side_depth"]["end_ge"],
            angle = "side"
        )

    return per_sensor


# -------------------- 示例 --------------------
if __name__ == "__main__":
    # 设定起止时间（可用 UNIX 秒 / FMT 字符串 / ISO 字符串）
    seg_df = pd.read_csv(r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\N\kinect\run_7\N_run_7_annotation.csv", skip_blank_lines=True) 
    
    # 路径示例（请替换为你的实际目录/文件；Kinect 这里示例用“图像目录”版本）
    # lidar_dir = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\vlp16\run_1\20251113_165615"
    # emg_L = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\mindrove\sub-J_ses-S001_task-DSamplePrepare_run-021_emg\left\sub-J_ses-S001_task-DSamplePrepare_run-021_emg_left_emg_imu.csv"
    # emg_R = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\mindrove\sub-J_ses-S001_task-DSamplePrepare_run-021_emg\right\sub-J_ses-S001_task-DSamplePrepare_run-021_emg_right_emg_imu.csv"
    k_front_rgb = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\N\kinect\run_7\001431512812\frames_rgb"
    # k_back_rgb  = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\kinect\run_1\001484412812\frames_rgb"
    # k_side_rgb  = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\kinect\run_1\001528512812\frames_rgb"
    # k_front_depth = r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_1\20250807_164249\001431512812\depth"
    # k_back_depth  = r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_1\20250807_164249\001484412812\depth"
    # k_side_depth  = r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_1\20250807_164249\001528512812\depth"

    out_root = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\test_structured_dataset"
    for i, row in seg_df.iterrows():
        start = row["start"]
        end = row["end"]
        action = row["action"]
        
        info = extract_all(
            start, end, out_root,
            action=f"{action}",
            session="None",
            participant="J",
            run="run1",
            count=i,
            # lidar_dir=lidar_dir,
            # emg_left_csv=emg_L, emg_right_csv=emg_R, emg_time_col="board_ts",
            kinect_front_dir_rgb=k_front_rgb, #kinect_back_dir_rgb=k_back_rgb, kinect_side_dir_rgb=k_side_rgb,
            # kinect_front_dir_depth=k_front_depth, kinect_back_dir_depth=k_back_depth, kinect_side_dir_depth=k_side_depth,
            tz=DEFAULT_TZ,
        )
        print(info)
