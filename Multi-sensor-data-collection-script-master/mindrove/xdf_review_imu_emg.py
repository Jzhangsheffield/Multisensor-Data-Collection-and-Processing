#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 需要可视化窗口
import matplotlib.pyplot as plt
import pyxdf
from scipy.signal import butter, sosfiltfilt, iirnotch, resample_poly

# ---- MindRove 两路流名称 ----
LEFT_NAME  = "MindRove_Desktop"  # 左手
RIGHT_NAME = "MindRove_Z16"      # 右手

# ---- 常量 ----
SIDES = ("left", "right")
MODS  = ("IMU", "EMG")
COLUMNS = ["time", "file", "L_IMU", "R_IMU", "L_EMG", "R_EMG"]

# ================= 基础工具 =================
def load_xdf(xdf_path: Path):
    streams, info = pyxdf.load_xdf(str(xdf_path),
                                   dejitter_timestamps=True,
                                   synchronize_clocks=True)
    return streams, info

def infer_fs(stream):
    for key in ("effective_srate", "nominal_srate"):
        try:
            fs = float(stream["info"][key][0])
            if fs > 0:
                return fs
        except Exception:
            pass
    ts = np.asarray(stream["time_stamps"])
    if len(ts) > 1:
        dt = np.median(np.diff(ts))
        if dt > 0:
            return 1.0 / dt
    return None

def find_stream_by_name(streams, name):
    for s in streams:
        if s["info"].get("name", [""])[0] == name:
            return s
    return None

# ================= 滤波器 =================
def butter_bandpass_sos(fs, low, high, order=4):
    nyq = 0.5 * fs
    lo = max(0.1, low) / nyq
    hi = min(high, 0.9 * nyq) / nyq
    if hi <= lo:
        hi = min(0.9, max(lo + 0.05, lo * 1.2))
    return butter(order, [lo, hi], btype="bandpass", output="sos")

def butter_lowpass_sos(fs, cutoff, order=4):
    nyq = 0.5 * fs
    wn = min(cutoff, 0.9 * nyq) / nyq
    return butter(order, wn, btype="lowpass", output="sos")

def notch_sos(fs, freq=50.0, q=30.0):
    if freq <= 0 or fs <= 0 or freq >= fs / 2:
        return None
    b, a = iirnotch(w0=freq / (fs / 2.0), Q=q)
    sos = np.zeros((1, 6), dtype=float)
    sos[0, :3] = b
    sos[0, 3:] = a
    return sos

def apply_sosfiltfilt(sos, x):
    if sos is None:
        return x
    return sosfiltfilt(sos, x, axis=0)

# ================= 重采样 & 包络 =================
def resample_if_needed(x, fs, target_fs=None):
    if (target_fs is None) or (abs(target_fs - fs) < 1e-9):
        return x, fs
    up = int(round(target_fs))
    down = int(round(fs))
    from math import gcd
    g = gcd(up, down)
    up //= g
    down //= g
    y = resample_poly(x, up, down, axis=0)
    return y, target_fs

def rectify_and_smooth(x, fs, win_ms=100.0):
    x = np.abs(x)
    win = int(max(1, round(win_ms / 1000.0 * fs)))
    kernel = np.ones(win) / win
    env = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), axis=0, arr=x)
    return env

# ================= 通道拆分（16ch） =================
def split_modalities_from_16ch(time_series):
    """
    输入：N×16（或更多）：
      ch1 = board_ts
      ch2-9 = 8×EMG
      ch10-12 = 3×Accel
      ch13-15 = 3×Gyro
      ch16 = 忽略
    返回: board_ts(N,), emg(N×8), acc(N×3), gyr(N×3)
    """
    X = np.asarray(time_series)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[1] < 15:
        raise ValueError("该流通道数 < 15，不符合 16 通道布局")
    board_ts = X[:, 0]
    emg = X[:, 1:9]
    acc = X[:, 9:12]
    gyr = X[:, 12:15]
    return board_ts, emg, acc, gyr

# ================= 处理流程 =================
def process_emg(emg, fs, emg_band, emg_bp_order, notch_list, notch_q, target_fs, rectify_flag, smooth_ms):
    emg = emg - np.mean(emg, axis=0, keepdims=True)
    sos_bp = butter_bandpass_sos(fs, emg_band[0], emg_band[1], order=emg_bp_order)
    y = apply_sosfiltfilt(sos_bp, emg)
    if notch_list:
        for f0 in notch_list:
            sos_n = notch_sos(fs, f0, q=notch_q)
            y = apply_sosfiltfilt(sos_n, y)
    y, fs_out = resample_if_needed(y, fs, target_fs)
    env = rectify_and_smooth(y, fs_out, win_ms=smooth_ms) if rectify_flag else None
    return y, env, fs_out

def process_imu(acc, gyr, fs, imu_lp, imu_lp_order, target_fs):
    acc = acc - np.mean(acc, axis=0, keepdims=True)
    gyr = gyr - np.mean(gyr, axis=0, keepdims=True)
    sos_lp = butter_lowpass_sos(fs, cutoff=imu_lp, order=imu_lp_order)
    acc_f = apply_sosfiltfilt(sos_lp, acc)
    gyr_f = apply_sosfiltfilt(sos_lp, gyr)
    acc_f, fs_out = resample_if_needed(acc_f, fs, target_fs)
    gyr_f, _      = resample_if_needed(gyr_f, fs, target_fs)
    return acc_f, gyr_f, fs_out

# ================= 绘图（全屏 & 分通道子图） =================
def make_time_axis(n_samples: int, fs_out: float):
    if fs_out is None or fs_out <= 0:
        return np.arange(n_samples)
    return np.arange(n_samples) / float(fs_out)

def fullscreen_current_fig():
    mng = plt.get_current_fig_manager()
    try:
        mng.full_screen_toggle()
    except Exception:
        try:
            mng.window.showMaximized()
        except Exception:
            pass

def plot_emg_channels(emg, env, fs, title):
    n_ch = emg.shape[1]
    fig, axes = plt.subplots(n_ch, 1, sharex=True, figsize=(15, max(8, n_ch*1.0)))
    if n_ch == 1:
        axes = [axes]
    t = make_time_axis(emg.shape[0], fs)
    for i, ax in enumerate(axes):
        ax.plot(t, emg[:, i], linewidth=0.8)
        if env is not None:
            ax.plot(t, env[:, i], linewidth=1.0, alpha=0.9)
        ax.set_ylabel(f"EMG{i+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fullscreen_current_fig()
    return fig

def plot_imu_channels(acc, gyr, fs, title):
    n_rows = acc.shape[1] + gyr.shape[1]  # 3 + 3
    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(15, max(8, n_rows*1.0)))
    if n_rows == 1:
        axes = [axes]
    t = make_time_axis(acc.shape[0], fs)
    labels_acc = ["Ax", "Ay", "Az"]
    for i in range(acc.shape[1]):
        axes[i].plot(t, acc[:, i], linewidth=0.9)
        axes[i].set_ylabel(labels_acc[i])
        axes[i].grid(True, alpha=0.3)
    t2 = make_time_axis(gyr.shape[0], fs)
    labels_gyr = ["Gx", "Gy", "Gz"]
    for i in range(gyr.shape[1]):
        axes[acc.shape[1]+i].plot(t2, gyr[:, i], linewidth=0.9)
        axes[acc.shape[1]+i].set_ylabel(labels_gyr[i])
        axes[acc.shape[1]+i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fullscreen_current_fig()
    return fig

# ================= 简洁日志：单行/文件 =================
def log_path(base_dir: Path):
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "mindrove_review_log.txt"

def parse_existing_log(base_dir: Path):
    """
    读取已有日志为 dict:
      key = 绝对路径字符串
      val = dict(fields={"time","file","L_IMU","R_IMU","L_EMG","R_EMG"})
    """
    path = log_path(base_dir)
    table = {}
    if not path.exists():
        return table
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        # 定宽/多空格 -> split
        parts = line.split()
        if len(parts) < 6:
            continue
        # 时间  文件名  L_IMU  R_IMU  L_EMG  R_EMG
        tm = " ".join(parts[0:2])
        fname = parts[2]
        cols = parts[3:7]
        if len(cols) != 4:
            continue
        entry = {
            "time": tm,
            "file": fname,
            "L_IMU": cols[0],
            "R_IMU": cols[1],
            "L_EMG": cols[2],
            "R_EMG": cols[3],
        }
        key = str((base_dir / fname).resolve())  # 用文件名在 in-dir 上级拼绝对路径（兼容之前行为）
        table[key] = entry
    return table

def write_log(base_dir: Path, table: dict):
    """重写日志文件，保证每个文件一行并对齐。"""
    path = log_path(base_dir)
    # 列宽
    W_TIME = 19  # 'YYYY-MM-DD HH:MM:SS' = 19
    W_FILE = max(35, max((len(v["file"]) for v in table.values()), default=35))
    def fmt(entry):
        return (f"{entry['time']:<{W_TIME}} {entry['file']:<{W_FILE}} "
                f"{entry['L_IMU']:<5} {entry['R_IMU']:<5} {entry['L_EMG']:<5} {entry['R_EMG']:<5}")
    header = (f"{'time':<{W_TIME}} {'file':<{W_FILE}} L_IMU R_IMU L_EMG R_EMG")
    sep = "-" * len(header)
    lines = ["# one line per file", header, sep]
    # 稳定顺序：按文件名排序
    for _, entry in sorted(table.items(), key=lambda kv: kv[1]["file"]):
        lines.append(fmt(entry))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[LOG] 写入：{path}")

def update_file_status(entry: dict, side: str, modality: str, value: str):
    key = f"{'L' if side=='left' else 'R'}_{modality.upper()}"
    entry[key] = value

# ================= 单模态审核（返回 PASS/ SKIP/ None-quit） =================
def review_modality(xdf_path: Path, stream, side: str, modality: str, args):
    fs = infer_fs(stream)
    if fs is None:
        print(f"[WARN] {xdf_path.name}/{side}: 无法推断采样率 -> 记为 SKIP")
        return "SKIP"

    board_ts, emg, acc, gyr = split_modalities_from_16ch(stream["time_series"])

    # 预处理
    emg_f, env, fs_emg = process_emg(
        emg, fs, args.emg_band, args.emg_bp_order,
        args.notch, args.notch_q, args.target_fs,
        args.rectify, args.smooth_ms
    )
    acc_f, gyr_f, fs_imu = process_imu(
        acc, gyr, fs, args.imu_lp, args.imu_lp_order, args.target_fs
    )

    if modality == "IMU":
        fig = plot_imu_channels(acc_f, gyr_f, fs_imu, f"{xdf_path.name} [{side}] IMU (fs={fs_imu:.1f}Hz)")
    else:
        fig = plot_emg_channels(emg_f, env, fs_emg, f"{xdf_path.name} [{side}] EMG (fs={fs_emg:.1f}Hz)")

    decision = {"val": "SKIP", "quit": False}
    def on_key(event):
        k = (event.key or "").lower()
        if k == 'p':     # 通过
            decision["val"] = "PASS"
            plt.close(fig)
        elif k == 'k':   # 跳过
            decision["val"] = "SKIP"
            plt.close(fig)
        elif k == 's':   # 保存
            out_dir = xdf_path.parent / xdf_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{xdf_path.stem}_{side}_{modality}.png", dpi=150)
        elif k == 'q':   # 退出所有
            decision["quit"] = True
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    print(f"Controls ({modality} {side}): [P]=Pass  [K]=Skip  [S]=Save  [Q]=Quit")
    plt.show(block=True)

    if decision["quit"]:
        return None
    return decision["val"]

# ================= 主流程：先 IMU 后 EMG，单行日志 =================
def main():
    pa = argparse.ArgumentParser("逐一人工审核 XDF：先左右手 IMU，再左右手 EMG；单行日志/文件")
    pa.add_argument("--in-dir", type=Path, required=True)
    pa.add_argument("--pattern", type=str, default="*.xdf")
    pa.add_argument("--recursive", action="store_true")

    # 重采样
    pa.add_argument("--target-fs", type=float, default=None)

    # EMG
    pa.add_argument("--emg-band", nargs=2, type=float, default=[20.0, 450.0])
    pa.add_argument("--emg-bp-order", type=int, default=4)
    pa.add_argument("--notch", nargs="*", type=float, default=50)
    pa.add_argument("--notch-q", type=float, default=30.0)
    pa.add_argument("--rectify", action="store_true")
    pa.add_argument("--smooth-ms", type=float, default=100.0)

    # IMU
    pa.add_argument("--imu-lp", type=float, default=20.0)
    pa.add_argument("--imu-lp-order", type=int, default=4)

    args = pa.parse_args()

    files = sorted([p for p in (args.in_dir.rglob(args.pattern) if args.recursive else args.in_dir.glob(args.pattern)) if p.is_file()])
    if not files:
        print("未找到匹配的 XDF 文件。")
        return

    base_dir = args.in_dir.parent
    log_table = parse_existing_log(base_dir)  # 读取已有（断点续跑）

    for xdf_path in files:
        # 构造/载入该文件的状态行
        key = str(xdf_path.resolve())
        if key in log_table:
            entry = log_table[key]
        else:
            entry = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": xdf_path.name,
                "L_IMU": "NA", "R_IMU": "NA", "L_EMG": "NA", "R_EMG": "NA"
            }
            log_table[key] = entry

        try:
            streams, _ = load_xdf(xdf_path)
        except Exception as e:
            print(f"[ERROR] {xdf_path.name}: {e}")
            # 标为 SKIP
            for col in ("L_IMU","R_IMU","L_EMG","R_EMG"):
                if entry[col] == "NA":
                    entry[col] = "SKIP"
            # 立刻刷新日志并继续
            write_log(base_dir, log_table)
            continue

        # 审核顺序：IMU left->right, 然后 EMG left->right
        order = [("left","IMU"), ("right","IMU"), ("left","EMG"), ("right","EMG")]
        name_map = {"left": LEFT_NAME, "right": RIGHT_NAME}

        for side, modality in order:
            col = f"{'L' if side=='left' else 'R'}_{modality}"
            # 已 PASS 则跳过
            if entry[col] == "PASS":
                continue

            stream = find_stream_by_name(streams, name_map[side])
            if stream is None:
                entry[col] = "NA"  # 该侧缺失
                continue

            print(f"\n[REVIEW] {xdf_path.name}  {side}-{modality}")
            res = review_modality(xdf_path, stream, side, modality, args)
            if res is None:  # 用户按 Q 退出
                # 写入当前进度
                entry["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_log(base_dir, log_table)
                print("[QUIT] 用户退出。")
                return
            update_file_status(entry, side, modality, res)

        # 一个文件结束：更新时间并写日志（保证一行/文件）
        entry["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_log(base_dir, log_table)

    print("\n全部完成。日志位于：", log_path(base_dir))

if __name__ == "__main__":
    main()
