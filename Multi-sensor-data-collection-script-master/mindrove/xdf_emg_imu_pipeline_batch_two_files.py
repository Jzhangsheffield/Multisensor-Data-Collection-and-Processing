#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.signal import butter, sosfiltfilt, iirnotch, resample_poly

# 新增：统计 CSV 行数所需
import csv

import pyxdf

# 新增：绘图（无界面后端）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LEFT_NAME  = "MindRove_Desktop"  # 左手
RIGHT_NAME = "MindRove_Z16"      # 右手


# ============== 基础工具 ==============
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


# ============== 滤波器 ==============
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


# ============== 重采样 & 包络 ==============
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


# ============== 通道拆分（16ch布局） ==============
def split_modalities_from_16ch(time_series):
    """
    输入：N×16（或更多）数组，其中：
      ch1 = board_ts（系统时间）
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
        raise ValueError("该流通道数 < 15，不符合设备的 16 通道布局要求")
    board_ts = X[:, 0]
    emg = X[:, 1:9]
    acc = X[:, 9:12]
    gyr = X[:, 12:15]
    return board_ts, emg, acc, gyr


# ============== 处理流程 ==============
def process_emg(emg, fs, emg_band, emg_bp_order, notch_list, notch_q, target_fs, rectify, smooth_ms):
    emg = emg - np.mean(emg, axis=0, keepdims=True) # 去直流
    sos_bp = butter_bandpass_sos(fs, emg_band[0], emg_band[1], order=emg_bp_order)
    y = apply_sosfiltfilt(sos_bp, emg)
    if notch_list:
        for f0 in notch_list:
            sos_n = notch_sos(fs, f0, q=notch_q)
            y = apply_sosfiltfilt(sos_n, y)
    y, fs_out = resample_if_needed(y, fs, target_fs)
    env = rectify_and_smooth(y, fs_out, win_ms=smooth_ms) if rectify else None
    return y, env, fs_out

def process_imu(acc, gyr, fs, imu_lp, imu_lp_order, target_fs):
    acc = acc - np.mean(acc, axis=0, keepdims=True) # 去直流
    gyr = gyr - np.mean(gyr, axis=0, keepdims=True)
    sos_lp = butter_lowpass_sos(fs, cutoff=imu_lp, order=imu_lp_order)
    acc_f = apply_sosfiltfilt(sos_lp, acc)
    gyr_f = apply_sosfiltfilt(sos_lp, gyr)
    acc_f, fs_out = resample_if_needed(acc_f, fs, target_fs)
    gyr_f, _      = resample_if_needed(gyr_f, fs, target_fs)
    return acc_f, gyr_f, fs_out


# ============== 新增：绘图工具 ==============
def _make_time_axis(n_samples: int, fs_out: float):
    if fs_out is None or fs_out <= 0:
        return np.arange(n_samples)
    return np.arange(n_samples) / float(fs_out)

def save_emg_plot(y, env, fs_out, save_path: Path, max_channels_to_plot: int = 8):
    """保存 EMG（含包络）波形图"""
    y = np.asarray(y)
    n, c = y.shape[:2]
    ch_to_plot = min(c, max_channels_to_plot)
    t = _make_time_axis(n, fs_out)

    plt.figure(figsize=(12, 7))
    for i in range(ch_to_plot):
        ax = plt.subplot(ch_to_plot, 1, i + 1)
        ax.plot(t, y[:, i], linewidth=0.7, label=f"EMG ch{i}")
        if env is not None and env.shape[0] == n:
            ax.plot(t, env[:, i], linewidth=0.9, alpha=0.9, label=f"Envelope ch{i}")
        ax.set_ylabel("ampl.")
        if i == 0:
            ax.legend(loc="upper right")
        if i == ch_to_plot - 1:
            ax.set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_imu_plot(acc_f, gyr_f, fs_out, save_path: Path):
    """保存 IMU（acc+gyro）波形图"""
    acc_f = np.asarray(acc_f)
    gyr_f = np.asarray(gyr_f)
    n = acc_f.shape[0]
    t = _make_time_axis(n, fs_out)

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, acc_f[:, 0], linewidth=0.7, label="Ax")
    ax1.plot(t, acc_f[:, 1], linewidth=0.7, label="Ay")
    ax1.plot(t, acc_f[:, 2], linewidth=0.7, label="Az")
    ax1.set_ylabel("acc")
    ax1.legend(loc="upper right")

    m = min(gyr_f.shape[0], t.shape[0])
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(t[:m], gyr_f[:m, 0], linewidth=0.7, label="Gx")
    ax2.plot(t[:m], gyr_f[:m, 1], linewidth=0.7, label="Gy")
    ax2.plot(t[:m], gyr_f[:m, 2], linewidth=0.7, label="Gz")
    ax2.set_ylabel("gyro")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def required_image_paths(out_dir: Path, xdf_stem: str, sides, modality: str):
    """
    返回本次处理应存在的所有图片路径（基于实际存在的 sides 与所选 modality）
    同一只手的 EMG/IMU 图像放在各自子文件夹内：out_dir/left, out_dir/right
    """
    need_emg = modality in ("emg", "both")
    need_imu = modality in ("imu", "both")
    req = []
    for s in sides:
        side_dir = out_dir / s
        if need_emg:
            req.append(side_dir / f"{xdf_stem}_{s}_emg.png")
        if need_imu:
            req.append(side_dir / f"{xdf_stem}_{s}_imu.png")
    return req

def all_exist(paths):
    return all(p.exists() for p in paths)


# ============== 新增：统计与 summary.txt 写入 ==============
def count_csv_rows(csv_path: Path):
    """返回CSV的有效数据行数（去掉1行header）；若不存在返回None。"""
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)

def write_summary_txt(out_dir: Path, xdf_stem: str, summary_dict: dict):
    """
    summary_dict 结构示例：
    {
        "left": {
            "orig_n": 12345,
            "emg_csv_n": 12340,   # 或 None
            "imu_csv_n": 12340,   # 或 None
            "emg_csv_path": Path(...),
            "imu_csv_path": Path(...),
        },
        "right": { ... }
    }
    """
    lines = []
    lines.append(f"XDF file: {xdf_stem}\n")
    for side in ("left", "right"):
        if side not in summary_dict:
            continue
        rec = summary_dict[side]
        lines.append(f"[{side}]")
        lines.append(f"  Original samples in stream: {rec.get('orig_n', 'N/A')}")
        emg_n = rec.get("emg_csv_n", None)
        imu_n = rec.get("imu_csv_n", None)
        emg_path = rec.get("emg_csv_path", None)
        imu_path = rec.get("imu_csv_path", None)
        lines.append(
            f"  EMG CSV samples: {('N/A' if emg_n is None else emg_n)}"
            + (f"  ({emg_path.name})" if emg_path else "")
        )
        lines.append(
            f"  IMU CSV samples: {('N/A' if imu_n is None else imu_n)}"
            + (f"  ({imu_path.name})" if imu_path else "")
        )
        lines.append("")  # 空行分隔
    txt_path = out_dir / f"{xdf_stem}_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


# ============== 处理一个 XDF（双流） ==============
def process_one_xdf(xdf_path: Path, args):
    try:
        streams, _ = load_xdf(xdf_path)

        left_stream  = find_stream_by_name(streams, LEFT_NAME)
        right_stream = find_stream_by_name(streams, RIGHT_NAME)

        if (left_stream is None) and (right_stream is None):
            return (str(xdf_path), "ERROR: 未找到期望的两个流：MindRove_Desktop / MindRove_Z16")

        # 实际存在的侧别
        available_sides = []
        if left_stream is not None:
            available_sides.append("left")
        if right_stream is not None:
            available_sides.append("right")

        # 输出目录：与 XDF 同目录的同名文件夹
        out_dir = xdf_path.parent / xdf_path.stem

        # 统计摘要容器
        summary = {}

        # —— 跳过判定：如果 out_dir 已存在且所有应有图片均存在，则尝试仅写/补 summary.txt 后返回 ——
        req_imgs = required_image_paths(out_dir, xdf_path.stem, available_sides, args.modality)
        if out_dir.exists() and all_exist(req_imgs):
            for side, stream in (("left", left_stream), ("right", right_stream)):
                if stream is None:
                    continue
                orig_n = int(np.asarray(stream["time_series"]).shape[0])
                rec = {"orig_n": orig_n}

                side_dir = out_dir / side
                combined_csv = side_dir / f"{xdf_path.stem}_{side}_emg_imu.csv"
                if combined_csv.exists():
                    n_rows = count_csv_rows(combined_csv)
                    if args.modality in ("emg", "both"):
                        rec["emg_csv_path"] = combined_csv
                        rec["emg_csv_n"] = n_rows
                    else:
                        rec["emg_csv_path"] = None
                        rec["emg_csv_n"] = None

                    if args.modality in ("imu", "both"):
                        rec["imu_csv_path"] = combined_csv
                        rec["imu_csv_n"] = n_rows
                    else:
                        rec["imu_csv_path"] = None
                        rec["imu_csv_n"] = None
                else:
                    # 如果没有合并 CSV，可以按需兼容旧格式，这里简单标记为 None
                    rec["emg_csv_path"] = None
                    rec["emg_csv_n"] = None
                    rec["imu_csv_path"] = None
                    rec["imu_csv_n"] = None

                summary[side] = rec

            try:
                write_summary_txt(out_dir, xdf_path.stem, summary)
            except Exception as e:
                print(f"[WARN] {xdf_path.stem} 写入 summary 失败：{e}")

            return (str(xdf_path), "SKIP: 已存在所有所需的 EMG/IMU 图片，视为已处理。")

        # 需要处理则确保总目录存在
        out_dir.mkdir(parents=True, exist_ok=True)

        # 处理函数（复用一套逻辑）
        def handle_stream(stream, side_label):
            if stream is None:
                return
            fs = infer_fs(stream)
            if fs is None:
                raise RuntimeError(f"{side_label}: 无法推断采样率。")

            board_ts, emg, acc, gyr = split_modalities_from_16ch(stream["time_series"])
            orig_n = int(np.asarray(stream["time_series"]).shape[0])
            rec = {"orig_n": orig_n}

            # 每只手一个子目录
            side_dir = out_dir / side_label
            side_dir.mkdir(parents=True, exist_ok=True)

            # 先置空，方便后面统一合并
            y = None
            env = None
            fs_emg = None
            acc_f = None
            gyr_f = None
            fs_imu = None

            # ---------- EMG ----------
            if args.modality in ("emg", "both"):
                y, env, fs_emg = process_emg(
                    emg, fs,
                    emg_band=args.emg_band, emg_bp_order=args.emg_bp_order,
                    notch_list=args.notch, notch_q=args.notch_q,
                    target_fs=args.target_fs,
                    rectify=args.rectify, smooth_ms=args.smooth_ms
                )
                # EMG 波形图
                emg_png = side_dir / f"{xdf_path.stem}_{side_label}_emg.png"
                try:
                    save_emg_plot(y, env, fs_emg, emg_png)
                except Exception as e:
                    print(f"[WARN] {xdf_path.stem}/{side_label} EMG 绘图失败：{e}")

            # ---------- IMU ----------
            if args.modality in ("imu", "both"):
                acc_f, gyr_f, fs_imu = process_imu(
                    acc, gyr, fs,
                    imu_lp=args.imu_lp, imu_lp_order=args.imu_lp_order,
                    target_fs=args.target_fs
                )
                imu_png = side_dir / f"{xdf_path.stem}_{side_label}_imu.png"
                try:
                    save_imu_plot(acc_f, gyr_f, fs_imu, imu_png)
                except Exception as e:
                    print(f"[WARN] {xdf_path.stem}/{side_label} IMU 绘图失败：{e}")

            # ---------- 合并到一个 CSV：EMG + IMU ----------
            combined_csv_path = side_dir / f"{xdf_path.stem}_{side_label}_emg_imu.csv"
            cols = []
            head = ["board_ts"]

            lengths = []
            if args.modality in ("emg", "both") and y is not None:
                lengths.append(y.shape[0])
            if args.modality in ("imu", "both") and acc_f is not None:
                lengths.append(acc_f.shape[0])

            if lengths:
                n_comb = min(lengths) if len(lengths) > 0 else board_ts.shape[0]
                n_comb = min(n_comb, board_ts.shape[0])

                cols.append(board_ts[:n_comb])

                if args.modality in ("emg", "both") and y is not None:
                    cols.append(y[:n_comb, :])
                    head += [f"emg{i+1}" for i in range(y.shape[1])]

                if args.modality in ("imu", "both") and acc_f is not None and gyr_f is not None:
                    cols.append(acc_f[:n_comb, :])
                    cols.append(gyr_f[:n_comb, :])
                    head += [f"acc_{ax}" for ax in "xyz"] + [f"gyro_{ax}" for ax in "xyz"]

                data = np.column_stack(cols)
                np.savetxt(
                    combined_csv_path,
                    data,
                    delimiter=",",
                    header=",".join(head),
                    comments="",
                    fmt="%.7f"
                )

                if args.modality in ("emg", "both"):
                    rec["emg_csv_path"] = combined_csv_path
                    rec["emg_csv_n"] = n_comb
                else:
                    rec["emg_csv_path"] = None
                    rec["emg_csv_n"] = None

                if args.modality in ("imu", "both"):
                    rec["imu_csv_path"] = combined_csv_path
                    rec["imu_csv_n"] = n_comb
                else:
                    rec["imu_csv_path"] = None
                    rec["imu_csv_n"] = None
            else:
                rec["emg_csv_path"] = None
                rec["emg_csv_n"] = None
                rec["imu_csv_path"] = None
                rec["imu_csv_n"] = None

            summary[side_label] = rec

        # 左手（Desktop）、右手（Z16）
        handle_stream(left_stream,  "left")
        handle_stream(right_stream, "right")

        # 写 summary
        try:
            write_summary_txt(out_dir, xdf_path.stem, summary)
        except Exception as e:
            print(f"[WARN] {xdf_path.stem} 写入 summary 失败：{e}")

        return (str(xdf_path), "OK")
    except Exception as e:
        return (str(xdf_path), f"ERROR: {e}")


# ============== 主程序：批处理多个 XDF ==============
def main():
    pa = argparse.ArgumentParser(
        "Batch XDF (two MindRove streams) -> per-file folder with left/right EMG+IMU CSV + PNG + summary.txt"
    )
    pa.add_argument("--in-dir", type=Path, required=True, help="包含 XDF 的目录")
    pa.add_argument("--pattern", type=str, default="*.xdf", help="文件匹配模式")
    pa.add_argument("--recursive", action="store_true", help="递归子目录")

    # 公共
    pa.add_argument("--target-fs", type=float, default=None, help="重采样频率(Hz)")

    # EMG
    pa.add_argument("--modality", choices=["emg", "imu", "both"], default="both")
    pa.add_argument("--emg-band", nargs=2, type=float, default=[50.0, 450.0])
    pa.add_argument("--emg-bp-order", type=int, default=4)
    pa.add_argument("--notch", nargs="*", type=float, default=[50.0], help="如 --notch 50 100 150")
    pa.add_argument("--notch-q", type=float, default=30.0)
    pa.add_argument("--rectify", action="store_true")
    pa.add_argument("--smooth-ms", type=float, default=100.0)

    # IMU
    pa.add_argument("--imu-lp", type=float, default=20.0)
    pa.add_argument("--imu-lp-order", type=int, default=4)

    # 并行
    pa.add_argument("--jobs", type=int, default=max(1, cpu_count() // 2))

    args = pa.parse_args()

    if args.recursive:
        files = sorted([p for p in args.in_dir.rglob(args.pattern) if p.is_file()])
    else:
        files = sorted([p for p in args.in_dir.glob(args.pattern) if p.is_file()])

    if not files:
        print("未找到匹配的 XDF 文件。")
        return

    print(f"待处理: {len(files)} 个 XDF")
    work_items = [(f, args) for f in files]

    if args.jobs == 1:
        results = [process_one_xdf(*wi) for wi in work_items]
    else:
        with Pool(processes=args.jobs) as pool:
            results = pool.starmap(process_one_xdf, work_items)

    ok   = sum(1 for _, s in results if s == "OK")
    skip = sum(1 for _, s in results if isinstance(s, str) and s.startswith("SKIP"))
    fail = len(results) - ok - skip
    for fname, status in results:
        if status != "OK":
            print(f"[INFO] {fname}: {status}")
    print(f"完成：OK={ok}, SKIP={skip}, FAIL={fail}")


if __name__ == "__main__":
    main()
