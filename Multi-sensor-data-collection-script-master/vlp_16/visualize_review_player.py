#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt


# ===========================
# 配置（按需修改）
# ===========================
DEFAULT_FPS = 180
DEFAULT_POINT_SIZE = 0.35
WINDOW_W, WINDOW_H = 1280, 900
# 初始姿态（按你之前习惯）
INIT_YAW, INIT_PITCH, INIT_ROLL = -90, -60, -5
# 颜色：'distance' 或 'intensity'
INIT_COLOR_BY = "distance"


# ===========================
# 工具 & 颜色映射
# ===========================
_TURBO_LUT = (plt.get_cmap('turbo', 256)(np.linspace(0, 1, 256))[:, :3]).astype(np.float32)

def colors_from_values(values: np.ndarray, p1: float, p99: float) -> np.ndarray:
    norm01 = (values - p1) / max(p99 - p1, 1e-6)
    idx = (np.clip(norm01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _TURBO_LUT[idx]

def compute_global_stats(csv_files, use_intensity, max_frames=30, max_pts_per_frame=5000):
    """采样若干帧估计稳健分位数，避免每帧都做percentile。"""
    samples = []
    for f in csv_files[:min(len(csv_files), max_frames)]:
        usecols = ['x','y','z'] + (['intensity'] if use_intensity else [])
        try:
            df = pd.read_csv(f, usecols=usecols, dtype={c:'float32' for c in usecols}, engine='c', memory_map=True)
        except Exception:
            df = pd.read_csv(f, usecols=usecols)
        if use_intensity and 'intensity' in df.columns:
            arr = df['intensity'].to_numpy()
        else:
            pts = df[['x','y','z']].to_numpy(dtype=np.float32, copy=False)
            arr = np.linalg.norm(pts, axis=1)
        if arr.size:
            step = max(1, arr.size // max_pts_per_frame)
            samples.append(arr[::step])
    if not samples:
        return 0.0, 1.0
    cat = np.concatenate(samples)
    p1, p99 = np.percentile(cat, [1, 99])
    if p99 <= p1:
        p1, p99 = float(cat.min()), float(cat.max()) + 1e-6
    return float(p1), float(p99)

def rotate_point_cloud(points: np.ndarray, yaw=0, pitch=0, roll=0) -> np.ndarray:
    yaw = np.radians(yaw).astype(np.float32)
    pitch = np.radians(pitch).astype(np.float32)
    roll = np.radians(roll).astype(np.float32)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R_yaw = np.array([[cy, -sy, 0],[sy, cy, 0],[0,0,1]], dtype=np.float32)
    R_pitch = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float32)
    R_roll  = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float32)
    R = R_yaw @ R_pitch @ R_roll
    return (points @ R.T).astype(np.float32)


# ===========================
# 日志
# ===========================
def load_pass_set(log_path: Path):
    passed = set()
    if log_path.exists():
        for line in log_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            line = line.strip()
            if not line:
                continue
            # 约定每行：<datetime>\t<MR>\t<run>\t<timestamp_dir>\t<total_frames>
            parts = line.split('\t')
            if len(parts) >= 4:
                key = parts[3]  # timestamp 目录完整路径
                passed.add(key)
    return passed

def append_pass_log(log_path: Path, mr: str, run: str, ts_dir: Path, total_frames: int):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{mr}\t{run}\t{str(ts_dir)}\t{total_frames}\n")


# ===========================
# 播放控制 & 回调
# ===========================
class PlayerState:
    def __init__(self, fps=DEFAULT_FPS, color_by=INIT_COLOR_BY):
        self.paused = False
        self.want_next = False
        self.want_prev = False
        self.want_quit = False
        self.want_pass = False
        self.want_skip = False
        self.want_screenshot = False
        self.point_size_delta = 0.0
        self.fps = fps
        self.color_by = color_by

def register_callbacks(vis: o3d.visualization.VisualizerWithKeyCallback, state: PlayerState, screenshot_dir: Path):
    def toggle_pause(_):
        state.paused = not state.paused
        print(f"[PAUSE] {state.paused}")
        return False

    def next_frame(_):
        state.want_next = True
        return False

    def prev_frame(_):
        state.want_prev = True
        return False

    def key_pass(_):
        state.want_pass = True
        return False

    def key_skip(_):
        state.want_skip = True
        return False

    def key_quit(_):
        state.want_quit = True
        return False

    def key_psize_inc(_):
        state.point_size_delta += +0.1
        return False

    def key_psize_dec(_):
        state.point_size_delta += -0.1
        return False

    def key_color_toggle(_):
        state.color_by = "intensity" if state.color_by == "distance" else "distance"
        print(f"[COLOR] {state.color_by}")
        return False

    def key_screenshot(_):
        state.want_screenshot = True
        return False

    vis.register_key_callback(ord(' '), toggle_pause)   # Space 播放/暂停
    vis.register_key_callback(262, next_frame)          # → 下一帧
    vis.register_key_callback(263, prev_frame)          # ← 上一帧
    vis.register_key_callback(ord('P'), key_pass)       # P 通过
    vis.register_key_callback(ord('K'), key_skip)       # K 跳过（不记通过）
    vis.register_key_callback(ord('Q'), key_quit)       # Q 退出整个程序
    vis.register_key_callback(ord(']'), key_psize_inc)  # ] 点大小+
    vis.register_key_callback(ord('['), key_psize_dec)  # [ 点大小-
    vis.register_key_callback(ord('C'), key_color_toggle) # C 切换着色
    vis.register_key_callback(ord('S'), key_screenshot) # S 截图保存到该目录


# ===========================
# 可视化一个时间戳目录（人工审核）
# ===========================
def review_one_sequence(ts_dir: Path,
                        yaw=INIT_YAW, pitch=INIT_PITCH, roll=INIT_ROLL,
                        init_fps=DEFAULT_FPS, init_point_size=DEFAULT_POINT_SIZE,
                        init_color_by=INIT_COLOR_BY):
    csv_files = sorted(glob.glob(str(ts_dir / "*.csv")))
    if not csv_files:
        print(f"[Skip] No CSV files in: {ts_dir}")
        return False, 0  # 未播放

    # 使用带回调的可视化器
    VisClass = getattr(o3d.visualization, "VisualizerWithKeyCallback", o3d.visualization.Visualizer)
    vis = VisClass()
    vis.create_window(window_name=f"Review: {ts_dir.name}", width=WINDOW_W, height=WINDOW_H, visible=True)

    pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.03, 0.03, 0.03], dtype=np.float32)
    opt.point_size = float(init_point_size)
    opt.light_on = False

    # 播放控制
    state = PlayerState(fps=init_fps, color_by=init_color_by)
    register_callbacks(vis, state, ts_dir)

    # 提前做稳健分位（减少每帧开销）
    use_intensity = (state.color_by == "intensity")
    p1, p99 = compute_global_stats(csv_files, use_intensity)

    # 初次相机设置需要一帧点云
    df0 = pd.read_csv(csv_files[0], usecols=['x','y','z'] + (['intensity'] if use_intensity else []))
    pts0 = df0[['x','y','z']].to_numpy(dtype=np.float32, copy=False)
    pts0 = rotate_point_cloud(pts0, yaw=yaw, pitch=pitch, roll=roll)
    pcd.points = o3d.utility.Vector3dVector(pts0)
    if use_intensity and 'intensity' in df0.columns:
        colors0 = colors_from_values(df0['intensity'].to_numpy(np.float32), p1, p99)
    else:
        d0 = np.linalg.norm(pts0, axis=1)
        colors0 = colors_from_values(d0, p1, p99)
    pcd.colors = o3d.utility.Vector3dVector(colors0)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    bbox = pcd.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center())
    ctr.set_front([0, -1, 0.5])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.5)

    print(
        " Controls: [SPACE]=Play/Pause  [→]/[←]=Next/Prev  [P]=Pass  [K]=Skip  [C]=Color  "
        "[[]/[]]=PointSize  [S]=Screenshot  [Q]=Quit"
    )

    idx = 0
    last_time = time.perf_counter()
    total_frames = len(csv_files)
    passed = False
    quit_all = False

    try:
        while True:
            # 键响应
            if state.want_quit:
                quit_all = True
                break
            if state.want_pass:
                passed = True
                break
            if state.want_skip:
                break

            # 点大小调整
            if abs(state.point_size_delta) > 1e-6:
                opt = vis.get_render_option()
                opt.point_size = float(np.clip(opt.point_size + state.point_size_delta, 0.1, 5.0))
                state.point_size_delta = 0.0

            # 着色切换（重算分位，保证鲁棒度）
            use_intensity_now = (state.color_by == "intensity")
            if use_intensity_now != use_intensity:
                use_intensity = use_intensity_now
                p1, p99 = compute_global_stats(csv_files, use_intensity)

            # 帧索引控制
            if state.want_next:
                idx = min(idx + 1, total_frames - 1)
                state.want_next = False
            elif state.want_prev:
                idx = max(idx - 1, 0)
                state.want_prev = False
            else:
                # 自动播放
                now = time.perf_counter()
                dt = now - last_time
                frame_interval = 1.0 / max(state.fps, 1e-6)
                if (not state.paused) and dt >= frame_interval:
                    if idx < total_frames - 1:
                        idx += 1
                    last_time = now

            # 载入并渲染当前帧
            df = pd.read_csv(csv_files[idx], usecols=['x','y','z'] + (['intensity'] if use_intensity else []))
            pts = df[['x','y','z']].to_numpy(dtype=np.float32, copy=False)
            pts = rotate_point_cloud(pts, yaw=yaw, pitch=pitch, roll=roll)
            if use_intensity and 'intensity' in df.columns:
                colors = colors_from_values(df['intensity'].to_numpy(np.float32), p1, p99)
            else:
                d = np.linalg.norm(pts, axis=1)
                colors = colors_from_values(d, p1, p99)

            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)

            # 截图
            if state.want_screenshot:
                shot_name = ts_dir / f"screenshot_{idx:05d}.png"
                o3d.io.write_image(str(shot_name), vis.capture_screen_float_buffer(do_render=True))
                print(f"[Screenshot] {shot_name}")
                state.want_screenshot = False

            vis.poll_events()
            vis.update_renderer()

            # 轻微让出时间片
            time.sleep(0.001)

    finally:
        vis.destroy_window()

    return (passed, total_frames, quit_all)


# ===========================
# 批量审核流程
# ===========================
def batch_review(root_folder: str):
    root_path = Path(root_folder)
    if not root_path.exists():
        print(f"[Error] Root not found: {root_folder}")
        return

    # 日志文件：根目录同级 logs/review_pass.txt
    logs_dir = root_path.parent / "logs"
    pass_log = logs_dir / "vlp16_review_log.txt"
    passed_set = load_pass_set(pass_log)

    # 收集所有含 CSV 的时间戳目录
    ts_dirs = []
    for dirpath, _, filenames in os.walk(root_path):
        if any(fn.lower().endswith(".csv") for fn in filenames):
            ts_dirs.append(Path(dirpath))
    ts_dirs.sort()

    print(f"[Info] Found {len(ts_dirs)} sequences to check.")
    print(f"[Info] Already passed: {len(passed_set)} (will be skipped).")

    for ts_dir in ts_dirs:
        ts_dir_str = str(ts_dir)
        if (ts_dir_str in passed_set) or ("deprecated" in ts_dir_str):
            print(f"[Skip PASS] {ts_dir}")
            continue

        # 提取 MR / run
        try:
            run_dir = ts_dir.parent.name
            mr_dir  = ts_dir.parent.parent.name
        except Exception:
            run_dir, mr_dir = "UNKNOWN_RUN", "UNKNOWN_MR"

        print(f"\n[Review] MR={mr_dir}, RUN={run_dir}, TS={ts_dir.name}")
        passed, total_frames, quit_all = review_one_sequence(ts_dir)
        if quit_all:
            print("[QUIT] Stop reviewing.")
            break

        if passed:
            append_pass_log(pass_log, mr_dir, run_dir, ts_dir, total_frames)
            print(f"[PASS] Logged: {ts_dir}")
        else:
            print(f"[NEXT] Not marked PASS: {ts_dir}")

    print(f"\n[Summary] Review finished. Log file: {pass_log}")


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    # 把 root 改成你的“MR 目录的上层”或更上层路径
    root = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\MR\vlp16"
    batch_review(root)
