#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
import threading
import queue
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===========================
# 异步视频写入（优先 imageio+ffmpeg，不行退回 OpenCV）
# ===========================
class AsyncVideoSink:
    def __init__(self, path, fps, width, height):
        self.path = str(path)
        self.fps = float(fps)
        self.w = int(width)
        self.h = int(height)
        self.q = queue.Queue(maxsize=8)  # 防止内存暴涨
        self._stop = object()
        self.backend = None
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def _run(self):
        writer = None
        try:
            # 优先尝试 imageio+ffmpeg（最快、兼容好）
            try:
                import imageio.v2 as imageio
                writer = imageio.get_writer(
                    self.path,
                    fps=self.fps,
                    codec='libx264',
                    quality=8,
                    ffmpeg_params=[
                        '-preset', 'ultrafast',  # 极快预设
                        '-crf', '25',            # 质量-码率平衡
                        '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart',
                        '-threads', '2',
                    ],
                )
                self.backend = "imageio"
                print("[Video] Using imageio+ffmpeg (ultrafast)")
            except Exception as e:
                print(f"[Video] imageio backend unavailable: {e}")
                # 退回 OpenCV
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                w = cv2.VideoWriter(self.path, fourcc, self.fps, (self.w, self.h))
                if not w.isOpened():
                    raise RuntimeError("cv2.VideoWriter not opened")
                writer = w
                self.backend = "cv2"
                print("[Video] Fallback to OpenCV VideoWriter (mp4v)")

            # 消费帧队列
            while True:
                frm = self.q.get()
                if frm is self._stop:
                    break
                if self.backend == "imageio":
                    writer.append_data(frm)
                else:
                    import cv2
                    if frm.shape[1] != self.w or frm.shape[0] != self.h:
                        frm = cv2.resize(frm, (self.w, self.h), interpolation=cv2.INTER_AREA)
                    writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
        finally:
            if writer is not None:
                try:
                    writer.close() if self.backend == "imageio" else writer.release()
                except Exception:
                    pass

    def write(self, frame_rgb_uint8: np.ndarray):
        self.q.put(frame_rgb_uint8, block=True)

    def close(self):
        self.q.put(self._stop, block=True)
        self._thr.join()


# ===========================
# 颜色映射（全局分位数 + Turbo LUT）
# ===========================
_TURBO_LUT = (plt.get_cmap('turbo', 256)(np.linspace(0, 1, 256))[:, :3]).astype(np.float32)

def _compute_global_stats(csv_files, use_intensity, max_frames=50, max_pts_per_frame=5000):
    """采样前若干帧，估计 1%/99% 分位，用于稳健归一化。"""
    samples = []
    for f in csv_files[:min(len(csv_files), max_frames)]:
        usecols = ['x', 'y', 'z'] + (['intensity'] if use_intensity else [])
        # 快速 CSV 读取：限定列/类型/C 引擎/内存映射
        df = pd.read_csv(
            f, usecols=usecols,
            dtype={c: 'float32' for c in usecols},
            engine='c', memory_map=True
        )
        if use_intensity and 'intensity' in df.columns:
            arr = df['intensity'].to_numpy()
        else:
            pts = df[['x', 'y', 'z']].to_numpy(dtype=np.float32, copy=False)
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

def _colors_from_values(values: np.ndarray, p1: float, p99: float) -> np.ndarray:
    norm01 = (values - p1) / max(p99 - p1, 1e-6)
    idx = (np.clip(norm01, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _TURBO_LUT[idx]


# ===========================
# 点云旋转
# ===========================
def rotate_point_cloud(points: np.ndarray, yaw=0, pitch=0, roll=0) -> np.ndarray:
    yaw = np.radians(yaw).astype(np.float32)
    pitch = np.radians(pitch).astype(np.float32)
    roll = np.radians(roll).astype(np.float32)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    R_yaw = np.array([[cy, -sy, 0],
                      [sy,  cy, 0],
                      [0,    0, 1]], dtype=np.float32)
    R_pitch = np.array([[ cp, 0, sp],
                        [  0, 1,  0],
                        [-sp, 0, cp]], dtype=np.float32)
    R_roll = np.array([[1,  0,  0],
                       [0, cr, -sr],
                       [0, sr,  cr]], dtype=np.float32)

    R = R_yaw @ R_pitch @ R_roll
    return (points @ R.T).astype(np.float32)


# ===========================
# 离线渲染 + 录制（单个时间戳文件夹）
# ===========================
def visualize_and_record_folder(
    folder_path,
    out_mp4_path,
    yaw=0, pitch=0, roll=0,
    fps=15, color_by="distance",  # 或 "intensity"
    point_size=0.35,
    bg_color=(0.03, 0.03, 0.03),
    width=1280, height=900,
    max_points=80000  # 每帧点数上限（随机下采样），None/0 表示不限制
):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"[Skip] No CSV files in: {folder_path}")
        return False

    # Open3D 离线窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='LiDAR Offscreen', width=width, height=height, visible=False)

    pcd = o3d.geometry.PointCloud()
    added = False

    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color, dtype=np.float32)
    opt.point_size = float(point_size)
    opt.light_on = False  # 关光照更快

    # 坐标轴（用于初次构图）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(axis)

    # 全局分位数（一次）
    use_intensity = (color_by == "intensity")
    p1, p99 = _compute_global_stats(csv_files, use_intensity)

    # 输出
    out_mp4_path = _unique_path(Path(out_mp4_path))
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    sink = AsyncVideoSink(out_mp4_path, fps=fps, width=width, height=height)
    print(f"[Start] {folder_path} -> {out_mp4_path}")

    try:
        for idx, csv_file in tqdm(
            enumerate(csv_files),
            total=len(csv_files),
            desc=f"{Path(folder_path).name}",
            ncols=100,
            dynamic_ncols=True
        ):
            # 快速 CSV 读取
            usecols = ['x', 'y', 'z'] + (['intensity'] if use_intensity else [])
            df = pd.read_csv(
                csv_file, usecols=usecols,
                dtype={c: 'float32' for c in usecols},
                engine='c', memory_map=True
            )

            pts = df[['x', 'y', 'z']].to_numpy(dtype=np.float32, copy=False)

            # 可选随机下采样：控制每帧点数，显著提速
            if max_points and pts.shape[0] > max_points:
                sel = np.random.choice(pts.shape[0], size=max_points, replace=False)
                pts = pts[sel]
                if use_intensity and 'intensity' in df.columns:
                    inten = df['intensity'].to_numpy(dtype=np.float32, copy=False)[sel]
            else:
                inten = df['intensity'].to_numpy(dtype=np.float32, copy=False) if (use_intensity and 'intensity' in df.columns) else None

            # 姿态旋转
            pts = rotate_point_cloud(pts, yaw=yaw, pitch=pitch, roll=roll)

            # 颜色映射（查 LUT，极快）
            if use_intensity and (inten is not None):
                colors = _colors_from_values(inten, p1, p99)
            else:
                d = np.linalg.norm(pts, axis=1)
                colors = _colors_from_values(d, p1, p99)

            # 更新点云
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if not added:
                vis.add_geometry(pcd)
                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                ctr.set_lookat(bbox.get_center())
                ctr.set_front([0, -1, 0.5])
                ctr.set_up([0, 0, 1])
                ctr.set_zoom(0.5)
                added = True
            else:
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

            # 抓帧并送编码线程
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            sink.write((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))

            # 离线渲染无需 sleep；保留最小 yield
            # time.sleep(0.0)

    finally:
        sink.close()
        vis.destroy_window()
        print(f"[Done] Saved video: {out_mp4_path}")

    return True


# ===========================
# 批量处理（外层 Run 级进度条）
# 目录结构：.../<MR_name>/<run_xx>/<timestamp>/*.csv
# 输出：<root 同级>/videos/<MR_name>/<MR_name>_<run_xx>.mp4
# ===========================
def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    k = 2
    while True:
        cand = path.with_name(f"{stem}_v{k}{suf}")
        if not cand.exists():
            return cand
        k += 1

def batch_process(
    root_folder,
    yaw=-90, pitch=-100, roll=-5,
    fps=15, color_by="distance", cmap="turbo",
    point_size=0.35, bg_color=(0.03, 0.03, 0.03),
    width=1280, height=900,
    max_points=80000
):
    root_path = Path(root_folder)
    if not root_path.exists():
        print(f"[Error] Root folder not found: {root_folder}")
        return

    videos_root = root_path.parent / "videos"
    print(f"[Info] Output root: {videos_root}")

    # 预扫描所有含 CSV 的时间戳目录
    csv_dirs = []
    for dirpath, _, filenames in os.walk(root_path):
        if any(fn.lower().endswith(".csv") for fn in filenames):
            csv_dirs.append(Path(dirpath))

    # 外层批处理进度条
    count = 0
    for csv_dir in tqdm(csv_dirs, total=len(csv_dirs), desc="Runs", ncols=100, dynamic_ncols=True):
        try:
            run_dir = csv_dir.parent.name         # run_xx
            mr_dir = csv_dir.parent.parent.name   # MR_*（参与者）
        except Exception:
            print(f"[Warn] Unrecognized path pattern: {csv_dir}")
            continue

        out_dir = videos_root / mr_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        video_name = f"{mr_dir}_{run_dir}.mp4"
        out_mp4 = out_dir / video_name

        ok = visualize_and_record_folder(
            folder_path=str(csv_dir),
            out_mp4_path=str(out_mp4),
            yaw=yaw, pitch=pitch, roll=roll,
            fps=fps, color_by=color_by,
            point_size=point_size, bg_color=bg_color,
            width=width, height=height,
            max_points=max_points
        )
        if ok:
            count += 1

    print(f"[Summary] Processed {count}/{len(csv_dirs)} folder(s) with CSV.")


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    # 把 root 改成你的“MR 目录的上层”或更上层路径
    root = r"F:\test"

    batch_process(
        root_folder=root,
        yaw=-90, pitch=-100, roll=-5,
        fps=30,                     # 可调：越低越快
        color_by="distance",        # 或 "intensity"
        point_size=0.35,
        bg_color=(0.03, 0.03, 0.03),
        width=1280, height=960,     # 可调：更低分辨率更快
        max_points=80000            # 可调：限制每帧点数（None 表示不限制）
    )
