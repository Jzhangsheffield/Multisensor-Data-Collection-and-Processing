import os
import glob
import time
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import shutil

# ---------- 配色与归一化 ----------
def normalize_intensity(intensity, cmap_name='turbo'):
    intensity = np.asarray(intensity)
    p_low, p_high = np.percentile(intensity, [1, 99])
    denom = max(p_high - p_low, 1e-6)
    norm = np.clip((intensity - p_low) / denom, 0, 1)
    cmap = plt.get_cmap(cmap_name)
    return cmap(norm)[:, :3]

def normalize_distance(points, cmap_name='turbo'):
    distances = np.linalg.norm(points, axis=1)
    p_low, p_high = np.percentile(distances, [1, 99])
    denom = max(p_high - p_low, 1e-6)
    norm = np.clip((distances - p_low) / denom, 0, 1)
    cmap = plt.get_cmap(cmap_name)
    return cmap(norm)[:, :3]

def rotate_point_cloud(points, yaw=0, pitch=0, roll=0):
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0,            0,           1]])
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0,             1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    R_roll = np.array([[1, 0,              0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll),  np.cos(roll)]])
    R = R_yaw @ R_pitch @ R_roll
    return points @ R.T


# ---------- 自动清理空文件 ----------
def remove_empty_csv_files(folder_path, move_to_subfolder=True):
    """Scan folder for empty CSVs and move or delete them."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    empty_files = [f for f in csv_files if os.path.getsize(f) == 0]

    if not empty_files:
        print("✅ No empty CSV files detected.")
        return

    if move_to_subfolder:
        empty_dir = os.path.join(folder_path, "_empty_csvs")
        os.makedirs(empty_dir, exist_ok=True)
        for f in empty_files:
            shutil.move(f, os.path.join(empty_dir, os.path.basename(f)))
        print(f"🧹 Moved {len(empty_files)} empty CSV files to '{empty_dir}'")
    else:
        for f in empty_files:
            os.remove(f)
        print(f"🧹 Deleted {len(empty_files)} empty CSV files.")


# ---------- Legacy 可视化 ----------
def visualize_lidar_3d_legacy(folder_path, yaw=0, pitch=0, roll=0, fps=15,
                              color_by="distance", cmap="turbo",
                              point_size=0.4, bg_color=(0.03, 0.03, 0.03)):

    # Step 1: 清理空 CSV 文件
    remove_empty_csv_files(folder_path, move_to_subfolder=True)

    # Step 2: 获取 CSV 列表
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print("No valid CSV files found after cleanup.")
        return

    VisClass = getattr(o3d.visualization, "VisualizerWithKeyCallback",
                       o3d.visualization.Visualizer)
    vis = VisClass()
    vis.create_window(window_name='LiDAR Viewer (Legacy)', width=1280, height=900)

    pcd = o3d.geometry.PointCloud()
    added = False

    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color, dtype=np.float32)
    opt.point_size = float(point_size)
    opt.light_on = True

    # 注册热键调整点大小
    def _set_psize(delta):
        def _fn(v):
            o = v.get_render_option()
            o.point_size = float(np.clip(o.point_size + delta, 0.1, 5.0))
            print(f"Point size -> {o.point_size:.2f}")
            return False
        return _fn

    if hasattr(vis, "register_key_callback"):
        vis.register_key_callback(ord('['), _set_psize(-0.1))
        vis.register_key_callback(ord(']'), _set_psize(+0.1))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(axis)

    # Step 3: 播放帧
    for idx, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            if df.empty or not {'x', 'y', 'z'}.issubset(df.columns):
                print(f"Skipping invalid file (missing x,y,z): {os.path.basename(csv_file)}")
                continue
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file (EmptyDataError): {os.path.basename(csv_file)}")
            continue
        except Exception as e:
            print(f"Skipping {os.path.basename(csv_file)} due to error: {e}")
            continue

        points = df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        points = rotate_point_cloud(points, yaw=yaw, pitch=pitch, roll=roll)

        if (color_by == "intensity") and ('intensity' in df.columns):
            colors = normalize_intensity(df['intensity'].to_numpy(), cmap_name=cmap)
        else:
            colors = normalize_distance(points, cmap_name=cmap)

        pcd.points = o3d.utility.Vector3dVector(points)
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
        print(f"Frame {idx+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        time.sleep(0.1 / max(fps, 1))

    vis.destroy_window()


# ---------- 主程序 ----------
if __name__ == "__main__":
    folder = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\vlp16\sample_prepare_Nov_21_clean\MR\run_43\20251121_174848"
    visualize_lidar_3d_legacy(folder,
                              yaw=-90, pitch=-100, roll=-5,
                              fps=15, color_by="distance", cmap="turbo",
                              point_size=0.35,
                              bg_color=(0.03, 0.03, 0.03))
