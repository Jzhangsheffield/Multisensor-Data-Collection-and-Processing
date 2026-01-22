import os
import glob
import numpy as np
import open3d as o3d
import pandas as pd
import time 

import matplotlib.pyplot as plt

def normalize_intensity(intensity, cmap_name='viridis'):
    """使用colormap将intensity映射为彩色"""
    cmap = plt.get_cmap(cmap_name)
    norm_intensity = np.clip(intensity, 0, 255) / 255.0
    colors = cmap(norm_intensity)[:, :3]  # cmap returns RGBA; drop A
    return colors

def normalize_distance(points, cmap_name='viridis'):
    """使用点到原点的距离进行颜色映射"""
    distances = np.linalg.norm(points, axis=1)
    norm_dist = (distances - distances.min()) / (distances.max() - distances.min())
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm_dist)[:, :3]
    return colors

def rotate_point_cloud(points, yaw=0, pitch=0, roll=0):
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    R_roll = np.array([
        [1, 0,              0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    R = R_yaw @ R_pitch @ R_roll
    return points @ R.T

# def normalize_intensity(intensity):
#     intensity = np.clip(intensity, 0, 255)
#     return np.stack([intensity, intensity, intensity], axis=1) / 255.0  # grayscale

def visualize_lidar_3d(folder_path, yaw=0, pitch=0, roll=0, fps=10):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print("No CSV files found.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='LiDAR Viewer', width=960, height=720)
    pcd = o3d.geometry.PointCloud()

    first = True
    for idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        points = df[['x', 'y', 'z']].to_numpy()
        intensity = df['intensity'].to_numpy()

        points = rotate_point_cloud(points, yaw=yaw, pitch=pitch, roll=roll)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(normalize_distance(points))

        if first:
            vis.add_geometry(pcd)
            first = False
        else:
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        print(f"Frame {idx+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        time.sleep(0.1 / fps)

    vis.destroy_window()



if __name__ == "__main__":
    folder = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\vlp16\sample_parpare_Oct_18_clean\J\run_15\20251019_181410"
    visualize_lidar_3d(folder, yaw=0, pitch=0, roll=0, fps=10)
