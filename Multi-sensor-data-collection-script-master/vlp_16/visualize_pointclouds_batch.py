import os
import glob
import numpy as np
import open3d as o3d
import pandas as pd
import time
import matplotlib.pyplot as plt


def normalize_intensity(intensity, cmap_name='plasma'):
    """Map LiDAR intensity values to a colormap."""
    cmap = plt.get_cmap(cmap_name)
    norm_intensity = np.clip(intensity, 0, 255) / 255.0
    colors = cmap(norm_intensity)[:, :3]
    return colors


def normalize_distance(points, cmap_name='viridis'):
    """Color points based on distance from the origin."""
    distances = np.linalg.norm(points, axis=1)
    norm_dist = (distances - distances.min()) / (distances.max() - distances.min())
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm_dist)[:, :3]
    return colors


def rotate_point_cloud(points, yaw=0, pitch=0, roll=0):
    """Apply yaw, pitch, roll rotation to the point cloud."""
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


def visualize_lidar_3d(folder_path, video_path, yaw=0, pitch=0, roll=0, fps=10, color_mode='distance'):
    """Visualize LiDAR CSVs as animated 3D point clouds and save to video."""
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"⚠️ No CSV files found in {folder_path}")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='🚗 LiDAR Viewer', width=1280, height=960)
    vis.get_render_option().background_color = np.asarray([0.05, 0.05, 0.1])
    vis.get_render_option().point_size = 2.5
    vis.get_render_option().show_coordinate_frame = True

    pcd = o3d.geometry.PointCloud()
    first = True

    recorder = o3d.visualization.VideoRecorder(video_path, size=(1280, 960), fps=fps)
    recorder.open()
    print(f"🎥 Recording: {video_path}")

    for idx, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        points = df[['x', 'y', 'z']].to_numpy()
        intensity = df['intensity'].to_numpy()

        points = rotate_point_cloud(points, yaw=yaw, pitch=pitch, roll=roll)
        pcd.points = o3d.utility.Vector3dVector(points)

        if color_mode == 'intensity':
            pcd.colors = o3d.utility.Vector3dVector(normalize_intensity(intensity))
        else:
            pcd.colors = o3d.utility.Vector3dVector(normalize_distance(points))

        if first:
            vis.add_geometry(pcd)
            first = False
        else:
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        recorder.capture_screen_float_buffer(False)

        print(f"🟢 Frame {idx+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        time.sleep(1.0 / fps)

    recorder.close()
    vis.destroy_window()
    print(f"✅ Saved video: {video_path}\n")


def process_all_runs(base_folder, yaw=0, pitch=0, roll=0, fps=10, color_mode='distance'):
    """Recursively find all 'run_' folders and process LiDAR data inside."""
    print(f"🔍 Scanning for LiDAR data under: {base_folder}")
    for root, dirs, files in os.walk(base_folder):
        # Identify folders like '.../run_xx/20251103_171334'
        if any(f.endswith(".csv") for f in files):
            if "run_" in os.path.basename(os.path.dirname(root)):
                run_folder = os.path.dirname(root)
                run_name = os.path.basename(run_folder)
                video_path = os.path.join(run_folder, f"{run_name}.mp4")

                print(f"📂 Processing {root}")
                visualize_lidar_3d(
                    folder_path=root,
                    video_path=video_path,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    fps=fps,
                    color_mode=color_mode
                )


if __name__ == "__main__":
    base_folder = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\vlp16"
    process_all_runs(
        base_folder,
        yaw=0,
        pitch=0,
        roll=0,
        fps=15,
        color_mode='distance'  # or 'intensity'
    )
