import os
import h5py
import numpy as np
import imageio
import argparse
from tqdm import tqdm


def save_frames_from_hdf5(hdf5_path: str, output_dir: str, max_frames: int = None):
    print(f"Loading data from: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as h5f:
        depth_data = h5f["depth"]
        rgb_data = h5f["rgb"]
        timestamps = h5f["timestamp"]

        total_frames = depth_data.shape[0]
        if max_frames:
            total_frames = min(total_frames, max_frames)

        serial = os.path.splitext(os.path.basename(hdf5_path))[0]
        save_dir_depth = os.path.join(output_dir, serial, "depth")
        save_dir_rgb = os.path.join(output_dir, serial, "rgb")
        os.makedirs(save_dir_depth, exist_ok=True)
        os.makedirs(save_dir_rgb, exist_ok=True)

        for i in tqdm(range(total_frames), desc=f"Saving frames from {serial}"):
            # RGB
            rgb_frame = rgb_data[i][:,:,::-1]
            rgb_path = os.path.join(save_dir_rgb, f"rgb_{i:05d}.png")
            imageio.imwrite(rgb_path, rgb_frame)

            # Depth - normalize to 0-255 for visualization
            depth_frame = depth_data[i]
            depth_norm = (depth_frame - np.min(depth_frame)) / (np.max(depth_frame) - np.min(depth_frame))
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_path = os.path.join(save_dir_depth, f"depth_{i:05d}.png")
            imageio.imwrite(depth_path, depth_uint8)

        print(f"Saved {total_frames} frames to: {os.path.join(output_dir, serial)}")


def main():
    parser = argparse.ArgumentParser(description="Export HDF5 Kinect frames to PNG images.")
    parser.add_argument("--input_hdf5", default=r"C:\multi_sensor_sync_save_data_test\kinect\20250716_161803\001484412812.hdf5", help="Path to .hdf5 file")
    parser.add_argument("--output_dir", default=r"C:\multi_sensor_sync_save_data_test\kinect\20250716_161803\output_images", help="Where to save .png files")
    parser.add_argument("--max_frames", type=int, default=30, help="Maximum number of frames to export")
    args = parser.parse_args()

    save_frames_from_hdf5(args.input_hdf5, args.output_dir, args.max_frames)


if __name__ == "__main__":
    main()
