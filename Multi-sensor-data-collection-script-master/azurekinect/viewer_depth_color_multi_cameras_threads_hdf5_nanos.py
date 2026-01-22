import os
import datetime
import time
import multiprocessing
from typing import Dict

import h5py
import numpy as np
from pyk4a import (
    Config,
    PyK4A,
    WiredSyncMode,
    DepthMode,
    connected_device_count,
    FPS,
)

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def enumerate_available_devices() -> Dict[int, str]:
    """Returns a mapping: device_id -> serial number."""
    cnt = connected_device_count()
    if not cnt:
        print("No devices available")
        raise RuntimeError("No Azure Kinect devices detected")
    print(f"Available devices: {cnt}")
    device_dict: Dict[int, str] = {}
    for device_id in range(cnt):
        device = PyK4A(device_id=device_id)
        device.open()
        device_dict[device_id] = device.serial
        print(f"{device_id}: {device.serial}")
        device.close()
    return device_dict


def create_hdf5_paths(base_dir: str, device_dict: Dict[int, str]) -> Dict[int, str]:
    """Create a timestamped folder and return the *.hdf5 path for each camera."""
    timestamp_folder = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, timestamp_folder)
    os.makedirs(session_dir, exist_ok=True)

    file_paths = {}
    for device_id, serial in device_dict.items():
        file_path = os.path.join(session_dir, f"{serial}.hdf5")
        file_paths[device_id] = file_path
    print(f"All HDF5 files will be saved under {session_dir}")
    return file_paths

# --------------------------------------------------------------------------------------
# Core acquisition logic
# --------------------------------------------------------------------------------------

def init_datasets(h5f: h5py.File, first_depth: np.ndarray, first_rgb: np.ndarray):
    """Create extendible datasets for depth, rgb and timestamp."""
    depth_shape = first_depth.shape
    rgb_shape = first_rgb.shape

    h5f.create_dataset(
        "depth",
        shape=(0, *depth_shape),
        maxshape=(None, *depth_shape),
        dtype=first_depth.dtype,
        chunks=(1, *depth_shape),
    )
    h5f.create_dataset(
        "rgb",
        shape=(0, *rgb_shape),
        maxshape=(None, *rgb_shape),
        dtype=first_rgb.dtype,
        chunks=(1, *rgb_shape),
    )
    h5f.create_dataset(
        "timestamp",
        shape=(0,),
        maxshape=(None,),
        dtype="i8",
        chunks=True,
    )


def append_frame(h5f: h5py.File, depth_frame: np.ndarray, rgb_frame: np.ndarray, ts: float):
    """Append one frame to all three datasets."""
    depth_ds = h5f["depth"]
    rgb_ds = h5f["rgb"]
    ts_ds = h5f["timestamp"]

    next_index = depth_ds.shape[0]
    depth_ds.resize((next_index + 1, *depth_ds.shape[1:]))
    rgb_ds.resize((next_index + 1, *rgb_ds.shape[1:]))
    ts_ds.resize((next_index + 1,))

    depth_ds[next_index] = depth_frame
    rgb_ds[next_index] = rgb_frame
    ts_ds[next_index] = ts


def camera_process(device_id: int, config: Config, hdf5_path: str, stop_event: multiprocessing.Event, start_event: multiprocessing.Event):
    print(f"[Process {device_id}] Initializing camera …")
    cam = PyK4A(device_id=device_id, config=config)
    cam.start()

    with h5py.File(hdf5_path, "w") as h5f:
        while True:
            capture = cam.get_capture()
            if capture.color is not None and capture.depth is not None:
                break
        first_depth = capture.depth
        first_rgb = capture.color[:, :, :3]  # drop alpha
        init_datasets(h5f, first_depth, first_rgb)
        print(f"[Process {device_id}] Ready. Waiting for start signal …")

        # Wait for all processes to be ready before recording
        start_event.wait()
        print(f"[Process {device_id}] Started recording …")

        append_frame(h5f, first_depth, first_rgb, time.time_ns())
        frame_count = 1
        start_time = datetime.datetime.now()

        try:
            while not stop_event.is_set():
                capture = cam.get_capture()
                if capture.color is None or capture.depth is None:
                    continue
                depth_frame = capture.depth
                rgb_frame = capture.color[:, :, :3]
                timestamp = time.time_ns()
                append_frame(h5f, depth_frame, rgb_frame, timestamp)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"[Process {device_id}] Saved {frame_count} frames → {os.path.basename(hdf5_path)}")
        except KeyboardInterrupt:
            print(f"[Process {device_id}] KeyboardInterrupt – stopping …")
        finally:
            cam.stop()
            end_time = datetime.datetime.now()
            print(f"[Process {device_id}] Stopped. Total frames: {frame_count}, total duration: {end_time - start_time}")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main():
    base_dir = r"C:\multi_sensor_sync_save_data_test\kinect\Mike_run"  # Adjust as needed

    device_dict = enumerate_available_devices()
    hdf5_paths = create_hdf5_paths(base_dir, device_dict)

    configs = {
        0: Config(
            # wired_sync_mode=WiredSyncMode.STANDALONE,
            wired_sync_mode=WiredSyncMode.MASTER,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
        ),
        1: Config(
            # wired_sync_mode=WiredSyncMode.STANDALONE,
            wired_sync_mode=WiredSyncMode.SUBORDINATE,
            subordinate_delay_off_master_usec=200,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
        ),
        2: Config(
            # wired_sync_mode=WiredSyncMode.STANDALONE,
            wired_sync_mode=WiredSyncMode.SUBORDINATE,
            subordinate_delay_off_master_usec=400,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
        ),
    }

    processes = []
    start_event = multiprocessing.Event()

    try:
        for device_id in device_dict:
            p_stop = multiprocessing.Event()
            p = multiprocessing.Process(
                target=camera_process,
                args=(device_id, configs[device_id], hdf5_paths[device_id], p_stop, start_event),
                daemon=True,
            )
            p.start()
            processes.append((p, p_stop))

        print("[Main] Waiting 5 seconds for all processes to initialize …")
        import time
        time.sleep(5)
        print("[Main] Triggering start for all cameras.")
        start_event.set()

        for p, _ in processes:
            p.join()
    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt – terminating all camera processes …")
        for p, ev in processes:
            ev.set()
            p.join()
    print("[Main] All camera processes terminated.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # Windows requirement
    main()
