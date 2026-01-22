import os
import datetime
import multiprocessing
import time

import cv2
import numpy as np

from pyk4a import Config, PyK4A, WiredSyncMode, DepthMode, connected_device_count
from helpers import colorize  # Assumes this exists and works


def enumerate_available_devices():
    cnt = connected_device_count()
    device_dict = dict()
    if not cnt:
        print("No devices available")
        exit()
    print(f"Available devices: {cnt}")
    for device_id in range(cnt):
        device = PyK4A(device_id=device_id)
        device.open()
        print(f"{device_id}: {device.serial}")
        device_dict[device_id] = device.serial
        device.close()
    return device_dict


def create_save_folders(base_dir, device_dict):
    _time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    updated_base_dir = os.path.join(base_dir, _time)
    os.makedirs(updated_base_dir, exist_ok=True)
    device_dirs = dict()
    for device_id, serial in device_dict.items():
        rgb_dir = os.path.join(updated_base_dir, serial, "RGB")
        depth_dir = os.path.join(updated_base_dir, serial, "Depth")
        os.makedirs(rgb_dir)
        os.makedirs(depth_dir)
        device_dirs[device_id] = {"RGB": rgb_dir, "Depth": depth_dir}
    print(f"All folders created at {updated_base_dir}")
    return device_dirs


def save_img(img, save_dir, is_color=True):
    timestamp = datetime.datetime.now().timestamp()
    ext = ".jpg" if is_color else ".png"
    img_name = f"{timestamp}{ext}"
    img_path = os.path.join(save_dir, img_name)

    if is_color:
        cv2.imwrite(img_path, img) #, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        cv2.imwrite(img_path, img)

    print(f"[Saved] {img_path}")


def camera_process(device_id, config, save_dir):
    print(f"Starting camera process {device_id}")
    cam = PyK4A(device_id=device_id, config=config)
    cam.start()

    try:
        while True:
            capture = cam.get_capture()

            if capture.color is not None:
                color_img = capture.color[:, :, :3]
                save_img(color_img, save_dir["RGB"], is_color=True)
                # cv2.imshow(f"Camera {device_id} - Color", color_img)
            if capture.depth is not None:
                depth_img = colorize(capture.depth, (100, 5000), cv2.COLORMAP_JET)
                # depth_img = capture.depth
                save_img(depth_img, save_dir["Depth"], is_color=False)
                # cv2.imshow(f"Camera {device_id} - Color", depth_img)
            # Optional: Slow down for testing
            # time.sleep(0.05)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except KeyboardInterrupt:
        print(f"Stopping camera process {device_id}...")

    cam.stop()
    print(f"Camera {device_id} stopped.")


def main():
    base_dir = "C:/multi_sensor_sync_save_data_test/kinect"
    device_dict = enumerate_available_devices()
    save_dirs = create_save_folders(base_dir, device_dict)

    # Prepare device configs
    configs = {
        0: Config(wired_sync_mode=WiredSyncMode.MASTER, depth_mode=DepthMode.NFOV_UNBINNED),
        1: Config(wired_sync_mode=WiredSyncMode.SUBORDINATE, subordinate_delay_off_master_usec=50, depth_mode=DepthMode.NFOV_UNBINNED),
        2: Config(wired_sync_mode=WiredSyncMode.SUBORDINATE, subordinate_delay_off_master_usec=100, depth_mode=DepthMode.NFOV_UNBINNED)
    }

    processes = []
    for device_id in device_dict:
        config = configs[device_id]
        save_dir = save_dirs[device_id]

        p = multiprocessing.Process(target=camera_process, args=(device_id, config, save_dir))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating all camera processes...")
        for p in processes:
            p.terminate()
            p.join()

    print("All camera processes terminated.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Important for Windows
    main()
