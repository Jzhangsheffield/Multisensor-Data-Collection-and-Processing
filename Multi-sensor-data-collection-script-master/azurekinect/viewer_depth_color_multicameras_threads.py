import os
import datetime
import threading
import time

import cv2
import numpy as np

from pyk4a import Config, PyK4A, WiredSyncMode, DepthMode, connected_device_count
from helpers import colorize  # Make sure this exists


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
        device_dict.update({device_id: device.serial})
        device.close()
    return device_dict


def create_save_folders(base_dir, device_dict):
    _time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    updated_base_dir = os.path.join(base_dir, _time)
    os.makedirs(updated_base_dir, exist_ok=True)
    device_dirs = dict()
    for item in device_dict.items():
        device_dir = {
            "RGB": os.path.join(updated_base_dir, item[1], "RGB"),
            "Depth": os.path.join(updated_base_dir, item[1], "Depth")
        }
        os.makedirs(device_dir["RGB"])
        os.makedirs(device_dir["Depth"])
        device_dirs[item[0]] = device_dir
    print(f"All folders created at {updated_base_dir}")
    return device_dirs


def save_img(img, save_dir, is_color=True):
    timestamp = datetime.datetime.now().timestamp()
    ext = ".jpg" if is_color else ".png"
    img_name = f"{timestamp}{ext}"
    img_path = os.path.join(save_dir, img_name)

    if is_color:
        cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        cv2.imwrite(img_path, img)  # Use .png or .tiff for depth

    print(f"[Saved immediately] {img_path}")


def camera_capture_loop(cam, save_dir, cam_id):

    while True:
        capture = cam.get_capture()

        if capture.color is not None:
            color_img = capture.color[:, :, :3]
            cv2.imshow(f"Camera {cam_id} - Color", color_img)
            save_img(color_img, save_dir["RGB"], is_color=True)

        if capture.depth is not None:
            depth_img = colorize(capture.depth, (500, 5000), cv2.COLORMAP_JET)
            cv2.imshow(f"Camera {cam_id} - Depth", depth_img)
            save_img(depth_img, save_dir["Depth"], is_color=False)

        if cv2.waitKey(1) != -1:
            break


def main():
    base_dir = "C:/multi_sensor_sync_save_data_test/kinect"
    device_dict = enumerate_available_devices()
    save_dirs = create_save_folders(base_dir, device_dict)

    # === Setup Kinect configurations ===
    config1 = Config(wired_sync_mode=WiredSyncMode.MASTER, depth_mode=DepthMode.NFOV_UNBINNED)
    config2 = Config(wired_sync_mode=WiredSyncMode.SUBORDINATE, subordinate_delay_off_master_usec=200, depth_mode=DepthMode.NFOV_UNBINNED)
    config3 = Config(wired_sync_mode=WiredSyncMode.SUBORDINATE, subordinate_delay_off_master_usec=400, depth_mode=DepthMode.NFOV_UNBINNED)

    # === Initialize and start devices ===
    device_master = PyK4A(device_id=0, config=config1)
    device_sub_1 = PyK4A(device_id=1, config=config2)
    device_sub_2 = PyK4A(device_id=2, config=config3)

    devices = [device_master, device_sub_1, device_sub_2]
    for cam in devices:
        cam.start()

    # === Start capture threads ===
    threads = []
    for i, cam in enumerate(devices):
        t = threading.Thread(target=camera_capture_loop, args=(cam, save_dirs[i], i), daemon=True)
        t.start()
        threads.append(t)

    # === Wait for key press or Ctrl+C ===
    try:
        while True:
            if cv2.waitKey(1) != -1:
                break
    except KeyboardInterrupt:
        pass

    print("Stopping all cameras...")

    # === Cleanup ===
    for cam in devices:
        cam.stop()

    cv2.destroyAllWindows()
    print("Finished.")


if __name__ == "__main__":
    main()
