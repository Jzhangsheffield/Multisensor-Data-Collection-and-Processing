import os
import datetime

import cv2
import numpy as np

import pyk4a
from pyk4a import Config, PyK4A, WiredSyncMode, DepthMode, connected_device_count
from helpers import colorize

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
        device_dir = {"RGB": os.path.join(updated_base_dir, item[1], "RGB"), "Depth": os.path.join(updated_base_dir, item[1], "Depth")}
        os.makedirs(device_dir["RGB"])
        os.makedirs(device_dir["Depth"])
        device_dirs[item[0]] = device_dir

    print(f"All folders created")

    return device_dirs


def save_img(img, save_dir):
    timestamp = datetime.datetime.now().timestamp()
    img_name = f"{timestamp}.png"
    img_path = os.path.join(save_dir, img_name)
    cv2.imwrite(img_path, img)
    # print(f"image saved to {img_path}")


def main():

    device_dict = enumerate_available_devices()
    save_dirs = create_save_folders(f"C:/multi_sensor_sync_save_data_test/kinect", device_dict)

    # Initialize all 3 cameras
    devices = []

    config1 = Config(wired_sync_mode=WiredSyncMode.MASTER, depth_mode=DepthMode.NFOV_UNBINNED)
    config2 = Config(wired_sync_mode=WiredSyncMode.SUBORDINATE, subordinate_delay_off_master_usec=200, depth_mode=DepthMode.NFOV_UNBINNED)
    config3 = Config(wired_sync_mode=WiredSyncMode.SUBORDINATE, subordinate_delay_off_master_usec=400, depth_mode=DepthMode.NFOV_UNBINNED)

    device_master = PyK4A(device_id=0, config=config1)
    device_subordinate_1 = PyK4A(device_id=1, config=config2)
    device_subordinate_2 = PyK4A(device_id=2, config=config3)
    device_master.start()
    device_subordinate_1.start()
    device_subordinate_2.start()

    devices.extend([device_master, device_subordinate_1, device_subordinate_2])
        

    while True:
        for i, cam in enumerate(devices):
            capture = cam.get_capture()
            in_timestamp = capture.depth_timestamp_usec
            sys_timestamp = capture.depth_system_timestamp_nsec
            print(f"camera {i}")
            print(f"internal timestamp: {in_timestamp}")
            print(f"system timestamp: {sys_timestamp}")
            print(f"-----------------")


            if capture.color is not None:
                color_img = capture.color[:, :, :3]
                cv2.imshow(f"Camera {i} - Color", color_img)

            if capture.depth is not None:
                depth_img = colorize(capture.depth, (0, 5000), cv2.COLORMAP_JET)
                # depth_img = capture.depth
                cv2.imshow(f"Camera {i} - Depth", depth_img)

            # save_dir = save_dirs[i]
            # save_img(color_img, save_dir["RGB"])
            # save_img(depth_img, save_dir["Depth"])

        key = cv2.waitKey(1)
        if key != -1:
            break

    cv2.destroyAllWindows()
    for cam in devices:
        cam.stop()


if __name__ == "__main__":
    main()