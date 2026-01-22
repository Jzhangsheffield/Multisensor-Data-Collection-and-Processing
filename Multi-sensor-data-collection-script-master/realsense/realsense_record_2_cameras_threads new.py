import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import multiprocessing

def configure_pipeline(serial_number, width=640, height=480, fps=30,
                       print_default_values=False, set_control_values=False,
                       depth_control_values={}, rgb_control_values={}, param_save_path=None):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    color_sensor = device.first_color_sensor()

    if print_default_values:
        depth_supported_options = depth_sensor.get_supported_options()
        color_supported_options = color_sensor.get_supported_options()
        print(f"Supported options for depth sensor: {depth_supported_options}")
        print(f"Supported options for color sensor: {color_supported_options}")
        for opt in depth_supported_options:
            print(f"{opt} value for depth sensor: {depth_sensor.get_option(opt)}")
        for opt in color_supported_options:
            print(f"{opt} value for color sensor: {color_sensor.get_option(opt)}")
            
        if param_save_path:
            save_path = os.path.join(param_save_path, f"camera_control_param_{serial_number}.txt")
            with open(save_path, "w") as f:
                for opt in depth_supported_options:
                    f.writelines(f"{opt} value for depth sensor: {depth_sensor.get_option(opt)} \n")
                f.writelines("========================\n")
                for opt in color_supported_options:
                    f.writelines(f"{opt} value for color sensor: {color_sensor.get_option(opt)} \n")
                f.writelines("========================\n")
            
                f.writelines(f"Depth scale: {depth_sensor.get_depth_scale()}")
            
    if set_control_values:
        if depth_control_values :
            for opt, value in depth_control_values.items():
                _opt = getattr(rs.option, opt)
                depth_sensor.set_option(_opt, value)
        if rgb_control_values:
            for opt, value in rgb_control_values.items():
                _opt = getattr(rs.option, opt)
                color_sensor.set_option(_opt, value)

    return pipeline


def create_colorizer(min_dist, max_dist):
    colorizer = rs.colorizer(0.0)
    colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
    colorizer.set_option(rs.option.min_distance, min_dist)
    colorizer.set_option(rs.option.max_distance, max_dist)
    return colorizer


def distance_threshold_filter(min_dist=0.0, max_dist=5.0):
    filter_ = rs.threshold_filter()
    filter_.set_option(rs.option.min_distance, min_dist)
    filter_.set_option(rs.option.max_distance, max_dist)
    return filter_, min_dist, max_dist


def save_image(image, serial_number, image_type, save_dir):
    timestamp = datetime.datetime.now().timestamp()
    filename = f"{serial_number}_{image_type}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"[{serial_number}] 图像已保存：{filepath}")


def camera_process(serial_number, save_dirs, min_dist, max_dist, param_save_dir):
    pipeline = configure_pipeline(
        serial_number,
        print_default_values=True,
        set_control_values=True,
        param_save_path=param_save_dir,
        depth_control_values={"exposure": 8800.0, "visual_preset": 1},
    )

    threshold_filter, _, _ = distance_threshold_filter(min_dist, max_dist)
    colorizer = create_colorizer(min_dist, max_dist)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            #thresholded_depth_frame = threshold_filter.process(depth_frame)
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # colorizer_depth = np.asanyarray(colorizer.colorize(thresholded_depth_frame).get_data())

            # Save
            save_image(color_image, serial_number, "RGB", save_dirs["RGB"])
            save_image(depth_image, serial_number, "Orginal Depth", save_dirs["Org Depth"])
            # save_image(colorizer_depth, serial_number, "thresholded", save_dirs["Depth"])

            # # Display
            ## cv2.imshow(f"Color - {serial_number}", color_image)
            ## cv2.imshow(f"Depth - {serial_number}", colorizer_depth)

            ## if cv2.waitKey(1) & 0xFF == ord('q'):
            ##     stop_event.set()
            ##    break

    except KeyboardInterrupt:
        print(f"Stopping camera process {serial_number}...")


    pipeline.stop()
    print(f"[{serial_number}] 线程已停止")


def get_connected_devices():
    context = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in context.query_devices()]


def main():
    serial_numbers = get_connected_devices()
    if not serial_numbers:
        print("未检测到相机")
        return

    recording_start_time = datetime.datetime.now()

    base_save_dir = "C:/multi_sensor_sync_save_data_test/kinect/"
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(base_save_dir, date_str)
    os.makedirs(base_save_dir, exist_ok=True)

    camera_dirs = {}
    for serial in serial_numbers:
        dirs = {
            "RGB": os.path.join(base_save_dir, f"camera_{serial}", "RGB"),
            "Depth": os.path.join(base_save_dir, f"camera_{serial}", "Depth"),
            "Org Depth": os.path.join(base_save_dir, f"camera_{serial}", "Org Depth")
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        camera_dirs[serial] = dirs

    # Start threads
    processes = []
    for serial in serial_numbers:
        p = multiprocessing.Process(target=camera_process, args=(serial, camera_dirs[serial], 0.0, 5.0, base_save_dir))
        p.start()
        processes.append(p)

    # Wait for threads to finish
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Terminating all camera processes...")
        for p in processes:
            p.terminate()
            p.join()

        print("All camera processes terminated.")

    duration = datetime.datetime.now() - recording_start_time
    print(f"录制总用时为: {duration}")


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")  # Important for Windows
    main()
