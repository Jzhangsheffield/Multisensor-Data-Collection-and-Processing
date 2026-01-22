import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime

# 配置相机流水线
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
    _colorizer = rs.colorizer(0.0)
    _colorizer.set_option(rs.option.histogram_equalization_enabled, 0)
    _colorizer.set_option(rs.option.min_distance, min_dist)
    _colorizer.set_option(rs.option.max_distance, max_dist)
    
    return _colorizer



def distance_threshold_filter(min_dist=0.0, max_dist=5.0):
    _min_dist = min_dist
    _max_dist = max_dist
    threshold_filter = rs.threshold_filter()
    threshold_filter.set_option(rs.option.min_distance, min_dist)
    threshold_filter.set_option(rs.option.max_distance, max_dist)
    return threshold_filter, _min_dist, _max_dist
    

# 获取连接的设备序列号
def get_connected_devices():
    context = rs.context()
    devices = context.query_devices()
    serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in devices]
    return serial_numbers

# 保存图像到文件
def save_image(image, serial_number, image_type, save_dir):
    # timestamp = datetime.datetime.now().strftime('%Y%m%d %H_%M_%S_%f')
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamp = datetime.datetime.now().timestamp()
    filename = f"{serial_number}_{image_type}_{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"图像已保存：{filepath}")

# 主函数
def main():
    serial_numbers = get_connected_devices()
    if len(serial_numbers) < 1:
        print("未检测到足够的相机。请确保连接了至少一台 D435i 相机。")
        return

    # 开始录制的时间:
    recording_start_time = datetime.datetime.now()
    
    # 为每台相机创建单独的保存目录
    base_save_dir = "F:/realsense/test"
    date = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    base_save_dir = os.path.join(base_save_dir, date)
    os.makedirs(base_save_dir, exist_ok=True)
    camera_dirs = {}
    for serial in serial_numbers:
        camera_dir = {"RGB": os.path.join(base_save_dir, f"camera_{serial}", f"RGB"), "Org Depth": os.path.join(base_save_dir, f"camera_{serial}", f"Org Depth"), "Depth": os.path.join(base_save_dir, f"camera_{serial}", f"Depth")}
        os.makedirs(camera_dir["RGB"], exist_ok=True)
        os.makedirs(camera_dir["Depth"], exist_ok=True)
        os.makedirs(camera_dir["Org Depth"], exist_ok=True)
        camera_dirs[serial] = camera_dir
        
    threshold_filter, _min_dist, _max_dist = distance_threshold_filter(max_dist=5.0)

    # 配置两台相机的流水线
    pipelines = [configure_pipeline(sn, print_default_values=True, set_control_values=True, 
                                    depth_control_values={"exposure": 8800.0, "visual_preset": 1}, 
                                    param_save_path=base_save_dir) for sn in serial_numbers]

    try:
        while True:
            frames = []
            for pipeline in pipelines:
                frames.append(pipeline.wait_for_frames())
                colorizer = create_colorizer(_min_dist, _max_dist)
                # depth_to_disparity = rs.disparity_transform(True)
                # disparity_to_depth = rs.disparity_transform(False)


            for i, frame_set in enumerate(frames):
                serial_number = serial_numbers[i]

                # 获取对齐的深度和颜色帧, 一般不要对齐，影响图片质量。
                # align = rs.align(rs.stream.depth)
                # aligned_frames = align.process(frame_set)
                # depth_frame = aligned_frames.get_depth_frame()
                # depth_frame = threshold_filter.process(depth_frame)
                # color_frame = aligned_frames.get_color_frame()
                
                depth_frame = frame_set.get_depth_frame()
                thresholded_depth_frame = threshold_filter.process(depth_frame)
                color_frame = frame_set.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # 将图像转换为 numpy 数组
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                colorizer_depth = np.asanyarray(colorizer.colorize(thresholded_depth_frame).get_data())

                # 深度图数据格式转换，uint16 → uint8
                # depth_8bit = cv2.convertScaleAbs(depth_image, alpha=0.1)
                # depth_8bit_inverted = 255 - depth_8bit
                # depth_colormap = cv2.applyColorMap(depth_8bit_inverted, cv2.COLORMAP_JET)

                # 显示图像
                cv2.imshow(f'Camera {i+1} - Color', color_image)
                cv2.imshow(f'Camera {i+1} - Depth', colorizer_depth)


                save_dir = camera_dirs[serial_number]
                save_image(color_image, serial_number, "RGB", save_dir["RGB"])
                save_image(depth_image, serial_number, "Orginal Depth", save_dir["Org Depth"])
                save_image(colorizer_depth, serial_number, "thresholded", save_dir["Depth"])

            # 按 'q' 键退出
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()
            # 结束录制的时间
        recording_end_time = datetime.datetime.now()

        # 录制用时:
        duration = recording_end_time - recording_start_time
        print(f"录制总用时为: {duration}")
        

# print(get_connected_devices())

if __name__ == '__main__':
    main()
