#导入所需要的包
import pyrealsense2 as rs
import numpy as np
import cv2
import time

#初始化管道和配置
pipeline = rs.pipeline()
config = rs.config()

#config.enable_record_to_file('test_record_file_2.bag')
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#开始进行流式传输
profile = pipeline.start(config)

#获取当前的深度映射系数
device = profile.get_device()
depth_sensor = device.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is:", depth_scale)

clipping_distance_in_meter = 1
clipping_distance = clipping_distance_in_meter / depth_scale

#深度图片与RGB图片对齐的配置
align = rs.align(rs.stream.color)

#设置保存视频的参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('aligned_color_video.mp4', fourcc, 30.0, (1280, 480))

try:
    start = time.time()
    
    while time.time() - start < 10:
    #for _ in range(4):

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frames = aligned_frames.get_depth_frame()
        color_frames = aligned_frames.get_color_frame()
        
        if not aligned_depth_frames or not color_frames:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frames.get_data())
        color_image = np.asanyarray(color_frames.get_data())
        
        grey_scale = 153
        
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d < 0), grey_scale, color_image)
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        #print(depth_colormap.shape, color_image.shape)
        #depth_colormaps = np.dstack((depth_colormap, depth_colormap, depth_colormap))
        
       
        images = np.hstack((depth_colormap, background_removed))
        
        video_writer.write(images)
        
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        #cv2.imwrite("aligned_images.png", images)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
            
finally:
    cv2.destroyAllWindows()
    video_writer.release()
    pipeline.stop()
    
        