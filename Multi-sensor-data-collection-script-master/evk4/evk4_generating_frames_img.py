import numpy as np
from metavision_core.event_io.raw_reader import RawReader
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import os
from datetime import datetime, timedelta

def create_image_from_events(events, height, width):
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    img[events['y'], events['x']] = 255 * events['p'][:, None]
    return img


# # # #------------------------------------写入视频-------------------------------------------------------------------
base_dir = r'E:\multi_sensor_sync_save_data_test\evk4_test\sample_prepare_Aug_08_clean\M'
file_dir = [os.path.join(base_dir, f, f2) for f in os.listdir(base_dir) for f2 in os.listdir(os.path.join(base_dir, f))]
print(file_dir)
for file in file_dir:
    if file.endswith('.raw'):
        raw_stream = RawReader(file)  # use empty string to open a camera
        print(file)
        save_dir = file[:-4]
        height, width = raw_stream.get_size()
        save_dir = save_dir + '.mp4'
        print(save_dir)

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        fps = 30
        video_writer = cv2.VideoWriter(save_dir, fourcc, fps, (width, height))

        while not raw_stream.is_done():
            events = raw_stream.load_delta_t(30000)
            frame_offset_us = int(events['t'][-1])
            delta = timedelta(microseconds=frame_offset_us)

            im = create_image_from_events(events, height, width)

            video_writer.write(im)


        video_writer.release()
        print(f"Video saved to {save_dir}")


# # #----------------------------------写入图片----------------------------------------------------
# raw_stream = RawReader(r"E:\multi_sensor_sync_save_data_test\evk4\recording_250718_100012.raw")  # use empty string to open a camera
# height, width = raw_stream.get_size()
# start_time = "250718_095608_000000"
# start_time = datetime.strptime(start_time, "%y%m%d_%H%M%S_%f")
# save_dir = r"E:\multi_sensor_sync_save_data_test\evk4\recording_250718_100012"
# os.makedirs(save_dir, exist_ok=True)

# while not raw_stream.is_done():
#     events = raw_stream.load_delta_t(50000)
#     frame_offset_us = int(events['t'][-1])
#     delta = timedelta(microseconds=frame_offset_us)
#     frame_timestamp = start_time + delta
    
#     img_name = frame_timestamp.strftime("%y%m%d_%H%M%S_%f") + ".png"
#     im = create_image_from_events(events, height, width)
#     cv2.imwrite(os.path.join(save_dir, img_name), im)


# # # # ---------------------------用于连续可视化显示图像-----------------------------------------
# # 初始化图像显示
# fig, ax = plt.subplots()
# im_display = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8), animated=True)
# plt.axis('off')

# # 更新函数
# def update(frame):
#     if raw_stream.is_done():
#         anim.event_source.stop()
#         return im_display

#     events = raw_stream.load_delta_t(50000)  # 微秒 = 0.1秒
#     img = create_image_from_events(events, height, width)
#     im_display.set_data(img)
#     return [im_display]

# # 动画
# anim = FuncAnimation(fig, update, interval=5, blit=True)  # interval 单位是 ms
# plt.show()