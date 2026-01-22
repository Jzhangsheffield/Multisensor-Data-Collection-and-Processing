import h5py
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import tqdm


base_path = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Nov_21_clean\MR" 



#### convert Unix timestamp
def timestamp_convert(t):
    us = int(np.rint(t * 1_000_000))
    sec = us // 1_000_000
    micro = us % 1_000_000
    return(f"{sec}_{micro:06d}")


### -----------save to videos .mp4---------------------------
# new_path = [os.path.join(base_path, f, f2, f3) for f in os.listdir(base_path) for f2 in os.listdir(os.path.join(base_path, f)) for f3 in os.listdir(os.path.join(base_path, f, f2)) if os.path.join(base_path, f, f2, f3).endswith(".hdf5")]
# print(new_path)

# # hdf5_files = [os.path.join(base_path, each, _file) for each in folders for _file in os.listdir(os.path.join(base_path, each)) if _file.endswith('.hdf5') ]
# # print(hdf5_files)
# # h5_path = r'C:\multi_sensor_sync_save_data_test\kinect\20250718_095527\001528512812.hdf5'
# # depth_img_folder = r'C:\multi_sensor_sync_save_data_test\kinect\20250718_095527\001528512812\depth'
# # os.makedirs(depth_img_folder, exist_ok=True)
# # output_path = h5_path.removesuffix(".hdf5")
# # os.makedirs(output_path, exist_ok=True)

hdf5_files = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".hdf5"):
            hdf5_files.append(os.path.join(root, file))

print(f"Found {len(hdf5_files)} HDF5 files")
print(hdf5_files)


for h5_path in tqdm.tqdm(hdf5_files):
    run_folder = os.path.basename(os.path.dirname(os.path.dirname(h5_path)))
    output_path = h5_path.removesuffix(".hdf5")
    os.makedirs(output_path, exist_ok=True)
    with h5py.File(h5_path, 'r') as file:
        print("keys:", list(file.keys()))
        key_list = list(file.keys())
        
        for _key in key_list:
            # if _key == "depth": #_key == "rgb" or _key == "depth":
            if _key == "rgb" or _key == "depth":
                u_output_path = os.path.join(output_path, (run_folder + '_' + _key + '.mp4'))
                # print(u_output_path)
                ds = file[_key]
                first_frame = ds[0]
                height = first_frame.shape[0]
                width = first_frame.shape[1]

                fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                fps = 30
                out = cv2.VideoWriter(u_output_path, fourcc, 30, (width, height), isColor=True)


                for i in range(len(ds)):
                    frame = ds[i]
                    if _key == "depth":
                        DEPTH_MIN = 0
                        DEPTH_MAX = 3000
                        colormap = cv2.COLORMAP_JET

                        valid_mask = frame > 0

                        # Clip depth to fixed range
                        frame_clipped = np.clip(frame, DEPTH_MIN, DEPTH_MAX)

                        # Normalize to 0–255 using fixed range
                        normalized = ((frame_clipped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)

                        # Apply colormap
                        frame = cv2.applyColorMap(normalized, colormap)

                        # Set invalid (0 depth) pixels to black
                        frame[~valid_mask] = [0, 0, 0]


                    out.write(frame)

                out.release()
                print(f"video saved to {u_output_path}")

# ###------------------------save to images .png--------------------------------------------
# new_path = [os.path.join(base_path, f, f2, f3, f4) for f in os.listdir(base_path) for f2 in os.listdir(os.path.join(base_path, f)) 
#             for f3 in os.listdir(os.path.join(base_path, f, f2)) for f4 in os.listdir(os.path.join(base_path, f, f2, f3)) if f4.endswith('.hdf5')]
# print(new_path)
# for h5_path in new_path:
#     output_path = h5_path.removesuffix(".hdf5")
#     os.makedirs(output_path, exist_ok=True)
    
#     with h5py.File(h5_path, 'r') as file:
#         print("keys:", list(file.keys()))
#         key_list = list(file.keys())
#         ds_timestamp = file['timestamp']
#         ds_timestamp_converted = pd.to_datetime(ds_timestamp, unit="s", utc=True).tz_convert("Europe/London")
#         ds_timestamp_converted = ds_timestamp_converted.round('us').sort_values().reset_index(drop=True)
#         ds_timestamp_foramtted = ds_timestamp_converted.strftime("%Y%m%d_%H%M%S_%f").tolist()
        
#         for _key in key_list:
#             if _key == "depth": # or _key == "depth":
#                 u_output_path = os.path.join(output_path, _key)
#                 os.makedirs(u_output_path, exist_ok=True)
#                 ds = file[_key]

#                 for i in range(len(ds)):
#                     frame = ds[i]
#                     timestamp = ds_timestamp_foramtted[i]
#                     # ts_converted = timestamp_convert(timestamp)

#                     # # visualize the depth image
#                     # if _key == "depth":
#                     #     DEPTH_MIN = 0
#                     #     DEPTH_MAX = 3000
#                     #     colormap = cv2.COLORMAP_JET

#                     #     valid_mask = frame > 0

#                     #     # Clip depth to fixed range
#                     #     frame_clipped = np.clip(frame, DEPTH_MIN, DEPTH_MAX)

#                     #     # Normalize to 0–255 using fixed range
#                     #     normalized = ((frame_clipped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)

#                     #     # Apply colormap
#                     #     frame = cv2.applyColorMap(normalized, colormap)

#                     #     # Set invalid (0 depth) pixels to black
#                     #     frame[~valid_mask] = [0, 0, 0]
#                     img_name = str(timestamp) + '.png'
#                     print(u_output_path, img_name)
#                     cv2.imwrite(os.path.join(u_output_path, img_name), frame)



###------------------------save to images .png no timestamp--------------------------------------------
# new_path = [os.path.join(base_path, f, f2, f3, f4) for f in os.listdir(base_path) for f2 in os.listdir(os.path.join(base_path, f)) 
#             for f3 in os.listdir(os.path.join(base_path, f, f2)) for f4 in os.listdir(os.path.join(base_path, f, f2, f3)) if f4.endswith('.hdf5')]
# print(new_path)

# for h5_path in new_path:
#     output_path = h5_path.removesuffix(".hdf5")
#     os.makedirs(output_path, exist_ok=True)
    
#     with h5py.File(h5_path, 'r') as file:
#         print("keys:", list(file.keys()))
#         key_list = list(file.keys())

#         # 获取路径中的信息
#         parts = Path(h5_path).parts
#         participant = parts[-4]
#         run = parts[-3]
#         device = Path(h5_path).stem

#         #print(participant, run, device)
#         for _key in key_list:
#             if _key == "rgb": # or _key == "depth":
#                 u_output_path = os.path.join(output_path, _key)
#                 os.makedirs(u_output_path, exist_ok=True)
#                 ds = file[_key]

#                 for i in range(len(ds)):
#                     frame = ds[i]

#                     # visualize the depth image
#                     if _key == "depth":
#                         DEPTH_MIN = 0
#                         DEPTH_MAX = 3000
#                         colormap = cv2.COLORMAP_JET

#                         valid_mask = frame > 0

#                         # Clip depth to fixed range
#                         frame_clipped = np.clip(frame, DEPTH_MIN, DEPTH_MAX)

#                         # Normalize to 0–255 using fixed range
#                         normalized = ((frame_clipped - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 255).astype(np.uint8)

#                         # Apply colormap
#                         frame = cv2.applyColorMap(normalized, colormap)

#                         # Set invalid (0 depth) pixels to black
#                         frame[~valid_mask] = [0, 0, 0]
                        
#                     img_name = participant + '_' + run + '_' +  device + '_' + f'{i:06d}' + '.png'
#                     print(u_output_path, img_name)
#                     cv2.imwrite(os.path.join(u_output_path, img_name), frame)




