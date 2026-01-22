import os
import cv2
import h5py
import pandas as pd
import numpy as np


# # blurred_videos = [r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\all_blurred_videos_side_4k\run1_rgb_b.mp4",
# #                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\all_blurred_videos_side_4k\run2_rgb_b.mp4",
# #                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\all_blurred_videos_side_4k\run3_rgb_b.mp4",
# #                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\all_blurred_videos_side_4k\run4_rgb_b.mp4",
# #                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\all_blurred_videos_side_4k\run5_rgb_b.mp4",
# #                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\all_blurred_videos_side_4k\run6_rgb_b.mp4",
# #                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\all_blurred_videos_side_4k\run7_rgb_b.mp4",
# #                   ]

# # output_dir = [r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_1\20250807_164249\001528512812\blurred_rgb",
# #               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_2\20250807_164658\001528512812\blurred_rgb",
# #               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_3\20250807_165159\001528512812\blurred_rgb",
# #               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_4\20250807_165533\001528512812\blurred_rgb",
# #               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_5\20250807_165945\001528512812\blurred_rgb",
# #               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_6\20250807_170338\001528512812\blurred_rgb",
# #               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_7\20250807_171018\001528512812\blurred_rgb",
# #               ]

# # h5_files = [r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_1\20250807_164249\001528512812.hdf5",
# #             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_2\20250807_164658\001528512812.hdf5",
# #             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_3\20250807_165159\001528512812.hdf5",
# #             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_4\20250807_165533\001528512812.hdf5",
# #             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_5\20250807_165945\001528512812.hdf5",
# #             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_6\20250807_170338\001528512812.hdf5",
# #             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_7\20250807_171018\001528512812.hdf5"]

# blurred_videos = [r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_1\20250807_164249\001484412812\rgb.mp4",
#                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_2\20250807_164658\001484412812\rgb.mp4",
#                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_3\20250807_165159\001484412812\rgb.mp4",
#                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_4\20250807_165533\001484412812\rgb.mp4",
#                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_5\20250807_165945\001484412812\rgb.mp4",
#                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_6\20250807_170338\001484412812\rgb.mp4",
#                   r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_7\20250807_171018\001484412812\rgb.mp4",
#               ]

# output_dir = [r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_1\20250807_164249\001484412812\rgb",
#               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_2\20250807_164658\001484412812\rgb",
#               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_3\20250807_165159\001484412812\rgb",
#               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_4\20250807_165533\001484412812\rgb",
#               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_5\20250807_165945\001484412812\rgb",
#               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_6\20250807_170338\001484412812\rgb",
#               r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_7\20250807_171018\001484412812\rgb",
#               ]

# h5_files = [r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_1\20250807_164249\001484412812.hdf5",
#             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_2\20250807_164658\001484412812.hdf5",
#             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_3\20250807_165159\001484412812.hdf5",
#             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_4\20250807_165533\001484412812.hdf5",
#             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_5\20250807_165945\001484412812.hdf5",
#             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_6\20250807_170338\001484412812.hdf5",
#             r"I:\7_Aug_test_run\kinect\sample_prepare_Aug_07_clean\M\run_7\20250807_171018\001484412812.hdf5"]

# img_ext = '.png'

# for video, h5, out_dir in zip(blurred_videos, h5_files, output_dir):
#     os.makedirs(out_dir, exist_ok=True)
    
#     with h5py.File(h5, 'r') as f:
#         ds_timestamp = f["timestamp"][:]
        
#     ds_timestamp_converted = pd.to_datetime(ds_timestamp, unit="s", utc=True).tz_convert("Europe/London")
#     ds_timestamp_converted = ds_timestamp_converted.round('us')
#     ds_timestamp_foramtted = ds_timestamp_converted.strftime("%Y%m%d_%H%M%S_%f").tolist()
    
#     cap = cv2.VideoCapture(str(video))
#     if not cap.isOpened():
#         raise RuntimeError(f"无法打开视频：{video}")

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     zpad = len(str(total_frames - 1))
    
#     count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         cv2.imwrite(os.path.join(out_dir, f"{ds_timestamp_foramtted[count]}{img_ext}"), frame)
#         count += 1

#     cap.release()
#     print(f"{video} done!!")
    



#-------------------------NEW CODE ------------------------
import os
import cv2
import pandas as pd

BASE_DIR = [#r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\MR\kinect", 
            r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\N\kinect",
            #r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\M\kinect", 
            #r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\kinect"
            ]

IMG_EXT = ".jpg"

def find_rgb_video(files):
    """Return the first file that contains 'rgb' and is a video."""
    for f in files:
        lower = f.lower()
        if "rgb_front_b" in lower and lower.endswith((".mp4", ".avi", ".mov", ".mkv")):
            return f
    return None

def find_csv(files):
    """Return the first CSV file."""
    for f in files:
        if f.lower().endswith(".csv"):
            return f
    return None

for DIR in BASE_DIR:

    # iterate all run folders
    for run_folder in os.listdir(DIR):
        if run_folder in ["run_13", "run_14", "run_15"]:
            run_path = os.path.join(DIR, run_folder)

            if not os.path.isdir(run_path):
                continue
            if not run_folder.startswith("run_"):
                continue

            print(f"\nProcessing {run_folder}")

            # iterate camera folders inside each run
            for cam_folder in os.listdir(run_path):
                if cam_folder == "001431512812":
                    cam_path = os.path.join(run_path, cam_folder)

                    if not os.path.isdir(cam_path):
                        continue

                    print(f"  Camera folder: {cam_folder}")

                    files = os.listdir(cam_path)

                    # -------------------------
                    # FIND THE CSV FILE
                    # -------------------------
                    csv_file = find_csv(files)
                    if csv_file is None:
                        print(f"    ❌ No CSV file found in {cam_path}")
                        continue

                    csv_path = os.path.join(cam_path, csv_file)

                    # -------------------------
                    # FIND THE RGB VIDEO FILE
                    # -------------------------
                    rgb_file = find_rgb_video(files)
                    if rgb_file is None:
                        print(f"    ❌ No RGB video found in {cam_path}")
                        continue

                    rgb_path = os.path.join(cam_path, rgb_file)

                    # output frames directory
                    out_dir = os.path.join(cam_path, "frames_rgb_blured")
                    if os.path.exists(out_dir) and os.listdir(out_dir):
                        print(f"Folder exists and not empty, skip this folder.")
                    else:
                        os.makedirs(out_dir, exist_ok=True)

                    # -------------------------
                    # STEP 1 — Read timestamps
                    # -------------------------
                    df = pd.read_csv(csv_path)
                    timestamps = df["timestamp"].tolist()

                    print(f"    ✔ Loaded {len(timestamps)} timestamps")

                    # -------------------------
                    # STEP 2 — Open RGB video
                    # -------------------------
                    cap = cv2.VideoCapture(rgb_path)
                    if not cap.isOpened():
                        print(f"    ❌ Cannot open video: {rgb_path}")
                        continue

                    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"    ✔ Video has {video_frames} frames")

                    # -------------------------
                    # STEP 3 — MATCH LENGTHS
                    # -------------------------
                    usable_count = min(len(timestamps), video_frames)
                    if len(timestamps) != video_frames:
                        print(f"    ⚠ MISMATCH — using first {usable_count}, dropping the rest")

                    timestamps = timestamps[:usable_count]

                    # -------------------------
                    # STEP 4 — Extract frames
                    # -------------------------
                    index = 0
                    while index < usable_count:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        ts = timestamps[index]
                        out_path = os.path.join(out_dir, f"{ts}{IMG_EXT}")
                        cv2.imwrite(out_path, frame)

                        index += 1

                    cap.release()
                    print(f"    ✔ Saved {index} frames to {out_dir}")

            print("\n🎉 ALL DONE — FRAMES EXTRACTED SUCCESSFULLY!")

    
#-----------------删除文件夹----------------------
        
# import os
# import shutil

# root_dir = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured"  # 改成你的目标路径

# deleted_count = 0

# for root, dirs, files in os.walk(root_dir):
#     if "frames_rgb_blured" in dirs:
#         target_dir = os.path.join(root, "frames_rgb_blured")
#         shutil.rmtree(target_dir)
#         deleted_count += 1
#         print(f"已删除: {target_dir}")

# print(f"\n共删除 {deleted_count} 个 frames_rgb_blured 文件夹")

        