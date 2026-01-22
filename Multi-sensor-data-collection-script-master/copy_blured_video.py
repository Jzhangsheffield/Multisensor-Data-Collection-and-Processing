import os
import shutil

# 修改为你自己的路径
A_root = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\blurred videos"
B_root = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured"

persons = ["N", "MR", "M", "J"]

for person in persons:
    A_person_dir = os.path.join(A_root, person)
    B_person_dir = os.path.join(B_root, person, "kinect")

    if not os.path.exists(A_person_dir) or not os.path.exists(B_person_dir):
        print(f"跳过 {person}，路径不存在")
        continue

    # 遍历 A 中的视频
    for video in os.listdir(A_person_dir):
        if not video.endswith(".mp4"):
            continue

        # 例子:
        # run_1_rgb_front.mp4 -> run_1
        # run_1_34_rgb_front.mp4 -> run_1_34
        run_name = video.replace("_rgb_front_b.mp4", "")

        src_video_path = os.path.join(A_person_dir, video)

        # 在 B 中寻找对应 run 目录
        run_dir = os.path.join(B_person_dir, run_name)
        target_dir = os.path.join(run_dir, "001431512812")

        if not os.path.exists(target_dir):
            print(f"目标路径不存在，跳过: {target_dir}")
            continue

        dst_video_path = os.path.join(target_dir, video)

        shutil.copy2(src_video_path, dst_video_path)
        print(f"已复制: {src_video_path} -> {dst_video_path}")
