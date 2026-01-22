import os
import hashlib

BASE_DIRS = [
             r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Nov_13_clean\J",
             r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Nov_13_clean\N",
             r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Nov_18_clean\M",
             r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Nov_21_clean\MR",
             r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Oct_18_clean\J",
             r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Oct_24_clean\M",
             r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Oct_27_clean\M",
             r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Oct_29_clean\MR"
             ]

# Camera folders → suffix mapping
CAM_MAP = {
    "001484412812": "back",
    "001431512812": "front",
    "001528512812": "side"
}

# Detect RGB video: no "depth" in name
def is_rgb(name):
    return "depth" not in name.lower() and name.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))

# Detect depth video: filename contains "depth"
def is_depth(name):
    return "depth" in name.lower() and name.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))

# Compute hash for duplicate detection
def file_hash(path, chunk_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

for BASE_DIR in BASE_DIRS:
    for run_folder in os.listdir(BASE_DIR):
        run_path = os.path.join(BASE_DIR, run_folder)
        time_folder = os.listdir(run_path)[0]
        run_path = os.path.join(run_path, time_folder)
        if not os.path.isdir(run_path) or not run_folder.startswith("run_"):
            continue

        print(f"\nProcessing run folder: {run_folder}")

        # loop through camera subfolders
        for cam_id, cam_suffix in CAM_MAP.items():
            cam_path = os.path.join(run_path, cam_id)

            if not os.path.isdir(cam_path):
                continue

            print(f"\n  Camera {cam_id} → {cam_suffix}")

            files = os.listdir(cam_path)

            # separate RGB and depth videos
            rgb_videos = [f for f in files if is_rgb(f)]
            depth_videos = [f for f in files if is_depth(f)]

            # --------------------------
            # Handle RGB videos
            # --------------------------
            if rgb_videos:
                print(f"    Found RGB videos: {rgb_videos}")

                # Deduplicate based on file hash
                hashes = {}
                duplicates = []

                for f in rgb_videos:
                    full_path = os.path.join(cam_path, f)
                    h = file_hash(full_path)

                    if h in hashes:  # duplicate found
                        duplicates.append(full_path)
                    else:
                        hashes[h] = full_path

                # Delete duplicates
                for d in duplicates:
                    print(f"    Deleting duplicate RGB: {d}")
                    os.remove(d)

                # Rename the remaining RGB video
                kept_rgb = list(hashes.values())[0]
                ext = os.path.splitext(kept_rgb)[1]

                new_rgb_name = f"{run_folder}_rgb_{cam_suffix}{ext}"
                new_rgb_path = os.path.join(cam_path, new_rgb_name)

                print(f"    Renaming RGB: {kept_rgb} → {new_rgb_path}")
                os.rename(kept_rgb, new_rgb_path)

            # --------------------------
            # Handle DEPTH videos
            # --------------------------
            if depth_videos:
                print(f"    Found depth videos: {depth_videos}")

                # assume only one depth file
                depth_path = os.path.join(cam_path, depth_videos[0])
                ext = os.path.splitext(depth_videos[0])[1]

                new_depth_name = f"{run_folder}_depth_{cam_suffix}{ext}"
                new_depth_path = os.path.join(cam_path, new_depth_name)

                print(f"    Renaming depth: {depth_path} → {new_depth_path}")
                os.rename(depth_path, new_depth_path)

    print("\nAll runs finished.")
