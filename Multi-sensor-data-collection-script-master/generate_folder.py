import os
import shutil

# generate folder
# source_root = r""
# dest_root = r""

# for i in range(1, 37):
#     folder = f"run_{i}"

#     os.makedirs(os.path.join(dest_root, folder), exist_ok=True)

# print(f"folder generated!")



# List of folders you want to copy
folders_to_copy = [r"001431512812", r"001484412812", r"001528512812"]

import os
import shutil

src_root = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Oct_18_clean\J"
dest_root = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\kinect"

# helper: path is run folder like "run_1"
def is_run_folder(name):
    return name.startswith("run_") and name.split("_")[-1].isdigit()

def is_empty_folder(path: str) -> bool:
    """Return True if folder exists and is empty, or does not exist at all."""
    return (not os.path.exists(path)) or (len(os.listdir(path)) == 0)

# For Kinect: loop through items under the source J folder 
for item in os.listdir(src_root):
    run_src = os.path.join(src_root, item)

    # check if it is a run folder
    if os.path.isdir(run_src) and is_run_folder(item):
        print(f"\n=== Processing {item} ===")

        # matching destination run folder
        run_dest = os.path.join(dest_root, item)
        if not is_empty_folder(run_dest):
            print(f"❌ Skipping {item}: destination is NOT empty → {run_dest}")
            continue  # Skip this run folder entirely

        # ensure destination run folder exists
        os.makedirs(run_dest, exist_ok=True)

        # inside each run folder, find the timestamp folder (like 20251113_165628)
        for inner in os.listdir(run_src):
            inner_path = os.path.join(run_src, inner)
            if os.path.isdir(inner_path):  # ensure folder
                # copy *its* subfolders (not the timestamp folder itself)
                for sub in os.listdir(inner_path):
                    sub_src = os.path.join(inner_path, sub)
                    if os.path.isdir(sub_src):  # only copy folders
                        sub_dest = os.path.join(run_dest, sub)

                        print(f"Copying {sub_src} → {sub_dest}")
                        shutil.copytree(sub_src, sub_dest, dirs_exist_ok=True)


# # For vlp16: loop through items under the source J folder
# for item in os.listdir(src_root):
#     run_src = os.path.join(src_root, item)

#     # Check if it's a run folder
#     if os.path.isdir(run_src) and is_run_folder(item):
#         print(f"\n=== Processing {item} ===")

#         run_dest = os.path.join(dest_root, item)

#         # Skip run folders in destination that are NOT empty
#         if not is_empty_folder(run_dest):
#             print(f"❌ Skipping {item}: destination is NOT empty → {run_dest}")
#             continue

#         # Ensure destination exists (empty)
#         os.makedirs(run_dest, exist_ok=True)
#         print(f"✓ Destination is empty, copying into: {run_dest}")

#         # Find timestamp folder(s)
#         for inner in os.listdir(run_src):
#             timestamp_path = os.path.join(run_src, inner)

#             # Only process timestamp folders
#             if os.path.isdir(timestamp_path):
#                 # Now copy only files inside timestamp folder
#                 for fname in os.listdir(timestamp_path):
#                     file_src = os.path.join(timestamp_path, fname)

#                     if os.path.isfile(file_src):  # ensure file
#                         file_dest = os.path.join(run_dest, fname)

#                         print(f"Copying {file_src} → {file_dest}")
#                         shutil.copy2(file_src, file_dest)  # preserves metadata






