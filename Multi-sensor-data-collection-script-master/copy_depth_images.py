import os
import shutil
import re

# ================== CONFIG ==================
src_root = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect\sample_prepare_Nov_13_clean\N"
dest_root = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\N\kinect"

TARGET_IDS = ["001431512812", "001484412812", "001528512812"]

SRC_SUBFOLDER = "depth"          # source folder name
DEST_SUBFOLDER = "frames_depth"  # destination folder name (rename)

# run folder name can be: run_x  or  run_x_x, where x are digits
RUN_RE = re.compile(r"^run_(\d+)(?:_(\d+))?$")
# ============================================


def is_run_folder(name: str) -> bool:
    """Valid run folder: run_<digits> or run_<digits>_<digits>."""
    return RUN_RE.match(name) is not None


def safe_listdir(path: str):
    """List directory safely."""
    try:
        return os.listdir(path)
    except FileNotFoundError:
        return []
    except PermissionError:
        return []


def main():
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"src_root not found: {src_root}")
    if not os.path.isdir(dest_root):
        raise FileNotFoundError(f"dest_root not found: {dest_root}")

    for item in safe_listdir(src_root):
        run_src = os.path.join(src_root, item)
        if not (os.path.isdir(run_src) and is_run_folder(item)):
            continue

        print(f"\n=== Processing {item} ===")

        run_dest = os.path.join(dest_root, item)

        # ✅ Must match run folder name EXACTLY and must already exist in destination
        if not os.path.isdir(run_dest):
            print(f"⏭ Skipped: destination run does NOT exist (must match exactly) → {run_dest}")
            continue

        # Traverse timestamp folders under run_xxx
        for inner in safe_listdir(run_src):
            inner_path = os.path.join(run_src, inner)
            if not os.path.isdir(inner_path):
                continue

            for target_id in TARGET_IDS:
                src_depth = os.path.join(inner_path, target_id, SRC_SUBFOLDER)
                if not os.path.isdir(src_depth):
                    # not found in this timestamp folder
                    continue

                dest_depth = os.path.join(run_dest, target_id, DEST_SUBFOLDER)

                # ✅ Only copy if destination frames_depth does NOT exist
                if os.path.exists(dest_depth):
                    print(f"⏭ Skip existing: {dest_depth}")
                    continue

                os.makedirs(os.path.dirname(dest_depth), exist_ok=True)

                print(f"Copying {src_depth} → {dest_depth}")
                shutil.copytree(src_depth, dest_depth, dirs_exist_ok=True)

    print("\n✅ All done.")


if __name__ == "__main__":
    main()
