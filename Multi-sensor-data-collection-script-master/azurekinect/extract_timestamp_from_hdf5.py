import os
import glob
import h5py
import pandas as pd
import tqdm 

root_h5_dir = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\kinect"

# ---------------------------------------------------------
# Scan for all HDF5 files under the run directory
# ---------------------------------------------------------
h5_files = sorted(glob.glob(os.path.join(root_h5_dir, "**", "*.hdf5"), recursive=True))

print("Found HDF5 files:")
for f in h5_files:
    print("   ", f)

# ---------------------------------------------------------
# For each HDF5 file, find its matching RGB video folder
# ---------------------------------------------------------
for h5 in tqdm.tqdm(h5_files, desc=r"extract timestamps"):

    # Extract the numeric ID (e.g., "001431512812")
    base_id = os.path.splitext(os.path.basename(h5))[0]

    # The corresponding RGB folder
    video_folder = os.path.join(os.path.dirname(h5), base_id)

    if not os.path.isdir(video_folder):
        print(f"⚠️ RGB directory not found for {h5}: {video_folder}")
        continue

    # CSV output filename
    csv_out_path = os.path.join(video_folder, f"{base_id}_timestamp.csv")

    print(f"\nProcessing:")
    print(f"  HDF5: {h5}")
    print(f"  CSV:  {csv_out_path}")

    # ---------------------------------------------------------
    # Read timestamps from HDF5
    # ---------------------------------------------------------
    with h5py.File(h5, "r") as f:
        ds_timestamp = f["timestamp"][:]

    # Convert timestamps
    ds_timestamp_converted = (
        pd.to_datetime(ds_timestamp, unit="s", utc=True)
          .tz_convert("Europe/London")
          .round("us")
    )

    # Format timestamps
    ds_timestamp_formatted = ds_timestamp_converted.strftime("%Y%m%d_%H%M%S_%f").tolist()

    # ---------------------------------------------------------
    # Save timestamps to CSV
    # ---------------------------------------------------------
    df = pd.DataFrame({"timestamp": ds_timestamp_formatted})
    df.to_csv(csv_out_path, index=False)

    print(f"✔️ Timestamp CSV saved: {csv_out_path}")

print("\n🎉 All timestamps extracted successfully!")
