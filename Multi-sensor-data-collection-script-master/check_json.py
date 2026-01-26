# import json

# with open(r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\RGB\rgb_meta_info.json\cam_001431512812.json", "r", encoding="utf-8") as f:
#     data_rgb = json.load(f)

# with open(r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\mindrove_npy\mindrove_meta_info.json", "r", encoding="utf-8") as f:
#     data_mindrove = json.load(f)

# # bad_keys = []
# # for k in data_rgb.keys():
# #     if not (k.endswith("_left") or k.endswith("_right") or k.endswith("_normal")):
# #         bad_keys.append(k)

# # print("Bad keys:", bad_keys)
# # print("Total bad:", len(bad_keys))
# rgb_keys = data_rgb.keys()

# mindrove_keys = data_mindrove.keys()




import json
import csv

def compare_json_keys_and_save_csv(json_path_a, json_path_b, output_csv):
    with open(json_path_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    with open(json_path_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    keys_a = set(data_a.keys())
    keys_b = set(data_b.keys())

    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    print(f"✅ Common keys: {len(common)}")
    print(f"❌ Only in A: {len(only_in_a)}")
    print(f"❌ Only in B: {len(only_in_b)}")

    # 写 CSV
    max_len = max(len(only_in_a), len(only_in_b), len(common))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["only_in_A", "only_in_B", "common"])

        for i in range(max_len):
            row = [
                only_in_a[i] if i < len(only_in_a) else "",
                only_in_b[i] if i < len(only_in_b) else "",
                common[i] if i < len(common) else "",
            ]
            writer.writerow(row)

    print(f"📄 CSV saved to: {output_csv}")


if __name__ == "__main__":
    json_a = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\mindrove\mindrove_meta_info.json"
    json_b = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\RGB\rgb_meta_info.json\cam_001431512812.json"
    out_csv = r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\RGB\rgb_meta_info.json\compare.csv"
    compare_json_keys_and_save_csv(json_a, json_b, out_csv)
