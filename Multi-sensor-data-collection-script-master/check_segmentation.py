"""
该文档用于检查 RGB, Depth, Vlp16, Mindrove 分割的片段是否一致, 对于 vlp16 和 mindrove 他们的片段中含有字样_npy， 在比较时会自动去除。
"""

from pathlib import Path
from collections import defaultdict


def normalize_segment_name(name: str) -> str:
    """
    统一片段文件夹名：
    - 去掉结尾的 '_npy'
    """
    if name.endswith("_npy"):
        return name[:-4]
    return name


def list_action_segment_pairs(
    root: str | Path,
    ignore_action_names=None,
    ignore_segment_names=None,
):
    """
    只认两层目录结构：
      root/
        action_dir/
          segment_dir/   ← 这里可能是 xxx 或 xxx_npy
            ...

    返回：
      set[(action_name, normalized_segment_name)]
    """
    root = Path(root)

    if ignore_action_names is None:
        ignore_action_names = set()
    if ignore_segment_names is None:
        ignore_segment_names = set()

    pairs = set()

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    # 第一层：动作文件夹
    for action_dir in root.iterdir():
        if not action_dir.is_dir():
            continue
        if action_dir.name.startswith("."):
            continue
        if action_dir.name in ignore_action_names:
            continue

        # 第二层：片段文件夹
        for seg_dir in action_dir.iterdir():
            if not seg_dir.is_dir():
                continue
            if seg_dir.name.startswith("."):
                continue
            if seg_dir.name in ignore_segment_names:
                continue

            # 🔥 新增：规范化片段名（去掉 _npy）
            seg_name_norm = normalize_segment_name(seg_dir.name)

            pairs.add((action_dir.name, seg_name_norm))

    return pairs


def compare_modalities(mod_roots: dict):
    """
    mod_roots:
      {
        "RGB": Path(...),
        "Depth": Path(...),
        "mindrove": Path(...)
      }
    """
    pairs_by_mod = {}

    for mod, root in mod_roots.items():
        pairs_by_mod[mod] = list_action_segment_pairs(root)

    mods = list(pairs_by_mod.keys())

    union_pairs = set().union(*pairs_by_mod.values())
    common_pairs = set.intersection(*pairs_by_mod.values())

    print("========== Summary ==========")
    for mod in mods:
        print(f"{mod:8s}: {len(pairs_by_mod[mod])} action/segment pairs")
    print(f"Union   : {len(union_pairs)}")
    print(f"Common  : {len(common_pairs)}")

    print("\n========== Differences ==========")
    for mod in mods:
        missing = union_pairs - pairs_by_mod[mod]
        extra   = pairs_by_mod[mod] - common_pairs

        print(f"\n[{mod}]")
        print(f"  Missing vs union : {len(missing)}")
        print(f"  Extra   vs common: {len(extra)}")

        if missing:
            print("  Example missing (up to 20):")
            for a, s in sorted(missing)[:20]:
                print(f"    - {a}/{s}")

        if extra:
            print("  Example extra (up to 20):")
            for a, s in sorted(extra)[:20]:
                print(f"    + {a}/{s}")

    return {
        "pairs_by_mod": pairs_by_mod,
        "union": union_pairs,
        "common": common_pairs,
    }


if __name__ == "__main__":
    mod_roots = {
        "RGB_M": r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\RGB\MR",
        "Depth": r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\Depth\MR",
        "mindrove": r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\mindrove_npy\MR",
        "vlp16": r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\vlp16_npy\MR"
    }

    compare_modalities(mod_roots)




