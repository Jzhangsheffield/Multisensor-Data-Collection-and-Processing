"""
该脚本用于检查 RGB, Depth, VLP16, MindRove 四个模态在“动作/片段”粒度上的切分是否一致。

数据结构假设（只认两层目录）：
root/
  action_dir/
    segment_dir/   <-- 片段文件夹名，可能包含一些后缀或标记（如 _npy / _mid / _elbow 等）
      ...

功能：
- 对每个模态根目录，扫描出所有 (action_name, normalized_segment_name) 的集合
- 计算各模态的 union / common
- 输出每个模态缺失(missing)与额外(extra)的片段示例，便于定位数据不一致

注意：
- normalize_segment_name 支持 remove_tokens，可配置要从片段名中去除的“字样”
"""

from pathlib import Path
from typing import Dict, Set, Tuple, Optional, Iterable


def normalize_segment_name(
    name: str,
    remove_tokens: Optional[Iterable[str]] = None,
) -> str:
    """
    统一片段文件夹名（做“命名层面”的对齐）：

    参数
    ----
    name : str
        原始片段文件夹名，比如：
          run_4_clip_000012_mid_npy
    remove_tokens : Iterable[str] | None
        要去除的字样列表，例如：
          ["_npy", "_mid", "_elbow", "_left", "_right"]
        - None 表示使用默认值 ["_npy"]

    返回
    ----
    str
        规范化后的片段名，比如：
          run_4_clip_000012
    """
    if remove_tokens is None:
        # 默认保持与你之前脚本一致：只去掉 _npy
        remove_tokens = ["_npy"]

    normalized = name

    # 逐个删除 token（出现多次也会全部删掉）
    for token in remove_tokens:
        if token:  # 防止传入空字符串导致死循环或意外行为
            normalized = normalized.replace(token, "")

    # 清理可能产生的多余下划线，例如：run__4__clip -> run_4_clip
    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    # 清理首尾下划线：_run_1_ -> run_1
    normalized = normalized.strip("_")

    return normalized


def list_action_segment_pairs(
    root: str | Path,
    ignore_action_names: Optional[Set[str]] = None,
    ignore_segment_names: Optional[Set[str]] = None,
    remove_tokens: Optional[Iterable[str]] = None,
) -> Set[Tuple[str, str]]:
    """
    扫描一个模态根目录，返回 (action_name, normalized_segment_name) 的集合。

    只认两层目录结构：
      root/
        action_dir/
          segment_dir/

    参数
    ----
    root : str | Path
        模态根目录路径
    ignore_action_names : set[str] | None
        忽略的动作文件夹名集合（完全匹配 action_dir.name）
    ignore_segment_names : set[str] | None
        忽略的片段文件夹名集合（完全匹配 seg_dir.name；注意这里是“原始名”）
    remove_tokens : Iterable[str] | None
        传给 normalize_segment_name 的配置，用于对齐片段命名差异

    返回
    ----
    set[(action_name, normalized_segment_name)]
    """
    root = Path(root)

    if ignore_action_names is None:
        ignore_action_names = set()
    if ignore_segment_names is None:
        ignore_segment_names = set()

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    pairs: Set[Tuple[str, str]] = set()

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

            # 规范化片段名：把 _npy / _mid / _elbow ... 等标记去掉，便于跨模态比较
            seg_name_norm = normalize_segment_name(
                seg_dir.name,
                remove_tokens=remove_tokens
            )

            pairs.add((action_dir.name, seg_name_norm))

    return pairs


def compare_modalities(
    mod_roots: Dict[str, str | Path],
    remove_tokens: Optional[Iterable[str]] = None,
    ignore_action_names: Optional[Set[str]] = None,
    ignore_segment_names: Optional[Set[str]] = None,
):
    """
    比较多个模态目录下的动作/片段是否一致。

    参数
    ----
    mod_roots : dict
        形如：
          {
            "RGB": Path(...),
            "Depth": Path(...),
            "mindrove": Path(...),
            "vlp16": Path(...)
          }
    remove_tokens : Iterable[str] | None
        用于 normalize_segment_name 的 token 列表（所有模态统一使用同一套规则）
    ignore_action_names / ignore_segment_names :
        可选忽略列表
    """
    pairs_by_mod: Dict[str, Set[Tuple[str, str]]] = {}

    # 1) 扫描每个模态的 (action, segment) 集合
    for mod, root in mod_roots.items():
        pairs_by_mod[mod] = list_action_segment_pairs(
            root,
            ignore_action_names=ignore_action_names,
            ignore_segment_names=ignore_segment_names,
            remove_tokens=remove_tokens,
        )

    mods = list(pairs_by_mod.keys())

    # 2) 计算全集（union）和交集（common）
    union_pairs = set().union(*pairs_by_mod.values()) if pairs_by_mod else set()
    common_pairs = set.intersection(*pairs_by_mod.values()) if pairs_by_mod else set()

    # 3) 输出统计信息
    print("========== Summary ==========")
    for mod in mods:
        print(f"{mod:8s}: {len(pairs_by_mod[mod])} action/segment pairs")
    print(f"Union   : {len(union_pairs)}")
    print(f"Common  : {len(common_pairs)}")

    # 4) 输出差异
    print("\n========== Differences ==========")
    for mod in mods:
        # missing：该模态缺少 union 里的某些 action/segment
        missing = union_pairs - pairs_by_mod[mod]

        # extra：该模态比 common 多出来的 action/segment
        # （常见原因：该模态多切了一段，或命名规则不一致仍未被 normalize 覆盖）
        extra = pairs_by_mod[mod] - common_pairs

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

    # ✅ 你可以在这里统一定义“需要从片段名中去除的标记”
    # - 如果只想保持旧行为，就设为 ["_npy"] 或者直接传 None
    REMOVE_TOKENS = ["_npy", "_mid", "_elbow"]

    mod_roots = {
        "RGB_M": r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\RGB\MR",
        "Depth": r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\kinect\Depth\MR",
        "mindrove": r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\mindrove_npy\MR",
        "vlp16": r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\Thermal_Crimping_Dataset\vlp16_npy\MR"
    }

    compare_modalities(
        mod_roots,
        remove_tokens=REMOVE_TOKENS,
        # ignore_action_names=set([...]),     # 如需忽略动作目录可在此配置
        # ignore_segment_names=set([...]),    # 如需忽略片段目录可在此配置
    )
