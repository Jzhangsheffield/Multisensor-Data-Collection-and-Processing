#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="合并 N_run_*_annotation.csv（排除 copy），增加 run 列并按 run 第一个数字排序"
    )
    parser.add_argument(
        "--input_root", type=str, required=True,
        help=r"输入根目录，例如 D:\_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\N\kinect"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help=r"输出的拼接 csv 路径，例如 D:\_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\N\N_annotation.csv"
    )
    return parser.parse_args()


# 提取 run 字符串和第一个数字
def parse_run_from_filename(filename: str):
    """
    只匹配两种形式的文件名：
      - N_run_24_annotation.csv
      - N_run_24-37_annotation.csv

    返回:
      run_str: 比如 "24" 或 "24-37"
      first_num: 用于排序的第一个数字 (int)
    """
    # 正则：N_run_ 后面跟数字和 -，直到 _annotation.csv
    pattern = re.compile(r"^J_run_([0-9][0-9\-]*)_annotation\.csv$")
    m = pattern.match(filename)
    if not m:
        return None, None

    run_str = m.group(1)              # "24" 或 "24-37"
    first_num_str = run_str.split("-")[0]
    first_num = int(first_num_str)
    return run_str, first_num


def collect_and_concat_csv(input_root: Path, output_csv: Path):
    # 递归找到所有 csv
    all_csv_paths = list(input_root.rglob("*.csv"))
    print(f"在 {input_root} 下共找到 {len(all_csv_paths)} 个 csv 文件。")

    # 1. 过滤：排除包含 "copy" 的；只保留符合 N_run_*_annotation.csv 格式的
    candidates = []
    for p in all_csv_paths:
        name_lower = p.name.lower()
        if "copy" in name_lower:
            continue

        run_str, first_num = parse_run_from_filename(p.name)
        if run_str is None:
            continue

        candidates.append((first_num, run_str, p))

    if not candidates:
        print("没有找到符合命名规则的 CSV 文件（N_run_*_annotation.csv）。")
        return

    # 2. 按 first_num 升序排序
    candidates.sort(key=lambda x: x[0])

    print("符合条件并排序后的文件列表：")
    for first_num, run_str, p in candidates:
        print(f"  first_num={first_num:3d}, run={run_str:>7s}, file={p}")

    dfs = []
    for first_num, run_str, csv_path in candidates:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"读取失败，跳过文件: {csv_path}，错误: {e}")
            continue

        # 增加一列 run = "24" 或 "24-37"
        df["run"] = run_str
        dfs.append(df)

    if not dfs:
        print("没有成功读取任何 CSV 文件。")
        return

    # 3. 按 run 第一个数字排序后的顺序拼接
    combined = pd.concat(dfs, ignore_index=True)

    # 确保输出目录存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n拼接完成，共 {len(combined)} 行，已保存到:\n  {output_csv}")


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_csv = Path(args.output_csv)

    collect_and_concat_csv(input_root, output_csv)


if __name__ == "__main__":
    main()
