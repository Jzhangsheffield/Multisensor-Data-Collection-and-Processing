#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def norm_run_token(x) -> str:
    s = str(x).strip().lower()
    # 处理 1.0 -> "1"
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        s = s.replace(".0", "")
    return s


def build_run2light_map(global_csv: Path) -> dict:
    df = pd.read_csv(global_csv)

    if "run" not in df.columns or "light" not in df.columns:
        raise ValueError(f"Global CSV must contain columns: run, light. Got: {list(df.columns)}")

    df = df[["run", "light"]].copy()
    df["run_key"] = df["run"].apply(norm_run_token)

    # 每个 run_key 的 light 必须唯一
    nunique = df.groupby("run_key")["light"].nunique(dropna=False)
    bad = nunique[nunique > 1]
    if len(bad) > 0:
        examples = []
        for r in bad.index[:10]:
            vals = df.loc[df["run_key"] == r, "light"].unique().tolist()
            examples.append(f"{r}: {vals}")
        raise ValueError(
            "Some runs have multiple different 'light' values in global CSV.\n"
            + "\n".join(examples)
        )

    return df.drop_duplicates("run_key").set_index("run_key")["light"].to_dict()


def is_copy_file(path: Path) -> bool:
    """忽略文件名里包含 copy 的文件（大小写不敏感）。"""
    return "copy" in path.name.lower()


def extract_run_token_from_annotation_filename(name: str) -> str | None:
    """
    支持：
      J_run_1_annotation.csv        -> "1"
      J_run_1-37_annotation.csv     -> "1-37"

    只处理严格以 _annotation.csv 结尾的文件（copy 会在上层被过滤掉）。
    """
    low = name.lower().strip()
    if not low.endswith("_annotation.csv"):
        return None

    parts = low.split("_")
    if len(parts) < 4:
        return None
    if parts[1] != "run":
        return None

    return parts[2]  # "1" or "1-37"


def find_annotation_csvs(root: Path) -> list[Path]:
    # 找所有 csv，再由逻辑过滤：必须是 *_annotation.csv 且不含 copy
    all_csv = sorted(root.rglob("*.csv"))
    out = []
    for p in all_csv:
        if is_copy_file(p):
            continue
        if p.name.lower().endswith("_annotation.csv"):
            out.append(p)
    return out


def add_light_and_save(ann_path: Path, run2light: dict):
    token = extract_run_token_from_annotation_filename(ann_path.name)
    if token is None:
        return False, f"Skip (unrecognized name): {ann_path.name}"

    run_key = norm_run_token(token)
    if run_key not in run2light:
        return False, f"Run token not found in global CSV: {token}  (file: {ann_path.name})"

    light_val = run2light[run_key]

    ann_df = pd.read_csv(ann_path)
    ann_df["light"] = light_val

    out_name = ann_path.name.replace("_annotation.csv", "_annotationlight.csv")
    out_path = ann_path.with_name(out_name)

    ann_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return True, f"OK: {ann_path.name} -> {out_path.name} (light={light_val})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_csv", type=str, required=True, help="整体标注CSV路径(含run, light两列)")
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\kinect",
        help="kinect 根目录，会递归查找 *_annotation.csv（自动忽略含 copy 的文件）",
    )
    args = parser.parse_args()

    global_csv = Path(args.global_csv)
    root = Path(args.root)

    if not global_csv.exists():
        raise FileNotFoundError(f"Global CSV not found: {global_csv}")
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    run2light = build_run2light_map(global_csv)
    ann_files = find_annotation_csvs(root)

    if len(ann_files) == 0:
        print(f"No valid annotation CSV found under: {root}")
        return

    ok = skipped = failed = 0
    for p in ann_files:
        success, msg = add_light_and_save(p, run2light)
        print(msg)
        if success:
            ok += 1
        else:
            if msg.startswith("Skip"):
                skipped += 1
            else:
                failed += 1

    print("\n========== Summary ==========")
    print(f"Total valid found: {len(ann_files)}")
    print(f"Written:          {ok}")
    print(f"Skipped:          {skipped}")
    print(f"Failed:           {failed}")


if __name__ == "__main__":
    main()
