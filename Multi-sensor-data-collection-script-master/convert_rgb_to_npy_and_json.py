# import os
# import json
# import argparse
# import re
# from pathlib import Path
# from typing import List, Dict, Tuple, Optional
# from datetime import datetime

# import numpy as np

# # --- image backend: prefer cv2, fallback to PIL ---
# try:
#     import cv2

#     def imread_rgb(path: str):
#         img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
#         if img_bgr is None:
#             raise RuntimeError(f"Failed to read image: {path}")
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#         return img_rgb

#     def resize_img(img, size_hw: Tuple[int, int]):
#         h, w = size_hw
#         return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

# except Exception:
#     cv2 = None
#     from PIL import Image

#     def imread_rgb(path: str):
#         with Image.open(path) as im:
#             im = im.convert("RGB")
#             return np.array(im, dtype=np.uint8)

#     def resize_img(img, size_hw: Tuple[int, int]):
#         h, w = size_hw
#         im = Image.fromarray(img)
#         im = im.resize((w, h), resample=Image.BILINEAR)
#         return np.array(im, dtype=np.uint8)


# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# # frame name: 20251113_170325_880571.jpg
# _TS_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<micro>\d+)$")


# def parse_frame_ts(filename_stem: str):
#     """
#     输入: '20251113_170325_880571'（不含扩展名）
#     输出: (datetime_obj, micro_int) 作为排序 key
#     """
#     m = _TS_RE.match(filename_stem)
#     if not m:
#         return None
#     d = m.group("date")     # YYYYMMDD
#     t = m.group("time")     # HHMMSS
#     micro = int(m.group("micro"))
#     dt = datetime.strptime(d + t, "%Y%m%d%H%M%S")
#     return (dt, micro)


# def list_images_in_dir(d: Path) -> List[Path]:
#     files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

#     def sort_key(p: Path):
#         ts = parse_frame_ts(p.stem)
#         if ts is None:
#             return (1, p.name)         # 解析失败的放后面
#         return (0, ts[0], ts[1])       # datetime + micro

#     files.sort(key=sort_key)
#     return files


# def find_leaf_frame_dirs(root: Path) -> List[Path]:
#     """
#     找到“叶子帧目录”：该目录包含图片帧，且其子目录没有再包含帧目录。
#     bottom-up 避免重复处理父目录。
#     """
#     frame_dirs = set()

#     for dirpath, dirnames, filenames in os.walk(root, topdown=False):
#         d = Path(dirpath)
#         has_imgs = any(Path(f).suffix.lower() in IMG_EXTS for f in filenames)
#         if not has_imgs:
#             continue

#         children = [d / sd for sd in dirnames]
#         # 若任何子目录已经是帧目录，则当前不是“最终帧目录”
#         if any((c in frame_dirs) or any(str(fd).startswith(str(c) + os.sep) for fd in frame_dirs) for c in children):
#             continue

#         frame_dirs.add(d)

#     return sorted(frame_dirs, key=lambda x: str(x))


# def parse_subject_action_run_cam(root: Path, frame_dir: Path) -> Tuple[str, str, str, str]:
#     """
#     期望结构：
#       root / subject / action / run_clip / cam_dir  (cam_dir 内是帧)
#     """
#     rel = frame_dir.relative_to(root)
#     parts = rel.parts
#     if len(parts) < 4:
#         raise ValueError(
#             f"Frame dir path too shallow. Expect root/subject/action/run_clip/cam. Got: {frame_dir}"
#         )
#     subject, action, run_clip, cam = parts[0], parts[1], parts[2], parts[3]
#     return subject, action, run_clip, cam


# def ensure_dir(p: Path):
#     p.mkdir(parents=True, exist_ok=True)


# def get_npy_meta(npy_path: Path) -> Optional[Dict]:
#     """用于 overwrite=False 且文件已存在时，读取 .npy 的 shape/dtype（mmap，不会把数据读进内存）"""
#     if not npy_path.exists():
#         return None
#     arr = np.load(str(npy_path), mmap_mode="r")
#     meta = {
#         "shape": list(arr.shape),
#         "dtype": str(arr.dtype),
#     }
#     del arr
#     return meta


# def pack_one_cam_folder(
#     frame_dir: Path,
#     out_dir: Path,
#     out_file: Path,
#     resize_hw: Optional[Tuple[int, int]] = None,
#     overwrite: bool = False,
# ) -> Dict:
#     """
#     将 frame_dir 下所有帧打包成 out_file (.npy).
#     使用 open_memmap 逐帧写入，避免内存爆。
#     返回打包元数据：orig_size, resized_size, n_frames, dtype, shape...
#     """
#     ensure_dir(out_dir)

#     imgs = list_images_in_dir(frame_dir)
#     if len(imgs) == 0:
#         return {
#             "status": "skipped_no_frames",
#             "n_frames": 0,
#             "path": str(out_file),
#         }

#     # 读第一帧得到原始尺寸
#     first_raw = imread_rgb(str(imgs[0]))
#     if first_raw.ndim != 3 or first_raw.shape[2] != 3:
#         raise RuntimeError(f"Unexpected image shape: {first_raw.shape} from {imgs[0]}")
#     orig_h, orig_w = first_raw.shape[0], first_raw.shape[1]

#     # 计算输出尺寸
#     if resize_hw is not None:
#         first = resize_img(first_raw, resize_hw)
#         out_h, out_w = resize_hw
#     else:
#         first = first_raw
#         out_h, out_w = orig_h, orig_w

#     T = len(imgs)

#     # 如果已存在且不覆盖：也返回 meta（shape/dtype 从 npy 读）
#     if out_file.exists() and not overwrite:
#         npy_meta = get_npy_meta(out_file) or {}
#         # 若 npy_meta 里能拿到 shape，则 out_h/out_w 以 npy 为准更靠谱
#         if "shape" in npy_meta and len(npy_meta["shape"]) == 4:
#             _, h, w, _ = npy_meta["shape"]
#             out_h, out_w = h, w
#         return {
#             "status": "skipped_exists",
#             "n_frames": T,
#             "path": str(out_file),
#             "orig_size_hw": [orig_h, orig_w],
#             "resized_size_hw": [out_h, out_w],
#             "dtype": npy_meta.get("dtype", "uint8"),
#             "shape": npy_meta.get("shape", [T, out_h, out_w, 3]),
#         }

#     # overwrite：先删旧文件
#     if out_file.exists():
#         out_file.unlink()

#     # 创建 .npy memmap 文件 (T, H, W, 3)
#     arr = np.lib.format.open_memmap(
#         filename=str(out_file),
#         mode="w+",
#         dtype=np.uint8,
#         shape=(T, out_h, out_w, 3),
#     )

#     # 写入第 0 帧
#     if first.shape != (out_h, out_w, 3):
#         raise RuntimeError(f"First frame shape mismatch: {first.shape} vs {(out_h, out_w, 3)}")
#     arr[0] = first

#     # 逐帧写入
#     for i in range(1, T):
#         img = imread_rgb(str(imgs[i]))
#         if resize_hw is not None:
#             img = resize_img(img, resize_hw)

#         if img.shape != (out_h, out_w, 3):
#             raise RuntimeError(
#                 f"Shape mismatch at {imgs[i]}: got {img.shape}, expected {(out_h, out_w, 3)}. "
#                 f"Tip: use --resize to force fixed size."
#             )
#         arr[i] = img

#     del arr  # flush

#     return {
#         "status": "ok",
#         "n_frames": T,
#         "path": str(out_file),
#         "orig_size_hw": [orig_h, orig_w],
#         "resized_size_hw": [out_h, out_w],
#         "dtype": "uint8",
#         "shape": [T, out_h, out_w, 3],
#     }


# def main():
#     parser = argparse.ArgumentParser(
#         description="Recursively pack leaf RGB frame folders into .npy, and write a JSON index with metadata."
#     )
#     parser.add_argument(
#         "--root",
#         type=str,
#         required=True,
#         help=r'RGB root, e.g. D:\_data\...\kinect\RGB',
#     )
#     parser.add_argument(
#         "--resize",
#         type=int,
#         nargs=2,
#         default=None,
#         metavar=("H", "W"),
#         help="Optional resize to (H W), e.g. --resize 256 256",
#     )
#     parser.add_argument(
#         "--suffix",
#         type=str,
#         default="_npy",
#         help='Output dir suffix. cam_xxx -> cam_xxx + suffix (default: "_npy")',
#     )
#     parser.add_argument(
#         "--index_json",
#         type=str,
#         default=None,
#         help="Where to save JSON index. Default: <root>/rgb_npy_index.json",
#     )
#     parser.add_argument(
#         "--overwrite",
#         action="store_true",
#         help="Overwrite existing npy files.",
#     )
#     parser.add_argument(
#         "--subject_map_json",
#         type=str,
#         default=None,
#         help=(
#             "Optional JSON for mapping subject folder name, e.g. "
#             r'{"J":"N"} to turn J_adjust_slider_* into N_adjust_slider_*'
#         ),
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Print per-folder logs.",
#     )
#     args = parser.parse_args()

#     root = Path(args.root).resolve()
#     if not root.exists():
#         raise FileNotFoundError(f"Root not found: {root}")

#     resize_hw = tuple(args.resize) if args.resize is not None else None

#     # subject mapping (optional)
#     subject_map = {}
#     if args.subject_map_json:
#         with open(args.subject_map_json, "r", encoding="utf-8") as f:
#             subject_map = json.load(f)

#     index_json_path = Path(args.index_json).resolve() if args.index_json else (root / "rgb_npy_index.json")

#     leaf_dirs = find_leaf_frame_dirs(root)
#     if args.verbose:
#         print(f"[INFO] Found {len(leaf_dirs)} leaf frame dirs under: {root}")

#     # index: key -> { cam_field: {path, orig_size_hw, resized_size_hw, n_frames, dtype, shape} }
#     index: Dict[str, Dict[str, Dict]] = {}
#     stats = {"ok": 0, "skipped_exists": 0, "skipped_no_frames": 0, "error": 0}

#     for frame_dir in leaf_dirs:
#         try:
#             subject, action, run_clip, cam = parse_subject_action_run_cam(root, frame_dir)
#             subject_mapped = subject_map.get(subject, subject)

#             key = f"{subject_mapped}_{action}_{run_clip}"
#             cam_field = f"rgb_{cam}"

#             out_dir = frame_dir.parent / f"{cam}{args.suffix}"
#             out_file = out_dir / f"{cam}.npy"

#             info = pack_one_cam_folder(
#                 frame_dir=frame_dir,
#                 out_dir=out_dir,
#                 out_file=out_file,
#                 resize_hw=resize_hw,
#                 overwrite=args.overwrite,
#             )

#             if key not in index:
#                 index[key] = {}

#             index[key][cam_field] = {
#                 "path": info["path"],
#                 "orig_size_hw": info.get("orig_size_hw"),
#                 "resized_size_hw": info.get("resized_size_hw"),
#                 "n_frames": info.get("n_frames"),
#                 "dtype": info.get("dtype"),
#                 "shape": info.get("shape"),
#             }

#             stats[info["status"]] = stats.get(info["status"], 0) + 1
#             if args.verbose:
#                 print(f"[{info['status']}] {frame_dir} -> {out_file} | n={info.get('n_frames')} | "
#                       f"orig={info.get('orig_size_hw')} | out={info.get('resized_size_hw')}")

#         except Exception as e:
#             stats["error"] += 1
#             if args.verbose:
#                 print(f"[ERROR] {frame_dir} | {repr(e)}")
#             else:
#                 print(f"[ERROR] {frame_dir} | {e}")

#     ensure_dir(index_json_path.parent)
#     with open(index_json_path, "w", encoding="utf-8") as f:
#         json.dump(index, f, ensure_ascii=False, indent=2)

#     print("\n[Done]")
#     print(f"Root: {root}")
#     print(f"Index JSON: {index_json_path}")
#     print("Stats:", stats)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

import numpy as np

# --- image backend: prefer cv2, fallback to PIL ---
try:
    import cv2

    def imread_rgb(path: str):
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

    def resize_img(img, size_hw: Tuple[int, int]):
        h, w = size_hw
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

except Exception:
    cv2 = None
    from PIL import Image

    def imread_rgb(path: str):
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im, dtype=np.uint8)

    def resize_img(img, size_hw: Tuple[int, int]):
        h, w = size_hw
        im = Image.fromarray(img)
        im = im.resize((w, h), resample=Image.BILINEAR)
        return np.array(im, dtype=np.uint8)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# frame name: 20251113_170325_880571.jpg
_TS_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<micro>\d+)$")


def parse_frame_ts(filename_stem: str):
    m = _TS_RE.match(filename_stem)
    if not m:
        return None
    d = m.group("date")
    t = m.group("time")
    micro = int(m.group("micro"))
    dt = datetime.strptime(d + t, "%Y%m%d%H%M%S")
    return (dt, micro)


def list_images_in_dir(d: Path) -> List[Path]:
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    def sort_key(p: Path):
        ts = parse_frame_ts(p.stem)
        if ts is None:
            return (1, p.name)
        return (0, ts[0], ts[1])

    files.sort(key=sort_key)
    return files


def find_leaf_frame_dirs(root: Path) -> List[Path]:
    frame_dirs = set()

    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        d = Path(dirpath)
        has_imgs = any(Path(f).suffix.lower() in IMG_EXTS for f in filenames)
        if not has_imgs:
            continue

        children = [d / sd for sd in dirnames]
        if any((c in frame_dirs) or any(str(fd).startswith(str(c) + os.sep) for fd in frame_dirs) for c in children):
            continue

        frame_dirs.add(d)

    return sorted(frame_dirs, key=lambda x: str(x))


def parse_subject_action_run_cam(root: Path, frame_dir: Path) -> Tuple[str, str, str, str]:
    rel = frame_dir.relative_to(root)
    parts = rel.parts
    if len(parts) < 4:
        raise ValueError(
            f"Frame dir path too shallow. Expect root/subject/action/run_clip/cam. Got: {frame_dir}"
        )
    subject, action, run_clip, cam = parts[0], parts[1], parts[2], parts[3]
    return subject, action, run_clip, cam


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pack_one_cam_folder(
    frame_dir: Path,
    out_file: Path,
    out_hw: Tuple[int, int],
    resize_hw: Optional[Tuple[int, int]] = None,
    overwrite: bool = False,
) -> Dict:
    """
    将 frame_dir 下所有帧打包成 out_file (.npy).
    - 不读取首帧来确定尺寸：尺寸由 out_hw/resize_hw 外部传入
    - 仍会逐帧读入写入 npy（这是必须的）
    """
    ensure_dir(out_file.parent)

    imgs = list_images_in_dir(frame_dir)
    T = len(imgs)
    if T == 0:
        return {"status": "skipped_no_frames", "n_frames": 0, "path": str(out_file)}

    out_h, out_w = out_hw

    if out_file.exists() and not overwrite:
        return {"status": "skipped_exists", "n_frames": T, "path": str(out_file)}

    if out_file.exists():
        out_file.unlink()

    arr = np.lib.format.open_memmap(
        filename=str(out_file),
        mode="w+",
        dtype=np.uint8,
        shape=(T, out_h, out_w, 3),
    )

    for i in range(T):
        img = imread_rgb(str(imgs[i]))
        if resize_hw is not None:
            img = resize_img(img, resize_hw)

        if img.shape != (out_h, out_w, 3):
            raise RuntimeError(
                f"Shape mismatch at {imgs[i]}: got {img.shape}, expected {(out_h, out_w, 3)}. "
                f"Tip: ensure --orig-size/--resize are correct."
            )
        arr[i] = img

    del arr
    return {"status": "ok", "n_frames": T, "path": str(out_file)}


def derive_out_root_from_in_root(in_root: Path) -> Path:
    """
    输入:
      ...\\Thermal_Crimping_Dataset\\kinect\\RGB
      ...\\Thermal_Crimping_Dataset\\kinect\\Depth
    输出:
      ...\\Thermal_Crimping_Dataset\\kinect_npy\\RGB
      ...\\Thermal_Crimping_Dataset\\kinect_npy\\Depth
    """
    # in_root = .../kinect/<modality>
    dataset_dir = in_root.parent.parent  # .../Thermal_Crimping_Dataset
    modality = in_root.name              # RGB or Depth or others
    return dataset_dir / "kinect_npy" / modality


def main():
    parser = argparse.ArgumentParser(
        description="Recursively pack leaf frame folders into .npy, and write a JSON index with metadata."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help=r'Input root, e.g. D:\...\Thermal_Crimping_Dataset\kinect\RGB  or  ...\kinect\Depth',
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help=(
            r'Output root, e.g. D:\...\Thermal_Crimping_Dataset\kinect_npy\RGB (or Depth). '
            r'If not set, will derive from --root.'
        ),
    )

    # 不读取帧来推断尺寸，运行时输入
    parser.add_argument(
        "--orig-size",
        type=int,
        nargs=2,
        required=True,
        metavar=("H", "W"),
        help="Original frame size (H W), e.g. --orig-size 360 640",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Optional resize to (H W), e.g. --resize 256 256",
    )

    parser.add_argument(
        "--index_json",
        type=str,
        default=None,
        help="Where to save JSON index. Default: <out_root>/frame_npy_index.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--subject_map_json", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    in_root = Path(args.root).resolve()
    if not in_root.exists():
        raise FileNotFoundError(f"Root not found: {in_root}")

    out_root = Path(args.out_root).resolve() if args.out_root else derive_out_root_from_in_root(in_root)
    ensure_dir(out_root)

    orig_hw = tuple(args.orig_size)  # (H, W)
    resize_hw = tuple(args.resize) if args.resize is not None else None
    out_hw = resize_hw if resize_hw is not None else orig_hw

    # subject mapping (optional)
    subject_map = {}
    if args.subject_map_json:
        with open(args.subject_map_json, "r", encoding="utf-8") as f:
            subject_map = json.load(f)

    index_json_path = Path(args.index_json).resolve() if args.index_json else (out_root / "frame_npy_index.json")

    leaf_dirs = find_leaf_frame_dirs(in_root)
    if args.verbose:
        print(f"[INFO] Found {len(leaf_dirs)} leaf frame dirs under: {in_root}")

    # index: key -> { meta + cams }
    # key: J_adjust_slider_run_3_clip_000003_npy
    # value:
    #   {
    #     "orig_size_hw": [H,W],
    #     "resize_size_hw": [H,W] or null,
    #     "out_size_hw": [H,W],
    #     "cams": {
    #        "cam_001431...": {"path": "...", "n_frames": 123},
    #        ...
    #     }
    #   }
    index: Dict[str, Dict[str, Any]] = {}
    stats = {"ok": 0, "skipped_exists": 0, "skipped_no_frames": 0, "error": 0}

    for frame_dir in leaf_dirs:
        try:
            subject, action, run_clip, cam = parse_subject_action_run_cam(in_root, frame_dir)
            subject_mapped = subject_map.get(subject, subject)

            key = f"{subject_mapped}_{action}_{run_clip}_npy"

            # 输出目录：
            # out_root / subject / action / (run_clip + "_npy") / cam / (cam + ".npy")
            rel_to_in = frame_dir.relative_to(in_root)  # subject/action/run_clip/cam
            rel_parts = list(rel_to_in.parts)
            rel_parts[2] = rel_parts[2] + "_npy"
            out_cam_dir = out_root.joinpath(*rel_parts)
            out_file = out_cam_dir / f"{cam}.npy"

            info = pack_one_cam_folder(
                frame_dir=frame_dir,
                out_file=out_file,
                out_hw=out_hw,
                resize_hw=resize_hw,
                overwrite=args.overwrite,
            )

            if key not in index:
                index[key] = {
                    "orig_size_hw": [int(orig_hw[0]), int(orig_hw[1])],
                    "resize_size_hw": [int(resize_hw[0]), int(resize_hw[1])] if resize_hw else None,
                    "out_size_hw": [int(out_hw[0]), int(out_hw[1])],
                    "cams": {},
                }

            index[key]["cams"][cam] = {
                "path": info["path"],
                "n_frames": int(info.get("n_frames", 0)),
            }

            stats[info["status"]] = stats.get(info["status"], 0) + 1
            if args.verbose:
                print(f"[{info['status']}] {frame_dir} -> {out_file} | n={info.get('n_frames')}")

        except Exception as e:
            stats["error"] += 1
            if args.verbose:
                print(f"[ERROR] {frame_dir} | {repr(e)}")
            else:
                print(f"[ERROR] {frame_dir} | {e}")

    ensure_dir(index_json_path.parent)
    with open(index_json_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n[Done]")
    print(f"Input Root : {in_root}")
    print(f"Output Root: {out_root}")
    print(f"Index JSON : {index_json_path}")
    print(f"orig_size_hw={list(orig_hw)}, resize_size_hw={list(resize_hw) if resize_hw else None}, out_size_hw={list(out_hw)}")
    print("Stats:", stats)


if __name__ == "__main__":
    main()
