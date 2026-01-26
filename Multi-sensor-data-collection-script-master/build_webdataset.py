#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flexible multimodal WebDataset packer (RGB / DEPTH / LiDAR / EMG), all optional.

You may provide any subset of the following inputs:
- --rgb_cam_json_dir     : directory containing per-camera RGB jsons (cam_*.json)
- --depth_cam_json_dir   : directory containing per-camera DEPTH jsons (cam_*.json)
- --lidar_json           : lidar manifest json (key -> npy_path, ...)
- --emg_json             : emg manifest json (key -> left_npy/right_npy, ...)

If an input is NOT provided, that modality is simply skipped.

------------------------------------------------------------
Expected per-camera json format (RGB or DEPTH), one json per camera:
  {
    "J_adjust_slider_run_11_clip_000003_left": {
      "cam": "cam_001431512812",
      "frames": ["D:\\...\\20251113_170325_880571.jpg", ...],
      "n_frames": 67,
      "frame_dir": "D:\\...\\cam_001431512812",
      ...
    },
    ...
  }

The script will merge all camera json files into:
  modality[key]["cams"][cam] = {"frames":[...], "n_frames":...}

------------------------------------------------------------
WebDataset grouping rule:
- All tar members that share the same prefix before the FIRST '.' are grouped as ONE sample.
  So we MUST name members like:
    {key}.rgb_<camid>.<000001>.jpg
    {key}.depth_<camid>.<000001>.png
    {key}.lidar.npy
    {key}.emg_left.npy
    {key}.labels.json
    {key}.meta.json

Then WebDataset will return one sample with __key__ = key.

------------------------------------------------------------
Memory / performance:
- Worker processes DO NOT read image bytes (avoid huge RAM usage).
- Workers only build a "plan" (tar member name + source file path).
- Main process streams files into tar sequentially.

Outputs:
- out_dir/shards/shard-000000.tar, ...
- out_dir/shards.txt
- out_dir/manifest.jsonl        (sample-level manifest)
- out_dir/wds_index.jsonl       (member-level tar offset index)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from multiprocessing import Pool


# -----------------------------
# Basic IO utilities
# -----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file into dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def json_to_bytes(obj: Any) -> bytes:
    """Serialize object into UTF-8 JSON bytes."""
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")

def bytes_to_tar(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    """
    Write small bytes content into tar.
    Suitable for labels/meta/debug json (tiny files).
    """
    ti = tarfile.TarInfo(name=arcname)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))

def file_to_tar(tar: tarfile.TarFile, arcname: str, src_path: Path) -> None:
    """
    Stream an existing file into tar without reading it fully into RAM.
    This is crucial for large jpg/png/npy.
    """
    st = src_path.stat()
    ti = tarfile.TarInfo(name=arcname)
    ti.size = st.st_size
    with src_path.open("rb") as f:
        tar.addfile(ti, f)


# -----------------------------
# Key -> labels parsing (your rule)
# -----------------------------

def parse_labels_from_key(key: str) -> Dict[str, Any]:
    """
    Your label rules (updated to match your key style):
    key example:
      J_cap_green_pen_run_22_clip_000027_right
      J_adjust_slider_run_8-37_clip_000003_normal

    label_1 = content between first token and 'run'
            = "cap_green_pen" or "adjust_slider"
    label_2 = the first token after first token
            = "cap" or "adjust"
    """
    tokens = key.split("_")
    prefix = tokens[0] if tokens else ""

    try:
        run_idx = tokens.index("run")
    except ValueError:
        run_idx = -1

    mid_tokens = tokens[1:run_idx] if (run_idx != -1 and run_idx > 1) else (tokens[1:] if len(tokens) > 1 else [])
    label_1 = "_".join(mid_tokens) if mid_tokens else ""
    label_2 = tokens[1] if len(tokens) > 1 else ""

    return {
        "label_1": label_1,
        "label_2": label_2,
        "prefix": prefix,
        "key_tokens": tokens,  # for debugging; remove if you don't want it
    }


# -----------------------------
# Load per-camera json directory (RGB or DEPTH)
# -----------------------------

def load_cam_json_dir(cam_json_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all *.json in a directory, each representing one camera.

    Return merged structure:
      out[key]["cams"][cam_name] = {
         "frames": [abs_paths...],
         "n_frames": N,
         "frame_dir": ...
      }

    Notes:
    - This function does NOT assume whether it's RGB or DEPTH.
      The caller decides which modality this dict belongs to.
    """
    out: Dict[str, Dict[str, Any]] = {}

    json_files = sorted([p for p in cam_json_dir.glob("*.json") if p.is_file()])
    if not json_files:
        raise FileNotFoundError(f"No .json files found in: {cam_json_dir}")

    for jp in json_files:
        cam_data = read_json(jp)  # key -> entry
        for key, entry in cam_data.items():
            cam = entry.get("cam") or jp.stem
            frames = entry.get("frames", []) or []
            if not frames:
                continue

            out.setdefault(key, {})
            out[key].setdefault("cams", {})
            out[key]["cams"][cam] = {
                "frames": frames,
                "n_frames": entry.get("n_frames", len(frames)),
                "frame_dir": entry.get("frame_dir", ""),
            }

    return out


# -----------------------------
# "Plan" structures: workers return these (lightweight)
# -----------------------------

@dataclass
class MemberPlan:
    """
    One tar member to be written.

    Exactly one of:
      - src_path (stream a file into tar)
      - data_bytes (write bytes into tar)
    """
    arcname: str
    src_path: Optional[str] = None
    data_bytes: Optional[bytes] = None


@dataclass
class SamplePlan:
    """
    All members belonging to a single sample (one __key__).
    """
    key: str
    members: List[MemberPlan]
    labels: Dict[str, Any]
    meta: Dict[str, Any]


def plan_frames_for_modality(key: str, modality: str, cams_dict: Dict[str, Any]) -> Tuple[List[MemberPlan], Dict[str, Any]]:
    """
    Create plans for frame-based modality (RGB or DEPTH).

    Tar naming:
      {key}.{modality}_{camid}.{frame_idx:06d}{ext}

    Example:
      J_xxx_run_5_clip_000003_left.rgb_001431512812.000010.jpg
      J_xxx_run_5_clip_000003_left.depth_001431512812.000010.png

    This guarantees:
    - Same prefix {key} => WebDataset groups them as one sample
    - modality is encoded in the extension name part => no collision between rgb & depth
    """
    members: List[MemberPlan] = []
    meta_mod: Dict[str, Any] = {"n_cams": len(cams_dict), "cams": {}}

    for cam_name, cam_info in cams_dict.items():
        cam_id = cam_name.replace("cam_", "")
        frames: List[str] = cam_info.get("frames", []) or []
        if not frames:
            continue

        # Sort by filename to preserve time order (your names are timestamp-like)
        frames_sorted = sorted(frames, key=lambda x: Path(x).name)
        meta_mod["cams"][cam_name] = {"n_frames": len(frames_sorted)}

        for i, fp in enumerate(frames_sorted):
            ext = Path(fp).suffix.lower()  # keep original (.jpg/.png)
            arc = f"{key}.{modality}_{cam_id}.{i:06d}{ext}"
            members.append(MemberPlan(arcname=arc, src_path=fp))

    return members, meta_mod


def build_sample_plan(args: Tuple[str,
                                  Optional[Dict[str, Any]],  # rgb_item
                                  Optional[Dict[str, Any]],  # depth_item
                                  Optional[Dict[str, Any]],  # lidar_item
                                  Optional[Dict[str, Any]],  # emg_item
                                  bool  # store_debug_json
                                  ]) -> Optional[SamplePlan]:
    """
    Worker: build a plan for one key using whatever modalities exist.
    This function must be lightweight: do NOT read image bytes here.
    """
    key, rgb_item, depth_item, lidar_item, emg_item, store_debug_json = args

    labels = parse_labels_from_key(key)
    meta: Dict[str, Any] = {"key": key, "has": {}}
    members: List[MemberPlan] = []

    # ---- RGB (optional) ----
    if rgb_item is not None and rgb_item.get("cams"):
        meta["has"]["rgb"] = True
        rgb_members, meta_rgb = plan_frames_for_modality(key, "rgb", rgb_item["cams"])
        members.extend(rgb_members)
        meta["rgb"] = meta_rgb
        if store_debug_json:
            members.append(MemberPlan(arcname=f"{key}.rgb.json", data_bytes=json_to_bytes(rgb_item)))
    else:
        meta["has"]["rgb"] = False

    # ---- DEPTH (optional) ----
    if depth_item is not None and depth_item.get("cams"):
        meta["has"]["depth"] = True
        depth_members, meta_depth = plan_frames_for_modality(key, "depth", depth_item["cams"])
        members.extend(depth_members)
        meta["depth"] = meta_depth
        if store_debug_json:
            members.append(MemberPlan(arcname=f"{key}.depth.json", data_bytes=json_to_bytes(depth_item)))
    else:
        meta["has"]["depth"] = False

    # ---- LiDAR (optional) ----
    if lidar_item is not None and lidar_item.get("npy_path"):
        meta["has"]["lidar"] = True
        meta["lidar"] = {
            "num_frames": lidar_item.get("num_frames"),
            "dtype": lidar_item.get("dtype"),
            "num_cols": lidar_item.get("num_cols"),
            "max_rows": lidar_item.get("max_rows"),
            "padded": lidar_item.get("padded"),
            "pad_value": lidar_item.get("pad_value"),
        }
        members.append(MemberPlan(arcname=f"{key}.lidar.npy", src_path=lidar_item["npy_path"]))
        if store_debug_json:
            members.append(MemberPlan(arcname=f"{key}.lidar.json", data_bytes=json_to_bytes(lidar_item)))
    else:
        meta["has"]["lidar"] = False

    # ---- EMG (optional) ----
    if emg_item is not None:
        left_path = emg_item.get("left_npy")
        right_path = emg_item.get("right_npy")
        has_any = bool(left_path) or bool(right_path)

        meta["has"]["emg"] = has_any
        meta["emg"] = {
            "left_shape": emg_item.get("left_shape"),
            "right_shape": emg_item.get("right_shape"),
            "dtype": emg_item.get("dtype"),
        }

        if left_path:
            members.append(MemberPlan(arcname=f"{key}.emg_left.npy", src_path=left_path))
        if right_path:
            members.append(MemberPlan(arcname=f"{key}.emg_right.npy", src_path=right_path))

        if store_debug_json:
            members.append(MemberPlan(arcname=f"{key}.emg.json", data_bytes=json_to_bytes(emg_item)))
    else:
        meta["has"]["emg"] = False

    # Always store labels/meta (small)
    members.append(MemberPlan(arcname=f"{key}.labels.json", data_bytes=json_to_bytes(labels)))
    members.append(MemberPlan(arcname=f"{key}.meta.json", data_bytes=json_to_bytes(meta)))

    # If the sample has no real data (only labels/meta), you may want to skip it.
    # Here we keep it ONLY if at least one modality exists.
    has_any_modality = any(meta["has"].values())
    if not has_any_modality:
        return None

    return SamplePlan(key=key, members=members, labels=labels, meta=meta)


# -----------------------------
# Tar writing (main process)
# -----------------------------

def write_shards(sample_plans_iter, shards_dir: Path, shard_size: int) -> List[Path]:
    """
    Consume SamplePlan iterator and write .tar shards.

    shard_size = number of SAMPLES per shard (not number of files).
    For large samples (many frames), choose small shard_size (e.g., 2~16).
    """
    ensure_dir(shards_dir)
    shard_paths: List[Path] = []

    shard_idx = 0
    in_shard = 0
    tar: Optional[tarfile.TarFile] = None

    def open_new_shard():
        nonlocal tar, shard_idx, in_shard
        if tar is not None:
            tar.close()
        out_path = shards_dir / f"shard-{shard_idx:06d}.tar"
        tar = tarfile.open(out_path, "w")
        shard_paths.append(out_path)
        shard_idx += 1
        in_shard = 0

    open_new_shard()

    for sp in sample_plans_iter:
        if sp is None:
            continue

        if in_shard >= shard_size:
            open_new_shard()

        # Write all members belonging to this sample
        for m in sp.members:
            if m.data_bytes is not None:
                bytes_to_tar(tar, m.arcname, m.data_bytes)
            else:
                try:
                    file_to_tar(tar, m.arcname, Path(m.src_path))
                except Exception as e:
                    # Do not crash; record an error text file in the same sample
                    err_name = f"{sp.key}.ERR.{Path(m.arcname).name}.txt"
                    bytes_to_tar(tar, err_name, str(e).encode("utf-8"))

        in_shard += 1

    if tar is not None:
        tar.close()

    return shard_paths


# -----------------------------
# Index files (optional but useful)
# -----------------------------

def build_tar_member_index(shard_paths: List[Path], out_index_jsonl: Path) -> None:
    """
    Member-level tar index (JSONL):
      {"shard": "...tar", "member": "KEY.xxx", "offset_data": ..., "size": ..., "key": "KEY"}

    Useful for debugging / custom random access.
    """
    with out_index_jsonl.open("w", encoding="utf-8") as f:
        for sp in shard_paths:
            with tarfile.open(sp, "r") as tar:
                for m in tar.getmembers():
                    if not m.isfile():
                        continue
                    member = m.name
                    key = member.split(".", 1)[0] if "." in member else member
                    row = {
                        "shard": str(sp),
                        "member": member,
                        "key": key,
                        "offset_data": m.offset_data,
                        "size": m.size,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sample_manifest(shard_paths: List[Path], out_manifest_jsonl: Path) -> None:
    """
    Sample-level manifest (JSONL):
      {"key": key, "shard": tar_path, "members": [...], "labels": {...}}

    We read {key}.labels.json inside tar to store labels.
    """
    with out_manifest_jsonl.open("w", encoding="utf-8") as f:
        for sp in shard_paths:
            with tarfile.open(sp, "r") as tar:
                by_key: Dict[str, List[tarfile.TarInfo]] = {}
                for m in tar.getmembers():
                    if not m.isfile():
                        continue
                    key = m.name.split(".", 1)[0] if "." in m.name else m.name
                    by_key.setdefault(key, []).append(m)

                for key, members in by_key.items():
                    member_names = sorted([m.name for m in members])

                    labels = {}
                    label_member = f"{key}.labels.json"
                    names_set = {m.name for m in members}
                    if label_member in names_set:
                        try:
                            ex = tar.extractfile(label_member)
                            if ex is not None:
                                labels = json.loads(ex.read().decode("utf-8"))
                        except Exception:
                            labels = {}

                    row = {
                        "key": key,
                        "shard": str(sp),
                        "members": member_names,
                        "labels": labels,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # Optional inputs: provide any subset
    ap.add_argument("--rgb_cam_json_dir", type=str, default="",
                    help="(Optional) directory containing RGB per-camera jsons (cam_*.json)")
    ap.add_argument("--depth_cam_json_dir", type=str, default="",
                    help="(Optional) directory containing DEPTH per-camera jsons (cam_*.json)")
    ap.add_argument("--lidar_json", type=str, default="",
                    help="(Optional) LiDAR manifest json (key -> npy_path, ...)")
    ap.add_argument("--emg_json", type=str, default="",
                    help="(Optional) EMG manifest json (key -> left_npy/right_npy, ...)")

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--shard_size", type=int, default=8,
                    help="Samples per shard. For large samples use 2~16.")
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--keys_intersection", action="store_true",
                    help="Only pack keys that exist in ALL PROVIDED modalities. Default: union.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--store_debug_json", action="store_true")

    args = ap.parse_args()

    # ---- Load whichever manifests are provided ----
    rgb = None
    depth = None
    lidar = None
    emg = None

    provided_key_sets: List[set] = []

    if args.rgb_cam_json_dir:
        rgb = load_cam_json_dir(Path(args.rgb_cam_json_dir))
        provided_key_sets.append(set(rgb.keys()))

    if args.depth_cam_json_dir:
        depth = load_cam_json_dir(Path(args.depth_cam_json_dir))
        provided_key_sets.append(set(depth.keys()))

    if args.lidar_json:
        lidar = read_json(Path(args.lidar_json))
        provided_key_sets.append(set(lidar.keys()))

    if args.emg_json:
        emg = read_json(Path(args.emg_json))
        provided_key_sets.append(set(emg.keys()))

    if not provided_key_sets:
        raise ValueError("You must provide at least one modality manifest (rgb/depth/lidar/emg).")

    # Determine which keys to pack:
    # - intersection: only keys present in all PROVIDED modalities
    # - union:        keys present in any PROVIDED modality
    if args.keys_intersection:
        keys = sorted(list(set.intersection(*provided_key_sets)))
    else:
        keys = sorted(list(set.union(*provided_key_sets)))

    if args.limit and args.limit > 0:
        keys = keys[:args.limit]

    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    ensure_dir(out_dir)
    ensure_dir(shards_dir)

    # Prepare worker tasks for all keys.
    # For any missing modality, pass None; worker will skip it.
    tasks: List[Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]] = []
    for k in keys:
        tasks.append((
            k,
            rgb.get(k) if rgb else None,
            depth.get(k) if depth else None,
            lidar.get(k) if lidar else None,
            emg.get(k) if emg else None,
            bool(args.store_debug_json),
        ))

    # Worker builds plans -> main writes tar
    with Pool(processes=args.num_workers) as pool:
        plan_iter = pool.imap_unordered(build_sample_plan, tasks, chunksize=8)
        shard_paths = write_shards(plan_iter, shards_dir, args.shard_size)

    # Write shards list
    shards_txt = out_dir / "shards.txt"
    with shards_txt.open("w", encoding="utf-8") as f:
        for sp in shard_paths:
            f.write(str(sp) + "\n")

    # Build indexes
    build_sample_manifest(shard_paths, out_dir / "manifest.jsonl")
    build_tar_member_index(shard_paths, out_dir / "wds_index.jsonl")

    print(f"✅ Done. packed_keys={len(keys)}, shards={len(shard_paths)}")
    print(f"- shards list:     {shards_txt}")
    print(f"- sample manifest: {out_dir / 'manifest.jsonl'}")
    print(f"- tar index:       {out_dir / 'wds_index.jsonl'}")


if __name__ == "__main__":
    main()
