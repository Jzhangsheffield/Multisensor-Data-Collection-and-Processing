#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flexible multimodal WebDataset packer (RGB / DEPTH / LiDAR / MindRove/EMG), all optional.

✅ Supports the NEW split JSON format only:

{
  "__meta__": {...},
  "samples": {
      "<key>": {
          "person": "...",
          "tier1": "...",
          ...
          "files": {
              "rgb":   {"cam_xxx":[...], "cam_yyy":[...]},
              "depth": {"cam_xxx":[...], ...},
              "mindrove": {"left":"...npy","right":"...npy"}  # or left/right dict
              "lidar": "....npy"  # or {"npy_path":"..."}
          },
          "missing_modalities": ["depth", ...]
      }
  }
}

✅ NEW: Selective packing
--------------------------------
You can choose:
1) which modalities to pack:
   --modalities rgb,depth,lidar,emg      (any subset of {rgb,depth,lidar,emg})

2) which camera ids to pack (for rgb/depth):
   --camids cam_001431...,001484...
   - accepts "cam_XXXX" or bare "XXXX"
   - empty means include all cams

3) which modalities the cam filter applies to:
   --cam_apply_to rgb,depth   (default both)

Optional (debug JSON):
- --store_debug_json
  If enabled, the script stores modality debug JSON members in each sample:
    <key>.rgb.json / <key>.depth.json / <key>.lidar.json / <key>.emg.json
  You can choose whether the stored rgb/depth debug JSON is:
    - original (unfiltered)
    - filtered to selected cams
  via:
    --debug_json_filtered_cams

Shard cutting rule:
- Primary: by target bytes (default 2 GiB): if adding a full sample would exceed threshold -> open new shard.
- Optional: also cap by sample count using --shard_size (0 means disable).

Members naming (IMPORTANT):
- WebDataset groups members by the prefix before the FIRST '.'.
- Therefore, we must ensure the prefix (key) DOES NOT contain '/' or '\\',
  otherwise tar will contain directory hierarchy and grouping breaks.

This version:
✅ Converts split_json original keys into a FLAT safe key like:
   J_adjust_slider_run_3_clip_000003_normal_elbow

Outputs (split-aware when --split_name is provided):
- out_dir/shards/shard-<split_name>-000000.tar, ...
- out_dir/shards_<split_name>.txt
- out_dir/manifest_<split_name>.jsonl
- out_dir/wds_index_<split_name>.jsonl
- out_dir/split_meta_<split_name>.json

Example usage
--------------------------------
Pack only RGB + EMG, and only one cam for RGB:
  python webdataset_packer_selective.py \
    --split_json /path/train.json --split_name train --out_dir /path/out \
    --modalities rgb,emg --camids cam_001431512812

Pack Depth only, two cameras:
  python webdataset_packer_selective.py \
    --split_json /path/val.json --split_name val --out_dir /path/out \
    --modalities depth --camids 001431512812,001484412812

Pack RGB+Depth, but cam filter applies only to RGB (Depth packs all cams):
  python webdataset_packer_selective.py \
    --split_json /path/train.json --split_name train --out_dir /path/out \
    --modalities rgb,depth --camids 001431512812 --cam_apply_to rgb
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterable, Set
from multiprocessing import Pool


# -----------------------------
# Basic IO utilities
# -----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def json_to_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")

def bytes_to_tar(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    """Write in-memory bytes as one tar member."""
    ti = tarfile.TarInfo(name=arcname)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))

def file_to_tar(tar: tarfile.TarFile, arcname: str, src_path: Path, known_size: Optional[int] = None) -> None:
    """
    Stream a file into tar without loading it into memory.
    known_size: if provided, avoids a redundant stat().
    """
    size = int(known_size) if known_size is not None else src_path.stat().st_size
    ti = tarfile.TarInfo(name=arcname)
    ti.size = size
    with src_path.open("rb") as f:
        tar.addfile(ti, f)


# -----------------------------
# Key utilities (flatten key)
# -----------------------------

def sanitize_key_basic(k: str) -> str:
    """
    Make a key safe for tar member names:
      - replace both '\\' and '/' with '_'
      - collapse whitespace
      - collapse repeated underscores
    This guarantees no directory hierarchy appears in tar.
    """
    if k is None:
        return ""
    s = str(k).strip()
    s = s.replace("\\", "_").replace("/", "_")
    s = "_".join(s.split())
    while "__" in s:
        s = s.replace("__", "_")
    return s

def build_safe_key_from_record(original_key: str, sample_entry: Dict[str, Any], style: str = "person_action_segment") -> str:
    """
    Convert split_json original key into a tar-safe key.

    style options:
      - person_action_segment (DEFAULT): person_action_segment -> J_adjust_slider_run_..._elbow
        uses sample_entry['person'], sample_entry['action'] or ['tier2'/'tier3'], sample_entry['segment']
        fallback: infer from original_key tokens after sanitization
      - sanitize_only: just sanitize original_key (replace separators), no reformatting
    """
    if style == "sanitize_only":
        return sanitize_key_basic(original_key)

    person = sample_entry.get("person")
    action = sample_entry.get("action") or sample_entry.get("tier2") or sample_entry.get("tier3")
    segment = sample_entry.get("segment")

    # Fallback inference from original_key when fields missing
    if not (person and action and segment):
        parts = [p for p in sanitize_key_basic(original_key).split("_") if p]
        if not person and len(parts) >= 1:
            person = parts[0]
        if not action and len(parts) >= 2:
            action = parts[1]
        if not segment and len(parts) >= 3:
            segment = "_".join(parts[2:])

    person = sanitize_key_basic(person or "UNKNOWN")
    action = sanitize_key_basic(action or "UNKNOWN_ACTION")
    segment = sanitize_key_basic(segment or "UNKNOWN_SEGMENT")

    return f"{person}_{action}_{segment}"


# -----------------------------
# Key -> labels parsing (kept for backward compatibility)
# -----------------------------

def parse_labels_from_key(key: str) -> Dict[str, Any]:
    """
    A lightweight parser for underscore keys.
    - label_1: tokens between prefix and 'run' (if 'run' exists), else everything after prefix
    - label_2: tokens[1] if exists
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
        "key_tokens": tokens,
    }


# -----------------------------
# Split JSON loader (NEW ONLY)
# -----------------------------

def load_split_json(split_json_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = read_json(split_json_path)
    meta = obj.get("__meta__", {}) or {}
    samples = obj.get("samples", {}) or {}
    if not isinstance(samples, dict):
        raise ValueError(f"Invalid split json: 'samples' must be a dict. got={type(samples)}")
    return meta, samples


# -----------------------------
# "Plan" structures
# -----------------------------

@dataclass
class MemberPlan:
    """
    One tar member to be written.

    Exactly one of:
      - src_path (file on disk)
      - data_bytes (in-memory bytes)
    should be non-None.
    """
    arcname: str
    src_path: Optional[str] = None
    data_bytes: Optional[bytes] = None
    size_bytes: Optional[int] = None   # raw file size, for shard byte estimation / avoid double-stat

@dataclass
class SamplePlan:
    """
    Represents one "sample" (one WebDataset group prefix).
    """
    original_key: str               # original split_json key (may contain '/' or '\\')
    key: str                        # safe key used for tar grouping (no separators)
    members: List[MemberPlan]       # all tar members for this sample
    labels: Dict[str, Any]          # labels JSON (will be stored)
    meta: Dict[str, Any]            # meta JSON (will be stored)


# -----------------------------
# Modality planning helpers
# -----------------------------

def _normalize_cam_files(cam_to_frames: Any) -> Dict[str, Dict[str, Any]]:
    """
    Normalize camera frames structure to:
      { cam_name: {"frames":[...], "n_frames": N}, ... }
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(cam_to_frames, dict):
        return out
    for cam, frames in cam_to_frames.items():
        if not frames:
            continue
        if isinstance(frames, list):
            out[str(cam)] = {"frames": frames, "n_frames": len(frames)}
    return out

def _filter_cams_dict_by_camids(cams_dict: Dict[str, Any], allowed_camids: Optional[Set[str]]) -> Dict[str, Any]:
    """
    Filter a cams_dict by allowed camids.
    cams_dict keys are typically like "cam_001431512812".
    allowed_camids are stored without "cam_" prefix.
    """
    if not allowed_camids:
        return cams_dict
    out: Dict[str, Any] = {}
    for cam_name, cam_info in cams_dict.items():
        cam_id = str(cam_name).replace("cam_", "")
        if cam_id in allowed_camids:
            out[cam_name] = cam_info
    return out

def _safe_getsize(p: str) -> Optional[int]:
    try:
        return os.path.getsize(p)
    except Exception:
        return None

def plan_frames_for_modality(
    safe_key: str,
    modality: str,
    cams_dict: Dict[str, Any],
    allowed_camids: Optional[Set[str]] = None,
) -> Tuple[List[MemberPlan], Dict[str, Any]]:
    """
    Create MemberPlan list for frame-based modalities (rgb/depth).
    Each frame becomes one tar member:
      <safe_key>.<modality>_<camid>.<frame_idx><ext>

    allowed_camids:
      - set of camera ids WITHOUT "cam_" prefix, e.g. {"001431512812"}
      - None or empty means include all cams
    """
    members: List[MemberPlan] = []
    meta_mod: Dict[str, Any] = {"n_cams": 0, "cams": {}}

    for cam_name in sorted(cams_dict.keys(), key=lambda x: str(x)):
        cam_info = cams_dict[cam_name]
        cam_id = str(cam_name).replace("cam_", "")

        # Selective camera packing
        if allowed_camids and (cam_id not in allowed_camids):
            continue

        frames: List[str] = cam_info.get("frames", []) or []
        if not frames:
            continue

        # Sort frames by filename for stable ordering
        frames_sorted = sorted(frames, key=lambda x: Path(x).name)

        meta_mod["cams"][cam_name] = {"n_frames": len(frames_sorted)}

        for i, fp in enumerate(frames_sorted):
            ext = Path(fp).suffix.lower()
            arc = f"{safe_key}.{modality}_{cam_id}.{i:06d}{ext}"
            members.append(MemberPlan(arcname=arc, src_path=fp, size_bytes=_safe_getsize(fp)))

    meta_mod["n_cams"] = len(meta_mod["cams"])
    return members, meta_mod

def _get_lidar_path(lidar_obj: Any) -> Optional[str]:
    """
    lidar can be:
      - "....npy"
      - {"npy_path":"..."} or {"path":"..."}
    """
    if isinstance(lidar_obj, str) and lidar_obj.strip():
        return lidar_obj
    if isinstance(lidar_obj, dict):
        p = lidar_obj.get("npy_path") or lidar_obj.get("path")
        if isinstance(p, str) and p.strip():
            return p
    return None

def _get_mindrove_paths(mindrove_obj: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    mindrove/emg can be:
      - {"left":"...npy","right":"...npy"}
      - {"left":{"npy":"..."},"right":{"npy":"..."}}
      - key variants: npy/path/npy_path
    """
    left = right = None
    if isinstance(mindrove_obj, dict):
        l = mindrove_obj.get("left")
        r = mindrove_obj.get("right")

        if isinstance(l, str):
            left = l
        elif isinstance(l, dict):
            lp = l.get("npy") or l.get("path") or l.get("npy_path")
            if isinstance(lp, str):
                left = lp

        if isinstance(r, str):
            right = r
        elif isinstance(r, dict):
            rp = r.get("npy") or r.get("path") or r.get("npy_path")
            if isinstance(rp, str):
                right = rp

    return left, right


# -----------------------------
# Worker: build sample plan (selective modalities + selective camids)
# -----------------------------

def build_sample_plan(args: Tuple[str, Dict[str, Any], bool, str, Set[str], Set[str], Set[str], bool]) -> Optional[SamplePlan]:
    """
    Worker builds a "plan" without reading full bytes.
    It may stat sizes to estimate shard size by bytes.

    args:
      (original_key, sample_entry,
       store_debug_json, key_style,
       want_modalities, want_camids, cam_apply_to,
       debug_json_filtered_cams)

    want_modalities:
      subset of {"rgb","depth","lidar","emg"}

    want_camids:
      set of cam ids WITHOUT "cam_" prefix, e.g. {"001431..."} (empty means all)

    cam_apply_to:
      subset of {"rgb","depth"}; indicates which modalities should apply cam filter
    """
    (original_key, sample_entry,
     store_debug_json, key_style,
     want_modalities, want_camids, cam_apply_to,
     debug_json_filtered_cams) = args

    # Build tar-safe key (no '/' or '\\')
    safe_key = build_safe_key_from_record(original_key, sample_entry, style=key_style)

    # Labels: start from key parser, then override with explicit fields
    labels = parse_labels_from_key(safe_key)
    for k in ("tier1", "tier2", "tier3", "action", "person", "run_token", "lighting", "pos", "segment"):
        if k in sample_entry and sample_entry[k] is not None:
            labels[k] = sample_entry[k]

    files = sample_entry.get("files", {}) or {}
    missing_modalities = sample_entry.get("missing_modalities", []) or []

    # Meta keeps traceability
    meta: Dict[str, Any] = {
        "original_key": original_key,
        "safe_key": safe_key,
        "has": {},
        "missing_modalities": missing_modalities,
        "selective": {
            "want_modalities": sorted(list(want_modalities)),
            "want_camids": sorted(list(want_camids)),
            "cam_apply_to": sorted(list(cam_apply_to)),
            "debug_json_filtered_cams": bool(debug_json_filtered_cams),
        }
    }

    members: List[MemberPlan] = []

    # -------------------------
    # RGB (frame modality)
    # -------------------------
    if "rgb" in want_modalities:
        rgb_files = files.get("rgb")
        rgb_cams = _normalize_cam_files(rgb_files)

        # Apply cam filter if requested
        allowed = want_camids if (want_camids and "rgb" in cam_apply_to) else None
        rgb_cams_filtered = _filter_cams_dict_by_camids(rgb_cams, allowed)

        if rgb_cams_filtered:
            rgb_members, meta_rgb = plan_frames_for_modality(safe_key, "rgb", rgb_cams_filtered, allowed_camids=None)
            # Note: plan_frames_for_modality already supports allowed_camids, but we filtered dict already.
            # This avoids repeated cam_id parsing and lets debug_json choose filtered/unfiltered cleanly.

            if rgb_members:
                meta["has"]["rgb"] = True
                members.extend(rgb_members)
                meta["rgb"] = meta_rgb

                if store_debug_json:
                    # Choose whether debug json stores filtered cams or original cams
                    dbg_obj = rgb_cams_filtered if debug_json_filtered_cams else (rgb_files if isinstance(rgb_files, dict) else rgb_files)
                    b = json_to_bytes(dbg_obj)
                    members.append(MemberPlan(arcname=f"{safe_key}.rgb.json", data_bytes=b, size_bytes=len(b)))
            else:
                meta["has"]["rgb"] = False
        else:
            meta["has"]["rgb"] = False
    else:
        meta["has"]["rgb"] = False

    # -------------------------
    # DEPTH (frame modality)
    # -------------------------
    if "depth" in want_modalities:
        depth_files = files.get("depth")
        depth_cams = _normalize_cam_files(depth_files)

        allowed = want_camids if (want_camids and "depth" in cam_apply_to) else None
        depth_cams_filtered = _filter_cams_dict_by_camids(depth_cams, allowed)

        if depth_cams_filtered:
            depth_members, meta_depth = plan_frames_for_modality(safe_key, "depth", depth_cams_filtered, allowed_camids=None)
            if depth_members:
                meta["has"]["depth"] = True
                members.extend(depth_members)
                meta["depth"] = meta_depth

                if store_debug_json:
                    dbg_obj = depth_cams_filtered if debug_json_filtered_cams else (depth_files if isinstance(depth_files, dict) else depth_files)
                    b = json_to_bytes(dbg_obj)
                    members.append(MemberPlan(arcname=f"{safe_key}.depth.json", data_bytes=b, size_bytes=len(b)))
            else:
                meta["has"]["depth"] = False
        else:
            meta["has"]["depth"] = False
    else:
        meta["has"]["depth"] = False

    # -------------------------
    # LiDAR (single file modality)
    # -------------------------
    if "lidar" in want_modalities:
        lidar_path = _get_lidar_path(files.get("lidar"))
        if lidar_path:
            meta["has"]["lidar"] = True
            meta["lidar"] = {"path": lidar_path}
            members.append(MemberPlan(
                arcname=f"{safe_key}.lidar.npy",
                src_path=lidar_path,
                size_bytes=_safe_getsize(lidar_path)
            ))
            if store_debug_json:
                b = json_to_bytes(files.get("lidar"))
                members.append(MemberPlan(arcname=f"{safe_key}.lidar.json", data_bytes=b, size_bytes=len(b)))
        else:
            meta["has"]["lidar"] = False
    else:
        meta["has"]["lidar"] = False

    # -------------------------
    # EMG (MindRove) (single/dual file modality)
    # -------------------------
    if "emg" in want_modalities:
        mindrove_obj = files.get("mindrove") or files.get("emg")
        left_path, right_path = _get_mindrove_paths(mindrove_obj)
        has_any_emg = bool(left_path) or bool(right_path)
        meta["has"]["emg"] = has_any_emg

        if has_any_emg:
            meta["emg"] = {"left_path": left_path, "right_path": right_path}
            if left_path:
                members.append(MemberPlan(
                    arcname=f"{safe_key}.emg_left.npy",
                    src_path=left_path,
                    size_bytes=_safe_getsize(left_path)
                ))
            if right_path:
                members.append(MemberPlan(
                    arcname=f"{safe_key}.emg_right.npy",
                    src_path=right_path,
                    size_bytes=_safe_getsize(right_path)
                ))
            if store_debug_json:
                b = json_to_bytes(mindrove_obj)
                members.append(MemberPlan(arcname=f"{safe_key}.emg.json", data_bytes=b, size_bytes=len(b)))
        else:
            meta["emg"] = {}
    else:
        meta["has"]["emg"] = False
        meta["emg"] = {}

    # Always store labels/meta (even if selective packing removes some modalities)
    b1 = json_to_bytes(labels)
    b2 = json_to_bytes(meta)
    members.append(MemberPlan(arcname=f"{safe_key}.labels.json", data_bytes=b1, size_bytes=len(b1)))
    members.append(MemberPlan(arcname=f"{safe_key}.meta.json", data_bytes=b2, size_bytes=len(b2)))

    # Skip samples with no included modality content (after filtering)
    if not any(meta["has"].values()):
        return None

    return SamplePlan(original_key=original_key, key=safe_key, members=members, labels=labels, meta=meta)


# -----------------------------
# Shard cutting by bytes (main process)
# -----------------------------

def _tar_member_contribution_bytes(raw_size: int) -> int:
    """
    Estimate bytes added to the tar stream for one regular file entry:
      - 512 bytes header
      - data padded up to a multiple of 512
    """
    if raw_size < 0:
        raw_size = 0
    blocks = (raw_size + 511) // 512
    return 512 + blocks * 512

def _estimate_sample_tar_bytes(sp: SamplePlan) -> int:
    """
    Estimate tar stream bytes for a whole sample (sum of member contributions).
    Uses known sizes from planning stage to avoid reading actual bytes.
    """
    total = 0
    for m in sp.members:
        if m.data_bytes is not None:
            raw = len(m.data_bytes)
        else:
            raw = m.size_bytes if m.size_bytes is not None else 0
        total += _tar_member_contribution_bytes(int(raw))
    return total

def write_shards_by_bytes(
    sample_plans_iter: Iterable[Optional[SamplePlan]],
    shards_dir: Path,
    shard_name_prefix: str,
    shard_max_bytes: int,
    shard_size_cap: int = 0,   # 0 -> disable sample-count cap
) -> List[Path]:
    """
    Stream SamplePlan objects into tar shards.

    Shard cut rule:
      1) bytes threshold (primary)
      2) optional sample-count threshold
    """
    ensure_dir(shards_dir)
    shard_paths: List[Path] = []

    shard_idx = 0
    in_shard = 0
    shard_bytes = 0
    tar: Optional[tarfile.TarFile] = None

    def open_new_shard():
        nonlocal tar, shard_idx, in_shard, shard_bytes
        if tar is not None:
            tar.close()
        out_path = shards_dir / f"{shard_name_prefix}-{shard_idx:06d}.tar"
        tar = tarfile.open(out_path, "w")
        shard_paths.append(out_path)
        shard_idx += 1
        in_shard = 0
        shard_bytes = 0

    open_new_shard()

    for sp in sample_plans_iter:
        if sp is None:
            continue

        sample_bytes = _estimate_sample_tar_bytes(sp)

        # Cut shard BEFORE writing this sample if it would exceed bytes threshold
        if in_shard > 0 and (shard_bytes + sample_bytes) > shard_max_bytes:
            open_new_shard()

        # Optional: also cut by sample-count cap
        if shard_size_cap and shard_size_cap > 0 and in_shard >= shard_size_cap:
            open_new_shard()

        # Write members
        for m in sp.members:
            if m.data_bytes is not None:
                bytes_to_tar(tar, m.arcname, m.data_bytes)
                shard_bytes += _tar_member_contribution_bytes(len(m.data_bytes))
            else:
                try:
                    file_to_tar(tar, m.arcname, Path(m.src_path), known_size=m.size_bytes)
                    raw = m.size_bytes
                    if raw is None:
                        try:
                            raw = Path(m.src_path).stat().st_size
                        except Exception:
                            raw = 0
                    shard_bytes += _tar_member_contribution_bytes(int(raw))
                except Exception as e:
                    # Write an error member instead of crashing the whole pack.
                    # Use safe key to avoid creating directories.
                    err_name = f"{sp.key}.ERR.{Path(m.arcname).name}.txt"
                    b = str(e).encode("utf-8")
                    bytes_to_tar(tar, err_name, b)
                    shard_bytes += _tar_member_contribution_bytes(len(b))

        in_shard += 1

    if tar is not None:
        tar.close()
    return shard_paths


# -----------------------------
# Index files
# -----------------------------

def build_tar_member_index(shard_paths: List[Path], out_index_jsonl: Path) -> None:
    """
    Create a member-level index:
      each line: shard path + member name + key prefix + tar offsets/sizes
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
    Create a sample-level manifest:
      each line: key + shard + list of members + decoded labels (if present)
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
# CLI / main
# -----------------------------

def _parse_csv_set(s: str) -> Set[str]:
    return {x.strip() for x in s.split(",") if x.strip()}

def _normalize_camids(camids_csv: str) -> Set[str]:
    """
    Normalize cam ids:
      - accept 'cam_001431...' or '001431...'
      - store as bare '001431...'
    """
    out: Set[str] = set()
    for x in _parse_csv_set(camids_csv):
        if x.startswith("cam_"):
            x = x.replace("cam_", "", 1)
        out.add(x)
    return out

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--split_json", type=str, required=True,
                    help="Split json with {'__meta__':..., 'samples':...} (train/val/test each one file)")
    ap.add_argument("--split_name", type=str, default="",
                    help="Optional split name (e.g., train/val/test). If set, outputs become split-aware filenames.")
    ap.add_argument("--out_dir", type=str, required=True)

    # Shard by bytes
    ap.add_argument("--shard_max_gb", type=float, default=2.0,
                    help="Target max shard size in GiB (1 GiB=1024^3). Default: 2.0")

    # Optional: cap by sample count
    ap.add_argument("--shard_size", type=int, default=0,
                    help="Optional cap: max samples per shard. 0 disables. (You can set 80 for dual-threshold.)")

    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--store_debug_json", action="store_true",
                    help="If set, store per-modality debug JSON members inside each sample.")
    ap.add_argument("--debug_json_filtered_cams", action="store_true",
                    help="If set, rgb/depth debug json will be filtered to selected camids (when cam filter is active).")

    # Key style control
    ap.add_argument(
        "--key_style",
        type=str,
        default="person_action_segment",
        choices=("person_action_segment", "sanitize_only"),
        help=(
            "How to convert split_json sample keys into tar-safe keys.\n"
            "  person_action_segment (default): person_action_segment -> J_adjust_slider_run_..._elbow\n"
            "  sanitize_only: replace /\\ with _ and keep original ordering\n"
        ),
    )

    # ✅ NEW: selective modalities + selective camids
    ap.add_argument(
        "--modalities",
        type=str,
        default="rgb,depth,lidar,emg",
        help="Comma-separated modalities to pack. Subset of {rgb,depth,lidar,emg}. Example: rgb,emg"
    )
    ap.add_argument(
        "--camids",
        type=str,
        default="",
        help=(
            "Only pack these camera ids for rgb/depth. "
            "Comma-separated. Accepts 'cam_001431...' or bare '001431...'. "
            "Empty means pack all cams."
        )
    )
    ap.add_argument(
        "--cam_apply_to",
        type=str,
        default="rgb,depth",
        help="Which modalities the cam filter applies to. Default: rgb,depth"
    )

    args = ap.parse_args()

    # Normalize split naming
    split_name = (args.split_name or "").strip()
    suffix = f"_{split_name}" if split_name else ""
    shard_prefix = f"shard-{split_name}" if split_name else "shard"

    out_dir = Path(args.out_dir)
    shards_dir = out_dir / "shards"
    ensure_dir(out_dir)
    ensure_dir(shards_dir)

    # Load split json
    meta, samples = load_split_json(Path(args.split_json))

    # -------------------------
    # Parse selective options
    # -------------------------
    valid_modalities = {"rgb", "depth", "lidar", "emg"}
    want_modalities = {m.strip().lower() for m in args.modalities.split(",") if m.strip()}
    want_modalities = want_modalities & valid_modalities
    if not want_modalities:
        raise ValueError(f"--modalities results in empty set. valid={sorted(list(valid_modalities))}")

    cam_apply_to = {m.strip().lower() for m in args.cam_apply_to.split(",") if m.strip()}
    cam_apply_to = cam_apply_to & {"rgb", "depth"}  # only makes sense for frame modalities

    want_camids = _normalize_camids(args.camids)

    # Deterministic ordering of sample keys
    original_keys = sorted(list(samples.keys()))
    if args.limit and args.limit > 0:
        original_keys = original_keys[:args.limit]

    # Worker tasks: include selection info
    tasks: List[Tuple[str, Dict[str, Any], bool, str, Set[str], Set[str], Set[str], bool]] = [
        (
            ok,
            samples[ok],
            bool(args.store_debug_json),
            args.key_style,
            want_modalities,
            want_camids,
            cam_apply_to,
            bool(args.debug_json_filtered_cams),
        )
        for ok in original_keys
    ]

    shard_max_bytes = int(args.shard_max_gb * (1024 ** 3))

    # Build plans in parallel; write shards in main process (streaming)
    with Pool(processes=args.num_workers) as pool:
        plan_iter = pool.imap_unordered(build_sample_plan, tasks, chunksize=8)
        shard_paths = write_shards_by_bytes(
            plan_iter,
            shards_dir=shards_dir,
            shard_name_prefix=shard_prefix,
            shard_max_bytes=shard_max_bytes,
            shard_size_cap=int(args.shard_size) if args.shard_size else 0,
        )

    # Save meta snapshot for traceability; include selection config used
    meta_out = dict(meta)
    meta_out["packer_key_style"] = args.key_style
    meta_out["packer_selective"] = {
        "want_modalities": sorted(list(want_modalities)),
        "want_camids": sorted(list(want_camids)),
        "cam_apply_to": sorted(list(cam_apply_to)),
        "store_debug_json": bool(args.store_debug_json),
        "debug_json_filtered_cams": bool(args.debug_json_filtered_cams),
        "shard_max_gb": float(args.shard_max_gb),
        "shard_size_cap": int(args.shard_size),
    }
    meta_path = out_dir / f"split_meta{suffix}.json"
    meta_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Text list of shards
    shards_txt = out_dir / f"shards{suffix}.txt"
    manifest_jsonl = out_dir / f"manifest{suffix}.jsonl"
    index_jsonl = out_dir / f"wds_index{suffix}.jsonl"

    with shards_txt.open("w", encoding="utf-8") as f:
        for sp in shard_paths:
            f.write(str(sp) + "\n")

    # Build indices (reads tar, can be slow for huge shards)
    build_sample_manifest(shard_paths, manifest_jsonl)
    build_tar_member_index(shard_paths, index_jsonl)

    print(f"✅ Done. shards={len(shard_paths)} split_name='{split_name}'")
    print(f"- key_style:                 {args.key_style}")
    print(f"- modalities:                {sorted(list(want_modalities))}")
    print(f"- camids:                    {sorted(list(want_camids)) if want_camids else 'ALL'}")
    print(f"- cam_apply_to:              {sorted(list(cam_apply_to))}")
    print(f"- shard_max_gb:              {args.shard_max_gb} (bytes={shard_max_bytes})")
    print(f"- shard_size_cap:            {args.shard_size}")
    print(f"- store_debug_json:          {bool(args.store_debug_json)}")
    print(f"- debug_json_filtered_cams:  {bool(args.debug_json_filtered_cams)}")
    print(f"- shards list:               {shards_txt}")
    print(f"- sample manifest:           {manifest_jsonl}")
    print(f"- tar index:                 {index_jsonl}")
    print(f"- split meta:                {meta_path}")


if __name__ == "__main__":
    main()
