#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
split_dataset_grouped_by_run_with_hierarchy_coverage_unified_json_v6_indexed.py

✅ What this v6 does:
  - SAME splitting logic as your previous version:
      * records are grouped by run: group_uid = person::run_token
      * random trials search to satisfy tier1 coverage across train/val/test (hard constraint)
      * tier2/tier3 coverage + size balance (soft objective)
  - NEW: Optimization A (fast):
      * build a file index ONCE per modality (single filesystem traversal per modality)
      * JSON generation uses O(1) dict lookup, no per-sample glob scanning
  - RGB/Depth:
      <root>/<person>/<action>/<segment>/
          cam_001431512812/*.png
          cam_001484412812/*.png
    JSON stores per-cam lists (cam split)

  - MindRove:
      <root>/<person>/<action>/<segment>_npy/
          left/*.npy  (prefer clip.npy)
          right/*.npy (prefer clip.npy)
    JSON stores dict: {"left": "...", "right": "..."} (only existing)

  - LiDAR:
      <root>/<person>/<action>/<segment>_npy/
          *.npy (prefer clip.npy)
    JSON stores a single npy path (string)

Key:
  key = person/action/segment_base   (segment_base strips trailing "_npy")

Output:
  - manifest.csv
  - run_split_map.csv
  - tier{1,2,3}_split_counts.csv
  - rare_actions_report.txt
  - train.json / val.json / test.json
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set, Any


# ------------------------- Defaults / constants -------------------------

DEFAULT_PERSONS = ("N", "M", "MR", "J")

TIER2_ACTIONS = [
    "adjust_slider",
    "cap_pen",
    "close_wire_cutter",
    "cut_wire",
    "cut_tape",
    "insert_wire_into_lug",
    "label_sample",
    "measure_wire_length",
    "move_ruler",
    "move_wire",
    "move_wire_cutter",
    "move_lug",
    "move_plier",
    "move_pen",
    "move_scissor",
    "move_tape",
    "open_wire_cutter",
    "position_wire",
    "press_wire_cutter",
    "pull_out_tape",
    "put_ruler",
    "put_tape",
    "put_wire",
    "put_lug",
    "put_plier",
    "put_pen",
    "put_cap",
    "put_sample",
    "put_scissor",
    "put_wire_cutter",
    "remove_cap",
    "remove_tape",
    "take_ruler",
    "take_pen",
    "take_tape",
    "take_wire",
    "take_lug",
    "take_plier",
    "take_cap",
    "take_sample",
    "take_scissor",
    "take_wire_cutter",
    "tear_tape",
    "wrap_sample_with_tape",
    "wrap_short_wire_with_tape",
]

LIGHTS = ("normal", "right", "left")
POSITIONS = ("elbow", "mid")
SPLITS = ("train", "val", "test")

DEFAULT_CAM_DIRS = ("cam_001431512812", "cam_001484412812")
RUN_TOKEN_RE = re.compile(r"^run_(\d+(?:-\d+)?)", re.IGNORECASE)


# ------------------------- Small string utils -------------------------

def norm_name(s: str) -> str:
    s = s.strip().lower().replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

def tokenize(name: str) -> List[str]:
    return [t for t in norm_name(name).split("_") if t]

def is_subsequence(needle: List[str], hay: List[str]) -> bool:
    if not needle:
        return True
    j = 0
    for tok in hay:
        if tok == needle[j]:
            j += 1
            if j == len(needle):
                return True
    return False

def strip_trailing_npy_suffix(seg_dirname: str) -> str:
    """Remove ONLY trailing '_npy' (case-insensitive)."""
    s = seg_dirname.strip()
    if s.lower().endswith("_npy"):
        return s[:-4]
    return s

def build_key(person: str, action_raw: str, segment_base: str) -> str:
    return f"{person}/{action_raw}/{segment_base}"


# ------------------------- Tier mapping -------------------------

def get_tier1_verb(tier3_action: str) -> str:
    t3 = tokenize(tier3_action)
    if len(t3) >= 2 and f"{t3[0]}_{t3[1]}" == "pull_out":
        return "pull_out"
    return t3[0] if t3 else "UNKNOWN"

def best_tier2_match(tier3_action: str, tier2_list: List[str]) -> Optional[str]:
    t3 = tokenize(tier3_action)
    verb = "pull_out" if (len(t3) >= 2 and f"{t3[0]}_{t3[1]}" == "pull_out") else (t3[0] if t3 else None)

    candidates: List[Tuple[int, str]] = []
    for t2 in tier2_list:
        t2n = tokenize(t2)
        if not t2n:
            continue
        t2_verb = "pull_out" if (len(t2n) >= 2 and f"{t2n[0]}_{t2n[1]}" == "pull_out") else t2n[0]
        if verb is not None and t2_verb != verb:
            continue
        if is_subsequence(t2n, t3):
            candidates.append((len(t2n), t2))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


# ------------------------- Segment meta parsing -------------------------

def parse_segment_meta_last2(seg_base: str) -> Tuple[str, str]:
    toks = tokenize(seg_base)
    if len(toks) < 2:
        return "unknown", "unknown"
    lighting = toks[-2]
    pos = toks[-1]
    if lighting not in LIGHTS:
        lighting = "unknown"
    if pos not in POSITIONS:
        pos = "unknown"
    return lighting, pos

def extract_run_token(seg_base: str) -> Optional[str]:
    m = RUN_TOKEN_RE.match(seg_base.strip())
    if not m:
        m = RUN_TOKEN_RE.match(norm_name(seg_base))
    if not m:
        return None
    return f"run_{m.group(1)}"


# ------------------------- Data model -------------------------

@dataclass
class SegmentRecord:
    person: str
    action_raw: str
    segment_raw: str
    segment_base: str

    tier3: str
    tier2: str
    tier1: str

    run_token: str
    group_uid: str

    lighting: str
    pos: str

    split: str = ""


# ------------------------- Arg helpers -------------------------

def parse_modality_args(modality_args: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in modality_args:
        if "=" not in item:
            raise ValueError(f"--modality expects name=PATH, got: {item}")
        name, p = item.split("=", 1)
        name = name.strip()
        p = p.strip().strip('"').strip("'")
        if not name:
            raise ValueError(f"Empty modality name in: {item}")
        out[name] = Path(p)
    return out

def choose_canonical_modality(modality_parents: Dict[str, Path]) -> str:
    """Prefer rgb, else choose first alphabetically."""
    if "rgb" in modality_parents:
        return "rgb"
    return sorted(modality_parents.keys())[0]


# ------------------------- Canonical scan (fast: only directories) -------------------------

def scan_canonical_records(
    canonical_root: Path,
    persons: Tuple[str, ...],
) -> List[SegmentRecord]:
    """
    Scan (person/action/segment) from canonical modality root.
    NOTE: This scan does NOT enumerate frames/npy files. It only walks directories.
    """
    if not canonical_root.exists():
        raise FileNotFoundError(f"Canonical root not found: {canonical_root}")

    tier2_norm = [norm_name(x) for x in TIER2_ACTIONS]
    records: List[SegmentRecord] = []

    for person in persons:
        person_dir = canonical_root / person
        if not person_dir.exists():
            raise FileNotFoundError(f"Canonical root missing person folder: {person_dir}")

        # list action folders
        for action_entry in os.scandir(person_dir):
            if not action_entry.is_dir():
                continue
            action_raw = action_entry.name

            tier3 = norm_name(action_raw)
            tier1 = norm_name(get_tier1_verb(tier3))
            tier2 = best_tier2_match(tier3, tier2_norm) or "UNMAPPED_T2"

            action_dir = Path(action_entry.path)

            # list segment folders
            for seg_entry in os.scandir(action_dir):
                if not seg_entry.is_dir():
                    continue
                seg_raw = seg_entry.name
                seg_base = strip_trailing_npy_suffix(seg_raw)

                run_token = extract_run_token(seg_base) or "run_UNKNOWN"
                lighting, pos = parse_segment_meta_last2(seg_base)

                records.append(
                    SegmentRecord(
                        person=person,
                        action_raw=action_raw,
                        segment_raw=seg_raw,
                        segment_base=seg_base,
                        tier3=tier3,
                        tier2=tier2,
                        tier1=tier1,
                        run_token=run_token,
                        group_uid=f"{person}::{run_token}",
                        lighting=lighting,
                        pos=pos,
                    )
                )

    return records


# ------------------------- Group helpers -------------------------

def build_groups(records: List[SegmentRecord]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[r.group_uid].append(i)
    return groups

def group_label_sets(records: List[SegmentRecord], groups: Dict[str, List[int]]) -> Dict[str, Dict[str, Set[str]]]:
    info: Dict[str, Dict[str, Set[str]]] = {}
    for gid, idxs in groups.items():
        t1, t2, t3 = set(), set(), set()
        for i in idxs:
            t1.add(records[i].tier1)
            t2.add(records[i].tier2)
            t3.add(records[i].tier3)
        info[gid] = {"tier1": t1, "tier2": t2, "tier3": t3}
    return info

def group_sizes(groups: Dict[str, List[int]]) -> Dict[str, int]:
    return {gid: len(idxs) for gid, idxs in groups.items()}


# ------------------------- Split search (same logic) -------------------------

def evaluate_assignment(
    assignment: Dict[str, str],
    group_info: Dict[str, Dict[str, Set[str]]],
    group_sz: Dict[str, int],
    all_t1: Set[str],
    target_counts: Dict[str, float],
) -> Tuple[float, Dict[str, Dict[str, Set[str]]], Dict[str, int]]:
    cov = {sp: {"tier1": set(), "tier2": set(), "tier3": set()} for sp in SPLITS}
    split_counts = {sp: 0 for sp in SPLITS}

    for gid, sp in assignment.items():
        split_counts[sp] += group_sz[gid]
        cov[sp]["tier1"].update(group_info[gid]["tier1"])
        cov[sp]["tier2"].update(group_info[gid]["tier2"])
        cov[sp]["tier3"].update(group_info[gid]["tier3"])

    # Hard constraint: all tier1 verbs must appear in each split
    missing_t1_total = sum(len(all_t1 - cov[sp]["tier1"]) for sp in SPLITS)
    if missing_t1_total > 0:
        score = -1e12 - 1e9 * missing_t1_total
        return score, cov, split_counts

    # Soft objective: maximize tier2 and tier3 coverage, penalize size deviation
    t2_score = sum(len(cov[sp]["tier2"]) for sp in SPLITS)
    t3_score = sum(len(cov[sp]["tier3"]) for sp in SPLITS)
    dev_pen = sum(abs(split_counts[sp] - target_counts[sp]) for sp in SPLITS)

    score = 1000.0 * t2_score + 100.0 * t3_score - 1.0 * dev_pen
    return score, cov, split_counts

def random_initial_assignment(group_ids: List[str], group_sz: Dict[str, int], target_counts: Dict[str, float], rng) -> Dict[str, str]:
    remaining = dict(target_counts)
    gids = group_ids[:]
    rng.shuffle(gids)
    assignment: Dict[str, str] = {}

    for gid in gids:
        best_sp, best_val = None, None
        for sp in SPLITS:
            val = remaining[sp] + rng.random() * 0.01
            if best_val is None or val > best_val:
                best_val = val
                best_sp = sp
        assignment[gid] = best_sp
        remaining[best_sp] -= group_sz[gid]

    return assignment

def search_best_split(
    group_ids: List[str],
    group_info: Dict[str, Dict[str, Set[str]]],
    group_sz: Dict[str, int],
    all_t1: Set[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    trials: int,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Set[str]]], Dict[str, int]]:
    import random
    rng = random.Random(seed)

    total_segments = sum(group_sz.values())
    target_counts = {
        "train": total_segments * train_ratio,
        "val": total_segments * val_ratio,
        "test": total_segments * test_ratio,
    }

    best_score = -1e30
    best_assignment: Dict[str, str] = {}
    best_cov = None
    best_counts = None

    for _ in range(trials):
        assn = random_initial_assignment(group_ids, group_sz, target_counts, rng)
        score, cov, scounts = evaluate_assignment(assn, group_info, group_sz, all_t1, target_counts)
        if score > best_score:
            best_score = score
            best_assignment = assn
            best_cov = cov
            best_counts = scounts

    if best_cov is None or best_counts is None:
        raise RuntimeError("Unexpected: no split found.")
    return best_assignment, best_cov, best_counts


# ------------------------- Reports / counts -------------------------

def per_tier_split_counts(records: List[SegmentRecord]) -> Dict[str, Dict[str, Counter]]:
    out = {
        "tier1": {sp: Counter() for sp in SPLITS},
        "tier2": {sp: Counter() for sp in SPLITS},
        "tier3": {sp: Counter() for sp in SPLITS},
    }
    for r in records:
        out["tier1"][r.split][r.tier1] += 1
        out["tier2"][r.split][r.tier2] += 1
        out["tier3"][r.split][r.tier3] += 1
    return out

def write_tier_split_counts_csv(path: Path, tier_name: str, split_counters: Dict[str, Counter]) -> None:
    labels = set()
    for sp in SPLITS:
        labels.update(split_counters[sp].keys())

    rows = []
    for lab in labels:
        tr = split_counters["train"].get(lab, 0)
        va = split_counters["val"].get(lab, 0)
        te = split_counters["test"].get(lab, 0)
        tot = tr + va + te
        rows.append((lab, tr, va, te, tot))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([tier_name, "train", "val", "test", "total"])
        for r in sorted(rows, key=lambda x: (-x[4], x[0])):
            w.writerow(list(r))

def count_groups_per_label(records: List[SegmentRecord], groups: Dict[str, List[int]], tier_attr: str) -> Dict[str, int]:
    label_to_groups: Dict[str, Set[str]] = defaultdict(set)
    for gid, idxs in groups.items():
        labels = {getattr(records[i], tier_attr) for i in idxs}
        for lab in labels:
            label_to_groups[lab].add(gid)
    return {lab: len(gset) for lab, gset in label_to_groups.items()}

def missing_labels_in_split(split_counter: Counter, all_labels: Set[str]) -> List[str]:
    return sorted([lab for lab in all_labels if split_counter.get(lab, 0) == 0])


# ------------------------- Index building (Optimization A) -------------------------

def pick_preferred_npy(npy_paths: List[str]) -> Optional[str]:
    """
    Prefer clip.npy if present; else first sorted path.
    Inputs are strings (full paths).
    """
    if not npy_paths:
        return None
    clip = [p for p in npy_paths if os.path.basename(p).lower() == "clip.npy"]
    if clip:
        return sorted(clip)[0]
    return sorted(npy_paths)[0]

def build_index_rgb_or_depth(
    mod_root: Path,
    persons: Tuple[str, ...],
    cam_dirs: Tuple[str, ...],
    frame_exts: Tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    """
    Build index for rgb/depth:
      key -> {
        "segment_dir": <abs segment dir>,
        "files": {cam_dir: [abs_frame_paths...] }   (only cams with frames)
      }
    Only traverses expected depth:
      mod_root/person/action/segment/cam/*.ext
    """
    index: Dict[str, Dict[str, Any]] = {}

    if not mod_root.exists():
        return index

    frame_exts_lc = tuple(e.lower() for e in frame_exts)

    for person in persons:
        person_dir = mod_root / person
        if not person_dir.exists():
            continue

        for action_entry in os.scandir(person_dir):
            if not action_entry.is_dir():
                continue
            action_raw = action_entry.name
            action_dir = Path(action_entry.path)

            for seg_entry in os.scandir(action_dir):
                if not seg_entry.is_dir():
                    continue
                seg_raw = seg_entry.name
                seg_dir = Path(seg_entry.path)

                seg_base = strip_trailing_npy_suffix(seg_raw)  # normally no _npy for rgb/depth, but safe
                key = build_key(person, action_raw, seg_base)

                files_by_cam: Dict[str, List[str]] = {}
                for cam in cam_dirs:
                    cam_path = seg_dir / cam
                    if not cam_path.exists():
                        continue
                    frames: List[str] = []
                    # scan files in cam folder
                    for f in os.scandir(cam_path):
                        if not f.is_file():
                            continue
                        # extension filter
                        ext = os.path.splitext(f.name)[1].lstrip(".").lower()
                        if ext in frame_exts_lc:
                            frames.append(os.path.abspath(f.path))
                    if frames:
                        frames.sort()
                        files_by_cam[cam] = frames

                if files_by_cam:
                    index[key] = {
                        "segment_dir": str(seg_dir.resolve()),
                        "files": files_by_cam,
                    }

    return index

def build_index_mindrove(
    mod_root: Path,
    persons: Tuple[str, ...],
    allow_suffix_npy: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Build index for mindrove:
      key -> {
        "segment_dir": <abs segment dir>,
        "files": {"left": <npy>, "right": <npy>}  (only hands that exist)
        "missing_hands": [...]
      }
    Expected:
      mod_root/person/action/segment[_npy]/left|right/*.npy
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not mod_root.exists():
        return index

    for person in persons:
        person_dir = mod_root / person
        if not person_dir.exists():
            continue

        for action_entry in os.scandir(person_dir):
            if not action_entry.is_dir():
                continue
            action_raw = action_entry.name
            action_dir = Path(action_entry.path)

            for seg_entry in os.scandir(action_dir):
                if not seg_entry.is_dir():
                    continue
                seg_raw = seg_entry.name
                seg_dir = Path(seg_entry.path)

                # Only accept suffix if allow_suffix_npy; otherwise still strip for key consistency.
                seg_base = strip_trailing_npy_suffix(seg_raw) if allow_suffix_npy else seg_raw
                key = build_key(person, action_raw, seg_base)

                files: Dict[str, str] = {}
                missing_hands: List[str] = []

                for hand in ("left", "right"):
                    hand_dir = seg_dir / hand
                    if not hand_dir.exists():
                        missing_hands.append(hand)
                        continue
                    npys: List[str] = []
                    for f in os.scandir(hand_dir):
                        if f.is_file() and f.name.lower().endswith(".npy"):
                            npys.append(os.path.abspath(f.path))
                    pick = pick_preferred_npy(npys)
                    if pick is None:
                        missing_hands.append(hand)
                    else:
                        files[hand] = pick

                if files:
                    index[key] = {
                        "segment_dir": str(seg_dir.resolve()),
                        "files": files,  # dict(left/right)
                        "missing_hands": missing_hands,
                    }

    return index

def build_index_lidar(
    mod_root: Path,
    persons: Tuple[str, ...],
    allow_suffix_npy: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Build index for lidar:
      key -> {
        "segment_dir": <abs segment dir>,
        "files": <npy_path_string>
      }
    Expected:
      mod_root/person/action/segment[_npy]/*.npy
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not mod_root.exists():
        return index

    for person in persons:
        person_dir = mod_root / person
        if not person_dir.exists():
            continue

        for action_entry in os.scandir(person_dir):
            if not action_entry.is_dir():
                continue
            action_raw = action_entry.name
            action_dir = Path(action_entry.path)

            for seg_entry in os.scandir(action_dir):
                if not seg_entry.is_dir():
                    continue
                seg_raw = seg_entry.name
                seg_dir = Path(seg_entry.path)

                seg_base = strip_trailing_npy_suffix(seg_raw) if allow_suffix_npy else seg_raw
                key = build_key(person, action_raw, seg_base)

                npys: List[str] = []
                for f in os.scandir(seg_dir):
                    if f.is_file() and f.name.lower().endswith(".npy"):
                        npys.append(os.path.abspath(f.path))
                pick = pick_preferred_npy(npys)
                if pick is not None:
                    index[key] = {
                        "segment_dir": str(seg_dir.resolve()),
                        "files": pick,  # string
                    }

    return index

def build_indexes(
    modality_parents: Dict[str, Path],
    persons: Tuple[str, ...],
    cam_dirs: Tuple[str, ...],
    frame_exts: Tuple[str, ...],
    suffix_modalities: Set[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build one index per modality.
    Returns:
      indexes[modality_name] = dict(key -> {"segment_dir":..., "files":...})
    """
    indexes: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for mn, root in modality_parents.items():
        root = Path(root)

        if mn in ("rgb", "depth"):
            indexes[mn] = build_index_rgb_or_depth(root, persons, cam_dirs, frame_exts)
        elif mn == "mindrove":
            indexes[mn] = build_index_mindrove(root, persons, allow_suffix_npy=(mn in suffix_modalities))
        elif mn == "lidar":
            indexes[mn] = build_index_lidar(root, persons, allow_suffix_npy=(mn in suffix_modalities))
        else:
            # Fallback: treat unknown modality as lidar-like npy-in-segment
            indexes[mn] = build_index_lidar(root, persons, allow_suffix_npy=(mn in suffix_modalities))

    return indexes


# ------------------------- JSON writing (O(1) lookup) -------------------------

def write_split_jsons_from_indexes(
    records: List[SegmentRecord],
    out_dir: Path,
    modality_parents: Dict[str, Path],
    indexes: Dict[str, Dict[str, Dict[str, Any]]],
    suffix_modalities: Set[str],
    cam_dirs: Tuple[str, ...],
    frame_exts: Tuple[str, ...],
    require_all_modalities: bool,
) -> None:
    """
    Generate train.json / val.json / test.json using the pre-built indexes.
    No filesystem scanning happens here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mod_names = sorted(modality_parents.keys())

    for sp in SPLITS:
        split_records = [r for r in records if r.split == sp]
        samples: Dict[str, Any] = {}
        unused: List[str] = []

        for r in split_records:
            key = build_key(r.person, r.action_raw, r.segment_base)

            files_by_mod: Dict[str, Any] = {}
            segdirs_by_mod: Dict[str, str] = {}
            missing_mods: List[str] = []
            extra: Dict[str, Any] = {}

            for mn in mod_names:
                idx = indexes.get(mn, {})
                hit = idx.get(key)

                if hit is None:
                    missing_mods.append(mn)
                    continue

                files_by_mod[mn] = hit["files"]
                segdirs_by_mod[mn] = hit["segment_dir"]

                # keep extra mindrove info if present
                if mn == "mindrove" and "missing_hands" in hit:
                    extra["mindrove_missing_hands"] = hit["missing_hands"]

            if require_all_modalities and missing_mods:
                unused.append(key)
                continue

            sample_obj = {
                "person": r.person,
                "action": r.action_raw,
                "segment": r.segment_base,
                "tier1": r.tier1,
                "tier2": r.tier2,
                "tier3": r.tier3,
                "run_token": r.run_token,
                "lighting": r.lighting,
                "pos": r.pos,
                "files": files_by_mod,
                "segment_dirs": segdirs_by_mod,
                "missing_modalities": missing_mods,
            }
            if extra:
                sample_obj["extra"] = extra

            samples[key] = sample_obj

        payload = {
            "__meta__": {
                "split": sp,
                "modalities": mod_names,
                "suffix_modalities": sorted(list(suffix_modalities)),
                "cam_dirs": list(cam_dirs),
                "frame_exts": list(frame_exts),
                "require_all_modalities": require_all_modalities,
                "unused_samples": unused,
                "modality_parents": {k: str(Path(v).resolve()) for k, v in modality_parents.items()},
                "note": (
                    "This JSON is generated using pre-built modality indexes (Optimization A). "
                    "rgb/depth store per-cam frame lists. mindrove stores dict(left/right)->npy. "
                    "lidar stores a single npy path string."
                ),
            },
            "samples": samples,
        }

        (out_dir / f"{sp}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------- CSV outputs -------------------------

def write_manifest_csv(out_csv: Path, records: List[SegmentRecord]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "person", "action_raw", "segment_raw", "segment_base",
            "tier3", "tier2", "tier1",
            "run_token", "group_uid",
            "lighting", "pos",
            "split",
        ])
        for r in records:
            w.writerow([
                r.person, r.action_raw, r.segment_raw, r.segment_base,
                r.tier3, r.tier2, r.tier1,
                r.run_token, r.group_uid,
                r.lighting, r.pos,
                r.split,
            ])

def write_run_split_map_csv(out_csv: Path, groups: Dict[str, List[int]], records: List[SegmentRecord], assignment: Dict[str, str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for gid, idxs in groups.items():
        person = records[idxs[0]].person if idxs else "UNKNOWN"
        run_token = records[idxs[0]].run_token if idxs else "run_UNKNOWN"
        split = assignment.get(gid, "")
        rows.append((person, run_token, gid, split, len(idxs)))

    def run_sort_key(rt: str) -> Tuple[int, str]:
        m = re.match(r"run_(\d+)(?:-(\d+))?$", rt)
        if m:
            return (int(m.group(1)), rt)
        return (10**9, rt)

    rows.sort(key=lambda x: (x[0], run_sort_key(x[1]), x[2]))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["person", "run_token", "group_uid", "split", "n_segments_in_group"])
        for r in rows:
            w.writerow(list(r))


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--modality",
        action="append",
        required=True,
        help="Repeatable: name=PATH. Example: --modality rgb=... --modality depth=... --modality mindrove=... --modality lidar=...",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--persons", default="N,M,MR,J", help="Comma-separated persons. Default: N,M,MR,J")

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)

    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--suffix_modalities",
        default="mindrove,lidar",
        help="Comma-separated modality names that may use segment folder suffix '_npy'. Default: mindrove,lidar",
    )

    ap.add_argument("--frame_exts", default="png", help="Comma-separated frame extensions for rgb/depth. Default: png")

    ap.add_argument(
        "--cam_dirs",
        default="cam_001431512812,cam_001484412812",
        help="Comma-separated camera subfolders under rgb/depth segment. Default: cam_001431512812,cam_001484412812",
    )

    ap.add_argument("--require_all_modalities", action="store_true", help="If set, drop samples missing any modality files.")

    args = ap.parse_args()

    s = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {s}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    persons = tuple([p.strip() for p in args.persons.split(",") if p.strip()])
    modality_parents = parse_modality_args(args.modality)

    frame_exts = tuple(e.strip().lower() for e in args.frame_exts.split(",") if e.strip())
    cam_dirs = tuple(x.strip() for x in args.cam_dirs.split(",") if x.strip())
    suffix_modalities = {x.strip() for x in args.suffix_modalities.split(",") if x.strip()}

    # 1) Canonical records scan (directory-only scan)
    canonical_name = choose_canonical_modality(modality_parents)
    canonical_root = Path(modality_parents[canonical_name])
    records = scan_canonical_records(canonical_root, persons=persons)

    all_t1 = {r.tier1 for r in records}
    all_t2 = {r.tier2 for r in records}
    all_t3 = {r.tier3 for r in records}

    # 2) Build groups by run
    groups = build_groups(records)
    ginfo = group_label_sets(records, groups)
    gsz = group_sizes(groups)
    group_ids = list(groups.keys())

    # 3) Find best split assignment
    assignment, _cov, split_counts = search_best_split(
        group_ids=group_ids,
        group_info=ginfo,
        group_sz=gsz,
        all_t1=all_t1,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        trials=args.trials,
        seed=args.seed,
    )

    for r in records:
        r.split = assignment[r.group_uid]

    # 4) CSV outputs
    manifest_path = out_dir / "manifest.csv"
    run_map_path = out_dir / "run_split_map.csv"
    write_manifest_csv(manifest_path, records)
    write_run_split_map_csv(run_map_path, groups, records, assignment)

    # 5) Count tables + rare report
    counts = per_tier_split_counts(records)
    write_tier_split_counts_csv(out_dir / "tier1_split_counts.csv", "tier1", counts["tier1"])
    write_tier_split_counts_csv(out_dir / "tier2_split_counts.csv", "tier2", counts["tier2"])
    write_tier_split_counts_csv(out_dir / "tier3_split_counts.csv", "tier3", counts["tier3"])

    t2_groups_per = count_groups_per_label(records, groups, tier_attr="tier2")
    t3_groups_per = count_groups_per_label(records, groups, tier_attr="tier3")
    rare_t2 = sorted([k for k, v in t2_groups_per.items() if v < 3])
    rare_t3 = sorted([k for k, v in t3_groups_per.items() if v < 3])

    missing_t1 = {sp: missing_labels_in_split(counts["tier1"][sp], all_t1) for sp in SPLITS}
    missing_t2 = {sp: missing_labels_in_split(counts["tier2"][sp], all_t2) for sp in SPLITS}
    missing_t3 = {sp: missing_labels_in_split(counts["tier3"][sp], all_t3) for sp in SPLITS}

    report_path = out_dir / "rare_actions_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("=== Canonical scan ===\n")
        f.write(f"canonical_modality = {canonical_name}\n")
        f.write(f"canonical_root     = {str(canonical_root.resolve())}\n\n")

        f.write("=== Split summary (segment counts) ===\n")
        for sp in SPLITS:
            f.write(f"{sp}: {split_counts[sp]}\n")
        f.write("\n")

        f.write("=== Tier1 coverage check (target: full in all splits) ===\n")
        for sp in SPLITS:
            f.write(f"{sp} missing Tier1 labels ({len(missing_t1[sp])}): {missing_t1[sp]}\n")
        f.write("\n")

        f.write("=== Labels impossible to cover in all 3 splits (too few distinct run-groups) ===\n")
        f.write(f"Tier2 labels with <3 groups ({len(rare_t2)}):\n")
        for lab in rare_t2:
            f.write(f"  {lab}  (groups={t2_groups_per[lab]})\n")
        f.write("\n")
        f.write(f"Tier3 labels with <3 groups ({len(rare_t3)}):\n")
        for lab in rare_t3:
            f.write(f"  {lab}  (groups={t3_groups_per[lab]})\n")
        f.write("\n")

        f.write("=== Missing labels in the chosen split (practical coverage) ===\n")
        f.write("Tier2 missing per split:\n")
        for sp in SPLITS:
            f.write(f"  {sp} missing ({len(missing_t2[sp])}): {missing_t2[sp][:80]}")
            if len(missing_t2[sp]) > 80:
                f.write(" ...")
            f.write("\n")
        f.write("\n")

        f.write("Tier3 missing per split:\n")
        for sp in SPLITS:
            f.write(f"  {sp} missing ({len(missing_t3[sp])}): {missing_t3[sp][:80]}")
            if len(missing_t3[sp]) > 80:
                f.write(" ...")
            f.write("\n")

    # 6) Build indexes ONCE per modality (Optimization A)
    indexes = build_indexes(
        modality_parents=modality_parents,
        persons=persons,
        cam_dirs=cam_dirs,
        frame_exts=frame_exts,
        suffix_modalities=suffix_modalities,
    )

    # 7) Write train/val/test JSON using indexes (fast lookup)
    write_split_jsons_from_indexes(
        records=records,
        out_dir=out_dir,
        modality_parents=modality_parents,
        indexes=indexes,
        suffix_modalities=suffix_modalities,
        cam_dirs=cam_dirs,
        frame_exts=frame_exts,
        require_all_modalities=args.require_all_modalities,
    )

    print("Done.")
    print(f"- manifest: {manifest_path}")
    print(f"- run map : {run_map_path}")
    print(f"- report  : {report_path}")
    print(f"- json    : {out_dir}/train.json , val.json , test.json")
    print(f"  canonical_modality = {canonical_name}")
    print(f"  suffix_modalities  = {sorted(list(suffix_modalities))}")
    print(f"  cam_dirs           = {list(cam_dirs)}")
    print(f"  frame_exts         = {list(frame_exts)}")


if __name__ == "__main__":
    main()
