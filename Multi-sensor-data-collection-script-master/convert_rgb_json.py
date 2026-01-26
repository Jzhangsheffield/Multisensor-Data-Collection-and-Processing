#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Traverse RGB frame folders and create ONE json per camera.

Folder layout (as you described):
RGB_ROOT/
  N|J|MR|M/
    action_name/                             e.g. adjust_slider
      run_5_clip_000003_normal/              clip folder (single run id)
        cam_001431512812/
          20251113_170325_880571.jpg
          ...
      run_8-37_clip_000003_normal/           clip folder (run range)
        cam_001431512812/
          ...

Output:
OUT_DIR/
  cam_001431512812.json
  cam_001484412812.json
  ...

JSON format (per camera):
{
  "J_adjust_slider_run_5_clip_000003_normal": {
      "cam": "cam_001431512812",
      "frames": ["D:\\...\\000001.jpg", "..."],
      "n_frames": 67,
      "frame_dir": "D:\\...\\cam_001431512812",
      "run": "5",
      "clip": "000003",
      "tag": "normal"
  },
  "J_adjust_slider_run_8-37_clip_000003_normal": {
      ...
  }
}

Key rule (user-defined):
- key = "{SUBJECT}_{ACTION}_run_{RUN}_clip_{CLIP}_{TAG}"
  where:
    SUBJECT = top-level folder under RGB (N/J/MR/M)
    ACTION  = action folder name
    RUN     = run id string, can be "5" or "8-37"
    CLIP    = clip id string, e.g. "000003"
    TAG     = tail string, e.g. "normal"/"left"/"right"/...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple


# -----------------------------
# Parsing helpers
# -----------------------------

# Support both:
#   run_5_clip_000003_normal
#   run_8-37_clip_000003_normal
#
# group(1) = run string ("5" or "8-37")
# group(2) = clip string ("000003")
# group(3) = tag string  ("normal" / "left" / "right" / ...)
CLIP_RE = re.compile(r"^run_([0-9]+(?:[-_][0-9]+)?)_clip_([0-9]+)_([A-Za-z0-9_]+)$")

IMG_EXTS = {".jpg", ".jpeg", ".png"}  # allow png just in case


def parse_clip_folder_name(name: str) -> Tuple[str, str, str]:
    """
    Parse clip folder name into (run_str, clip_str, tag).
    """
    m = CLIP_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized clip folder name: {name}")
    run_str, clip_str, tag = m.group(1), m.group(2), m.group(3)
    return run_str, clip_str, tag


def list_frames_sorted(frame_dir: Path) -> List[str]:
    """
    List frame files in a camera folder, sorted by filename.

    Your filenames are like: YYYYMMDD_HHMMSS_micro.jpg
    Lexicographic sort matches chronological order in that format.
    Returns absolute path strings.
    """
    frames = []
    for p in frame_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            frames.append(p)
    frames.sort(key=lambda x: x.name)
    return [str(p) for p in frames]


# -----------------------------
# Main traversal
# -----------------------------

def build_camera_jsons(rgb_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Return:
      cam_to_entries:
        {
          "cam_001431512812": {
              key1: entry1,
              key2: entry2,
              ...
          },
          ...
        }
    """
    cam_to_entries: Dict[str, Dict[str, Any]] = {}

    # Traverse: subject/action/clip/cam/frames
    for subject_dir in sorted([p for p in rgb_root.iterdir() if p.is_dir()]):
        subject = subject_dir.name  # N, J, MR, M

        for action_dir in sorted([p for p in subject_dir.iterdir() if p.is_dir()]):
            action = action_dir.name  # adjust_slider, etc.

            for clip_dir in sorted([p for p in action_dir.iterdir() if p.is_dir()]):
                clip_name = clip_dir.name

                try:
                    run_str, clip_str, tag = parse_clip_folder_name(clip_name)
                except ValueError:
                    # skip folders that do not match your naming rule
                    print(f"[SKIP] Unrecognized clip folder: {clip_name}")
                    continue

                # Build the key you requested
                key = f"{subject}_{action}_run_{run_str}_clip_{clip_str}_{tag}"

                # Under clip_dir: camera folders
                for cam_dir in sorted([p for p in clip_dir.iterdir() if p.is_dir()]):
                    cam = cam_dir.name  # cam_001431512812

                    frames = list_frames_sorted(cam_dir)
                    if not frames:
                        continue

                    entry = {
                        "cam": cam,
                        "frames": frames,
                        "n_frames": len(frames),
                        "frame_dir": str(cam_dir),
                        "run": run_str,
                        "clip": clip_str,
                        "tag": tag,
                    }

                    cam_to_entries.setdefault(cam, {})
                    # If duplicate key appears for same cam (shouldn't), later one overwrites.
                    cam_to_entries[cam][key] = entry

    return cam_to_entries


def save_camera_jsons(cam_to_entries: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam, entries in sorted(cam_to_entries.items(), key=lambda x: x[0]):
        out_path = out_dir / f"{cam}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root", type=str, required=True,
                    help=r'RGB root directory, e.g. D:\...\Thermal_Crimping_Dataset\kinect\RGB')
    ap.add_argument("--out_dir", type=str, required=True,
                    help=r'Output directory for per-camera jsons')
    args = ap.parse_args()

    rgb_root = Path(args.rgb_root)
    out_dir = Path(args.out_dir)

    cam_to_entries = build_camera_jsons(rgb_root)
    save_camera_jsons(cam_to_entries, out_dir)

    # Summary
    n_cams = len(cam_to_entries)
    n_keys_total = sum(len(v) for v in cam_to_entries.values())
    print(f"✅ Done. cameras={n_cams}, total clip-entries={n_keys_total}")
    for cam, entries in sorted(cam_to_entries.items()):
        print(f"  - {cam}: {len(entries)} clips -> {out_dir / (cam + '.json')}")

if __name__ == "__main__":
    main()
