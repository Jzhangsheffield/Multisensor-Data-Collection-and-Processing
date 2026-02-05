#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimate time offset (and optional drift) between RGB (30 FPS) and LiDAR (10 FPS)
using motion-energy signals derived from:
- RGB: frame difference energy (midpoint timestamp between frames)
- LiDAR: distance-histogram difference (robust) and/or centroid speed and/or point count change
  (also midpoint timestamp between consecutive LiDAR frames)

Input layout (example):
RGB:
  ...\\kinect\\run_1\\001431512812\\frames_rgb_blured\\20251105_180412_789655.jpg

LiDAR:
  ...\\vlp16\\run_1\\20251105_172532\\20251105_180412_686366_cloud.csv

CSV columns: x, y, z, intensity
Many rows can be all-zeros due to angular crop -> filtered.

Per-subject outputs (each subject into its own folder):
- <subject_out_dir>/plots/<run_name>__<lidar_session>__<lidar_feature>.png
- <subject_out_dir>/run_offsets.json      : summary list (written at end)
- <subject_out_dir>/run_offsets.jsonl     : one JSON per run (append; written immediately after each run)
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import matplotlib.pyplot as plt


TS_RE = re.compile(r"(\d{8}_\d{6}_\d{6})")  # YYYYMMDD_HHMMSS_micro


# -------------------------- time parsing --------------------------
def parse_ts_from_name(name: str) -> Optional[float]:
    """
    Parse 'YYYYMMDD_HHMMSS_micro' from filename and return "epoch seconds" as float.

    IMPORTANT (Windows):
      datetime.timestamp() can raise OSError for certain ranges.
      So we compute (dt - epoch).total_seconds() which is Windows-safe.

    Returns:
      float epoch seconds, or None if cannot parse.
    """
    m = TS_RE.search(name)
    if not m:
        return None
    s = m.group(1)
    try:
        dt = datetime.strptime(s, "%Y%m%d_%H%M%S_%f")
    except Exception:
        return None

    epoch = datetime(1970, 1, 1)
    return (dt - epoch).total_seconds()


def list_files_sorted_by_ts(folder: Path, suffix: str) -> List[Tuple[float, Path]]:
    """
    List files in folder matching suffix, parse timestamps from filenames,
    drop invalid/bogus timestamps (e.g., 1970), return sorted by time.

    We filter out any timestamp earlier than 2010-01-01 to remove bogus frames like:
      19700101_010000_000000.jpg
    """
    items: List[Tuple[float, Path]] = []
    for p in folder.glob(f"*{suffix}"):
        t = parse_ts_from_name(p.name)
        if t is None or not np.isfinite(t):
            continue

        # Filter bogus epoch-like timestamps
        if t < 1262304000.0:  # 2010-01-01 00:00:00
            continue

        items.append((t, p))

    items.sort(key=lambda x: x[0])
    return items


# -------------------------- signal utils --------------------------
def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standard z-score normalization (mean/std)."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean() if x.size else 0.0
    sd = x.std() if x.size else 1.0
    if sd < eps:
        return x * 0.0
    return (x - mu) / (sd + eps)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """
    Length-preserving moving average.
    Ensures output length equals input length even if win is even.
    """
    x = np.asarray(x, dtype=np.float64)
    if win <= 1 or x.size == 0:
        return x
    win = int(win)

    # pad_left + pad_right = win - 1
    pad_left = (win - 1) // 2
    pad_right = (win - 1) - pad_left

    xp = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    y = np.convolve(xp, kernel, mode="valid")  # length == len(x)

    # safety
    if y.shape[0] != x.shape[0]:
        y = y[:x.shape[0]]
    return y


def interp_to_grid(t: np.ndarray, v: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate samples (t, v) onto a unified time grid (t_grid).
    Out-of-range is clamped to boundary values via np.clip.
    """
    if len(t) < 2:
        return np.zeros_like(t_grid, dtype=np.float64)
    tmin, tmax = t[0], t[-1]
    vg = np.interp(np.clip(t_grid, tmin, tmax), t, v)
    return vg


def corr_at_shift(a: np.ndarray, b: np.ndarray) -> float:
    """
    Correlation coefficient between same-length arrays a and b.
    """
    if a.size == 0 or b.size == 0:
        return -1.0
    aa = a - a.mean()
    bb = b - b.mean()
    denom = (np.sqrt((aa * aa).sum()) * np.sqrt((bb * bb).sum()))
    if denom < 1e-12:
        return -1.0
    return float((aa * bb).sum() / denom)


def estimate_offset_by_scan(
    tg: np.ndarray,
    rgb_g: np.ndarray,
    lidar_g: np.ndarray,
    search_s: float,
    step_s: float,
) -> Tuple[float, float]:
    """
    Scan delta in [-search_s, +search_s] and find best correlation.

    Convention:
      delta > 0 means compare rgb(t) with lidar(t + delta)
      i.e., LiDAR is shifted later in the comparison.
    """
    deltas = np.arange(-search_s, search_s + 1e-9, step_s, dtype=np.float64)

    best_delta = 0.0
    best_corr = -1e9

    tmin, tmax = tg[0], tg[-1]

    for d in deltas:
        # Overlap region when shifting LiDAR by d
        mask = (tg + d >= tmin) & (tg + d <= tmax)

        # Need enough overlap, at least ~0.5 seconds worth of samples
        if mask.sum() < max(50, int(0.5 / (tg[1] - tg[0]))):
            continue

        a = rgb_g[mask]
        b = np.interp(tg[mask] + d, tg, lidar_g)
        c = corr_at_shift(a, b)

        if c > best_corr:
            best_corr = c
            best_delta = float(d)

    return best_delta, float(best_corr)


# -------------------------- RGB motion signal --------------------------
def compute_rgb_motion(
    frames: List[Tuple[float, Path]],
    down_w: int = 160,
    down_h: int = 90,
    diff_threshold: int = 0,
    smooth_win_s: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RGB motion signal from consecutive RGB frames.

    Returns:
      t_mid: midpoint timestamps between frame i and i+1
      m_rgb: motion magnitude (mean abs diff, or thresholded ratio)

    NOTE:
      Since this is an "edge" measure between frames,
      midpoint timestamp is the best aligned time for the measurement.
    """
    if len(frames) < 2:
        return np.array([]), np.array([])

    t0, p0 = frames[0]
    im0 = cv2.imread(str(p0), cv2.IMREAD_GRAYSCALE)
    if im0 is None:
        raise RuntimeError(f"Failed to read image: {p0}")
    im0 = cv2.resize(im0, (down_w, down_h), interpolation=cv2.INTER_AREA)

    t_mid = []
    m = []

    for (t1, p1) in frames[1:]:
        im1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
        if im1 is None:
            continue
        im1 = cv2.resize(im1, (down_w, down_h), interpolation=cv2.INTER_AREA)

        d = cv2.absdiff(im1, im0).astype(np.float32)

        if diff_threshold > 0:
            val = float((d > diff_threshold).mean())
        else:
            val = float(d.mean())

        t_mid.append(0.5 * (t0 + t1))
        m.append(val)

        t0, im0 = t1, im1

    t_mid = np.asarray(t_mid, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)

    # Smooth while preserving length
    if len(t_mid) >= 3 and smooth_win_s > 0:
        dt = np.median(np.diff(t_mid))
        win = max(1, int(round(smooth_win_s / dt)))
        m = moving_average(m, win)

    return t_mid, m


# -------------------------- LiDAR motion signal --------------------------
def load_lidar_points_csv(csv_path: Path) -> np.ndarray:
    """
    Load LiDAR CSV with columns x,y,z,intensity.
    Filters out:
      - NaN rows
      - all-zero rows (x=y=z=intensity=0)
    Handles header/no-header.

    Returns:
      Nx4 float32 array
    """
    try:
        arr = np.genfromtxt(str(csv_path), delimiter=",", dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {csv_path}: {e}")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # If header caused issues, try skip_header
    if arr.shape[1] < 4:
        arr2 = np.genfromtxt(str(csv_path), delimiter=",", dtype=np.float32, skip_header=1)
        if arr2.ndim == 1:
            arr2 = arr2.reshape(1, -1)
        arr = arr2

    if arr.shape[1] > 4:
        arr = arr[:, :4]

    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    nz = np.any(arr != 0.0, axis=1)
    arr = arr[nz]
    return arr.astype(np.float32, copy=False)


def compute_lidar_motion(
    frames: List[Tuple[float, Path]],
    feature: str = "histdiff",
    hist_bins: int = 40,
    r_max: float = 3.0,
    smooth_win_s: float = 0.3,
    min_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LiDAR motion signal from consecutive LiDAR frames.

    Returns:
      t_mid: midpoint timestamps between LiDAR frame i and i+1
      m_lidar: motion magnitude (depending on feature)

    feature:
      - histdiff: L1 difference between distance histograms (robust)
      - centroid: centroid speed (norm(c1-c0) / dt)
      - count   : abs difference of point counts
    """
    if len(frames) < 2:
        return np.array([]), np.array([])

    def frame_feat(points: np.ndarray):
        if points.shape[0] < min_points:
            return None

        xyz = points[:, :3]
        if feature == "centroid":
            return xyz.mean(axis=0)
        elif feature == "count":
            return float(points.shape[0])
        elif feature == "histdiff":
            r = np.linalg.norm(xyz, axis=1)
            r = r[(r >= 0) & (r <= r_max)]
            if r.size < min_points:
                return None
            h, _ = np.histogram(r, bins=hist_bins, range=(0.0, r_max), density=True)
            return h.astype(np.float32)
        else:
            raise ValueError(f"Unknown feature: {feature}")

    feats = []
    ts = []
    for t, p in frames:
        pts = load_lidar_points_csv(p)
        feats.append(frame_feat(pts))
        ts.append(t)

    t_mid = []
    m = []
    for i in range(1, len(frames)):
        f0 = feats[i - 1]
        f1 = feats[i]
        if f0 is None or f1 is None:
            continue

        t0 = ts[i - 1]
        t1 = ts[i]
        if t1 <= t0:
            continue

        if feature == "centroid":
            val = float(np.linalg.norm(f1 - f0) / (t1 - t0))
        elif feature == "count":
            val = float(abs(f1 - f0))
        elif feature == "histdiff":
            val = float(np.abs(f1 - f0).sum())
        else:
            val = 0.0

        t_mid.append(0.5 * (t0 + t1))
        m.append(val)

    t_mid = np.asarray(t_mid, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)

    if len(t_mid) >= 3 and smooth_win_s > 0:
        dt = np.median(np.diff(t_mid))
        win = max(1, int(round(smooth_win_s / dt)))
        m = moving_average(m, win)

    return t_mid, m


# -------------------------- drift estimation --------------------------
def estimate_drift_piecewise(
    tg: np.ndarray,
    rgb_g: np.ndarray,
    lidar_g: np.ndarray,
    search_s: float,
    step_s: float,
    win_s: float = 20.0,
    hop_s: float = 10.0,
) -> Dict[str, float]:
    """
    Estimate drift by splitting timeline into windows and estimating offset per window,
    then fitting offset(T) = p*T + q.

    Returns:
      slope p, intercept q, and mean correlation across windows.
    """
    t0, t1 = tg[0], tg[-1]
    centers = []
    offsets = []
    corrs = []

    t = t0
    while t + win_s <= t1 + 1e-9:
        mask = (tg >= t) & (tg <= t + win_s)
        if mask.sum() < 200:
            t += hop_s
            continue
        d, c = estimate_offset_by_scan(tg[mask], rgb_g[mask], lidar_g[mask], search_s, step_s)
        centers.append(float(t + 0.5 * win_s))
        offsets.append(float(d))
        corrs.append(float(c))
        t += hop_s

    if len(centers) < 2:
        return {
            "slope": 0.0,
            "intercept": float(offsets[0]) if offsets else 0.0,
            "mean_corr": float(np.mean(corrs)) if corrs else -1.0,
        }

    X = np.asarray(centers, dtype=np.float64)
    Y = np.asarray(offsets, dtype=np.float64)

    A = np.vstack([X, np.ones_like(X)]).T
    p, q = np.linalg.lstsq(A, Y, rcond=None)[0]

    return {"slope": float(p), "intercept": float(q), "mean_corr": float(np.mean(corrs))}


# -------------------------- plotting --------------------------
def plot_alignment(
    out_png: Path,
    t_grid: np.ndarray,
    rgb_g: np.ndarray,
    lidar_g: np.ndarray,
    best_delta: float,
    title: str,
):
    """
    Plot (z-scored) RGB and LiDAR motion signals on the unified grid,
    plus LiDAR shifted by best_delta.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(t_grid - t_grid[0], zscore(rgb_g), label="RGB motion (grid, z)")
    plt.plot(t_grid - t_grid[0], zscore(lidar_g), label="LiDAR motion (grid, z)")

    lidar_shifted = np.interp(t_grid + best_delta, t_grid, lidar_g, left=np.nan, right=np.nan)
    plt.plot(
        t_grid - t_grid[0],
        zscore(np.nan_to_num(lidar_shifted, nan=0.0)),
        label=f"LiDAR shifted by Δt={best_delta:+.3f}s",
    )

    plt.xlabel("Time since start (s)")
    plt.ylabel("Z-scored motion")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -------------------------- per-run processing --------------------------
def process_one_run(
    rgb_dir: Path,
    lidar_dir: Path,
    out_dir: Path,
    lidar_feature: str,
    search_s: float,
    step_s: float,
    grid_dt: float,
    do_drift: bool,
    drift_win_s: float,
    drift_hop_s: float,
    lidar_r_max: float,
    lidar_hist_bins: int,
    warmup_s: float,
) -> Dict:
    """
    Process one run:
      1) list RGB frames and LiDAR frames (sorted by timestamp, filtered)
      2) compute motion signals (RGB and LiDAR) with midpoint timestamps
      3) compute intersection time range
      4) drop warm-up seconds AFTER intersection start
      5) interpolate both signals to unified time grid
      6) scan offset that maximizes correlation
      7) optional drift estimation
      8) save plot
      9) return a dict with results and metadata
    """
    rgb_frames = list_files_sorted_by_ts(rgb_dir, ".jpg")
    if not rgb_frames:
        raise RuntimeError(f"No RGB frames found in: {rgb_dir}")

    lidar_frames = list_files_sorted_by_ts(lidar_dir, ".csv")
    lidar_frames = [(t, p) for (t, p) in lidar_frames if p.name.endswith("_cloud.csv")]
    if not lidar_frames:
        raise RuntimeError(f"No LiDAR CSV frames found in: {lidar_dir}")

    # 1) motion signals (midpoint timestamps)
    t_rgb, m_rgb = compute_rgb_motion(
        rgb_frames, down_w=160, down_h=90, diff_threshold=0, smooth_win_s=0.2
    )
    t_lidar, m_lidar = compute_lidar_motion(
        lidar_frames,
        feature=lidar_feature,
        smooth_win_s=0.3,
        hist_bins=lidar_hist_bins,
        r_max=lidar_r_max,
    )

    if len(t_rgb) < 10 or len(t_lidar) < 5:
        raise RuntimeError(f"Not enough motion samples. RGB={len(t_rgb)}, LiDAR={len(t_lidar)}")

    if len(t_rgb) != len(m_rgb):
        raise RuntimeError(f"RGB length mismatch: len(t_rgb)={len(t_rgb)} != len(m_rgb)={len(m_rgb)}")
    if len(t_lidar) != len(m_lidar):
        raise RuntimeError(f"LiDAR length mismatch: len(t_lidar)={len(t_lidar)} != len(m_lidar)={len(m_lidar)}")

    # 2) intersection time range
    t_start0 = max(t_rgb[0], t_lidar[0])
    t_end = min(t_rgb[-1], t_lidar[-1])

    # 3) drop warm-up seconds AFTER intersection start
    t_start = t_start0 + float(max(0.0, warmup_s))

    if t_end - t_start < 5.0:
        raise RuntimeError(
            f"Overlap too short after warmup: {(t_end - t_start):.2f}s "
            f"(raw_overlap={(t_end - t_start0):.2f}s, warmup_s={warmup_s:.2f}s)"
        )

    # 4) unified time grid
    t_grid = np.arange(t_start, t_end, grid_dt, dtype=np.float64)

    # 5) interpolate motion signals to grid
    rgb_g = interp_to_grid(t_rgb, zscore(m_rgb), t_grid)
    lidar_g = interp_to_grid(t_lidar, zscore(m_lidar), t_grid)

    # 6) estimate offset
    best_delta, best_corr = estimate_offset_by_scan(
        t_grid, rgb_g, lidar_g, search_s=search_s, step_s=step_s
    )

    run_name = rgb_dir.parents[2].name      # .../kinect/run_x/<cam>/frames_rgb_blured
    lidar_session = lidar_dir.name          # .../vlp16/run_x/<session>

    out_png = out_dir / "plots" / f"{run_name}__{lidar_session}__{lidar_feature}.png"

    result: Dict = {
        "run_name": run_name,
        "rgb_run": run_name,
        "lidar_run": run_name,
        "lidar_session": lidar_session,
        "rgb_dir": str(rgb_dir),
        "lidar_dir": str(lidar_dir),
        "plot_path": str(out_png),

        "n_rgb_frames": int(len(rgb_frames)),
        "n_lidar_frames": int(len(lidar_frames)),
        "n_rgb_motion": int(len(t_rgb)),
        "n_lidar_motion": int(len(t_lidar)),

        "lidar_feature": str(lidar_feature),
        "lidar_r_max": float(lidar_r_max),
        "lidar_hist_bins": int(lidar_hist_bins),
        "search_s": float(search_s),
        "step_s": float(step_s),
        "grid_dt": float(grid_dt),
        "warmup_s": float(warmup_s),

        "t_start_intersection": float(t_start0),
        "t_start_used": float(t_start),
        "t_end_used": float(t_end),
        "overlap_raw_s": float(t_end - t_start0),
        "overlap_used_s": float(t_end - t_start),

        "best_delta_s": float(best_delta),
        "best_corr": float(best_corr),

        "drift": None,
    }

    if do_drift:
        drift = estimate_drift_piecewise(
            t_grid, rgb_g, lidar_g,
            search_s=search_s, step_s=step_s,
            win_s=drift_win_s, hop_s=drift_hop_s,
        )
        result["drift"] = drift

    plot_alignment(
        out_png=out_png,
        t_grid=t_grid,
        rgb_g=rgb_g,
        lidar_g=lidar_g,
        best_delta=best_delta,
        title=f"{run_name} | session={lidar_session} | feature={lidar_feature} | corr={best_corr:.3f}",
    )

    return result


# -------------------------- incremental JSONL writer --------------------------
def append_jsonl(path: Path, obj: Dict):
    """
    Append one JSON object as a single line (JSONL).
    This is robust against interruptions (Ctrl+C / crashes).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


# -------------------------- subject processing --------------------------
def process_one_subject(subject_root: Path, args) -> None:
    """
    Process one subject root:
      - find runs under subject_root/kinect/run_*
      - match LiDAR under subject_root/vlp16/run_*/<session>
      - process each run and write outputs to subject-specific out_dir
    """
    kinect_root = subject_root / "kinect"
    vlp16_root = subject_root / "vlp16"

    if not kinect_root.exists():
        print(f"[SKIP subject] Missing: {kinect_root}")
        return
    if not vlp16_root.exists():
        print(f"[SKIP subject] Missing: {vlp16_root}")
        return

    # decide per-subject out_dir
    if args.out_root:
        subject_name = subject_root.name
        out_dir = Path(args.out_root) / subject_name / "_sync_out"
    else:
        out_dir = subject_root / "_sync_out"

    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "run_offsets.jsonl"
    out_json = out_dir / "run_offsets.json"

    only_run_set = set(args.only_run) if args.only_run else None

    def should_run(run_name: str) -> bool:
        if only_run_set is None:
            return True
        return run_name in only_run_set

    results: List[Dict] = []

    # run loop
    run_dirs = sorted([p for p in kinect_root.glob("run_*") if p.is_dir()])
    if not run_dirs:
        print(f"[SKIP subject] No run_* found under: {kinect_root}")
        return

    print(f"\n========== SUBJECT: {subject_root} ==========")
    print(f"Output -> {out_dir}")

    for run_dir in run_dirs:
        run_name = run_dir.name
        if not should_run(run_name):
            continue

        rgb_dir = run_dir / args.camera_id / "frames_rgb_blured"
        if not rgb_dir.exists():
            # 兼容：如果你某些 run 用 frames_rgb 而不是 frames_rgb_blured，可在这里加 fallback
            # rgb_dir2 = run_dir / args.camera_id / "frames_rgb"
            # if rgb_dir2.exists(): rgb_dir = rgb_dir2
            continue

        lidar_run_dir = vlp16_root / run_name
        if not lidar_run_dir.exists():
            continue

        sessions = [p for p in lidar_run_dir.iterdir() if p.is_dir()]
        if not sessions:
            continue
        sessions.sort(key=lambda p: p.name)
        lidar_dir = sessions[-1]

        print(f"[{run_name}] RGB={rgb_dir} | LiDAR={lidar_dir}")

        try:
            res = process_one_run(
                rgb_dir=rgb_dir,
                lidar_dir=lidar_dir,
                out_dir=out_dir,
                lidar_feature=args.lidar_feature,
                search_s=args.search_s,
                step_s=args.step_s,
                grid_dt=args.grid_dt,
                do_drift=args.do_drift,
                drift_win_s=args.drift_win_s,
                drift_hop_s=args.drift_hop_s,
                lidar_r_max=args.lidar_r_max,
                lidar_hist_bins=args.lidar_hist_bins,
                warmup_s=args.warmup_s,
            )

            results.append(res)
            append_jsonl(out_jsonl, res)

            print(f"  -> best_delta_s={res['best_delta_s']:+.3f}, corr={res['best_corr']:.3f}")
            print(f"  -> plot: {res['plot_path']}")

        except Exception as e:
            err = {
                "subject_root": str(subject_root),
                "run_name": run_name,
                "rgb_dir": str(rgb_dir),
                "lidar_dir": str(lidar_dir),
                "error": str(e),
            }
            append_jsonl(out_jsonl, err)
            print(f"  !! FAILED: {run_name}: {e}")
            continue

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[Subject done] {subject_root}")
    print(f"Saved summary JSON: {out_json}")
    print(f"Saved per-run JSONL: {out_jsonl}")
    print(f"Plots folder: {(out_dir / 'plots')}")
    if only_run_set:
        print(f"Only-runs filter used: {sorted(only_run_set)}")


# -------------------------- main --------------------------
def main():
    ap = argparse.ArgumentParser()

    # ✅ new: multiple subject roots
    ap.add_argument(
        "--subject_roots", type=str, nargs="+", required=True,
        help=r"One or more subject roots, e.g. D:\...\_raw_data_structured\N D:\...\_raw_data_structured\MR"
    )

    ap.add_argument(
        "--camera_id", type=str, default="001431512812",
        help="Camera folder under kinect/run_x/<camera_id>/frames_rgb_blured"
    )
    ap.add_argument(
        "--lidar_feature", type=str, default="histdiff",
        choices=["histdiff", "centroid", "count"],
        help="LiDAR motion feature"
    )
    ap.add_argument("--search_s", type=float, default=2.0, help="Search range for offset in seconds (+/-)")
    ap.add_argument("--step_s", type=float, default=0.01, help="Scan step for offset in seconds")
    ap.add_argument("--grid_dt", type=float, default=0.01, help="Unified time grid dt in seconds")

    ap.add_argument("--do_drift", action="store_true", help="Estimate drift using piecewise windows")
    ap.add_argument("--drift_win_s", type=float, default=20.0)
    ap.add_argument("--drift_hop_s", type=float, default=10.0)

    # ✅ new: global output root (optional)
    ap.add_argument(
        "--out_root", type=str, default=None,
        help="If set, write outputs to out_root/<subject_name>/_sync_out. Otherwise per-subject <subject_root>/_sync_out"
    )

    ap.add_argument("--lidar_r_max", type=float, default=3.0)
    ap.add_argument("--lidar_hist_bins", type=int, default=40)

    ap.add_argument(
        "--warmup_s", type=float, default=5.0,
        help="Drop warm-up seconds AFTER intersection start (e.g., 1~3). Default 5."
    )

    ap.add_argument(
        "--only_run", nargs="*", default=None,
        help="Only process specified run(s), e.g. --only_run run_10 or --only_run run_1 run_8-37"
    )

    args = ap.parse_args()

    # ensure out_root exists if provided
    if args.out_root:
        Path(args.out_root).mkdir(parents=True, exist_ok=True)

    # process each subject sequentially
    for s in args.subject_roots:
        subject_root = Path(s)
        if not subject_root.exists():
            print(f"[SKIP subject] Not found: {subject_root}")
            continue
        process_one_subject(subject_root, args)


if __name__ == "__main__":
    main()
