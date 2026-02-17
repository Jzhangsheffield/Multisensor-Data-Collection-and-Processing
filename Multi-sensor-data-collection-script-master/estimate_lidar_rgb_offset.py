#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimate time offset (and optional drift) between RGB (30 FPS) and LiDAR (10 FPS)
using motion-energy signals derived from:
- RGB: frame difference energy (midpoint timestamp between frames)
- LiDAR: distance-histogram difference (robust) and/or centroid speed and/or point count change
  (also midpoint timestamp between consecutive LiDAR frames)

✅ This FULL VERSION adds:
  1) Per-run parallel processing via ProcessPoolExecutor (--jobs)
  2) Safe JSONL writing ONLY in the main process (avoid file corruption)
  3) Optional timezone-aware parsing for filename timestamps (--ts_tz)
     - Default is "naive" behavior (keeps your old result)
     - If you need DST-safe behavior, set: --ts_tz Europe/London or --ts_tz UTC

✅ NEW FEATURE (manual visual alignment mode):
  4) Manual shift plotting mode (--manual_shift_s)
     - If provided, NO cross-correlation scan will be performed.
     - The script will plot:
         a) RGB motion energy (grid)
         b) original LiDAR motion energy (grid)
         c) shifted LiDAR motion energy (grid + manual Δt)
     - This lets you visually inspect alignment and decide the shift by hand.
     - When manual mode is enabled, drift estimation is automatically disabled
       to respect "no correlation/scan" requirement.

Output per subject:
- <out_dir>/plots/<run_name>__<lidar_session>__<lidar_feature>.png
- <out_dir>/run_offsets.json      : summary list (written at end)
- <out_dir>/run_offsets.jsonl     : one JSON per run (append as each run finishes)

Windows notes:
- Must run under "if __name__ == '__main__':" for multiprocessing.
- Worker must be top-level function (pickleable).
"""

import argparse
import json
import re
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2

# ---- Use non-GUI backend for safety (especially on Windows / servers) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional tz parsing support (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# ======================================================================
# 0) Timestamp parsing from filenames
# ======================================================================

TS_RE = re.compile(r"(\d{8}_\d{6}_\d{6})")  # substring: YYYYMMDD_HHMMSS_micro


def parse_ts_from_name(name: str, tz_name: Optional[str] = None) -> Optional[float]:
    """
    Parse 'YYYYMMDD_HHMMSS_micro' from filename and return "epoch seconds" as float.

    IMPORTANT:
      - If tz_name is None:
          We keep the original behavior: naive datetime, then (dt - epoch).total_seconds().
          This is "Windows-safe" and matches your old script exactly, BUT it treats the
          filename timestamp as if it were in an unspecified time base (often ends up
          equivalent to "UTC naive").
      - If tz_name is provided (e.g. 'Europe/London' or 'UTC'):
          We interpret the filename timestamp as local time in that timezone, and then
          use dt.timestamp() to get true epoch seconds (DST-aware).

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

    if tz_name:
        if ZoneInfo is None:
            raise RuntimeError("zoneinfo is not available, cannot use --ts_tz")
        # Interpret filename as "local time in tz_name", then convert to epoch.
        dt = dt.replace(tzinfo=ZoneInfo(tz_name))
        return float(dt.timestamp())

    # Original behavior (no tz)
    epoch = datetime(1970, 1, 1)
    return float((dt - epoch).total_seconds())


def list_files_sorted_by_ts(folder: Path, suffix: str, tz_name: Optional[str]) -> List[Tuple[float, Path]]:
    """
    List files in folder matching suffix, parse timestamps from filenames,
    drop invalid/bogus timestamps (e.g., 1970), return sorted by time.

    Filter: timestamps earlier than 2010-01-01 are treated as bogus.
    """
    items: List[Tuple[float, Path]] = []
    for p in folder.glob(f"*{suffix}"):
        t = parse_ts_from_name(p.name, tz_name=tz_name)
        if t is None or not np.isfinite(t):
            continue
        if t < 1262304000.0:  # 2010-01-01
            continue
        items.append((float(t), p))

    items.sort(key=lambda x: x[0])
    return items


# ======================================================================
# 1) Signal utilities
# ======================================================================

def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standard z-score normalization (mean/std)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    mu = float(x.mean())
    sd = float(x.std())
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

    pad_left = (win - 1) // 2
    pad_right = (win - 1) - pad_left

    xp = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    y = np.convolve(xp, kernel, mode="valid")  # length == len(x)

    if y.shape[0] != x.shape[0]:
        y = y[: x.shape[0]]
    return y


def interp_to_grid(t: np.ndarray, v: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Interpolate samples (t, v) onto a unified time grid (t_grid), clamping out-of-range."""
    if len(t) < 2:
        return np.zeros_like(t_grid, dtype=np.float64)
    tmin, tmax = float(t[0]), float(t[-1])
    return np.interp(np.clip(t_grid, tmin, tmax), t, v)


def corr_at_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation coefficient between same-length arrays a and b."""
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

    tmin, tmax = float(tg[0]), float(tg[-1])
    dt_grid = float(tg[1] - tg[0]) if len(tg) >= 2 else 0.01
    min_need = max(50, int(0.5 / max(dt_grid, 1e-6)))

    for d in deltas:
        # Overlap region when shifting LiDAR by d
        mask = (tg + d >= tmin) & (tg + d <= tmax)
        if int(mask.sum()) < min_need:
            continue

        a = rgb_g[mask]
        b = np.interp(tg[mask] + d, tg, lidar_g)
        c = corr_at_shift(a, b)

        if c > best_corr:
            best_corr = float(c)
            best_delta = float(d)

    return float(best_delta), float(best_corr)


# ======================================================================
# 2) RGB motion signal
# ======================================================================

def compute_rgb_motion(
    frames: List[Tuple[float, Path]],
    down_w: int = 160,
    down_h: int = 90,
    diff_threshold: int = 0,
    smooth_win_s: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RGB motion energy between consecutive frames:
      motion = mean(absdiff(gray(frame[i+1]), gray(frame[i])))
      timestamp = midpoint (t_i, t_{i+1})

    If diff_threshold > 0:
      motion = fraction of pixels whose absdiff > diff_threshold
    """
    if len(frames) < 2:
        return np.array([]), np.array([])

    t0, p0 = frames[0]
    im0 = cv2.imread(str(p0), cv2.IMREAD_GRAYSCALE)
    if im0 is None:
        raise RuntimeError(f"Failed to read image: {p0}")
    im0 = cv2.resize(im0, (down_w, down_h), interpolation=cv2.INTER_AREA)

    t_mid, m = [], []
    for (t1, p1) in frames[1:]:
        im1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
        if im1 is None:
            continue
        im1 = cv2.resize(im1, (down_w, down_h), interpolation=cv2.INTER_AREA)

        d = cv2.absdiff(im1, im0).astype(np.float32)
        val = float((d > diff_threshold).mean()) if diff_threshold > 0 else float(d.mean())

        t_mid.append(0.5 * (float(t0) + float(t1)))
        m.append(val)

        t0, im0 = t1, im1

    t_mid = np.asarray(t_mid, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)

    if len(t_mid) >= 3 and smooth_win_s > 0:
        dt = float(np.median(np.diff(t_mid)))
        win = max(1, int(round(smooth_win_s / max(dt, 1e-6))))
        m = moving_average(m, win)

    return t_mid, m


# ======================================================================
# 3) LiDAR motion signal
# ======================================================================

def load_lidar_points_csv(csv_path: Path) -> np.ndarray:
    """
    Load LiDAR CSV with columns x,y,z,intensity.
    Filters out:
      - NaN rows
      - all-zero rows

    NOTE: np.genfromtxt is robust but slow; keep as-is for correctness.
    If you want faster, replace with np.loadtxt or pandas.read_csv(usecols=[0,1,2,3]).
    """
    try:
        arr = np.genfromtxt(str(csv_path), delimiter=",", dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {csv_path}: {e}")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # If header caused issues, try skip_header=1
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

    feats, ts = [], []
    for t, p in frames:
        pts = load_lidar_points_csv(p)
        feats.append(frame_feat(pts))
        ts.append(float(t))

    t_mid, m = [], []
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
        dt = float(np.median(np.diff(t_mid)))
        win = max(1, int(round(smooth_win_s / max(dt, 1e-6))))
        m = moving_average(m, win)

    return t_mid, m


# ======================================================================
# 4) Drift estimation (optional)
# ======================================================================

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

    NOTE: This routine relies on estimate_offset_by_scan(), so it is DISABLED when
    manual shift mode is enabled (--manual_shift_s), to satisfy "no correlation scan".
    """
    t0, t1 = float(tg[0]), float(tg[-1])
    centers, offsets, corrs = [], [], []

    t = t0
    while t + win_s <= t1 + 1e-9:
        mask = (tg >= t) & (tg <= t + win_s)
        if int(mask.sum()) < 200:
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


# ======================================================================
# 5) Plotting
# ======================================================================

def plot_alignment(
    out_png: Path,
    t_grid: np.ndarray,
    rgb_g: np.ndarray,
    lidar_g: np.ndarray,
    best_delta: float,
    title: str,
):
    """
    Plot z-scored RGB and LiDAR motion signals on the unified grid, plus shifted LiDAR.

    - Always plots:
        1) RGB motion (grid)
        2) LiDAR motion (grid, original)
        3) LiDAR motion shifted by Δt (either auto-estimated or manual)
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


# ======================================================================
# 6) Per-run processing (core)
# ======================================================================

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
    tail_trim_s: float,
    ts_tz: Optional[str],
    manual_shift_s: Optional[float],  # ✅ NEW
) -> Dict:
    """
    Process one run:
      1) list RGB frames and LiDAR frames (sorted by timestamp, filtered)
      2) compute motion signals (RGB and LiDAR) with midpoint timestamps
      3) compute intersection time range
      4) drop warm-up seconds AFTER intersection start + trim tail seconds
      5) interpolate both signals to unified time grid
      6) offset:
          - auto mode: scan offset that maximizes correlation
          - manual mode (--manual_shift_s): use given Δt, NO scanning
      7) optional drift estimation (auto mode only)
      8) save plot
      9) return a dict with results and metadata
    """
    rgb_frames = list_files_sorted_by_ts(rgb_dir, ".jpg", tz_name=ts_tz)
    if not rgb_frames:
        raise RuntimeError(f"No RGB frames found in: {rgb_dir}")

    lidar_frames = list_files_sorted_by_ts(lidar_dir, ".csv", tz_name=ts_tz)
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

    # 2) intersection time range
    t_start0 = float(max(t_rgb[0], t_lidar[0]))
    t_end0 = float(min(t_rgb[-1], t_lidar[-1]))

    # 3) warmup + tail trim (both applied INSIDE intersection)
    t_start = t_start0 + float(max(0.0, warmup_s))
    t_end = t_end0 - float(max(0.0, tail_trim_s))

    if t_end <= t_start:
        raise RuntimeError(
            f"Invalid trimmed window: t_end<=t_start. "
            f"intersection=[{t_start0:.3f},{t_end0:.3f}] warmup={warmup_s} tail_trim={tail_trim_s}"
        )

    if (t_end - t_start) < 5.0:
        raise RuntimeError(
            f"Overlap too short after warmup/tail trim: {(t_end - t_start):.2f}s "
            f"(warmup_s={warmup_s:.2f}s, tail_trim_s={tail_trim_s:.2f}s)"
        )

    # 4) unified time grid
    t_grid = np.arange(t_start, t_end, float(grid_dt), dtype=np.float64)
    if len(t_grid) < 100:
        raise RuntimeError(f"Too few grid points: {len(t_grid)} (grid_dt={grid_dt})")

    # 5) interpolate motion signals to grid
    rgb_g = interp_to_grid(t_rgb, zscore(m_rgb), t_grid)
    lidar_g = interp_to_grid(t_lidar, zscore(m_lidar), t_grid)

    # 6) estimate offset OR use manual shift
    manual_mode = (manual_shift_s is not None)
    if manual_mode:
        best_delta = float(manual_shift_s)
        best_corr = None  # explicitly indicate "not computed"
        # Respect "no cross-correlation scan" requirement:
        do_drift_effective = False
        drift = None
    else:
        best_delta, best_corr = estimate_offset_by_scan(
            t_grid, rgb_g, lidar_g, search_s=float(search_s), step_s=float(step_s)
        )
        do_drift_effective = bool(do_drift)
        drift = None

    run_name = rgb_dir.parents[1].name      # .../kinect/run_x/<cam>/frames_rgb_blured
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
        "tail_trim_s": float(tail_trim_s),

        "ts_tz": None if not ts_tz else str(ts_tz),

        "t_start_intersection": float(t_start0),
        "t_end_intersection": float(t_end0),
        "t_start_used": float(t_start),
        "t_end_used": float(t_end),
        "overlap_raw_s": float(t_end0 - t_start0),
        "overlap_used_s": float(t_end - t_start),

        # ✅ NEW: record mode
        "mode": "manual" if manual_mode else "auto",
        "manual_shift_s": float(best_delta) if manual_mode else None,

        "best_delta_s": float(best_delta),
        "best_corr": None if best_corr is None else float(best_corr),

        "drift": None,
    }

    # 7) optional drift estimation (auto mode only)
    if do_drift_effective:
        drift = estimate_drift_piecewise(
            t_grid, rgb_g, lidar_g,
            search_s=float(search_s), step_s=float(step_s),
            win_s=float(drift_win_s), hop_s=float(drift_hop_s),
        )
        result["drift"] = drift

    # 8) plot
    if manual_mode:
        title = (
            f"{run_name} | session={lidar_session} | feature={lidar_feature} | "
            f"MANUAL Δt={best_delta:+.3f}s"
        )
    else:
        title = (
            f"{run_name} | session={lidar_session} | feature={lidar_feature} | "
            f"corr={float(best_corr):.3f}"
        )

    plot_alignment(
        out_png=out_png,
        t_grid=t_grid,
        rgb_g=rgb_g,
        lidar_g=lidar_g,
        best_delta=best_delta,
        title=title,
    )

    return result


# ======================================================================
# 7) Robust JSONL writer (main-process only!)
# ======================================================================

def append_jsonl(path: Path, obj: Dict):
    """
    Append one JSON object as a single line (JSONL).

    IMPORTANT:
      - Only call this from the MAIN process.
      - Do NOT let worker processes write to the same JSONL file concurrently.
        (That can corrupt the output.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


# ======================================================================
# 8) Worker function (must be top-level for Windows multiprocessing)
# ======================================================================

def worker_process_one_run(job: Dict) -> Dict:
    """
    One run worker.
    This must be top-level (not nested) so Windows can pickle it.

    The worker:
      - limits OpenCV internal threads to avoid CPU oversubscription
      - calls process_one_run(...) and returns a result dict
      - returns a dict with "error" key if something fails

    job includes all needed paths and parameters in plain JSON-serializable types.
    """
    # Prevent OpenCV from using multiple threads inside each process.
    # (Without this, 4 processes x 8 threads can destroy performance.)
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        res = process_one_run(
            rgb_dir=Path(job["rgb_dir"]),
            lidar_dir=Path(job["lidar_dir"]),
            out_dir=Path(job["out_dir"]),
            lidar_feature=str(job["lidar_feature"]),
            search_s=float(job["search_s"]),
            step_s=float(job["step_s"]),
            grid_dt=float(job["grid_dt"]),
            do_drift=bool(job["do_drift"]),
            drift_win_s=float(job["drift_win_s"]),
            drift_hop_s=float(job["drift_hop_s"]),
            lidar_r_max=float(job["lidar_r_max"]),
            lidar_hist_bins=int(job["lidar_hist_bins"]),
            warmup_s=float(job["warmup_s"]),
            tail_trim_s=float(job["tail_trim_s"]),
            ts_tz=job.get("ts_tz", None),
            manual_shift_s=job.get("manual_shift_s", None),  # ✅ NEW
        )
        res["subject_root"] = str(job["subject_root"])
        return res

    except Exception as e:
        return {
            "subject_root": str(job["subject_root"]),
            "run_name": str(job["run_name"]),
            "rgb_dir": str(job["rgb_dir"]),
            "lidar_dir": str(job["lidar_dir"]),
            "error": str(e),
        }


# ======================================================================
# 9) Per-subject driver (collect jobs, then run sequential/parallel)
# ======================================================================

def process_one_subject(subject_root: Path, args) -> None:
    """
    Process one subject root:
      - find runs under subject_root/kinect/run_*
      - match LiDAR under subject_root/vlp16/run_*/<session>
      - build a job list
      - run jobs:
          - sequential if --jobs 1
          - parallel if --jobs > 1
      - write JSONL incrementally as each run finishes (main process only)
      - write a final JSON summary at the end
    """
    kinect_root = subject_root / "kinect"
    vlp16_root = subject_root / "vlp16"

    if not kinect_root.exists():
        print(f"[SKIP subject] Missing: {kinect_root}")
        return
    if not vlp16_root.exists():
        print(f"[SKIP subject] Missing: {vlp16_root}")
        return

    # Decide output directory per subject
    if args.out_root:
        out_dir = Path(args.out_root) / subject_root.name / args.out_folder
    else:
        out_dir = subject_root / args.out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "run_offsets.jsonl"
    out_json = out_dir / "run_offsets.json"

    # Optional run filter
    only_run_set = set(args.only_run) if args.only_run else None

    def should_run(run_name: str) -> bool:
        return True if only_run_set is None else (run_name in only_run_set)

    run_dirs = sorted([p for p in kinect_root.glob("run_*") if p.is_dir()])
    if not run_dirs:
        print(f"[SKIP subject] No run_* found under: {kinect_root}")
        return

    print(f"\n========== SUBJECT: {subject_root} ==========")
    print(f"Output -> {out_dir}")

    # If manual mode: remind user drift will be disabled.
    if args.manual_shift_s is not None and args.do_drift:
        print("[NOTE] --manual_shift_s is set, so drift estimation will be disabled (no correlation scans).")

    # ----------------------------------------------------------
    # Build job list
    # ----------------------------------------------------------
    jobs = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        if not should_run(run_name):
            continue

        # RGB folder
        rgb_dir = run_dir / args.camera_id / "frames_rgb_blured"
        if not rgb_dir.exists():
            # Optional fallback:
            # rgb_dir2 = run_dir / args.camera_id / "frames_rgb"
            # if rgb_dir2.exists(): rgb_dir = rgb_dir2
            continue

        # LiDAR folder: subject_root/vlp16/run_x/<session>
        lidar_run_dir = vlp16_root / run_name
        if not lidar_run_dir.exists():
            continue

        sessions = [p for p in lidar_run_dir.iterdir() if p.is_dir()]
        if not sessions:
            continue
        sessions.sort(key=lambda p: p.name)
        lidar_dir = sessions[-1]  # choose the latest by name sort

        jobs.append({
            "subject_root": str(subject_root),
            "run_name": run_name,
            "rgb_dir": str(rgb_dir),
            "lidar_dir": str(lidar_dir),
            "out_dir": str(out_dir),

            "lidar_feature": args.lidar_feature,
            "search_s": args.search_s,
            "step_s": args.step_s,
            "grid_dt": args.grid_dt,

            # drift is allowed only in auto-mode; in manual-mode we will disable inside process_one_run
            "do_drift": args.do_drift,
            "drift_win_s": args.drift_win_s,
            "drift_hop_s": args.drift_hop_s,

            "lidar_r_max": args.lidar_r_max,
            "lidar_hist_bins": args.lidar_hist_bins,

            "warmup_s": args.warmup_s,
            "tail_trim_s": args.tail_trim_s,

            "ts_tz": args.ts_tz if args.ts_tz else None,

            # ✅ NEW: manual shift per run (same value for all runs; you can extend later if needed)
            "manual_shift_s": args.manual_shift_s if args.manual_shift_s is not None else None,
        })

    if not jobs:
        print(f"[SKIP subject] No runnable runs for: {subject_root}")
        return

    # ----------------------------------------------------------
    # Execute jobs
    # ----------------------------------------------------------
    results: List[Dict] = []
    n_workers = max(1, int(args.jobs))

    if n_workers == 1:
        # --------- Sequential (original behavior) ---------
        for job in jobs:
            print(f"[{job['run_name']}] RGB={job['rgb_dir']} | LiDAR={job['lidar_dir']}")
            res = worker_process_one_run(job)

            # Main process writes JSONL immediately
            append_jsonl(out_jsonl, res)
            results.append(res)

            if "error" in res:
                print(f"  !! FAILED: {job['run_name']}: {res['error']}")
            else:
                if res.get("mode") == "manual":
                    print(f"  -> MANUAL shift Δt={res['best_delta_s']:+.3f}s")
                else:
                    print(f"  -> best_delta_s={res['best_delta_s']:+.3f}, corr={res['best_corr']:.3f}")
                print(f"  -> plot: {res['plot_path']}")

    else:
        # --------- Parallel (per-run) ---------
        # NOTE: If your data is on HDD, too many workers can be slower.
        print(f"[Parallel] workers={n_workers}, total_runs={len(jobs)}")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_to_job = {ex.submit(worker_process_one_run, job): job for job in jobs}

            # as_completed yields futures as soon as they finish
            for fut in as_completed(fut_to_job):
                job = fut_to_job[fut]
                res = fut.result()

                # Main process writes JSONL immediately (safe)
                append_jsonl(out_jsonl, res)
                results.append(res)

                if "error" in res:
                    print(f"  !! FAILED: {job['run_name']}: {res['error']}")
                else:
                    if res.get("mode") == "manual":
                        print(f"  -> DONE {job['run_name']}: MANUAL shift Δt={res['best_delta_s']:+.3f}s")
                    else:
                        print(f"  -> DONE {job['run_name']}: best_delta_s={res['best_delta_s']:+.3f}, corr={res['best_corr']:.3f}")

    # ----------------------------------------------------------
    # Write final summary JSON (list of results)
    # ----------------------------------------------------------
    # Sorting is optional. You can keep completion order, or sort by run_name:
    results_sorted = sorted(results, key=lambda d: d.get("run_name", ""))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    print(f"\n[Subject done] {subject_root}")
    print(f"Saved summary JSON: {out_json}")
    print(f"Saved per-run JSONL: {out_jsonl}")
    print(f"Plots folder: {(out_dir / 'plots')}")
    if only_run_set:
        print(f"Only-runs filter used: {sorted(only_run_set)}")


# ======================================================================
# 10) main
# ======================================================================

def main():
    ap = argparse.ArgumentParser()

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

    ap.add_argument(
        "--out_root", type=str, default=None,
        help="If set, write outputs to out_root/<subject_name>/<out_folder>. Otherwise per-subject <subject_root>/<out_folder>"
    )

    ap.add_argument(
        "--out_folder", type=str, default="_sync_out",
        help="Output folder name inside each subject output directory"
    )

    ap.add_argument("--lidar_r_max", type=float, default=3.0)
    ap.add_argument("--lidar_hist_bins", type=int, default=40)

    ap.add_argument(
        "--warmup_s", type=float, default=6.0,
        help="Drop warm-up seconds AFTER intersection start"
    )

    ap.add_argument(
        "--tail_trim_s", type=float, default=0.0,
        help="Trim last seconds from the intersection end"
    )

    ap.add_argument(
        "--only_run", nargs="*", default=None,
        help="Only process specified run(s), e.g. --only_run run_10 run_1 run_8-37"
    )

    # ✅ per-run parallelism
    ap.add_argument(
        "--jobs", type=int, default=3,
        help="Number of parallel processes PER SUBJECT. 1 = no parallel. "
             "If data is on HDD, try small values (2~4). If on SSD/NVMe, higher can help."
    )

    # ✅ timezone-aware parsing option
    ap.add_argument(
        "--ts_tz", type=str, default=None,
        help="Timezone used to interpret filename timestamps (YYYYMMDD_HHMMSS_micro). "
             "Default None keeps old naive behavior. Examples: Europe/London, UTC"
    )

    # ✅ NEW: manual shift mode (no correlation scan)
    ap.add_argument(
        "--manual_shift_s", type=float, default=None,
        help="Manual time shift Δt in seconds. If set, the script will NOT perform "
             "cross-correlation scan. It will directly plot RGB, original LiDAR, and "
             "LiDAR shifted by this Δt for visual inspection. "
             "Convention is the same as auto mode: delta > 0 means compare rgb(t) with lidar(t + delta)."
    )

    args = ap.parse_args()

    # Ensure global out_root exists if provided
    if args.out_root:
        Path(args.out_root).mkdir(parents=True, exist_ok=True)

    # Process each subject sequentially (inside each subject we may parallelize runs)
    for s in args.subject_roots:
        subject_root = Path(s)
        if not subject_root.exists():
            print(f"[SKIP subject] Not found: {subject_root}")
            continue
        process_one_subject(subject_root, args)


if __name__ == "__main__":
    # On Windows, you MUST protect multiprocessing entry point like this.
    # Also: consider setting the start method explicitly if you run into issues.
    # import multiprocessing
    # multiprocessing.set_start_method("spawn", force=True)

    main()
