#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
estimate_rgb_lidar_piecewise_offsets.py

Run-level piecewise time-offset estimation between RGB (30 FPS) and LiDAR (10 FPS),
using motion-energy signals on a unified time grid.

✅ Features (current version)
------------------------------------------------------------
A) Sliding-window local offset estimation:
   - For each window (length win_s, hop hop_s), scan delta in [-search_s, +search_s]
     to maximize correlation between RGB motion (grid) and LiDAR motion (grid shifted).

B) Changepoint detection (up to K changepoints):
   - CLI: --max_cps K
   - Dynamic programming (DP) fits a piecewise-constant offset model on offsets(t_center)
     (only using windows whose corr >= min_corr).
   - Returns segment medians [Δ1..ΔM] and changepoint times [t1*..tK*].

C) Plotting:
   - offset_timeseries plot:
       * Good windows (corr >= min_corr): connected line + markers
       * Bad windows (corr < min_corr): gray scatter
       * Every point annotated with its corr value (text)
       * Draws horizontal lines Δi, vertical lines t*i
   - alignment_piecewise plot:
       * RGB motion (z)
       * LiDAR motion (z)
       * LiDAR motion with piecewise shift (z)
       * legend shows corr per piece + overall

D) Record each window's RGB timestamp range (for future "window-based offset control"):
   For each sliding window we record:
     - window absolute time range: win_start_abs, win_end_abs (epoch seconds)
     - RGB motion coverage inside window:
         rgb_in_win_count
         rgb_win_start_abs, rgb_win_end_abs (epoch seconds of RGB motion midpoints in that window)
     - Nearest RGB frame timestamp strings from filenames:
         rgb_win_start_ts, rgb_win_end_ts

✅ NEW (requested in this message) - plotting only
------------------------------------------------------------
1) Always plot gray points for corr < min_corr (do NOT rely on decision.keep_mask).
   Gray points also show their corr text.
2) At each point, annotate BOTH corr and estimated offset Δ.
3) In legend, show range of estimated offsets among VALID points (corr >= min_corr):
      Δrange = max(Δ_good) - min(Δ_good), and also min/max.

Delta convention:
------------------------------------------------------------
delta > 0 means compare rgb(t) with lidar(t + delta)
(i.e., LiDAR is shifted later in the comparison)

Windows note:
------------------------------------------------------------
Multiprocessing MUST be guarded by:
    if __name__ == "__main__":
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional tz parsing support (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


# =========================================================
# 0) Timestamp parsing from filenames
# =========================================================

TS_RE = re.compile(r"(\d{8}_\d{6}_\d{6})")  # YYYYMMDD_HHMMSS_micro


def parse_ts_from_name(name: str, tz_name: Optional[str] = None) -> Optional[float]:
    """
    Parse 'YYYYMMDD_HHMMSS_micro' from filename and return epoch seconds.

    - tz_name is None: naive behavior (keeps old logic)
    - tz_name provided: interpret filename time as local time in tz_name (DST-safe)

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
            raise RuntimeError("zoneinfo not available, cannot use --ts_tz")
        dt = dt.replace(tzinfo=ZoneInfo(tz_name))
        return float(dt.timestamp())

    epoch = datetime(1970, 1, 1)
    return float((dt - epoch).total_seconds())


def list_files_sorted_by_ts(folder: Path, suffix: str, tz_name: Optional[str]) -> List[Tuple[float, Path]]:
    """
    List files matching suffix, parse timestamps, filter bogus timestamps (<2010-01-01),
    return sorted by time.
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


# =========================================================
# 1) Signal utilities
# =========================================================

def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standard z-score normalization."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    mu = float(x.mean())
    sd = float(x.std())
    if sd < eps:
        return x * 0.0
    return (x - mu) / (sd + eps)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Length-preserving moving average."""
    x = np.asarray(x, dtype=np.float64)
    if win <= 1 or x.size == 0:
        return x
    win = int(win)
    pad_left = (win - 1) // 2
    pad_right = (win - 1) - pad_left
    xp = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    y = np.convolve(xp, kernel, mode="valid")
    if y.shape[0] != x.shape[0]:
        y = y[: x.shape[0]]
    return y


def interp_to_grid(t: np.ndarray, v: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Interpolate (t, v) to t_grid, clamping at ends."""
    if len(t) < 2:
        return np.zeros_like(t_grid, dtype=np.float64)
    tmin, tmax = float(t[0]), float(t[-1])
    return np.interp(np.clip(t_grid, tmin, tmax), t, v)


def corr_at_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation coefficient between same-length arrays."""
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
    Scan delta in [-search_s, +search_s], find best correlation.

    Convention:
      delta > 0 means compare rgb(t) with lidar(t + delta)

    Returns:
      (best_delta, best_corr)
    """
    deltas = np.arange(-search_s, search_s + 1e-9, step_s, dtype=np.float64)

    best_delta = 0.0
    best_corr = -1e9

    tmin, tmax = float(tg[0]), float(tg[-1])
    dt_grid = float(tg[1] - tg[0]) if len(tg) >= 2 else 0.01
    min_need = max(80, int(0.8 / max(dt_grid, 1e-6)))  # require enough overlap

    for d in deltas:
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


def corr_for_delta_on_mask(
    tg: np.ndarray,
    rgb_g: np.ndarray,
    lidar_g: np.ndarray,
    delta: float,
    mask: np.ndarray,
) -> float:
    """Compute corr between rgb_g and lidar_g shifted by delta on mask."""
    if mask is None or int(mask.sum()) < 10:
        return -1.0
    a = rgb_g[mask]
    b = np.interp(tg[mask] + float(delta), tg, lidar_g)
    return corr_at_shift(a, b)


def nearest_frame_ts_str(
    t_query: float,
    rgb_frame_times: np.ndarray,
    rgb_frame_ts_strs: List[str],
) -> Optional[str]:
    """
    Map an epoch time to the nearest RGB frame's timestamp string (from filename).
    """
    if rgb_frame_times is None or len(rgb_frame_times) == 0:
        return None
    t_query = float(t_query)
    idx = int(np.searchsorted(rgb_frame_times, t_query))
    if idx <= 0:
        return rgb_frame_ts_strs[0]
    if idx >= len(rgb_frame_times):
        return rgb_frame_ts_strs[-1]
    if abs(rgb_frame_times[idx] - t_query) < abs(rgb_frame_times[idx - 1] - t_query):
        return rgb_frame_ts_strs[idx]
    return rgb_frame_ts_strs[idx - 1]


# =========================================================
# 2) RGB motion signal
# =========================================================

def compute_rgb_motion(
    frames: List[Tuple[float, Path]],
    down_w: int = 160,
    down_h: int = 90,
    diff_threshold: int = 0,
    smooth_win_s: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RGB motion energy between consecutive frames:
      motion = mean(absdiff(gray[i+1], gray[i]))
      timestamp = midpoint(t_i, t_{i+1})

    If diff_threshold>0:
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


# =========================================================
# 3) LiDAR motion signal
# =========================================================

def load_lidar_points_csv(csv_path: Path) -> np.ndarray:
    """
    Load LiDAR CSV (x,y,z,intensity).
    Filters out NaN rows + all-zero rows.
    """
    try:
        arr = np.genfromtxt(str(csv_path), delimiter=",", dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {csv_path}: {e}")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # If header exists, try skipping it
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
    LiDAR motion from consecutive frames:
      - histdiff: L1 diff between distance histograms (robust)
      - centroid: centroid speed
      - count   : abs diff of point counts
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


# =========================================================
# 4) Sliding-window offsets (+ record window RGB timestamp ranges)
# =========================================================

def sliding_window_offsets(
    t_grid: np.ndarray,
    rgb_g: np.ndarray,
    lidar_g: np.ndarray,
    win_s: float,
    hop_s: float,
    search_s: float,
    step_s: float,
    min_points: int = 400,
    # provide RGB time info so each window can record "RGB timestamps"
    t_rgb_mid: Optional[np.ndarray] = None,          # RGB motion midpoint times (epoch seconds)
    rgb_frame_times: Optional[np.ndarray] = None,    # RGB frame times from filenames (epoch seconds)
    rgb_frame_ts_strs: Optional[List[str]] = None,   # RGB frame timestamp strings from filenames
) -> Dict[str, Any]:
    """
    Compute offset per window center by scanning delta that maximizes correlation.

    Returns dict with centers/offsets/corrs and per-window time records.
    """
    t0 = float(t_grid[0])
    t1 = float(t_grid[-1])

    centers_abs: List[float] = []
    centers_s: List[float] = []
    offsets: List[float] = []
    corrs: List[float] = []

    win_start_abs_list: List[float] = []
    win_end_abs_list: List[float] = []

    rgb_in_win_count_list: List[int] = []
    rgb_win_start_abs_list: List[float] = []
    rgb_win_end_abs_list: List[float] = []

    rgb_win_start_ts_list: List[Optional[str]] = []
    rgb_win_end_ts_list: List[Optional[str]] = []

    if t_rgb_mid is not None:
        t_rgb_mid = np.asarray(t_rgb_mid, dtype=np.float64)
    if rgb_frame_times is not None:
        rgb_frame_times = np.asarray(rgb_frame_times, dtype=np.float64)

    cur = t0
    while cur + win_s <= t1 + 1e-9:
        mask = (t_grid >= cur) & (t_grid <= cur + win_s)
        n = int(mask.sum())
        if n >= min_points:
            tg = t_grid[mask]
            a = rgb_g[mask]
            b = lidar_g[mask]

            d, c = estimate_offset_by_scan(tg, a, b, search_s=search_s, step_s=step_s)

            center = float(cur + 0.5 * win_s)
            centers_abs.append(center)
            centers_s.append(center - t0)
            offsets.append(float(d))
            corrs.append(float(c))

            win_start_abs = float(cur)
            win_end_abs = float(cur + win_s)
            win_start_abs_list.append(win_start_abs)
            win_end_abs_list.append(win_end_abs)

            # RGB motion coverage inside the window
            if t_rgb_mid is not None and len(t_rgb_mid) > 0:
                m_rgbw = (t_rgb_mid >= win_start_abs) & (t_rgb_mid <= win_end_abs)
                rgb_in_win_count = int(m_rgbw.sum())
                rgb_in_win_count_list.append(rgb_in_win_count)

                if rgb_in_win_count > 0:
                    rgb_win_start_abs = float(t_rgb_mid[m_rgbw][0])
                    rgb_win_end_abs = float(t_rgb_mid[m_rgbw][-1])
                else:
                    rgb_win_start_abs = float("nan")
                    rgb_win_end_abs = float("nan")

                rgb_win_start_abs_list.append(rgb_win_start_abs)
                rgb_win_end_abs_list.append(rgb_win_end_abs)
            else:
                rgb_in_win_count_list.append(0)
                rgb_win_start_abs_list.append(float("nan"))
                rgb_win_end_abs_list.append(float("nan"))

            # nearest RGB frame timestamp strings to window boundaries
            if (
                rgb_frame_times is not None
                and rgb_frame_ts_strs is not None
                and len(rgb_frame_ts_strs) == len(rgb_frame_times)
                and len(rgb_frame_times) > 0
            ):
                rgb_win_start_ts_list.append(nearest_frame_ts_str(win_start_abs, rgb_frame_times, rgb_frame_ts_strs))
                rgb_win_end_ts_list.append(nearest_frame_ts_str(win_end_abs, rgb_frame_times, rgb_frame_ts_strs))
            else:
                rgb_win_start_ts_list.append(None)
                rgb_win_end_ts_list.append(None)

        cur += hop_s

    return {
        "centers_s": centers_s,
        "centers_abs": centers_abs,
        "offsets": offsets,
        "corrs": corrs,
        "win_s": float(win_s),
        "hop_s": float(hop_s),

        "win_start_abs": win_start_abs_list,
        "win_end_abs": win_end_abs_list,

        "rgb_in_win_count": rgb_in_win_count_list,
        "rgb_win_start_abs": rgb_win_start_abs_list,
        "rgb_win_end_abs": rgb_win_end_abs_list,

        "rgb_win_start_ts": rgb_win_start_ts_list,
        "rgb_win_end_ts": rgb_win_end_ts_list,
    }


# =========================================================
# 5) Multi-changepoint detection (up to K changepoints)
# =========================================================

def _prefix_sums_for_sse(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return prefix sums of y and y^2 for O(1) segment SSE queries."""
    y = np.asarray(y, dtype=np.float64)
    csum = np.cumsum(y)
    csum2 = np.cumsum(y * y)
    return csum, csum2


def _seg_sse(csum: np.ndarray, csum2: np.ndarray, i0: int, i1: int) -> float:
    """SSE for segment y[i0:i1) fitted by a constant mean, computed in O(1)."""
    n = i1 - i0
    if n <= 0:
        return 0.0
    s = float(csum[i1 - 1] - (csum[i0 - 1] if i0 > 0 else 0.0))
    s2 = float(csum2[i1 - 1] - (csum2[i0 - 1] if i0 > 0 else 0.0))
    return float(s2 - (s * s) / max(n, 1))


def best_piecewise_constant_up_to_k_cps(
    y: np.ndarray,
    max_cps: int,
    min_seg: int,
    cp_penalty: float,
) -> Dict[str, Any]:
    """
    DP fit piecewise-constant model to y with up to max_cps changepoints.

    total_cost = sum(SSE(segment)) + cp_penalty * (num_segments - 1)
    """
    y = np.asarray(y, dtype=np.float64)
    n = int(y.size)

    if n == 0:
        return {"best_m": 1, "best_cps": [], "best_sse": 0.0, "best_cost": 0.0, "all_models": []}

    max_segments_by_data = max(1, n // max(1, min_seg))
    max_segments = min(max_cps + 1, max_segments_by_data)

    csum, csum2 = _prefix_sums_for_sse(y)

    INF = 1e300
    dp = np.full((max_segments + 1, n + 1), INF, dtype=np.float64)
    back = np.full((max_segments + 1, n + 1), -1, dtype=np.int32)

    # base: 1 segment
    for j in range(min_seg, n + 1):
        dp[1, j] = _seg_sse(csum, csum2, 0, j)
        back[1, j] = 0

    # transitions
    for m in range(2, max_segments + 1):
        j_min = m * min_seg
        for j in range(j_min, n + 1):
            i_min = (m - 1) * min_seg
            i_max = j - min_seg

            best_val = INF
            best_i = -1
            for i in range(i_min, i_max + 1):
                prev = dp[m - 1, i]
                if prev >= INF / 2:
                    continue
                sse_last = _seg_sse(csum, csum2, i, j)
                val = prev + sse_last + float(cp_penalty)  # penalty per added CP
                if val < best_val:
                    best_val = float(val)
                    best_i = int(i)

            dp[m, j] = best_val
            back[m, j] = best_i

    # choose best m
    all_models = []
    best_cost = INF
    best_m = 1
    for m in range(1, max_segments + 1):
        cost = float(dp[m, n])
        if cost < best_cost:
            best_cost = cost
            best_m = m
        sse_only = float(cost - float(cp_penalty) * max(0, m - 1))
        all_models.append({"m": int(m), "cost": cost, "sse": sse_only})

    # reconstruct CP indices in y
    cps = []
    j = n
    m = best_m
    while m > 1:
        i = int(back[m, j])
        cps.append(i)
        j = i
        m -= 1
    cps = sorted(cps)

    best_sse = float(best_cost - float(cp_penalty) * max(0, best_m - 1))
    return {
        "best_m": int(best_m),
        "best_cps": cps,
        "best_sse": best_sse,
        "best_cost": float(best_cost),
        "all_models": all_models,
    }


def decide_piecewise_offsets_multi(
    centers_s_all: np.ndarray,
    offsets_all: np.ndarray,
    corrs_all: np.ndarray,
    min_corr: float,
    max_cps: int,
    min_seg_windows: int,
    cp_penalty: float,
    min_improve_ratio: float,
) -> Dict[str, Any]:
    """Decide whether to use single or piecewise offsets (uses only corr>=min_corr windows)."""
    centers_s_all = np.asarray(centers_s_all, dtype=np.float64)
    offsets_all = np.asarray(offsets_all, dtype=np.float64)
    corrs_all = np.asarray(corrs_all, dtype=np.float64)

    keep = np.isfinite(offsets_all) & np.isfinite(corrs_all) & (corrs_all >= float(min_corr))

    if int(keep.sum()) < max(2 * min_seg_windows + 1, min_seg_windows):
        finite = np.isfinite(offsets_all)
        d = float(np.median(offsets_all[finite])) if finite.any() else 0.0
        return {
            "mode": "single",
            "deltas": [d],
            "t_stars_s": [],
            "k_stars": [],
            "keep_mask": keep.astype(bool).tolist(),
            "used_windows": int(keep.sum()),
            "total_windows": int(len(offsets_all)),
            "min_corr": float(min_corr),
            "note": "Not enough windows after min_corr filtering; using median of finite offsets.",
        }

    c = centers_s_all[keep]
    y = offsets_all[keep]

    mu0 = float(np.mean(y))
    sse0 = float(np.sum((y - mu0) ** 2))
    eps = 1e-12

    if int(max_cps) <= 0:
        d_single = float(np.median(y))
        return {
            "mode": "single",
            "deltas": [d_single],
            "t_stars_s": [],
            "k_stars": [],
            "keep_mask": keep.astype(bool).tolist(),
            "used_windows": int(len(y)),
            "total_windows": int(len(offsets_all)),
            "min_corr": float(min_corr),
            "sse0": sse0,
            "improve_ratio": 0.0,
            "note": "--max_cps=0 so forcing single.",
        }

    fit = best_piecewise_constant_up_to_k_cps(
        y=y,
        max_cps=int(max_cps),
        min_seg=int(min_seg_windows),
        cp_penalty=float(cp_penalty),
    )

    best_m = int(fit["best_m"])
    best_sse = float(fit["best_sse"])
    improve_ratio = float((sse0 - best_sse) / max(sse0, eps))

    if best_m <= 1 or improve_ratio < float(min_improve_ratio):
        d_single = float(np.median(y))
        return {
            "mode": "single",
            "deltas": [d_single],
            "t_stars_s": [],
            "k_stars": [],
            "keep_mask": keep.astype(bool).tolist(),
            "used_windows": int(len(y)),
            "total_windows": int(len(offsets_all)),
            "min_corr": float(min_corr),
            "sse0": sse0,
            "best_sse": best_sse,
            "improve_ratio": improve_ratio,
            "fit": fit,
        }

    cps = [int(k) for k in fit["best_cps"]]
    bounds = [0] + cps + [len(y)]

    deltas = [float(np.median(y[a:b])) for a, b in zip(bounds[:-1], bounds[1:])]

    t_stars = []
    for k in cps:
        if k <= 0:
            t_stars.append(float(c[0]))
        elif k >= len(c):
            t_stars.append(float(c[-1]))
        else:
            t_stars.append(float(0.5 * (c[k - 1] + c[k])))

    return {
        "mode": "piecewise",
        "deltas": deltas,
        "t_stars_s": t_stars,
        "k_stars": cps,
        "keep_mask": keep.astype(bool).tolist(),
        "used_windows": int(len(y)),
        "total_windows": int(len(offsets_all)),
        "min_corr": float(min_corr),
        "sse0": sse0,
        "best_sse": best_sse,
        "improve_ratio": improve_ratio,
        "fit": fit,
    }


# =========================================================
# 6) Compute final correlations for chosen piecewise deltas
# =========================================================

def compute_final_corrs_multi(
    t_grid: np.ndarray,
    rgb_g: np.ndarray,
    lidar_g: np.ndarray,
    deltas: List[float],
    t_stars_s: List[float],
) -> Dict[str, Any]:
    """Compute corr per piece and overall corr for piecewise-shifted LiDAR."""
    deltas = [float(d) for d in deltas]
    t_stars_s = [float(t) for t in t_stars_s]

    if len(deltas) <= 0:
        return {"corr_overall": -1.0, "corr_per_piece": []}

    t0 = float(t_grid[0])
    t_stars_abs = [t0 + ts for ts in t_stars_s]

    seg_idx = np.zeros_like(t_grid, dtype=np.int32)
    for tstar in t_stars_abs:
        seg_idx[t_grid >= tstar] += 1
    seg_idx = np.clip(seg_idx, 0, len(deltas) - 1)

    corr_per_piece = []
    shifted = np.full_like(lidar_g, np.nan, dtype=np.float64)

    for s in range(len(deltas)):
        mask = seg_idx == s
        d = deltas[s]
        corr_s = corr_for_delta_on_mask(t_grid, rgb_g, lidar_g, d, mask)
        corr_per_piece.append(float(corr_s))
        shifted[mask] = np.interp(t_grid[mask] + d, t_grid, lidar_g, left=np.nan, right=np.nan)

    shifted_filled = np.nan_to_num(shifted, nan=0.0)
    corr_overall = corr_at_shift(rgb_g, shifted_filled)

    return {
        "corr_overall": float(corr_overall),
        "corr_per_piece": corr_per_piece,
    }


# =========================================================
# 7) Plotting
# =========================================================

def plot_offset_timeseries(
    out_png: Path,
    centers_s: np.ndarray,
    offsets: np.ndarray,
    corrs: np.ndarray,
    min_corr: float,            # ✅ NEW: always compute gray/good by corrs vs min_corr
    decision: Dict[str, Any],
    title: str,
):
    """
    Plot offset(t_center) with corr + offset annotations.

    ✅ Behavior (robust):
      - Gray points: corr < min_corr  (always computed from `corrs`, NOT from decision.keep_mask)
      - Good points: corr >= min_corr (line + markers)
      - Every point shows:
          c=<corr>
          Δ=<offset>
        (two-line label)

      - Legend includes:
          * used/filtered label with threshold
          * Δrange among GOOD points:
              max(Δ_good) - min(Δ_good), plus min/max

      - Decision overlays:
          * horizontal lines for each segment delta
          * vertical lines at changepoints t*
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    centers_s = np.asarray(centers_s, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    corrs = np.asarray(corrs, dtype=np.float64)
    min_corr = float(min_corr)

    # finite mask
    finite = np.isfinite(centers_s) & np.isfinite(offsets) & np.isfinite(corrs)

    # ✅ robust good/bad computed HERE
    good = finite & (corrs >= min_corr)
    bad = finite & (corrs < min_corr)

    plt.figure(figsize=(12, 5))

    # --- gray points (filtered) ---
    if bad.any():
        plt.scatter(
            centers_s[bad],
            offsets[bad],
            s=42,
            marker="o",
            color="gray",
            alpha=0.9,
            zorder=3,
            label=f"filtered: corr < {min_corr:g}",
        )

    # --- good line + markers ---
    if good.any():
        # connect only good points
        plt.plot(
            centers_s[good],
            offsets[good],
            marker="o",
            linewidth=1.2,
            zorder=4,
            label=f"used: corr >= {min_corr:g}",
        )

    # --- annotate EACH point with corr + offset ---
    # two-line label; keep it small; offset a bit so it doesn't cover marker
    for x, d, c in zip(centers_s, offsets, corrs):
        if not (np.isfinite(x) and np.isfinite(d) and np.isfinite(c)):
            continue
        txt = f"c={float(c):.2f}\nΔ={float(d):+.3f}"
        plt.text(
            float(x),
            float(d),
            txt,
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=5,
        )

    # --- legend add: Δrange among GOOD points ---
    if good.any():
        d_min = float(np.min(offsets[good]))
        d_max = float(np.max(offsets[good]))
        d_range = float(d_max - d_min)
        # add an "empty" legend entry
        plt.plot(
            [],
            [],
            linestyle="None",
            label=f"Δrange(used)={d_range:.3f}s  min={d_min:+.3f}  max={d_max:+.3f}",
        )
    else:
        plt.plot([], [], linestyle="None", label="Δrange(used)=N/A (no corr>=min_corr points)")

    # --- decision overlays (piecewise or single) ---
    mode = decision.get("mode", "single")
    if mode == "piecewise":
        deltas = [float(d) for d in decision.get("deltas", [])]
        t_stars = [float(t) for t in decision.get("t_stars_s", [])]

        for i, d in enumerate(deltas):
            plt.axhline(d, linestyle="--", linewidth=1.0, label=f"Δ{i+1}={d:+.3f}s")
        for i, t in enumerate(t_stars):
            plt.axvline(t, linestyle=":", linewidth=2.0, label=f"t*{i+1}={t:.2f}s")
    else:
        d = float(decision.get("deltas", [0.0])[0])
        plt.axhline(d, linestyle="--", linewidth=1.2, label=f"Δ={d:+.3f}s")

    plt.xlabel("Time since run start (s) [grid]")
    plt.ylabel("Estimated offset (s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_alignment_piecewise(
    out_png: Path,
    t_grid: np.ndarray,
    rgb_g: np.ndarray,
    lidar_g: np.ndarray,
    deltas: List[float],
    t_stars_s: List[float],
    final_corrs: Dict[str, Any],
    title: str,
):
    """
    Plot:
      - RGB motion (z)
      - LiDAR motion (z)
      - piecewise-shifted LiDAR motion (z)
    with vertical lines at changepoints.

    Legend includes:
      corr per piece + corr overall.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    t0 = float(t_grid[0])
    x = t_grid - t0

    rgb_z = zscore(rgb_g)
    lidar_z = zscore(lidar_g)

    deltas = [float(d) for d in deltas]
    t_stars_s = [float(t) for t in t_stars_s]

    t_stars_abs = [t0 + ts for ts in t_stars_s]
    seg_idx = np.zeros_like(t_grid, dtype=np.int32)
    for tstar in t_stars_abs:
        seg_idx[t_grid >= tstar] += 1
    seg_idx = np.clip(seg_idx, 0, len(deltas) - 1)

    shifted = np.full_like(lidar_g, np.nan, dtype=np.float64)
    for s in range(len(deltas)):
        mask = seg_idx == s
        d = deltas[s]
        shifted[mask] = np.interp(t_grid[mask] + d, t_grid, lidar_g, left=np.nan, right=np.nan)

    shifted_z = zscore(np.nan_to_num(shifted, nan=0.0))

    corr_overall = float(final_corrs.get("corr_overall", float("nan")))
    corr_per_piece = final_corrs.get("corr_per_piece", [])
    corr_str = " ".join([f"c{i+1}={float(c):.2f}" for i, c in enumerate(corr_per_piece)])
    label = f"LiDAR shifted piecewise | {corr_str} overall={corr_overall:.2f}"

    plt.figure(figsize=(12, 6))
    plt.plot(x, rgb_z, label="RGB motion (grid, z)")
    plt.plot(x, lidar_z, label="LiDAR motion (grid, z)")
    plt.plot(x, shifted_z, label=label)

    for i, t in enumerate(t_stars_s):
        plt.axvline(float(t), linestyle=":", linewidth=2.0, label=f"t*{i+1}")

    plt.xlabel("Time since start (s)")
    plt.ylabel("Z-scored motion")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# =========================================================
# 8) Core per-run processing
# =========================================================

def process_one_run_piecewise(
    rgb_dir: Path,
    lidar_dir: Path,
    out_dir: Path,
    lidar_feature: str,
    search_s: float,
    step_s: float,
    grid_dt: float,
    warmup_s: float,
    tail_trim_s: float,
    lidar_r_max: float,
    lidar_hist_bins: int,
    ts_tz: Optional[str],
    win_s: float,
    hop_s: float,
    min_corr: float,
    min_seg_windows: int,
    cp_penalty: float,
    min_improve_ratio: float,
    max_cps: int,
) -> Dict[str, Any]:
    """One run end-to-end."""
    rgb_frames = list_files_sorted_by_ts(rgb_dir, ".jpg", tz_name=ts_tz)
    if not rgb_frames:
        raise RuntimeError(f"No RGB frames found in: {rgb_dir}")

    lidar_frames = list_files_sorted_by_ts(lidar_dir, ".csv", tz_name=ts_tz)
    lidar_frames = [(t, p) for (t, p) in lidar_frames if p.name.endswith("_cloud.csv")]
    if not lidar_frames:
        raise RuntimeError(f"No LiDAR CSV frames found in: {lidar_dir}")

    # RGB frame times + ts strings for window logging
    rgb_frame_times = np.asarray([float(t) for (t, _) in rgb_frames], dtype=np.float64)
    rgb_frame_ts_strs: List[str] = []
    for _, p in rgb_frames:
        m = TS_RE.search(p.name)
        rgb_frame_ts_strs.append(m.group(1) if m else p.name)

    t_rgb, m_rgb = compute_rgb_motion(rgb_frames, down_w=160, down_h=90, diff_threshold=0, smooth_win_s=0.2)
    t_lidar, m_lidar = compute_lidar_motion(
        lidar_frames, feature=lidar_feature, smooth_win_s=0.3, hist_bins=lidar_hist_bins, r_max=lidar_r_max
    )

    if len(t_rgb) < 10 or len(t_lidar) < 5:
        raise RuntimeError(f"Not enough motion samples. RGB={len(t_rgb)}, LiDAR={len(t_lidar)}")

    t_start0 = float(max(t_rgb[0], t_lidar[0]))
    t_end0 = float(min(t_rgb[-1], t_lidar[-1]))

    t_start = t_start0 + float(max(0.0, warmup_s))
    t_end = t_end0 - float(max(0.0, tail_trim_s))
    if t_end <= t_start:
        raise RuntimeError(
            f"Invalid trimmed window: t_end<=t_start. intersection=[{t_start0:.3f},{t_end0:.3f}] "
            f"warmup={warmup_s} tail_trim={tail_trim_s}"
        )
    if (t_end - t_start) < max(8.0, win_s + 2.0):
        raise RuntimeError(
            f"Overlap too short after trim: {(t_end - t_start):.2f}s; need at least ~{max(8.0, win_s + 2.0):.1f}s"
        )

    t_grid = np.arange(t_start, t_end, float(grid_dt), dtype=np.float64)
    if len(t_grid) < 300:
        raise RuntimeError(f"Too few grid points: {len(t_grid)} (grid_dt={grid_dt})")

    rgb_g = interp_to_grid(t_rgb, zscore(m_rgb), t_grid)
    lidar_g = interp_to_grid(t_lidar, zscore(m_lidar), t_grid)

    sw = sliding_window_offsets(
        t_grid=t_grid,
        rgb_g=rgb_g,
        lidar_g=lidar_g,
        win_s=float(win_s),
        hop_s=float(hop_s),
        search_s=float(search_s),
        step_s=float(step_s),
        min_points=max(300, int(0.7 * win_s / max(grid_dt, 1e-6))),
        t_rgb_mid=t_rgb,
        rgb_frame_times=rgb_frame_times,
        rgb_frame_ts_strs=rgb_frame_ts_strs,
    )

    centers_s = np.asarray(sw["centers_s"], dtype=np.float64)
    offsets = np.asarray(sw["offsets"], dtype=np.float64)
    corrs = np.asarray(sw["corrs"], dtype=np.float64)

    # decide piecewise
    if offsets.size < max(2 * min_seg_windows + 1, min_seg_windows + 1):
        best_delta, best_corr = estimate_offset_by_scan(t_grid, rgb_g, lidar_g, search_s=float(search_s), step_s=float(step_s))
        decision = {
            "mode": "single",
            "deltas": [float(best_delta)],
            "t_stars_s": [],
            "k_stars": [],
            "keep_mask": [True] * int(len(offsets)),  # NOTE: plotting no longer depends on this
            "note": "Too few sliding windows; fallback to global scan on full run.",
            "fallback_best_corr": float(best_corr),
        }
    else:
        decision = decide_piecewise_offsets_multi(
            centers_s_all=centers_s,
            offsets_all=offsets,
            corrs_all=corrs,
            min_corr=float(min_corr),
            max_cps=int(max_cps),
            min_seg_windows=int(min_seg_windows),
            cp_penalty=float(cp_penalty),
            min_improve_ratio=float(min_improve_ratio),
        )

    deltas = decision["deltas"]
    t_stars_s = decision.get("t_stars_s", [])

    final_corrs = compute_final_corrs_multi(t_grid, rgb_g, lidar_g, deltas=deltas, t_stars_s=t_stars_s)
    decision["final_corrs"] = final_corrs

    run_name = rgb_dir.parents[1].name
    lidar_session = lidar_dir.name

    out_plots = out_dir / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)

    out_offset_png = out_plots / f"{run_name}__{lidar_session}__{lidar_feature}__offset_timeseries.png"
    out_align_png = out_plots / f"{run_name}__{lidar_session}__{lidar_feature}__alignment_piecewise.png"

    title1 = f"{run_name} | session={lidar_session} | feature={lidar_feature} | offsets(t)"

    corr_overall = float(final_corrs.get("corr_overall", float("nan")))
    corr_per_piece = final_corrs.get("corr_per_piece", [])
    if decision.get("mode") == "piecewise":
        corr_str = " ".join([f"c{i+1}={float(c):.2f}" for i, c in enumerate(corr_per_piece)])
        title2 = (
            f"{run_name} | session={lidar_session} | feature={lidar_feature} | "
            f"PIECEWISE segs={len(deltas)} cps={len(t_stars_s)} | {corr_str} overall={corr_overall:.2f}"
        )
    else:
        title2 = f"{run_name} | session={lidar_session} | feature={lidar_feature} | SINGLE Δ={float(deltas[0]):+.3f}s | corr={corr_overall:.2f}"

    # ✅ PLOT: pass min_corr (do NOT pass keep_mask anymore)
    plot_offset_timeseries(
        out_png=out_offset_png,
        centers_s=centers_s,
        offsets=offsets,
        corrs=corrs,
        min_corr=float(min_corr),
        decision=decision,
        title=title1,
    )

    plot_alignment_piecewise(
        out_png=out_align_png,
        t_grid=t_grid,
        rgb_g=rgb_g,
        lidar_g=lidar_g,
        deltas=deltas,
        t_stars_s=t_stars_s,
        final_corrs=final_corrs,
        title=title2,
    )

    result: Dict[str, Any] = {
        "run_name": run_name,
        "lidar_session": lidar_session,
        "rgb_dir": str(rgb_dir),
        "lidar_dir": str(lidar_dir),

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

        "sliding_window": {
            "win_s": float(win_s),
            "hop_s": float(hop_s),
            "min_corr": float(min_corr),

            "centers_s": sw["centers_s"],
            "centers_abs": sw["centers_abs"],

            "win_start_abs": sw.get("win_start_abs", []),
            "win_end_abs": sw.get("win_end_abs", []),

            "rgb_in_win_count": sw.get("rgb_in_win_count", []),
            "rgb_win_start_abs": sw.get("rgb_win_start_abs", []),
            "rgb_win_end_abs": sw.get("rgb_win_end_abs", []),

            "rgb_win_start_ts": sw.get("rgb_win_start_ts", []),
            "rgb_win_end_ts": sw.get("rgb_win_end_ts", []),

            "offsets": sw["offsets"],
            "corrs": sw["corrs"],
        },

        "piecewise": decision,

        "plot_offset_timeseries": str(out_offset_png),
        "plot_alignment_piecewise": str(out_align_png),
    }
    return result


# =========================================================
# 9) Safe JSONL writer (main process only)
# =========================================================

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append one JSON object per line (JSONL). Must be main process only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


# =========================================================
# 10) Worker (top-level for Windows)
# =========================================================

def worker_process_one_run(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: compute one run piecewise offsets."""
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        res = process_one_run_piecewise(
            rgb_dir=Path(job["rgb_dir"]),
            lidar_dir=Path(job["lidar_dir"]),
            out_dir=Path(job["out_dir"]),
            lidar_feature=str(job["lidar_feature"]),
            search_s=float(job["search_s"]),
            step_s=float(job["step_s"]),
            grid_dt=float(job["grid_dt"]),
            warmup_s=float(job["warmup_s"]),
            tail_trim_s=float(job["tail_trim_s"]),
            lidar_r_max=float(job["lidar_r_max"]),
            lidar_hist_bins=int(job["lidar_hist_bins"]),
            ts_tz=job.get("ts_tz", None),
            win_s=float(job["win_s"]),
            hop_s=float(job["hop_s"]),
            min_corr=float(job["min_corr"]),
            min_seg_windows=int(job["min_seg_windows"]),
            cp_penalty=float(job["cp_penalty"]),
            min_improve_ratio=float(job["min_improve_ratio"]),
            max_cps=int(job["max_cps"]),
        )
        res["subject_root"] = str(job["subject_root"])
        return res

    except Exception as e:
        return {
            "subject_root": str(job["subject_root"]),
            "run_name": str(job.get("run_name", "")),
            "rgb_dir": str(job["rgb_dir"]),
            "lidar_dir": str(job["lidar_dir"]),
            "error": str(e),
        }


# =========================================================
# 11) Per-subject driver
# =========================================================

def process_one_subject(subject_root: Path, args) -> None:
    """Traverse runs, match LiDAR sessions, dispatch jobs."""
    kinect_root = subject_root / "kinect"
    vlp16_root = subject_root / "vlp16"

    if not kinect_root.exists():
        print(f"[SKIP subject] Missing: {kinect_root}")
        return
    if not vlp16_root.exists():
        print(f"[SKIP subject] Missing: {vlp16_root}")
        return

    if args.out_root:
        out_dir = Path(args.out_root) / subject_root.name / args.out_folder
    else:
        out_dir = subject_root / args.out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "run_piecewise_offsets.jsonl"
    out_json = out_dir / "run_piecewise_offsets.json"

    only_run_set = set(args.only_run) if args.only_run else None

    def should_run(run_name: str) -> bool:
        return True if only_run_set is None else (run_name in only_run_set)

    run_dirs = sorted([p for p in kinect_root.glob("run_*") if p.is_dir()])
    if not run_dirs:
        print(f"[SKIP subject] No run_* found under: {kinect_root}")
        return

    print(f"\n========== SUBJECT: {subject_root} ==========")
    print(f"Output -> {out_dir}")

    jobs = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        if not should_run(run_name):
            continue

        rgb_dir = run_dir / args.camera_id / "frames_rgb_blured"
        if not rgb_dir.exists():
            continue

        lidar_run_dir = vlp16_root / run_name
        if not lidar_run_dir.exists():
            continue

        sessions = [p for p in lidar_run_dir.iterdir() if p.is_dir()]
        if not sessions:
            continue
        sessions.sort(key=lambda p: p.name)
        lidar_dir = sessions[-1]

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

            "warmup_s": args.warmup_s,
            "tail_trim_s": args.tail_trim_s,

            "lidar_r_max": args.lidar_r_max,
            "lidar_hist_bins": args.lidar_hist_bins,

            "ts_tz": args.ts_tz if args.ts_tz else None,

            "win_s": args.win_s,
            "hop_s": args.hop_s,
            "min_corr": args.min_corr,

            "min_seg_windows": args.min_seg_windows,
            "cp_penalty": args.cp_penalty,
            "min_improve_ratio": args.min_improve_ratio,

            "max_cps": args.max_cps,
        })

    if not jobs:
        print(f"[SKIP subject] No runnable runs for: {subject_root}")
        return

    results: List[Dict[str, Any]] = []
    n_workers = max(1, int(args.jobs))

    if n_workers == 1:
        for job in jobs:
            print(f"[{job['run_name']}] RGB={job['rgb_dir']} | LiDAR={job['lidar_dir']}")
            res = worker_process_one_run(job)

            append_jsonl(out_jsonl, res)
            results.append(res)

            if "error" in res:
                print(f"  !! FAILED: {job['run_name']}: {res['error']}")
            else:
                pw = res.get("piecewise", {})
                fc = pw.get("final_corrs", {})
                if pw.get("mode") == "piecewise":
                    deltas = pw.get("deltas", [])
                    corr_per = fc.get("corr_per_piece", [])
                    print(
                        f"  -> PIECEWISE segs={len(deltas)} cps={len(pw.get('t_stars_s', []))} "
                        f"deltas={[round(float(d),3) for d in deltas]} "
                        f"corrs={[round(float(c),2) for c in corr_per]} overall={float(fc.get('corr_overall', float('nan'))):.2f}"
                    )
                else:
                    print(f"  -> SINGLE Δ={float(pw.get('deltas', [0.0])[0]):+.3f}s | corr={float(fc.get('corr_overall', float('nan'))):.2f}")
                print(f"  -> plot(offset): {res['plot_offset_timeseries']}")
                print(f"  -> plot(align) : {res['plot_alignment_piecewise']}")

    else:
        print(f"[Parallel] workers={n_workers}, total_runs={len(jobs)}")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            fut_to_job = {ex.submit(worker_process_one_run, job): job for job in jobs}
            for fut in as_completed(fut_to_job):
                job = fut_to_job[fut]
                res = fut.result()

                append_jsonl(out_jsonl, res)
                results.append(res)

                if "error" in res:
                    print(f"  !! FAILED: {job['run_name']}: {res['error']}")
                else:
                    pw = res.get("piecewise", {})
                    fc = pw.get("final_corrs", {})
                    if pw.get("mode") == "piecewise":
                        deltas = pw.get("deltas", [])
                        print(f"  -> DONE {job['run_name']}: PIECEWISE segs={len(deltas)} cps={len(pw.get('t_stars_s', []))} overall={float(fc.get('corr_overall', float('nan'))):.2f}")
                    else:
                        print(f"  -> DONE {job['run_name']}: SINGLE Δ={float(pw.get('deltas', [0.0])[0]):+.3f}s | corr={float(fc.get('corr_overall', float('nan'))):.2f}")

    results_sorted = sorted(results, key=lambda d: d.get("run_name", ""))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    print(f"\n[Subject done] {subject_root}")
    print(f"Saved summary JSON: {out_json}")
    print(f"Saved per-run JSONL: {out_jsonl}")
    print(f"Plots folder: {(out_dir / 'plots')}")
    if only_run_set:
        print(f"Only-runs filter used: {sorted(only_run_set)}")


# =========================================================
# 12) main
# =========================================================

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
    ap.add_argument("--grid_dt", type=float, default=0.01, help="Unified time grid dt (seconds)")

    ap.add_argument(
        "--out_root", type=str, default=None,
        help="If set, write outputs to out_root/<subject_name>/<out_folder>. Otherwise per-subject <subject_root>/<out_folder>"
    )
    ap.add_argument("--out_folder", type=str, default="_sync_piecewise_out", help="Output folder name inside each subject output directory")

    ap.add_argument("--lidar_r_max", type=float, default=3.0)
    ap.add_argument("--lidar_hist_bins", type=int, default=40)

    ap.add_argument("--warmup_s", type=float, default=6.0, help="Drop warm-up seconds AFTER intersection start")
    ap.add_argument("--tail_trim_s", type=float, default=0.0, help="Trim last seconds from intersection end")

    ap.add_argument("--only_run", nargs="*", default=None, help="Only process specified run(s), e.g. --only_run run_10 run_1 run_8-37")

    ap.add_argument("--jobs", type=int, default=3, help="Parallel processes PER SUBJECT. 1=no parallel")

    ap.add_argument("--ts_tz", type=str, default=None, help="Timezone used to interpret filename timestamps. Examples: Europe/London, UTC")

    ap.add_argument("--win_s", type=float, default=15.0, help="Sliding window length (seconds) for local offset estimation")
    ap.add_argument("--hop_s", type=float, default=10.0, help="Sliding window hop (seconds)")
    ap.add_argument("--min_corr", type=float, default=0.15, help="Windows with corr < min_corr are filtered (gray in plot; not used for CP)")

    ap.add_argument("--max_cps", type=int, default=1, help="Max number of changepoints to detect (0 => force single)")
    ap.add_argument("--min_seg_windows", type=int, default=3, help="Min number of windows per segment")
    ap.add_argument("--cp_penalty", type=float, default=0.0, help="Penalty per changepoint (bigger => fewer CPs)")
    ap.add_argument("--min_improve_ratio", type=float, default=0.10, help="Accept piecewise only if SSE improves by at least this ratio (0~1)")

    args = ap.parse_args()

    if args.out_root:
        Path(args.out_root).mkdir(parents=True, exist_ok=True)

    for s in args.subject_roots:
        subject_root = Path(s)
        if not subject_root.exists():
            print(f"[SKIP subject] Not found: {subject_root}")
            continue
        process_one_subject(subject_root, args)


if __name__ == "__main__":
    main()
