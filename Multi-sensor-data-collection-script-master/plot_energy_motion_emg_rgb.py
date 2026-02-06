#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Multi-subject: Visualize alignment between RGB frames (30Hz) and Armband (MindRove EMG+IMU)
by plotting motion-energy curves (NO cross-correlation, NO offset estimation).

NEW FEATURES (added without breaking existing workflow):
  A) (Optional) Top x-axis: map trimmed window back to FULL RGB timeline
     - bottom axis: time since trimmed start (unchanged)
     - top axis   : time since RGB run start (full RGB timeline)

  B) Peak detection + map peaks back to nearest original RGB frame filename
     - detect peaks on rgb_z / sig_z (z-score space) inside the trimmed window
     - optionally require event mask (both strong) before considering peaks
     - for each detected peak time, locate nearest RGB frame in the FULL run frames list
       (binary search; robust and fast)
     - save peak->frame mapping into JSON/JSONL summaries
     - (optional) annotate peaks on plot

The purpose:
  When you see a peak, you can immediately know:
    - the peak happens at which second of the original run (full RGB timeline)
    - the nearest original RGB frame filename (timestamped jpg) to check the video

Inputs:
RGB:
  subject_root/kinect/run_*/<camera_id>/<rgb_subdir>/*.jpg
  timestamp in filename: YYYYMMDD_HHMMSS_micro

Armband:
  subject_root/mindrove/sub-*_run-xxx[_yyy]_emg/left|right/*_emg_imu.csv
  timestamp in CSV: board_ts (epoch seconds float)

Outputs per subject:
  <subject_out_dir>/
    run_plots.jsonl
    run_plots.json
    plots/
      run_1__armband__left.png
      run_1__armband__right.png
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from bisect import bisect_left
from typing import List, Tuple, Optional, Dict
from zoneinfo import ZoneInfo

import numpy as np
import cv2

# ---- Force non-GUI backend to avoid Qt warnings (important on Windows) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Timestamp parsing (RGB)
# =========================
TS_RE = re.compile(r"(\d{8}_\d{6}_\d{6})")  # YYYYMMDD_HHMMSS_micro


def parse_ts_from_name(name: str, tz_name: str = "Europe/London") -> Optional[float]:
    m = TS_RE.search(name)
    if not m:
        return None
    s = m.group(1)
    try:
        dt = datetime.strptime(s, "%Y%m%d_%H%M%S_%f")
        dt = dt.replace(tzinfo=ZoneInfo(tz_name))   # 关键：解释为 tz_name 的本地时间
        return dt.timestamp()                       # 关键：转成真正 epoch 秒
    except Exception:
        return None


def list_files_sorted_by_ts(folder: Path, suffix: str, tz_name: str) -> List[Tuple[float, Path]]:
    """
    List files in folder matching suffix, parse timestamps from filenames,
    filter out bogus timestamps (< 2010-01-01), return sorted (t, path).
    """
    items: List[Tuple[float, Path]] = []
    for p in folder.glob(f"*{suffix}"):
        t = parse_ts_from_name(p.name, tz_name=tz_name)
        if t is None or not np.isfinite(t):
            continue
        if t < 1262304000.0:  # 2010-01-01
            continue
        items.append((t, p))
    items.sort(key=lambda x: x[0])
    return items


def epoch_to_timestr(t_epoch: float, tz_name: str = "Europe/London") -> str:
    dt = datetime.fromtimestamp(float(t_epoch), tz=ZoneInfo(tz_name))
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# =========================
# Signal utils
# =========================
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
    """Interpolate samples (t,v) onto time grid, clamping out-of-range."""
    if len(t) < 2:
        return np.zeros_like(t_grid, dtype=np.float64)
    tmin, tmax = float(t[0]), float(t[-1])
    return np.interp(np.clip(t_grid, tmin, tmax), t, v)


# =========================
# Event gating (optional overlay)
# =========================
def build_event_mask(
    rgb_z: np.ndarray,
    sig_z: np.ndarray,
    event_z: float,
    dilate_bins: int = 0,
) -> np.ndarray:
    """
    Build boolean event mask where BOTH signals are strong:
      mask = (rgb_z > event_z) & (sig_z > event_z)

    Optional dilation (to make events more continuous):
      dilate_bins = number of bins to dilate on each side.
    """
    mask = (rgb_z > event_z) & (sig_z > event_z)
    if dilate_bins and dilate_bins > 0:
        k = 2 * int(dilate_bins) + 1
        x = np.convolve(mask.astype(np.float64), np.ones(k, dtype=np.float64), mode="same")
        mask = x > 0
    return mask.astype(bool)


# =========================
# RGB motion signal
# =========================
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

        t_mid.append(0.5 * (t0 + t1))
        m.append(val)

        t0, im0 = t1, im1

    t_mid = np.asarray(t_mid, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)

    if len(t_mid) >= 3 and smooth_win_s > 0:
        dt = float(np.median(np.diff(t_mid)))
        win = max(1, int(round(smooth_win_s / dt)))
        m = moving_average(m, win)

    return t_mid, m


# =========================
# Armband motion signal
# =========================
def load_armband_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Read armband CSV (header required). Expected columns:
      board_ts, emg1..emg8, acc_x acc_y acc_z, gyro_x gyro_y gyro_z
    """
    try:
        data = np.genfromtxt(
            str(csv_path),
            delimiter=",",
            names=True,
            dtype=np.float64,
            encoding="utf-8",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read armband CSV {csv_path}: {e}")

    if data.size == 0:
        raise RuntimeError(f"Empty armband CSV: {csv_path}")

    def col(name: str) -> np.ndarray:
        if name not in data.dtype.names:
            raise RuntimeError(f"Missing column '{name}' in {csv_path.name}")
        return np.asarray(data[name], dtype=np.float64)

    t = col("board_ts")
    ok = np.isfinite(t)
    if ok.sum() < 10:
        raise RuntimeError(f"Too few valid board_ts in {csv_path}")

    # keep finite rows only
    t = t[ok]
    idx = np.argsort(t)
    t = t[idx]

    emg_cols = [f"emg{i}" for i in range(1, 9)]
    emg = np.stack([col(c)[ok][idx] for c in emg_cols], axis=1)

    gyro = np.stack(
        [col("gyro_x")[ok][idx], col("gyro_y")[ok][idx], col("gyro_z")[ok][idx]],
        axis=1,
    )

    # drop duplicate timestamps (sometimes logging repeats same board_ts)
    uniq_mask = np.concatenate([[True], np.diff(t) > 0])
    t = t[uniq_mask]
    emg = emg[uniq_mask]
    gyro = gyro[uniq_mask]

    return {"t": t, "emg": emg, "gyro": gyro}


def compute_armband_motion(
    csv_path: Path,
    mode: str = "fusion",
    smooth_win_s_emg: float = 0.08,
    smooth_win_s_imu: float = 0.15,
    emg_hpf_like: bool = True,
    imu_use_energy: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build armband motion-energy.

    EMG (~500Hz):
      envelope = mean(abs(emg1..8))
      motion   = abs(diff(envelope))  (high-pass-like) or envelope itself if emg_hpf_like=False
      then smooth with smooth_win_s_emg

    IMU (~50Hz):
      gyro_mag = norm(gyro)
      motion   = smooth( gyro_mag^2 ) if imu_use_energy else smooth( gyro_mag )
      (No diff -> stable peaks, easier to eyeball)

    Returns:
      t_mid: midpoint timestamps between consecutive samples (after dt filtering)
      m: motion energy (same length as t_mid)
    """
    d = load_armband_csv(csv_path)
    t = d["t"]
    emg = d["emg"]
    gyro = d["gyro"]

    if len(t) < 3:
        return np.array([]), np.array([])

    # midpoint time between consecutive samples
    t_mid = 0.5 * (t[:-1] + t[1:])
    dt = np.diff(t)

    # filter weird dt (logging hiccup)
    good = (dt > 0.0005) & (dt < 0.2)
    if good.sum() < 50:
        good = np.ones_like(dt, dtype=bool)

    t_mid = t_mid[good]
    dt_good = dt[good]

    # ---------- EMG motion ----------
    emg_env = np.mean(np.abs(emg), axis=1)          # (N,)
    if emg_hpf_like:
        emg_raw = np.abs(np.diff(emg_env))          # (N-1,)
    else:
        emg_raw = emg_env[1:]                       # (N-1,)
    emg_raw = emg_raw[good]

    if len(t_mid) >= 5 and smooth_win_s_emg > 0:
        dt_med = float(np.median(dt_good))
        win = max(1, int(round(smooth_win_s_emg / dt_med)))
        emg_raw = moving_average(emg_raw, win)

    # ---------- IMU motion ----------
    gyro_mag = np.linalg.norm(gyro, axis=1)         # (N,)
    imu_base = gyro_mag**2 if imu_use_energy else gyro_mag
    imu_raw = imu_base[1:][good]                    # align to (N-1,)

    if len(t_mid) >= 5 and smooth_win_s_imu > 0:
        dt_med = float(np.median(dt_good))
        win = max(1, int(round(smooth_win_s_imu / dt_med)))
        imu_raw = moving_average(imu_raw, win)

    # ---------- combine ----------
    if mode == "emg":
        m = emg_raw
    elif mode == "imu":
        m = imu_raw
    elif mode == "fusion":
        m = np.abs(zscore(emg_raw) + zscore(imu_raw))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return np.asarray(t_mid, dtype=np.float64), np.asarray(m, dtype=np.float64)


# =========================
# Run matching: kinect run_*  <->  mindrove run-xxx or run-xxx-yyy
# =========================
RUN_KINECT_RE = re.compile(r"^run_(\d+)(?:-(\d+))?$", re.IGNORECASE)
RUN_MINDROVE_RE = re.compile(r"run-(\d+)(?:-(\d+))?_emg", re.IGNORECASE)


def parse_kinect_run_range(run_name: str) -> Optional[Tuple[int, int]]:
    m = RUN_KINECT_RE.match(run_name.strip())
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    return (min(a, b), max(a, b))


def parse_mindrove_run_range(folder_name: str) -> Optional[Tuple[int, int]]:
    m = RUN_MINDROVE_RE.search(folder_name)
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    return (min(a, b), max(a, b))


def run_overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def find_armband_folder_for_run(mindrove_root: Path, kinect_run_name: str) -> Optional[Path]:
    """
    Match mindrove folder to kinect run:
      - overlap run range
      - prefer exact match
      - else choose smallest range length (more specific)
    """
    kr = parse_kinect_run_range(kinect_run_name)
    if kr is None:
        return None

    cand = []
    for p in mindrove_root.iterdir():
        if not p.is_dir():
            continue
        rr = parse_mindrove_run_range(p.name)
        if rr is None:
            continue
        if run_overlaps(kr, rr):
            cand.append((rr, p))

    if not cand:
        return None

    # exact first
    for rr, p in cand:
        if rr == kr:
            return p

    # smallest range length
    cand.sort(key=lambda x: (x[0][1] - x[0][0], x[1].name))
    return cand[0][1]


def find_armband_csv(armband_folder: Path, side: str) -> Optional[Path]:
    """Find *_emg_imu.csv in <armband_folder>/<side>/."""
    side_dir = armband_folder / side
    if not side_dir.exists():
        return None
    cands = sorted(side_dir.glob("*_emg_imu.csv"))
    if not cands:
        cands = sorted(side_dir.glob("*.csv"))
    return cands[0] if cands else None


# =========================
# Peak detection + mapping to nearest RGB frame
# =========================
def find_local_peaks_1d(
    y: np.ndarray,
    x: np.ndarray,
    min_height: float = 1.0,
    min_prominence: float = 0.3,
    min_distance_s: float = 0.35,
    max_peaks: int = 30,
    valid_mask: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Simple, dependency-free local peak finder.

    Inputs:
      y: signal values (e.g., z-score curve)
      x: same length, time in seconds (e.g., x_trim = t_grid - t_grid[0])
      min_height: reject peaks below this y
      min_prominence: reject peaks whose (peak - max(neighbor_min)) too small
                      (rough prominence; not exact SciPy definition)
      min_distance_s: enforce spacing between peaks (keep stronger one)
      max_peaks: keep at most this many peaks (strongest first)
      valid_mask: optional boolean mask; only consider peaks where valid_mask[idx]=True

    Output:
      list of peak indices into y/x (sorted by time).
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = y.size
    if n < 3:
        return []

    if valid_mask is None:
        valid_mask = np.ones(n, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.size != n:
            raise ValueError("valid_mask length must equal y length")

    # 1) raw local maxima candidates
    cand = []
    for i in range(1, n - 1):
        if not valid_mask[i]:
            continue
        if y[i] >= min_height and y[i] > y[i - 1] and y[i] >= y[i + 1]:
            cand.append(i)
    if not cand:
        return []

    # 2) rough prominence filtering
    #    prominence ~ peak - max(min(left_window), min(right_window))
    #    Use a small window in time to estimate "local baseline"
    #    window size derived from min_distance_s (reasonable scale)
    half_win_s = max(0.2, 0.5 * float(min_distance_s))
    keep = []
    for i in cand:
        t0 = x[i] - half_win_s
        t1 = x[i] + half_win_s
        L = np.where((x >= t0) & (x <= x[i]))[0]
        R = np.where((x >= x[i]) & (x <= t1))[0]
        if L.size < 2 or R.size < 2:
            continue
        left_min = float(np.min(y[L]))
        right_min = float(np.min(y[R]))
        prom = float(y[i] - max(left_min, right_min))
        if prom >= min_prominence:
            keep.append((i, y[i], prom))
    if not keep:
        return []

    # 3) enforce min_distance_s (non-maximum suppression in time)
    #    Sort by peak height desc; pick and suppress neighbors within min_distance_s
    keep.sort(key=lambda t: (t[1], t[2]), reverse=True)
    selected = []
    for i, height, prom in keep:
        ok = True
        for j in selected:
            if abs(float(x[i] - x[j])) < min_distance_s:
                ok = False
                break
        if ok:
            selected.append(i)
        if len(selected) >= max_peaks:
            break

    selected = sorted(selected)
    return selected


def nearest_rgb_frame(
    rgb_frames: List[Tuple[float, Path]],
    t_epoch: float,
) -> Optional[Dict]:
    """
    Given full RGB frames list (sorted by time) and a target epoch time,
    return info of the nearest frame (by absolute time difference).

    Output dict:
      {
        "nearest_frame_name": "...jpg",
        "nearest_frame_path": "...",
        "nearest_frame_epoch": 123.456,
        "nearest_frame_timestr": "...",
        "nearest_frame_dt_s": 0.0123,   # frame_time - target_time
      }
    """
    if not rgb_frames:
        return None
    times = [t for t, _ in rgb_frames]
    k = bisect_left(times, float(t_epoch))

    # candidates: k-1 and k
    best_idx = None
    best_abs = None
    for idx in [k - 1, k]:
        if 0 <= idx < len(rgb_frames):
            dt = float(times[idx] - float(t_epoch))
            adt = abs(dt)
            if best_abs is None or adt < best_abs:
                best_abs = adt
                best_idx = idx

    if best_idx is None:
        return None

    tt, pp = rgb_frames[best_idx]
    return {
        "nearest_frame_name": pp.name,
        "nearest_frame_path": str(pp),
        "nearest_frame_epoch": float(tt),
        "nearest_frame_timestr": epoch_to_timestr(float(tt)),
        "nearest_frame_dt_s": float(float(tt) - float(t_epoch)),
    }


def build_peak_records(
    kind: str,
    peak_indices: List[int],
    t_grid: np.ndarray,
    x_trim: np.ndarray,
    y_used: np.ndarray,
    rgb_frames: List[Tuple[float, Path]],
    rgb_run_start_epoch: float,
    t_trim_start_epoch: float,
) -> List[Dict]:
    """
    Convert peak indices into serializable records, including mapping to nearest RGB frame.

    kind: "rgb" | "sig" | "joint"
    """
    recs = []
    for idx in peak_indices:
        t_peak = float(t_grid[idx])         # epoch seconds
        x_peak = float(x_trim[idx])         # seconds since trimmed start
        y_peak = float(y_used[idx])         # value on the curve we used to detect peaks (zscore space by default)

        m = nearest_rgb_frame(rgb_frames, t_peak)
        rec = {
            "kind": kind,
            "idx_grid": int(idx),
            "t_peak_epoch": t_peak,
            "t_peak_timestr": epoch_to_timestr(t_peak),
            "t_peak_since_trim_s": x_peak,
            "t_peak_since_rgb_start_s": float(t_peak - float(rgb_run_start_epoch)),
            "y_peak": y_peak,
            # nearest original RGB frame for video lookup
            "nearest_rgb": m,
            # convenience: "nearest frame's time since RGB start"
            "nearest_frame_since_rgb_start_s": None if m is None else float(m["nearest_frame_epoch"] - float(rgb_run_start_epoch)),
            "nearest_frame_since_trim_s": None if m is None else float(m["nearest_frame_epoch"] - float(t_trim_start_epoch)),
        }
        recs.append(rec)
    return recs


# =========================
# Plotting
# =========================
def plot_energy(
    out_png: Path,
    t_grid: np.ndarray,
    rgb_y: np.ndarray,
    sig_y: np.ndarray,
    title: str,
    mask_evt: Optional[np.ndarray] = None,
    ylabel: str = "Motion energy",
    # NEW: top axis mapping + optional peak annotation
    rgb_run_start_epoch: Optional[float] = None,
    add_top_axis_rgb: bool = False,
    add_top_axis_datetime: bool = False,
    annotate_peaks: bool = False,
    peak_xs: Optional[List[float]] = None, 
    top_axis_tick_s: float = 2.0  # x positions in seconds since trimmed start
):
    """
    Plot RGB and Armband motion energy curves on the same time axis.

    Bottom axis (unchanged): time since trimmed start (t_grid[0]).
    Top axis (optional): time since RGB run start (first RGB frame).
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # bottom axis: unchanged behavior
    x_trim = t_grid - t_grid[0]
    ax.plot(x_trim, rgb_y, label="RGB motion")
    ax.plot(x_trim, sig_y, label="Armband motion")

    if mask_evt is not None:
        y0 = min(float(np.min(rgb_y)), float(np.min(sig_y))) - 0.5
        y1 = y0 + 0.25
        band = mask_evt.astype(np.float64) * (y1 - y0) + y0
        ax.plot(x_trim, band, label="Event mask (band)")

    # NEW: annotate peaks by vertical lines (optional)
    if annotate_peaks and peak_xs:
        for xx in peak_xs:
            ax.axvline(float(xx), linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Time since trimmed start (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(False)

    # ---------- NEW: top axis mapping to full RGB timeline ----------
    if add_top_axis_rgb and (rgb_run_start_epoch is not None) and len(t_grid) > 0:
        offset_s = float(t_grid[0] - float(rgb_run_start_epoch))

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())

        # ===== Top axis ticks: every 2 seconds (avoid sparse ticks) =====
        # Bottom axis is x_trim (seconds since trimmed start)
        x0, x1 = ax.get_xlim()  # in seconds since trimmed start
        step = float(top_axis_tick_s)  # fallback

        # create ticks at multiples of step within [x0, x1]
        start_tick = np.ceil(x0 / step) * step
        ticks = np.arange(start_tick, x1 + 1e-6, step, dtype=np.float64)

        # label is seconds since RGB start: (since_trim + offset_s)
        labels = [f"{(float(t) + offset_s):.0f}" for t in ticks]

        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels)

        # Make font smaller + rotate slightly to prevent overlap
        ax2.tick_params(axis="x", labelsize=8, pad=2)
        for lab in ax2.get_xticklabels():
            lab.set_rotation(30)
            lab.set_ha("left")

        ax2.set_xlabel("Time since RGB run start (s)")

        # Optional sparse datetime hints (avoid clutter)
        if add_top_axis_datetime:
            quart = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 4)
            for q in quart:
                t_epoch = float(rgb_run_start_epoch) + (float(q) + offset_s)
                ax2.text(
                    q, 1.02,
                    epoch_to_timestr(t_epoch),
                    transform=ax2.get_xaxis_transform(),
                    ha="center", va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# =========================
# JSONL writer
# =========================
def append_jsonl(path: Path, obj: Dict):
    """Append one JSON object to a JSONL file (robust for long runs/crashes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


# =========================
# Per-run processing (NO correlation, only plotting)
# =========================
def process_one_run_plot(
    run_name: str,
    rgb_dir: Path,
    armband_csv: Path,
    out_dir: Path,
    side: str,
    args,
) -> Dict:
    """
    Plot RGB vs Armband motion energy curves for one run & one side.

    Steps:
      1) RGB motion (midpoint timestamps)
      2) Armband motion (midpoint timestamps)
      3) Intersection time window + trimming (warmup/taildrop)
      4) Interpolate both motion signals to unified time grid
      5) Optional zscore plotting (raw or z-score)
      6) Optional event mask overlay (computed in z-score space)
      7) NEW: detect peaks (rgb/sig/joint) and map peaks back to nearest original RGB frame
      8) Save plot + write rich JSON summary
    """
    # ---- RGB frames -> motion signal
    rgb_frames = list_files_sorted_by_ts(rgb_dir, ".jpg", tz_name=args.rgb_tz)
    if len(rgb_frames) < 2:
        raise RuntimeError(f"No/too few RGB frames in: {rgb_dir}")

    # Full RGB timeline origin (run start)
    rgb_run_start_epoch = float(rgb_frames[0][0])

    t_rgb, m_rgb = compute_rgb_motion(
        rgb_frames,
        down_w=args.rgb_down_w,
        down_h=args.rgb_down_h,
        diff_threshold=args.rgb_diff_threshold,
        smooth_win_s=args.rgb_smooth_s,
    )
    if len(t_rgb) < 10:
        raise RuntimeError(f"Not enough RGB motion samples: {len(t_rgb)}")

    # ---- Armband CSV -> motion signal
    t_sig, m_sig = compute_armband_motion(
        armband_csv,
        mode=args.armband_mode,
        smooth_win_s_emg=args.armband_smooth_emg,
        smooth_win_s_imu=args.armband_smooth_imu,
        emg_hpf_like=(not args.armband_emg_no_hpf),
        imu_use_energy=(not args.imu_no_energy),
    )
    if len(t_sig) < 50:
        raise RuntimeError(f"Not enough armband motion samples: {len(t_sig)}")

    # ---- Intersection window (raw)
    t_start0 = max(float(t_rgb[0]), float(t_sig[0]))
    t_end0 = min(float(t_rgb[-1]), float(t_sig[-1]))

    # ---- Apply head/tail trimming inside the intersection
    warmup_s = float(max(0.0, args.warmup_s))
    taildrop_s = float(max(0.0, args.taildrop_s))
    t_start = t_start0 + warmup_s
    t_end = t_end0 - taildrop_s

    if t_end <= t_start:
        raise RuntimeError(
            f"Invalid trimmed window: t_end<=t_start. "
            f"raw=[{t_start0:.3f},{t_end0:.3f}] warmup_s={warmup_s} taildrop_s={taildrop_s}"
        )

    overlap_used = t_end - t_start
    if overlap_used < args.min_overlap_s:
        raise RuntimeError(
            f"Overlap too short after warmup/taildrop: used={overlap_used:.2f}s "
            f"(min_overlap_s={args.min_overlap_s}, warmup_s={warmup_s}, taildrop_s={taildrop_s})"
        )

    # ---- Unified time grid (epoch seconds)
    t_grid = np.arange(t_start, t_end, args.grid_dt, dtype=np.float64)
    if len(t_grid) < args.min_grid_n:
        raise RuntimeError(
            f"Too few grid samples: {len(t_grid)} (grid_dt={args.grid_dt}, min_grid_n={args.min_grid_n})"
        )

    # ---- Interpolate to grid
    rgb_g = interp_to_grid(t_rgb, m_rgb, t_grid)
    sig_g = interp_to_grid(t_sig, m_sig, t_grid)

    # ---- Compute z-score curves (we always compute them for gating/peaks)
    rgb_z = zscore(rgb_g)
    sig_z = zscore(sig_g)

    # ---- Decide what to plot (existing behavior)
    if args.no_zscore:
        rgb_plot = rgb_g
        sig_plot = sig_g
        ylabel = "Motion energy (raw)"
    else:
        rgb_plot = rgb_z
        sig_plot = sig_z
        ylabel = "Motion energy (z-score)"

    # ---- Optional event mask overlay (existing behavior)
    mask_evt = None
    evt_meta = {"event_count": None, "event_ratio": None}
    if args.plot_event_mask:
        mask_evt = build_event_mask(
            rgb_z, sig_z,
            event_z=args.event_z,
            dilate_bins=args.event_dilate_bins,
        )
        evt_meta = {
            "event_count": float(mask_evt.sum()),
            "event_ratio": float(mask_evt.mean()),
        }

    # =========================
    # NEW: Peak detection
    # =========================
    # Time axis in seconds since trimmed start (handy for plot and min_distance)
    x_trim = t_grid - t_grid[0]

    # valid mask for peak candidates:
    # - if --peaks_require_event_mask: only search peaks where event mask True
    # - else: consider all points
    if args.peaks_require_event_mask:
        # if user requires event mask but plot_event_mask wasn't enabled, we still build it here
        _mask_evt_for_peaks = build_event_mask(
            rgb_z, sig_z, event_z=args.event_z, dilate_bins=args.event_dilate_bins
        )
        valid_mask = _mask_evt_for_peaks
    else:
        valid_mask = np.ones_like(rgb_z, dtype=bool)

    # Choose peak detection curve space:
    # - use z-score curves (recommended) regardless of plotting choice
    y_rgb_for_peaks = rgb_z
    y_sig_for_peaks = sig_z

    peaks_rgb_idx = []
    peaks_sig_idx = []
    peaks_joint_idx = []

    if args.find_peaks_rgb:
        peaks_rgb_idx = find_local_peaks_1d(
            y=y_rgb_for_peaks, x=x_trim,
            min_height=args.peak_min_z,
            min_prominence=args.peak_min_prom,
            min_distance_s=args.peak_min_dist_s,
            max_peaks=args.peak_max_n,
            valid_mask=valid_mask,
        )

    if args.find_peaks_sig:
        peaks_sig_idx = find_local_peaks_1d(
            y=y_sig_for_peaks, x=x_trim,
            min_height=args.peak_min_z,
            min_prominence=args.peak_min_prom,
            min_distance_s=args.peak_min_dist_s,
            max_peaks=args.peak_max_n,
            valid_mask=valid_mask,
        )

    if args.find_peaks_joint:
        # joint peaks: peaks on (rgb_z + sig_z) / 2, still under the same valid_mask
        y_joint = 0.5 * (y_rgb_for_peaks + y_sig_for_peaks)
        peaks_joint_idx = find_local_peaks_1d(
            y=y_joint, x=x_trim,
            min_height=args.peak_min_z,
            min_prominence=args.peak_min_prom,
            min_distance_s=args.peak_min_dist_s,
            max_peaks=args.peak_max_n,
            valid_mask=valid_mask,
        )

    # Convert peak indices to rich records with nearest RGB frame mapping
    peaks_rgb = build_peak_records(
        kind="rgb",
        peak_indices=peaks_rgb_idx,
        t_grid=t_grid,
        x_trim=x_trim,
        y_used=y_rgb_for_peaks,
        rgb_frames=rgb_frames,
        rgb_run_start_epoch=rgb_run_start_epoch,
        t_trim_start_epoch=float(t_grid[0]),
    )
    peaks_sig = build_peak_records(
        kind="sig",
        peak_indices=peaks_sig_idx,
        t_grid=t_grid,
        x_trim=x_trim,
        y_used=y_sig_for_peaks,
        rgb_frames=rgb_frames,
        rgb_run_start_epoch=rgb_run_start_epoch,
        t_trim_start_epoch=float(t_grid[0]),
    )
    # for joint peaks, store y_peak from y_joint (not from rgb/sig)
    peaks_joint = []
    if args.find_peaks_joint and peaks_joint_idx:
        y_joint = 0.5 * (y_rgb_for_peaks + y_sig_for_peaks)
        peaks_joint = build_peak_records(
            kind="joint",
            peak_indices=peaks_joint_idx,
            t_grid=t_grid,
            x_trim=x_trim,
            y_used=y_joint,
            rgb_frames=rgb_frames,
            rgb_run_start_epoch=rgb_run_start_epoch,
            t_trim_start_epoch=float(t_grid[0]),
        )

    # Optional: annotate *some* peaks on plot (avoid clutter by using joint or rgb peaks)
    peak_xs_for_plot = None
    if args.annotate_peaks_on_plot:
        # Priority: joint peaks, else rgb peaks, else sig peaks
        idxs = peaks_joint_idx if peaks_joint_idx else (peaks_rgb_idx if peaks_rgb_idx else peaks_sig_idx)
        if idxs:
            # keep only first N for visualization
            idxs = idxs[: min(len(idxs), args.annotate_peaks_max_n)]
            peak_xs_for_plot = [float(x_trim[i]) for i in idxs]

    # ---- Plot
    out_png = out_dir / "plots" / f"{run_name}__armband__{side}.png"
    plot_energy(
        out_png=out_png,
        t_grid=t_grid,
        rgb_y=rgb_plot,
        sig_y=sig_plot,
        mask_evt=mask_evt,
        ylabel=ylabel,
        title=(
            f"{run_name} | {side} | mode={args.armband_mode} | "
            f"zscore={'off' if args.no_zscore else 'on'} | "
            f"warmup={warmup_s:.1f}s taildrop={taildrop_s:.1f}s"
        ),
        rgb_run_start_epoch=rgb_run_start_epoch,
        add_top_axis_rgb=bool(args.top_axis_rgb_time),
        add_top_axis_datetime=bool(args.top_axis_show_datetime),
        annotate_peaks=bool(args.annotate_peaks_on_plot),
        peak_xs=peak_xs_for_plot,
        top_axis_tick_s=args.top_axis_tick_s
    )

    return {
        # identifiers
        "run_name": run_name,
        "side": side,
        "armband_mode": args.armband_mode,
        "rgb_dir": str(rgb_dir),
        "armband_csv": str(armband_csv),
        "plot_path": str(out_png),

        # counts
        "n_rgb_frames": int(len(rgb_frames)),
        "n_rgb_motion": int(len(t_rgb)),
        "n_armband_motion": int(len(t_sig)),
        "n_grid": int(len(t_grid)),

        # grid/trim info
        "grid_dt": float(args.grid_dt),
        "warmup_s": float(warmup_s),
        "taildrop_s": float(taildrop_s),
        "min_overlap_s": float(args.min_overlap_s),

        "t_start_intersection": float(t_start0),
        "t_end_intersection": float(t_end0),
        "t_start_used": float(t_start),
        "t_end_used": float(t_end),
        "overlap_raw_s": float(t_end0 - t_start0),
        "overlap_used_s": float(overlap_used),

        # plotting
        "no_zscore": bool(args.no_zscore),
        "plot_event_mask": bool(args.plot_event_mask),
        "event_z": float(args.event_z),
        "event_dilate_bins": int(args.event_dilate_bins),
        **evt_meta,

        # NEW: mapping back to full RGB timeline
        "rgb_run_start_epoch": float(rgb_run_start_epoch),
        "rgb_run_start_timestr": epoch_to_timestr(rgb_run_start_epoch),
        "trim_start_offset_from_rgb_s": float(t_start - rgb_run_start_epoch),
        "trim_end_offset_from_rgb_s": float(t_end - rgb_run_start_epoch),

        # armband feature params
        "armband_smooth_emg": float(args.armband_smooth_emg),
        "armband_smooth_imu": float(args.armband_smooth_imu),
        "armband_emg_hpf_like": bool(not args.armband_emg_no_hpf),
        "imu_use_energy": bool(not args.imu_no_energy),

        # NEW: peak parameters + outputs
        "peaks": {
            "enabled": True,
            "peaks_require_event_mask": bool(args.peaks_require_event_mask),
            "peak_min_z": float(args.peak_min_z),
            "peak_min_prom": float(args.peak_min_prom),
            "peak_min_dist_s": float(args.peak_min_dist_s),
            "peak_max_n": int(args.peak_max_n),
            "find_peaks_rgb": bool(args.find_peaks_rgb),
            "find_peaks_sig": bool(args.find_peaks_sig),
            "find_peaks_joint": bool(args.find_peaks_joint),
            "n_peaks_rgb": int(len(peaks_rgb)),
            "n_peaks_sig": int(len(peaks_sig)),
            "n_peaks_joint": int(len(peaks_joint)),
            "peaks_rgb": peaks_rgb,
            "peaks_sig": peaks_sig,
            "peaks_joint": peaks_joint,
        }
    }


# =========================
# Per-subject processing
# =========================
def get_subject_out_dir(subject_root: Path, out_root: Optional[Path], out_folder) -> Path:
    """
    Output layout:
      - if out_root is None: <subject_root>/<out_folder>
      - else: out_root/<subject_name>/<out_folder>
    """
    if out_root is None:
        return subject_root / out_folder
    return out_root / subject_root.name / out_folder


def process_one_subject(subject_root: Path, args) -> None:
    """
    Process all runs under one subject.
    For each run, find matching mindrove folder, then plot left/right csvs.
    """
    kinect_root = subject_root / "kinect"
    mindrove_root = subject_root / "mindrove"

    if not kinect_root.exists():
        print(f"[SKIP subject] Missing kinect: {kinect_root}")
        return
    if not mindrove_root.exists():
        print(f"[SKIP subject] Missing mindrove: {mindrove_root}")
        return

    out_root = Path(args.out_root) if args.out_root else None
    out_dir = get_subject_out_dir(subject_root, out_root, args.out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "run_plots.jsonl"
    out_json = out_dir / "run_plots.json"

    only_run_set = set(args.only_run) if args.only_run else None

    def should_run(rn: str) -> bool:
        return True if only_run_set is None else (rn in only_run_set)

    results = []

    run_dirs = sorted([p for p in kinect_root.glob("run_*") if p.is_dir()])
    if not run_dirs:
        print(f"[SKIP subject] No run_* under: {kinect_root}")
        return

    print(f"\n========== SUBJECT: {subject_root} ==========")
    print(f"Output -> {out_dir}")

    for run_dir in run_dirs:
        run_name = run_dir.name
        if not should_run(run_name):
            continue

        rgb_dir = run_dir / args.camera_id / args.rgb_subdir
        if not rgb_dir.exists():
            continue

        armband_folder = find_armband_folder_for_run(mindrove_root, run_name)
        if armband_folder is None:
            err = {
                "subject_root": str(subject_root),
                "run_name": run_name,
                "rgb_dir": str(rgb_dir),
                "error": "No matching mindrove folder",
            }
            append_jsonl(out_jsonl, err)
            print(f"[{run_name}] !! no mindrove match")
            continue

        left_csv = find_armband_csv(armband_folder, "left")
        right_csv = find_armband_csv(armband_folder, "right")

        print(f"[{run_name}] RGB={rgb_dir}")
        print(f"         mindrove={armband_folder.name}")
        print(f"         left={left_csv.name if left_csv else None} | right={right_csv.name if right_csv else None}")

        per_run_summary = {
            "subject_root": str(subject_root),
            "run_name": run_name,
            "rgb_dir": str(rgb_dir),
            "mindrove_folder": str(armband_folder),
            "left": None,
            "right": None,
        }

        # LEFT
        if left_csv is not None:
            try:
                res_l = process_one_run_plot(run_name, rgb_dir, left_csv, out_dir, "left", args)
                per_run_summary["left"] = res_l
                print(f"  -> LEFT  plot={Path(res_l['plot_path']).name} peaks(joint)={res_l['peaks']['n_peaks_joint']}")
            except Exception as e:
                per_run_summary["left"] = {"error": str(e), "armband_csv": str(left_csv)}
                print(f"  !! LEFT failed: {e}")

        # RIGHT
        if right_csv is not None:
            try:
                res_r = process_one_run_plot(run_name, rgb_dir, right_csv, out_dir, "right", args)
                per_run_summary["right"] = res_r
                print(f"  -> RIGHT plot={Path(res_r['plot_path']).name} peaks(joint)={res_r['peaks']['n_peaks_joint']}")
            except Exception as e:
                per_run_summary["right"] = {"error": str(e), "armband_csv": str(right_csv)}
                print(f"  !! RIGHT failed: {e}")

        append_jsonl(out_jsonl, per_run_summary)
        results.append(per_run_summary)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary JSON: {out_json}")
    print(f"Saved per-run JSONL: {out_jsonl}")
    print(f"Plots folder: {(out_dir / 'plots')}")


# =========================
# main
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--subject_roots", type=str, nargs="+", required=True,
        help=r"One or more subject roots, e.g. D:\...\_raw_data_structured\MR D:\...\_raw_data_structured\N"
    )

    ap.add_argument("--rgb_tz", type=str, default="Europe/London",
                help="Timezone used to interpret RGB filename timestamps (default: Europe/London). "
                     "Try UTC if your filenames are in UTC.")

    ap.add_argument("--camera_id", type=str, default="001431512812")
    ap.add_argument("--rgb_subdir", type=str, default="frames_rgb_blured",
                    help="frames_rgb_blured or frames_rgb")

    # RGB motion params
    ap.add_argument("--rgb_down_w", type=int, default=160)
    ap.add_argument("--rgb_down_h", type=int, default=90)
    ap.add_argument("--rgb_diff_threshold", type=int, default=0,
                    help="0 -> mean abs diff; >0 -> ratio of pixels exceeding threshold")
    ap.add_argument("--rgb_smooth_s", type=float, default=0.2)

    # armband mode + smoothing
    ap.add_argument("--armband_mode", type=str, default="imu",
                    choices=["fusion", "emg", "imu"])
    ap.add_argument("--armband_smooth_emg", type=float, default=0.08,
                    help="EMG smoothing window seconds (0.05~0.15 typical)")
    ap.add_argument("--armband_smooth_imu", type=float, default=0.08,
                    help="IMU smoothing window seconds (try 0.05~0.20 to preserve peaks)")
    ap.add_argument("--armband_emg_no_hpf", action="store_true",
                    help="Disable EMG diff (use raw envelope instead).")
    ap.add_argument("--imu_no_energy", action="store_true",
                    help="Use gyro magnitude instead of gyro energy (gyro^2).")

    # unified grid
    ap.add_argument("--grid_dt", type=float, default=0.02,
                    help="Unified time grid dt seconds (0.02 ~= 50Hz)")

    # trimming inside intersection
    ap.add_argument("--warmup_s", type=float, default=10.0,
                    help="Drop warm-up seconds AFTER intersection start")
    ap.add_argument("--taildrop_s", type=float, default=6.0,
                    help="Drop tail seconds BEFORE intersection end (e.g., 1~5). Default 0.")

    ap.add_argument("--min_overlap_s", type=float, default=8.0,
                    help="Minimum overlap seconds after warmup/taildrop")
    ap.add_argument("--min_grid_n", type=int, default=200,
                    help="Minimum grid points required to plot (avoid too-short windows)")

    # plotting / normalization
    ap.add_argument("--no_zscore", action="store_true",
                    help="Plot RAW energy instead of z-scored energy (zscore recommended for peak comparison).")

    # event mask overlay
    ap.add_argument("--event_z", type=float, default=0.8,
                    help="Event threshold on z-scored signals (both must exceed). Only affects mask overlay / peak gating.")
    ap.add_argument("--event_dilate_bins", type=int, default=2,
                    help="Dilate event mask by N bins on each side to connect events")
    ap.add_argument("--plot_event_mask", action="store_true",
                    help="Overlay event mask band on plots")

    # ---------- NEW: top axis mapping ----------
    ap.add_argument(
        "--top_axis_rgb_time", action="store_true",
        help="Add a TOP x-axis showing time since RGB run start (full RGB timeline). "
             "Bottom axis remains time since trimmed start (original behavior)."
    )
    ap.add_argument(
        "--top_axis_show_datetime", action="store_true",
        help="If --top_axis_rgb_time is set, also show approximate wall-clock datetime labels (sparse)."
    )

    ap.add_argument("--top_axis_tick_s", type=float, default=2.0,
                help="Tick spacing (seconds) for TOP x-axis.")

    # ---------- NEW: peak detection ----------
    ap.add_argument("--find_peaks_rgb", action="store_true",
                    help="Detect peaks on RGB z-score curve (inside trimmed window).")
    ap.add_argument("--find_peaks_sig", action="store_true",
                    help="Detect peaks on Armband z-score curve (inside trimmed window).")
    ap.add_argument("--find_peaks_joint", action="store_true",
                    help="Detect peaks on joint curve 0.5*(rgb_z + sig_z) (inside trimmed window).")

    # default behavior: find joint peaks (most useful), and also rgb peaks
    ap.set_defaults(find_peaks_rgb=True, find_peaks_sig=False, find_peaks_joint=True)

    ap.add_argument("--peaks_require_event_mask", action="store_true",
                    help="Only search peaks where BOTH rgb_z and sig_z exceed event_z (gated by event mask).")

    ap.add_argument("--peak_min_z", type=float, default=1.0,
                    help="Peak min height in z-score space (lower -> more peaks).")
    ap.add_argument("--peak_min_prom", type=float, default=0.35,
                    help="Peak min prominence (rough) in z-score space (lower -> more peaks).")
    ap.add_argument("--peak_min_dist_s", type=float, default=0.35,
                    help="Minimum time distance between peaks (seconds).")
    ap.add_argument("--peak_max_n", type=int, default=25,
                    help="Maximum number of peaks per curve to keep.")

    ap.add_argument("--annotate_peaks_on_plot", action="store_true",
                    help="Draw vertical lines at detected peaks (to help eyeballing).")
    ap.add_argument("--annotate_peaks_max_n", type=int, default=10,
                    help="Max number of peak lines to draw on plot (avoid clutter).")

    # output routing
    ap.add_argument(
        "--out_root", type=str, default=None,
        help="If set: outputs go to out_root/<subject_name>/<out_folder>. "
             "Otherwise: <subject_root>/<out_folder>"
    )
    ap.add_argument(
        "--out_folder", type=str, default="_sync_out_emg",
        help="If set: outputs go to out_root/<subject_name>/<out_folder>. "
             "Otherwise: <subject_root>/<out_folder>"
    )

    # run filter
    ap.add_argument(
        "--only_run", nargs="*", default=None,
        help="Only process specific runs, e.g. --only_run run_36 run_29-49"
    )

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
