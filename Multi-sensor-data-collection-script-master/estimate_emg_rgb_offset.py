#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Multi-subject: Estimate time offset between RGB frames (30Hz) and Armband (MindRove EMG+IMU).

Key upgrades for reliability:
  1) Event gating: align only on time bins where BOTH RGB and armband motion are strong
  2) Coarse-to-fine scan: first coarse scan, then fine scan around the peak
  3) Peak confidence metrics:
       - best_corr
       - second_best_corr
       - peak_margin = best - second
       - peak_width_s: width around best where corr >= best - peak_drop
       - event_ratio: fraction of grid samples used as events
  4) More stable IMU feature:
       - default IMU motion uses smoothed gyro magnitude/energy (no diff)
  5) Matplotlib forced to Agg backend -> avoid Qt font warnings on Windows.
  6) NEW: tail drop (end trimming) inside the intersection window:
       - warmup_s: drop head seconds AFTER intersection start
       - taildrop_s: drop tail seconds BEFORE intersection end

Offset convention:
  delta > 0 means compare rgb(t) with armband(t + delta)
  => signal is shifted later in the comparison.

Inputs:
RGB:
  subject_root/kinect/run_*/<camera_id>/<rgb_subdir>/*.jpg
  timestamp in filename: YYYYMMDD_HHMMSS_micro

Armband:
  subject_root/mindrove/sub-*_run-xxx[_yyy]_emg/left|right/*_emg_imu.csv
  timestamp in CSV: board_ts (epoch seconds float)

Outputs per subject:
  <subject_out_dir>/
    run_offsets.jsonl   # per run (append immediately)
    run_offsets.json    # summary (written at end)
    plots/
      run_1__armband__left.png
      run_1__armband__right.png

Usage example:
  python estimate_rgb_emg_offset.py ^
    --subject_roots D:\...\_raw_data_structured\N D:\...\_raw_data_structured\MR ^
    --armband_mode imu ^
    --event_z 0.8 --min_event_s 2.0 ^
    --search_s 3.0 --coarse_step_s 0.02 --fine_step_s 0.005 ^
    --grid_dt 0.02 --warmup_s 5.0 --taildrop_s 3.0
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

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


def parse_ts_from_name(name: str) -> Optional[float]:
    """
    Parse timestamp substring 'YYYYMMDD_HHMMSS_micro' from a filename and return epoch seconds float.
    Returns None if parsing fails.
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
    filter out bogus timestamps (< 2010-01-01), return sorted (t, path).
    """
    items: List[Tuple[float, Path]] = []
    for p in folder.glob(f"*{suffix}"):
        t = parse_ts_from_name(p.name)
        if t is None or not np.isfinite(t):
            continue
        if t < 1262304000.0:  # 2010-01-01
            continue
        items.append((t, p))
    items.sort(key=lambda x: x[0])
    return items


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


def corr_at_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation."""
    if a.size == 0 or b.size == 0:
        return -1.0
    aa = a - a.mean()
    bb = b - b.mean()
    denom = (np.sqrt((aa * aa).sum()) * np.sqrt((bb * bb).sum()))
    if denom < 1e-12:
        return -1.0
    return float((aa * bb).sum() / denom)


def scan_correlations(
    tg: np.ndarray,
    rgb_g: np.ndarray,
    sig_g: np.ndarray,
    deltas: np.ndarray,
) -> np.ndarray:
    """
    Compute correlation for each delta in `deltas`.

    We assume tg is a uniform grid. We use np.interp on the same grid.
    """
    tmin, tmax = tg[0], tg[-1]
    corrs = np.full((len(deltas),), -1e9, dtype=np.float64)

    for i, d in enumerate(deltas):
        mask = (tg + d >= tmin) & (tg + d <= tmax)
        if mask.sum() < 10:
            continue
        a = rgb_g[mask]
        b = np.interp(tg[mask] + d, tg, sig_g)
        corrs[i] = corr_at_shift(a, b)

    return corrs


def estimate_offset_coarse_to_fine(
    tg: np.ndarray,
    rgb_g: np.ndarray,
    sig_g: np.ndarray,
    search_s: float,
    coarse_step_s: float,
    fine_step_s: float,
    fine_half_window_s: float = 0.15,
) -> Dict[str, float]:
    """
    Coarse-to-fine offset estimation:
      1) coarse scan in [-search_s, +search_s] with coarse_step_s
      2) pick best coarse delta
      3) fine scan around best coarse delta +/- fine_half_window_s with fine_step_s

    Returns dict:
      best_delta_s, best_corr,
      second_best_corr, peak_margin,
      peak_width_s (width where corr >= best - peak_drop),
      coarse_best_delta_s, coarse_best_corr
    """
    # coarse scan
    deltas_c = np.arange(-search_s, search_s + 1e-12, coarse_step_s, dtype=np.float64)
    corrs_c = scan_correlations(tg, rgb_g, sig_g, deltas_c)
    ic = int(np.argmax(corrs_c))
    coarse_best_delta = float(deltas_c[ic])
    coarse_best_corr = float(corrs_c[ic])

    # fine scan around coarse peak
    fine_lo = max(-search_s, coarse_best_delta - fine_half_window_s)
    fine_hi = min(+search_s, coarse_best_delta + fine_half_window_s)
    deltas_f = np.arange(fine_lo, fine_hi + 1e-12, fine_step_s, dtype=np.float64)
    corrs_f = scan_correlations(tg, rgb_g, sig_g, deltas_f)

    ib = int(np.argmax(corrs_f))
    best_delta = float(deltas_f[ib])
    best_corr = float(corrs_f[ib])

    # second best: remove a small neighborhood around best to avoid counting the same peak
    corrs_tmp = corrs_f.copy()
    nb = 2  # neighborhood bins on each side
    lo = max(0, ib - nb)
    hi = min(len(corrs_tmp), ib + nb + 1)
    corrs_tmp[lo:hi] = -1e9
    second_best_corr = float(np.max(corrs_tmp))
    peak_margin = float(best_corr - second_best_corr)

    # peak width: region where corr >= best_corr - peak_drop (a rough sharpness measure)
    peak_drop = 0.05
    thr = best_corr - peak_drop
    good = corrs_f >= thr
    if good.any() and good[ib]:
        left = ib
        while left - 1 >= 0 and good[left - 1]:
            left -= 1
        right = ib
        while right + 1 < len(good) and good[right + 1]:
            right += 1
        peak_width_s = float(deltas_f[right] - deltas_f[left])
    else:
        peak_width_s = 0.0

    return {
        "best_delta_s": best_delta,
        "best_corr": best_corr,
        "second_best_corr": second_best_corr,
        "peak_margin": peak_margin,
        "peak_width_s": peak_width_s,
        "coarse_best_delta_s": coarse_best_delta,
        "coarse_best_corr": coarse_best_corr,
    }


# =========================
# Event gating (critical for reliability)
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


def apply_mask_or_fallback(
    t_grid: np.ndarray,
    rgb_z: np.ndarray,
    sig_z: np.ndarray,
    mask_evt: np.ndarray,
    min_event_s: float,
    grid_dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    If event mask has enough samples (>= min_event_s), use it.
    Else fallback to full signal, but mark low_confidence.
    """
    need = int(np.ceil(min_event_s / grid_dt))
    used_mask = mask_evt
    low_conf = False

    if used_mask.sum() < need:
        used_mask = np.ones_like(mask_evt, dtype=bool)
        low_conf = True

    meta = {
        "event_count": float(mask_evt.sum()),
        "event_ratio": float(mask_evt.mean()),
        "mask_used_ratio": float(used_mask.mean()),
        "low_conf_event_fallback": bool(low_conf),
    }
    return t_grid[used_mask], rgb_z[used_mask], sig_z[used_mask], meta


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

    acc = np.stack(
        [col("acc_x")[ok][idx], col("acc_y")[ok][idx], col("acc_z")[ok][idx]],
        axis=1,
    )
    gyro = np.stack(
        [col("gyro_x")[ok][idx], col("gyro_y")[ok][idx], col("gyro_z")[ok][idx]],
        axis=1,
    )

    # drop duplicate timestamps (sometimes logging repeats same board_ts)
    uniq_mask = np.concatenate([[True], np.diff(t) > 0])
    t = t[uniq_mask]
    emg = emg[uniq_mask]
    acc = acc[uniq_mask]
    gyro = gyro[uniq_mask]

    return {"t": t, "emg": emg, "acc": acc, "gyro": gyro}


def compute_armband_motion(
    csv_path: Path,
    mode: str = "fusion",
    smooth_win_s_emg: float = 0.08,
    smooth_win_s_imu: float = 0.35,
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
      (No diff -> much more stable peaks than diff)

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
    # - EMG sample dt should be ~0.002s
    # - keep a broad safe range (0.0005..0.2) to tolerate occasional gaps
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

    # ---------- IMU motion (gyro-dominant, stable) ----------
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
        # zscore each then sum to balance scales
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
# Plotting
# =========================
def plot_alignment(
    out_png: Path,
    t_grid: np.ndarray,
    rgb_z: np.ndarray,
    sig_z: np.ndarray,
    best_delta: float,
    mask_evt: Optional[np.ndarray],
    title: str,
):
    """
    Plot z-scored RGB/armband signals on t_grid, plus shifted armband.
    If mask_evt provided, also show event mask as a faint band.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    x = t_grid - t_grid[0]
    plt.plot(x, rgb_z, label="RGB motion (z)")
    plt.plot(x, sig_z, label="Armband motion (z)")

    sig_shifted = np.interp(t_grid + best_delta, t_grid, sig_z, left=np.nan, right=np.nan)
    plt.plot(x, np.nan_to_num(sig_shifted, nan=0.0), label=f"Armband shifted Δt={best_delta:+.3f}s")

    if mask_evt is not None:
        # visualize event mask as a thin band near the bottom
        y0 = min(float(np.min(rgb_z)), float(np.min(sig_z))) - 0.5
        y1 = y0 + 0.25
        band = mask_evt.astype(np.float64) * (y1 - y0) + y0
        plt.plot(x, band, label="Event mask (band)")

    plt.xlabel("Time since start (s)")
    plt.ylabel("Z-score / shifted z-score")
    plt.title(title)
    plt.legend()
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
# Per-run processing
# =========================
def process_one_run_rgb_armband(
    run_name: str,
    rgb_dir: Path,
    armband_csv: Path,
    out_dir: Path,
    side: str,
    args,
) -> Dict:
    """
    Robust alignment for one run & one side.

    Steps:
      1) RGB motion (midpoint timestamps)
      2) Armband motion (midpoint timestamps)
      3) Compute intersection time window:
           [t_start0, t_end0]
         Then apply trimming:
           t_start = t_start0 + warmup_s
           t_end   = t_end0   - taildrop_s   <-- NEW
      4) Interpolate both motion signals to a unified time grid
      5) z-score both signals
      6) event gating -> use bins where both are strong; else fallback to full window
      7) coarse-to-fine scan on selected bins
      8) compute confidence metrics and save plot
    """
    # ---- RGB frames -> motion signal
    rgb_frames = list_files_sorted_by_ts(rgb_dir, ".jpg")
    if len(rgb_frames) < 2:
        raise RuntimeError(f"No/too few RGB frames in: {rgb_dir}")

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
    taildrop_s = float(max(0.0, args.taildrop_s))  # NEW
    t_start = t_start0 + warmup_s
    t_end = t_end0 - taildrop_s

    # basic sanity
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

    # ---- Unified time grid
    t_grid = np.arange(t_start, t_end, args.grid_dt, dtype=np.float64)
    if len(t_grid) < 200:
        raise RuntimeError(f"Too few grid samples: {len(t_grid)} (grid_dt={args.grid_dt})")

    # ---- Interpolate to grid (use raw motion first, then zscore)
    rgb_g = interp_to_grid(t_rgb, m_rgb, t_grid)
    sig_g = interp_to_grid(t_sig, m_sig, t_grid)
    rgb_z = zscore(rgb_g)
    sig_z = zscore(sig_g)

    # ---- Event gating: only keep bins where both are strong
    mask_evt = build_event_mask(
        rgb_z, sig_z,
        event_z=args.event_z,
        dilate_bins=args.event_dilate_bins,
    )

    tg_use, rgb_use, sig_use, evt_meta = apply_mask_or_fallback(
        t_grid, rgb_z, sig_z, mask_evt,
        min_event_s=args.min_event_s,
        grid_dt=args.grid_dt,
    )

    # ---- Coarse-to-fine scan on chosen samples
    scan_meta = estimate_offset_coarse_to_fine(
        tg_use,
        rgb_use,
        sig_use,
        search_s=args.search_s,
        coarse_step_s=args.coarse_step_s,
        fine_step_s=args.fine_step_s,
        fine_half_window_s=args.fine_half_window_s,
    )

    best_delta = scan_meta["best_delta_s"]
    best_corr = scan_meta["best_corr"]

    # ---- Confidence heuristics (tunable thresholds)
    confidence = {
        "pass_corr": bool(best_corr >= args.conf_min_corr),
        "pass_margin": bool(scan_meta["peak_margin"] >= args.conf_min_margin),
        "pass_event_ratio": bool(evt_meta["event_ratio"] >= args.conf_min_event_ratio),
        "pass_peak_width": bool(scan_meta["peak_width_s"] <= args.conf_max_peak_width_s),
    }
    confidence["is_reliable"] = bool(all(confidence.values()))

    # ---- Plot
    out_png = out_dir / "plots" / f"{run_name}__armband__{side}.png"
    plot_alignment(
        out_png=out_png,
        t_grid=t_grid,
        rgb_z=rgb_z,
        sig_z=sig_z,
        best_delta=best_delta,
        mask_evt=(mask_evt if args.plot_event_mask else None),
        title=(
            f"{run_name} | {side} | mode={args.armband_mode} | "
            f"best={best_delta:+.3f}s corr={best_corr:.3f} "
            f"margin={scan_meta['peak_margin']:.3f} evt={evt_meta['event_ratio']:.3f} "
            f"reliable={confidence['is_reliable']}"
        ),
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

        # params
        "search_s": float(args.search_s),
        "coarse_step_s": float(args.coarse_step_s),
        "fine_step_s": float(args.fine_step_s),
        "fine_half_window_s": float(args.fine_half_window_s),
        "grid_dt": float(args.grid_dt),

        "warmup_s": float(warmup_s),
        "taildrop_s": float(taildrop_s),  # NEW
        "min_overlap_s": float(args.min_overlap_s),

        "event_z": float(args.event_z),
        "min_event_s": float(args.min_event_s),
        "event_dilate_bins": int(args.event_dilate_bins),

        "armband_smooth_emg": float(args.armband_smooth_emg),
        "armband_smooth_imu": float(args.armband_smooth_imu),
        "armband_emg_hpf_like": bool(not args.armband_emg_no_hpf),
        "imu_use_energy": bool(not args.imu_no_energy),

        # overlap info (raw + used)
        "t_start_intersection": float(t_start0),
        "t_end_intersection": float(t_end0),          # NEW: record raw end
        "t_start_used": float(t_start),
        "t_end_used": float(t_end),
        "overlap_raw_s": float(t_end0 - t_start0),
        "overlap_used_s": float(overlap_used),

        # event gating meta
        **evt_meta,

        # scan results + confidence
        **scan_meta,
        "confidence": confidence,
    }


# =========================
# Per-subject processing
# =========================
def get_subject_out_dir(subject_root: Path, out_root: Optional[Path]) -> Path:
    """
    Output layout:
      - if out_root is None: <subject_root>/_sync_out_emg
      - else: out_root/<subject_name>/_sync_out_emg
    """
    if out_root is None:
        return subject_root / "_sync_out_emg"
    return out_root / subject_root.name / "_sync_out_emg"


def process_one_subject(subject_root: Path, args) -> None:
    """
    Process all runs under one subject.
    For each run, find matching mindrove folder, then process left/right csvs.
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
    out_dir = get_subject_out_dir(subject_root, out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "run_offsets.jsonl"
    out_json = out_dir / "run_offsets.json"

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
            # no RGB frames for that camera/subdir
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
                res_l = process_one_run_rgb_armband(run_name, rgb_dir, left_csv, out_dir, "left", args)
                per_run_summary["left"] = res_l
                print(
                    f"  -> LEFT  best={res_l['best_delta_s']:+.3f}s corr={res_l['best_corr']:.3f} "
                    f"margin={res_l['peak_margin']:.3f} evt={res_l['event_ratio']:.3f} "
                    f"reliable={res_l['confidence']['is_reliable']}"
                )
            except Exception as e:
                per_run_summary["left"] = {"error": str(e), "armband_csv": str(left_csv)}
                print(f"  !! LEFT failed: {e}")

        # RIGHT
        if right_csv is not None:
            try:
                res_r = process_one_run_rgb_armband(run_name, rgb_dir, right_csv, out_dir, "right", args)
                per_run_summary["right"] = res_r
                print(
                    f"  -> RIGHT best={res_r['best_delta_s']:+.3f}s corr={res_r['best_corr']:.3f} "
                    f"margin={res_r['peak_margin']:.3f} evt={res_r['event_ratio']:.3f} "
                    f"reliable={res_r['confidence']['is_reliable']}"
                )
            except Exception as e:
                per_run_summary["right"] = {"error": str(e), "armband_csv": str(right_csv)}
                print(f"  !! RIGHT failed: {e}")

        append_jsonl(out_jsonl, per_run_summary)
        results.append(per_run_summary)

    # final summary (nice for downstream analysis)
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
    ap.add_argument("--armband_mode", type=str, default="fusion",
                    choices=["fusion", "emg", "imu"])
    ap.add_argument("--armband_smooth_emg", type=float, default=0.08,
                    help="EMG smoothing window seconds (0.05~0.15 typical)")
    ap.add_argument("--armband_smooth_imu", type=float, default=0.15,
                    help="IMU smoothing window seconds (0.25~0.6 typical)")
    ap.add_argument("--armband_emg_no_hpf", action="store_true",
                    help="Disable EMG diff (use raw envelope instead).")
    ap.add_argument("--imu_no_energy", action="store_true",
                    help="Use gyro magnitude instead of gyro energy (gyro^2).")

    # time/grid/scan
    ap.add_argument("--search_s", type=float, default=2.0, help="Search range +/- seconds")
    ap.add_argument("--coarse_step_s", type=float, default=0.02, help="Coarse scan step (seconds)")
    ap.add_argument("--fine_step_s", type=float, default=0.005, help="Fine scan step (seconds)")
    ap.add_argument("--fine_half_window_s", type=float, default=0.15,
                    help="Fine scan half window around coarse best")

    ap.add_argument("--grid_dt", type=float, default=0.02,
                    help="Unified time grid dt seconds (0.02 ~= 50Hz)")

    # trimming inside intersection
    ap.add_argument("--warmup_s", type=float, default=10.0,
                    help="Drop warm-up seconds AFTER intersection start")
    ap.add_argument("--taildrop_s", type=float, default=3.0,
                    help="Drop tail seconds BEFORE intersection end (e.g., 1~5). Default 0.")  # NEW

    ap.add_argument("--min_overlap_s", type=float, default=8.0,
                    help="Minimum overlap seconds after warmup/taildrop")

    # event gating controls
    ap.add_argument("--event_z", type=float, default=0.3,
                    help="Event threshold on z-scored signals (both must exceed)")
    ap.add_argument("--min_event_s", type=float, default=0.3,
                    help="Minimum event seconds; else fallback to full window")
    ap.add_argument("--event_dilate_bins", type=int, default=2,
                    help="Dilate event mask by N bins on each side to connect events")
    ap.add_argument("--plot_event_mask", action="store_true",
                    help="Overlay event mask band on plots")

    # confidence thresholds (heuristics)
    ap.add_argument("--conf_min_corr", type=float, default=0.35,
                    help="Minimum best_corr to call alignment reliable")
    ap.add_argument("--conf_min_margin", type=float, default=0.05,
                    help="Minimum (best - second_best) to call alignment reliable")
    ap.add_argument("--conf_min_event_ratio", type=float, default=0.01,
                    help="Minimum fraction of event bins to call reliable (avoid tiny events)")
    ap.add_argument("--conf_max_peak_width_s", type=float, default=0.20,
                    help="Maximum peak width (seconds) to call reliable (avoid very flat peaks)")

    # output routing
    ap.add_argument(
        "--out_root", type=str, default=None,
        help="If set: outputs go to out_root/<subject_name>/_sync_out_emg. "
             "Otherwise: <subject_root>/_sync_out_emg"
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
