#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
h5_export_frames_and_points_csv.py

Tasks:
1) Export /frames metadata to ONE CSV file.
2) Optionally export /points into per-frame CSV files, naming each file by either
   frames/t_recv_ns or frames/t_msg_ns (user selectable), with London timezone naming.

Per-frame CSV naming (same behavior as your h5_to_frame_csv_london.py):
  YYYYMMDD_HHMMSS_microseconds_cloud.csv   (Europe/London)
Example:
  20251018_134438_780640_cloud.csv

Per-frame CSV columns:
  x,y,z,intensity

Usage examples:
  # Export frames.csv only
  python3 h5_export_frames_and_points_csv.py \
    --h5 /path/to/run_1_20260216_120000.h5 \
    --frames_csv /path/to/out/frames.csv

  # Export frames.csv + per-frame point CSVs named by t_recv_ns
  python3 h5_export_frames_and_points_csv.py \
    --h5 /path/to/run_1_20260216_120000.h5 \
    --frames_csv /path/to/out/frames.csv \
    --export_points \
    --points_out_dir /path/to/out/points_csv \
    --time_key t_recv_ns

  # Export frames.csv + per-frame point CSVs named by t_msg_ns, no header, higher precision
  python3 h5_export_frames_and_points_csv.py \
    --h5 /path/to/run_1_20260216_120000.h5 \
    --frames_csv /path/to/out/frames.csv \
    --export_points \
    --points_out_dir /path/to/out/points_csv \
    --time_key t_msg_ns \
    --no_header \
    --float_fmt %.9f
"""

import os
import csv
import argparse
from zoneinfo import ZoneInfo
from datetime import datetime

import numpy as np
import h5py


def ts_ns_to_london_name(t_ns: int, suffix: str = "_cloud.csv") -> str:
    """Convert epoch nanoseconds -> Europe/London local time string: YYYYMMDD_HHMMSS_%f + suffix."""
    dt_utc = datetime.fromtimestamp(t_ns / 1e9, tz=ZoneInfo("UTC"))
    dt_lon = dt_utc.astimezone(ZoneInfo("Europe/London"))
    return dt_lon.strftime("%Y%m%d_%H%M%S_%f") + suffix


def export_frames_to_csv(h5_path: str, frames_csv_path: str) -> int:
    """
    Export /frames datasets into a single CSV.
    Returns: n_frames
    """
    os.makedirs(os.path.dirname(os.path.abspath(frames_csv_path)), exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        if "frames" not in f:
            raise RuntimeError("Invalid HDF5: missing /frames")
        g = f["frames"]

        required = ["t_recv_ns", "t_recv_s", "t_msg_ns", "frame_start", "frame_count", "frame_id"]
        missing = [k for k in required if k not in g]
        if missing:
            raise RuntimeError(f"Invalid HDF5: missing /frames/{missing}")

        d_t_recv_ns = g["t_recv_ns"]
        d_t_recv_s = g["t_recv_s"]
        d_t_msg_ns = g["t_msg_ns"]
        d_start = g["frame_start"]
        d_count = g["frame_count"]
        d_fid = g["frame_id"]

        n_frames = int(d_t_recv_ns.shape[0])
        # Basic consistency checks
        for d in [d_t_recv_s, d_t_msg_ns, d_start, d_count, d_fid]:
            if int(d.shape[0]) != n_frames:
                raise RuntimeError("Invalid HDF5: /frames datasets have inconsistent lengths")

        with open(frames_csv_path, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["idx", "t_recv_ns", "t_recv_s", "t_msg_ns", "frame_start", "frame_count", "frame_id"])

            # Stream row-by-row (no need to load all into RAM)
            for i in range(n_frames):
                # frame_id can be bytes or str depending on h5py version/dtype; normalize to str
                fid = d_fid[i]
                if isinstance(fid, (bytes, np.bytes_)):
                    fid = fid.decode("utf-8", errors="replace")
                else:
                    fid = str(fid)

                w.writerow([
                    i,
                    int(d_t_recv_ns[i]),
                    float(d_t_recv_s[i]),
                    int(d_t_msg_ns[i]),
                    int(d_start[i]),
                    int(d_count[i]),
                    fid,
                ])

    return n_frames


def export_points_to_frame_csvs(
    h5_path: str,
    out_dir: str,
    time_key: str = "t_recv_ns",
    float_fmt: str = "%.6f",
    no_header: bool = False,
    progress_every: int = 50,
) -> int:
    """
    Export /points into per-frame CSVs using /frames/frame_start & /frames/frame_count.
    Returns: n_frames exported
    """
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        if "frames" not in f or "points" not in f:
            raise RuntimeError("Invalid HDF5: missing /frames or /points")

        g = f["frames"]
        for k in (time_key, "frame_start", "frame_count"):
            if k not in g:
                raise RuntimeError(f"Invalid HDF5: missing /frames/{k}")

        d_time = g[time_key]          # (n_frames,)
        d_start = g["frame_start"]    # (n_frames,)
        d_count = g["frame_count"]    # (n_frames,)
        d_points = f["points"]        # (N_total,4)

        n_frames = int(d_time.shape[0])
        if n_frames == 0:
            print("No frames found. Nothing to export.")
            return 0

        if d_points.ndim != 2 or int(d_points.shape[1]) != 4:
            raise RuntimeError(f"Invalid /points shape: {d_points.shape}, expected (N,4)")

        header = "" if no_header else "x,y,z,intensity\n"

        for i in range(n_frames):
            t_ns = int(d_time[i])
            start = int(d_start[i])
            cnt = int(d_count[i])

            # slice points for this frame (reads only this chunk)
            pts = d_points[start:start + cnt, :]  # (cnt,4)

            # build filename (London timezone)
            fname = ts_ns_to_london_name(t_ns, suffix="_cloud.csv")
            out_path = os.path.join(out_dir, fname)

            np.savetxt(
                out_path,
                pts,
                delimiter=",",
                fmt=float_fmt,
                header=header.strip("\n") if header else "",
                comments="",  # prevent '#' prefix
            )

            if (i + 1) % max(1, progress_every) == 0 or (i + 1) == n_frames:
                print(f"Exported points {i+1}/{n_frames}: {out_path}")

    return n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Input HDF5 file")
    ap.add_argument("--frames_csv", required=True, help="Output CSV path for frames metadata")

    # points export options
    ap.add_argument(
        "--export_points",
        action="store_true",
        help="If set, also export per-frame point CSVs",
    )
    ap.add_argument(
        "--points_out_dir",
        default=None,
        help="Output directory for per-frame point CSVs (required if --export_points)",
    )
    ap.add_argument(
        "--time_key",
        default="t_recv_ns",
        choices=["t_recv_ns", "t_msg_ns"],
        help="Which per-frame timestamp to use for naming files (default: t_recv_ns)",
    )
    ap.add_argument(
        "--float_fmt",
        default="%.6f",
        help="Float format in point CSVs (default: %%.6f). Example: %%.9f",
    )
    ap.add_argument(
        "--no_header",
        action="store_true",
        help="If set, do not write point CSV header line",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Print progress every N frames when exporting point CSVs (default: 50)",
    )
    args = ap.parse_args()

    # 1) frames -> CSV
    n_frames = export_frames_to_csv(args.h5, args.frames_csv)
    print(f"Frames CSV written: {args.frames_csv} (n_frames={n_frames})")

    # 2) optional: points -> per-frame CSVs
    if args.export_points:
        if not args.points_out_dir:
            raise SystemExit("ERROR: --points_out_dir is required when --export_points is set")

        n2 = export_points_to_frame_csvs(
            h5_path=args.h5,
            out_dir=args.points_out_dir,
            time_key=args.time_key,
            float_fmt=args.float_fmt,
            no_header=args.no_header,
            progress_every=args.progress_every,
        )
        print(f"Per-frame point CSVs exported: {args.points_out_dir} (n_frames={n2})")


if __name__ == "__main__":
    main()

