#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
count_action_hierarchy_segments_with_joint_and_plots.py

✅ What this script does
------------------------------------------------------------
You provide 4 dataset roots. Each root contains many *Tier-3 action folders*:

  root/
    <tier3_action_dir>/                 e.g., insert_short_wire_into_lug
      <segment_dir>/                    e.g., run_3_clip_000011_normal_elbow
        ... (data inside)

We treat "number of segment folders" as the number of action segments.

We aggregate segment counts across ALL provided roots for 3 hierarchy tiers:

  Tier3: original action folder name
  Tier2: (action + object) mapped from Tier3 by token-subsequence best match
  Tier1: verb-only extracted from Tier3 (supports multiword verb: pull_out)

For EACH tier-key, we also compute a 3x2 joint table (lighting x armband position):
  lighting ∈ {normal, right, left}
  pos      ∈ {elbow, mid}

Segment folder naming rule (HARD rule, per your requirement):
  run_3_clip_000011_normal_elbow
  - second last token = lighting
  - last token        = pos

Outputs:
  out_dir/
    tier1_counts.csv
    tier2_counts.csv
    tier3_counts.csv
    unmapped_tier2_actions.txt
    plots/
      tier1_bar.png
      tier2_bar.png
      tier3_bar_topN.png

Plot enhancement (NEW):
  - On each bar, annotate the exact count (integer) on top of the bar.

------------------------------------------------------------
Usage example (Windows):
  python count_action_hierarchy_segments_with_joint_and_plots.py ^
    --roots D:\data\root1 D:\data\root2 D:\data\root3 D:\data\root4 ^
    --out_dir D:\data\stats_out ^
    --top_n_t3 60

"""

import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # Safe for servers / headless execution
import matplotlib.pyplot as plt


# ------------------------- Your hierarchy definitions -------------------------

# Tier1 verbs (normalized to underscores)
TIER1_VERBS = [
    "adjust", "cap", "close", "cut", "insert", "label", "measure",
    "move", "open", "position", "press", "pull_out", "put",
    "remove", "take", "tear", "wrap"
]

# Tier2: action + object (normalized to underscores)
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

# Allowed meta values (fixed)
LIGHTS = ("normal", "right", "left")
POSITIONS = ("elbow", "mid")


# ------------------------- Small utility functions -------------------------

def norm_name(s: str) -> str:
    """
    Normalize a name:
      - lower-case
      - spaces -> underscores
      - collapse multiple underscores
    """
    s = s.strip().lower().replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

def tokenize(name: str) -> List[str]:
    """Split normalized string into underscore tokens."""
    return [t for t in norm_name(name).split("_") if t]

def is_subsequence(needle: List[str], hay: List[str]) -> bool:
    """
    Return True if `needle` tokens appear in `hay` in order (not necessarily contiguous).
    Example:
      needle: ['insert','wire','into','lug']
      hay:    ['insert','short','wire','into','lug']  -> True
    """
    if not needle:
        return True
    j = 0
    for tok in hay:
        if tok == needle[j]:
            j += 1
            if j == len(needle):
                return True
    return False


# ------------------------- Tier mapping rules -------------------------

def get_tier1_verb(tier3_action: str) -> str:
    """
    Extract Tier1 verb from Tier3 folder name.
    Special case: multiword verb "pull_out" -> tokens start with ['pull','out',...]
    """
    t3 = tokenize(tier3_action)
    if len(t3) >= 2 and f"{t3[0]}_{t3[1]}" == "pull_out":
        return "pull_out"
    if t3:
        return t3[0]
    return "UNKNOWN"

def best_tier2_match(tier3_action: str, tier2_list: List[str]) -> Optional[str]:
    """
    Map Tier3 -> Tier2 by "best token-subsequence match" with verb consistency.

    We:
      1) Determine Tier3's verb (including pull_out)
      2) Among Tier2 candidates with the same verb:
         choose the one whose tokens are a subsequence of Tier3 tokens,
         and that has the maximum token length (most specific match).

    Returns:
      - matched Tier2 string (normalized) or None if no match
    """
    t3 = tokenize(tier3_action)

    # detect Tier3 verb
    if len(t3) >= 2 and f"{t3[0]}_{t3[1]}" == "pull_out":
        verb = "pull_out"
    else:
        verb = t3[0] if t3 else None

    candidates: List[Tuple[int, str]] = []
    for t2 in tier2_list:
        t2n = tokenize(t2)
        if not t2n:
            continue

        # detect Tier2 verb
        t2_verb = "pull_out" if len(t2n) >= 2 and f"{t2n[0]}_{t2n[1]}" == "pull_out" else t2n[0]

        # require verb consistency when possible
        if verb is not None and t2_verb != verb:
            continue

        if is_subsequence(t2n, t3):
            candidates.append((len(t2n), t2))

    if not candidates:
        return None

    # prefer longest match; stable tie-break by name
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


# ------------------------- Segment meta parsing -------------------------

def parse_segment_meta_last2(seg_name: str) -> Tuple[str, str]:
    """
    NEW HARD RULE (per your dataset):
      segment folder name like: run_3_clip_000011_normal_elbow

    We ONLY use the last two underscore tokens:
      - second last token = lighting (normal/right/left)
      - last token        = position (elbow/mid)

    Any invalid value -> 'unknown'
    """
    toks = tokenize(seg_name)
    if len(toks) < 2:
        return "unknown", "unknown"

    lighting = toks[-2]
    pos = toks[-1]

    if lighting not in LIGHTS:
        lighting = "unknown"
    if pos not in POSITIONS:
        pos = "unknown"

    return lighting, pos


# ------------------------- Data structure for counts -------------------------

JOINT_KEYS = (
    "normal_elbow", "normal_mid",
    "right_elbow", "right_mid",
    "left_elbow", "left_mid",
)
UNKNOWN_KEY = "unknown_unknown"


@dataclass
class Counts:
    """
    Stores:
      - total number of segments
      - 3x2 joint table counts
      - unknown bucket (if parsing fails)
    """
    total: int = 0
    normal_elbow: int = 0
    normal_mid: int = 0
    right_elbow: int = 0
    right_mid: int = 0
    left_elbow: int = 0
    left_mid: int = 0
    unknown_unknown: int = 0

    def add(self, lighting: str, pos: str, n: int = 1) -> None:
        """Add n segments into the appropriate joint-bin and total."""
        self.total += n
        if lighting in LIGHTS and pos in POSITIONS:
            k = f"{lighting}_{pos}"
            if hasattr(self, k):
                setattr(self, k, getattr(self, k) + n)
            else:
                self.unknown_unknown += n
        else:
            self.unknown_unknown += n


# ------------------------- CSV writer -------------------------

def write_csv(path: Path, key_name: str, data: Dict[str, Counts]) -> None:
    """Write counts dict to CSV sorted by descending total count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key_name, "total", *JOINT_KEYS, UNKNOWN_KEY]

    rows = []
    for k, c in sorted(data.items(), key=lambda kv: (-kv[1].total, kv[0])):
        d = asdict(c)
        d[key_name] = k
        rows.append(d)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({fn: r.get(fn, "") for fn in fieldnames})


# ------------------------- Plotting (bar + annotations) -------------------------

def plot_bar_counts(
    out_png: Path,
    title: str,
    data: Dict[str, Counts],
    top_n: Optional[int] = None,
) -> None:
    """
    Draw a bar chart:
      x = action key
      y = total segments

    NEW:
      - Annotate each bar with its integer value on top.
    """
    items = sorted(data.items(), key=lambda kv: kv[1].total, reverse=True)
    if top_n is not None:
        items = items[:top_n]

    labels = [k for k, _ in items]
    values = [c.total for _, c in items]

    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic figure size to reduce overlap of x labels
    n = max(1, len(labels))
    fig_w = min(26, max(10, 0.38 * n))
    fig_h = 6

    plt.figure(figsize=(fig_w, fig_h))
    bars = plt.bar(range(len(values)), values)

    plt.title(title)
    plt.ylabel("Number of segments (folders)")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")

    # --- Annotate each bar with its value ---
    # Put text slightly above the bar top.
    # For very small values, still works.
    y_max = max(values) if values else 0
    y_pad = max(1, int(round(y_max * 0.01)))  # small padding based on scale

    for rect, val in zip(bars, values):
        x = rect.get_x() + rect.get_width() / 2.0
        y = rect.get_height()
        plt.text(
            x, y + y_pad,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0
        )

    # Expand y-limit a bit so text doesn't get clipped
    if y_max > 0:
        plt.ylim(0, y_max + 5 * y_pad)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ------------------------- Main scan -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Dataset root folders (provide 4). Example: --roots D:/A D:/B D:/C D:/D",
    )
    ap.add_argument("--out_dir", required=True, help="Output directory for CSVs and plots")
    ap.add_argument(
        "--top_n_t3",
        type=int,
        default=70,
        help="Top-N Tier3 actions to plot (Tier3 can be large). Default: 40",
    )
    ap.add_argument(
        "--top_n_t2",
        type=int,
        default=None,
        help="Top-N Tier2 actions to plot. Default: plot all",
    )
    ap.add_argument(
        "--top_n_t1",
        type=int,
        default=None,
        help="Top-N Tier1 verbs to plot. Default: plot all",
    )
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    out_dir = Path(args.out_dir)

    # three tier dictionaries: key -> Counts
    tier1_counts: Dict[str, Counts] = defaultdict(Counts)
    tier2_counts: Dict[str, Counts] = defaultdict(Counts)
    tier3_counts: Dict[str, Counts] = defaultdict(Counts)

    # for debugging mapping failures
    unmapped_t2_actions = set()

    # normalize lists
    tier2_norm = [norm_name(x) for x in TIER2_ACTIONS]
    tier1_norm = set(norm_name(x) for x in TIER1_VERBS)  # not strictly required, but useful

    # scan all roots
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Root not found: {root}")

        # action folders are direct children of root
        for action_dir in root.iterdir():
            if not action_dir.is_dir():
                continue

            tier3 = norm_name(action_dir.name)

            # segment folders are direct children of action_dir
            for seg_dir in action_dir.iterdir():
                if not seg_dir.is_dir():
                    continue

                lighting, pos = parse_segment_meta_last2(seg_dir.name)

                # Tier3
                tier3_counts[tier3].add(lighting, pos, 1)

                # Tier1
                t1 = norm_name(get_tier1_verb(tier3))
                # Keep unknown verbs too (to surface surprises)
                tier1_counts[t1].add(lighting, pos, 1)

                # Tier2
                t2 = best_tier2_match(tier3, tier2_norm)
                if t2 is None:
                    unmapped_t2_actions.add(tier3)
                    t2 = "UNMAPPED_T2"
                tier2_counts[t2].add(lighting, pos, 1)

    # write CSV outputs
    write_csv(out_dir / "tier1_counts.csv", "tier1", tier1_counts)
    write_csv(out_dir / "tier2_counts.csv", "tier2", tier2_counts)
    write_csv(out_dir / "tier3_counts.csv", "tier3", tier3_counts)

    # mapping debug output
    (out_dir / "unmapped_tier2_actions.txt").write_text(
        "\n".join(sorted(unmapped_t2_actions)),
        encoding="utf-8",
    )

    # plots
    plots_dir = out_dir / "plots"
    plot_bar_counts(
        plots_dir / "tier1_bar.png",
        "Tier1: segment counts per verb",
        tier1_counts,
        top_n=args.top_n_t1,
    )
    plot_bar_counts(
        plots_dir / "tier2_bar.png",
        "Tier2: segment counts per action+object",
        tier2_counts,
        top_n=args.top_n_t2,
    )
    plot_bar_counts(
        plots_dir / f"tier3_bar_top{args.top_n_t3}.png",
        f"Tier3: segment counts per action (Top {args.top_n_t3})",
        tier3_counts,
        top_n=args.top_n_t3,
    )

    # console summary
    print("Done.")
    print(f"- Tier1 CSV: {out_dir / 'tier1_counts.csv'}")
    print(f"- Tier2 CSV: {out_dir / 'tier2_counts.csv'}")
    print(f"- Tier3 CSV: {out_dir / 'tier3_counts.csv'}")
    print(f"- Unmapped Tier2 list: {out_dir / 'unmapped_tier2_actions.txt'}")
    print(f"- Plots dir: {plots_dir}")


if __name__ == "__main__":
    main()
