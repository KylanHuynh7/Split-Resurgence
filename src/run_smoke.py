"""End-to-end smoke test on a single week of Statcast data.

Run from repo root:
    uv run python -m src.run_smoke

This pulls 7 days, runs every layer (pull -> features -> aggregate -> cluster),
and prints sanity-check stats. If this completes without error and the numbers
look reasonable, the data layer is wired up correctly.
"""

from __future__ import annotations

import sys
from textwrap import indent

import pandas as pd

from src.data.pull_statcast import pull_range
from src.data.build_features import build_pitch_features
from src.data.aggregate import aggregate_pitcher_season
from src.data.pitcher_metadata import resolve_roster
from src.analysis.splinker_subcluster import fit_velocity_subclusters, cluster_summary


SMOKE_WINDOW = ("2024-07-01", "2024-07-07")  # one week, all 30 teams active
SMOKE_LABEL = "smoke_2024w27"


def banner(s: str) -> None:
    print(f"\n=== {s} ===")


def main() -> int:
    banner("1. Pull Statcast (7 days)")
    raw = pull_range(*SMOKE_WINDOW, label=SMOKE_LABEL)
    print(f"pulled {len(raw):,} rows, {raw['pitcher'].nunique()} unique pitchers")
    print(f"pitch type counts (top 10):\n{indent(str(raw['pitch_type'].value_counts().head(10)), '  ')}")

    # Sanity: required columns present
    required = {"pitcher", "pitch_type", "release_speed", "pfx_z", "delta_run_exp"}
    missing = required - set(raw.columns)
    if missing:
        print(f"FAIL: missing required columns: {missing}")
        return 1

    banner("2. Build per-pitch features (FS + CH only)")
    feats = build_pitch_features(raw)
    print(f"feature rows: {len(feats):,}")
    print(f"by pitch type:\n{indent(str(feats['pitch_type'].value_counts()), '  ')}")
    print(f"velo_diff_vs_fastball summary (FS only):")
    fs_feats = feats[feats["pitch_type"] == "FS"]
    if not fs_feats.empty:
        print(indent(str(fs_feats["velo_diff_vs_fastball"].describe()), "  "))
    else:
        print("  (no FS rows in window)")

    banner("3. Aggregate to pitcher-season")
    # In a one-week smoke we lower the min_pitches threshold so we get nonzero rows.
    agg = aggregate_pitcher_season(feats, raw_full=raw, min_pitches=10)
    print(f"pitcher-season-pitchtype rows (min 10 pitches): {len(agg)}")
    print(f"head:\n{indent(str(agg.head().to_string(max_cols=8)), '  ')}")

    banner("4. Velocity-band sub-cluster (FS only)")
    clustered = fit_velocity_subclusters(agg)
    summary = cluster_summary(clustered)
    if summary.empty:
        print("  (too few FS pitcher-seasons in smoke window to cluster)")
    else:
        print(summary.to_string(index=False))

    banner("5. Resolve focal roster MLBAM IDs")
    try:
        roster = resolve_roster()
        # Check: did each focal pitcher resolve?
        unresolved = roster[roster["mlbam_id"].isna()]
        print(f"resolved {len(roster) - len(unresolved)}/{len(roster)} pitchers")
        if not unresolved.empty:
            print("unresolved:")
            print(indent(unresolved[["full_name"]].to_string(index=False), "  "))
        else:
            print("all focal pitchers resolved.")
        # Spot-check: how many of the focal pitchers appear in this week's data?
        focal_ids = set(roster["mlbam_id"].dropna().astype(int))
        present = focal_ids & set(raw["pitcher"].astype(int))
        present_names = roster[roster["mlbam_id"].isin(present)]["full_name"].tolist()
        print(f"focal pitchers active in smoke window: {present_names}")
    except Exception as e:
        print(f"roster resolution failed (network or pybaseball issue): {e}")
        return 2

    banner("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
