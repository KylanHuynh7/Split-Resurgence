"""Per-pitch feature table for splitter and changeup analysis.

The output of `build_pitch_features()` is the working dataset for everything
downstream: the velocity-band sub-cluster, the pitcher-season aggregator, the
physical-features baseline model, and the residual analysis.

Filtering decisions made here:
  - Keep pitch_type in {FS, CH}. Changeups are the comparison pitch from the
    revised plan.
  - Keep `delta_run_exp` non-null (this is the dependent variable). Statcast
    sets it for every pitch but rare missing values do occur.
  - Do NOT filter by pitcher here. The 100-FS minimum is applied at the
    aggregation step so the feature table can also be used for league-wide
    distributional plots.

Engineered features:
  - velo_diff_vs_fastball: pitcher-season-level velocity gap from primary FB
  - ivb_diff_vs_fastball:  same idea for induced vertical break
  - is_two_strike, is_three_ball: count-state indicators
  - prev_pitch_type:       lag-1 pitch type within the same plate appearance
                           (for sequencing analysis later — not used in v1
                           model unless we get there)
"""

from __future__ import annotations

import pandas as pd

from src.data.pitcher_metadata import build_fastball_baseline

# Statcast columns we actually use. Pulling everything is fine but explicitly
# listing makes downstream contracts clear.
KEEP_COLS = [
    # identity
    "pitcher", "batter", "game_year", "game_pk", "game_date", "at_bat_number",
    "pitch_number", "pitch_type",
    # physical
    "release_speed", "release_pos_x", "release_pos_z", "release_extension",
    "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "plate_x", "plate_z", "spin_axis", "release_spin_rate",
    # context
    "stand", "p_throws", "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    # outcome
    "description", "events", "delta_run_exp",
]

PITCH_TYPES_OF_INTEREST = ("FS", "CH")


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        # Statcast schema is stable but log loudly if it shifts.
        print(f"[build_features] WARNING: missing Statcast cols: {missing}")
    return df[available].copy()


def build_pitch_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Produce the per-pitch feature table for FS + CH pitches.

    `raw` is one or more concatenated season pulls from `pull_statcast.pull_season`.
    """
    df = _select_columns(raw)

    # 1. Fastball baseline per pitcher-season (computed from FF/SI/FC across the
    #    FULL pitch set, before we filter to FS/CH).
    fb = build_fastball_baseline(df)

    # 2. Compute lag-1 pitch type within each at-bat from the FULL pitch set,
    #    so the prior pitch reflects the actual sequence (e.g., fastball
    #    followed by splitter), not "prior FS/CH only".
    full_sorted = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    full_sorted["prev_pitch_type"] = (
        full_sorted.groupby(["game_pk", "at_bat_number"])["pitch_type"].shift(1)
    )
    prev_lookup = full_sorted[
        ["game_pk", "at_bat_number", "pitch_number", "prev_pitch_type"]
    ]

    # 3. Filter to splitters + changeups.
    df = df[df["pitch_type"].isin(PITCH_TYPES_OF_INTEREST)].copy()

    # 4. Drop pitches without a run-value tag — these are unusable as the DV.
    df = df[df["delta_run_exp"].notna()].copy()

    # 5. Join fastball baseline + prev-pitch lag.
    df = df.merge(fb, on=["pitcher", "game_year"], how="left")
    df = df.merge(
        prev_lookup, on=["game_pk", "at_bat_number", "pitch_number"], how="left"
    )

    # 6. Engineered features.
    df["velo_diff_vs_fastball"] = df["release_speed"] - df["fastball_velo_mean"]
    df["ivb_diff_vs_fastball"] = df["pfx_z"] - df["fastball_pfx_z_mean"]
    df["is_two_strike"] = (df["strikes"] == 2).astype(int)
    df["is_three_ball"] = (df["balls"] == 3).astype(int)
    df["same_handed"] = (df["stand"] == df["p_throws"]).astype(int)

    return df.reset_index(drop=True)
