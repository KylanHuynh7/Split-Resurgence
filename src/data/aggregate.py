"""Aggregate per-pitch features to pitcher-season level.

This is the unit of analysis for the mixed-effects regression. Each row is one
(pitcher, season, pitch_type) combination with summary stats:

  - run_value_per_100: 100 * mean(delta_run_exp). Statcast convention is run
    value FROM the pitcher's perspective; lower is better. We negate so that
    higher = better pitch (pitcher prevented more runs).
  - n_pitches
  - mean velocity, IVB, horizontal break, release point, extension
  - mean velocity differential and IVB differential vs primary fastball
  - whiff_rate (swings_and_misses / swings)
  - usage_share: this pitch as fraction of all pitches the pitcher threw

Pitcher-season-pitch-type rows below `min_pitches` are dropped. Default 100
matches the project plan.
"""

from __future__ import annotations

import pandas as pd

# Statcast `description` values that count as a swing.
SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "missed_bunt", "foul_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}


def aggregate_pitcher_season(
    pitch_features: pd.DataFrame,
    raw_full: pd.DataFrame,
    min_pitches: int = 100,
) -> pd.DataFrame:
    """Aggregate to (pitcher, season, pitch_type).

    `pitch_features` is the output of `build_features.build_pitch_features`
    (already filtered to FS + CH).
    `raw_full` is the unfiltered pitch set, needed to compute the pitcher's
    total pitch count for usage_share.
    """
    df = pitch_features.copy()
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)

    grouped = df.groupby(["pitcher", "game_year", "pitch_type"]).agg(
        n_pitches=("delta_run_exp", "size"),
        run_value_sum=("delta_run_exp", "sum"),
        run_value_mean=("delta_run_exp", "mean"),
        velo_mean=("release_speed", "mean"),
        velo_std=("release_speed", "std"),
        pfx_x_mean=("pfx_x", "mean"),
        pfx_z_mean=("pfx_z", "mean"),
        release_x_mean=("release_pos_x", "mean"),
        release_z_mean=("release_pos_z", "mean"),
        extension_mean=("release_extension", "mean"),
        spin_mean=("release_spin_rate", "mean"),
        velo_diff_mean=("velo_diff_vs_fastball", "mean"),
        ivb_diff_mean=("ivb_diff_vs_fastball", "mean"),
        plate_z_mean=("plate_z", "mean"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        # Carry through the fastball baseline for joins downstream.
        fastball_type=("fastball_type", "first"),
        fastball_velo_mean=("fastball_velo_mean", "first"),
        fastball_pfx_z_mean=("fastball_pfx_z_mean", "first"),
    ).reset_index()

    # Negate so higher = better pitch (pitcher's perspective).
    grouped["run_value_per_100"] = -100 * grouped["run_value_mean"]
    grouped["whiff_rate"] = grouped["whiffs"] / grouped["swings"].replace(0, pd.NA)

    # Total pitches per pitcher-season (denominator for usage share).
    totals = (
        raw_full.groupby(["pitcher", "game_year"])
        .size()
        .reset_index(name="total_pitches_pitcher_season")
    )
    grouped = grouped.merge(totals, on=["pitcher", "game_year"], how="left")
    grouped["usage_share"] = (
        grouped["n_pitches"] / grouped["total_pitches_pitcher_season"]
    )

    return grouped[grouped["n_pitches"] >= min_pitches].reset_index(drop=True)
