"""Step 8: aggregate per-pitch residuals to pitcher-season-pitchtype.

`model_resistance = -100 * mean(residual)` — sign-flipped so higher is better
for the pitcher (mirrors the `run_value_per_100` convention in aggregate.py).
A pitcher with model_resistance > 0 is OUTPERFORMING what the physical model
predicts; one with model_resistance < 0 is UNDERPERFORMING it.

Joins three things onto each row:
  - Focal-roster metadata (group, npb_years, mlb_debut) for the 10 named pitchers
  - Chadwick-register MLB debut year for everyone else, so `mlb_seasons_in`
    is right for non-focal pitchers too (the regression needs it as a control)
  - A `is_novelty_year` flag = (game_year == mlb_debut)

Run:
    uv run python -m src.models.residual_analysis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pybaseball import playerid_reverse_lookup

from src.data.pitcher_metadata import resolve_roster

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PREDICTIONS_PATH = PROCESSED_DIR / "pitches_with_predictions.parquet"
RESIDUALS_PATH = PROCESSED_DIR / "pitcher_season_residuals.parquet"
DEBUTS_CACHE = PROCESSED_DIR / "pitcher_debuts.csv"


def aggregate_residuals(
    pred_df: pd.DataFrame, min_pitches: int = 100
) -> pd.DataFrame:
    """Per-pitcher-season-pitchtype summary of residuals + run value."""
    grouped = (
        pred_df.groupby(["pitcher", "game_year", "pitch_type"])
        .agg(
            n_pitches=("residual", "size"),
            residual_mean=("residual", "mean"),
            residual_sum=("residual", "sum"),
            run_value_mean=("delta_run_exp", "mean"),
            velo_mean=("release_speed", "mean"),
            velo_diff_mean=("velo_diff_vs_fastball", "mean"),
            ivb_diff_mean=("ivb_diff_vs_fastball", "mean"),
            pfx_x_mean=("pfx_x", "mean"),
            pfx_z_mean=("pfx_z", "mean"),
            extension_mean=("release_extension", "mean"),
            spin_mean=("release_spin_rate", "mean"),
            fastball_velo_mean=("fastball_velo_mean", "first"),
        )
        .reset_index()
    )
    grouped["model_resistance"] = -100 * grouped["residual_mean"]
    grouped["run_value_per_100"] = -100 * grouped["run_value_mean"]
    return grouped[grouped["n_pitches"] >= min_pitches].reset_index(drop=True)


def lookup_debuts(mlbam_ids: list[int]) -> pd.DataFrame:
    """Return a DataFrame [mlbam_id, mlb_played_first] for the given IDs.

    Cached to disk after first call — Chadwick lookup is slow.
    """
    if DEBUTS_CACHE.exists():
        cached = pd.read_csv(DEBUTS_CACHE)
        missing = sorted(set(mlbam_ids) - set(cached["mlbam_id"]))
        if not missing:
            return cached
    else:
        cached = pd.DataFrame(columns=["mlbam_id", "mlb_played_first"])
        missing = sorted(set(mlbam_ids))

    if missing:
        print(f"[residual_analysis] looking up {len(missing)} new MLBAM debuts...")
        df = playerid_reverse_lookup(missing, key_type="mlbam")
        new = df[["key_mlbam", "mlb_played_first"]].rename(
            columns={"key_mlbam": "mlbam_id"}
        )
        cached = pd.concat([cached, new], ignore_index=True).drop_duplicates(
            "mlbam_id", keep="last"
        )
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cached.to_csv(DEBUTS_CACHE, index=False)

    return cached


def enrich_with_metadata(agg: pd.DataFrame) -> pd.DataFrame:
    """Join roster + Chadwick debut + experience features onto an aggregated table."""
    roster = resolve_roster()
    roster_lookup = roster[
        ["mlbam_id", "full_name", "group", "npb_years", "mlb_debut"]
    ].copy()
    roster_lookup["mlbam_id"] = roster_lookup["mlbam_id"].astype("Int64")

    out = agg.copy()
    out["pitcher"] = out["pitcher"].astype("Int64")
    out = out.merge(
        roster_lookup, left_on="pitcher", right_on="mlbam_id", how="left"
    ).drop(columns=["mlbam_id"])

    # Chadwick debut for the rest.
    debuts = lookup_debuts([int(p) for p in out["pitcher"].dropna().unique()])
    debuts["mlbam_id"] = debuts["mlbam_id"].astype("Int64")
    out = out.merge(
        debuts.rename(columns={"mlb_played_first": "chadwick_debut"}),
        left_on="pitcher", right_on="mlbam_id", how="left",
    ).drop(columns=["mlbam_id"])

    # Resolved debut: prefer hand-curated roster, fall back to Chadwick.
    out["debut_year"] = out["mlb_debut"].fillna(out["chadwick_debut"])
    out["mlb_seasons_in"] = out["game_year"] - out["debut_year"]
    out["is_novelty_year"] = (out["mlb_seasons_in"] == 0).astype("Int64")

    # Non-NPB pitchers default to 0 NPB years and "Domestic" group.
    out["npb_years"] = out["npb_years"].fillna(0).astype(int)
    out["group"] = out["group"].fillna("Domestic")

    # Left-censored flag — pitchers whose Chadwick debut is before our 2021 cutoff
    # AND aren't in the focal roster (where we have correct debuts). Their
    # mlb_seasons_in is correct via Chadwick, but flag it for downstream
    # sensitivity analysis if needed.
    out["left_censored"] = (
        out["debut_year"].notna() & (out["debut_year"] < 2021)
        & out["full_name"].isna()
    ).astype(int)

    return out


def main() -> None:
    print("[residual_analysis] loading per-pitch predictions...")
    pred_df = pd.read_parquet(PREDICTIONS_PATH)
    print(f"[residual_analysis]   {len(pred_df):,} pitches with predictions")

    print("[residual_analysis] aggregating residuals (min 100 pitches)...")
    agg = aggregate_residuals(pred_df, min_pitches=100)
    print(f"[residual_analysis]   {len(agg)} pitcher-season-pitchtype rows")

    print("[residual_analysis] enriching with roster + Chadwick metadata...")
    enriched = enrich_with_metadata(agg)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(RESIDUALS_PATH, index=False)
    print(f"[residual_analysis] wrote to {RESIDUALS_PATH}")

    # Quick descriptive: focal-pitcher rows.
    focal = enriched[enriched["full_name"].notna()].sort_values(
        ["full_name", "pitch_type", "game_year"]
    )
    print("[residual_analysis] focal-pitcher residuals:")
    cols = [
        "full_name", "game_year", "pitch_type", "n_pitches",
        "model_resistance", "run_value_per_100", "velo_mean",
    ]
    print(focal[cols].to_string(index=False))


if __name__ == "__main__":
    main()
