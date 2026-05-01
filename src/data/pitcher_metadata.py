"""Pitcher metadata for the focal roster.

Two parts:

1. A hand-curated biographical roster (`FOCAL_ROSTER`): the splitter throwers
   named in the project plan, plus their NPB years and MLB debut. This is the
   source of the "NPB development years" feature used by the regression.
   Edit this dict to add or correct pitchers.

2. A `build_fastball_baseline()` function that computes each pitcher-season's
   average fastball velocity and induced vertical break from raw Statcast.
   The "fastball" is whichever of {FF, SI, FC} the pitcher threw most that
   season — the splitter's tunneling partner depends on the pitcher.

The MLBAM `pitcher` ID column in Statcast is what we join on. We resolve names
to IDs via `pybaseball.playerid_lookup` once and persist to
`data/processed/pitcher_roster.csv` so subsequent runs don't re-query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from pybaseball import playerid_lookup

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ROSTER_CSV = PROCESSED_DIR / "pitcher_roster.csv"


@dataclass
class PitcherEntry:
    last: str
    first: str
    group: str  # "NPB" or "Domestic"
    mlb_debut: int
    # Years pitching in NPB before MLB. 0 for domestic pitchers.
    npb_years: int = 0
    # Notes flagged in the project plan (e.g. "splinker", "small sample").
    notes: str = ""
    # Filled in at resolution time.
    mlbam_id: Optional[int] = None


# Hand-curated. Edit to extend.
# NPB years are pro NPB seasons before MLB debut (rounded down). Sources:
# Baseball Reference + NPB Wikipedia.
FOCAL_ROSTER: list[PitcherEntry] = [
    # --- Group A: NPB-developed -------------------------------------------------
    PitcherEntry(
        last="Yamamoto", first="Yoshinobu",
        group="NPB", mlb_debut=2024, npb_years=7,
        notes="Dodgers ace; primary splitter weapon",
    ),
    PitcherEntry(
        last="Ohtani", first="Shohei",
        group="NPB", mlb_debut=2018, npb_years=5,
        notes="Pitching seasons 2018-2023 only; no pitching 2024+",
    ),
    PitcherEntry(
        last="Imanaga", first="Shota",
        group="NPB", mlb_debut=2024, npb_years=8,
        notes="Cubs; splitter-heavy arsenal",
    ),
    PitcherEntry(
        last="Sasaki", first="Roki",
        group="NPB", mlb_debut=2025, npb_years=4,
        notes="Dodgers; low-spin splitter; small MLB sample",
    ),
    PitcherEntry(
        last="Senga", first="Kodai",
        group="NPB", mlb_debut=2023, npb_years=11,
        notes="Mets; ghost fork variant",
    ),
    # --- Group B: Domestic ------------------------------------------------------
    PitcherEntry(
        last="Gausman", first="Kevin",
        group="Domestic", mlb_debut=2013,
        notes="Gold standard domestic splitter; most whiffs in Statcast era",
    ),
    PitcherEntry(
        last="Eovaldi", first="Nathan",
        group="Domestic", mlb_debut=2011,
        notes="Horizontal splitter profile; contrast to NPB vertical drop",
    ),
    PitcherEntry(
        last="Skenes", first="Paul",
        group="Domestic", mlb_debut=2024,
        notes="Splinker (94 mph); reclassified FS in 2025; velocity outlier",
    ),
    PitcherEntry(
        last="Walker", first="Taijuan",
        group="Domestic", mlb_debut=2012,
        notes="Cautionary case; worst splitter by run value 2024 (-16)",
    ),
    PitcherEntry(
        last="Yesavage", first="Trey",
        group="Domestic", mlb_debut=2025,
        notes="Over-the-top delivery; 57% whiff in 2025; small sample",
    ),
]


def _lookup_id(entry: PitcherEntry) -> Optional[int]:
    """Resolve an MLBAM ID via pybaseball. Returns None if no match."""
    df = playerid_lookup(entry.last, entry.first)
    if df.empty:
        return None
    # If multiple matches, prefer the one whose mlb_played_first matches debut.
    df = df[df["mlb_played_first"].notna()]
    if df.empty:
        return None
    debut_match = df[df["mlb_played_first"] == entry.mlb_debut]
    if not debut_match.empty:
        df = debut_match
    return int(df.iloc[0]["key_mlbam"])


def resolve_roster(force: bool = False) -> pd.DataFrame:
    """Materialize the focal roster as a DataFrame with MLBAM IDs.

    Caches to `data/processed/pitcher_roster.csv`. Pass `force=True` to re-resolve.
    """
    if ROSTER_CSV.exists() and not force:
        return pd.read_csv(ROSTER_CSV)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in FOCAL_ROSTER:
        e.mlbam_id = _lookup_id(e)
        rows.append({
            "mlbam_id": e.mlbam_id,
            "last": e.last,
            "first": e.first,
            "full_name": f"{e.first} {e.last}",
            "group": e.group,
            "mlb_debut": e.mlb_debut,
            "npb_years": e.npb_years,
            "notes": e.notes,
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROSTER_CSV, index=False)
    return df


# --- Fastball baseline ------------------------------------------------------

# Which pitch types count as "fastball" for tunneling-partner purposes.
FASTBALL_TYPES = ("FF", "SI", "FC")


def build_fastball_baseline(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per pitcher-season, the primary fastball type and its avg velo + IVB.

    Input: raw Statcast pitches with columns
        pitcher, game_year, pitch_type, release_speed, pfx_z (vertical movement
        with gravity removed, ft) — pfx_z is Statcast's IVB-equivalent in feet,
        we'll convert to inches downstream.

    Output cols:
        pitcher, game_year, fastball_type, fastball_count,
        fastball_velo_mean, fastball_pfx_z_mean
    """
    fb = pitches[pitches["pitch_type"].isin(FASTBALL_TYPES)].copy()
    if fb.empty:
        return pd.DataFrame(
            columns=[
                "pitcher", "game_year", "fastball_type", "fastball_count",
                "fastball_velo_mean", "fastball_pfx_z_mean",
            ]
        )

    # Pick each pitcher-season's most-thrown fastball type.
    counts = (
        fb.groupby(["pitcher", "game_year", "pitch_type"])
        .size()
        .reset_index(name="n")
    )
    primary = (
        counts.sort_values("n", ascending=False)
        .drop_duplicates(["pitcher", "game_year"])
        .rename(columns={"pitch_type": "fastball_type", "n": "fastball_count"})
    )

    # Now compute means restricted to that primary type per pitcher-season.
    fb_keyed = fb.merge(
        primary[["pitcher", "game_year", "fastball_type"]],
        on=["pitcher", "game_year"],
    )
    fb_keyed = fb_keyed[fb_keyed["pitch_type"] == fb_keyed["fastball_type"]]

    means = (
        fb_keyed.groupby(["pitcher", "game_year"])
        .agg(
            fastball_velo_mean=("release_speed", "mean"),
            fastball_pfx_z_mean=("pfx_z", "mean"),
        )
        .reset_index()
    )

    return primary.merge(means, on=["pitcher", "game_year"])
