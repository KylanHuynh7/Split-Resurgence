"""Pull pitch-level Statcast data and cache to parquet.

Public API:
    pull_season(season, overwrite=False) -> pd.DataFrame
        Returns all pitches in [start, end] for the regular + postseason of `season`,
        cached to data/raw/statcast_{season}.parquet.

    pull_range(start, end, label) -> pd.DataFrame
        Arbitrary date range, cached to data/raw/statcast_{label}.parquet.
        Used for the smoke test.

We pull league-wide and filter to FS / CH downstream rather than per-pitcher,
because the physical-features baseline model needs the full distribution
of changeups for comparison and the splitter analysis needs all FS pitchers
above the 100-pitch threshold.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from pybaseball import statcast
from pybaseball import cache as pyb_cache

# Enable pybaseball's own on-disk HTTP cache so re-pulls are fast.
pyb_cache.enable()

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Regular + postseason windows. Statcast's `statcast()` accepts any date range
# and returns whatever pitches are within it; spring training is excluded by
# default. Postseason runs into early November in recent years.
SEASON_WINDOWS: dict[int, tuple[str, str]] = {
    2021: ("2021-04-01", "2021-11-03"),
    2022: ("2022-04-07", "2022-11-06"),
    2023: ("2023-03-30", "2023-11-02"),
    2024: ("2024-03-28", "2024-11-01"),
    2025: ("2025-03-27", "2025-11-02"),
}


def _cache_path(label: str) -> Path:
    return RAW_DIR / f"statcast_{label}.parquet"


def _month_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into monthly chunks (inclusive). Last chunk truncated."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    windows: list[tuple[str, str]] = []
    cur = s
    while cur <= e:
        # Last day of `cur`'s month, or `e`, whichever is earlier.
        if cur.month == 12:
            month_end = date(cur.year, 12, 31)
        else:
            month_end = date(cur.year, cur.month + 1, 1) - timedelta(days=1)
        chunk_end = min(month_end, e)
        windows.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + timedelta(days=1)
    return windows


def _statcast_with_retry(
    start: str, end: str, attempts: int = 3, backoff: float = 5.0
) -> pd.DataFrame:
    """Call pybaseball.statcast() with retries on transient parser/HTTP failures.

    Baseball Savant occasionally returns a malformed CSV (HTML error page,
    partial response) that pandas can't tokenize. This is transient — retrying
    after a short wait usually works.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            df = statcast(start_dt=start, end_dt=end, verbose=False)
            return df
        except (pd.errors.ParserError, ValueError, OSError) as err:
            last_err = err
            print(
                f"[statcast {start}..{end}] attempt {attempt}/{attempts} failed: "
                f"{type(err).__name__}: {err}"
            )
            if attempt < attempts:
                time.sleep(backoff * attempt)
    assert last_err is not None
    raise last_err


def pull_range(
    start: str,
    end: str,
    label: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Pull all Statcast pitches in [start, end] (inclusive) and cache to parquet.

    `label` is the cache filename suffix (e.g., "2024", "smoke_2025w32").
    Re-pulls are skipped if the parquet exists unless `overwrite=True`.

    For ranges longer than a month, chunks monthly and retries each chunk
    independently — one bad day from Baseball Savant doesn't kill the whole
    range. Per-month parquets are also cached so resuming after a crash is fast.
    """
    out = _cache_path(label)
    if out.exists() and not overwrite:
        return pd.read_parquet(out)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    windows = _month_windows(start, end)

    parts: list[pd.DataFrame] = []
    for ws, we in windows:
        chunk_label = f"{label}_chunk_{ws}_{we}"
        chunk_path = _cache_path(chunk_label)
        if chunk_path.exists() and not overwrite:
            parts.append(pd.read_parquet(chunk_path))
            continue
        chunk_df = _statcast_with_retry(ws, we)
        chunk_df = chunk_df[chunk_df["pitch_type"].notna()].copy()
        chunk_df.to_parquet(chunk_path, index=False)
        parts.append(chunk_df)
        print(f"[statcast] {ws}..{we}: {len(chunk_df):,} pitches")

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.to_parquet(out, index=False)
    return df


def pull_season(season: int, overwrite: bool = False) -> pd.DataFrame:
    """Pull a full season (regular + postseason) and cache."""
    if season not in SEASON_WINDOWS:
        raise ValueError(
            f"Season {season} not configured. Add to SEASON_WINDOWS or use pull_range()."
        )
    start, end = SEASON_WINDOWS[season]
    return pull_range(start, end, label=str(season), overwrite=overwrite)


def pull_all(
    seasons: Optional[list[int]] = None, overwrite: bool = False
) -> dict[int, pd.DataFrame]:
    """Pull every configured season. Returns {season: df}."""
    seasons = seasons or sorted(SEASON_WINDOWS)
    return {s: pull_season(s, overwrite=overwrite) for s in seasons}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pull Statcast season(s) to parquet.")
    parser.add_argument("--season", type=int, help="Single season to pull")
    parser.add_argument("--all", action="store_true", help="Pull all configured seasons")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all:
        for s, df in pull_all(overwrite=args.overwrite).items():
            print(f"{s}: {len(df):,} pitches -> {_cache_path(str(s))}")
    elif args.season:
        df = pull_season(args.season, overwrite=args.overwrite)
        print(f"{args.season}: {len(df):,} pitches -> {_cache_path(str(args.season))}")
    else:
        parser.error("Pass --season YYYY or --all")
