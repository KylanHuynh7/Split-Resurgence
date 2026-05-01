# Split-Resurgence

Do run-expectancy models systematically undervalue the splitter, and if so, what
movement and sequencing characteristics does the pitch have that existing
metrics fail to capture?

Status: **data layer complete, modeling layer pending.**

## What's here

```
src/
├── data/
│   ├── pull_statcast.py       # season-level Statcast pulls, parquet cached
│   ├── pitcher_metadata.py    # hand-curated NPB-vs-domestic roster + fastball baseline
│   ├── build_features.py      # per-pitch feature table (FS + CH)
│   └── aggregate.py           # pitcher-season aggregator (run value / 100, etc.)
├── analysis/
│   └── splinker_subcluster.py # GMM on FS pitches: traditional vs splinker
└── run_smoke.py               # end-to-end smoke test (7-day pull)
```

## Reproducing

```sh
# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Smoke test (one week of pitches, ~30 seconds)
uv run python -m src.run_smoke

# Full data pull (slow — pulls 2021-2025 league-wide, several minutes per season)
uv run python -m src.data.pull_statcast --all
```

Cached pulls land in `data/raw/statcast_{season}.parquet` and are gitignored
(regenerable from the pybaseball source). The hand-curated roster lands in
`data/processed/pitcher_roster.csv` and is gitignored too — re-run the smoke
test or call `resolve_roster()` to regenerate it.

## Project plan

The full research plan, mentor revisions, and methodological commitments live
in [`docs/plan.md`](docs/plan.md) (move from chat-attached spec when ready).

The plan calls for three modeling steps next:

> 7. Fit physical-features baseline model (XGBoost predicting per-pitch run value)
> 8. Compute per-pitcher-season residuals = "model resistance" score
> 9. Mixed-effects regression of residuals on NPB years, MLB experience, etc.

The choice of model family for step 7 (XGBoost vs LightGBM vs something
simpler) and the exact specification for step 9 are deferred to the modeling
phase.

## What the data layer guarantees

- Pitch-level FS and CH dataset for 2021-2025, with run value (`delta_run_exp`)
  attached to every pitch.
- Velocity and IVB differential features computed against each pitcher-season's
  primary fastball (FF/SI/FC, whichever was thrown most).
- A pitcher-season grain table with `run_value_per_100` (sign-flipped so
  higher = better pitch), whiff rate, and usage share.
- Sub-clustering of the FS tag into "traditional" and "splinker" classes, so
  Skenes-style 94 mph splinkers don't pollute traditional-splitter analyses.

## Honest limitations of the data layer

- Statcast `pfx_z` is in feet, not the inches typically used in pitch-grading
  literature. Multiply by 12 if you want IVB in inches.
- The FS tag changed for Skenes mid-2025; raw data reflects whatever Statcast
  decided that day. The sub-cluster is the right way to handle this, not the
  pitch_type column directly.
- `delta_run_exp` is base-out-state aware but doesn't fully account for
  pitcher/batter quality. The residual analysis (downstream) is what
  separates pitch-level effects from context.
