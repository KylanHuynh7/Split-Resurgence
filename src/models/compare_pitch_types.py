"""Step 11 deliverable: side-by-side comparison of FS and CH regressions.

Loads the two coefficient CSVs produced by `mixed_effects.py` and writes a
human-readable comparison to `reports/fs_vs_ch_comparison.txt`. The point
of the comparison is the question from the build plan:

    "Repeat steps 7-9 for changeups. Does NPB development predict residuals
    on changeups too? If yes, the effect isn't splitter-specific."

The headline answer is structural rather than statistical: none of the focal
NPB-developed pitchers throw enough changeups to estimate `npb_years` on CH
at all (0 rows pass the 100-pitch threshold). So `npb_years` is dropped from
the CH model. That itself is a finding — it confirms the focal NPB pitchers
are splitter specialists by arsenal construction, not by some general
deception advantage that should generalize across off-speed pitches.

For the predictors that ARE comparable (mlb_seasons_in, fastball_velo_mean,
usage_share, is_novelty_year), the comparison tells us whether a pattern
seen in FS generalizes to CH or is splitter-specific.

Run:
    uv run python -m src.models.compare_pitch_types
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
OUT_PATH = REPORTS_DIR / "fs_vs_ch_comparison.txt"


def load_coeffs(pitch_type: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"mixed_effects_coeffs_{pitch_type}.csv")


def render_row(term: str, fs: pd.DataFrame, ch: pd.DataFrame) -> str:
    fs_row = fs[fs["term"] == term]
    ch_row = ch[ch["term"] == term]
    fs_str = (
        f"{fs_row.iloc[0]['estimate']:+.3f} (p={fs_row.iloc[0]['p_value']:.3f})"
        if len(fs_row) else "—"
    )
    ch_str = (
        f"{ch_row.iloc[0]['estimate']:+.3f} (p={ch_row.iloc[0]['p_value']:.3f})"
        if len(ch_row) else "DROPPED"
    )
    return f"  {term:<22} FS: {fs_str:<22}  CH: {ch_str}"


def render_residuals_summary() -> str:
    """Pitcher-season counts at the 100-pitch threshold for each pitch type."""
    df = pd.read_parquet(PROCESSED_DIR / "pitcher_season_residuals.parquet")
    out_lines = []
    for pt in ("FS", "CH"):
        sub = df[df["pitch_type"] == pt]
        n_focal = sub["full_name"].notna().sum()
        n_npb = (sub["group"] == "NPB").sum()
        out_lines.append(
            f"  {pt}: {len(sub)} pitcher-seasons, {sub['pitcher'].nunique()} pitchers, "
            f"{n_focal} focal-roster rows, {n_npb} NPB-flagged rows"
        )
    return "\n".join(out_lines)


def main() -> None:
    fs = load_coeffs("FS")
    ch = load_coeffs("CH")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("=== FS vs CH MixedLM coefficient comparison ===\n\n")
        f.write("Sample sizes (pitcher-seasons with >= 100 pitches):\n")
        f.write(render_residuals_summary())
        f.write("\n\nFixed-effects coefficients side-by-side:\n")
        for term in [
            "Intercept",
            "npb_years",
            "mlb_seasons_in",
            "fastball_velo_mean",
            "usage_share",
            "is_novelty_year",
        ]:
            f.write(render_row(term, fs, ch) + "\n")

        f.write("\n--- Reading the comparison ---\n")
        f.write(
            "  npb_years is DROPPED on CH because none of the 5 focal NPB-developed\n"
            "  pitchers (Yamamoto, Ohtani, Imanaga, Sasaki, Senga) reached the\n"
            "  100-changeup threshold in any season. The Roberts hypothesis is\n"
            "  about splitters specifically; this data confirms NPB-developed\n"
            "  pitchers do not throw changeups in volumes large enough to test\n"
            "  whether NPB development would generalize to other off-speed pitches.\n\n"
            "  For the remaining predictors, similar coefficient signs and\n"
            "  magnitudes between FS and CH would mean the effect is general\n"
            "  off-speed behavior. Different magnitudes mean it is specific to\n"
            "  the splitter — those are the more interesting findings.\n"
        )

    print(f"[compare_pitch_types] wrote {OUT_PATH}")
    # Also print to stdout for quick inspection.
    with open(OUT_PATH) as f:
        print(f.read())


if __name__ == "__main__":
    main()
