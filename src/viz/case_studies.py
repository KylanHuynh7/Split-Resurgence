"""Figure 3: case-study trajectories — Yamamoto, Gausman, Skenes.

Two stacked panels:
  Top:    Year-over-year FS model_resistance for each case-study pitcher
          (line + markers). The y=0 reference is "physics-only model exactly
          predicts this pitcher's run value."
  Bottom: Against-handedness splits — model_resistance computed per
          (pitcher, season, batter-handedness) cell. Tests whether splitter
          model-resistance is uniform across LHB and RHB.

The bottom panel goes back to per-pitch predictions (FS rows only), so it
needs `pitches_with_predictions.parquet` — regenerable via
`uv run python -m src.models.stuff_baseline` if not present.

Run:
    uv run python -m src.viz.case_studies
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.pitcher_metadata import resolve_roster
from src.viz.style import COLORS, apply_style, save_figure

REPO_ROOT = Path(__file__).resolve().parents[2]
RESIDUALS_PATH = REPO_ROOT / "data" / "processed" / "pitcher_season_residuals.parquet"
PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "pitches_with_predictions.parquet"
FIG_PATH = REPO_ROOT / "reports" / "figures" / "fig03_case_studies.png"

CASE_STUDIES = [
    ("Yoshinobu Yamamoto", COLORS["yamamoto"]),
    ("Kevin Gausman", COLORS["gausman"]),
    ("Paul Skenes", COLORS["skenes"]),
]


def trajectory_panel(ax, residuals: pd.DataFrame) -> None:
    """Year-over-year model_resistance for each case-study pitcher."""
    fs = residuals[residuals["pitch_type"] == "FS"]
    for name, color in CASE_STUDIES:
        sub = fs[fs["full_name"] == name].sort_values("game_year")
        if sub.empty:
            continue
        ax.plot(
            sub["game_year"], sub["model_resistance"],
            marker="o", markersize=7, linewidth=2.0, color=color, label=name,
        )
        # Annotate sample size next to each marker.
        for _, row in sub.iterrows():
            ax.annotate(
                f"n={int(row['n_pitches'])}",
                (row["game_year"], row["model_resistance"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=7, color=color,
            )

    ax.axhline(0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Season")
    ax.set_ylabel("Model resistance (RV/100)")
    ax.set_title("FS model resistance by season — case studies")
    ax.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax.legend(loc="best", fontsize=9)


def handedness_panel(ax, predictions: pd.DataFrame, roster: pd.DataFrame) -> None:
    """Model resistance broken out by batter handedness for each case study."""
    fs = predictions[predictions["pitch_type"] == "FS"].copy()
    fs = fs.merge(
        roster[["mlbam_id", "full_name"]], left_on="pitcher", right_on="mlbam_id",
        how="inner",
    )

    # Per (pitcher, season, stand): mean residual → model_resistance
    agg = (
        fs.groupby(["full_name", "game_year", "stand"])
        .agg(n=("residual", "size"), residual_mean=("residual", "mean"))
        .reset_index()
    )
    agg = agg[agg["n"] >= 30]  # don't show tiny cells
    agg["model_resistance"] = -100 * agg["residual_mean"]

    case_names = [c[0] for c in CASE_STUDIES]
    case_colors = {n: c for n, c in CASE_STUDIES}
    agg = agg[agg["full_name"].isin(case_names)]

    # Grouped bar: x=year, color=stand, one cluster per pitcher
    pitchers = [n for n in case_names if n in agg["full_name"].unique()]
    n_pitchers = len(pitchers)
    n_groups = max(agg["game_year"].nunique(), 1)
    bar_w = 0.18

    # We'll plot one set of bars per pitcher in a single axis, with x = year.
    # Marker shape encodes handedness; color encodes pitcher.
    handedness_marker = {"L": "o", "R": "s"}
    for name in pitchers:
        sub = agg[agg["full_name"] == name].sort_values(["game_year", "stand"])
        for stand, marker in handedness_marker.items():
            sub_h = sub[sub["stand"] == stand]
            if sub_h.empty:
                continue
            ax.scatter(
                sub_h["game_year"] + (0.10 if stand == "R" else -0.10),
                sub_h["model_resistance"],
                marker=marker, s=80, color=case_colors[name],
                edgecolors="white", linewidth=1.0,
                label=f"{name} vs {'RHB' if stand == 'R' else 'LHB'}",
            )

    ax.axhline(0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Season")
    ax.set_ylabel("Model resistance (RV/100)")
    ax.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax.set_title(
        "FS against-handedness splits — circle = vs LHB, square = vs RHB"
    )
    # De-duplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="best", fontsize=8, ncol=2)


def main() -> None:
    apply_style()
    residuals = pd.read_parquet(RESIDUALS_PATH)
    roster = resolve_roster()

    fig, (ax_traj, ax_hand) = plt.subplots(2, 1, figsize=(9, 9), sharex=False)
    trajectory_panel(ax_traj, residuals)

    if PREDICTIONS_PATH.exists():
        predictions = pd.read_parquet(PREDICTIONS_PATH)
        handedness_panel(ax_hand, predictions, roster)
    else:
        ax_hand.text(
            0.5, 0.5,
            "pitches_with_predictions.parquet missing — "
            "run `uv run python -m src.models.stuff_baseline` to regenerate",
            transform=ax_hand.transAxes, ha="center", va="center",
            fontsize=10, color="#888888",
        )
        ax_hand.set_axis_off()

    fig.suptitle(
        "Case studies: Yamamoto / Gausman / Skenes splitter trajectories",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    save_figure(fig, FIG_PATH)


if __name__ == "__main__":
    main()
