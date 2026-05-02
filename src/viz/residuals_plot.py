"""Figure 1: residuals landscape.

Scatter of every FS pitcher-season:
    x: run_value_per_100   — actual performance (higher = better for pitcher)
    y: model_resistance    — actual minus physics-only prediction
                             (higher = pitcher beat the model)

The y=0 line is "physics fully explains the pitcher's performance." Pitchers
above the line outperform what physics alone predicts; pitchers below
underperform. Focal pitchers labeled; NPB-developed in vermillion.

Run:
    uv run python -m src.viz.residuals_plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.viz.style import COLORS, apply_style, save_figure

REPO_ROOT = Path(__file__).resolve().parents[2]
RESIDUALS_PATH = REPO_ROOT / "data" / "processed" / "pitcher_season_residuals.parquet"
FIG_PATH = REPO_ROOT / "reports" / "figures" / "fig01_residuals_landscape.png"


def main() -> None:
    apply_style()
    df = pd.read_parquet(RESIDUALS_PATH)
    fs = df[df["pitch_type"] == "FS"].copy()

    fig, ax = plt.subplots(figsize=(8.5, 6))

    # Pool (non-focal) pitcher-seasons.
    pool = fs[fs["full_name"].isna()]
    ax.scatter(
        pool["run_value_per_100"], pool["model_resistance"],
        s=14, c=COLORS["pool"], alpha=0.4, edgecolors="none",
        label=f"All FS pitchers (n={len(pool)})",
    )

    # Focal pitchers.
    focal = fs[fs["full_name"].notna()].copy()
    npb = focal[focal["group"] == "NPB"]
    dom = focal[focal["group"] == "Domestic"]
    ax.scatter(
        npb["run_value_per_100"], npb["model_resistance"],
        s=70, c=COLORS["npb"], edgecolors="white", linewidth=1.0,
        label=f"NPB-developed (n={len(npb)})", zorder=3,
    )
    ax.scatter(
        dom["run_value_per_100"], dom["model_resistance"],
        s=70, c=COLORS["domestic"], edgecolors="white", linewidth=1.0,
        label=f"Domestic focal (n={len(dom)})", zorder=3,
    )

    # Labels for focal pitcher-seasons.
    for _, row in focal.iterrows():
        label = f"{row['full_name'].split()[-1]} '{str(int(row['game_year']))[-2:]}"
        ax.annotate(
            label,
            (row["run_value_per_100"], row["model_resistance"]),
            xytext=(4, 3), textcoords="offset points",
            fontsize=7.5, color="#333333",
        )

    # Reference: physics-fully-explains-it line.
    ax.axhline(0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
    ax.text(
        ax.get_xlim()[1], 0.05,
        "physics-only prediction = actual",
        fontsize=8, ha="right", color="#444444",
    )

    ax.set_xlabel("Run value per 100 pitches (higher = better for pitcher)")
    ax.set_ylabel(
        "Model resistance (residual, RV/100)\n"
        "(higher = pitcher beat physics-only model)"
    )
    ax.set_title("FS pitcher-seasons, 2021–2025: actual run value vs physics-model residual")
    ax.legend(loc="lower right", fontsize=9)

    save_figure(fig, FIG_PATH)


if __name__ == "__main__":
    main()
