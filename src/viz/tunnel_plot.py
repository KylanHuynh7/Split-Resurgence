"""Figure 2: tunneling overlays for the three case-study pitchers.

For each of Yamamoto, Gausman, Skenes:
  - Top panel: release-point density (release_pos_x vs release_pos_z).
    A tight overlap between the fastball and splitter clouds means the
    hitter sees the same release point — the deception window the Roberts
    hypothesis is about.
  - Bottom panel: plate-crossing location (plate_x vs plate_z) with the
    typical strike zone box. Where the splitter actually lands.

Ranges are pitcher-specific so each cloud fills its panel — comparison is
within-pitcher (FB vs FS), not across pitchers.

Run:
    uv run python -m src.viz.tunnel_plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from src.data.pitcher_metadata import resolve_roster
from src.data.pull_statcast import SEASON_WINDOWS, _cache_path
from src.viz.style import apply_style, save_figure

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_PATH = REPO_ROOT / "reports" / "figures" / "fig02_tunnels.png"

CASE_STUDY_PITCHERS = ["Yoshinobu Yamamoto", "Kevin Gausman", "Paul Skenes"]
FB_TYPES = ("FF", "SI", "FC")
FS_TYPE = "FS"

# Strike-zone approximation for a generic batter (feet).
SZ_LEFT, SZ_RIGHT = -0.7083, 0.7083
SZ_BOT, SZ_TOP = 1.5, 3.5


def load_pitcher_pitches(name: str) -> pd.DataFrame:
    roster = resolve_roster()
    row = roster[roster["full_name"] == name].iloc[0]
    mlbam = int(row["mlbam_id"])

    parts = []
    for season in sorted(SEASON_WINDOWS):
        path = _cache_path(str(season))
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=[
            "pitcher", "pitch_type", "release_pos_x", "release_pos_z",
            "plate_x", "plate_z", "release_speed",
        ])
        parts.append(df[df["pitcher"] == mlbam])
    out = pd.concat(parts, ignore_index=True)
    return out[out["pitch_type"].isin([*FB_TYPES, FS_TYPE])].copy()


def plot_release(ax, df: pd.DataFrame, title: str) -> None:
    fb = df[df["pitch_type"].isin(FB_TYPES)]
    fs = df[df["pitch_type"] == FS_TYPE]
    ax.scatter(
        fb["release_pos_x"], fb["release_pos_z"],
        s=4, c="#0072B2", alpha=0.15, edgecolors="none", label=f"FB (n={len(fb):,})",
    )
    ax.scatter(
        fs["release_pos_x"], fs["release_pos_z"],
        s=4, c="#D55E00", alpha=0.35, edgecolors="none", label=f"FS (n={len(fs):,})",
    )
    ax.set_title(title)
    ax.set_xlabel("Release X (ft)")
    ax.set_ylabel("Release Z (ft)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="lower left", markerscale=3)


def plot_plate(ax, df: pd.DataFrame) -> None:
    fb = df[df["pitch_type"].isin(FB_TYPES)]
    fs = df[df["pitch_type"] == FS_TYPE]
    ax.scatter(
        fb["plate_x"], fb["plate_z"],
        s=4, c="#0072B2", alpha=0.10, edgecolors="none",
    )
    ax.scatter(
        fs["plate_x"], fs["plate_z"],
        s=4, c="#D55E00", alpha=0.30, edgecolors="none",
    )
    sz = Rectangle(
        (SZ_LEFT, SZ_BOT), SZ_RIGHT - SZ_LEFT, SZ_TOP - SZ_BOT,
        linewidth=1.0, edgecolor="#444444", facecolor="none", linestyle="--",
    )
    ax.add_patch(sz)
    ax.set_xlabel("Plate X (ft)")
    ax.set_ylabel("Plate Z (ft)")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(0.0, 4.5)
    ax.set_aspect("equal", adjustable="box")


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for col, name in enumerate(CASE_STUDY_PITCHERS):
        df = load_pitcher_pitches(name)
        plot_release(axes[0, col], df, name)
        plot_plate(axes[1, col], df)

    fig.suptitle(
        "Tunneling overlay: release point (top) and plate location (bottom)\n"
        "Blue = fastball (FF/SI/FC), Orange = splitter (FS) — 2021-2025 cumulative",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, FIG_PATH)


if __name__ == "__main__":
    main()
