"""Shared matplotlib styling for project figures.

The plan explicitly forbids default Seaborn styling for final figures, so this
module sets clean rcParams once and is imported by every viz module.

Design choices:
  - Sans-serif (Helvetica / Arial / DejaVu fallback) at 10pt body, 12pt title.
  - Light gridlines, no top/right spines — keeps focus on the data.
  - A 4-color qualitative palette: NPB pitchers (warm), Domestic (cool),
    league pool (gray), accent (highlight).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Color palette. Hex chosen for color-blind safety (Wong 2011-style).
COLORS = {
    "npb": "#D55E00",        # vermillion
    "domestic": "#0072B2",   # blue
    "pool": "#999999",       # neutral gray
    "accent": "#CC79A7",     # pink (for callouts)
    "yamamoto": "#D55E00",
    "gausman": "#0072B2",
    "skenes": "#009E73",     # green — third distinct hue
}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
    })


def save_figure(fig: plt.Figure, path) -> None:
    """Save with consistent settings; ensures parent dir exists."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    print(f"[viz] saved {p}")
