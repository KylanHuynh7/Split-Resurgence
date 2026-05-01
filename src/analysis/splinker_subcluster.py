"""Velocity-band sub-clustering on the FS pitch tag.

The FS label bundles three meaningfully different pitches:
  - Traditional splitter (~85 mph, large velo gap from FB)
  - Splinker / hard splitter (Skenes ~94 mph, small velo gap)
  - Ghost fork (Senga, very low spin, idiosyncratic)

The revised plan moves this from "secondary research question" to "gating step
before Phase 1." We cluster on (mean velocity, mean velo differential vs FB)
at pitcher-season grain and tag each FS pitcher-season with a sub-class so
downstream models can choose to pool, exclude, or stratify.

GMM over k-means: the boundary between traditional and splinker is fuzzy and
soft assignments (posterior probabilities) are useful. k=2 to start (the
plan's headline distinction); inspect and revisit if a third cluster is
clearly justified by the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURES = ("velo_mean", "velo_diff_mean")


def fit_velocity_subclusters(
    season_agg: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 0,
) -> pd.DataFrame:
    """Fit a GMM on FS pitcher-seasons. Adds a `fs_subcluster` column.

    `season_agg` is the output of `aggregate.aggregate_pitcher_season`. Only
    rows with pitch_type == 'FS' are clustered; CH rows are passed through
    with `fs_subcluster = NaN`.

    Cluster labels are renamed by mean velocity so they're stable across runs:
      0 = "traditional" (lower velo)
      1 = "splinker"    (higher velo)
    """
    out = season_agg.copy()
    out["fs_subcluster"] = pd.NA
    out["fs_subcluster_label"] = pd.NA

    fs_mask = out["pitch_type"] == "FS"
    fs_rows = out[fs_mask].dropna(subset=list(CLUSTER_FEATURES))
    if len(fs_rows) < n_components:
        # Smoke-test guard: too few rows to fit.
        return out

    X = fs_rows[list(CLUSTER_FEATURES)].to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    gmm = GaussianMixture(
        n_components=n_components, random_state=random_state, covariance_type="full",
    ).fit(Xs)
    raw_labels = gmm.predict(Xs)

    # Reorder labels by ascending mean velocity so 0 = traditional, 1 = splinker.
    order = np.argsort(
        [fs_rows["velo_mean"].to_numpy()[raw_labels == k].mean()
         for k in range(n_components)]
    )
    relabel = {old: new for new, old in enumerate(order)}
    new_labels = np.array([relabel[r] for r in raw_labels])

    label_names = {0: "traditional", 1: "splinker"}
    if n_components > 2:
        label_names = {i: f"cluster_{i}" for i in range(n_components)}
        label_names[0] = "traditional"
        label_names[n_components - 1] = "splinker"

    out.loc[fs_rows.index, "fs_subcluster"] = new_labels
    out.loc[fs_rows.index, "fs_subcluster_label"] = [
        label_names[i] for i in new_labels
    ]
    return out


def cluster_summary(season_agg: pd.DataFrame) -> pd.DataFrame:
    """Quick descriptive table of each cluster's centroid stats."""
    fs = season_agg[season_agg["pitch_type"] == "FS"].dropna(subset=["fs_subcluster"])
    if fs.empty:
        return pd.DataFrame()
    return (
        fs.groupby("fs_subcluster_label")
        .agg(
            n_pitcher_seasons=("pitcher", "nunique"),
            velo_mean=("velo_mean", "mean"),
            velo_diff_mean=("velo_diff_mean", "mean"),
            ivb_diff_mean=("ivb_diff_mean", "mean"),
            run_value_per_100_mean=("run_value_per_100", "mean"),
        )
        .reset_index()
    )
