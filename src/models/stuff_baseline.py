"""Step 7: physical-features baseline model.

Trains a single XGBoost regressor predicting per-pitch `delta_run_exp` from
physical features only (velocity, break, release point, plate location,
count state, handedness, pitch type). Pitcher and batter identity are
deliberately EXCLUDED — residuals downstream measure how much each pitcher
beats or trails the physics-only prediction.

Trained on FS + CH + FF + SI + FC pitches across 2021-2025 (the unified
design we picked over per-pitch-type submodels). Evaluated and predicted via
GroupKFold cross-validation grouped by (pitcher, season) so within-PA leakage
and pitcher-season memorization are blocked.

Persists per-pitch out-of-sample predictions for FS and CH only — those are
the analytical targets. The fastball rows participated only as training data.

Run:
    uv run python -m src.models.stuff_baseline
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from src.data.build_features import build_pitch_features
from src.data.pull_statcast import SEASON_WINDOWS, _cache_path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PREDICTIONS_PATH = PROCESSED_DIR / "pitches_with_predictions.parquet"
MODEL_PATH = PROCESSED_DIR / "stuff_baseline.json"

TRAIN_PITCH_TYPES = ("FS", "CH", "FF", "SI", "FC")
ANALYZE_PITCH_TYPES = ("FS", "CH")

FEATURES_NUMERIC = [
    "release_speed",
    "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z", "release_extension",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "spin_axis", "release_spin_rate",
    "plate_x", "plate_z",
    "velo_diff_vs_fastball", "ivb_diff_vs_fastball",
    "balls", "strikes",
]
FEATURES_CATEGORICAL = ["pitch_type", "stand", "p_throws"]
FEATURE_COLS = FEATURES_NUMERIC + FEATURES_CATEGORICAL
TARGET = "delta_run_exp"


def load_all_seasons() -> pd.DataFrame:
    """Concatenate cached parquets for every configured season."""
    parts = []
    for season in sorted(SEASON_WINDOWS):
        path = _cache_path(str(season))
        if not path.exists():
            print(f"[stuff_baseline] missing cache for {season}, skipping")
            continue
        parts.append(pd.read_parquet(path))
    return pd.concat(parts, ignore_index=True)


def prepare_modeling_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Feature table for FS+CH+fastballs, no NaN rows on essential features."""
    feats = build_pitch_features(raw, pitch_types=TRAIN_PITCH_TYPES)
    feats = feats.dropna(
        subset=[TARGET, "release_speed", "pfx_z", "plate_x", "plate_z"]
    )
    # XGBoost needs categoricals as pandas category dtype.
    for col in FEATURES_CATEGORICAL:
        feats[col] = feats[col].astype("category")
    return feats


def make_model(random_state: int = 0) -> xgb.XGBRegressor:
    """Modest GBT — we want the model to capture physics, not memorize subspaces.

    Defaults chosen to under-fit slightly (depth=6, lr=0.05, subsample=0.7).
    The downstream regression on residuals is what extracts pitcher-level
    signal; we don't want the baseline soaking up all the variance.
    """
    return xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=20,
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        n_jobs=-1,
        random_state=random_state,
    )


def cv_predict_groupkfold(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 0
) -> np.ndarray:
    """Out-of-sample predictions via GroupKFold on (pitcher, season).

    Each fold trains on ~80% of pitcher-seasons and predicts the held-out 20%,
    so predictions for any pitcher-season come from a model that didn't see
    that pitcher-season. Pitcher identity isn't a feature, but this still
    guards against within-PA correlation leakage.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET].to_numpy()
    groups = (
        df["pitcher"].astype(str) + "_" + df["game_year"].astype(str)
    ).to_numpy()

    preds = np.full(len(df), np.nan, dtype=np.float64)
    kf = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y, groups), 1):
        model = make_model(random_state=random_state + fold)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X.iloc[test_idx])
        print(
            f"[stuff_baseline] fold {fold}/{n_splits}: "
            f"train={len(train_idx):,} test={len(test_idx):,}"
        )
    return preds


def fit_final_model(df: pd.DataFrame, random_state: int = 0) -> xgb.XGBRegressor:
    """Refit on all data for case-study predictions and Importance plots."""
    model = make_model(random_state=random_state)
    model.fit(df[FEATURE_COLS], df[TARGET].to_numpy())
    return model


def main() -> None:
    print("[stuff_baseline] loading all seasons...")
    raw = load_all_seasons()
    print(f"[stuff_baseline]   raw pitches: {len(raw):,}")

    print("[stuff_baseline] building modeling table (FS+CH+fastballs)...")
    feats = prepare_modeling_table(raw)
    print(f"[stuff_baseline]   modeling rows: {len(feats):,}")
    print(
        f"[stuff_baseline]   by pitch type:\n{feats['pitch_type'].value_counts()}"
    )

    print("[stuff_baseline] cross-validated predictions (5-fold GroupKFold)...")
    feats["pred_run_value"] = cv_predict_groupkfold(feats)
    feats["residual"] = feats[TARGET] - feats["pred_run_value"]

    rmse = float(np.sqrt(np.mean(feats["residual"] ** 2)))
    bias = float(feats["residual"].mean())
    print(f"[stuff_baseline]   OOS RMSE = {rmse:.4f}, mean residual = {bias:.4g}")

    print("[stuff_baseline] fitting final full-data model (for case studies)...")
    final = fit_final_model(feats)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = feats[feats["pitch_type"].isin(ANALYZE_PITCH_TYPES)].copy()
    # XGBoost categorical columns don't round-trip through parquet cleanly;
    # cast back to plain object before writing.
    for col in FEATURES_CATEGORICAL:
        out[col] = out[col].astype(str)
    out.to_parquet(PREDICTIONS_PATH, index=False)
    final.save_model(MODEL_PATH)
    print(f"[stuff_baseline] wrote {len(out):,} FS+CH rows to {PREDICTIONS_PATH}")
    print(f"[stuff_baseline] wrote final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
