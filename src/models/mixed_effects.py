"""Step 9: mixed-effects regression of FS model-resistance + bootstrap CI on NPB.

Model:
    model_resistance ~ npb_years + mlb_seasons_in + fastball_velo_mean
                       + usage_share + is_novelty_year
                       + (1 | pitcher)

Restricted to FS pitcher-seasons with 100+ pitches. All pitchers (focal AND
non-focal) are included so the regression has hundreds of pitcher-seasons of
information for the controls. NPB-developed pitchers contribute via positive
`npb_years`; everyone else has `npb_years = 0`.

Sensitivity analysis: cluster bootstrap by pitcher (2000 resamples) on the
`npb_years` coefficient. Resampling at the pitcher level gives a nonparametric
CI that doesn't depend on the asymptotic approximations MixedLM uses, which
are the things that get unreliable when the predictor's variation comes from
~5 distinct pitchers.

Run:
    uv run python -m src.models.mixed_effects
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

RESIDUALS_PATH = PROCESSED_DIR / "pitcher_season_residuals.parquet"


def coeffs_path(pitch_type: str) -> Path:
    return PROCESSED_DIR / f"mixed_effects_coeffs_{pitch_type}.csv"


def bootstrap_path(pitch_type: str) -> Path:
    return PROCESSED_DIR / f"bootstrap_npb_coefficient_{pitch_type}.csv"


def summary_path(pitch_type: str) -> Path:
    return REPORTS_DIR / f"mixed_effects_summary_{pitch_type}.txt"


PREDICTORS = [
    "npb_years", "mlb_seasons_in", "fastball_velo_mean",
    "usage_share", "is_novelty_year",
]


def build_formula(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Drop predictors with no variation (e.g. npb_years on CH where none of
    the focal NPB pitchers crossed the 100-pitch threshold)."""
    used = [p for p in PREDICTORS if df[p].nunique() > 1]
    return "model_resistance ~ " + " + ".join(used), used


def load_pitch_table(pitch_type: str) -> pd.DataFrame:
    """Rows for `pitch_type` only, drop rows missing any covariate the formula needs."""
    df = pd.read_parquet(RESIDUALS_PATH)
    df = df[df["pitch_type"] == pitch_type].copy()
    needed = [
        "model_resistance", "npb_years", "mlb_seasons_in",
        "fastball_velo_mean", "is_novelty_year",
    ]
    df = df.dropna(subset=needed)
    # usage_share isn't in the residuals table — it lives in aggregate.py's
    # output. Recompute here from n_pitches per pitcher-season.
    season_totals = (
        df.groupby(["pitcher", "game_year"])["n_pitches"].sum().reset_index(
            name="total_fs_only"
        )
    )
    # We don't have total pitches in this file, so use a proxy: FS share among
    # the pitcher's tracked pitch types in our data. For non-focal pitchers
    # this is approximate, but consistent.
    df = df.merge(season_totals, on=["pitcher", "game_year"])
    df["usage_share"] = df["n_pitches"] / df["total_fs_only"]  # = 1 here, see below
    # Better: pull total from the predictions parquet.
    pred = pd.read_parquet(PROCESSED_DIR / "pitches_with_predictions.parquet")
    pitcher_season_totals = (
        pred.groupby(["pitcher", "game_year"]).size().reset_index(name="total_fs_ch")
    )
    df = df.drop(columns=["usage_share", "total_fs_only"]).merge(
        pitcher_season_totals, on=["pitcher", "game_year"]
    )
    df["usage_share"] = df["n_pitches"] / df["total_fs_ch"]

    # MixedLM needs numeric ints/floats — coerce nullable Int64 etc.
    df["npb_years"] = df["npb_years"].astype(float)
    df["mlb_seasons_in"] = df["mlb_seasons_in"].astype(float)
    df["is_novelty_year"] = df["is_novelty_year"].astype(float)
    df["fastball_velo_mean"] = df["fastball_velo_mean"].astype(float)
    df["pitcher"] = df["pitcher"].astype(int).astype(str)
    return df


def fit_mixedlm(df: pd.DataFrame, formula: str):
    """Fit with optimizer fallback chain.

    L-BFGS-B can hit a singular matrix in the score computation when many
    random-effect groups have only one observation (the random-intercept
    variance is then near-degenerate). Chain through more robust optimizers
    that don't rely on analytic gradients.
    """
    model = smf.mixedlm(formula, df, groups=df["pitcher"])
    last_err: Exception | None = None
    for method in ("lbfgs", "bfgs", "cg", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return model.fit(method=method)
        except Exception as err:
            last_err = err
            print(f"[mixed_effects]   {method} failed: {type(err).__name__}: {err}")
    raise RuntimeError("all MixedLM optimizers failed") from last_err


def cluster_bootstrap_npb(
    df: pd.DataFrame, formula: str, n_resamples: int = 2000, seed: int = 0
) -> np.ndarray:
    """Resample pitchers (with replacement), refit MixedLM, collect npb coeff."""
    rng = np.random.default_rng(seed)
    pitchers = df["pitcher"].unique()
    n_pitchers = len(pitchers)
    pitcher_to_rows = {p: df[df["pitcher"] == p] for p in pitchers}

    estimates = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        sampled = rng.choice(pitchers, size=n_pitchers, replace=True)
        # Re-key duplicates so MixedLM treats them as distinct clusters.
        parts = []
        for k, p in enumerate(sampled):
            chunk = pitcher_to_rows[p].copy()
            chunk["pitcher"] = f"{p}__b{k}"
            parts.append(chunk)
        boot = pd.concat(parts, ignore_index=True)
        # If the resample happens to contain zero NPB rows, the coefficient
        # is unidentified; record NaN and move on.
        if (boot["npb_years"] > 0).sum() == 0:
            continue
        model = smf.mixedlm(formula, boot, groups=boot["pitcher"])
        for method in ("lbfgs", "bfgs", "cg", "powell"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = model.fit(method=method)
                estimates[i] = fit.params.get("npb_years", np.nan)
                break
            except Exception:
                continue
        if (i + 1) % 200 == 0:
            ok = np.sum(~np.isnan(estimates[: i + 1]))
            print(f"[mixed_effects] bootstrap {i+1}/{n_resamples} ({ok} converged)")
    return estimates


def main(pitch_type: str = "FS", n_bootstrap: int = 2000) -> None:
    print(f"[mixed_effects] loading {pitch_type} pitcher-season residuals...")
    df = load_pitch_table(pitch_type)
    print(
        f"[mixed_effects]   {len(df)} rows, "
        f"{df['pitcher'].nunique()} pitchers, "
        f"{(df['npb_years'] > 0).sum()} NPB-flagged rows"
    )

    formula, used_predictors = build_formula(df)
    dropped = [p for p in PREDICTORS if p not in used_predictors]
    if dropped:
        print(f"[mixed_effects] dropping zero-variation predictors: {dropped}")
    print(f"[mixed_effects] formula: {formula}")
    print("[mixed_effects] fitting primary MixedLM...")
    fit = fit_mixedlm(df, formula)
    print(fit.summary())

    coeffs = pd.DataFrame(
        {
            "term": fit.params.index,
            "estimate": fit.params.values,
            "se": fit.bse.values,
            "p_value": fit.pvalues.values,
        }
    )
    coeffs["ci_low"] = coeffs["estimate"] - 1.96 * coeffs["se"]
    coeffs["ci_high"] = coeffs["estimate"] + 1.96 * coeffs["se"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    coeffs_p = coeffs_path(pitch_type)
    coeffs.to_csv(coeffs_p, index=False)
    print(f"[mixed_effects] wrote coeffs to {coeffs_p}")

    if "npb_years" in used_predictors:
        print(
            f"[mixed_effects] cluster bootstrap on npb_years ({n_bootstrap} resamples)..."
        )
        boot = cluster_bootstrap_npb(df, formula, n_resamples=n_bootstrap)
        boot_clean = boot[~np.isnan(boot)]
        if len(boot_clean) == 0:
            print("[mixed_effects] WARNING: no bootstrap fits converged.")
            boot_lo = boot_hi = float("nan")
        else:
            boot_lo, boot_hi = np.percentile(boot_clean, [2.5, 97.5])
            print(
                f"[mixed_effects]   bootstrap 95% CI: [{boot_lo:.3f}, {boot_hi:.3f}] "
                f"(from {len(boot_clean)}/{n_bootstrap} convergent fits)"
            )
        pd.DataFrame({"estimate": boot}).to_csv(bootstrap_path(pitch_type), index=False)
    else:
        print(
            "[mixed_effects] skipping npb_years bootstrap — predictor was "
            "dropped (no variation in this dataset)"
        )
        boot_clean = np.array([])
        boot_lo = boot_hi = float("nan")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_p = summary_path(pitch_type)
    with open(summary_p, "w") as f:
        f.write(f"=== MixedLM regression on {pitch_type} pitcher-seasons ===\n\n")
        if dropped:
            f.write(f"NOTE: predictors dropped (no variation): {dropped}\n\n")
        f.write(str(fit.summary()))
        if "npb_years" in used_predictors:
            npb_row = coeffs[coeffs["term"] == "npb_years"].iloc[0]
            f.write("\n\n=== Sensitivity check on npb_years coefficient ===\n")
            f.write(f"point estimate (MixedLM): {npb_row['estimate']:.4f}\n")
            f.write(
                f"asymptotic 95% CI:        "
                f"[{npb_row['ci_low']:.4f}, {npb_row['ci_high']:.4f}]\n"
            )
            f.write(
                f"cluster-bootstrap 95% CI: [{boot_lo:.4f}, {boot_hi:.4f}] "
                f"(n={len(boot_clean)} convergent of {n_bootstrap})\n"
            )
            f.write(
                "\nThe bootstrap CI is the honest interval — the asymptotic CI assumes\n"
                "more variation in `npb_years` than the data actually has, since the\n"
                "predictor's nonzero values come from ~5 NPB pitchers.\n"
            )
        else:
            f.write(
                "\n\n=== npb_years was dropped from this model ===\n"
                "None of the focal NPB-developed pitchers (Yamamoto, Ohtani,\n"
                f"Imanaga, Sasaki, Senga) reached the 100-{pitch_type} threshold\n"
                "in any season. The NPB-development hypothesis cannot be tested\n"
                "directly on this pitch type using the focal roster.\n"
            )
    print(f"[mixed_effects] summary written to {summary_p}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--pitch-type",
        type=str,
        default="FS",
        choices=["FS", "CH"],
        help="Pitch type to fit on (default FS).",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of cluster-bootstrap resamples (default 2000; use ~50 for smoke).",
    )
    args = p.parse_args()
    main(pitch_type=args.pitch_type, n_bootstrap=args.bootstrap)
