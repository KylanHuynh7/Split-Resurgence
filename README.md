# Split-Resurgence

**Do run-value models undervalue the splitter, and does NPB development explain
which splitters beat them?**

Dave Roberts said in 2025 that "the analytics hasn't caught up to the split."
This project tests that claim with every Statcast pitch from 2021 to 2025. It
uses a physics-only XGBoost run-value model, measures how far each pitcher's
splitter beats or trails that model, and runs a mixed-effects regression with
a cluster bootstrap on the five-pitcher NPB term. Changeups serve as the control.

**Paper:** [`paper/split_resurgence.pdf`](paper/split_resurgence.pdf) ·
**Design doc:** [`docs/design.md`](docs/design.md)

## Findings

| | Splitter (FS) | Changeup (CH) |
|---|---|---|
| NPB years (RV/100 per year) | +0.10, not significant; bootstrap 95% CI includes 0 | can't be estimated (no NPB pitcher throws 100 CH) |
| Pitcher-specific variance | **0.35** | 0.14 |
| Debut-season effect | **−0.87** (p = .03) | −0.13 (n.s.) |
| Usage share | +3.15 (p = .002) | +2.34 (p < .001) |

- **There is no detectable NPB-development effect.** Five NPB pitchers aren't enough to tell it from zero.
- **Splitters carry about 2.4× more pitcher-specific signal** than changeups once physics is accounted for.
- **Debut-season pitchers underperform the model on splitters**, with no matching effect on changeups.
- **The usage reward is general to off-speed pitches**, not specific to splitters.

## Repository

```
src/
├── data/        pull_statcast · pitcher_metadata · build_features · aggregate
├── analysis/    splinker_subcluster (GMM: traditional splitter vs splinker)
├── models/      stuff_baseline · residual_analysis · mixed_effects · compare_pitch_types
├── viz/         residuals_plot · tunnel_plot · case_studies · style
└── run_smoke.py
paper/           LaTeX source and compiled PDF
docs/design.md   pipeline design, data contracts, decisions, known issues
reports/         regression summaries, figures, original markdown draft
data/processed/  small derived outputs (committed); raw pulls are regenerated
```

## Reproducing

```sh
uv sync --extra modeling
uv run python -m src.run_smoke                                  # ~30 s sanity check
uv run python -m src.data.pull_statcast --all                   # ~30 min, network-bound
uv run python -m src.models.stuff_baseline                      # ~10 min
uv run python -m src.models.residual_analysis
uv run python -m src.models.mixed_effects --pitch-type FS --bootstrap 2000   # ~30 min
uv run python -m src.models.mixed_effects --pitch-type CH
uv run python -m src.models.compare_pitch_types
uv run python -m src.viz.residuals_plot
uv run python -m src.viz.tunnel_plot
uv run python -m src.viz.case_studies
tectonic paper/split_resurgence.tex                             # rebuild the PDF
```

The repo commits every small output the paper depends on: the trained model,
residuals, coefficients, bootstrap draws and figures. A fresh clone can rerun
the regressions and rebuild the paper without the ~480 MB raw pull. See
[`docs/design.md` §4](docs/design.md#4-artifacts-and-whats-committed) for what
is regenerated and how.

## Data notes

- Statcast `pfx_z` is in feet. Multiply by 12 for inches.
- The FS tag mixes traditional splitters with splinkers such as Skenes' pitch.
  `splinker_subcluster` separates them, but the v1 regression pools them.
- `delta_run_exp` accounts for base-out state but not batter quality.
