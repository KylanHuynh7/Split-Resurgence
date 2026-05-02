# The Splitter Analytics Gap

*Inspired by Dave Roberts' 2025 comments to Jeff Passan. Draft v1 — feedback welcome.*

---

## 1. The question

> *"I don't think the analytics has caught up to the split. It doesn't register as well with models. It's like a slow fastball. The split is so unique and it doesn't take into account a hitter's approach against it."*
> — Dave Roberts, LA Dodgers manager, to Jeff Passan, 2025

That's a precise, falsifiable claim from a manager whose pitching staff is built around the pitch. It deserves a precise answer. The question this paper takes on is one part of it:

> *What physical and contextual features of a splitter predict its resistance to pitch-grading models, and does NPB development add explanatory power once those features are controlled for?*

This is a deliberate reframing of the more obvious version — *"do Japanese-developed splitters beat the model more than domestic splitters?"* That's the question every fan asks while watching Yoshinobu Yamamoto, but it can't be answered cleanly. With only five NPB-developed pitchers in MLB throwing 100-plus splitters in any recent season, a direct group comparison runs into n=5, posting-system selection bias, and the awkward fact that Kevin Gausman exists. The regression framing absorbs nationality as one feature among many and lets the statistical machinery weigh it appropriately, which in this dataset turns out to be: not very heavily, and not for the reasons you might assume.

The splitter is having a moment. League-wide usage rose from 1.6% of pitches in 2021 to 3.3% in 2025 — the highest rate since pitch tracking began — and reached 6.6–6.8% in the 2025 postseason, where batters hit .154 / .206 / .250 against it, the worst slash line for any pitch type in October. The resurgence is being driven primarily by NPB-trained pitchers, in a league that mostly stopped throwing the pitch in the early 2000s on injury concerns. Roberts' claim that models haven't caught up is testable. Here's what happens when you actually test it.

## 2. Background

For roughly two decades, MLB organizations actively discouraged the splitter on the theory that it caused elbow injuries. The historical evidence for that claim is thin and Roberts dismisses it directly in the same conversation. We're not testing the injury question — that requires UCL-strain data we don't have, and the answer wouldn't change the pitch-grading question. What matters here is that for twenty years a generation of MLB pitchers grew up not throwing it, while in NPB it remained a standard-issue weapon. Yamamoto, Shohei Ohtani, Shota Imanaga, Roki Sasaki, and Kodai Senga all developed it through years of professional reps in Japan. They've now collectively brought it back to MLB, and the resurgence has expanded as domestic pitchers — Kevin Gausman first, then Nathan Eovaldi, Paul Skenes, others — adopted or rediscovered the pitch.

Roberts' claim has three sub-parts that are worth disentangling. **First**: that run-expectancy models systematically undervalue the pitch — that there's a gap between what the model says a splitter is worth and what it's actually worth. **Second**: that the pitch "is like a slow fastball" — meaning, presumably, that its physical signature deceives both the model and the hitter for longer than other off-speed pitches. **Third**: that current models don't account for hitter cognition, the part where the batter "doesn't know what's going on up here."

This paper tests claim 1 directly via a residuals analysis on a physics-only run-value model. It addresses claim 2 partially via tunneling figures and an explicit velocity-differential feature in the model. It does not test claim 3 — that requires bat-tracking or hitter biometric data, both of which exist in 2024+ but weren't incorporated in v1.

The rest of the paper goes through the data and methods (Section 3), the findings (Section 4), what the paper doesn't establish (Section 5), and a section laying out concretely what evidence would change my mind (Section 6).

## 3. Data and methods

### 3.1 Data

The analysis uses every regular-season and postseason pitch tracked by Statcast from 2021 through 2025 — 3,614,368 pitches in total, pulled via `pybaseball` and cached locally. The focal roster is ten pitchers, hand-curated to span the NPB-developed and domestic-developed populations: Yamamoto, Ohtani (pitching seasons 2018–2023 only), Imanaga, Sasaki, and Senga on the NPB side; Gausman, Eovaldi, Skenes, Taijuan Walker, and Trey Yesavage on the domestic side. NPB years and MLB debut were filled in by hand from Baseball Reference and NPB Wikipedia. Every other pitcher in the regression has their MLB debut year resolved via the Chadwick Bureau register.

Two additional data choices matter. First, the analytical universe is FS plus CH — splitters and changeups — because the changeup is the natural comparison off-speed pitch and the most plausible answer to *"is this splitter-specific or just an off-speed thing?"* Second, the modeling table also includes fastball pitches (FF / SI / FC) so the unified XGBoost baseline learns the run-value mapping across all five pitch types, not just within splitters.

### 3.2 The physical-features baseline

The first model in the pipeline is a single XGBoost regressor predicting per-pitch `delta_run_exp` — Statcast's per-pitch run value, base-out aware — from physical features only. The features are velocity, induced vertical break, horizontal break, release point (x and z), release extension, the initial velocity vector and acceleration components Statcast records, spin axis and rate, plate location, count state, batter and pitcher handedness, and two engineered features: each pitch's velocity differential and IVB differential against that pitcher's primary fastball that season. Pitcher and batter identity are deliberately excluded.

The model is cross-validated by `(pitcher, season)` GroupKFold. Each fold trains on roughly 80% of the pitcher-seasons in the dataset and predicts the held-out 20%, so every pitch in the analytical set ends up with an out-of-sample prediction made by a model that didn't see that pitcher-season's data. Hyperparameters are deliberately modest: depth 6, learning rate 0.05, subsample 0.7, 400 trees. The reason for under-tuning is that the residuals — actual minus predicted run value — are the dependent variable for the regression downstream. An over-tuned baseline produces residuals that are pure noise; an under-tuned one produces residuals that contain the pitcher-specific signal we're trying to explain.

OOS RMSE is 0.224 against the per-pitch dependent variable; mean residual is ≈ 0 (the model is unbiased). For context, per-pitch `delta_run_exp` is mostly noise — most pitches end in a ball or called strike with run values near zero, with a heavy tail of hard-contact events. RMSE around 0.22 is in the expected range for this kind of per-pitch prediction.

The per-pitch residuals are aggregated to the (pitcher, season, pitch_type) grain as `model_resistance = -100 × mean(residual)`. The sign flip matches the convention for run-value-per-100: positive numbers mean the pitcher *outperformed* what the physics-only model predicted, and the units are run value per 100 pitches.

### 3.3 The mixed-effects regression

The regression is restricted to pitcher-seasons with at least 100 pitches of the pitch type being modeled. That leaves 239 FS pitcher-seasons across 111 unique pitchers (8 of those rows from the 5 focal NPB-developed pitchers) and 1,166 CH pitcher-seasons across 520 pitchers. The model is a linear mixed-effects regression with a random pitcher intercept:

```
model_resistance ~ npb_years + mlb_seasons_in + fastball_velo_mean
                   + usage_share + is_novelty_year + (1 | pitcher)
```

`npb_years` is years in NPB before MLB debut, hand-coded for the focal NPB pitchers and 0 for everyone else. `mlb_seasons_in` is years since MLB debut (so 0 in the rookie season). `fastball_velo_mean` is the pitcher's primary-fastball average velocity that season. `usage_share` is the share of FS or CH within that pitcher's tracked pitches in our data. `is_novelty_year` is the indicator for `mlb_seasons_in == 0`. The random pitcher intercept absorbs within-pitcher correlation across seasons — Yamamoto's 2024 and 2025 splitters aren't independent observations.

Statsmodels' `MixedLM` with the L-BFGS-B optimizer hits a singular matrix on the FS dataset because many random-effect groups have only one observation; the implementation falls back through bfgs / cg / powell until one converges. The first fallback succeeds.

### 3.4 The honesty layer

There is one coefficient in this model that the asymptotic confidence intervals can't be trusted on, and that's `npb_years`. The variation in that predictor comes from 8 rows tied to 5 underlying pitchers; the asymptotic SE machinery treats those 8 rows as if they carried 8 independent observations of information, which they don't. To get an honest interval on that one coefficient, the procedure is a cluster bootstrap: resample the 111 pitchers with replacement (so each resample is 111 pitchers, with duplicates), refit the same MixedLM on the resampled dataset, record the `npb_years` coefficient, repeat 2000 times. The 2.5th and 97.5th percentiles of the resulting estimates form a nonparametric 95% CI that doesn't lean on any asymptotic approximation — it directly measures the sampling variability under the actual data structure.

The bootstrap is *only* applied to the `npb_years` coefficient. For the other coefficients, the asymptotic intervals are reported as-is, because their variation comes from hundreds of pitcher-seasons and the asymptotic approximation holds.

## 4. Findings

### 4.1 The headline: the NPB-development effect is not detectable

The point estimate on `npb_years` is +0.10 RV/100 per NPB year. The asymptotic 95% CI is [−0.10, +0.30]. The cluster-bootstrap 95% CI is [−0.07, +0.21], with 1,964 of 2,000 resamples converging.

Both intervals comfortably include zero. The data does not support a detectable NPB-development effect on splitter model-resistance.

It's worth noticing that the bootstrap CI is *tighter* than the asymptotic CI here, not wider. That's the opposite of what's usually true with small groups — typically the asymptotic interval is over-confident and the bootstrap pulls it apart. In this dataset the asymptotic interval is, if anything, slightly under-confident. So the underpowered conclusion isn't an artifact of optimistic standard errors; the data really does put the effect within shouting distance of zero from both directions.

This is not the result the original framing of the project was looking for. It's worth being clear about what it does and doesn't say. It doesn't say there's no NPB effect. It says that *with eight pitcher-seasons of NPB-developed splitters across five distinct pitchers in our dataset*, there isn't enough signal to distinguish a real effect from zero. A larger NPB sample — Sasaki and Yamamoto are still adding pitcher-seasons, and Roki Sasaki only has 271 splitters in our 2025 data — could move the answer either direction.

→ **Insert Figure 1: residuals landscape.** The clearest way to see the null is graphically: each dot is one FS pitcher-season, with run value per 100 on the x-axis and model resistance (the residual) on the y-axis. The y = 0 line is "physics-only model exactly predicts this pitcher's run value." If the NPB-developed pitchers were systematically beating physics, their dots would cluster up and to the right. They don't.

### 4.2 The splitter-specific patterns the regression *does* find

Where the regression has hundreds of pitcher-seasons of variation to work with — on the predictors that aren't tied to five pitchers — three patterns are clear and informative.

**`usage_share` is the only coefficient reaching p < 0.05 on FS, and it's roughly 2× the size of the same coefficient on CH.** On splitters, each unit of FS-share-of-arsenal predicts about +2.8 RV/100 of model-beating performance (p = 0.04). On changeups, the same coefficient is +1.5 (p = 0.13). The honest read is *splitters reward arsenal-specialization more than changeups do*. Pitchers who commit to throwing the splitter more frequently throw a meaningfully better splitter, in a way that doesn't carry over to the changeup. This is consistent with Roberts' framing — there's something specifically about splitter-throwing as a craft that's different from generic off-speed throwing — but the data here can't tell us *why* (more reps, better tunneling, batter never sees it twice in the right context, etc.). It can only tell us the pattern is real and is FS-specific within this comparison.

**The novelty-year coefficient is negative on splitters and roughly four times stronger than on changeups.** Rookie-year pitchers underperform what the physics-only model predicts on their splitters by about 0.66 RV/100 (p = 0.09). The same effect on changeups is −0.16 (p = 0.38). This contradicts the stylized "Japanese debutants outperform peripherals" intuition that floats around discussion of Yamamoto and Imanaga, but it doesn't contradict the case-study evidence — it's saying the *population* of debut-year FS throwers tends to leave run value on the table relative to model expectations, even though specific high-profile rookies sometimes don't. The population includes plenty of domestic rookies whose splitter is a work in progress, and those drag the coefficient negative.

**The pitcher random-intercept variance is about 2.4× larger on FS than on CH** — 0.40 versus 0.17 in the variance-component estimates. That number describes how much pitcher-to-pitcher variation in model resistance there is *after* you've stripped out everything the physical features explain. Splitters, by this measure, carry roughly two and a half times more pitcher-specific signal that physics alone cannot capture, compared to changeups. This may be the single most important finding in the paper. It's not a claim about NPB development; it's a claim about the splitter as a pitch type. Consistent with Roberts: the splitter has more pitcher-specific structure than the comparison off-speed pitch carries.

### 4.3 The structural finding from the comparison-pitch check

The build plan called for repeating the regression on changeups and asking *does NPB development predict CH residuals too — and if so, the effect isn't splitter-specific.* The implementation was straightforward; the result was unexpected.

**None of the five focal NPB-developed pitchers cleared the 100-changeup threshold in any of the five seasons.** Yamamoto, Ohtani, Imanaga, Sasaki, and Senga all throw a splitter or fork as their primary off-speed weapon. They throw changeups in trace amounts or not at all. The `npb_years` predictor in the CH regression therefore has zero variation — every CH pitcher-season has `npb_years = 0` — and the regression mechanically drops it.

This is a structural finding, not a statistical one. It tells us something Roberts' framing implies but doesn't say outright: *the NPB pitchers' deception advantage, whatever it is, isn't a generalized off-speed deception that happens to show up most clearly on the splitter*. It's specific to that arsenal. The comparison-pitch check can't directly test the NPB hypothesis because the NPB-developed pitchers in our data are splitter specialists by arsenal construction. That, by itself, is a piece of evidence supporting Roberts' first claim — the pitch is unique enough that the people who develop it specialize in it.

### 4.4 Three case studies

The case studies do work the regression can't. With 5 NPB pitchers and 5 domestic focal pitchers, group-level inference is tight; pitcher-level description is wide open.

→ **Insert Figure 2: tunneling overlays for Yamamoto, Gausman, Skenes.** Top row: release-point density. Bottom row: plate location. Blue is fastball; orange is splitter.

**Yoshinobu Yamamoto.** Model resistance roughly zero in both 2024 (−0.06, n = 391) and 2025 (+0.21, n = 830). Yamamoto's splitter is so physically dominant that a physics-only model already predicts its run value correctly. There's no model-resistance gap to explain — the pitch beats hitters because of *what* it is, not because of some unmeasured pitcher-specific quality on top of that. **A surprise in the figure**: his release-point cloud has two distinct clusters, one at roughly release_pos_x ≈ −1.4 ft and one at ≈ −2.0 ft. He changed arm slot somewhere between 2024 and 2025. We haven't decomposed his residuals by cluster yet — that's a clean follow-up, listed in Section 6 — but the visual is striking enough to flag now.

**Kevin Gausman.** Model resistance hovers around zero across all five seasons in our data: +1.33 (2021), −0.06 (2022), −0.67 (2023), +0.34 (2024), +0.12 (2025). The "Gausman exception" hypothesis from the project plan was *does his domestic-trained splitter look more like the NPB cluster than other domestic splitters?* Read against the data: not really. His trajectory is in line with the median domestic FS thrower over this window. Gausman's reputation as a splitter savant comes from sustained excellence at the *run-value* level, not from beating the physics-only baseline. His pitch is great because of what it is — measurably, in the physical features — not because of an additional unmeasured edge.

**Paul Skenes.** Model resistance was +1.44 in 2024 (n = 603) and −1.15 in 2025 (n = 405). That is the largest year-over-year drop in any focal pitcher's model resistance in the dataset. It's the documented mid-2025 splinker collapse, showing up in the residual data with no prompting. Three plausible stories for it: hitters caught up to a pitch they hadn't seen at 94 mph before; Skenes made a mechanical change that hurt the pitch; or the pitch's effectiveness was always partially a velocity-outlier illusion that the model under-priced in its first year and corrected for in its second. The current data can't separate these, but it can rule out *no change* — the residual move is real. The Skenes case is also why the build plan called out the classification problem early: Statcast's FS tag bundles his 94 mph splinker with traditional 86 mph splitters, and the pitch's behavior is plausibly different enough that pooling them is a modeling choice with consequences. The velocity-band sub-clustering in the codebase puts him in the "splinker" cluster and Yamamoto / Imanaga / Gausman in the "traditional" cluster, but the regression in this v1 doesn't stratify on the cluster. That's also a follow-up.

→ **Insert Figure 3: case-study trajectories and against-handedness splits.** The Skenes 2024 → 2025 drop is the most prominent feature. Yamamoto and Gausman are nearly flat at zero. The handedness panel is noisy — there's no clean systematic LHB / RHB pattern across the three pitchers, and the cell sizes get small once you split by season and stand. Worth showing for honesty; not worth claiming a pattern from.

### 4.5 Walker as the cautionary case

One footnote worth promoting. Taijuan Walker's 2024 FS pitcher-season has the most extreme negative residual in the entire dataset — model resistance of −5.49 RV/100 across 318 splitters. The project plan, written before any modeling was done, flagged him as the worst splitter in baseball by run value in 2024 (−16 total). The model's residual analysis reproduces that public flag without being told to. That's a sanity check on the residual decomposition: when the data says a pitch was bad, the model agrees, and when the project plan said a pitcher was a cautionary case based on outside reporting, the residuals point at the same season. The model isn't off in some idiosyncratic direction.

## 5. What this paper doesn't establish

This is not a causal claim. The regression is associational. Even if `npb_years` had come in clearly nonzero, "NPB development causes model resistance" isn't a conclusion the design supports — the n=5 selection bias problem (only the elite NPB pitchers post and sign in MLB) is unfixable with available data. What the regression can show is association after controls, and in this dataset there isn't enough signal to do even that for the NPB term.

This paper does not test Roberts' third claim — the cognitive disruption piece, the "you don't know what's going on up here." That requires hitter-side data: bat-tracking metrics, gaze data, or swing decisions modeled in detail. Statcast made bat-tracking public in 2024; a v2 of this work could plausibly partially address the cognition question with that data. We didn't.

The NPB-side data extension was cut from v1 deliberately. Translating NPB Tracker / DELTA pitch data to MLB equivalents would let us model splitter quality in Japan and ask whether NPB peripherals predict MLB outcomes. The data quality on the NPB side is significantly worse than Statcast's, and would dominate the error budget. That's a separate paper.

There's a left-censoring issue for older pitchers in the regression — for a pitcher who debuted in MLB before 2021, `mlb_seasons_in` is correctly computed via the Chadwick lookup, but their career-history features (e.g., prior-decade splitter usage) aren't observed in our 5-season window. The regression's controls partly handle this, but the population we're estimating effects across isn't fully homogeneous on tenure.

The classification problem is partially addressed and partially not. The codebase sub-clusters FS-tagged pitches into "traditional splitter" and "splinker" classes via a 2-component Gaussian mixture on velocity and velocity-differential. The clusters separate cleanly: traditional ≈ 85.7 mph with a −9.8 mph fastball gap, splinker ≈ 87.5 mph with a −5.9 mph gap. The regression in v1 doesn't stratify on the cluster — every FS pitcher-season is pooled regardless of which sub-class their splitter belongs to. Stratifying is a v2 sensitivity analysis; if Skenes-class pitches behave systematically differently, the pooled regression is averaging over a heterogeneous population.

## 6. What would change my mind

This is the section I'd want a peer reviewer to read most carefully. Each item is a concrete, falsifiable update condition.

**The NPB-development effect is real after all** if (a) extending the dataset through the 2026 season — by which point Sasaki, Yamamoto, and any new NPB arrivals will have added pitcher-seasons — produces a bootstrap CI on `npb_years` that excludes zero, or (b) refitting with a Bayesian hierarchical model under a weakly-informative prior produces a posterior with more than 95% mass above zero. Either of those would be a reason to update toward "yes, NPB development matters, the v1 sample was underpowered."

**The usage-share effect is splitter-specific** if replicating with a third comparison off-speed pitch — curveball or sweeper would be the natural choices — shows the FS coefficient still about 2× the CH and the third pitch's coefficients. If the FS effect is just "off-speed pitches reward specialization in general," it would replicate on every off-speed type at roughly the same magnitude. If only FS shows the doubled effect, the splitter-specific reading is supported.

**The Skenes 2025 collapse is hitter adaptation, not mechanical change** if a within-season decomposition of his 2025 splitters shows the residual worsening monotonically across the season as opposed to dropping at a specific date. Monotonic worsening is consistent with hitters catching up; a sharp date-localized drop is consistent with a mechanical change that broke the pitch.

**The pitcher-specific variance gap between FS and CH is real signal, not noise** if a sensitivity analysis with bootstrap CIs on the variance-component estimates puts both FS and CH variances clearly nonzero, and the FS variance is strictly larger than CH in essentially every bootstrap draw. That would convert the headline 2.4× ratio from a point estimate into a probabilistic claim.

**The Yamamoto arm-slot shift matters analytically** if splitting his 2024 and 2025 pitches by release-point cluster and re-running the residual analysis shows different model resistance per cluster. This is a half-day of additional work and the most concrete unfinished thread in the project. The shift might be invisible in the residuals, in which case the slot change was a delivery refinement that didn't move the model gap. Or it might show up as one cluster at +0.5 and another at −0.5, in which case Yamamoto has been silently running two different splitters and the apparent stability of his model resistance was an averaging artifact. Either answer is informative.

**The classification critique is structural, not cosmetic** if running the regression separately on the velocity-band traditional cluster vs. the splinker cluster produces meaningfully different `usage_share` and `is_novelty_year` coefficients. If the splinker-cluster regression looks just like the traditional-cluster regression, the FS tag is bundling fine and pooling was a reasonable choice. If they differ, then Statcast's pitch classification is suppressing real signal in any pooled analysis, and the headline numbers in this paper need to be read with that caveat in mind.

## 7. Reproducibility

The full project lives at https://github.com/KylanHuynh7/Split-Resurgence on the `claude/heuristic-pare-4b0721` branch. Six commits, each with a self-contained scope: data layer, modeling layer, modeling outputs, case-study figures, changeup comparison, writeup outline.

Everything is reproducible from a fresh clone with `uv` installed:

```
git clone https://github.com/KylanHuynh7/Split-Resurgence.git
cd Split-Resurgence
git checkout claude/heuristic-pare-4b0721
uv sync --extra modeling
uv run python -m src.run_smoke
uv run python -m src.data.pull_statcast --all          # ~30 min, network bound
uv run python -m src.models.stuff_baseline             # ~10 min, XGBoost CV
uv run python -m src.models.residual_analysis
uv run python -m src.models.mixed_effects --pitch-type FS --bootstrap 2000  # ~30 min
uv run python -m src.models.mixed_effects --pitch-type CH
uv run python -m src.models.compare_pitch_types
uv run python -m src.viz.residuals_plot
uv run python -m src.viz.tunnel_plot
uv run python -m src.viz.case_studies
```

All small derived outputs — the trained XGBoost model, the residuals parquet, the regression coefficients, the bootstrap distribution, the comparison summary, the three figures — are committed to the repo. The only large file that's not committed is the per-pitch predictions parquet (~54MB), which `stuff_baseline` regenerates in about ten minutes from the cached raw data and the trained model.

The README under [README.md](https://github.com/KylanHuynh7/Split-Resurgence/blob/claude/heuristic-pare-4b0721/README.md) documents data provenance and the pipeline at a higher level; the docstring at the top of every `src/` module describes that module's role specifically. The full project plan, including the mentor revisions that reframed the central question, is preserved in chat history; the relevant subset is summarized in Sections 1–3 above.

---

*Comments and corrections: open an issue on the repo or DM me. Particular interest in pushback on the usage-share interpretation in Section 4.2 — the 2× FS-vs-CH gap is the most aggressive claim in the paper and the most natural place to be skeptical.*
