# Writeup outline — Split-Resurgence

This is a skeleton. Each section lists the bullets to hit, the specific
numbers/figures to cite, and tone notes. Replace each bullet block with prose
in your voice. Target: 8–10 pages of finished writing.

---

## Section 1: Lede (~½ page)

**Job**: hook with the Roberts quote, then state the question in one sentence.

- Open with the Roberts / Passan quote in full ("I don't think the analytics has
  caught up to the split…"). Cite as: Roberts to Jeff Passan, ESPN, 2025.
- One sentence translating it into a research question — the version from the
  revised plan, not the original NPB-vs-domestic framing:
  *"What physical and contextual features of a splitter predict its
  resistance to pitch-grading models, and does NPB development add
  explanatory power once those features are controlled for?"*
- One paragraph framing why this matters now: splitter usage went from 1.6%
  in 2021 to 3.3% in 2025 (postseason 6.6–6.8%); 2025 postseason batters hit
  .154/.206/.250 against splitters; the resurgence is real and is being
  driven primarily by NPB-trained pitchers.

**Tone**: confident, journalistic. Don't editorialize on the result yet.

---

## Section 2: Background (~1 page)

**Job**: catch a non-baseball reader up enough to read the rest, but don't
restate Wikipedia.

- The historical aversion: splitter discouraged in MLB ~2000s on injury
  concerns; flag that Roberts disagrees with the injury claim, but explicitly
  say *we are not testing the injury question* — that's a separate paper.
- Why NPB matters: standard-issue pitch in Japan; the focal NPB-developed
  pitchers (Yamamoto, Ohtani, Imanaga, Sasaki, Senga) had years of
  professional reps before MLB.
- The Roberts claim, decomposed into the three sub-claims from the project
  plan: (1) models undervalue the pitch, (2) "it's like a slow fastball,"
  (3) cognitive disruption isn't captured. State which of these we test
  directly (1 and partially 2 via residuals + tunneling figure) and which
  we don't (3 — the "cognitive" piece would need hitter biometric data we
  don't have).
- One short paragraph on why the original NPB-vs-domestic framing was
  reframed in the revised plan: n=5 in the NPB group, posting-system
  selection bias, the Gausman exception. Reframe as a regression with
  controls, lean on case studies for the NPB pitchers individually.

**Tone**: clear, brief. Reader should leave knowing what we test and what
we don't.

---

## Section 3: Data and methods (~2 pages)

**Job**: enough detail that someone could replicate; not so much that the
narrative drowns.

### 3.1 Data
- Statcast pitch-level data, 2021–2025 (regular + postseason). 3,614,368
  total pitches across the five seasons.
- Pulled via `pybaseball`, cached as parquet, monthly chunked with retry
  on transient parser/HTTP failures.
- Hand-curated focal roster of 10 pitchers (5 NPB-developed, 5 domestic);
  MLB debut years for the rest of the pitcher pool from the Chadwick register.

### 3.2 The physical-features baseline (step 7)
- One XGBoost regressor, single model trained on all of FS, CH, FF, SI, FC
  with `pitch_type` as a categorical feature. Pitcher and batter identity
  deliberately excluded.
- Features used: release_speed, pfx_x/z (movement), release point,
  release_extension, vx0/vy0/vz0/ax/ay/az, spin_axis, release_spin_rate,
  plate location, balls/strikes count, batter and pitcher handedness,
  velo and IVB differential vs each pitcher's primary fastball.
- Cross-validated by `(pitcher, season)` GroupKFold so predictions for any
  pitcher-season come from a model that didn't see it. Modest hyperparameters
  (depth 6, lr 0.05, subsample 0.7) — under-tuned by design so residuals
  carry the signal we want to explain. OOS RMSE = 0.224, mean residual
  ≈ 0 (model unbiased).
- Per-pitch out-of-sample residuals are aggregated to (pitcher, season,
  pitch_type) as `model_resistance = -100 * mean(residual)`. Sign-flipped
  so positive = pitcher beat the physics-only prediction.

### 3.3 The mixed-effects regression (step 9)
- Restrict to pitcher-seasons with ≥100 pitches of the pitch type being
  modeled (FS or CH). 239 FS pitcher-seasons, 1,166 CH pitcher-seasons.
- Random pitcher intercept; fixed effects for `npb_years`, `mlb_seasons_in`,
  `fastball_velo_mean`, `usage_share`, `is_novelty_year`. Optimizer fallback
  chain (lbfgs → bfgs → cg → powell) handles degenerate fits when many
  random-effect groups have only one observation.

### 3.4 The honesty layer
- Cluster bootstrap on pitcher (2000 resamples) for the `npb_years`
  coefficient specifically. Resampling pitchers with replacement, refitting
  each time, and taking percentiles of the resulting estimates gives a
  nonparametric CI that doesn't rely on the asymptotic approximations the
  frequentist mixed-effects machinery uses — those approximations are the
  things that get unreliable when a predictor's variation is tied to ~5
  pitchers.
- Be explicit that this is the *only* coefficient where we don't trust the
  asymptotic CI; for the others we report MixedLM's standard intervals.

**Tone**: technical but plain. Reserve hedging for findings, not method.

---

## Section 4: Findings (~3 pages — load-bearing section)

**Job**: state what we found, in order of importance, with the right amount
of confidence on each.

### 4.1 The headline: NPB development effect is not detectable

- Point estimate: `npb_years` coefficient = +0.10 RV/100 per NPB year.
- Asymptotic 95% CI: [-0.10, +0.30].
- Cluster-bootstrap 95% CI: [-0.07, +0.21] (1964 / 2000 fits converged).
- Both intervals comfortably include zero. With 8 NPB-flagged pitcher-seasons
  drawn from 5 underlying pitchers, this is genuinely underpowered — *do
  not claim an NPB effect*.
- Important: the bootstrap CI is *narrower* than the asymptotic CI here, not
  wider. The asymptotic SE is overstating uncertainty. So the underpowered
  conclusion isn't because the asymptotic interval was being optimistic — it
  literally wasn't being optimistic enough.

→ Insert **Figure 1** (`fig01_residuals_landscape.png`) here. Reading note for
the reader: each dot is one FS pitcher-season; the y=0 line is "physics
fully explains it." If NPB pitchers were systematically above the line we'd
see vermillion dots clustered up and to the right — they're not.

### 4.2 Splitter-specific patterns

- `usage_share` coefficient: **+2.80** on FS (p=0.04) vs +1.46 on CH
  (p=0.13). Pitchers who throw more splitters per pitch outperform model
  predictions by ~2.8 RV/100 per unit of share. The same pattern on
  changeups is ~half as large and not significant. *The splitter rewards
  arsenal-specialization more than the changeup does.*
- `is_novelty_year` coefficient: -0.66 on FS (p=0.09) vs -0.15 on CH
  (p=0.38). Rookie-year pitchers underperform the model's splitter prediction
  by about 0.66 RV/100; the same pattern on changeups is ~4× weaker and
  far from significant. Notably, this contradicts the "Japanese debutants
  outperform peripherals" intuition — the small NPB sample we have happens
  to land near zero, but the population-level rookie pattern is *underperformance*,
  not the reverse.
- Pitcher random-intercept variance: 0.40 on FS vs 0.17 on CH. The
  pitcher-level signal that physics doesn't capture is roughly 2.4× larger
  on splitters than changeups. **This may be the most important finding in
  the paper**: it's quantitative evidence that splitters carry more
  pitcher-specific structure than the comparison off-speed pitch carries.

### 4.3 The structural finding from the comparison-pitch check

- *None* of the 5 focal NPB-developed pitchers reached the 100-changeup
  threshold in any of the five seasons. The `npb_years` coefficient is
  unidentifiable on changeups by data construction.
- This is itself a substantive finding: the NPB-developed pitchers we have
  access to are **splitter specialists by arsenal**, not deception
  specialists who happen to favor the splitter. Roberts' framing
  ("doesn't take into account a hitter's approach") was about the splitter
  specifically; the data confirms it isn't a generalized off-speed thing
  for these pitchers.

### 4.4 Case studies

→ Insert **Figure 2** (`fig02_tunnels.png`) here.

- **Yamamoto**: FS model_resistance ~0 in both 2024 (-0.06) and 2025 (+0.21).
  His splitter is so physically dominant that the physics-only model
  already predicts it. **Surprise finding**: the release-point panel shows
  two distinct clusters — he changed arm slot between 2024 and 2025. We
  haven't isolated whether residual changed within either cluster; that's
  a follow-up.
- **Gausman**: hovers near zero (range -0.67 to +1.33 across five seasons),
  dipped slightly negative in 2023. The "Gausman exception" hypothesis
  from the project plan was: *does his domestic-trained splitter look more
  like the NPB cluster than other domestic splitters?* Answer with the
  current data: not really — his trajectory is in line with the median
  domestic FS thrower.
- **Skenes**: 2024 model_resistance = +1.44, 2025 = -1.15. **The documented
  mid-2025 splinker collapse is the clearest year-over-year drop in the
  whole focal roster.** Worth one paragraph speculating on adaptation
  vs. mechanical change vs. velocity outlier penalty. The plan flagged the
  splinker as a velocity outlier (94 mph vs ~86.5 mph average) and the
  classification problem (Statcast lumps splinker and traditional splitter
  as FS); the residual collapse in year 2 is consistent with hitters
  catching up to a pitch they hadn't seen before.

→ Insert **Figure 3** (`fig03_case_studies.png`) here.

### 4.5 Walker as the cautionary case

- Walker '24 is the single most extreme negative residual in the dataset
  (-5.49 RV/100). Plan flagged him as the worst splitter in baseball by
  run value in 2024 (-16 total). Worth a short paragraph: the model's
  residuals reproduce that public flag without being told to.

**Tone**: report what's there, hedge appropriately. Don't oversell the
splitter-specific patterns as proven; they're directional evidence.

---

## Section 5: What this paper doesn't establish (~½ page)

**Job**: pre-empt the reasonable objections.

- *Causation*. The regression is associational. We can't say "NPB
  development causes model resistance" even if the coefficient were nonzero
  — the n=5 selection bias problem is unfixable with available data.
- *Sequencing / cognitive disruption*. Roberts' third claim ("you don't know
  what's going on up here") is about hitter cognition. We have pitch-level
  outcomes, not hitter biometrics. A bat-tracking dataset (now public for
  2024+) could partially address this; we didn't use it in v1.
- *NPB-side data*. The plan considered translating NPB Tracker / DELTA data
  to MLB equivalents. That extension was cut from v1 — NPB data quality is
  significantly worse than Statcast and would dominate the error budget.
- *Left-censoring*. For pitchers in the regression who debuted before 2021,
  `mlb_seasons_in` is correctly computed via Chadwick lookup, but their
  career-history features (e.g., career splitter usage trend) aren't
  observed in our 5-year window.
- *Classification problem partially addressed*. We sub-clustered FS into
  "traditional" vs "splinker" via GMM but didn't run separate regressions
  on the two clusters. Could be a v2.

**Tone**: matter-of-fact. Acknowledge limits without apologizing for them.

---

## Section 6: What would change my mind (~½ page)

**Job**: state in concrete, falsifiable terms what evidence would update
each conclusion. This is the section reviewers will trust most.

- **NPB effect is real after all** if: (a) the bootstrap CI on `npb_years`
  excludes zero after we add 2026 data (Sasaki and Yamamoto are still adding
  pitcher-seasons), or (b) refitting with a Bayesian hierarchical model
  with weakly-informative priors gives a posterior that puts >95% mass
  above zero.
- **Usage-share effect is splitter-specific** if: replicating with a third
  comparison pitch (curveball or sweeper) shows the FS coefficient still
  ~2× the others. If the FS effect is just "off-speed pitches reward
  specialization in general," it would replicate on every off-speed type.
- **Skenes 2025 collapse is mechanical, not adaptation** if: a within-season
  decomposition of his 2025 FS shows the residual dropping starting at a
  specific date (suggesting a delivery change), rather than monotonically
  worsening across the season (suggesting hitters caught up).
- **The pitcher-specific variance is real signal, not noise** if: bootstrap
  CIs on the variance-component estimate (would need a new sensitivity
  analysis) put both FS and CH variances clearly nonzero, and the FS
  variance is strictly larger than CH in the bootstrap distribution.
- **The Yamamoto arm-slot shift matters** if: splitting his pitches by
  release-point cluster and re-running the residual analysis shows different
  model_resistance per cluster. This is a half-day of additional work and
  would be a clean follow-up.

---

## Section 7: Reproducibility (~¼ page)

- Repo URL.
- One-line reproduction: `uv sync --extra modeling && uv run python -m
  src.run_smoke && uv run python -m src.data.pull_statcast --all && uv run
  python -m src.models.stuff_baseline && uv run python -m
  src.models.residual_analysis && uv run python -m src.models.mixed_effects
  --pitch-type FS && uv run python -m src.models.mixed_effects --pitch-type
  CH && uv run python -m src.models.compare_pitch_types`. Mention runtime
  expectations (full data pull ~30 min, XGBoost CV ~10 min, FS bootstrap
  ~30 min).
- Note that all small outputs are committed to the repo; the only file
  that's regenerable rather than committed is the 54MB
  `pitches_with_predictions.parquet`.

---

## Closing notes for drafting

- **Voice**: informed-amateur-talking-to-peers, not academic. The audience is
  FanGraphs Community + r/sabermetrics, not a peer-reviewed journal.
- **Length budget**: section 4 is the heart, give it the space. Sections
  5–6 are short on purpose — they signal honesty without padding.
- **Three figures, no more.** Fig 1 is the headline (residuals landscape),
  fig 2 is the tunneling overlay (carries section 4.4 visually), fig 3 is
  the case-study trajectories.
- **What to cut if over budget**: Walker subsection (4.5) is optional; the
  reproducibility section can be 3 lines pointing at the README.
- **What to add if under budget**: a "what's next" paragraph on the
  Yamamoto arm-slot follow-up, since it's the most concrete unfinished
  thread.
