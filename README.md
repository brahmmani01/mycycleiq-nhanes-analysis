# Beyond Cycle Tracking — Adolescent Menstrual Health, Sleep & Depression

Reproducible analysis pipeline behind the MSHI capstone research
**"Beyond Cycle Tracking: Examining the Intersection of Mental Health,
Gamification, and Privacy in Adolescent Menstrual Health Technology"**
(Northeastern University, HINF 7701, April 2026), conducted in support
of the **MyCycleIQ** research collaboration.

**Research question:** Is adolescent menstrual irregularity independently
associated with clinical depression, or does sleep disturbance explain
the relationship — and what does that imply for how a menstrual-health
app should prioritize what it monitors?

## Data

Two data sources, matching the original study:

1. **NHANES 2005-2018** (7 CDC survey cycles), adolescent females
   ages 12-19, final analytic sample **N = 1,046**. `data/nhanes_synthetic.csv`
   is a *synthetic, individual-level reconstruction* calibrated so its
   marginal and joint distributions match every published aggregate
   statistic in the paper (race/ethnicity mix, menstrual-regularity and
   sleep-disturbance prevalence, PHQ-9 severity bands, and depression
   prevalence by subgroup) — see `generate_dataset.py` for exactly how
   each figure was matched. Real NHANES microdata requires pulling from
   the CDC's data portal, which this offline environment can't reach;
   this reconstruction lets the full analysis run end-to-end with the
   correct statistical signal.
2. **App evaluation rubric** (`data/app_evaluation_scores.csv`) — a
   30-point scoring of 8 commercial menstrual-health apps (Flo, Clue,
   Euki, Stardust, Clover, Glow, Period Tracker, My Calendar) across
   Gamification, Mental Health Integration, and Privacy domains.

## Method

- Variables: menstrual regularity (NHANES RHQ031), PHQ-9 depression
  screen (DPQ010-090, dichotomized at ≥10), sleep disturbance (SLQ050),
  age at menarche (RHD043), age, race/ethnicity.
- Two logistic regression models: **Model 1** — depression ~ menstrual
  irregularity (unadjusted); **Model 2** — depression ~ irregularity +
  sleep disturbance + age + race/ethnicity (adjusted).
- `analysis.R` mirrors the original pipeline exactly (R 4.3.2,
  `tidyverse` + `gtsummary` + `ggplot2`, as used in the capstone).
  `analysis.py` is a from-scratch Python re-implementation — logistic
  regression via Newton-Raphson IRLS rather than a black-box call —
  runnable in this repo without an R environment.

## Key finding

Sleep disturbance, not menstrual irregularity, is the variable that
predicts depression risk. That result is the reason a "cycle tracker"
alone is the wrong product shape for adolescent mental-health
monitoring — sleep needs to be a first-class signal.

| Model | Predictor | Odds Ratio | p-value |
|---|---|---|---|
| 1 (unadjusted) | Menstrual irregularity | ~1.0, not significant | n.s. |
| 2 (adjusted) | Sleep disturbance | **~4x**, significant | p < .001 |
| 2 (adjusted) | Menstrual irregularity | ~1.0, not significant | n.s. |

*(Exact coefficients differ slightly from the published paper because
this repo's dataset is a calibrated reconstruction, not the original
NHANES extract — the direction and significance pattern match.)*

## Files

```
data/
  nhanes_synthetic.csv         synthetic individual-level dataset (N=1,046)
  app_evaluation_scores.csv    8-app rubric scores (Table 3 in the paper)
generate_dataset.py            builds nhanes_synthetic.csv, documents every calibration target
analysis.py                    full pipeline: regression + all 5 figures (Python/numpy/matplotlib)
analysis.R                     same pipeline in R (tidyverse/gtsummary/ggplot2), matching original method
figures/                        output plots (age at menarche, PHQ-9 by subgroup, depression prevalence, forest plot, app scores)
```

## Running it

```bash
python3 generate_dataset.py   # builds data/nhanes_synthetic.csv
python3 analysis.py           # prints regression tables, writes figures/
# or, with R installed:
Rscript analysis.R
```

## Why this matters for health data / clinical informatics work

This project is the applied version of what a health data analyst
does day to day: pull a public health survey (NHANES), operationalize
clinical instruments (PHQ-9) and constructs (menstrual regularity,
sleep disturbance) into analysis-ready variables, run and interpret
regression models correctly (unadjusted vs. adjusted, odds ratios,
confidence intervals), and translate the statistical result into a
concrete product/design recommendation.
