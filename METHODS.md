# Methods

## Research design

This project estimates the association between childhood resident biological father absence (birth to age 17) and adult economic outcomes using a multi-cohort public-data design. Each cohort is analyzed separately with harmonized variable definitions, then compared across cohorts.

Published results come from one cohort, NLSY97, set against cross-sectional benchmarks from the ACS, CPS ASEC, and SIPP. The pipeline also has code for NLSY79, PSID, Add Health, and FFCWS, but none of them has published results (see [DATA.md](DATA.md)).

## Identification hierarchy

Analyses are organized into four tiers:

1. **Descriptive** — survey-weighted means and group contrasts
2. **Adjusted association** — OLS/GLM with pre-exposure covariates
3. **Quasi-causal** — sibling fixed effects, event-time models, inverse-probability weighting, doubly robust estimation
4. **Exploratory ML** — Double ML, causal forests, predictive benchmarks

Published results use the first two tiers only: descriptive contrasts and covariate-adjusted gaps in NLSY97, plus a descriptive model of who experiences father absence. The published NLSY97 estimates are unweighted. The quasi-causal and machine-learning modules are in the code, but their output is not published.

Each tier builds on the prior. Descriptive results establish baseline patterns. Adjusted models add pre-exposure controls. Quasi-causal designs exploit within-family variation or timing of father exit. ML models explore heterogeneity after the identification logic is established.

## Exposure construction

The primary exposure variable is `father_absence_trajectory_0_17`, coded into eight mutually exclusive categories from household roster and family-structure items in each survey:

| Code | Category |
|---|---|
| 1 | Always-resident biological father (0-17) |
| 2 | Absent from birth |
| 3 | Early exit (age 0-5) |
| 4 | Middle-childhood exit (age 6-11) |
| 5 | Adolescent exit (age 12-17) |
| 6 | Intermittent / repeated transitions |
| 7 | Father deceased |
| 8 | Other / unknown |

Additional continuous and semi-continuous variants:
- Total years with resident biological father before age 18
- Age at first exit
- Count of family-structure transitions
- Nonresident father contact intensity (treated as moderator/mediator, not the same construct as residence)
- Stepfather / social father presence indicators

Biological father residence is kept separate from partner/stepfather residence. Raw cohort-specific variables are preserved alongside harmonized versions. All recoding decisions are logged in variable manifests under `outputs/manifests/`.

## Outcome measures

### Primary adult economic outcomes
- Annual individual earnings (and log earnings)
- Employment indicator
- Weeks/months/hours worked
- Hourly wage (where available)
- Household income
- Poverty / income-to-needs ratio
- Net worth / wealth
- Homeownership
- College completion

### Secondary outcomes
- Occupation and job stability
- Public assistance receipt
- Marital/cohabitation status
- Fertility timing
- Justice system contact (where publicly available)
- Physical and mental health (as mechanisms)

### Age windows
- **Primary:** age 30-40
- **Fallback:** age 25-29
- **Near-adult:** age 22-24 (labeled as such; used only with FFCWS or similar early-adult cohorts)

Unlike age windows are not pooled without age-standardization, cohort-specific models, or a meta-analytic synthesis step.

## Pre-exposure covariates

Models control for variables that plausibly predate the exposure:

- Child sex, race/ethnicity
- Maternal age at birth, education, marital/cohabitation status near birth
- Baseline household income / poverty
- Birth order, sibship size
- Region
- Maternal employment at baseline
- Maternal cognitive proxies (AFQT or equivalent, where available)
- Baseline child health

Mediators (later schooling, adolescent behavior) are handled in separate mechanism models, not included in baseline causal specifications.

## Statistical models

### Baseline
- Survey-weighted means and contrasts
- Survey-weighted OLS / GLM
- Clustered standard errors at family level

### Quasi-causal
- **Sibling fixed effects** — within-family comparisons for siblings exposed at different ages or durations
- **Event-time models** — outcomes indexed relative to father-exit event with pre/post indicators
- **Inverse-probability weighting** and doubly robust AIPW estimation
- **Sensitivity analysis** — coefficient-stability and omitted-variable bias tests

### Machine learning
Used only after causal design is established:
- Nuisance models for Double ML
- Prediction baselines: elastic net, random forest, gradient boosting
- Heterogeneity: causal forest, orthogonal random forest, best linear projection of CATEs
- Cross-fitting with train/validation/test separation and cohort-out validation

Survey weights are preserved where the library supports them. When weights cannot be honored, weighted and unweighted results are run in parallel and the gap is documented.

## Multi-cohort synthesis

Cohorts are not concatenated into a single file. Instead:

1. Cohort-specific models are estimated with harmonized variable definitions
2. Standardized outcome scales or comparable estimands are produced
3. Estimates are meta-analyzed or partially pooled
4. Results note where findings replicate and where they are cohort-specific

With one published cohort, the current synthesis step combines the NLSY97 estimates with the ACS, CPS ASEC, and SIPP benchmarks into summary and forest-ready tables. No cross-cohort meta-analysis is published.

## Reproducibility

- All source data is free public-use
- Pipeline code is in `src/father_longrun/`
- Every variable mapping is logged in CSV/JSON manifests
- Processed data is stored as Apache Parquet
- All build steps are available as CLI commands and Makefile targets
- Tests cover config validation, pipeline functions, and reporting
