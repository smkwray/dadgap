# dadgap

A reproducible, multi-cohort research pipeline that estimates associations between childhood resident biological father absence and adult economic outcomes using free public-use data.

## What this project does

dadgap builds analysis-ready datasets from longitudinal surveys (NLSY79, NLSY97, PSID) and cross-sectional benchmarks (ACS, CPS, SIPP), constructs harmonized exposure and outcome variables, and runs descriptive, quasi-causal, and machine-learning analyses on the relationship between father absence during childhood and later earnings, employment, and education.

The pipeline is structured in phases: cohort-specific data ingestion, variable harmonization, within-cohort analysis, and cross-cohort synthesis. Each phase produces structured artifacts (parquet files, CSV tables, markdown diagnostics) that are version-controlled and reproducible.

## Results

All estimates below are descriptive associations from public-use data, not causal effects. Source tables are in `outputs/tables/`.

### NLSY97: father-absence prevalence

43.2% of NLSY97 respondents (n = 8,982) did not live with their biological father continuously from birth to age 17.

| Subgroup | Father-absent | Father-present | n |
|---|---|---|---|
| Overall | 43.2% | 56.8% | 8,982 |
| Female | 45.4% | 54.6% | 4,384 |
| Male | 41.1% | 58.9% | 4,598 |
| Black | 65.4% | 34.6% | 2,334 |
| Hispanic | 40.4% | 59.6% | 1,900 |
| Non-Black, non-Hispanic | 33.4% | 66.6% | 4,748 |

<details>
<summary>Prevalence by parent education quartile</summary>

| Group | Band | Father-absent | n |
|---|---|---|---|
| Combined parent education | q1 (low) | 46.0% | 1,855 |
| Combined parent education | q2 | 45.0% | 1,855 |
| Combined parent education | q3 | 39.9% | 1,855 |
| Combined parent education | q4 (high) | 36.6% | 1,855 |
| Combined parent education | missing | 49.5% | 1,562 |
| Mother education | q1 (low) | 39.0% | 1,834 |
| Mother education | q2 | 53.2% | 1,833 |
| Mother education | q3 | 35.2% | 1,833 |
| Mother education | q4 (high) | 39.7% | 1,834 |
| Mother education | missing | 49.6% | 1,648 |
| Father education | q1 (low) | 38.2% | 1,686 |
| Father education | q2 | 46.5% | 1,685 |
| Father education | q3 | 38.0% | 1,685 |
| Father education | q4 (high) | 38.1% | 1,686 |
| Father education | missing | 52.3% | 2,240 |

</details>

### NLSY97: adult earnings and employment gaps

NLSY97 respondents who grew up with a resident biological father had higher mean 2021 earnings and employment rates than those who did not.

| Group | Mean 2021 earnings | Employment rate | Mean household income | n |
|---|---|---|---|---|
| Overall | $63,003 | 79.7% | $103,795 | 7,293 |
| Father-present | $70,784 | 83.0% | $121,212 | 4,185 |
| Father-absent | $51,731 | 75.3% | $80,283 | 3,106 |
| **Gap (present − absent)** | **$19,053** | **7.7 pp** | **$40,930** | |

<details>
<summary>Earnings and employment by race, sex, and father-presence status</summary>

**Father-present group:**

| Race/ethnicity | Sex | Mean 2021 earnings | Employment rate | n |
|---|---|---|---|---|
| Black | Female | $53,579 | 78.8% | 316 |
| Black | Male | $58,497 | 77.9% | 343 |
| Hispanic | Female | $50,535 | 74.4% | 439 |
| Hispanic | Male | $66,143 | 88.6% | 479 |
| Non-Black, non-Hispanic | Female | $65,024 | 80.0% | 1,213 |
| Non-Black, non-Hispanic | Male | $89,417 | 89.5% | 1,395 |

**Father-absent group:**

| Race/ethnicity | Sex | Mean 2021 earnings | Employment rate | n |
|---|---|---|---|---|
| Black | Female | $43,975 | 72.6% | 586 |
| Black | Male | $46,339 | 72.7% | 614 |
| Hispanic | Female | $44,052 | 68.5% | 326 |
| Hispanic | Male | $57,287 | 79.7% | 309 |
| Non-Black, non-Hispanic | Female | $47,898 | 74.0% | 667 |
| Non-Black, non-Hispanic | Male | $69,564 | 84.6% | 604 |

**Father-present minus father-absent earnings gap by subgroup:**

| Race/ethnicity | Sex | Earnings gap |
|---|---|---|
| Black | Female | +$9,604 |
| Black | Male | +$12,158 |
| Hispanic | Female | +$6,483 |
| Hispanic | Male | +$8,856 |
| Non-Black, non-Hispanic | Female | +$17,126 |
| Non-Black, non-Hispanic | Male | +$19,853 |

</details>

### NLSY97: predictors of father absence

Logistic regression (HC1 standard errors, n = 8,982) with father absence as the outcome. Race/ethnicity is the strongest predictor. Coefficients are associations, not causal estimates.

| Predictor | Odds ratio | p-value |
|---|---|---|
| Non-Black, non-Hispanic (ref: Black) | 0.287 | 1.14 × 10⁻¹¹² |
| Hispanic (ref: Black) | 0.361 | 4.80 × 10⁻⁵⁵ |
| Father education missing | 1.431 | 3.99 × 10⁻⁶ |
| Birth year (centered) | 0.958 | 0.007 |
| Male | 0.843 | < 0.001 |
| Mother education (filled) | 0.977 | 0.310 |
| Father education (filled) | 0.980 | 0.302 |
| Mother education missing | 0.909 | 0.269 |

### ACS: child father-presence context

ACS 2024 public-use microdata (n = 596,271 children; weighted population 69.0 million) provides a cross-sectional snapshot of children's current living arrangements using the Census `ESP` variable as a proxy for father presence. This is not a longitudinal family-history measure.

| Subgroup | Father-absent | Father-present | Weighted children |
|---|---|---|---|
| Overall | 25.6% | 74.4% | 69.0M |
| Black | 54.4% | 45.6% | 8.3M |
| Hispanic | 32.0% | 68.0% | 18.6M |
| Non-Black, non-Hispanic | 17.1% | 82.9% | 42.2M |

<details>
<summary>ACS child father-presence by poverty and household income</summary>

**By poverty band:**

| Poverty band | Father-absent | Father-present | Weighted children |
|---|---|---|---|
| Below 100% | 59.2% | 40.8% | 10.1M |
| 100–124% | 43.6% | 56.4% | 3.2M |
| 125–149% | 39.6% | 60.4% | 3.3M |
| 150%+ | 17.1% | 82.9% | 52.4M |

**By household income quartile:**

| Income quartile | Father-absent | Father-present | Mean household income |
|---|---|---|---|
| q1 (low) | 50.5% | 49.5% | $34,242 |
| q2 | 23.9% | 76.1% | $86,690 |
| q3 | 13.7% | 86.3% | $143,430 |
| q4 (high) | 7.6% | 92.4% | $328,204 |

</details>

### Public benchmark context

External survey benchmarks for adults age 25-54, for comparison with NLSY97 estimates.

| Source | Year | Period | Mean earnings | Employment | Poverty rate | Weighted pop. |
|---|---|---|---|---|---|---|
| ACS PUMS | 2024 | Annual | $55,894 | 81.1% | 10.6% | 74.2M |
| CPS ASEC | 2023 | Annual | $53,341 | 80.6% | 10.0% | 71.6M |
| CPS ASEC | 2024 | Annual | $55,682 | 80.4% | 10.1% | 71.8M |
| CPS ASEC | 2025 | Annual | $58,996 | 80.5% | 9.9% | 73.6M |
| SIPP | 2023 | Monthly | $6,137 | 82.4% | 11.6% | — |
| NLSY97 | 2021 | Annual | $63,003 | 79.7% | — | — |

NLSY97 mean earnings are higher than the population-weighted ACS/CPS benchmarks. NLSY97 estimates are unweighted and reflect a specific birth cohort (born 1980-1984) observed at ages 37-41. ACS and CPS are population-weighted cross-sections covering all adults 25-54. SIPP reports monthly earnings and is not directly comparable to annual figures.

## Data sources

**Core longitudinal cohorts:**
- **NLSY79** (main + Child/Young Adult) — childhood exposure timing, maternal history, sibling comparisons
- **NLSY97** — adult earnings, employment, education through age 40+
- **PSID** (+ family-history files, PSID-SHELF, CDS/TAS) — multigenerational income and wealth

**Secondary replication:**
- **Add Health** public-use (Waves I-VI) — adult outcome replication
- **FFCWS** public-use — father involvement and early-adult outcomes

**Benchmark / context:**
- **ACS PUMS** — descriptive prevalence and subgroup benchmarks
- **CPS ASEC** — income and poverty benchmarks
- **SIPP** — short-run household dynamics
- **FRED / BEA / BLS / Census APIs** — macro-context overlays

See [DATA.md](DATA.md) for source details and access instructions.

## Quick start

```bash
# 1. Bootstrap environment (creates venv outside repo)
export DADGAP_VENV_PATH="${HOME}/venvs/dadgap"
./scripts/bootstrap_project.sh

# 2. Configure
cp config/user_inputs.example.yaml config/user_inputs.local.yaml
# Edit user_inputs.local.yaml with your local data paths

# 3. Set API credentials (optional, for benchmark modules)
cp .env.example .env.local
# Add keys for FRED, BEA, BLS, Census, IPUMS as needed

# 4. Validate setup
father-longrun check-config --config config/user_inputs.local.yaml
father-longrun source-status
```

Requires Python 3.10+. Install with analysis and ML extras:

```bash
pip install -e ".[dev,ml]"
```

## Build sequence

The pipeline runs as a series of CLI commands. Each step produces artifacts that feed the next.

```bash
# NLSY intake and treatment construction
make build-nlsy-pilot
make build-reviewed-layers
make build-analysis-ready-treatments
make build-fatherlessness-profiles
make build-nlsy97-longitudinal-panel

# Quasi-causal and ML analysis
make build-quasi-causal-scaffold
make build-ml-benchmarks

# Public benchmarks
make build-benchmarks
make build-public-microdata
make build-public-benchmark-profiles
make build-cross-cohort-benchmarks

# Final output synthesis
make build-results-appendix
```

All commands accept `CONFIG=path/to/config.yaml`. Default is `config/user_inputs.local.yaml`.

Run `make test` to execute the test suite.

## Repo layout

```
config/           Configuration templates and variable manifests
src/father_longrun/
  cli.py          CLI entry point (Typer)
  pipelines/      Data ingestion, harmonization, reporting
  models/         OLS, quasi-causal, ML analysis modules
tests/            Smoke tests
data/
  raw/            Untouched source files
  external/       Downloaded public datasets and API snapshots
  interim/        Standardized intermediate files
  processed/      Analysis-ready parquet and CSV
outputs/
  tables/         Final publication tables
  models/         Quasi-causal and ML workbenches
  manifests/      Variable mappings, diagnostics, handoff memos
```

## Exposure definitions

Father absence is coded from household roster and family-structure survey items into eight categories:

1. Always-resident biological father (birth to 17)
2. Absent from birth
3. Early exit (age 0-5)
4. Middle-childhood exit (age 6-11)
5. Adolescent exit (age 12-17)
6. Intermittent / repeated transitions
7. Father deceased
8. Other / unknown

These categories are constructed separately for each cohort, then harmonized using a common variable library. See [METHODS.md](METHODS.md) for identification strategy and modeling details.

## Key limitations

- NLSY97 predictor coefficients are descriptive associations, not causal estimates.
- ACS child father-presence uses the Census `ESP` (subfamily relationship) variable as a proxy and does not capture full family history.
- SIPP provides monthly earnings context and is not directly comparable to annual measures from NLSY97 or CPS.
- Public-use data imposes sample and geographic restrictions relative to restricted-access files.

## License

MIT. See [LICENSE](LICENSE).
