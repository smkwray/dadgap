# dadgap

**[Browse the interactive results site](https://smkwray.github.io/dadgap/)**

`dadgap` is a reproducible research pipeline that measures how childhood absence of a resident biological father relates to adult earnings, employment, and household income, using free public-use U.S. survey data. The published results come from the National Longitudinal Survey of Youth 1997 (NLSY97), with cross-sectional benchmarks from the American Community Survey (ACS), the Current Population Survey Annual Social and Economic Supplement (CPS ASEC), and the Survey of Income and Program Participation (SIPP).

In NLSY97 (n = 8,982), 43.2% of respondents did not live continuously with their biological father from birth to age 17. Respondents who did averaged $19,053 more in 2021 earnings ($70,784 vs. $51,731) and had a 7.7 percentage-point higher employment rate (83.0% vs. 75.3%). All published estimates are descriptive associations, not causal effects.

## Headline findings

| NLSY97, 2021 | Father-present | Father-absent | Gap |
|---|---|---|---|
| Mean earnings | $70,784 | $51,731 | $19,053 |
| Employment rate | 83.0% | 75.3% | 7.7 pp |
| Mean household income | $121,212 | $80,283 | $40,930 |
| Respondents with outcomes | 4,185 | 3,106 | |

- Father absence in NLSY97 is 65.4% among Black respondents, 40.4% among Hispanic respondents, and 33.4% among other respondents.
- In ACS 2024 (596,271 children), 25.6% of children under 18 have no resident father in their current household. The share is 59.2% for children below the poverty line and 7.6% in the top household-income quartile.
- NLSY97 mean 2021 earnings ($63,003) are higher than the ACS 2024 ($55,894) and CPS ASEC 2023 ($53,341) means for adults aged 25–54. NLSY97 figures are unweighted and cover one birth cohort (1980–1984) at ages 37–41.

The site also reports subgroup splits by race, sex, and parent education, a logistic model of who experiences father absence, and NLSY97 gaps in cognitive scores, education, health, mental health, family formation, and occupation.

## Method

Father absence is coded from household roster and family-structure items into eight categories: always-resident father, absent from birth, exit at ages 0–5, 6–11, or 12–17, repeated transitions, father deceased, and other or unknown. Results compare father-present and father-absent groups, with covariate-adjusted gaps where noted. The site's [methods page](https://smkwray.github.io/dadgap/methods.html) describes the exposure coding, outcomes, and covariates.

## Data sources

| Source | Use |
|---|---|
| NLSY97 | Exposure and adult outcomes (published results) |
| ACS public-use microdata (PUMS) 2024 | Child father-presence context and adult benchmarks (published) |
| CPS ASEC 2023–2025 | Adult earnings, employment, and poverty benchmarks (published) |
| SIPP 2023 | Monthly earnings benchmark (published) |
| NLSY79, Panel Study of Income Dynamics (PSID), Add Health, Fragile Families (FFCWS) | Intake code only; no published results |
| FRED, BEA, BLS, and Census APIs | Optional macroeconomic context |

No data are committed. All sources are free public-use files; [DATA.md](DATA.md) lists where to get each one. The code also contains quasi-causal and machine-learning modules whose output is not part of the published results.

## Reproduce

Requires Python 3.10 or later. The Python package is `father_longrun` and the command-line tool is `father-longrun`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analysis]"        # add ".[ml]" for the machine-learning modules
cp config/user_inputs.example.yaml config/user_inputs.local.yaml   # set your local data paths
cp .env.example .env.local              # optional API keys for the benchmark sources
father-longrun check-config --config config/user_inputs.local.yaml
make test
```

`father-longrun` reads `.env.local` from the repository root and resolves relative paths in the config against it. With NLSY97 and the benchmark files in place, build the NLSY layers and then the published results:

```bash
make build-nlsy-pilot build-reviewed-layers build-analysis-ready-treatments
make public-results     # profiles, benchmarks, results appendix, synthesis, docs/results.json
father-longrun doctor --config config/user_inputs.local.yaml   # check config and the published artifacts
```

Every `make` target accepts `CONFIG=path/to/config.yaml`. To preview the site locally, run `python3 -m http.server --directory docs`.

## Layout

```
config/             Config templates and variable manifests
src/father_longrun/ CLI, data pipelines, and models
tests/              Test suite
data/               Raw, external, interim, and processed data (not committed)
outputs/            Generated tables and manifests (not committed)
docs/               Static results site; docs/results.json holds the published numbers
```

## Limitations

- All results are descriptive associations, including the predictor model.
- The ACS measure uses the parent-status variable `ESP`, so it covers only children living with at least one parent and describes the current household, not childhood history.
- SIPP reports monthly earnings and is not directly comparable to annual NLSY97, ACS, or CPS figures.
- Public-use files have smaller samples and less geographic detail than restricted-access versions.

## License

MIT. See [LICENSE](LICENSE).
