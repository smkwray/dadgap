# Data sources

All data used in this project is free public-use. No restricted-access datasets are required.

Published results come from NLSY97, with cross-sectional benchmarks from the ACS, CPS ASEC, and SIPP. The pipeline code also covers NLSY79, PSID, Add Health, and FFCWS, but none of them has published results.

## Published cohort

### NLSY97

The National Longitudinal Survey of Youth 1997. Provides adult earnings, employment, educational attainment, and household composition through the most recent survey rounds.

- Access: [NLS Investigator](https://www.nlsinfo.org/investigator/)
- Role: Primary analysis cohort; source of every published cohort estimate
- Note: Exposure definition relies on retrospective household composition items; less direct than NLSY79 Child roster-based constructs

## Published benchmarks

### ACS PUMS

American Community Survey Public Use Microdata Sample. Used for descriptive prevalence benchmarks, subgroup context, and national calibration.

- Access: [Census Bureau](https://www.census.gov/programs-surveys/acs/microdata.html) or [IPUMS USA](https://usa.ipums.org/)
- Note: ACS child father-presence is an `ESP`-based proxy, not a longitudinal family-history measure. `ESP` is the ACS parent-status variable (employment status of parents): it records whether a child lives with two parents, with the father only, or with the mother only. It is defined only for own children of the householder and children in subfamilies, and it describes the current household

### CPS ASEC

Current Population Survey Annual Social and Economic Supplement. Provides official income and poverty benchmarks and family structure prevalence checks.

- Access: [Census Bureau](https://www.census.gov/programs-surveys/cps.html) or [IPUMS CPS](https://cps.ipums.org/)

### SIPP

Survey of Income and Program Participation. Used for short-run household dynamics, income transitions, transfer receipt, and maternal labor-supply context.

- Access: [Census Bureau](https://www.census.gov/programs-surveys/sipp.html)
- Note: SIPP reports monthly earnings; not directly comparable to annual measures from NLSY97 or CPS

## Optional macro context

Aggregate economic context from federal statistical agencies:

| Source | Purpose | Credential env var |
|---|---|---|
| FRED | Labor market, inflation, business cycle | `FRED_API_KEY` |
| BEA | Regional income, GDP, transfers | `BEA_API_KEY` |
| BLS | Labor force, unemployment, CPI, wages | `BLS_API_KEY` |
| Census API | ACS/CPS/SIPP table pulls | `CENSUS_API_KEY` |
| IPUMS API | Repeatable microdata extracts | `IPUMS_API_KEY` |

API credentials go in `.env.local` (not tracked by git). Check credential status with `father-longrun source-status`.

## Cohorts without published results

The pipeline has intake code for these cohorts, but no results from them are published.

### NLSY79 (main + Child/Young Adult)

The National Longitudinal Survey of Youth 1979 and its Child/Young Adult supplements. Provides childhood exposure timing, maternal background, pre-birth covariates, father-presence/contact constructs, repeated child and young-adult outcomes, and sibling/cousin comparison potential.

- Access: [NLS Investigator](https://www.nlsinfo.org/investigator/)
- Role: Pipeline and sibling fixed-effects design code; no published results
- Key constructs: household roster, family structure history, maternal interview, adult earnings and employment

### PSID (+ family-history files, PSID-SHELF, CDS/TAS)

The Panel Study of Income Dynamics and associated files. Multigenerational design with long-run income, wealth/net worth, parent-child linkage, and family structure history spanning decades.

- Access: [PSID Data Center](https://psidonline.isr.umich.edu/)
- Role: Intake code only; no published results
- Components covered: main interview, Parent Identification File, Childbirth and Adoption History, Marriage History, CDS, TAS

### Add Health (public-use)

The National Longitudinal Study of Adolescent to Adult Health. Public-use data spans Waves I-VI and supports adult outcome replication across employment, education, family formation, and health.

- Access: [Add Health](https://addhealth.cpc.unc.edu/)
- Role: Intake code only; no published results
- Limitation: Public-use is a reduced sample; friend/sibling linkages are not available in public-use files

### FFCWS (public-use)

The Fragile Families and Child Wellbeing Study. Strong measures of father involvement, nonresident father contact, union instability, and transition-to-adulthood outcomes through Year 22.

- Access: [FFCWS / ICPSR](https://fragilefamilies.princeton.edu/)
- Role: Intake code only; no published results
- Limitation: Outcomes are measured at approximately age 22, which is earlier in the life course than NLSY or PSID adult windows

## Data directory layout

```
data/
  raw/              Untouched source files
  external/
    public_benchmarks/raw/   FRED, BEA, BLS, Census API snapshots
    ipums/raw/               IPUMS extract snapshots
    acs_pums/raw/            Local ACS PUMS files
  interim/
    nlsy_refresh/            Refreshed NLSY extracts with treatment variables
  processed/
    nlsy/                    Harmonized NLSY layers (parquet)
    public_benchmarks/       Normalized benchmark tables
    public_microdata/        ACS/CPS/SIPP harmonized subsets
    ipums/                   IPUMS processed extracts
```

Raw and external data files are not committed to the repository. The pipeline regenerates processed outputs from source data using the CLI.
