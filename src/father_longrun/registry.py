from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetMeta:
    key: str
    tier: str
    role: str
    first_pass_priority: bool


@dataclass(frozen=True)
class ExternalSourceMeta:
    key: str
    label: str
    role: str
    env_var: str
    category: str


DATASET_REGISTRY: tuple[DatasetMeta, ...] = (
    DatasetMeta("nlsy79_main", "core", "maternal history and pre-birth context", True),
    DatasetMeta("nlsy79_child", "core", "child exposure timing and child context", True),
    DatasetMeta("nlsy79_young_adult", "core", "adult outcomes", True),
    DatasetMeta("nlsy97", "core", "adult replication", True),
    DatasetMeta("psid_main", "core", "multigenerational panel", False),
    DatasetMeta("psid_shelf", "core", "harmonized long-run income and wealth", False),
    DatasetMeta("add_health_public", "secondary", "adult replication", False),
    DatasetMeta("ffcws_public", "secondary", "early-adult replication", False),
    DatasetMeta("acs_pums", "benchmark", "descriptive context", False),
    DatasetMeta("cps_asec", "benchmark", "official income / poverty benchmark", False),
    DatasetMeta("sipp", "benchmark", "short-run event-study benchmark", False),
)


EXTERNAL_SOURCE_REGISTRY: tuple[ExternalSourceMeta, ...] = (
    ExternalSourceMeta("bea", "BEA", "regional income and GDP context", "BEA_API_KEY", "macro context"),
    ExternalSourceMeta("fred", "FRED", "macro and labor-market time series", "FRED_API_KEY", "macro context"),
    ExternalSourceMeta("bls", "BLS", "labor-market benchmark series", "BLS_API_KEY", "macro context"),
    ExternalSourceMeta("census", "Census API", "ACS/CPS/SIPP programmatic pulls", "CENSUS_API_KEY", "microdata/program access"),
    ExternalSourceMeta("ipums", "IPUMS", "extract automation for ACS/CPS", "IPUMS_API_KEY", "microdata access"),
    ExternalSourceMeta("noaa_ncdc", "NOAA NCEI", "weather and climate context covariates", "NOAA_NCDC_API_TOKEN", "geographic context"),
    ExternalSourceMeta("usda_quickstats", "USDA Quick Stats", "agricultural/rural context covariates", "USDA_QUICK_STATS_API_KEY", "geographic context"),
)
