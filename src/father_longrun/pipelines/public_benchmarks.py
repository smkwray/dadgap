from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from father_longrun.pipelines.harmonize import (
    build_poverty_band_from_percent,
    build_poverty_band_from_ratio,
    map_public_race_ethnicity_3cat,
    map_public_sex,
    standardize_public_profile_frame,
)
from father_longrun.registry import EXTERNAL_SOURCE_REGISTRY, ExternalSourceMeta


FRED_SERIES_DEFAULTS: tuple[dict[str, str], ...] = (
    {"series_id": "UNRATE", "canonical_name": "us_unemployment_rate", "category": "labor_market"},
    {"series_id": "PAYEMS", "canonical_name": "us_total_nonfarm_payrolls", "category": "labor_market"},
    {"series_id": "CPIAUCSL", "canonical_name": "us_cpi_all_items", "category": "prices"},
)

BLS_SERIES_DEFAULTS: tuple[dict[str, str], ...] = (
    {"series_id": "LNS14000000", "canonical_name": "us_unemployment_rate", "category": "labor_market"},
    {"series_id": "CES0000000001", "canonical_name": "us_total_nonfarm_payrolls", "category": "labor_market"},
    {"series_id": "CUUR0000SA0", "canonical_name": "us_cpi_all_items", "category": "prices"},
)

BEA_REQUEST_DEFAULTS: tuple[dict[str, str], ...] = (
    {
        "dataset": "Regional",
        "table_name": "SAINC1",
        "line_code": "3",
        "geo_fips": "STATE",
        "year": "ALL",
        "canonical_name": "state_per_capita_personal_income",
    },
    {
        "dataset": "Regional",
        "table_name": "SAGDP9N",
        "line_code": "2",
        "geo_fips": "STATE",
        "year": "ALL",
        "canonical_name": "state_real_gdp",
    },
)

CENSUS_REQUEST_DEFAULTS: tuple[dict[str, Any], ...] = (
    {
        "dataset": "2024/acs/acs5",
        "canonical_name": "acs5_state_context_2024",
        "geography": "state:*",
        "variables": [
            "NAME",
            "B19013_001E",
            "B17001_001E",
            "B17001_002E",
            "B23025_003E",
            "B23025_005E",
            "B15003_001E",
            "B15003_022E",
            "B15003_023E",
            "B15003_024E",
            "B15003_025E",
        ],
    },
)

IPUMS_EXTRACT_DEFAULTS: tuple[dict[str, Any], ...] = (
    {
        "collection": "cps",
        "description": "dadgap_cps_asec_benchmark",
        "samples": [
            "cps2023_03s",
            "cps2024_03s",
            "cps2025_03s",
        ],
        "variables": [
            "AGE",
            "SEX",
            "RACE",
            "HISPAN",
            "STATEFIP",
            "EDUC",
            "EMPSTAT",
            "LABFORCE",
            "INCWAGE",
            "INCTOT",
            "POVERTY",
            "FAMSIZE",
            "MARST",
            "RELATE",
        ],
        "submit": "false",
    },
)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"
BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BEA_BASE_URL = "https://apps.bea.gov/api/data"
CENSUS_BASE_URL = "https://api.census.gov/data"
IPUMS_BASE_URL = "https://api.ipums.org/extracts"

SIPP_RMESR_LABELS: dict[int, str] = {
    1: "with_job_entire_month_worked_all_weeks",
    2: "with_job_entire_month_absent_without_pay_not_layoff",
    3: "with_job_entire_month_absent_without_pay_due_to_layoff",
    4: "with_job_part_month_no_layoff_or_search",
    5: "with_job_part_month_some_layoff_or_search",
    6: "no_job_all_month_on_layoff_or_looking_all_weeks",
    7: "no_job_all_month_some_layoff_or_looking",
    8: "no_job_all_month_no_layoff_and_no_search",
}

ACS_ESP_LABELS: dict[int, str] = {
    1: "living_with_two_parents_both_in_labor_force",
    2: "living_with_two_parents_father_only_in_labor_force",
    3: "living_with_two_parents_mother_only_in_labor_force",
    4: "living_with_two_parents_neither_in_labor_force",
    5: "living_with_father_in_labor_force",
    6: "living_with_father_not_in_labor_force",
    7: "living_with_mother_in_labor_force",
    8: "living_with_mother_not_in_labor_force",
}


@dataclass(frozen=True)
class ExternalSourceStatus:
    key: str
    label: str
    role: str
    env_var: str
    configured: bool
    category: str


@dataclass(frozen=True)
class SourceBuildResult:
    source: str
    raw_json_path: Path
    observations_path: Path
    metadata_path: Path
    row_count: int


@dataclass(frozen=True)
class BenchmarkBuildResult:
    manifest_path: Path
    results: tuple[SourceBuildResult, ...]


@dataclass(frozen=True)
class IpumsWorkflowResult:
    request_path: Path
    extracts_path: Path
    metadata_path: Path
    status_path: Path
    raw_json_path: Path
    row_count: int
    submitted: bool


@dataclass(frozen=True)
class PublicMicrodataArtifact:
    source: str
    source_path: Path
    parquet_path: Path
    metadata_path: Path
    row_count: int
    column_count: int


@dataclass(frozen=True)
class PublicMicrodataResult:
    manifest_path: Path
    artifacts: tuple[PublicMicrodataArtifact, ...]


@dataclass(frozen=True)
class PublicBenchmarkProfileResult:
    profiles_path: Path
    mapping_path: Path
    summary_path: Path
    subgroup_summary_path: Path
    sipp_employment_codebook_path: Path
    acs_child_context_path: Path
    acs_child_summary_path: Path
    acs_child_report_path: Path
    row_count: int


@dataclass(frozen=True)
class CrossCohortBenchmarkResult:
    profiles_path: Path
    summary_path: Path
    subgroup_summary_path: Path
    report_path: Path
    row_count: int


def source_statuses() -> tuple[ExternalSourceStatus, ...]:
    statuses: list[ExternalSourceStatus] = []
    for item in EXTERNAL_SOURCE_REGISTRY:
        statuses.append(
            ExternalSourceStatus(
                key=item.key,
                label=item.label,
                role=item.role,
                env_var=item.env_var,
                configured=bool(os.environ.get(item.env_var)),
                category=item.category,
            )
        )
    return tuple(statuses)


def prioritized_public_sources() -> tuple[ExternalSourceMeta, ...]:
    """Order macro/context sources ahead of download-heavy microdata integrations."""
    priority_order = ("fred", "bea", "bls", "census", "ipums", "noaa_ncdc", "usda_quickstats")
    by_key = {item.key: item for item in EXTERNAL_SOURCE_REGISTRY}
    return tuple(by_key[key] for key in priority_order if key in by_key)


def _http_request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> Any:
    full_url = f"{url}?{urlencode(params)}" if params else url
    request = Request(
        full_url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers=headers or {},
        method=method,
    )
    with urlopen(request) as response:  # noqa: S310 - official public APIs only
        return json.loads(response.read().decode("utf-8"))


def _http_get_json(url: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    return _http_request_json(url, params=params, headers=headers, method="GET")


def _http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    return _http_request_json(url, payload=payload, headers=merged_headers, method="POST")


def _sanitize_payload(value: Any, *, secrets: Iterable[str]) -> Any:
    secrets_tuple = tuple(secret for secret in secrets if secret)
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, nested in value.items():
            lowered = key.lower()
            if lowered in {"api_key", "apikey", "registrationkey", "userid", "user_id", "token", "email"}:
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _sanitize_payload(nested, secrets=secrets_tuple)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_payload(item, secrets=secrets_tuple) for item in value]
    if isinstance(value, str):
        sanitized = value
        for secret in secrets_tuple:
            sanitized = sanitized.replace(secret, "<redacted>")
        return sanitized
    return value


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(".", pd.NA), errors="coerce")


def _read_source_specs(config: dict[str, Any], key: str, defaults: tuple[dict[str, str], ...]) -> list[dict[str, str]]:
    benchmarks = config.get("benchmarks", {}) if isinstance(config.get("benchmarks", {}), dict) else {}
    nested = benchmarks.get(key)
    if isinstance(nested, list) and nested:
        rows: list[dict[str, str]] = []
        for item in nested:
            if isinstance(item, dict):
                rows.append({k: str(v) for k, v in item.items()})
        if rows:
            return rows
    return [dict(item) for item in defaults]


def _read_flexible_specs(config: dict[str, Any], key: str, defaults: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    benchmarks = config.get("benchmarks", {}) if isinstance(config.get("benchmarks", {}), dict) else {}
    nested = benchmarks.get(key)
    if isinstance(nested, list) and nested:
        rows: list[dict[str, Any]] = []
        for item in nested:
            if isinstance(item, dict):
                rows.append(dict(item))
        if rows:
            return rows
    return [dict(item) for item in defaults]


def _write_frame_bundle(frame: pd.DataFrame, metadata: pd.DataFrame, *, source: str, processed_root: Path) -> tuple[Path, Path]:
    processed_root.mkdir(parents=True, exist_ok=True)
    observations_path = processed_root / f"{source}_observations.parquet"
    metadata_path = processed_root / f"{source}_metadata.csv"
    frame.to_parquet(observations_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    return observations_path, metadata_path


def _write_sanitized_json(payload: dict[str, Any], *, source: str, raw_root: Path, secrets: Iterable[str]) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    path = raw_root / f"{source}_snapshot.json"
    path.write_text(json.dumps(_sanitize_payload(payload, secrets=secrets), indent=2), encoding="utf-8")
    return path


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.where(denominator.ne(0)).divide(denominator.where(denominator.ne(0)))


def _series_or_na(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _public_value(path: Path) -> str:
    suffix = path.parts[-2:] if len(path.parts) >= 2 else path.parts
    return f"<local_path>/{'/'.join(suffix)}"


def _largest_matching_file(root: Path, suffixes: tuple[str, ...]) -> Path | None:
    matches = [path for path in root.rglob("*") if path.is_file() and path.name.lower().endswith(suffixes)]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_size)


def _benchmark_dir(config: dict[str, Any], key: str) -> Path | None:
    benchmarks = config.get("benchmarks", {}) if isinstance(config.get("benchmarks", {}), dict) else {}
    raw_value = benchmarks.get(key)
    if not isinstance(raw_value, str) or not raw_value or raw_value.startswith("/ABSOLUTE/PATH/TO"):
        return None
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path.resolve()


def _sipp_selected_spec() -> tuple[tuple[str, str], ...]:
    return (
        ("SSUID", "sample_unit_id"),
        ("SHHADID", "household_id_interview"),
        ("SPANEL", "panel_year"),
        ("SWAVE", "interview_wave"),
        ("PNUM", "person_number"),
        ("WPFINWGT", "person_weight"),
        ("TAGE", "age"),
        ("ESEX", "sex_code"),
        ("ERACE", "race_code"),
        ("EHISPAN", "hispanic_code"),
        ("EEDUC", "education_code"),
        ("ERELRPE", "relationship_to_reference_code"),
        ("TMARPATH", "marital_path_code"),
        ("RMESR", "monthly_employment_status_code"),
        ("TPEARN", "monthly_personal_earnings"),
        ("TPTOTINC", "monthly_person_total_income"),
        ("THTOTINC", "monthly_household_total_income"),
        ("TFTOTINC", "monthly_family_total_income"),
        ("RHPOV", "household_poverty_threshold"),
        ("RFPOV", "family_poverty_threshold"),
        ("THINCPOV", "household_income_poverty_ratio"),
        ("TFINCPOV", "family_income_poverty_ratio"),
    )


def _cps_selected_spec() -> tuple[tuple[str, str], ...]:
    return (
        ("YEAR", "survey_year"),
        ("MONTH", "survey_month"),
        ("SERIAL", "household_serial"),
        ("CPSID", "household_id"),
        ("ASECFLAG", "asec_flag"),
        ("ASECWTH", "household_weight"),
        ("STATEFIP", "state_fips"),
        ("PERNUM", "person_number"),
        ("CPSIDP", "person_id"),
        ("CPSIDV", "person_id_versioned"),
        ("ASECWT", "person_weight"),
        ("RELATE", "relationship_to_reference_code"),
        ("AGE", "age"),
        ("SEX", "sex_code"),
        ("RACE", "race_code"),
        ("MARST", "marital_status_code"),
        ("FAMSIZE", "family_size"),
        ("HISPAN", "hispanic_code"),
        ("EMPSTAT", "employment_status_code"),
        ("LABFORCE", "labor_force_status_code"),
        ("EDUC", "education_code"),
        ("INCTOT", "total_income"),
        ("INCWAGE", "wage_income"),
        ("POVERTY", "poverty_percent_code"),
    )


def _acs_person_selected_spec() -> tuple[tuple[str, str], ...]:
    return (
        ("SERIALNO", "household_serial"),
        ("SPORDER", "person_number"),
        ("STATE", "state_code"),
        ("PUMA", "puma_code"),
        ("PWGTP", "person_weight"),
        ("AGEP", "age"),
        ("SEX", "sex_code"),
        ("RAC1P", "race_code"),
        ("HISP", "hispanic_code"),
        ("SCHL", "education_code"),
        ("RELSHIPP", "relationship_to_reference_code"),
        ("ESP", "parent_status_code"),
        ("MAR", "marital_status_code"),
        ("ESR", "employment_status_code"),
        ("WAGP", "wage_income"),
        ("PERNP", "earnings_income"),
        ("PINCP", "total_income"),
        ("POVPIP", "poverty_percent_ratio"),
    )


def _acs_housing_selected_spec() -> tuple[tuple[str, str], ...]:
    return (
        ("SERIALNO", "household_serial"),
        ("NP", "household_size"),
        ("HINCP", "household_income"),
        ("FINCP", "family_income"),
        ("WGTP", "household_weight"),
    )


def _sipp_labels(path: Path, columns: list[str]) -> dict[str, str]:
    try:
        import pyreadstat
    except ImportError:
        return {column: "" for column in columns}

    _, metadata = pyreadstat.read_dta(str(path), usecols=columns, row_limit=1)
    labels = getattr(metadata, "column_names_to_labels", {}) or {}
    return {column: str(labels.get(column, "")) for column in columns}


def _build_sipp_microdata(*, config: dict[str, Any], processed_root: Path) -> PublicMicrodataArtifact | None:
    root = _benchmark_dir(config, "sipp_dir")
    if root is None or not root.exists():
        return None
    source_path = _largest_matching_file(root, (".dta",))
    if source_path is None:
        return None

    selected_spec = _sipp_selected_spec()
    raw_columns = [raw for raw, _ in selected_spec]
    frame = pd.read_stata(
        source_path,
        columns=raw_columns,
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    frame = frame.rename(columns={raw: canonical for raw, canonical in selected_spec})
    frame.insert(0, "source_file", source_path.name)
    frame.insert(1, "source_dataset", "sipp_2024")

    labels = _sipp_labels(source_path, raw_columns)
    metadata = pd.DataFrame(
        [
            {
                "source": "sipp",
                "source_file": source_path.name,
                "raw_column": raw,
                "canonical_name": canonical,
                "label": labels.get(raw, ""),
            }
            for raw, canonical in selected_spec
        ]
    )
    parquet_path = processed_root / "sipp_selected_person_records.parquet"
    metadata_path = processed_root / "sipp_selected_person_records_metadata.csv"
    processed_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    return PublicMicrodataArtifact(
        source="sipp",
        source_path=source_path,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        row_count=int(len(frame.index)),
        column_count=int(len(frame.columns)),
    )


def _build_cps_microdata(*, config: dict[str, Any], processed_root: Path) -> PublicMicrodataArtifact | None:
    root = _benchmark_dir(config, "cps_asec_dir")
    if root is None or not root.exists():
        return None
    source_path = _largest_matching_file(root, (".csv.gz", ".csv"))
    if source_path is None:
        return None

    selected_spec = _cps_selected_spec()
    raw_columns = [raw for raw, _ in selected_spec]
    frame = pd.read_csv(source_path, usecols=raw_columns, compression="infer")
    frame = frame.rename(columns={raw: canonical for raw, canonical in selected_spec})
    frame.insert(0, "source_file", source_path.name)
    frame.insert(1, "source_dataset", "ipums_cps_asec")

    metadata = pd.DataFrame(
        [
            {
                "source": "cps_asec",
                "source_file": source_path.name,
                "raw_column": raw,
                "canonical_name": canonical,
                "label": f"Imported from IPUMS CPS ASEC extract column {raw}.",
            }
            for raw, canonical in selected_spec
        ]
    )
    parquet_path = processed_root / "cps_asec_selected_person_records.parquet"
    metadata_path = processed_root / "cps_asec_selected_person_records_metadata.csv"
    processed_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    return PublicMicrodataArtifact(
        source="cps_asec",
        source_path=source_path,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        row_count=int(len(frame.index)),
        column_count=int(len(frame.columns)),
    )


def _acs_zip_member_frames(zip_path: Path, raw_columns: list[str]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
        for member in members:
            with archive.open(member) as handle:
                frames.append(pd.read_csv(handle, usecols=raw_columns, low_memory=False))
    return frames


def _acs_zip_path(root: Path, pattern: str) -> Path | None:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    return matches[-1] if matches else None


def _build_acs_microdata(*, config: dict[str, Any], processed_root: Path) -> PublicMicrodataArtifact | None:
    root = _benchmark_dir(config, "acs_pums_dir")
    if root is None or not root.exists():
        return None

    person_zip = _acs_zip_path(root, "csv_p*.zip")
    housing_zip = _acs_zip_path(root, "csv_h*.zip")
    if person_zip is None or housing_zip is None:
        return None

    person_spec = _acs_person_selected_spec()
    housing_spec = _acs_housing_selected_spec()
    person_raw_columns = [raw for raw, _ in person_spec]
    housing_raw_columns = [raw for raw, _ in housing_spec]

    person_frames = _acs_zip_member_frames(person_zip, person_raw_columns)
    housing_frames = _acs_zip_member_frames(housing_zip, housing_raw_columns)
    if not person_frames or not housing_frames:
        return None

    person_frame = pd.concat(person_frames, ignore_index=True).rename(columns={raw: canonical for raw, canonical in person_spec})
    housing_frame = pd.concat(housing_frames, ignore_index=True).rename(
        columns={raw: canonical for raw, canonical in housing_spec}
    )
    frame = person_frame.merge(housing_frame, how="left", on="household_serial", validate="m:1")
    frame.insert(0, "source_file", person_zip.name)
    frame.insert(1, "source_dataset", "acs_pums_2024")

    metadata = pd.DataFrame(
        [
            {
                "source": "acs_pums",
                "source_file": person_zip.name,
                "raw_column": raw,
                "canonical_name": canonical,
                "label": f"Imported from ACS 2024 PUMS person file column {raw}.",
            }
            for raw, canonical in person_spec
        ]
        + [
            {
                "source": "acs_pums",
                "source_file": housing_zip.name,
                "raw_column": raw,
                "canonical_name": canonical,
                "label": f"Imported from ACS 2024 PUMS housing file column {raw}.",
            }
            for raw, canonical in housing_spec
        ]
    )
    parquet_path = processed_root / "acs_pums_selected_person_records.parquet"
    metadata_path = processed_root / "acs_pums_selected_person_records_metadata.csv"
    processed_root.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    return PublicMicrodataArtifact(
        source="acs_pums",
        source_path=person_zip,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        row_count=int(len(frame.index)),
        column_count=int(len(frame.columns)),
    )


def _coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    frame = pd.DataFrame({"value": _coerce_float(values), "weight": _coerce_float(weights)})
    frame = frame.dropna()
    frame = frame.loc[frame["weight"].gt(0)]
    if frame.empty:
        return None
    return float((frame["value"] * frame["weight"]).sum() / frame["weight"].sum())


def _weighted_share(indicator: pd.Series, weights: pd.Series) -> float | None:
    frame = pd.DataFrame({"value": _coerce_float(indicator), "weight": _coerce_float(weights)})
    frame = frame.dropna()
    frame = frame.loc[frame["weight"].gt(0)]
    if frame.empty:
        return None
    return float((frame["value"] * frame["weight"]).sum() / frame["weight"].sum())


def _sipp_employment_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(SIPP_RMESR_LABELS).astype("string")


def _sipp_employment_broad(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    result.loc[numeric.isin([1, 2, 3, 4, 5])] = "employed_some_or_all_month"
    result.loc[numeric.isin([6, 7])] = "not_employed_search_or_layoff"
    result.loc[numeric.eq(8)] = "not_employed_no_search_or_layoff"
    return result


def _sipp_employed_any(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[numeric.isin([1, 2, 3, 4, 5])] = True
    result.loc[numeric.isin([6, 7, 8])] = False
    return result


def _acs_employed_any(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[numeric.isin([1, 2, 4, 5])] = True
    result.loc[numeric.isin([3, 6])] = False
    return result


def _cps_employed_any(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[numeric.isin([10, 12])] = True
    result.loc[numeric.ge(20)] = False
    return result


def _clean_binary_employment(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.mask(numeric < 0)
    numeric = numeric.mask(numeric == 0, 0)
    numeric = numeric.mask(numeric.isin([1, 2]), 1)
    return numeric


def _harmonize_sipp_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    age = _coerce_float(frame["age"])
    weight = _coerce_float(frame["person_weight"])
    poverty_ratio = _coerce_float(frame["family_income_poverty_ratio"])
    employment_status = _coerce_float(frame["monthly_employment_status_code"])
    harmonized = pd.DataFrame(
        {
            "source": "sipp",
            "source_dataset": frame["source_dataset"],
            "reference_year": 2023,
            "measure_period": "monthly",
            "person_weight": weight,
            "age": age,
            "adult_window_primary": age.between(25, 40, inclusive="both"),
            "sex": map_public_sex(frame["sex_code"]),
            "female": _coerce_float(frame["sex_code"]).eq(2),
            "race_ethnicity_3cat": map_public_race_ethnicity_3cat(
                race=frame["race_code"],
                hispanic=frame["hispanic_code"],
                source="sipp",
            ),
            "education_code": _coerce_float(frame["education_code"]),
            "relationship_to_reference_code": _coerce_float(frame["relationship_to_reference_code"]),
            "marital_path_code": _coerce_float(frame["marital_path_code"]),
            "employment_status_code": employment_status,
            "employment": _sipp_employed_any(frame["monthly_employment_status_code"]),
            "earnings": _coerce_float(frame["monthly_personal_earnings"]),
            "person_income": _coerce_float(frame["monthly_person_total_income"]),
            "household_income": _coerce_float(frame["monthly_household_total_income"]),
            "family_income": _coerce_float(frame["monthly_family_total_income"]),
            "poverty_ratio": poverty_ratio,
            "below_poverty": poverty_ratio.lt(1),
            "poverty_band": pd.Series(pd.NA, index=frame.index, dtype="string"),
        }
    )
    harmonized["poverty_band"] = build_poverty_band_from_ratio(
        poverty_ratio,
        missing_label=pd.NA,
    )
    return harmonized


def _harmonize_cps_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    age = _coerce_float(frame["age"])
    poverty_code = _coerce_float(frame["poverty_percent_code"])
    poverty_band = pd.Series(pd.NA, index=frame.index, dtype="string")
    poverty_band.loc[poverty_code.eq(10)] = "BELOW_100_PCT"
    poverty_band.loc[poverty_code.eq(21)] = "100_124_PCT"
    poverty_band.loc[poverty_code.eq(22)] = "125_149_PCT"
    poverty_band.loc[poverty_code.eq(23)] = "150_PLUS_PCT"
    return pd.DataFrame(
        {
            "source": "cps_asec",
            "source_dataset": frame["source_dataset"],
            "reference_year": _coerce_float(frame["survey_year"]),
            "measure_period": "annual",
            "person_weight": _coerce_float(frame["person_weight"]),
            "age": age,
            "adult_window_primary": age.between(25, 40, inclusive="both"),
            "sex": map_public_sex(frame["sex_code"]),
            "female": _coerce_float(frame["sex_code"]).eq(2),
            "race_ethnicity_3cat": map_public_race_ethnicity_3cat(
                race=frame["race_code"],
                hispanic=frame["hispanic_code"],
                source="cps_asec",
            ),
            "education_code": _coerce_float(frame["education_code"]),
            "relationship_to_reference_code": _coerce_float(frame["relationship_to_reference_code"]),
            "marital_path_code": _coerce_float(frame["marital_status_code"]),
            "employment_status_code": _coerce_float(frame["employment_status_code"]),
            "employment": _cps_employed_any(frame["employment_status_code"]),
            "earnings": _coerce_float(frame["wage_income"]),
            "person_income": _coerce_float(frame["total_income"]),
            "household_income": pd.Series([pd.NA] * len(frame.index), index=frame.index, dtype="Float64"),
            "family_income": pd.Series([pd.NA] * len(frame.index), index=frame.index, dtype="Float64"),
            "poverty_ratio": poverty_code,
            "below_poverty": poverty_code.eq(10),
            "poverty_band": poverty_band,
        }
    )


def _harmonize_acs_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    age = _coerce_float(frame["age"])
    poverty_percent = _coerce_float(frame["poverty_percent_ratio"])
    poverty_ratio = poverty_percent / 100
    return pd.DataFrame(
        {
            "source": "acs_pums",
            "source_dataset": frame["source_dataset"],
            "reference_year": 2024,
            "measure_period": "annual",
            "person_weight": _coerce_float(frame["person_weight"]),
            "age": age,
            "adult_window_primary": age.between(25, 40, inclusive="both"),
            "sex": map_public_sex(frame["sex_code"]),
            "female": _coerce_float(frame["sex_code"]).eq(2),
            "race_ethnicity_3cat": map_public_race_ethnicity_3cat(
                race=frame["race_code"],
                hispanic=frame["hispanic_code"],
                source="acs_pums",
            ),
            "education_code": _coerce_float(frame["education_code"]),
            "relationship_to_reference_code": _coerce_float(frame["relationship_to_reference_code"]),
            "marital_path_code": _coerce_float(frame["marital_status_code"]),
            "employment_status_code": _coerce_float(frame["employment_status_code"]),
            "employment": _acs_employed_any(frame["employment_status_code"]),
            "earnings": _coerce_float(frame["earnings_income"]),
            "person_income": _coerce_float(frame["total_income"]),
            "household_income": _coerce_float(frame["household_income"]),
            "family_income": _coerce_float(frame["family_income"]),
            "poverty_ratio": poverty_ratio,
            "below_poverty": poverty_percent.lt(100),
            "poverty_band": build_poverty_band_from_percent(poverty_percent, missing_label=pd.NA),
        }
    )


def _standardize_public_profile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return standardize_public_profile_frame(frame)


def _income_band(series: pd.Series) -> pd.Series:
    numeric = _coerce_float(series)
    result = pd.Series("missing", index=series.index, dtype="string")
    observed = numeric.dropna()
    if observed.empty:
        return result
    if observed.nunique() >= 4:
        ranked = observed.rank(method="first")
        bands = pd.qcut(ranked, 4, labels=["q1_low", "q2", "q3", "q4_high"])
        result.loc[bands.index] = bands.astype("string")
        return result
    median = float(observed.median())
    result.loc[numeric.notna() & numeric.le(median)] = "lower_or_equal_median"
    result.loc[numeric.notna() & numeric.gt(median)] = "above_median"
    return result


def _acs_child_context(
    frame: pd.DataFrame,
    *,
    processed_root: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    def _fmt_metric(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "unavailable"

    child = frame.copy()
    child["age"] = _coerce_float(child["age"])
    child["parent_status_code"] = _coerce_float(child["parent_status_code"])
    child = child.loc[child["age"].lt(18) & child["parent_status_code"].isin(ACS_ESP_LABELS)].copy()
    child["sex"] = map_public_sex(child["sex_code"])
    child["race_ethnicity_3cat"] = map_public_race_ethnicity_3cat(
        race=child["race_code"],
        hispanic=child["hispanic_code"],
        source="acs_pums",
    )
    child["parent_status_label"] = child["parent_status_code"].round().astype("Int64").map(ACS_ESP_LABELS).astype("string")
    child["father_presence_group"] = pd.Series(pd.NA, index=child.index, dtype="string")
    child.loc[child["parent_status_code"].isin([1, 2, 3, 4]), "father_presence_group"] = "two_parents"
    child.loc[child["parent_status_code"].isin([5, 6]), "father_presence_group"] = "father_only"
    child.loc[child["parent_status_code"].isin([7, 8]), "father_presence_group"] = "mother_only"
    child["resident_father_present_proxy"] = child["parent_status_code"].isin([1, 2, 3, 4, 5, 6]).astype("boolean")
    child["resident_father_absent_proxy"] = child["parent_status_code"].isin([7, 8]).astype("boolean")
    poverty_percent = _coerce_float(child["poverty_percent_ratio"])
    child["poverty_band"] = build_poverty_band_from_percent(
        poverty_percent,
        missing_label="missing",
        lowercase=True,
    )
    child["household_income_band"] = _income_band(child["household_income"])
    child["sex_x_race_ethnicity"] = child["sex"].fillna("missing") + " | " + child["race_ethnicity_3cat"].fillna("missing")

    context_path = processed_root / "acs_child_father_presence_context.parquet"
    child.to_parquet(context_path, index=False)

    def _summary(group_col: str, label: str | None = None) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for group_value, subset in child.groupby(group_col, dropna=False):
            weights = subset["person_weight"]
            rows.append(
                {
                    "group_type": label or group_col,
                    "group_value": group_value,
                    "row_count": int(len(subset.index)),
                    "weighted_children": float(_coerce_float(weights).sum(skipna=True)),
                    "father_present_share": _weighted_share(subset["resident_father_present_proxy"], weights),
                    "father_absent_share": _weighted_share(subset["resident_father_absent_proxy"], weights),
                    "two_parent_share": _weighted_share(subset["father_presence_group"].eq("two_parents"), weights),
                    "father_only_share": _weighted_share(subset["father_presence_group"].eq("father_only"), weights),
                    "mother_only_share": _weighted_share(subset["father_presence_group"].eq("mother_only"), weights),
                    "mean_household_income": _weighted_mean(subset["household_income"], weights),
                }
            )
        return pd.DataFrame(rows)

    summary = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "group_type": "overall",
                        "group_value": "overall",
                        "row_count": int(len(child.index)),
                        "weighted_children": float(_coerce_float(child["person_weight"]).sum(skipna=True)),
                        "father_present_share": _weighted_share(child["resident_father_present_proxy"], child["person_weight"]),
                        "father_absent_share": _weighted_share(child["resident_father_absent_proxy"], child["person_weight"]),
                        "two_parent_share": _weighted_share(child["father_presence_group"].eq("two_parents"), child["person_weight"]),
                        "father_only_share": _weighted_share(child["father_presence_group"].eq("father_only"), child["person_weight"]),
                        "mother_only_share": _weighted_share(child["father_presence_group"].eq("mother_only"), child["person_weight"]),
                        "mean_household_income": _weighted_mean(child["household_income"], child["person_weight"]),
                    }
                ]
            ),
            _summary("sex"),
            _summary("race_ethnicity_3cat"),
            _summary("sex_x_race_ethnicity"),
            _summary("poverty_band"),
            _summary("household_income_band"),
        ],
        ignore_index=True,
    )
    summary_path = output_dir / "acs_child_father_presence_summary.csv"
    summary.to_csv(summary_path, index=False)

    race_rows = summary.loc[summary["group_type"] == "race_ethnicity_3cat"].sort_values(
        "father_absent_share", ascending=False
    )
    top_race = race_rows.iloc[0] if not race_rows.empty else None
    low_race = race_rows.iloc[-1] if len(race_rows.index) > 1 else None
    poverty_rows = summary.loc[summary["group_type"] == "poverty_band"].sort_values(
        "father_absent_share", ascending=False
    )
    top_poverty = poverty_rows.iloc[0] if not poverty_rows.empty else None
    report_lines = [
        "# ACS Child Father Presence Context",
        "",
        "Universe: children under age 18 with non-missing ACS PUMS `ESP`, which covers own children of the householder and children in subfamilies.",
        "Interpretation: `father_absent_share` here is a public-use proxy for children coded as living with mother only, not a full census of all fatherlessness arrangements.",
        "",
        f"- Child rows in scope: {len(child.index)}",
        f"- Weighted father-present share: {_fmt_metric(_weighted_share(child['resident_father_present_proxy'], child['person_weight']))}",
        f"- Weighted father-absent share: {_fmt_metric(_weighted_share(child['resident_father_absent_proxy'], child['person_weight']))}",
        f"- Weighted two-parent share: {_fmt_metric(_weighted_share(child['father_presence_group'].eq('two_parents'), child['person_weight']))}",
        f"- Weighted father-only share: {_fmt_metric(_weighted_share(child['father_presence_group'].eq('father_only'), child['person_weight']))}",
        f"- Weighted mother-only share: {_fmt_metric(_weighted_share(child['father_presence_group'].eq('mother_only'), child['person_weight']))}",
        f"- Highest race father-absent share: {top_race['group_value']} = {top_race['father_absent_share']:.4f}" if top_race is not None else "- Highest race father-absent share: unavailable",
        f"- Lowest race father-absent share: {low_race['group_value']} = {low_race['father_absent_share']:.4f}" if low_race is not None else "- Lowest race father-absent share: unavailable",
        f"- Highest poverty-band father-absent share: {top_poverty['group_value']} = {top_poverty['father_absent_share']:.4f}" if top_poverty is not None else "- Highest poverty-band father-absent share: unavailable",
        "",
        "Artifacts:",
        f"- Context parquet: `{context_path.name}`",
        f"- Summary CSV: `{summary_path.name}`",
    ]
    report_path = output_dir / "acs_child_father_presence_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return context_path, summary_path, report_path


def build_public_benchmark_profiles(
    *,
    config: dict[str, Any],
    processed_root: Path,
    output_dir: Path,
) -> PublicBenchmarkProfileResult:
    microdata_root = processed_root / "public_microdata"
    microdata_result = build_public_microdata_artifacts(
        config=config,
        processed_root=microdata_root,
        output_dir=output_dir,
    )

    profile_frames: list[pd.DataFrame] = []
    acs_frame: pd.DataFrame | None = None
    for artifact in microdata_result.artifacts:
        if artifact.source == "acs_pums":
            acs_frame = pd.read_parquet(artifact.parquet_path)
            profile_frames.append(_standardize_public_profile_frame(_harmonize_acs_profiles(acs_frame)))
        elif artifact.source == "sipp":
            profile_frames.append(_standardize_public_profile_frame(_harmonize_sipp_profiles(pd.read_parquet(artifact.parquet_path))))
        elif artifact.source == "cps_asec":
            profile_frames.append(_standardize_public_profile_frame(_harmonize_cps_profiles(pd.read_parquet(artifact.parquet_path))))
    profiles = pd.concat(profile_frames, ignore_index=True)
    profiles_path = processed_root / "public_microdata" / "public_benchmark_profiles.parquet"
    profiles.to_parquet(profiles_path, index=False)
    acs_child_context_path = processed_root / "public_microdata" / "acs_child_father_presence_context.parquet"
    acs_child_summary_path = output_dir / "acs_child_father_presence_summary.csv"
    acs_child_report_path = output_dir / "acs_child_father_presence_report.md"
    if acs_frame is not None:
        acs_child_context_path, acs_child_summary_path, acs_child_report_path = _acs_child_context(
            acs_frame,
            processed_root=processed_root / "public_microdata",
            output_dir=output_dir,
        )

    sipp_employment_codebook_path = output_dir / "sipp_employment_status_codebook.csv"
    pd.DataFrame(
        [
            {
                "code": code,
                "label": label,
                "broad_status": (
                    "employed_some_or_all_month"
                    if code in {1, 2, 3, 4, 5}
                    else "not_employed_search_or_layoff"
                    if code in {6, 7}
                    else "not_employed_no_search_or_layoff"
                ),
                "employed_any": code in {1, 2, 3, 4, 5},
                "source_note": "Official Census SIPP RMESR definitions from the core-file data dictionary.",
            }
            for code, label in SIPP_RMESR_LABELS.items()
        ]
    ).to_csv(sipp_employment_codebook_path, index=False)

    mapping_rows = [
        {
            "source": "acs_pums",
            "canonical_name": "earnings",
            "source_column": "earnings_income",
            "status": "ready",
            "note": "Annual personal earnings from ACS PUMS `PERNP` for the past 12 months.",
        },
        {
            "source": "acs_pums",
            "canonical_name": "person_income",
            "source_column": "total_income",
            "status": "ready",
            "note": "Annual total personal income from ACS PUMS `PINCP` for the past 12 months.",
        },
        {
            "source": "acs_pums",
            "canonical_name": "household_income",
            "source_column": "household_income",
            "status": "ready",
            "note": "Annual household income from ACS PUMS housing file `HINCP`, merged by `SERIALNO`.",
        },
        {
            "source": "acs_pums",
            "canonical_name": "below_poverty",
            "source_column": "poverty_percent_ratio",
            "status": "ready",
            "note": "Derived from ACS PUMS `POVPIP`; observations below 100 are below poverty.",
        },
        {
            "source": "acs_pums",
            "canonical_name": "child_father_presence_proxy",
            "source_column": "parent_status_code",
            "status": "ready_child_universe_only",
            "note": "Derived from ACS PUMS `ESP` for children under 18 with valid `ESP`; codes 1-6 imply resident father present, 7-8 imply living with mother only.",
        },
        {
            "source": "sipp",
            "canonical_name": "person_income",
            "source_column": "monthly_person_total_income",
            "status": "ready",
            "note": "Monthly total personal income from the 2024 SIPP file covering the January-December 2023 reference period.",
        },
        {
            "source": "sipp",
            "canonical_name": "poverty_ratio",
            "source_column": "family_income_poverty_ratio",
            "status": "ready",
            "note": "Continuous family income-to-poverty ratio; values below 1 indicate below-poverty observations.",
        },
        {
            "source": "sipp",
            "canonical_name": "employment_status_code",
            "source_column": "monthly_employment_status_code",
            "status": "ready",
            "note": "Monthly employment-status recode decoded from official Census RMESR labels; see `sipp_employment_status_codebook.csv`.",
        },
        {
            "source": "sipp",
            "canonical_name": "employment",
            "source_column": "monthly_employment_status_code",
            "status": "derived_from_official_codebook",
            "note": "Derived employed-any indicator treating RMESR codes 1-5 as employed during at least part of the month.",
        },
        {
            "source": "cps_asec",
            "canonical_name": "person_income",
            "source_column": "total_income",
            "status": "ready",
            "note": "Annual total income from the prior calendar year in the IPUMS CPS ASEC extract.",
        },
        {
            "source": "cps_asec",
            "canonical_name": "poverty_band",
            "source_column": "poverty_percent_code",
            "status": "ready",
            "note": "Four-category official poverty interval using the observed ASEC codes 10, 21, 22, and 23.",
        },
        {
            "source": "cps_asec",
            "canonical_name": "below_poverty",
            "source_column": "poverty_percent_code",
            "status": "inferred_from_official_categories",
            "note": "Treated code 10 as below poverty based on the IPUMS CPS official four-category poverty definition for 1976+ samples.",
        },
    ]
    mapping_path = output_dir / "public_benchmark_mapping.csv"
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    primary = profiles.loc[profiles["adult_window_primary"]].copy()
    summary_rows: list[dict[str, object]] = []
    for keys, group in primary.groupby(["source", "reference_year", "measure_period"], dropna=False):
        source, reference_year, measure_period = keys
        summary_rows.append(
            {
                "source": source,
                "reference_year": int(reference_year) if pd.notna(reference_year) else pd.NA,
                "measure_period": measure_period,
                "row_count": int(len(group.index)),
                "weighted_population": float(_coerce_float(group["person_weight"]).sum(skipna=True)),
                "weighted_female_share": _weighted_share(group["female"], group["person_weight"]),
                "weighted_employment_share": _weighted_share(group["employment"], group["person_weight"]),
                "weighted_mean_earnings": _weighted_mean(group["earnings"], group["person_weight"]),
                "weighted_mean_person_income": _weighted_mean(group["person_income"], group["person_weight"]),
                "weighted_poverty_share": _weighted_share(group["below_poverty"], group["person_weight"]),
            }
        )
    summary_frame = pd.DataFrame(summary_rows).sort_values(["source", "reference_year"]).reset_index(drop=True)
    summary_path = output_dir / "public_benchmark_profile_summary.csv"
    summary_frame.to_csv(summary_path, index=False)

    subgroup_rows: list[dict[str, object]] = []
    for keys, group in primary.groupby(["source", "reference_year", "sex", "race_ethnicity_3cat"], dropna=False):
        source, reference_year, sex, race_ethnicity = keys
        subgroup_rows.append(
            {
                "source": source,
                "reference_year": int(reference_year) if pd.notna(reference_year) else pd.NA,
                "sex": sex,
                "race_ethnicity_3cat": race_ethnicity,
                "row_count": int(len(group.index)),
                "weighted_population": float(_coerce_float(group["person_weight"]).sum(skipna=True)),
                "weighted_employment_share": _weighted_share(group["employment"], group["person_weight"]),
                "weighted_mean_earnings": _weighted_mean(group["earnings"], group["person_weight"]),
                "weighted_mean_person_income": _weighted_mean(group["person_income"], group["person_weight"]),
                "weighted_poverty_share": _weighted_share(group["below_poverty"], group["person_weight"]),
            }
        )
    subgroup_frame = pd.DataFrame(subgroup_rows).sort_values(
        ["source", "reference_year", "sex", "race_ethnicity_3cat"]
    ).reset_index(drop=True)
    subgroup_summary_path = output_dir / "public_benchmark_subgroup_summary.csv"
    subgroup_frame.to_csv(subgroup_summary_path, index=False)

    return PublicBenchmarkProfileResult(
        profiles_path=profiles_path,
        mapping_path=mapping_path,
        summary_path=summary_path,
        subgroup_summary_path=subgroup_summary_path,
        sipp_employment_codebook_path=sipp_employment_codebook_path,
        acs_child_context_path=acs_child_context_path,
        acs_child_summary_path=acs_child_summary_path,
        acs_child_report_path=acs_child_report_path,
        row_count=int(len(profiles.index)),
    )


def _nlsy97_benchmark_profiles(processed_root: Path) -> pd.DataFrame:
    nlsy97 = pd.read_parquet(processed_root / "nlsy" / "nlsy97_analysis_ready.parquet").copy()
    age_2021 = 2021 - _coerce_float(nlsy97["birth_year"])
    employment = _clean_binary_employment(nlsy97["employment_2021"])
    sex = map_public_sex(nlsy97["sex_raw"])
    base = pd.DataFrame(
        {
            "source": "nlsy97",
            "source_dataset": "nlsy97",
            "source_group": "overall",
            "reference_year": 2021,
            "measure_period": "annual",
            "weighting_method": "unweighted",
            "person_weight": pd.Series([pd.NA] * len(nlsy97.index), index=nlsy97.index, dtype="Float64"),
            "age": age_2021,
            "adult_window_primary": age_2021.between(30, 40, inclusive="both"),
            "sex": sex,
            "female": _coerce_float(nlsy97["sex_raw"]).eq(2),
            "race_ethnicity_3cat": nlsy97["race_ethnicity_3cat"].astype("string"),
            "earnings": _coerce_float(nlsy97["annual_earnings_2021_clean"]),
            "person_income": pd.Series([pd.NA] * len(nlsy97.index), index=nlsy97.index, dtype="Float64"),
            "household_income": _coerce_float(nlsy97["household_income_2021_clean"]),
            "employment": employment,
            "below_poverty": pd.Series([pd.NA] * len(nlsy97.index), index=nlsy97.index, dtype="boolean"),
        }
    )
    profiles = [base]
    observed = nlsy97["primary_treatment_label_nlsy97"].notna()
    for label in ("resident_bio_father_present", "resident_bio_father_absent"):
        subset = base.loc[observed & (nlsy97["primary_treatment_label_nlsy97"] == label)].copy()
        subset["source_group"] = label
        profiles.append(subset)
    return pd.concat(profiles, ignore_index=True)


def _standardize_comparison_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    standardized = frame.reindex(columns=columns).copy()
    dtype_map = {
        "source": "string",
        "source_dataset": "string",
        "source_group": "string",
        "reference_year": "Int64",
        "measure_period": "string",
        "weighting_method": "string",
        "person_weight": "Float64",
        "age": "Float64",
        "adult_window_primary": "boolean",
        "sex": "string",
        "female": "boolean",
        "race_ethnicity_3cat": "string",
        "earnings": "Float64",
        "person_income": "Float64",
        "household_income": "Float64",
        "employment": "Float64",
        "below_poverty": "boolean",
    }
    for column, dtype in dtype_map.items():
        standardized[column] = standardized[column].astype(dtype)
    return standardized


def _summarize_profile_group(frame: pd.DataFrame, *, weighted: bool) -> dict[str, object]:
    result: dict[str, object] = {"row_count": int(len(frame.index))}
    if weighted:
        weights = frame["person_weight"]
        result["population"] = float(_coerce_float(weights).sum(skipna=True))
        result["female_share"] = _weighted_share(frame["female"], weights)
        result["mean_earnings"] = _weighted_mean(frame["earnings"], weights)
        result["mean_person_income"] = _weighted_mean(frame["person_income"], weights)
        result["mean_household_income"] = _weighted_mean(frame["household_income"], weights)
        result["employment_rate"] = _weighted_share(frame["employment"], weights) if "employment" in frame.columns else None
        result["poverty_share"] = _weighted_share(frame["below_poverty"], weights)
    else:
        result["population"] = pd.NA
        result["female_share"] = float(frame["female"].mean()) if frame["female"].notna().any() else None
        result["mean_earnings"] = float(_coerce_float(frame["earnings"]).mean()) if frame["earnings"].notna().any() else None
        result["mean_person_income"] = float(_coerce_float(frame["person_income"]).mean()) if frame["person_income"].notna().any() else None
        result["mean_household_income"] = float(_coerce_float(frame["household_income"]).mean()) if frame["household_income"].notna().any() else None
        result["employment_rate"] = float(_coerce_float(frame["employment"]).mean()) if frame["employment"].notna().any() else None
        result["poverty_share"] = float(_coerce_float(frame["below_poverty"]).mean()) if frame["below_poverty"].notna().any() else None
    return result


def build_cross_cohort_benchmark_comparison(
    *,
    config: dict[str, Any],
    processed_root: Path,
    output_dir: Path,
) -> CrossCohortBenchmarkResult:
    benchmark_result = build_public_benchmark_profiles(
        config=config,
        processed_root=processed_root,
        output_dir=output_dir,
    )
    benchmark_profiles = pd.read_parquet(benchmark_result.profiles_path).copy()
    benchmark_profiles["source_group"] = benchmark_profiles["source"].astype("string")
    benchmark_profiles["weighting_method"] = "person_weighted"
    benchmark_profiles["employment"] = pd.Series([pd.NA] * len(benchmark_profiles.index), index=benchmark_profiles.index, dtype="boolean")

    nlsy97_profiles = _nlsy97_benchmark_profiles(processed_root)

    benchmark_primary = benchmark_profiles.loc[benchmark_profiles["adult_window_primary"]].copy()
    annual_bench = benchmark_primary.loc[benchmark_primary["measure_period"] == "annual"].copy()
    annual_acs = annual_bench.loc[annual_bench["source"] == "acs_pums"].copy()
    annual_acs["source_group"] = "acs_pums_2024_context"
    annual_acs["reference_year"] = 2024
    pooled_cps = annual_bench.loc[annual_bench["source"] == "cps_asec"].copy()
    pooled_cps["source_group"] = "cps_asec_2023_2025_pooled"
    pooled_cps["reference_year"] = 2023
    monthly_sipp = benchmark_primary.loc[benchmark_primary["source"] == "sipp"].copy()
    monthly_sipp["source_group"] = "sipp_2023_monthly_context"
    monthly_sipp["reference_year"] = 2023

    comparison_columns = [
        "source",
        "source_dataset",
        "source_group",
        "reference_year",
        "measure_period",
        "weighting_method",
        "person_weight",
        "age",
        "adult_window_primary",
        "sex",
        "female",
        "race_ethnicity_3cat",
        "earnings",
        "person_income",
        "household_income",
        "employment",
        "below_poverty",
    ]
    comparison_profiles = pd.concat(
        [
            _standardize_comparison_frame(
                nlsy97_profiles.loc[nlsy97_profiles["adult_window_primary"], comparison_columns].copy(),
                comparison_columns,
            ),
            _standardize_comparison_frame(annual_acs.loc[:, comparison_columns].copy(), comparison_columns),
            _standardize_comparison_frame(pooled_cps.loc[:, comparison_columns].copy(), comparison_columns),
            _standardize_comparison_frame(monthly_sipp.loc[:, comparison_columns].copy(), comparison_columns),
        ],
        ignore_index=True,
    )
    profiles_path = processed_root / "public_microdata" / "cross_cohort_benchmark_profiles.parquet"
    comparison_profiles.to_parquet(profiles_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for keys, group in comparison_profiles.groupby(
        ["source", "source_group", "reference_year", "measure_period", "weighting_method"], dropna=False
    ):
        source, source_group, reference_year, measure_period, weighting_method = keys
        metrics = _summarize_profile_group(group, weighted=weighting_method == "person_weighted")
        summary_rows.append(
            {
                "source": source,
                "source_group": source_group,
                "reference_year": reference_year,
                "measure_period": measure_period,
                "weighting_method": weighting_method,
                **metrics,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["measure_period", "source", "source_group", "reference_year"]
    ).reset_index(drop=True)
    summary_path = output_dir / "cross_cohort_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    subgroup_rows: list[dict[str, object]] = []
    for keys, group in comparison_profiles.groupby(
        ["source", "source_group", "reference_year", "measure_period", "weighting_method", "sex", "race_ethnicity_3cat"],
        dropna=False,
    ):
        source, source_group, reference_year, measure_period, weighting_method, sex, race_ethnicity = keys
        metrics = _summarize_profile_group(group, weighted=weighting_method == "person_weighted")
        subgroup_rows.append(
            {
                "source": source,
                "source_group": source_group,
                "reference_year": reference_year,
                "measure_period": measure_period,
                "weighting_method": weighting_method,
                "sex": sex,
                "race_ethnicity_3cat": race_ethnicity,
                **metrics,
            }
        )
    subgroup_summary = pd.DataFrame(subgroup_rows).sort_values(
        ["measure_period", "source", "source_group", "sex", "race_ethnicity_3cat"]
    ).reset_index(drop=True)
    subgroup_summary_path = output_dir / "cross_cohort_benchmark_subgroup_summary.csv"
    subgroup_summary.to_csv(subgroup_summary_path, index=False)

    def _row(source_group: str) -> pd.Series:
        return summary.loc[summary["source_group"] == source_group].iloc[0]

    overall = _row("overall")
    present = _row("resident_bio_father_present")
    absent = _row("resident_bio_father_absent")
    cps_pool = _row("cps_asec_2023_2025_pooled")
    sipp_context = _row("sipp_2023_monthly_context")
    acs_available = "acs_pums_2024_context" in set(summary["source_group"])
    acs_context = _row("acs_pums_2024_context") if acs_available else None

    report_lines = [
        "# Cross-Cohort Benchmark Comparison",
        "",
        "Headline layer: `nlsy97` age 30-40 in 2021 against pooled annual CPS ASEC 2023-2025 benchmarks.",
        "SIPP 2023 is reported separately as monthly context and is not directly annual-comparable to NLSY97 or CPS ASEC earnings.",
        "",
        f"- NLSY97 adult-window rows: {int(overall['row_count'])}",
        f"- NLSY97 mean 2021 annual earnings: {overall['mean_earnings']:.2f}",
        f"- NLSY97 father-present mean 2021 annual earnings: {present['mean_earnings']:.2f}",
        f"- NLSY97 father-absent mean 2021 annual earnings: {absent['mean_earnings']:.2f}",
        f"- NLSY97 present-minus-absent earnings gap: {(present['mean_earnings'] - absent['mean_earnings']):.2f}",
        f"- CPS ASEC pooled 2023-2025 weighted mean annual wage income: {cps_pool['mean_earnings']:.2f}",
        f"- CPS ASEC pooled 2023-2025 weighted poverty share: {cps_pool['poverty_share']:.4f}",
        f"- SIPP 2023 weighted mean monthly earnings: {sipp_context['mean_earnings']:.2f}",
        f"- SIPP 2023 weighted poverty share: {sipp_context['poverty_share']:.4f}",
        "",
        "Interpretation notes:",
        "- `nlsy97` rows are unweighted cohort means; ACS PUMS, CPS ASEC, and SIPP rows are person-weighted public benchmarks.",
        "- ACS PUMS and CPS ASEC are annual context benchmarks for the adult-window NLSY97 earnings and income outputs.",
        "- The SIPP comparison is monthly context only; do not interpret it as a direct annual earnings benchmark.",
    ]
    if acs_context is not None:
        report_lines[9:9] = [
            f"- ACS PUMS 2024 weighted mean annual earnings: {acs_context['mean_earnings']:.2f}",
            f"- ACS PUMS 2024 weighted mean household income: {acs_context['mean_household_income']:.2f}",
            f"- ACS PUMS 2024 weighted poverty share: {acs_context['poverty_share']:.4f}",
        ]
    report_path = output_dir / "cross_cohort_benchmark_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return CrossCohortBenchmarkResult(
        profiles_path=profiles_path,
        summary_path=summary_path,
        subgroup_summary_path=subgroup_summary_path,
        report_path=report_path,
        row_count=int(len(comparison_profiles.index)),
    )


def build_fred_snapshot(*, config: dict[str, Any], raw_root: Path, processed_root: Path) -> SourceBuildResult:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not configured.")

    specs = _read_source_specs(config, "fred", FRED_SERIES_DEFAULTS)
    metadata_rows: list[dict[str, Any]] = []
    observation_frames: list[pd.DataFrame] = []
    raw_payload = {"source": "fred", "fetched_at": datetime.now(timezone.utc).isoformat(), "series": []}

    for spec in specs:
        series_id = spec["series_id"]
        meta_payload = _http_get_json(
            f"{FRED_BASE_URL}/series",
            {"series_id": series_id, "api_key": api_key, "file_type": "json"},
        )
        obs_payload = _http_get_json(
            f"{FRED_BASE_URL}/series/observations",
            {"series_id": series_id, "api_key": api_key, "file_type": "json"},
        )
        raw_payload["series"].append(
            {"series_id": series_id, "metadata": meta_payload, "observations": obs_payload}
        )
        for item in meta_payload.get("seriess", []):
            metadata_rows.append(
                {
                    "series_id": series_id,
                    "canonical_name": spec.get("canonical_name", ""),
                    "category": spec.get("category", ""),
                    "title": item.get("title", ""),
                    "units": item.get("units", ""),
                    "frequency": item.get("frequency", ""),
                    "seasonal_adjustment": item.get("seasonal_adjustment", ""),
                    "notes": item.get("notes", ""),
                }
            )
        obs_frame = pd.DataFrame(obs_payload.get("observations", []))
        if obs_frame.empty:
            continue
        obs_frame.insert(0, "series_id", series_id)
        obs_frame.insert(1, "canonical_name", spec.get("canonical_name", ""))
        obs_frame.insert(2, "category", spec.get("category", ""))
        obs_frame["value_numeric"] = _coerce_numeric(obs_frame["value"])
        observation_frames.append(obs_frame)

    if not observation_frames:
        raise RuntimeError("FRED returned no observations for the configured series.")

    observations = pd.concat(observation_frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows).drop_duplicates()
    raw_json_path = _write_sanitized_json(raw_payload, source="fred", raw_root=raw_root, secrets=[api_key])
    observations_path, metadata_path = _write_frame_bundle(
        observations,
        metadata,
        source="fred",
        processed_root=processed_root,
    )
    return SourceBuildResult(
        source="fred",
        raw_json_path=raw_json_path,
        observations_path=observations_path,
        metadata_path=metadata_path,
        row_count=int(len(observations.index)),
    )


def _year_chunks(start_year: int, end_year: int, *, max_span: int = 20) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    current = start_year
    while current <= end_year:
        upper = min(current + max_span - 1, end_year)
        windows.append((current, upper))
        current = upper + 1
    return windows


def build_bls_snapshot(*, config: dict[str, Any], raw_root: Path, processed_root: Path) -> SourceBuildResult:
    api_key = os.environ.get("BLS_API_KEY")
    if not api_key:
        raise RuntimeError("BLS_API_KEY is not configured.")

    specs = _read_source_specs(config, "bls", BLS_SERIES_DEFAULTS)
    series_ids = [item["series_id"] for item in specs]
    current_year = datetime.now(timezone.utc).year
    raw_payload = {"source": "bls", "fetched_at": datetime.now(timezone.utc).isoformat(), "responses": []}
    metadata_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    canonical_by_id = {item["series_id"]: item for item in specs}

    for start_year, end_year in _year_chunks(1980, current_year):
        payload = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
            "registrationkey": api_key,
            "catalog": True,
        }
        response = _http_post_json(BLS_BASE_URL, payload)
        raw_payload["responses"].append(response)
        series_rows = response.get("Results", {}).get("series", [])
        for item in series_rows:
            series_id = item.get("seriesID", "")
            spec = canonical_by_id.get(series_id, {})
            catalog = item.get("catalog", {})
            metadata_rows.append(
                {
                    "series_id": series_id,
                    "canonical_name": spec.get("canonical_name", ""),
                    "category": spec.get("category", ""),
                    "series_title": catalog.get("series_title", ""),
                    "survey_name": catalog.get("survey_name", ""),
                    "seasonality": catalog.get("seasonality", ""),
                    "measure_data_type": catalog.get("measure_data_type", ""),
                }
            )
            for obs in item.get("data", []):
                row = {
                    "series_id": series_id,
                    "canonical_name": spec.get("canonical_name", ""),
                    "category": spec.get("category", ""),
                    "year": obs.get("year", ""),
                    "period": obs.get("period", ""),
                    "period_name": obs.get("periodName", ""),
                    "latest": obs.get("latest", ""),
                    "value": obs.get("value", ""),
                    "footnotes": "; ".join(
                        footnote.get("text", "")
                        for footnote in obs.get("footnotes", [])
                        if isinstance(footnote, dict) and footnote.get("text")
                    ),
                }
                observation_rows.append(row)

    observations = pd.DataFrame(observation_rows).drop_duplicates()
    if observations.empty:
        raise RuntimeError("BLS returned no observations for the configured series.")
    observations["value_numeric"] = _coerce_numeric(observations["value"])
    metadata = pd.DataFrame(metadata_rows).drop_duplicates()
    raw_json_path = _write_sanitized_json(raw_payload, source="bls", raw_root=raw_root, secrets=[api_key])
    observations_path, metadata_path = _write_frame_bundle(
        observations,
        metadata,
        source="bls",
        processed_root=processed_root,
    )
    return SourceBuildResult(
        source="bls",
        raw_json_path=raw_json_path,
        observations_path=observations_path,
        metadata_path=metadata_path,
        row_count=int(len(observations.index)),
    )


def build_bea_snapshot(*, config: dict[str, Any], raw_root: Path, processed_root: Path) -> SourceBuildResult:
    api_key = os.environ.get("BEA_API_KEY")
    if not api_key:
        raise RuntimeError("BEA_API_KEY is not configured.")

    specs = _read_source_specs(config, "bea", BEA_REQUEST_DEFAULTS)
    raw_payload = {"source": "bea", "fetched_at": datetime.now(timezone.utc).isoformat(), "requests": []}
    observation_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for spec in specs:
        line_meta = _http_get_json(
            BEA_BASE_URL,
            {
                "UserID": api_key,
                "method": "GetParameterValuesFiltered",
                "datasetname": spec["dataset"],
                "TargetParameter": "LineCode",
                "TableName": spec["table_name"],
                "ResultFormat": "JSON",
            },
        )
        data_payload = _http_get_json(
            BEA_BASE_URL,
            {
                "UserID": api_key,
                "method": "GetData",
                "datasetname": spec["dataset"],
                "TableName": spec["table_name"],
                "LineCode": spec["line_code"],
                "GeoFips": spec["geo_fips"],
                "Year": spec["year"],
                "ResultFormat": "JSON",
            },
        )
        raw_payload["requests"].append(
            {
                "spec": spec,
                "line_metadata": line_meta,
                "data": data_payload,
            }
        )
        line_description = ""
        for item in line_meta.get("BEAAPI", {}).get("Results", {}).get("ParamValue", []):
            if str(item.get("Key")) == str(spec["line_code"]):
                line_description = item.get("Desc", "")
                break
        metadata_rows.append(
            {
                "dataset": spec["dataset"],
                "table_name": spec["table_name"],
                "line_code": spec["line_code"],
                "geo_fips": spec["geo_fips"],
                "year": spec["year"],
                "canonical_name": spec["canonical_name"],
                "line_description": line_description,
            }
        )
        data_rows = data_payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
        frame = pd.DataFrame(data_rows)
        if frame.empty:
            continue
        frame.insert(0, "canonical_name", spec["canonical_name"])
        frame.insert(1, "dataset", spec["dataset"])
        frame.insert(2, "table_name", spec["table_name"])
        frame.insert(3, "line_code", spec["line_code"])
        frame["data_value_numeric"] = _coerce_numeric(frame["DataValue"])
        observation_frames.append(frame)

    if not observation_frames:
        raise RuntimeError("BEA returned no data rows for the configured requests.")

    observations = pd.concat(observation_frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows).drop_duplicates()
    raw_json_path = _write_sanitized_json(raw_payload, source="bea", raw_root=raw_root, secrets=[api_key])
    observations_path, metadata_path = _write_frame_bundle(
        observations,
        metadata,
        source="bea",
        processed_root=processed_root,
    )
    return SourceBuildResult(
        source="bea",
        raw_json_path=raw_json_path,
        observations_path=observations_path,
        metadata_path=metadata_path,
        row_count=int(len(observations.index)),
    )


def build_census_snapshot(*, config: dict[str, Any], raw_root: Path, processed_root: Path) -> SourceBuildResult:
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        raise RuntimeError("CENSUS_API_KEY is not configured.")

    specs = _read_flexible_specs(config, "census", CENSUS_REQUEST_DEFAULTS)
    raw_payload: dict[str, Any] = {"source": "census", "fetched_at": datetime.now(timezone.utc).isoformat(), "requests": []}
    observation_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for spec in specs:
        variables_value = spec.get("variables", [])
        if isinstance(variables_value, str):
            variables = [item.strip() for item in variables_value.split(",") if item.strip()]
        else:
            variables = [str(item) for item in variables_value if str(item).strip()]
        if not variables:
            raise RuntimeError("Census benchmark specs must include at least one variable.")
        params = {
            "get": ",".join(variables),
            "for": str(spec.get("geography", "state:*")),
            "key": api_key,
        }
        dataset = str(spec.get("dataset", spec.get("dataset_path", "2024/acs/acs5")))
        response = _http_get_json(f"{CENSUS_BASE_URL}/{dataset}", params)
        raw_payload["requests"].append({"spec": {"dataset": dataset, "variables": variables}, "response": response})
        if not isinstance(response, list) or len(response) < 2:
            raise RuntimeError(f"Census returned no tabular rows for dataset {dataset}.")

        header = [str(item) for item in response[0]]
        rows = response[1:]
        frame = pd.DataFrame(rows, columns=header)
        for column in header:
            if column not in {"NAME", "state"}:
                frame[column] = _coerce_numeric(frame[column])

        bachelors_or_higher = (
            _series_or_na(frame, "B15003_022E").fillna(0)
            + _series_or_na(frame, "B15003_023E").fillna(0)
            + _series_or_na(frame, "B15003_024E").fillna(0)
            + _series_or_na(frame, "B15003_025E").fillna(0)
        )
        normalized = pd.DataFrame(
            {
                "canonical_name": str(spec.get("canonical_name", "acs_context")),
                "dataset": dataset,
                "geography": str(spec.get("geography", "state:*")),
                "state": _series_or_na(frame, "state"),
                "state_name": _series_or_na(frame, "NAME"),
                "median_household_income": _series_or_na(frame, "B19013_001E"),
                "poverty_universe": _series_or_na(frame, "B17001_001E"),
                "poverty_below_threshold": _series_or_na(frame, "B17001_002E"),
                "poverty_rate": _safe_ratio(_series_or_na(frame, "B17001_002E"), _series_or_na(frame, "B17001_001E")),
                "labor_force": _series_or_na(frame, "B23025_003E"),
                "unemployed": _series_or_na(frame, "B23025_005E"),
                "unemployment_rate": _safe_ratio(_series_or_na(frame, "B23025_005E"), _series_or_na(frame, "B23025_003E")),
                "education_universe": _series_or_na(frame, "B15003_001E"),
                "bachelors_or_higher_count": bachelors_or_higher,
                "bachelors_or_higher_share": _safe_ratio(bachelors_or_higher, _series_or_na(frame, "B15003_001E")),
            }
        )
        observation_frames.append(normalized)
        metadata_rows.extend(
            [
                {
                    "canonical_name": str(spec.get("canonical_name", "acs_context")),
                    "dataset": dataset,
                    "metric": "median_household_income",
                    "source_variables": "B19013_001E",
                    "note": "ACS 5-year median household income in the past 12 months.",
                },
                {
                    "canonical_name": str(spec.get("canonical_name", "acs_context")),
                    "dataset": dataset,
                    "metric": "poverty_rate",
                    "source_variables": "B17001_002E / B17001_001E",
                    "note": "Share of the poverty universe below poverty threshold.",
                },
                {
                    "canonical_name": str(spec.get("canonical_name", "acs_context")),
                    "dataset": dataset,
                    "metric": "unemployment_rate",
                    "source_variables": "B23025_005E / B23025_003E",
                    "note": "Share unemployed among the labor force.",
                },
                {
                    "canonical_name": str(spec.get("canonical_name", "acs_context")),
                    "dataset": dataset,
                    "metric": "bachelors_or_higher_share",
                    "source_variables": "B15003_022E + B15003_023E + B15003_024E + B15003_025E, divided by B15003_001E",
                    "note": "Share age 25+ with bachelor's degree or higher.",
                },
            ]
        )

    observations = pd.concat(observation_frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows).drop_duplicates()
    raw_json_path = _write_sanitized_json(raw_payload, source="census", raw_root=raw_root, secrets=[api_key])
    observations_path, metadata_path = _write_frame_bundle(
        observations,
        metadata,
        source="census",
        processed_root=processed_root,
    )
    return SourceBuildResult(
        source="census",
        raw_json_path=raw_json_path,
        observations_path=observations_path,
        metadata_path=metadata_path,
        row_count=int(len(observations.index)),
    )


def build_ipums_extract_workflow(
    *,
    config: dict[str, Any],
    raw_root: Path,
    processed_root: Path,
    output_dir: Path,
) -> IpumsWorkflowResult:
    api_key = os.environ.get("IPUMS_API_KEY")
    if not api_key:
        raise RuntimeError("IPUMS_API_KEY is not configured.")

    specs = _read_flexible_specs(config, "ipums_extracts", IPUMS_EXTRACT_DEFAULTS)
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    submitted = False
    extracts_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    raw_payload: dict[str, Any] = {"source": "ipums", "fetched_at": datetime.now(timezone.utc).isoformat(), "requests": []}
    request_records: list[dict[str, Any]] = []

    for spec in specs:
        collection = str(spec.get("collection", "cps"))
        variables_value = spec.get("variables", [])
        samples_value = spec.get("samples", [])
        if isinstance(variables_value, str):
            variables = [item.strip() for item in variables_value.split(",") if item.strip()]
        else:
            variables = [str(item) for item in variables_value if str(item).strip()]
        if isinstance(samples_value, str):
            samples = [item.strip() for item in samples_value.split(",") if item.strip()]
        else:
            samples = [str(item) for item in samples_value if str(item).strip()]
        if not variables or not samples:
            raise RuntimeError("IPUMS extract specs must include both samples and variables.")

        payload = {
            "description": str(spec.get("description", f"dadgap_{collection}_benchmark")),
            "dataStructure": {"rectangular": {"on": str(spec.get("record_type", "P"))}},
            "dataFormat": str(spec.get("data_format", "csv")),
            "caseSelectWho": str(spec.get("case_select_who", "individuals")),
            "samples": {sample: {} for sample in samples},
            "variables": {variable: {} for variable in variables},
            "collection": collection,
            "version": 2,
        }
        request_records.append(payload)
        headers = {"Authorization": api_key}
        extract_list = _http_get_json(IPUMS_BASE_URL, {"collection": collection, "version": 2}, headers=headers)
        response_record: dict[str, Any] = {"collection": collection, "list": extract_list, "request": payload}

        submit_value = str(spec.get("submit", "false")).strip().lower()
        should_submit = submit_value in {"1", "true", "yes", "on"}
        submission_response: Any = None
        if should_submit:
            submission_response = _http_post_json(
                f"{IPUMS_BASE_URL}?{urlencode({'collection': collection, 'version': 2})}",
                payload,
                headers=headers,
            )
            response_record["submission"] = submission_response
            submitted = True
        raw_payload["requests"].append(response_record)

        extracts = extract_list.get("data", []) if isinstance(extract_list, dict) else []
        frame = pd.DataFrame(extracts)
        if not frame.empty:
            normalized = pd.DataFrame(
                {
                    "collection": collection,
                    "number": pd.to_numeric(frame.get("number"), errors="coerce"),
                    "status": frame.get("status", pd.Series("", index=frame.index)),
                    "description": frame.get("extractDefinition", pd.Series([{}] * len(frame.index))).apply(
                        lambda item: item.get("description", "") if isinstance(item, dict) else ""
                    ),
                    "sample_count": frame.get("extractDefinition", pd.Series([{}] * len(frame.index))).apply(
                        lambda item: len(item.get("samples", {})) if isinstance(item, dict) else 0
                    ),
                    "variable_count": frame.get("extractDefinition", pd.Series([{}] * len(frame.index))).apply(
                        lambda item: len(item.get("variables", {})) if isinstance(item, dict) else 0
                    ),
                    "download_links_present": frame.get("downloadLinks", pd.Series([{}] * len(frame.index))).apply(
                        lambda item: bool(item) if isinstance(item, dict) else False
                    ),
                    "warning_count": frame.get("warnings", pd.Series([[]] * len(frame.index))).apply(
                        lambda item: len(item) if isinstance(item, list) else 0
                    ),
                }
            )
            extracts_frames.append(normalized)

        metadata_rows.append(
            {
                "collection": collection,
                "description": payload["description"],
                "samples": ",".join(samples),
                "sample_count": len(samples),
                "variables": ",".join(variables),
                "variable_count": len(variables),
                "submit_requested": should_submit,
                "submitted_extract_number": submission_response.get("number") if isinstance(submission_response, dict) else pd.NA,
            }
        )

    extracts_frame = pd.concat(extracts_frames, ignore_index=True) if extracts_frames else pd.DataFrame(columns=["collection"])
    extracts_path = processed_root / "ipums_extracts.parquet"
    metadata_path = processed_root / "ipums_request_metadata.csv"
    extracts_frame.to_parquet(extracts_path, index=False)
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    request_path = output_dir / "ipums_extract_request.json"
    request_path.write_text(
        json.dumps(_sanitize_payload({"requests": request_records}, secrets=[api_key]), indent=2),
        encoding="utf-8",
    )
    raw_json_path = _write_sanitized_json(raw_payload, source="ipums", raw_root=raw_root, secrets=[api_key])

    status_path = output_dir / "ipums_extract_status.md"
    lines = [
        "# IPUMS Extract Workflow",
        "",
        f"- fetched_at_utc: {raw_payload['fetched_at']}",
        f"- request_specs: {len(request_records)}",
        f"- live_extract_rows: {len(extracts_frame.index)}",
        f"- submission_attempted: {'yes' if submitted else 'no'}",
        "",
        "| collection | latest_extract_number | latest_status | extract_count |",
        "| --- | ---: | --- | ---: |",
    ]
    if extracts_frame.empty:
        for item in metadata_rows:
            lines.append(f"| {item['collection']} |  | no_extracts_seen | 0 |")
    else:
        for collection, group in extracts_frame.groupby("collection", dropna=False):
            latest = group.sort_values("number", ascending=False).iloc[0]
            lines.append(
                f"| {collection} | {int(latest['number']) if pd.notna(latest['number']) else ''} | "
                f"{latest.get('status', '')} | {len(group.index)} |"
            )
    lines.extend(
        [
            "",
            "## Request artifact",
            "",
            f"- request_json: `{request_path.name}`",
            f"- raw_snapshot: `{raw_json_path.name}`",
            f"- processed_extracts: `{extracts_path.name}`",
            f"- request_metadata: `{metadata_path.name}`",
        ]
    )
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return IpumsWorkflowResult(
        request_path=request_path,
        extracts_path=extracts_path,
        metadata_path=metadata_path,
        status_path=status_path,
        raw_json_path=raw_json_path,
        row_count=int(len(extracts_frame.index)),
        submitted=submitted,
    )


def build_public_microdata_artifacts(
    *,
    config: dict[str, Any],
    processed_root: Path,
    output_dir: Path,
) -> PublicMicrodataResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    artifacts = [
        artifact
        for artifact in (
            _build_acs_microdata(config=config, processed_root=processed_root),
            _build_sipp_microdata(config=config, processed_root=processed_root),
            _build_cps_microdata(config=config, processed_root=processed_root),
        )
        if artifact is not None
    ]
    if not artifacts:
        raise RuntimeError("No local public microdata directories were configured or no supported files were found.")

    manifest_path = output_dir / "public_microdata_intake.md"
    lines = [
        "# Public Microdata Intake",
        "",
        "| source | discovered_file | source_location | rows | columns | parquet | metadata |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for artifact in artifacts:
        lines.append(
            f"| {artifact.source} | {artifact.source_path.name} | {_public_value(artifact.source_path)} | "
            f"{artifact.row_count} | {artifact.column_count} | {artifact.parquet_path.name} | {artifact.metadata_path.name} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `acs_pums` materializes selected ACS 2024 PUMS person records plus household fields merged from the housing ZIP.",
            "- `sipp` materializes a selected person-level subset from the local 2024 public-use Stata file.",
            "- `cps_asec` materializes the selected person-level columns from the downloaded IPUMS CPS ASEC extract.",
        ]
    )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PublicMicrodataResult(manifest_path=manifest_path, artifacts=tuple(artifacts))


def build_public_benchmark_snapshot(
    *,
    config: dict[str, Any],
    raw_root: Path,
    processed_root: Path,
    output_dir: Path,
    sources: tuple[str, ...] = ("fred", "bea", "bls", "census"),
) -> BenchmarkBuildResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    build_fns = {
        "fred": build_fred_snapshot,
        "bea": build_bea_snapshot,
        "bls": build_bls_snapshot,
        "census": build_census_snapshot,
    }
    results: list[SourceBuildResult] = []
    for source in sources:
        builder = build_fns.get(source)
        if builder is None:
            raise KeyError(f"Unsupported benchmark source: {source}")
        results.append(
            builder(
                config=config,
                raw_root=raw_root / source,
                processed_root=processed_root,
            )
        )

    manifest_path = output_dir / "public_benchmark_snapshot.md"
    lines = [
        "# Public Benchmark Snapshot",
        "",
        "| source | rows | raw_json | observations | metadata |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item.source} | {item.row_count} | {item.raw_json_path.name} | "
            f"{item.observations_path.name} | {item.metadata_path.name} |"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BenchmarkBuildResult(manifest_path=manifest_path, results=tuple(results))
