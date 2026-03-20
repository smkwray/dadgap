from __future__ import annotations

from typing import Any

import pandas as pd


PUBLIC_PROFILE_COLUMNS: tuple[str, ...] = (
    "source",
    "source_dataset",
    "reference_year",
    "measure_period",
    "person_weight",
    "age",
    "adult_window_primary",
    "sex",
    "female",
    "race_ethnicity_3cat",
    "education_code",
    "relationship_to_reference_code",
    "marital_path_code",
    "employment_status_code",
    "employment",
    "earnings",
    "person_income",
    "household_income",
    "family_income",
    "poverty_ratio",
    "below_poverty",
    "poverty_band",
)

PUBLIC_PROFILE_DTYPES: dict[str, str] = {
    "source": "string",
    "source_dataset": "string",
    "reference_year": "Int64",
    "measure_period": "string",
    "person_weight": "Float64",
    "age": "Float64",
    "adult_window_primary": "boolean",
    "sex": "string",
    "female": "boolean",
    "race_ethnicity_3cat": "string",
    "education_code": "Float64",
    "relationship_to_reference_code": "Float64",
    "marital_path_code": "Float64",
    "employment_status_code": "Float64",
    "employment": "boolean",
    "earnings": "Float64",
    "person_income": "Float64",
    "household_income": "Float64",
    "family_income": "Float64",
    "poverty_ratio": "Float64",
    "below_poverty": "boolean",
    "poverty_band": "string",
}


def build_poverty_band_from_ratio(
    series: pd.Series,
    *,
    missing_label: str = "missing",
    lowercase: bool = False,
) -> pd.Series:
    """Bucket a poverty ratio into the shared public-profile bands."""

    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(missing_label, index=series.index, dtype="string")
    lower_100 = "below_100_pct" if lowercase else "BELOW_100_PCT"
    band_100_124 = "100_124_pct" if lowercase else "100_124_PCT"
    band_125_149 = "125_149_pct" if lowercase else "125_149_PCT"
    upper_150 = "150_plus_pct" if lowercase else "150_PLUS_PCT"
    result.loc[values.lt(1)] = lower_100
    result.loc[values.ge(1) & values.lt(1.25)] = band_100_124
    result.loc[values.ge(1.25) & values.lt(1.5)] = band_125_149
    result.loc[values.ge(1.5)] = upper_150
    return result


def build_poverty_band_from_percent(
    series: pd.Series,
    *,
    missing_label: str = "missing",
    lowercase: bool = False,
) -> pd.Series:
    """Bucket a poverty percentage into the shared public-profile bands."""

    return build_poverty_band_from_ratio(
        pd.to_numeric(series, errors="coerce") / 100.0,
        missing_label=missing_label,
        lowercase=lowercase,
    )


def map_public_sex(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").map({1: "MALE", 2: "FEMALE"}).astype("string")


def map_public_race_ethnicity_3cat(*, race: pd.Series, hispanic: pd.Series, source: str) -> pd.Series:
    race_num = pd.to_numeric(race, errors="coerce")
    hispanic_num = pd.to_numeric(hispanic, errors="coerce")
    if source in {"sipp", "acs_pums"}:
        hispanic_indicator = hispanic_num.gt(1)
        black_indicator = race_num.eq(2)
    else:
        hispanic_indicator = hispanic_num.ne(0)
        black_indicator = race_num.eq(200)

    values = pd.Series("NON-BLACK, NON-HISPANIC", index=race.index, dtype="string")
    values.loc[black_indicator.fillna(False)] = "BLACK"
    values.loc[hispanic_indicator.fillna(False)] = "HISPANIC"
    return values


def standardize_public_profile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Project a source-specific public profile frame onto the shared schema."""

    standardized = frame.reindex(columns=PUBLIC_PROFILE_COLUMNS).copy()
    for column, dtype in PUBLIC_PROFILE_DTYPES.items():
        standardized[column] = standardized[column].astype(dtype)
    return standardized


def to_serializable_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert pandas scalar values to JSON-safe Python values."""

    serializable: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            serializable[key] = None
        elif isinstance(value, pd.Timestamp):
            serializable[key] = value.isoformat()
        else:
            serializable[key] = value
    return serializable
