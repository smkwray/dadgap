from __future__ import annotations

import pandas as pd

from father_longrun.pipelines.harmonize import (
    PUBLIC_PROFILE_COLUMNS,
    build_poverty_band_from_percent,
    build_poverty_band_from_ratio,
    map_public_race_ethnicity_3cat,
    map_public_sex,
    standardize_public_profile_frame,
)


def test_build_poverty_band_from_ratio_uses_shared_labels() -> None:
    series = pd.Series([0.8, 1.1, 1.3, 1.8, None])

    result = build_poverty_band_from_ratio(series)

    assert result.tolist() == [
        "BELOW_100_PCT",
        "100_124_PCT",
        "125_149_PCT",
        "150_PLUS_PCT",
        "missing",
    ]


def test_build_poverty_band_from_percent_supports_lowercase_labels() -> None:
    series = pd.Series([80, 110, 130, 180, None])

    result = build_poverty_band_from_percent(series, lowercase=True)

    assert result.tolist() == [
        "below_100_pct",
        "100_124_pct",
        "125_149_pct",
        "150_plus_pct",
        "missing",
    ]


def test_standardize_public_profile_frame_projects_shared_schema() -> None:
    frame = pd.DataFrame(
        {
            "source": ["acs_pums"],
            "reference_year": [2024],
            "person_weight": [1.5],
            "sex": ["FEMALE"],
            "female": [True],
            "poverty_band": ["below_100_pct"],
            "extra_column": ["ignored"],
        }
    )

    standardized = standardize_public_profile_frame(frame)

    assert list(standardized.columns) == list(PUBLIC_PROFILE_COLUMNS)
    assert standardized.loc[0, "source"] == "acs_pums"
    assert standardized.loc[0, "reference_year"] == 2024
    assert "extra_column" not in standardized.columns


def test_map_public_sex_uses_shared_labels() -> None:
    result = map_public_sex(pd.Series([1, 2, 9]))

    assert result.iloc[0] == "MALE"
    assert result.iloc[1] == "FEMALE"
    assert pd.isna(result.iloc[2])


def test_map_public_race_ethnicity_3cat_handles_source_specific_hispanic_codes() -> None:
    sipp = map_public_race_ethnicity_3cat(
        race=pd.Series([2, 1, 1]),
        hispanic=pd.Series([1, 2, 1]),
        source="sipp",
    )
    cps = map_public_race_ethnicity_3cat(
        race=pd.Series([200, 100, 100]),
        hispanic=pd.Series([0, 1, 0]),
        source="cps_asec",
    )

    assert sipp.tolist() == ["BLACK", "HISPANIC", "NON-BLACK, NON-HISPANIC"]
    assert cps.tolist() == ["BLACK", "HISPANIC", "NON-BLACK, NON-HISPANIC"]
