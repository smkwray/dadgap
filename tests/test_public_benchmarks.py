from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import pandas as pd

from father_longrun.pipelines import public_benchmarks
from father_longrun.pipelines.public_benchmarks import (
    build_cross_cohort_benchmark_comparison,
    build_ipums_extract_workflow,
    build_public_benchmark_profiles,
    build_public_microdata_artifacts,
    build_public_benchmark_snapshot,
    prioritized_public_sources,
    source_statuses,
)


def test_source_statuses_cover_expected_sources() -> None:
    statuses = source_statuses()
    keys = {item.key for item in statuses}
    assert {"bea", "fred", "bls", "census", "ipums", "noaa_ncdc", "usda_quickstats"} <= keys


def test_public_source_priority_starts_with_macro_sources() -> None:
    ordered = prioritized_public_sources()
    assert [item.key for item in ordered[:3]] == ["fred", "bea", "bls"]


def test_build_public_benchmark_snapshot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FRED_API_KEY", "fred-secret")
    monkeypatch.setenv("BEA_API_KEY", "bea-secret")
    monkeypatch.setenv("BLS_API_KEY", "bls-secret")
    monkeypatch.setenv("CENSUS_API_KEY", "census-secret")

    def fake_get(url: str, params: dict[str, str], headers: dict[str, str] | None = None) -> object:
        if "stlouisfed.org" in url and url.endswith("/series"):
            return {
                "seriess": [
                    {
                        "title": f"Series {params['series_id']}",
                        "units": "Index",
                        "frequency": "Monthly",
                        "seasonal_adjustment": "SA",
                        "notes": "",
                    }
                ]
            }
        if "stlouisfed.org" in url and url.endswith("/series/observations"):
            return {"observations": [{"date": "2026-01-01", "value": "1.5"}]}
        if params.get("method") == "GetParameterValuesFiltered":
            return {
                "BEAAPI": {
                    "Results": {
                        "ParamValue": [
                            {"Key": "3", "Desc": "Per capita personal income"},
                            {"Key": "2", "Desc": "Real GDP"},
                        ]
                    }
                }
            }
        if params.get("method") == "GetData":
            return {
                "BEAAPI": {
                    "Request": {"UserID": "bea-secret"},
                    "Results": {
                        "Data": [
                            {
                                "GeoFips": "01000",
                                "GeoName": "Alabama",
                                "TimePeriod": "2024",
                                "DataValue": "123.4",
                                "LineCode": params["LineCode"],
                                "TableName": params["TableName"],
                            }
                        ]
                    },
                }
            }
        if "api.census.gov" in url:
            return [
                [
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
                    "state",
                ],
                ["Alabama", "60000", "1000", "100", "500", "25", "800", "80", "40", "20", "10", "01"],
            ]
        raise AssertionError(f"Unexpected GET call: {url} {params}")

    def fake_post(url: str, payload: dict[str, object]) -> dict[str, object]:
        return {
            "request": {"registrationkey": "bls-secret"},
            "Results": {
                "series": [
                    {
                        "seriesID": series_id,
                        "catalog": {
                            "series_title": f"Series {series_id}",
                            "survey_name": "Survey",
                            "seasonality": "S",
                            "measure_data_type": "Rate",
                        },
                        "data": [
                            {
                                "year": "2024",
                                "period": "M01",
                                "periodName": "January",
                                "latest": "true",
                                "value": "4.2",
                                "footnotes": [{"text": "note"}],
                            }
                        ],
                    }
                    for series_id in payload["seriesid"]
                ]
            },
        }

    monkeypatch.setattr(public_benchmarks, "_http_get_json", fake_get)
    monkeypatch.setattr(public_benchmarks, "_http_post_json", fake_post)

    result = build_public_benchmark_snapshot(
        config={},
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        output_dir=tmp_path / "outputs",
        sources=("fred", "bea", "bls", "census"),
    )

    assert result.manifest_path.exists()
    assert len(result.results) == 4
    for item in result.results:
        assert item.raw_json_path.exists()
        assert item.observations_path.exists()
        assert item.metadata_path.exists()

    bea_raw = (tmp_path / "raw" / "bea" / "bea_snapshot.json").read_text(encoding="utf-8")
    bls_raw = (tmp_path / "raw" / "bls" / "bls_snapshot.json").read_text(encoding="utf-8")
    census_raw = (tmp_path / "raw" / "census" / "census_snapshot.json").read_text(encoding="utf-8")
    assert "bea-secret" not in bea_raw
    assert "bls-secret" not in bls_raw
    assert "census-secret" not in census_raw


def test_build_ipums_extract_workflow(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IPUMS_API_KEY", "ipums-secret")

    def fake_get(url: str, params: dict[str, object], headers: dict[str, str] | None = None) -> object:
        assert url == public_benchmarks.IPUMS_BASE_URL
        assert params == {"collection": "cps", "version": 2}
        assert headers == {"Authorization": "ipums-secret"}
        return {
            "data": [
                {
                    "number": 12,
                    "status": "completed",
                    "email": "person@example.com",
                    "downloadLinks": {"csv": "https://example.test/file.csv.gz"},
                    "extractDefinition": {
                        "description": "dadgap_cps_asec_benchmark",
                        "samples": {"cps2024_03s": {}, "cps2025_03s": {}},
                        "variables": {"AGE": {}, "INCTOT": {}, "POVERTY": {}},
                    },
                    "warnings": ["sample note"],
                }
            ]
        }

    monkeypatch.setattr(public_benchmarks, "_http_get_json", fake_get)

    result = build_ipums_extract_workflow(
        config={},
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        output_dir=tmp_path / "outputs",
    )

    assert result.request_path.exists()
    assert result.extracts_path.exists()
    assert result.metadata_path.exists()
    assert result.status_path.exists()
    assert result.raw_json_path.exists()
    assert result.row_count == 1
    assert result.submitted is False

    extracts = public_benchmarks.pd.read_parquet(result.extracts_path)
    assert extracts.loc[0, "number"] == 12
    assert extracts.loc[0, "sample_count"] == 2
    assert extracts.loc[0, "variable_count"] == 3
    assert "email" not in extracts.columns

    raw_text = result.raw_json_path.read_text(encoding="utf-8")
    assert "ipums-secret" not in raw_text
    assert "person@example.com" not in raw_text


def test_build_public_microdata_artifacts(tmp_path: Path) -> None:
    acs_dir = tmp_path / "acs"
    sipp_dir = tmp_path / "sipp"
    cps_dir = tmp_path / "cps"
    acs_dir.mkdir()
    sipp_dir.mkdir()
    cps_dir.mkdir()

    acs_person_a = pd.DataFrame(
        {
            "SERIALNO": ["A1"],
            "SPORDER": [1],
            "STATE": [36],
            "PUMA": [100],
            "PWGTP": [10],
            "AGEP": [30],
            "SEX": [1],
            "RAC1P": [2],
            "HISP": [1],
            "SCHL": [21],
            "RELSHIPP": [0],
            "ESP": [1],
            "MAR": [1],
            "ESR": [1],
            "WAGP": [40000],
            "PERNP": [42000],
            "PINCP": [50000],
            "POVPIP": [220],
        }
    )
    acs_person_b = pd.DataFrame(
        {
            "SERIALNO": ["A2"],
            "SPORDER": [1],
            "STATE": [6],
            "PUMA": [200],
            "PWGTP": [12],
            "AGEP": [34],
            "SEX": [2],
            "RAC1P": [1],
            "HISP": [2],
            "SCHL": [16],
            "RELSHIPP": [0],
            "ESP": [8],
            "MAR": [2],
            "ESR": [3],
            "WAGP": [15000],
            "PERNP": [18000],
            "PINCP": [20000],
            "POVPIP": [90],
        }
    )
    acs_housing_a = pd.DataFrame({"SERIALNO": ["A1"], "NP": [3], "HINCP": [90000], "FINCP": [88000], "WGTP": [9]})
    acs_housing_b = pd.DataFrame({"SERIALNO": ["A2"], "NP": [2], "HINCP": [28000], "FINCP": [25000], "WGTP": [11]})
    with zipfile.ZipFile(acs_dir / "csv_pus.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("psam_pusa.csv", acs_person_a.to_csv(index=False))
        archive.writestr("psam_pusb.csv", acs_person_b.to_csv(index=False))
    with zipfile.ZipFile(acs_dir / "csv_hus.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("psam_husa.csv", acs_housing_a.to_csv(index=False))
        archive.writestr("psam_husb.csv", acs_housing_b.to_csv(index=False))

    sipp_frame = pd.DataFrame(
        {
            "SSUID": ["0001", "0002"],
            "SHHADID": ["H1", "H2"],
            "SPANEL": [2021, 2021],
            "SWAVE": [1, 2],
            "PNUM": [101, 102],
            "WPFINWGT": [100.0, 120.0],
            "TAGE": [25, 35],
            "ESEX": [1, 2],
            "ERACE": [1, 2],
            "EHISPAN": [1, 2],
            "EEDUC": [39, 43],
            "ERELRPE": [1, 2],
            "TMARPATH": [1, 3],
            "RMESR": [1, 8],
            "TPEARN": [5000, 4200],
            "TPTOTINC": [5400, 4700],
            "THTOTINC": [8800, 6400],
            "TFTOTINC": [8700, 6300],
            "RHPOV": [1500, 1600],
            "RFPOV": [1300, 1400],
            "THINCPOV": [2.5, 1.75],
            "TFINCPOV": [2.45, 0.7],
        }
    )
    sipp_path = sipp_dir / "pu2024.dta"
    sipp_frame.to_stata(sipp_path, write_index=False)

    cps_frame = pd.DataFrame(
        {
            "YEAR": [2024, 2024],
            "MONTH": [3, 3],
            "SERIAL": [1, 1],
            "CPSID": [123, 123],
            "ASECFLAG": [1, 1],
            "ASECWTH": [10.0, 10.0],
            "STATEFIP": [36, 36],
            "PERNUM": [1, 2],
            "CPSIDP": [1231, 1232],
            "CPSIDV": [12311, 12322],
            "ASECWT": [11.0, 12.0],
            "RELATE": [101, 201],
            "AGE": [40, 38],
            "SEX": [1, 2],
            "RACE": [100, 100],
            "MARST": [1, 1],
            "FAMSIZE": [4, 4],
            "HISPAN": [0, 0],
            "EMPSTAT": [10, 10],
            "LABFORCE": [2, 2],
            "EDUC": [73, 73],
            "INCTOT": [80000, 65000],
            "INCWAGE": [80000, 65000],
            "POVERTY": [300, 300],
        }
    )
    cps_path = cps_dir / "cps_00010.csv.gz"
    with gzip.open(cps_path, "wt", encoding="utf-8", newline="") as handle:
        cps_frame.to_csv(handle, index=False)

    result = build_public_microdata_artifacts(
        config={"benchmarks": {"acs_pums_dir": str(acs_dir), "sipp_dir": str(sipp_dir), "cps_asec_dir": str(cps_dir)}},
        processed_root=tmp_path / "processed",
        output_dir=tmp_path / "outputs",
    )

    assert result.manifest_path.exists()
    assert len(result.artifacts) == 3
    by_source = {item.source: item for item in result.artifacts}
    assert by_source["acs_pums"].row_count == 2
    assert by_source["sipp"].row_count == 2
    assert by_source["cps_asec"].row_count == 2

    acs_selected = pd.read_parquet(by_source["acs_pums"].parquet_path)
    sipp_selected = pd.read_parquet(by_source["sipp"].parquet_path)
    cps_selected = pd.read_parquet(by_source["cps_asec"].parquet_path)
    assert "household_income" in acs_selected.columns
    assert "parent_status_code" in acs_selected.columns
    assert "monthly_household_total_income" in sipp_selected.columns
    assert "wage_income" in cps_selected.columns


def test_build_public_benchmark_profiles(tmp_path: Path) -> None:
    acs_dir = tmp_path / "acs"
    sipp_dir = tmp_path / "sipp"
    cps_dir = tmp_path / "cps"
    acs_dir.mkdir()
    sipp_dir.mkdir()
    cps_dir.mkdir()

    acs_person = pd.DataFrame(
        {
            "SERIALNO": ["A1", "A2"],
            "SPORDER": [1, 1],
            "STATE": [36, 6],
            "PUMA": [100, 200],
            "PWGTP": [10, 12],
            "AGEP": [31, 33],
            "SEX": [1, 2],
            "RAC1P": [2, 1],
            "HISP": [1, 2],
            "SCHL": [21, 16],
            "RELSHIPP": [0, 0],
            "ESP": [1, 8],
            "MAR": [1, 2],
            "ESR": [1, 3],
            "WAGP": [60000, 5000],
            "PERNP": [62000, 9000],
            "PINCP": [70000, 12000],
            "POVPIP": [250, 90],
        }
    )
    acs_housing = pd.DataFrame(
        {"SERIALNO": ["A1", "A2"], "NP": [3, 2], "HINCP": [95000, 18000], "FINCP": [90000, 15000], "WGTP": [9, 11]}
    )
    with zipfile.ZipFile(acs_dir / "csv_pus.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("psam_pusa.csv", acs_person.to_csv(index=False))
    with zipfile.ZipFile(acs_dir / "csv_hus.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("psam_husa.csv", acs_housing.to_csv(index=False))

    pd.DataFrame(
        {
            "SSUID": ["0001", "0002"],
            "SHHADID": ["H1", "H2"],
            "SPANEL": [2021, 2021],
            "SWAVE": [1, 1],
            "PNUM": [101, 102],
            "WPFINWGT": [100.0, 120.0],
            "TAGE": [30, 32],
            "ESEX": [1, 2],
            "ERACE": [2, 1],
            "EHISPAN": [1, 2],
            "EEDUC": [43, 39],
            "ERELRPE": [1, 2],
            "TMARPATH": [2, 3],
            "RMESR": [1, 8],
            "TPEARN": [4000, 0],
            "TPTOTINC": [5000, 1800],
            "THTOTINC": [7000, 2200],
            "TFTOTINC": [6800, 2100],
            "RHPOV": [1600, 1600],
            "RFPOV": [1400, 1400],
            "THINCPOV": [2.0, 0.8],
            "TFINCPOV": [1.9, 0.75],
        }
    ).to_stata(sipp_dir / "pu2024.dta", write_index=False)

    cps_frame = pd.DataFrame(
        {
            "YEAR": [2024, 2024],
            "MONTH": [3, 3],
            "SERIAL": [1, 2],
            "CPSID": [123, 456],
            "ASECFLAG": [1, 1],
            "ASECWTH": [10.0, 10.0],
            "STATEFIP": [36, 36],
            "PERNUM": [1, 1],
            "CPSIDP": [1231, 4561],
            "CPSIDV": [12311, 45611],
            "ASECWT": [11.0, 12.0],
            "RELATE": [101, 101],
            "AGE": [30, 33],
            "SEX": [1, 2],
            "RACE": [200, 100],
            "MARST": [1, 1],
            "FAMSIZE": [4, 4],
            "HISPAN": [0, 100],
            "EMPSTAT": [10, 36],
            "LABFORCE": [2, 1],
            "EDUC": [73, 81],
            "INCTOT": [80000, 20000],
            "INCWAGE": [80000, 0],
            "POVERTY": [23, 10],
        }
    )
    with gzip.open(cps_dir / "cps_00010.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        cps_frame.to_csv(handle, index=False)

    result = build_public_benchmark_profiles(
        config={"benchmarks": {"acs_pums_dir": str(acs_dir), "sipp_dir": str(sipp_dir), "cps_asec_dir": str(cps_dir)}},
        processed_root=tmp_path / "processed",
        output_dir=tmp_path / "outputs",
    )

    assert result.profiles_path.exists()
    assert result.mapping_path.exists()
    assert result.summary_path.exists()
    assert result.subgroup_summary_path.exists()
    assert result.sipp_employment_codebook_path.exists()
    assert result.acs_child_context_path.exists()
    assert result.acs_child_summary_path.exists()
    assert result.acs_child_report_path.exists()

    profiles = pd.read_parquet(result.profiles_path)
    assert set(profiles["source"]) == {"acs_pums", "sipp", "cps_asec"}
    assert profiles["adult_window_primary"].all()

    acs_profile = profiles.loc[profiles["source"] == "acs_pums"].iloc[1]
    sipp_profile = profiles.loc[profiles["source"] == "sipp"].iloc[1]
    cps_profile = profiles.loc[profiles["source"] == "cps_asec"].iloc[1]
    assert acs_profile["below_poverty"]
    assert acs_profile["race_ethnicity_3cat"] == "HISPANIC"
    assert sipp_profile["below_poverty"]
    assert sipp_profile["race_ethnicity_3cat"] == "HISPANIC"
    assert cps_profile["below_poverty"]
    assert cps_profile["race_ethnicity_3cat"] == "HISPANIC"

    summary = pd.read_csv(result.summary_path)
    assert "weighted_employment_share" in summary.columns

    codebook = pd.read_csv(result.sipp_employment_codebook_path)
    assert "with_job_entire_month_worked_all_weeks" in set(codebook["label"])

    acs_child = pd.read_csv(result.acs_child_summary_path)
    assert "father_absent_share" in acs_child.columns
    assert "overall" in set(acs_child["group_type"])


def test_build_cross_cohort_benchmark_comparison(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    (processed_root / "nlsy").mkdir(parents=True)
    acs_dir = tmp_path / "acs"
    acs_dir.mkdir()

    nlsy97 = pd.DataFrame(
        {
            "respondent_id": [1, 2, 3],
            "sex_raw": [1, 2, 2],
            "birth_year": [1985, 1987, 1982],
            "race_ethnicity_3cat": ["BLACK", "HISPANIC", "NON-BLACK, NON-HISPANIC"],
            "annual_earnings_2021_clean": [40000.0, 30000.0, 70000.0],
            "household_income_2021_clean": [60000.0, 45000.0, 100000.0],
            "employment_2021": [1, 0, 2],
            "primary_treatment_label_nlsy97": [
                "resident_bio_father_present",
                "resident_bio_father_absent",
                "resident_bio_father_present",
            ],
        }
    )
    nlsy97.to_parquet(processed_root / "nlsy" / "nlsy97_analysis_ready.parquet", index=False)

    sipp_dir = tmp_path / "sipp"
    cps_dir = tmp_path / "cps"
    sipp_dir.mkdir()
    cps_dir.mkdir()
    acs_person = pd.DataFrame(
        {
            "SERIALNO": ["A1", "A2"],
            "SPORDER": [1, 1],
            "STATE": [36, 6],
            "PUMA": [100, 200],
            "PWGTP": [10, 12],
            "AGEP": [30, 33],
            "SEX": [1, 2],
            "RAC1P": [2, 1],
            "HISP": [1, 2],
            "SCHL": [21, 16],
            "RELSHIPP": [0, 0],
            "ESP": [1, 8],
            "MAR": [1, 2],
            "ESR": [1, 3],
            "WAGP": [55000, 7000],
            "PERNP": [58000, 9000],
            "PINCP": [65000, 12000],
            "POVPIP": [230, 90],
        }
    )
    acs_housing = pd.DataFrame(
        {"SERIALNO": ["A1", "A2"], "NP": [3, 2], "HINCP": [85000, 16000], "FINCP": [80000, 14000], "WGTP": [9, 11]}
    )
    with zipfile.ZipFile(acs_dir / "csv_pus.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("psam_pusa.csv", acs_person.to_csv(index=False))
    with zipfile.ZipFile(acs_dir / "csv_hus.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("psam_husa.csv", acs_housing.to_csv(index=False))
    pd.DataFrame(
        {
            "SSUID": ["0001", "0002"],
            "SHHADID": ["H1", "H2"],
            "SPANEL": [2021, 2021],
            "SWAVE": [1, 1],
            "PNUM": [101, 102],
            "WPFINWGT": [100.0, 120.0],
            "TAGE": [30, 32],
            "ESEX": [1, 2],
            "ERACE": [2, 1],
            "EHISPAN": [1, 2],
            "EEDUC": [43, 39],
            "ERELRPE": [1, 2],
            "TMARPATH": [2, 3],
            "RMESR": [1, 8],
            "TPEARN": [4000, 0],
            "TPTOTINC": [5000, 1800],
            "THTOTINC": [7000, 2200],
            "TFTOTINC": [6800, 2100],
            "RHPOV": [1600, 1600],
            "RFPOV": [1400, 1400],
            "THINCPOV": [2.0, 0.8],
            "TFINCPOV": [1.9, 0.75],
        }
    ).to_stata(sipp_dir / "pu2024.dta", write_index=False)
    cps_frame = pd.DataFrame(
        {
            "YEAR": [2023, 2024],
            "MONTH": [3, 3],
            "SERIAL": [1, 2],
            "CPSID": [123, 456],
            "ASECFLAG": [1, 1],
            "ASECWTH": [10.0, 10.0],
            "STATEFIP": [36, 36],
            "PERNUM": [1, 1],
            "CPSIDP": [1231, 4561],
            "CPSIDV": [12311, 45611],
            "ASECWT": [11.0, 12.0],
            "RELATE": [101, 101],
            "AGE": [30, 33],
            "SEX": [1, 2],
            "RACE": [200, 100],
            "MARST": [1, 1],
            "FAMSIZE": [4, 4],
            "HISPAN": [0, 100],
            "EMPSTAT": [10, 36],
            "LABFORCE": [2, 1],
            "EDUC": [73, 81],
            "INCTOT": [80000, 20000],
            "INCWAGE": [80000, 0],
            "POVERTY": [23, 10],
        }
    )
    with gzip.open(cps_dir / "cps_00010.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        cps_frame.to_csv(handle, index=False)

    result = build_cross_cohort_benchmark_comparison(
        config={"benchmarks": {"acs_pums_dir": str(acs_dir), "sipp_dir": str(sipp_dir), "cps_asec_dir": str(cps_dir)}},
        processed_root=processed_root,
        output_dir=output_dir,
    )

    assert result.profiles_path.exists()
    assert result.summary_path.exists()
    assert result.subgroup_summary_path.exists()
    assert result.report_path.exists()

    summary = pd.read_csv(result.summary_path)
    assert {
        "overall",
        "resident_bio_father_present",
        "resident_bio_father_absent",
        "acs_pums_2024_context",
        "cps_asec_2023_2025_pooled",
        "sipp_2023_monthly_context",
    } <= set(summary["source_group"])
    assert set(summary.loc[summary["source_group"] == "acs_pums_2024_context", "measure_period"]) == {"annual"}
    assert set(summary.loc[summary["source_group"] == "cps_asec_2023_2025_pooled", "measure_period"]) == {"annual"}
    assert set(summary.loc[summary["source_group"] == "sipp_2023_monthly_context", "measure_period"]) == {"monthly"}
