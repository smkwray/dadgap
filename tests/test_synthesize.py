from __future__ import annotations

import csv
import json
from math import isclose
from pathlib import Path

import pytest

from father_longrun.pipelines.contracts import RESULTS_SCHEMA_VERSION, SITE_PAYLOAD_VERSION
from father_longrun.pipelines.synthesize import build_synthesis


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fixture_manifest_rows() -> list[dict[str, str]]:
    return [
        {
            "artifact": "nlsy_prevalence_table",
            "path": "table_nlsy97_fatherlessness_prevalence.csv",
            "purpose": "NLSY97 fatherlessness prevalence by total, race, sex, and parental education.",
        },
        {
            "artifact": "nlsy_predictor_table",
            "path": "table_nlsy97_fatherlessness_predictors.csv",
            "purpose": "Descriptive predictor model for NLSY97 fatherlessness.",
        },
        {
            "artifact": "nlsy_outcome_gap_table",
            "path": "table_nlsy97_outcome_gaps_vs_public_context.csv",
            "purpose": "Adult outcome gaps and public benchmark context.",
        },
        {
            "artifact": "benchmark_context_table",
            "path": "table_public_benchmark_context.csv",
            "purpose": "ACS, CPS, and SIPP weighted context table.",
        },
        {
            "artifact": "acs_child_context_table",
            "path": "table_acs_child_father_presence_context.csv",
            "purpose": "ACS child father-presence proxy by race, poverty, and income.",
        },
        {
            "artifact": "nlsy_race_gap_table",
            "path": "table_nlsy97_race_sex_outcome_gaps.csv",
            "purpose": "NLSY97 race-by-sex adult outcome splits by father-presence status.",
        },
    ]


def _write_fixture_artifacts(root: Path) -> None:
    manifests_root = root / "manifests"
    tables_root = root / "tables"
    _write_csv(
        manifests_root / "results_appendix_manifest.csv",
        ["artifact", "path", "purpose"],
        _fixture_manifest_rows(),
    )

    _write_csv(
        tables_root / "table_nlsy97_fatherlessness_prevalence.csv",
        [
            "group_type",
            "group_value",
            "n",
            "fatherlessness_rate",
            "mother_education_mean",
            "father_education_mean",
            "father_present_rate",
            "fatherlessness_pct",
            "father_present_pct",
        ],
        [
            {
                "group_type": "overall",
                "group_value": "overall",
                "n": 100,
                "fatherlessness_rate": 0.40,
                "mother_education_mean": 2.0,
                "father_education_mean": 2.2,
                "father_present_rate": 0.60,
                "fatherlessness_pct": 40.0,
                "father_present_pct": 60.0,
            },
            {
                "group_type": "sex",
                "group_value": "female",
                "n": 40,
                "fatherlessness_rate": 0.50,
                "mother_education_mean": 2.1,
                "father_education_mean": 2.3,
                "father_present_rate": 0.50,
                "fatherlessness_pct": 50.0,
                "father_present_pct": 50.0,
            },
        ],
    )

    _write_csv(
        tables_root / "table_nlsy97_fatherlessness_predictors.csv",
        ["term", "coefficient", "std_error", "p_value", "odds_ratio", "model", "n"],
        [
            {
                "term": "const",
                "coefficient": 0.7,
                "std_error": 0.07,
                "p_value": 0.001,
                "odds_ratio": 2.0,
                "model": "logit_hc1",
                "n": 100,
            },
            {
                "term": "sex_male",
                "coefficient": -0.2,
                "std_error": 0.05,
                "p_value": 0.002,
                "odds_ratio": 0.82,
                "model": "logit_hc1",
                "n": 100,
            },
            {
                "term": "race_HISPANIC",
                "coefficient": -1.0,
                "std_error": 0.06,
                "p_value": 0.0001,
                "odds_ratio": 0.36,
                "model": "logit_hc1",
                "n": 100,
            },
        ],
    )

    _write_csv(
        tables_root / "table_nlsy97_outcome_gaps_vs_public_context.csv",
        [
            "source",
            "source_group",
            "reference_year",
            "measure_period",
            "weighting_method",
            "row_count",
            "population",
            "female_share",
            "mean_earnings",
            "mean_person_income",
            "mean_household_income",
            "employment_rate",
            "poverty_share",
        ],
        [
            {
                "source": "nlsy97",
                "source_group": "overall",
                "reference_year": 2021,
                "measure_period": "annual",
                "weighting_method": "unweighted",
                "row_count": 100,
                "population": "",
                "female_share": 0.50,
                "mean_earnings": 50000,
                "mean_person_income": "",
                "mean_household_income": 75000,
                "employment_rate": 0.80,
                "poverty_share": 0.10,
            },
            {
                "source": "nlsy97",
                "source_group": "resident_bio_father_present",
                "reference_year": 2021,
                "measure_period": "annual",
                "weighting_method": "unweighted",
                "row_count": 60,
                "population": "",
                "female_share": 0.45,
                "mean_earnings": 60000,
                "mean_person_income": "",
                "mean_household_income": 85000,
                "employment_rate": 0.85,
                "poverty_share": 0.08,
            },
            {
                "source": "nlsy97",
                "source_group": "resident_bio_father_absent",
                "reference_year": 2021,
                "measure_period": "annual",
                "weighting_method": "unweighted",
                "row_count": 40,
                "population": "",
                "female_share": 0.55,
                "mean_earnings": 40000,
                "mean_person_income": "",
                "mean_household_income": 65000,
                "employment_rate": 0.72,
                "poverty_share": 0.14,
            },
        ],
    )

    _write_csv(
        tables_root / "table_public_benchmark_context.csv",
        [
            "source",
            "reference_year",
            "measure_period",
            "row_count",
            "weighted_population",
            "weighted_female_share",
            "weighted_employment_share",
            "weighted_mean_earnings",
            "weighted_mean_person_income",
            "weighted_poverty_share",
            "weighted_employment_pct",
            "weighted_poverty_pct",
        ],
        [
            {
                "source": "acs_pums",
                "reference_year": 2024,
                "measure_period": "annual",
                "row_count": 200,
                "weighted_population": 1000,
                "weighted_female_share": 0.50,
                "weighted_employment_share": 0.81,
                "weighted_mean_earnings": 55000,
                "weighted_mean_person_income": 57000,
                "weighted_poverty_share": 0.10,
                "weighted_employment_pct": 81.0,
                "weighted_poverty_pct": 10.0,
            },
            {
                "source": "cps_asec",
                "reference_year": 2023,
                "measure_period": "annual",
                "row_count": 180,
                "weighted_population": 900,
                "weighted_female_share": 0.49,
                "weighted_employment_share": 0.80,
                "weighted_mean_earnings": 53000,
                "weighted_mean_person_income": 56000,
                "weighted_poverty_share": 0.11,
                "weighted_employment_pct": 80.0,
                "weighted_poverty_pct": 11.0,
            },
        ],
    )

    _write_csv(
        tables_root / "table_acs_child_father_presence_context.csv",
        [
            "group_type",
            "group_value",
            "row_count",
            "weighted_children",
            "father_present_share",
            "father_absent_share",
            "two_parent_share",
            "father_only_share",
            "mother_only_share",
            "mean_household_income",
            "father_absent_pct",
            "father_present_pct",
        ],
        [
            {
                "group_type": "overall",
                "group_value": "overall",
                "row_count": 300,
                "weighted_children": 1000,
                "father_present_share": 0.74,
                "father_absent_share": 0.26,
                "two_parent_share": 0.66,
                "father_only_share": 0.08,
                "mother_only_share": 0.26,
                "mean_household_income": 130000,
                "father_absent_pct": 26.0,
                "father_present_pct": 74.0,
            },
            {
                "group_type": "race_ethnicity_3cat",
                "group_value": "BLACK",
                "row_count": 120,
                "weighted_children": 250,
                "father_present_share": 0.46,
                "father_absent_share": 0.54,
                "two_parent_share": 0.37,
                "father_only_share": 0.09,
                "mother_only_share": 0.54,
                "mean_household_income": 90000,
                "father_absent_pct": 54.0,
                "father_present_pct": 46.0,
            },
        ],
    )

    _write_csv(
        tables_root / "table_nlsy97_race_sex_outcome_gaps.csv",
        [
            "source",
            "source_group",
            "reference_year",
            "measure_period",
            "weighting_method",
            "sex",
            "race_ethnicity_3cat",
            "row_count",
            "population",
            "female_share",
            "mean_earnings",
            "mean_person_income",
            "mean_household_income",
            "employment_rate",
            "poverty_share",
        ],
        [
            {
                "source": "nlsy97",
                "source_group": "overall",
                "reference_year": 2021,
                "measure_period": "annual",
                "weighting_method": "unweighted",
                "sex": "FEMALE",
                "race_ethnicity_3cat": "BLACK",
                "row_count": 30,
                "population": "",
                "female_share": 1.0,
                "mean_earnings": 47000,
                "mean_person_income": "",
                "mean_household_income": 70000,
                "employment_rate": 0.75,
                "poverty_share": 0.11,
            }
        ],
    )


def test_build_synthesis_from_artifacts(tmp_path: Path) -> None:
    _write_fixture_artifacts(tmp_path)

    result = build_synthesis(outputs_root=tmp_path, project_root=tmp_path)

    assert result.summary_path.exists()
    assert result.forest_ready_path.exists()
    assert result.memo_path.exists()
    assert result.site_payload_path.exists()

    summary_rows = _read_csv(result.summary_path)
    forest_rows = _read_csv(result.forest_ready_path)
    memo = result.memo_path.read_text(encoding="utf-8")
    site_payload = json.loads(result.site_payload_path.read_text(encoding="utf-8"))

    assert any(row["artifact"] == "nlsy_prevalence_table" for row in summary_rows)
    assert any(row["artifact"] == "benchmark_context_table" for row in summary_rows)
    assert any(row["artifact"] == "acs_child_context_table" for row in summary_rows)
    assert any(row["artifact"] == "nlsy_predictor_table" and row["metric"] == "coefficient" for row in summary_rows)

    overall_prevalence = next(
        row
        for row in summary_rows
        if row["artifact"] == "nlsy_prevalence_table"
        and row["metric"] == "fatherlessness_rate"
        and row["group_value"] == "overall"
    )
    assert isclose(float(overall_prevalence["value"]), 0.40, rel_tol=1e-9)

    assert any(row["metric"] == "coefficient" and "sex_male" in row["label"] for row in forest_rows)
    assert any(row["metric"] == "fatherlessness_rate" and "female" in row["label"] for row in forest_rows)
    predictor_row = next(row for row in forest_rows if row["metric"] == "coefficient" and "sex_male" in row["label"])
    assert float(predictor_row["lower_ci"]) < float(predictor_row["estimate"]) < float(predictor_row["upper_ci"])

    rate_row = next(row for row in forest_rows if row["metric"] == "fatherlessness_rate" and row["group_value"] == "overall")
    assert isclose(float(rate_row["estimate"]), 0.40, rel_tol=1e-9)
    assert float(rate_row["std_error"]) > 0.0

    assert "Cross-Cohort Synthesis" in memo
    assert "forest-ready" in memo
    assert "Appendix manifest artifacts" in memo
    assert site_payload["schema_version"] == RESULTS_SCHEMA_VERSION
    assert site_payload["site_payload_version"] == SITE_PAYLOAD_VERSION
    assert site_payload["generated_at_utc"]
    assert site_payload["source_manifest"]
    assert len(site_payload["synthesis_artifacts"]) == 3
    assert site_payload["pages"]["home"]["stats"]["respondents"] == "100"
    assert site_payload["pages"]["prevalence"]["predictors"][0]["predictor"] == "Hispanic (vs Black)"
    assert "cognition_table" in site_payload["pages"]["outcomes"]
    assert "prevalence_answer" in site_payload["pages"]["faq"]


def test_build_synthesis_requires_committed_artifacts(tmp_path: Path) -> None:
    manifests_root = tmp_path / "manifests"
    _write_csv(
        manifests_root / "results_appendix_manifest.csv",
        ["artifact", "path", "purpose"],
        _fixture_manifest_rows(),
    )
    _write_csv(
        tmp_path / "tables" / "table_nlsy97_fatherlessness_prevalence.csv",
        ["group_type", "group_value", "n", "fatherlessness_rate", "mother_education_mean", "father_education_mean", "father_present_rate", "fatherlessness_pct", "father_present_pct"],
        [
            {
                "group_type": "overall",
                "group_value": "overall",
                "n": 100,
                "fatherlessness_rate": 0.40,
                "mother_education_mean": 2.0,
                "father_education_mean": 2.2,
                "father_present_rate": 0.60,
                "fatherlessness_pct": 40.0,
                "father_present_pct": 60.0,
            }
        ],
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        build_synthesis(outputs_root=tmp_path)
    assert "table_nlsy97_fatherlessness_predictors.csv" in str(excinfo.value)
