from __future__ import annotations

from pathlib import Path

import pandas as pd

from father_longrun.pipelines.reporting import build_results_appendix


def test_build_results_appendix(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir(parents=True)

    pd.DataFrame(
        [
            {"group_type": "overall", "group_value": "overall", "n": 100, "fatherlessness_rate": 0.4, "mother_education_mean": 2.0, "father_education_mean": 2.1, "father_present_rate": 0.6},
            {"group_type": "race_ethnicity_3cat", "group_value": "BLACK", "n": 30, "fatherlessness_rate": 0.6, "mother_education_mean": 1.8, "father_education_mean": 1.7, "father_present_rate": 0.4},
            {"group_type": "race_ethnicity_3cat", "group_value": "NON-BLACK, NON-HISPANIC", "n": 70, "fatherlessness_rate": 0.3, "mother_education_mean": 2.4, "father_education_mean": 2.5, "father_present_rate": 0.7},
        ]
    ).to_csv(manifests / "nlsy97_fatherlessness_group_summary.csv", index=False)
    pd.DataFrame(
        [
            {"term": "const", "coefficient": 0.1, "std_error": 0.02, "p_value": 0.001, "odds_ratio": 1.1, "model": "logit_hc1", "n": 100},
            {"term": "sex_male", "coefficient": -0.2, "std_error": 0.05, "p_value": 0.01, "odds_ratio": 0.82, "model": "logit_hc1", "n": 100},
        ]
    ).to_csv(manifests / "nlsy97_fatherlessness_predictors.csv", index=False)
    pd.DataFrame(
        [
            {"group_type": "overall", "group_value": "overall", "row_count": 1000, "weighted_children": 1000.0, "father_present_share": 0.7, "father_absent_share": 0.3, "two_parent_share": 0.6, "father_only_share": 0.1, "mother_only_share": 0.3, "mean_household_income": 100000.0},
            {"group_type": "race_ethnicity_3cat", "group_value": "BLACK", "row_count": 300, "weighted_children": 300.0, "father_present_share": 0.4, "father_absent_share": 0.6, "two_parent_share": 0.3, "father_only_share": 0.1, "mother_only_share": 0.6, "mean_household_income": 70000.0},
            {"group_type": "poverty_band", "group_value": "below_100_pct", "row_count": 200, "weighted_children": 200.0, "father_present_share": 0.35, "father_absent_share": 0.65, "two_parent_share": 0.25, "father_only_share": 0.1, "mother_only_share": 0.65, "mean_household_income": 20000.0},
        ]
    ).to_csv(manifests / "acs_child_father_presence_summary.csv", index=False)
    pd.DataFrame(
        [
            {"source": "nlsy97", "source_group": "overall", "reference_year": 2021, "measure_period": "annual", "weighting_method": "unweighted", "row_count": 100, "population": pd.NA, "female_share": 0.5, "mean_earnings": 60000.0, "mean_person_income": pd.NA, "mean_household_income": 100000.0, "employment_rate": 0.8, "poverty_share": pd.NA},
            {"source": "nlsy97", "source_group": "resident_bio_father_present", "reference_year": 2021, "measure_period": "annual", "weighting_method": "unweighted", "row_count": 60, "population": pd.NA, "female_share": 0.48, "mean_earnings": 70000.0, "mean_person_income": pd.NA, "mean_household_income": 120000.0, "employment_rate": 0.84, "poverty_share": pd.NA},
            {"source": "nlsy97", "source_group": "resident_bio_father_absent", "reference_year": 2021, "measure_period": "annual", "weighting_method": "unweighted", "row_count": 40, "population": pd.NA, "female_share": 0.53, "mean_earnings": 45000.0, "mean_person_income": pd.NA, "mean_household_income": 75000.0, "employment_rate": 0.73, "poverty_share": pd.NA},
            {"source": "acs_pums", "source_group": "acs_pums_2024_context", "reference_year": 2024, "measure_period": "annual", "weighting_method": "person_weighted", "row_count": 500, "population": 5000.0, "female_share": 0.5, "mean_earnings": 55000.0, "mean_person_income": 57000.0, "mean_household_income": 130000.0, "employment_rate": pd.NA, "poverty_share": 0.1},
            {"source": "cps_asec", "source_group": "cps_asec_2023_2025_pooled", "reference_year": 2023, "measure_period": "annual", "weighting_method": "person_weighted", "row_count": 500, "population": 6000.0, "female_share": 0.5, "mean_earnings": 56000.0, "mean_person_income": 61000.0, "mean_household_income": pd.NA, "employment_rate": pd.NA, "poverty_share": 0.09},
            {"source": "sipp", "source_group": "sipp_2023_monthly_context", "reference_year": 2023, "measure_period": "monthly", "weighting_method": "person_weighted", "row_count": 500, "population": 7000.0, "female_share": 0.5, "mean_earnings": 6000.0, "mean_person_income": 5300.0, "mean_household_income": 12000.0, "employment_rate": pd.NA, "poverty_share": 0.12},
        ]
    ).to_csv(manifests / "cross_cohort_benchmark_summary.csv", index=False)
    pd.DataFrame(
        [
            {"source": "nlsy97", "source_group": "overall", "reference_year": 2021, "measure_period": "annual", "weighting_method": "unweighted", "sex": "FEMALE", "race_ethnicity_3cat": "BLACK", "row_count": 20, "population": pd.NA, "female_share": 1.0, "mean_earnings": 40000.0, "mean_person_income": pd.NA, "mean_household_income": 70000.0, "employment_rate": 0.7, "poverty_share": pd.NA},
            {"source": "nlsy97", "source_group": "resident_bio_father_present", "reference_year": 2021, "measure_period": "annual", "weighting_method": "unweighted", "sex": "MALE", "race_ethnicity_3cat": "NON-BLACK, NON-HISPANIC", "row_count": 20, "population": pd.NA, "female_share": 0.0, "mean_earnings": 90000.0, "mean_person_income": pd.NA, "mean_household_income": 140000.0, "employment_rate": 0.9, "poverty_share": pd.NA},
        ]
    ).to_csv(manifests / "cross_cohort_benchmark_subgroup_summary.csv", index=False)
    pd.DataFrame(
        [
            {"source": "acs_pums", "reference_year": 2024, "measure_period": "annual", "row_count": 500, "weighted_population": 5000.0, "weighted_female_share": 0.5, "weighted_employment_share": 0.81, "weighted_mean_earnings": 55000.0, "weighted_mean_person_income": 57000.0, "weighted_poverty_share": 0.10},
            {"source": "sipp", "reference_year": 2023, "measure_period": "monthly", "row_count": 500, "weighted_population": 7000.0, "weighted_female_share": 0.5, "weighted_employment_share": 0.82, "weighted_mean_earnings": 6000.0, "weighted_mean_person_income": 5300.0, "weighted_poverty_share": 0.12},
        ]
    ).to_csv(manifests / "public_benchmark_profile_summary.csv", index=False)

    result = build_results_appendix(outputs_root=tmp_path)

    assert result.manifest_path.exists()
    assert result.handoff_path.exists()
    assert result.synthesis_path.exists()
    assert result.nlsy_prevalence_table_path.exists()
    assert result.nlsy_predictor_table_path.exists()
    assert result.nlsy_outcome_gap_table_path.exists()
    assert result.nlsy_race_gap_table_path.exists()
    assert result.acs_child_context_table_path.exists()
    assert result.benchmark_context_table_path.exists()

    handoff = result.handoff_path.read_text(encoding="utf-8")
    assert "ready for frontend and documentation work" in handoff.lower()
