from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from father_longrun.pipelines.contracts import RESULTS_SCHEMA_VERSION
from father_longrun.pipelines.reporting import build_results_appendix


def test_build_results_appendix(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    models = tmp_path / "models"
    nlsy_refresh = tmp_path / "data" / "interim" / "nlsy_refresh" / "nlsy97"
    processed_nlsy = tmp_path / "data" / "processed" / "nlsy"
    manifests.mkdir(parents=True)
    models.mkdir(parents=True)
    nlsy_refresh.mkdir(parents=True)
    processed_nlsy.mkdir(parents=True)

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

    raw_rows = []
    for respondent_id in range(1, 9):
        birth_year = 1980 + (respondent_id % 4)
        high = respondent_id <= 4
        use_neg = respondent_id in {3, 4, 7, 8}
        base = 900.0 if high else 500.0
        row = {"R0000100": respondent_id, "R0536402": birth_year}
        specs = [
            ("R9705200", "R9706400"),
            ("R9705300", "R9706500"),
            ("R9705400", "R9706600"),
            ("R9705500", "R9706700"),
            ("R9705600", "R9706800"),
            ("R9705700", "R9706900"),
            ("R9705800", "R9707000"),
            ("R9705900", "R9707100"),
            ("R9706000", "R9707200"),
            ("R9706100", "R9707300"),
            ("R9706200", "R9707400"),
        ]
        for idx, (pos_col, neg_col) in enumerate(specs):
            if use_neg:
                row[pos_col] = -4
                row[neg_col] = base + idx
            else:
                row[pos_col] = base + idx
                row[neg_col] = -4
        raw_rows.append(row)
    pd.DataFrame(raw_rows).to_csv(nlsy_refresh / "panel_extract.csv", index=False)

    pd.DataFrame(
        [
            {"respondent_id": 1, "resident_bio_father_absent_1997": 0.0, "sex_raw": 1, "birth_year": 1981, "race_ethnicity_3cat": "BLACK", "mother_education": 3.2, "father_education": 2.9, "parent_education": 3.0},
            {"respondent_id": 2, "resident_bio_father_absent_1997": 0.0, "sex_raw": 2, "birth_year": 1982, "race_ethnicity_3cat": "BLACK", "mother_education": 3.6, "father_education": 3.4, "parent_education": 3.5},
            {"respondent_id": 3, "resident_bio_father_absent_1997": 0.0, "sex_raw": 1, "birth_year": 1983, "race_ethnicity_3cat": "HISPANIC", "mother_education": 2.9, "father_education": 2.7, "parent_education": 2.8},
            {"respondent_id": 4, "resident_bio_father_absent_1997": 0.0, "sex_raw": 2, "birth_year": 1980, "race_ethnicity_3cat": "NON-BLACK, NON-HISPANIC", "mother_education": 4.2, "father_education": 3.8, "parent_education": 4.0},
            {"respondent_id": 5, "resident_bio_father_absent_1997": 1.0, "sex_raw": 1, "birth_year": 1981, "race_ethnicity_3cat": "BLACK", "mother_education": 1.2, "father_education": 0.8, "parent_education": 1.0},
            {"respondent_id": 6, "resident_bio_father_absent_1997": 1.0, "sex_raw": 2, "birth_year": 1982, "race_ethnicity_3cat": "HISPANIC", "mother_education": 1.5, "father_education": 1.2, "parent_education": 1.4},
            {"respondent_id": 7, "resident_bio_father_absent_1997": 1.0, "sex_raw": 1, "birth_year": 1983, "race_ethnicity_3cat": "NON-BLACK, NON-HISPANIC", "mother_education": 1.9, "father_education": 1.6, "parent_education": 1.8},
            {"respondent_id": 8, "resident_bio_father_absent_1997": 1.0, "sex_raw": 2, "birth_year": 1980, "race_ethnicity_3cat": "NON-BLACK, NON-HISPANIC", "mother_education": 2.1, "father_education": 1.9, "parent_education": 2.0},
        ]
    ).to_parquet(processed_nlsy / "nlsy97_analysis_ready.parquet", index=False)

    pd.DataFrame(
        [
            {"respondent_id": 1, "panel_year": 2007, "age_at_wave": 26, "education_years_snapshot": 16.0, "sat_math_bin": 5.0, "sat_verbal_bin": 5.0, "act_bin": 5.0},
            {"respondent_id": 2, "panel_year": 2007, "age_at_wave": 25, "education_years_snapshot": 15.5, "sat_math_bin": 4.5, "sat_verbal_bin": 4.0, "act_bin": 4.5},
            {"respondent_id": 3, "panel_year": 2007, "age_at_wave": 24, "education_years_snapshot": 15.0, "sat_math_bin": 4.0, "sat_verbal_bin": 4.0, "act_bin": 4.0},
            {"respondent_id": 4, "panel_year": 2007, "age_at_wave": 27, "education_years_snapshot": 16.5, "sat_math_bin": 5.0, "sat_verbal_bin": 5.0, "act_bin": 5.0},
            {"respondent_id": 5, "panel_year": 2007, "age_at_wave": 26, "education_years_snapshot": 12.0, "sat_math_bin": 3.0, "sat_verbal_bin": 3.0, "act_bin": 3.0},
            {"respondent_id": 6, "panel_year": 2007, "age_at_wave": 25, "education_years_snapshot": 12.5, "sat_math_bin": 3.0, "sat_verbal_bin": 3.5, "act_bin": 3.0},
            {"respondent_id": 7, "panel_year": 2007, "age_at_wave": 24, "education_years_snapshot": 13.0, "sat_math_bin": 3.5, "sat_verbal_bin": 3.0, "act_bin": 3.5},
            {"respondent_id": 8, "panel_year": 2007, "age_at_wave": 27, "education_years_snapshot": 12.5, "sat_math_bin": 3.0, "sat_verbal_bin": 3.0, "act_bin": 3.0},
            {"respondent_id": 1, "panel_year": 2011, "age_at_wave": 30, "bmi": 24.0, "occupation_code": 2100},
            {"respondent_id": 2, "panel_year": 2011, "age_at_wave": 29, "bmi": 25.5, "occupation_code": 2200},
            {"respondent_id": 3, "panel_year": 2011, "age_at_wave": 28, "bmi": 26.0, "occupation_code": 4300},
            {"respondent_id": 4, "panel_year": 2011, "age_at_wave": 31, "bmi": 23.5, "occupation_code": 2400},
            {"respondent_id": 5, "panel_year": 2011, "age_at_wave": 30, "bmi": 31.0, "occupation_code": 8600},
            {"respondent_id": 6, "panel_year": 2011, "age_at_wave": 29, "bmi": 32.5, "occupation_code": 4700},
            {"respondent_id": 7, "panel_year": 2011, "age_at_wave": 28, "bmi": 33.0, "occupation_code": 6200},
            {"respondent_id": 8, "panel_year": 2011, "age_at_wave": 31, "bmi": 34.5, "occupation_code": 9000},
            {"respondent_id": 1, "panel_year": 2015, "age_at_wave": 34, "marijuana_days_30": 0.0, "occupation_code": 2100},
            {"respondent_id": 2, "panel_year": 2015, "age_at_wave": 33, "marijuana_days_30": 1.0, "occupation_code": 2200},
            {"respondent_id": 3, "panel_year": 2015, "age_at_wave": 32, "marijuana_days_30": 0.0, "occupation_code": 4300},
            {"respondent_id": 4, "panel_year": 2015, "age_at_wave": 35, "marijuana_days_30": 0.0, "occupation_code": 2400},
            {"respondent_id": 5, "panel_year": 2015, "age_at_wave": 34, "marijuana_days_30": 6.0, "occupation_code": 8600},
            {"respondent_id": 6, "panel_year": 2015, "age_at_wave": 33, "marijuana_days_30": 4.0, "occupation_code": 4700},
            {"respondent_id": 7, "panel_year": 2015, "age_at_wave": 32, "marijuana_days_30": 2.0, "occupation_code": 6200},
            {"respondent_id": 8, "panel_year": 2015, "age_at_wave": 35, "marijuana_days_30": 5.0, "occupation_code": 9000},
            {"respondent_id": 1, "panel_year": 2019, "age_at_wave": 38, "occupation_code": 2100, "annual_earnings": 70000.0, "household_income": 120000.0, "govt_program_income": 0.0},
            {"respondent_id": 2, "panel_year": 2019, "age_at_wave": 37, "occupation_code": 2200, "annual_earnings": 68000.0, "household_income": 118000.0, "govt_program_income": 0.0},
            {"respondent_id": 3, "panel_year": 2019, "age_at_wave": 36, "occupation_code": 4300, "annual_earnings": 62000.0, "household_income": 98000.0, "govt_program_income": 0.0},
            {"respondent_id": 4, "panel_year": 2019, "age_at_wave": 39, "occupation_code": 2400, "annual_earnings": 76000.0, "household_income": 130000.0, "govt_program_income": 0.0},
            {"respondent_id": 5, "panel_year": 2019, "age_at_wave": 38, "occupation_code": 8600, "annual_earnings": 42000.0, "household_income": 70000.0, "govt_program_income": 1.0},
            {"respondent_id": 6, "panel_year": 2019, "age_at_wave": 37, "occupation_code": 4700, "annual_earnings": 45000.0, "household_income": 76000.0, "govt_program_income": 1.0},
            {"respondent_id": 7, "panel_year": 2019, "age_at_wave": 36, "occupation_code": 6200, "annual_earnings": 50000.0, "household_income": 80000.0, "govt_program_income": 1.0},
            {"respondent_id": 8, "panel_year": 2019, "age_at_wave": 39, "occupation_code": 9000, "annual_earnings": 47000.0, "household_income": 78000.0, "govt_program_income": 1.0},
            {"respondent_id": 1, "panel_year": 2021, "age_at_wave": 40, "occupation_code": 2100, "annual_earnings": 72000.0, "household_income": 125000.0, "first_marriage_year": 2008.0, "first_marriage_end": pd.NA, "total_bio_children": 2.0, "total_marriages": 1.0, "marital_status_collapsed": 1.0, "household_type_40": 1.0},
            {"respondent_id": 2, "panel_year": 2021, "age_at_wave": 39, "occupation_code": 2200, "annual_earnings": 70000.0, "household_income": 122000.0, "first_marriage_year": 2010.0, "first_marriage_end": pd.NA, "total_bio_children": 1.0, "total_marriages": 1.0, "marital_status_collapsed": 1.0, "household_type_40": 1.0},
            {"respondent_id": 3, "panel_year": 2021, "age_at_wave": 38, "occupation_code": 4300, "annual_earnings": 65000.0, "household_income": 100000.0, "first_marriage_year": pd.NA, "first_marriage_end": pd.NA, "total_bio_children": 1.0, "total_marriages": 0.0, "marital_status_collapsed": 2.0, "household_type_40": 2.0},
            {"respondent_id": 4, "panel_year": 2021, "age_at_wave": 41, "occupation_code": 2400, "annual_earnings": 78000.0, "household_income": 135000.0, "first_marriage_year": 2007.0, "first_marriage_end": pd.NA, "total_bio_children": 2.0, "total_marriages": 1.0, "marital_status_collapsed": 1.0, "household_type_40": 1.0},
            {"respondent_id": 5, "panel_year": 2021, "age_at_wave": 40, "occupation_code": 8600, "annual_earnings": 44000.0, "household_income": 72000.0, "first_marriage_year": pd.NA, "first_marriage_end": pd.NA, "total_bio_children": 1.0, "total_marriages": 0.0, "marital_status_collapsed": 3.0, "household_type_40": 3.0},
            {"respondent_id": 6, "panel_year": 2021, "age_at_wave": 39, "occupation_code": 4700, "annual_earnings": 47000.0, "household_income": 78000.0, "first_marriage_year": 2018.0, "first_marriage_end": pd.NA, "total_bio_children": 1.0, "total_marriages": 1.0, "marital_status_collapsed": 1.0, "household_type_40": 2.0},
            {"respondent_id": 7, "panel_year": 2021, "age_at_wave": 38, "occupation_code": 6200, "annual_earnings": 52000.0, "household_income": 82000.0, "first_marriage_year": pd.NA, "first_marriage_end": pd.NA, "total_bio_children": 0.0, "total_marriages": 0.0, "marital_status_collapsed": 2.0, "household_type_40": 4.0},
            {"respondent_id": 8, "panel_year": 2021, "age_at_wave": 41, "occupation_code": 9000, "annual_earnings": 49000.0, "household_income": 79000.0, "first_marriage_year": 2016.0, "first_marriage_end": 2019.0, "total_bio_children": 2.0, "total_marriages": 1.0, "marital_status_collapsed": 4.0, "household_type_40": 3.0},
            {"respondent_id": 1, "panel_year": 2023, "age_at_wave": 42, "health_status": 2.0, "smoking_days_30": 0.0, "alcohol_days_30": 3.0, "binge_days_30": 0.0, "cesd_score": 6.0},
            {"respondent_id": 2, "panel_year": 2023, "age_at_wave": 41, "health_status": 2.0, "smoking_days_30": 1.0, "alcohol_days_30": 4.0, "binge_days_30": 1.0, "cesd_score": 8.0},
            {"respondent_id": 3, "panel_year": 2023, "age_at_wave": 40, "health_status": 3.0, "smoking_days_30": 0.0, "alcohol_days_30": 4.0, "binge_days_30": 1.0, "cesd_score": 9.0},
            {"respondent_id": 4, "panel_year": 2023, "age_at_wave": 43, "health_status": 2.0, "smoking_days_30": 0.0, "alcohol_days_30": 5.0, "binge_days_30": 0.0, "cesd_score": 7.0},
            {"respondent_id": 5, "panel_year": 2023, "age_at_wave": 42, "health_status": 4.0, "smoking_days_30": 10.0, "alcohol_days_30": 2.0, "binge_days_30": 3.0, "cesd_score": 17.0},
            {"respondent_id": 6, "panel_year": 2023, "age_at_wave": 41, "health_status": 4.0, "smoking_days_30": 8.0, "alcohol_days_30": 3.0, "binge_days_30": 2.0, "cesd_score": 16.0},
            {"respondent_id": 7, "panel_year": 2023, "age_at_wave": 40, "health_status": 3.0, "smoking_days_30": 6.0, "alcohol_days_30": 2.0, "binge_days_30": 1.0, "cesd_score": 15.0},
            {"respondent_id": 8, "panel_year": 2023, "age_at_wave": 43, "health_status": 5.0, "smoking_days_30": 12.0, "alcohol_days_30": 1.0, "binge_days_30": 4.0, "cesd_score": 18.0},
        ]
    ).to_parquet(processed_nlsy / "nlsy97_longitudinal_outcome_panel.parquet", index=False)

    pd.DataFrame(
        [
            {"control_history_type": "present_no_history_detail", "anchor_rule": "exact_stratum_median", "event_time_window": "post_3plus", "outcome": "schooling_engagement_months", "scale": "months", "design_rows": 100, "treated_rows": 40, "control_rows": 60, "treated_respondents": 40, "control_respondents": 60, "overlap_strata_n": 5, "row_weighted_strata_att": -1.2, "respondent_collapsed_strata_att": -1.1, "adjusted_treatment_coef": -1.0, "adjusted_treatment_se_hc1": 0.2, "adjusted_status": "ok", "headline_interpretation": "headline_supported"},
            {"control_history_type": "present_no_history_detail", "anchor_rule": "exact_stratum_median", "event_time_window": "post_3plus", "outcome": "arrest_any", "scale": "share", "design_rows": 100, "treated_rows": 40, "control_rows": 60, "treated_respondents": 40, "control_respondents": 60, "overlap_strata_n": 5, "row_weighted_strata_att": 0.02, "respondent_collapsed_strata_att": 0.02, "adjusted_treatment_coef": 0.03, "adjusted_treatment_se_hc1": 0.01, "adjusted_status": "ok", "headline_interpretation": "headline_supported"},
        ]
    ).to_csv(models / "nlsy97_event_time_post_only_preferred_summary.csv", index=False)
    pd.DataFrame(
        [
            {"event_time_window": "post_3plus", "outcome": "schooling_engagement_months", "scale": "months", "spec": "row_weighted_strata_att", "estimate": -1.2, "std_error": 0.2, "n_rows": 100, "n_treated": 40, "n_control": 60, "n_strata": 5, "status": "ok"},
            {"event_time_window": "post_3plus", "outcome": "schooling_engagement_months", "scale": "months", "spec": "respondent_collapsed_strata_att", "estimate": -1.1, "std_error": 0.25, "n_rows": 100, "n_treated": 40, "n_control": 60, "n_strata": 5, "status": "ok"},
            {"event_time_window": "post_3plus", "outcome": "arrest_any", "scale": "share", "spec": "ols_stratum_panel_year_fe_hc1", "estimate": 0.03, "std_error": 0.01, "n_rows": 100, "n_treated": 40, "n_control": 60, "n_strata": 5, "status": "ok"},
        ]
    ).to_csv(models / "nlsy97_event_time_post_only_robustness.csv", index=False)

    result = build_results_appendix(outputs_root=tmp_path, project_root=tmp_path)

    assert result.manifest_path.exists()
    assert result.results_json_path.exists()
    assert result.handoff_path.exists()
    assert result.synthesis_path.exists()
    assert result.nlsy_prevalence_table_path.exists()
    assert result.nlsy_predictor_table_path.exists()
    assert result.nlsy_outcome_gap_table_path.exists()
    assert result.nlsy_race_gap_table_path.exists()
    assert result.nlsy_cognitive_table_path.exists()
    assert result.nlsy_cognitive_subgroup_table_path.exists()
    assert result.nlsy_near_term_effects_table_path.exists()
    assert result.nlsy_near_term_robustness_table_path.exists()
    assert result.nlsy_health_table_path.exists()
    assert result.nlsy_mental_health_table_path.exists()
    assert result.nlsy_family_formation_table_path.exists()
    assert result.nlsy_occupation_summary_table_path.exists()
    assert result.nlsy_occupation_effect_table_path.exists()
    assert result.nlsy_effect_heterogeneity_table_path.exists()
    assert result.nlsy_group_residual_table_path.exists()
    assert result.acs_child_context_table_path.exists()
    assert result.benchmark_context_table_path.exists()

    handoff = result.handoff_path.read_text(encoding="utf-8")
    assert "ready for frontend and documentation work" in handoff.lower()
    assert "g_proxy" in handoff
    assert "post-only" in handoff
    assert "mental-health" in handoff
    assert "occupation" in handoff

    cognitive = pd.read_csv(result.nlsy_cognitive_table_path)
    assert "g_proxy_1997" in set(cognitive["outcome"])
    near_term = pd.read_csv(result.nlsy_near_term_effects_table_path)
    assert set(near_term["outcome"]) == {"schooling_engagement_months", "arrest_any"}
    health = pd.read_csv(result.nlsy_health_table_path)
    assert "poor_health_2023" in set(health["outcome"])
    mental_health = pd.read_csv(result.nlsy_mental_health_table_path)
    assert "cesd_score_2023" in set(mental_health["outcome"])
    family = pd.read_csv(result.nlsy_family_formation_table_path)
    assert "ever_married_2021" in set(family["outcome"])
    occupation_effect = pd.read_csv(result.nlsy_occupation_effect_table_path)
    assert "high_skill_occupation_latest" in set(occupation_effect["outcome"])
    heterogeneity = pd.read_csv(result.nlsy_effect_heterogeneity_table_path)
    assert set(heterogeneity["group_type"]) >= {"sex", "race_ethnicity_3cat", "parent_education_band"}
    residuals = pd.read_csv(result.nlsy_group_residual_table_path)
    assert "outcome" in residuals.columns

    artifact_index = json.loads(result.results_json_path.read_text(encoding="utf-8"))
    assert artifact_index["schema_version"] == RESULTS_SCHEMA_VERSION
    assert artifact_index["generated_at_utc"]
    assert artifact_index["source_manifest"] == "results_appendix_manifest.csv"
    assert artifact_index["synthesis_artifacts"]
    assert "artifacts" in artifact_index
    artifact_names = {item["artifact"] for item in artifact_index["artifacts"]}
    assert {"nlsy_prevalence_table", "benchmark_context_table", "results_synthesis"} <= artifact_names
