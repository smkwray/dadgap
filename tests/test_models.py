from pathlib import Path

import numpy as np
import pandas as pd

from father_longrun.models.ml import build_ml_benchmarks
from father_longrun.models.quasi_causal import build_quasi_causal_scaffold


def test_build_quasi_causal_scaffold(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    processed_root.mkdir(parents=True)

    sibling_rows: list[dict[str, float | int | str | bool]] = []
    for mother_id in range(1, 13):
        for child_offset, treatment in enumerate((0, 1), start=1):
            sibling_rows.append(
                {
                    "mother_id": mother_id,
                    "child_id": mother_id * 10 + child_offset,
                    "home_0_2_eligible_1990": True,
                    "adult_outcome_eligible_2014": True,
                    "primary_treatment_cnlsy_1990": treatment,
                    "primary_treatment_label_cnlsy_1990": "no_father_figure_present" if treatment == 1 else "father_figure_present",
                    "employment_2014_clean": 1 - treatment * 0.2,
                    "annual_earnings_2014_clean": 30000 - treatment * 5000 + mother_id * 100,
                    "age_2014_clean": 24 + child_offset,
                    "child_birth_year": 1989 + child_offset,
                }
            )
    pd.DataFrame(sibling_rows).to_parquet(processed_root / "nlsy79_cnlsy_backbone_analysis_ready.parquet", index=False)

    nlsy97 = pd.DataFrame(
        [
            {
                "respondent_id": idx,
                "primary_treatment_nlsy97": idx % 2,
                "employment_2021": 1,
                "birth_year": 1980 + (idx % 5),
                "sex_raw": 1,
                "race_ethnicity_3cat": "white_non_hispanic",
                "last_year_lived_with_bio_father": 1990 + idx,
            }
            for idx in range(20)
        ]
    )
    nlsy97.to_parquet(processed_root / "nlsy97_analysis_ready.parquet", index=False)
    history_rows: list[dict[str, float | int]] = []
    for idx in range(20):
        history_rows.append(
            {
                "respondent_id": idx,
                "childhood_year": 1995,
                "father_presence_observed": 1,
                "father_presence_imputed": 1,
                "localized_exit_year_available": 1 if idx % 2 else 0,
            }
        )
        history_rows.append(
            {
                "respondent_id": idx,
                "childhood_year": 1997,
                "father_presence_observed": 1,
                "father_presence_imputed": 0 if idx % 2 else 1,
                "localized_exit_year_available": 1 if idx % 2 else 0,
            }
        )
    pd.DataFrame(history_rows).to_parquet(processed_root / "nlsy97_childhood_exposure_history.parquet", index=False)
    panel_rows: list[dict[str, float | int]] = []
    for idx in range(20):
        if idx % 2 == 0:
            childhood_history_type = "stable_present_until_1997" if idx % 4 == 0 else "present_no_history_detail"
            for panel_year, k12_months, college_months, arrest_months, incarc_months, bkrpt_weeks, bkrpt_hours in (
                (2000, 11, 0, 0, 0, 4, 8),
                (2001, 9, 1, 0, 0, 5, 9),
                (2002, 7, 2, 0, 0, 6, 10),
            ):
                panel_rows.append(
                    {
                        "respondent_id": idx,
                        "panel_year": panel_year,
                        "age_at_wave": panel_year - (1980 + (idx % 5)),
                        "first_absent_year": np.nan,
                        "event_time_from_first_absent_year": np.nan,
                        "localized_exit_year_available": 0,
                        "primary_treatment_nlsy97": 0,
                        "childhood_history_type": childhood_history_type,
                        "k12_enrolled_months": k12_months,
                        "k12_vacation_months": 0,
                        "k12_disciplinary_or_other_months": 0,
                        "college_enrolled_months": college_months,
                        "college_4yrplus_months": 0 if college_months == 0 else college_months,
                        "arrest_months": arrest_months,
                        "incarceration_months": incarc_months,
                        "bkrpt_weeks": bkrpt_weeks,
                        "bkrpt_hours": bkrpt_hours,
                    }
                )
            continue
        for panel_year, event_time_value, k12_months, college_months, arrest_months, incarc_months, bkrpt_weeks, bkrpt_hours in (
            (1998, -1, 10, 0, 0, 0, np.nan, np.nan),
            (1999, 0, 8, 0, 1, 0, np.nan, np.nan),
            (2000, 1, 0, 8, 1, 1, 6, 10),
            (2001, 2, 0, 10, 0, 0, 8, 12),
            (2002, 3, 0, 9, 1, 0, 9, 13),
        ):
            panel_rows.append(
                {
                    "respondent_id": idx,
                    "panel_year": panel_year,
                    "age_at_wave": panel_year - (1980 + (idx % 5)),
                    "first_absent_year": 1999,
                    "event_time_from_first_absent_year": event_time_value,
                    "localized_exit_year_available": 1,
                    "primary_treatment_nlsy97": 1,
                    "childhood_history_type": "localized_exit_before_1997",
                    "k12_enrolled_months": k12_months,
                    "k12_vacation_months": 0,
                    "k12_disciplinary_or_other_months": 0,
                    "college_enrolled_months": college_months,
                    "college_4yrplus_months": 0 if college_months == 0 else college_months,
                    "arrest_months": arrest_months,
                    "incarceration_months": incarc_months,
                    "bkrpt_weeks": bkrpt_weeks,
                    "bkrpt_hours": bkrpt_hours,
                }
            )
    pd.DataFrame(panel_rows).to_parquet(processed_root / "nlsy97_longitudinal_outcome_panel.parquet", index=False)

    result = build_quasi_causal_scaffold(processed_root=processed_root, output_dir=output_dir)

    assert result.sibling_design_path.exists()
    assert result.sibling_fe_path.exists()
    assert result.event_time_path.exists()
    assert result.event_time_design_path.exists()
    assert result.event_time_window_summary_path.exists()
    assert result.event_time_comparison_candidates_path.exists()
    assert result.event_time_comparison_support_path.exists()
    assert result.event_time_strategy_path.exists()
    assert result.event_time_post_only_design_path.exists()
    assert result.event_time_post_only_summary_path.exists()
    assert result.event_time_post_only_robustness_path.exists()
    assert result.event_time_post_only_sensitivity_path.exists()
    assert result.event_time_post_only_sensitivity_report_path.exists()
    assert result.event_time_post_only_preferred_summary_path.exists()
    assert result.event_time_post_only_preferred_report_path.exists()
    assert result.event_time_post_only_report_path.exists()
    assert result.readiness_path.exists()
    assert result.summary_path.exists()

    sibling_fe = pd.read_csv(result.sibling_fe_path)
    readiness = pd.read_csv(result.readiness_path)
    event_time = pd.read_csv(result.event_time_path)
    event_time_design = pd.read_parquet(result.event_time_design_path)
    event_time_window_summary = pd.read_csv(result.event_time_window_summary_path)
    comparison_candidates = pd.read_parquet(result.event_time_comparison_candidates_path)
    comparison_support = pd.read_csv(result.event_time_comparison_support_path)
    post_only_design = pd.read_parquet(result.event_time_post_only_design_path)
    post_only_summary = pd.read_csv(result.event_time_post_only_summary_path)
    post_only_robustness = pd.read_csv(result.event_time_post_only_robustness_path)
    post_only_sensitivity = pd.read_csv(result.event_time_post_only_sensitivity_path)
    preferred_post_only = pd.read_csv(result.event_time_post_only_preferred_summary_path)
    assert set(sibling_fe["status"]) == {"estimated"}
    assert readiness.loc[readiness["design"] == "cnlsy_sibling_fe", "status"].iloc[0] == "ready"
    assert event_time.loc[event_time["metric"] == "childhood_history_rows_available", "count"].iloc[0] == 40
    assert event_time.loc[event_time["metric"] == "localized_exit_respondents", "count"].iloc[0] == 10
    assert event_time.loc[event_time["metric"] == "near_term_design_rows_available", "count"].iloc[0] == 50
    assert event_time.loc[event_time["metric"] == "baseline_present_control_candidates", "count"].iloc[0] == 5
    assert len(event_time_design.index) == 50
    assert event_time_design["respondent_id"].nunique() == 10
    assert set(event_time_design["event_time_window"]) == {"pre_1", "event_year", "post_1", "post_2", "post_3plus"}
    assert event_time_window_summary.loc[event_time_window_summary["event_time_window"] == "event_year", "rows"].iloc[0] == 10
    assert event_time_window_summary.loc[event_time_window_summary["event_time_window"] == "post_1", "bkrpt_observed_rows"].iloc[0] == 10
    assert comparison_candidates.loc[comparison_candidates["comparison_role"] == "control_baseline_present", "respondent_id"].nunique() == 5
    assert comparison_candidates.loc[comparison_candidates["comparison_role"] == "treated_localized_exit", "exact_overlap_available"].sum() == 10
    assert comparison_support["exact_overlap"].sum() >= 1
    assert set(post_only_design["comparison_group"]) == {"treated", "control"}
    assert post_only_summary.loc[post_only_summary["event_time_window"] == "post_1", "control_rows"].max() > 0
    assert post_only_summary.loc[post_only_summary["outcome"] == "schooling_engagement_months", "strata_att_difference"].notna().any()
    assert post_only_summary.loc[post_only_summary["outcome"] == "schooling_engagement_months", "respondent_collapsed_strata_att_difference"].notna().any()
    assert post_only_summary.loc[post_only_summary["outcome"] == "schooling_engagement_months", "adjusted_treatment_coef"].notna().any()
    assert "ols_stratum_panel_year_fe_hc1" in set(post_only_robustness["spec"])
    assert post_only_robustness.loc[post_only_robustness["spec"] == "ols_stratum_panel_year_fe_hc1", "estimate"].notna().any()
    assert {"exact_stratum_median", "birth_year_median", "overall_median"} <= set(post_only_sensitivity["anchor_rule"])
    assert {"stable_present_until_1997", "present_no_history_detail"} <= set(post_only_sensitivity["control_history_type"])
    assert post_only_sensitivity.loc[post_only_sensitivity["adjusted_status"] == "estimated", "adjusted_treatment_coef"].notna().any()
    assert set(preferred_post_only["control_history_type"]) == {"present_no_history_detail"}
    assert set(preferred_post_only["anchor_rule"]) == {"exact_stratum_median"}
    assert set(preferred_post_only["event_time_window"]) == {"post_3plus"}
    assert {"headline_supported"} <= set(preferred_post_only["headline_interpretation"])


def test_build_ml_benchmarks(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    processed_root.mkdir(parents=True)

    rng = np.random.default_rng(97)
    n = 180
    treatment = rng.binomial(1, 0.35, size=n)
    parent_education = rng.normal(13, 2, size=n)
    birth_year = rng.integers(1980, 1985, size=n)
    hh2010 = rng.normal(60000, 15000, size=n)
    earn2019 = rng.normal(45000, 12000, size=n)
    hh2019 = rng.normal(85000, 20000, size=n)
    emp2019 = rng.choice([0, 1], size=n, p=[0.2, 0.8])
    sex = rng.choice([1, 2], size=n)
    race = rng.choice(["white_non_hispanic", "black_non_hispanic", "hispanic"], size=n)
    earn2021 = 10000 + 1800 * parent_education + 0.3 * earn2019 + 0.15 * hh2010 - 6000 * treatment + rng.normal(0, 6000, size=n)
    employment_score = -0.5 + 0.15 * parent_education + 1.2 * emp2019 - 0.6 * treatment + rng.normal(0, 0.7, size=n)
    emp2021 = (employment_score > 0).astype(int)

    pd.DataFrame(
        {
            "respondent_id": np.arange(n),
            "birth_year": birth_year,
            "parent_education": parent_education,
            "household_income_2010": hh2010,
            "annual_earnings_2019": earn2019,
            "household_income_2019": hh2019,
            "employment_2019": emp2019,
            "employment_2021": emp2021,
            "primary_treatment_nlsy97": treatment,
            "primary_treatment_observed_nlsy97": 1,
            "sex_raw": sex,
            "race_ethnicity_3cat": race,
            "annual_earnings_2021_clean": earn2021,
        }
    ).to_parquet(processed_root / "nlsy97_analysis_ready.parquet", index=False)

    result = build_ml_benchmarks(processed_root=processed_root, output_dir=output_dir)

    assert result.readiness_path.exists()
    assert result.metrics_path.exists()
    assert result.feature_importance_path.exists()
    assert result.predictions_path.exists()
    assert result.summary_path.exists()

    metrics = pd.read_csv(result.metrics_path)
    readiness = pd.read_csv(result.readiness_path)
    assert not metrics.empty
    assert "annual_earnings_2021" in set(metrics["task"])
    assert {"annual_earnings_2021_clean", "employment_2021_clean"} <= set(readiness["target"])
    assert readiness.loc[0, "status"] in {"ready", "sklearn_unavailable"}
