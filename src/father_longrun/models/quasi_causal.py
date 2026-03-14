from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MIN_DISCORDANT_FAMILIES = 10
SPECIAL_MISSING_CODES = frozenset({-1, -2, -3, -4, -5, -7})
EVENT_TIME_WINDOW_ORDER = ("pre_3plus", "pre_2", "pre_1", "event_year", "post_1", "post_2", "post_3plus")
POST_ONLY_EVENT_TIME_WINDOWS = ("post_1", "post_2", "post_3plus")
CONTROL_HISTORY_PRIORITY = (
    "stable_present_until_1997",
    "present_no_history_detail",
    "present_but_prior_gap_unlocalized",
)
SENSITIVITY_ANCHOR_RULES = (
    "exact_stratum_median",
    "birth_year_median",
    "overall_median",
)
PRIMARY_POST_ONLY_CONTROL_HISTORY = "present_no_history_detail"
PRIMARY_POST_ONLY_ANCHOR_RULE = "exact_stratum_median"
PRIMARY_POST_ONLY_WINDOW = "post_3plus"


@dataclass(frozen=True)
class QuasiCausalBuildResult:
    sibling_design_path: Path
    sibling_fe_path: Path
    event_time_path: Path
    event_time_design_path: Path
    event_time_window_summary_path: Path
    event_time_comparison_candidates_path: Path
    event_time_comparison_support_path: Path
    event_time_strategy_path: Path
    event_time_post_only_design_path: Path
    event_time_post_only_summary_path: Path
    event_time_post_only_robustness_path: Path
    event_time_post_only_sensitivity_path: Path
    event_time_post_only_sensitivity_report_path: Path
    event_time_post_only_preferred_summary_path: Path
    event_time_post_only_preferred_report_path: Path
    event_time_post_only_report_path: Path
    readiness_path: Path
    summary_path: Path


def _clean_binary_employment(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.mask(numeric < 0)
    numeric = numeric.mask(numeric == 0, 0)
    numeric = numeric.mask(numeric.isin([1, 2]), 1)
    return numeric


def _clean_special_missing(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(~numeric.isin(SPECIAL_MISSING_CODES))


def _event_time_window_label(value: float | int | None) -> str | None:
    if pd.isna(value):
        return None
    event_time = int(value)
    if event_time <= -3:
        return "pre_3plus"
    if event_time == -2:
        return "pre_2"
    if event_time == -1:
        return "pre_1"
    if event_time == 0:
        return "event_year"
    if event_time == 1:
        return "post_1"
    if event_time == 2:
        return "post_2"
    return "post_3plus"


def _indicator_from_positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    indicator = pd.Series(np.nan, index=numeric.index, dtype="float64")
    indicator.loc[numeric.notna()] = 0.0
    indicator.loc[numeric > 0] = 1.0
    return indicator


def _safe_float_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    value = numeric.mean()
    return float(value) if not pd.isna(value) else np.nan


def _strata_att_difference(frame: pd.DataFrame, *, outcome: str) -> tuple[float, int]:
    design = frame[["stratum_key_exact", "comparison_group", outcome]].dropna().copy()
    if design.empty:
        return (np.nan, 0)
    summary = (
        design.groupby(["stratum_key_exact", "comparison_group"], dropna=False)[outcome]
        .agg(["mean", "count"])
        .reset_index()
    )
    pivot = summary.pivot(index="stratum_key_exact", columns="comparison_group")
    if ("mean", "treated") not in pivot.columns or ("mean", "control") not in pivot.columns:
        return (np.nan, 0)
    complete = pivot.dropna(subset=[("mean", "treated"), ("mean", "control")]).copy()
    if complete.empty:
        return (np.nan, 0)
    treated_weights = complete[("count", "treated")].fillna(0)
    total_weight = treated_weights.sum()
    if total_weight <= 0:
        return (np.nan, int(len(complete.index)))
    diff = complete[("mean", "treated")] - complete[("mean", "control")]
    att = float((diff * treated_weights).sum() / total_weight)
    return (att, int(len(complete.index)))


def _respondent_collapsed_contrast(frame: pd.DataFrame, *, outcome: str) -> dict[str, float | int]:
    design = frame[["respondent_id", "stratum_key_exact", "comparison_group", outcome]].dropna().copy()
    if design.empty:
        return {
            "respondent_collapsed_rows": 0,
            "respondent_collapsed_treated_respondents": 0,
            "respondent_collapsed_control_respondents": 0,
            "respondent_collapsed_treated_mean": np.nan,
            "respondent_collapsed_control_mean": np.nan,
            "respondent_collapsed_raw_difference": np.nan,
            "respondent_collapsed_strata_att_difference": np.nan,
            "respondent_collapsed_overlap_strata_n": 0,
        }
    respondent_design = (
        design.groupby(["respondent_id", "stratum_key_exact", "comparison_group"], dropna=False)[outcome]
        .mean()
        .reset_index()
    )
    treated = respondent_design.loc[respondent_design["comparison_group"] == "treated", outcome]
    control = respondent_design.loc[respondent_design["comparison_group"] == "control", outcome]
    respondent_att, overlap_strata_n = _strata_att_difference(respondent_design, outcome=outcome)
    treated_mean = _safe_float_mean(treated) if not treated.empty else np.nan
    control_mean = _safe_float_mean(control) if not control.empty else np.nan
    return {
        "respondent_collapsed_rows": int(len(respondent_design.index)),
        "respondent_collapsed_treated_respondents": int(
            respondent_design.loc[respondent_design["comparison_group"] == "treated", "respondent_id"].nunique()
        ),
        "respondent_collapsed_control_respondents": int(
            respondent_design.loc[respondent_design["comparison_group"] == "control", "respondent_id"].nunique()
        ),
        "respondent_collapsed_treated_mean": treated_mean,
        "respondent_collapsed_control_mean": control_mean,
        "respondent_collapsed_raw_difference": (
            treated_mean - control_mean if not pd.isna(treated_mean) and not pd.isna(control_mean) else np.nan
        ),
        "respondent_collapsed_strata_att_difference": respondent_att,
        "respondent_collapsed_overlap_strata_n": overlap_strata_n,
    }


def _adjusted_treatment_effect(
    frame: pd.DataFrame,
    *,
    outcome: str,
    fixed_effects: tuple[str, ...],
) -> dict[str, object]:
    required = ["comparison_group", outcome, *fixed_effects]
    design = frame[required].dropna().copy()
    if design.empty:
        return {
            "adjusted_status": "no_complete_rows",
            "adjusted_n_rows": 0,
            "adjusted_n_treated_rows": 0,
            "adjusted_n_control_rows": 0,
            "adjusted_n_strata": 0,
            "adjusted_n_panel_years": 0,
            "adjusted_treatment_coef": np.nan,
            "adjusted_treatment_se_hc1": np.nan,
        }

    design["treatment"] = (design["comparison_group"] == "treated").astype(float)
    if "stratum_key_exact" in design.columns:
        stratum_support = design.groupby("stratum_key_exact")["comparison_group"].nunique(dropna=True)
        supported_strata = stratum_support.loc[stratum_support >= 2].index
        design = design.loc[design["stratum_key_exact"].isin(supported_strata)].copy()
    if design.empty:
        return {
            "adjusted_status": "insufficient_overlap",
            "adjusted_n_rows": 0,
            "adjusted_n_treated_rows": 0,
            "adjusted_n_control_rows": 0,
            "adjusted_n_strata": 0,
            "adjusted_n_panel_years": 0,
            "adjusted_treatment_coef": np.nan,
            "adjusted_treatment_se_hc1": np.nan,
        }
    if design["treatment"].nunique(dropna=True) < 2:
        return {
            "adjusted_status": "single_group_only",
            "adjusted_n_rows": int(len(design.index)),
            "adjusted_n_treated_rows": int(design["treatment"].sum()),
            "adjusted_n_control_rows": int((1 - design["treatment"]).sum()),
            "adjusted_n_strata": int(design["stratum_key_exact"].nunique()) if "stratum_key_exact" in design.columns else 0,
            "adjusted_n_panel_years": int(design["panel_year"].nunique()) if "panel_year" in design.columns else 0,
            "adjusted_treatment_coef": np.nan,
            "adjusted_treatment_se_hc1": np.nan,
        }

    x_parts: list[pd.DataFrame] = [pd.DataFrame({"intercept": np.ones(len(design.index)), "treatment": design["treatment"]}, index=design.index)]
    for feature in fixed_effects:
        dummies = pd.get_dummies(design[feature].astype("string"), prefix=feature, drop_first=True, dtype=float)
        if not dummies.empty:
            x_parts.append(dummies)
    x_frame = pd.concat(x_parts, axis=1)
    y = pd.to_numeric(design[outcome], errors="coerce")
    keep = y.notna() & x_frame.notna().all(axis=1)
    if not keep.any():
        return {
            "adjusted_status": "no_complete_rows",
            "adjusted_n_rows": 0,
            "adjusted_n_treated_rows": 0,
            "adjusted_n_control_rows": 0,
            "adjusted_n_strata": 0,
            "adjusted_n_panel_years": 0,
            "adjusted_treatment_coef": np.nan,
            "adjusted_treatment_se_hc1": np.nan,
        }
    x = x_frame.loc[keep].to_numpy(dtype=float)
    y_vec = y.loc[keep].to_numpy(dtype=float)
    n_obs, n_coef = x.shape
    if n_obs <= n_coef:
        return {
            "adjusted_status": "insufficient_degrees_of_freedom",
            "adjusted_n_rows": int(n_obs),
            "adjusted_n_treated_rows": int(design.loc[keep, "treatment"].sum()),
            "adjusted_n_control_rows": int((1 - design.loc[keep, "treatment"]).sum()),
            "adjusted_n_strata": int(design.loc[keep, "stratum_key_exact"].nunique()) if "stratum_key_exact" in design.columns else 0,
            "adjusted_n_panel_years": int(design.loc[keep, "panel_year"].nunique()) if "panel_year" in design.columns else 0,
            "adjusted_treatment_coef": np.nan,
            "adjusted_treatment_se_hc1": np.nan,
        }

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y_vec
    residuals = y_vec - x @ beta
    if np.allclose(x[:, 1], x[0, 1]):
        return {
            "adjusted_status": "no_treatment_variation",
            "adjusted_n_rows": int(n_obs),
            "adjusted_n_treated_rows": int(design.loc[keep, "treatment"].sum()),
            "adjusted_n_control_rows": int((1 - design.loc[keep, "treatment"]).sum()),
            "adjusted_n_strata": int(design.loc[keep, "stratum_key_exact"].nunique()) if "stratum_key_exact" in design.columns else 0,
            "adjusted_n_panel_years": int(design.loc[keep, "panel_year"].nunique()) if "panel_year" in design.columns else 0,
            "adjusted_treatment_coef": np.nan,
            "adjusted_treatment_se_hc1": np.nan,
        }
    score = x * residuals[:, None]
    meat = score.T @ score
    hc1_scale = float(n_obs / max(n_obs - n_coef, 1))
    vcov = hc1_scale * (xtx_inv @ meat @ xtx_inv)
    treatment_se = float(np.sqrt(max(vcov[1, 1], 0.0)))
    return {
        "adjusted_status": "estimated",
        "adjusted_n_rows": int(n_obs),
        "adjusted_n_treated_rows": int(design.loc[keep, "treatment"].sum()),
        "adjusted_n_control_rows": int((1 - design.loc[keep, "treatment"]).sum()),
        "adjusted_n_strata": int(design.loc[keep, "stratum_key_exact"].nunique()) if "stratum_key_exact" in design.columns else 0,
        "adjusted_n_panel_years": int(design.loc[keep, "panel_year"].nunique()) if "panel_year" in design.columns else 0,
        "adjusted_treatment_coef": float(beta[1]),
        "adjusted_treatment_se_hc1": treatment_se,
    }


def _assign_control_anchors(
    controls: pd.DataFrame,
    *,
    anchor_rule: str,
    exact_anchor: pd.Series,
    birth_year_anchor: pd.Series,
    overall_anchor: float,
    exact_overlap_keys: set[str],
) -> pd.DataFrame:
    anchored = controls.copy()
    anchored["exact_overlap_available"] = anchored["stratum_key_exact"].isin(exact_overlap_keys).astype(int)
    if anchor_rule == "exact_stratum_median":
        anchored["event_anchor_year"] = anchored["stratum_key_exact"].map(exact_anchor)
    elif anchor_rule == "birth_year_median":
        anchored["event_anchor_year"] = anchored["stratum_key_birth_year"].map(birth_year_anchor)
    elif anchor_rule == "overall_median":
        anchored["event_anchor_year"] = overall_anchor
    else:
        raise ValueError(f"Unsupported anchor_rule: {anchor_rule}")
    anchored["anchor_strategy"] = anchor_rule
    return anchored


def _build_control_post_only_rows(
    *,
    nlsy97_panel: pd.DataFrame | None,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    if nlsy97_panel is None or nlsy97_panel.empty or controls.empty:
        return pd.DataFrame(
            columns=[
                "respondent_id",
                "panel_year",
                "event_anchor_year",
                "event_time_window",
                "anchor_strategy",
                "exact_overlap_available",
                "stratum_key_exact",
                "stratum_key_birth_year",
                "k12_enrolled_months",
                "college_enrolled_months",
                "schooling_engagement_months",
                "schooling_observed",
                "arrest_any",
                "arrest_months",
                "incarceration_any",
                "incarceration_months",
                "justice_observed",
                "bkrpt_weeks",
                "bkrpt_hours",
                "bkrpt_observed",
                "comparison_group",
            ]
        )
    control_post_only = nlsy97_panel.loc[
        nlsy97_panel["respondent_id"].isin(controls["respondent_id"])
        & nlsy97_panel["panel_year"].between(1998, 2005, inclusive="both")
    ].copy()
    control_post_only = control_post_only.merge(
        controls[
            [
                "respondent_id",
                "event_anchor_year",
                "anchor_strategy",
                "exact_overlap_available",
                "stratum_key_exact",
                "stratum_key_birth_year",
            ]
        ],
        on="respondent_id",
        how="left",
        validate="many_to_one",
    )
    control_post_only["event_time_from_anchor_year"] = (
        pd.to_numeric(control_post_only["panel_year"], errors="coerce")
        - pd.to_numeric(control_post_only["event_anchor_year"], errors="coerce")
    )
    control_post_only["event_time_window"] = control_post_only["event_time_from_anchor_year"].map(_event_time_window_label).astype("string")
    control_schooling_observed = control_post_only[
        [
            "k12_enrolled_months",
            "k12_vacation_months",
            "k12_disciplinary_or_other_months",
            "college_enrolled_months",
            "college_4yrplus_months",
        ]
    ].notna().any(axis=1)
    control_post_only["schooling_engagement_months"] = (
        pd.to_numeric(control_post_only["k12_enrolled_months"], errors="coerce").fillna(0)
        + pd.to_numeric(control_post_only["k12_vacation_months"], errors="coerce").fillna(0)
        + pd.to_numeric(control_post_only["college_enrolled_months"], errors="coerce").fillna(0)
    ).where(control_schooling_observed)
    control_post_only["schooling_observed"] = control_schooling_observed.astype(int)
    control_post_only["arrest_any"] = _indicator_from_positive(control_post_only["arrest_months"])
    control_post_only["incarceration_any"] = _indicator_from_positive(control_post_only["incarceration_months"])
    control_post_only["justice_observed"] = control_post_only[["arrest_months", "incarceration_months"]].notna().any(axis=1).astype(int)
    control_post_only["bkrpt_observed"] = control_post_only[["bkrpt_weeks", "bkrpt_hours"]].notna().any(axis=1).astype(int)
    control_post_only = control_post_only.loc[
        control_post_only["event_time_window"].isin(POST_ONLY_EVENT_TIME_WINDOWS)
    ].copy()
    control_post_only["comparison_group"] = "control"
    return control_post_only


def _within_family_fe(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    family_id: str,
    controls: tuple[str, ...] = (),
) -> dict[str, object]:
    required = [family_id, treatment, outcome, *controls]
    design = frame[required].dropna().copy()
    if design.empty:
        return {
            "outcome": outcome,
            "status": "no_complete_rows",
            "n_rows": 0,
            "n_families": 0,
            "n_discordant_families": 0,
            "coefficient": np.nan,
            "std_error": np.nan,
        }

    treatment_variation = design.groupby(family_id)[treatment].nunique(dropna=True)
    discordant_families = treatment_variation.loc[treatment_variation >= 2].index
    n_discordant = int(len(discordant_families))
    if n_discordant < MIN_DISCORDANT_FAMILIES:
        return {
            "outcome": outcome,
            "status": "insufficient_within_family_variation",
            "n_rows": int(len(design.index)),
            "n_families": int(design[family_id].nunique()),
            "n_discordant_families": n_discordant,
            "coefficient": np.nan,
            "std_error": np.nan,
        }

    design = design.loc[design[family_id].isin(discordant_families)].copy()
    demean_cols = [outcome, treatment, *controls]
    family_means = design.groupby(family_id)[demean_cols].transform("mean")
    demeaned = design[demean_cols] - family_means
    y = demeaned[outcome].to_numpy(dtype=float)
    x_cols = [treatment, *controls]
    x = demeaned[x_cols].to_numpy(dtype=float)
    keep = ~np.isnan(y) & ~np.isnan(x).any(axis=1)
    y = y[keep]
    x = x[keep]
    if len(y) <= len(x_cols) or np.allclose(x[:, 0], 0):
        return {
            "outcome": outcome,
            "status": "insufficient_post_demean_variation",
            "n_rows": int(len(y)),
            "n_families": int(design[family_id].nunique()),
            "n_discordant_families": n_discordant,
            "coefficient": np.nan,
            "std_error": np.nan,
        }

    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ beta
    dof = max(len(y) - x.shape[1], 1)
    sigma2 = float((residuals @ residuals) / dof)
    xtx_inv = np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return {
        "outcome": outcome,
        "status": "estimated",
        "n_rows": int(len(y)),
        "n_families": int(design[family_id].nunique()),
        "n_discordant_families": n_discordant,
        "coefficient": float(beta[0]),
        "std_error": float(se[0]),
    }


def build_quasi_causal_scaffold(*, processed_root: Path, output_dir: Path) -> QuasiCausalBuildResult:
    processed_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cnlsy = pd.read_parquet(processed_root / "nlsy79_cnlsy_backbone_analysis_ready.parquet").copy()
    nlsy97 = pd.read_parquet(processed_root / "nlsy97_analysis_ready.parquet").copy()
    panel_path = processed_root / "nlsy97_longitudinal_outcome_panel.parquet"
    nlsy97_panel = pd.read_parquet(panel_path) if panel_path.exists() else None
    childhood_history_path = processed_root / "nlsy97_childhood_exposure_history.parquet"
    nlsy97_history = pd.read_parquet(childhood_history_path) if childhood_history_path.exists() else None
    panel_wave_count = int(nlsy97_panel["panel_year"].nunique()) if nlsy97_panel is not None else 0
    panel_row_count = int(len(nlsy97_panel.index)) if nlsy97_panel is not None else 0

    cnlsy["employment_2014_clean"] = _clean_binary_employment(cnlsy.get("employment_2014_clean", cnlsy.get("employment_2014")))
    sibling_design = cnlsy.loc[
        (cnlsy["home_0_2_eligible_1990"] == True)
        & (cnlsy["adult_outcome_eligible_2014"] == True)
        & (cnlsy["primary_treatment_cnlsy_1990"].notna())
    , [
        "mother_id",
        "child_id",
        "primary_treatment_cnlsy_1990",
        "primary_treatment_label_cnlsy_1990",
        "employment_2014_clean",
        "annual_earnings_2014_clean",
        "age_2014_clean",
        "child_birth_year",
    ]].copy()
    sibling_design_path = processed_root / "cnlsy_sibling_fe_design.parquet"
    sibling_design.to_parquet(sibling_design_path, index=False)

    family_sizes = sibling_design.groupby("mother_id")["child_id"].count()
    treatment_var = sibling_design.groupby("mother_id")["primary_treatment_cnlsy_1990"].nunique(dropna=True)
    emp_var = sibling_design.loc[sibling_design["employment_2014_clean"].notna()].groupby("mother_id")["primary_treatment_cnlsy_1990"].nunique(dropna=True)
    earn_var = sibling_design.loc[sibling_design["annual_earnings_2014_clean"].notna()].groupby("mother_id")["primary_treatment_cnlsy_1990"].nunique(dropna=True)

    fe_rows = [
        _within_family_fe(
            sibling_design,
            outcome="employment_2014_clean",
            treatment="primary_treatment_cnlsy_1990",
            family_id="mother_id",
            controls=("age_2014_clean",),
        ),
        _within_family_fe(
            sibling_design,
            outcome="annual_earnings_2014_clean",
            treatment="primary_treatment_cnlsy_1990",
            family_id="mother_id",
            controls=("age_2014_clean",),
        ),
    ]
    sibling_fe = pd.DataFrame(fe_rows)
    sibling_fe["mothers_with_two_or_more_children"] = int((family_sizes >= 2).sum())
    sibling_fe["mothers_with_treatment_variation"] = int((treatment_var >= 2).sum())
    sibling_fe["employment_outcome_mothers_with_variation"] = int((emp_var >= 2).sum())
    sibling_fe["earnings_outcome_mothers_with_variation"] = int((earn_var >= 2).sum())
    sibling_fe_path = output_dir / "cnlsy_sibling_fe_results.csv"
    sibling_fe.to_csv(sibling_fe_path, index=False)

    nlsy97["employment_2021_clean"] = _clean_binary_employment(nlsy97.get("employment_2021"))
    nlsy97["last_year_lived_with_bio_father_clean"] = _clean_special_missing(nlsy97.get("last_year_lived_with_bio_father"))
    nlsy97["birth_year_clean"] = pd.to_numeric(nlsy97.get("birth_year"), errors="coerce")
    nlsy97["age_at_last_coresidence"] = nlsy97["last_year_lived_with_bio_father_clean"] - nlsy97["birth_year_clean"]
    absent = nlsy97.loc[nlsy97["primary_treatment_nlsy97"] == 1].copy()
    if nlsy97_history is not None and not nlsy97_history.empty:
        history_respondents = int(nlsy97_history["respondent_id"].nunique())
        history_rows = int(len(nlsy97_history.index))
        localized_history = (
            nlsy97_history.loc[nlsy97_history["localized_exit_year_available"] == 1, "respondent_id"].drop_duplicates()
        )
        pre_post_history = (
            nlsy97_history.loc[nlsy97_history["father_presence_observed"] == 1]
            .groupby("respondent_id")["father_presence_imputed"]
            .nunique(dropna=True)
        )
        localized_exit_respondents = int(localized_history.nunique())
        pre_post_respondents = int((pre_post_history >= 2).sum())
    else:
        history_respondents = 0
        history_rows = 0
        localized_exit_respondents = 0
        pre_post_respondents = 0

    near_term_design_columns = [
        "respondent_id",
        "panel_year",
        "age_at_wave",
        "first_absent_year",
        "event_time_from_first_absent_year",
        "event_time_window",
        "k12_enrolled_months",
        "k12_vacation_months",
        "k12_disciplinary_or_other_months",
        "college_enrolled_months",
        "college_4yrplus_months",
        "schooling_engagement_months",
        "schooling_observed",
        "arrest_months",
        "arrest_any",
        "incarceration_months",
        "incarceration_any",
        "justice_observed",
        "bkrpt_weeks",
        "bkrpt_hours",
        "bkrpt_observed",
    ]
    if nlsy97_panel is not None and not nlsy97_panel.empty:
        near_term_panel = nlsy97_panel.copy()
        for column in (
            "panel_year",
            "age_at_wave",
            "first_absent_year",
            "event_time_from_first_absent_year",
            "localized_exit_year_available",
            "primary_treatment_nlsy97",
            "k12_enrolled_months",
            "k12_vacation_months",
            "k12_disciplinary_or_other_months",
            "college_enrolled_months",
            "college_4yrplus_months",
            "arrest_months",
            "incarceration_months",
            "bkrpt_weeks",
            "bkrpt_hours",
        ):
            if column in near_term_panel.columns:
                near_term_panel[column] = pd.to_numeric(near_term_panel[column], errors="coerce")
        near_term_panel = near_term_panel.loc[
            near_term_panel["panel_year"].between(1998, 2005, inclusive="both")
            & (near_term_panel["localized_exit_year_available"] == 1)
            & (near_term_panel["primary_treatment_nlsy97"] == 1)
            & near_term_panel["event_time_from_first_absent_year"].notna()
        ].copy()
        near_term_panel["event_time_window"] = near_term_panel["event_time_from_first_absent_year"].map(_event_time_window_label).astype("string")
        schooling_observed = near_term_panel[
            [
                "k12_enrolled_months",
                "k12_vacation_months",
                "k12_disciplinary_or_other_months",
                "college_enrolled_months",
                "college_4yrplus_months",
            ]
        ].notna().any(axis=1)
        near_term_panel["schooling_engagement_months"] = (
            near_term_panel["k12_enrolled_months"].fillna(0)
            + near_term_panel["k12_vacation_months"].fillna(0)
            + near_term_panel["college_enrolled_months"].fillna(0)
        ).where(schooling_observed)
        near_term_panel["schooling_observed"] = schooling_observed.astype(int)
        near_term_panel["arrest_any"] = _indicator_from_positive(near_term_panel["arrest_months"])
        near_term_panel["incarceration_any"] = _indicator_from_positive(near_term_panel["incarceration_months"])
        near_term_panel["justice_observed"] = near_term_panel[["arrest_months", "incarceration_months"]].notna().any(axis=1).astype(int)
        near_term_panel["bkrpt_observed"] = near_term_panel[["bkrpt_weeks", "bkrpt_hours"]].notna().any(axis=1).astype(int)
        near_term_panel = near_term_panel[near_term_design_columns].copy()
    else:
        near_term_panel = pd.DataFrame(columns=near_term_design_columns)
    event_time_design_path = processed_root / "nlsy97_event_time_design.parquet"
    near_term_panel.to_parquet(event_time_design_path, index=False)

    window_summary_rows: list[dict[str, object]] = []
    for window in EVENT_TIME_WINDOW_ORDER:
        window_frame = near_term_panel.loc[near_term_panel["event_time_window"] == window].copy()
        window_summary_rows.append(
            {
                "event_time_window": window,
                "rows": int(len(window_frame.index)),
                "respondents": int(window_frame["respondent_id"].nunique()) if not window_frame.empty else 0,
                "panel_year_min": float(window_frame["panel_year"].min()) if not window_frame.empty else np.nan,
                "panel_year_max": float(window_frame["panel_year"].max()) if not window_frame.empty else np.nan,
                "event_time_min": float(window_frame["event_time_from_first_absent_year"].min()) if not window_frame.empty else np.nan,
                "event_time_max": float(window_frame["event_time_from_first_absent_year"].max()) if not window_frame.empty else np.nan,
                "schooling_observed_rows": int(window_frame["schooling_observed"].sum()) if not window_frame.empty else 0,
                "schooling_engagement_months_mean": _safe_float_mean(window_frame["schooling_engagement_months"]) if not window_frame.empty else np.nan,
                "k12_enrolled_months_mean": _safe_float_mean(window_frame["k12_enrolled_months"]) if not window_frame.empty else np.nan,
                "college_enrolled_months_mean": _safe_float_mean(window_frame["college_enrolled_months"]) if not window_frame.empty else np.nan,
                "justice_observed_rows": int(window_frame["justice_observed"].sum()) if not window_frame.empty else 0,
                "arrest_any_rate": _safe_float_mean(window_frame["arrest_any"]) if not window_frame.empty else np.nan,
                "arrest_months_mean": _safe_float_mean(window_frame["arrest_months"]) if not window_frame.empty else np.nan,
                "incarceration_any_rate": _safe_float_mean(window_frame["incarceration_any"]) if not window_frame.empty else np.nan,
                "incarceration_months_mean": _safe_float_mean(window_frame["incarceration_months"]) if not window_frame.empty else np.nan,
                "bkrpt_observed_rows": int(window_frame["bkrpt_observed"].sum()) if not window_frame.empty else 0,
                "bkrpt_weeks_mean": _safe_float_mean(window_frame["bkrpt_weeks"]) if not window_frame.empty else np.nan,
                "bkrpt_hours_mean": _safe_float_mean(window_frame["bkrpt_hours"]) if not window_frame.empty else np.nan,
            }
        )
    event_time_window_summary = pd.DataFrame(window_summary_rows)
    event_time_window_summary_path = output_dir / "nlsy97_event_time_window_summary.csv"
    event_time_window_summary.to_csv(event_time_window_summary_path, index=False)
    windows_with_support = int((event_time_window_summary["rows"] > 0).sum()) if not event_time_window_summary.empty else 0
    supported_windows = ", ".join(
        event_time_window_summary.loc[event_time_window_summary["rows"] > 0, "event_time_window"].tolist()
    ) or "none"

    if nlsy97_panel is not None and not nlsy97_panel.empty:
        respondent_panel = (
            nlsy97_panel[
                [
                    "respondent_id",
                    "childhood_history_type",
                    "first_absent_year",
                ]
            ]
            .drop_duplicates(subset=["respondent_id"])
            .copy()
        )
    else:
        respondent_panel = pd.DataFrame(columns=["respondent_id", "childhood_history_type", "first_absent_year"])
    comparison_base = pd.DataFrame({"respondent_id": nlsy97["respondent_id"]})
    comparison_base["birth_year"] = nlsy97.get("birth_year")
    comparison_base["sex_raw"] = nlsy97.get("sex_raw")
    comparison_base["race_ethnicity_3cat"] = nlsy97.get("race_ethnicity_3cat")
    comparison_base["primary_treatment_nlsy97"] = nlsy97.get("primary_treatment_nlsy97")
    comparison_base["primary_treatment_label_nlsy97"] = nlsy97.get(
        "primary_treatment_label_nlsy97",
        nlsy97.get("primary_treatment_nlsy97").map({0: "resident_bio_father_present", 1: "resident_bio_father_absent"})
        if "primary_treatment_nlsy97" in nlsy97.columns
        else pd.Series(pd.NA, index=nlsy97.index, dtype="string"),
    )
    comparison_base = comparison_base.merge(respondent_panel, on="respondent_id", how="left")
    comparison_base["birth_year"] = pd.to_numeric(comparison_base["birth_year"], errors="coerce")
    comparison_base["sex_raw"] = pd.to_numeric(comparison_base["sex_raw"], errors="coerce")
    comparison_base["first_absent_year"] = pd.to_numeric(comparison_base["first_absent_year"], errors="coerce")
    comparison_base["race_ethnicity_3cat"] = comparison_base["race_ethnicity_3cat"].astype("string")
    comparison_base["childhood_history_type"] = comparison_base["childhood_history_type"].astype("string")
    comparison_base["stratum_key_exact"] = (
        comparison_base["birth_year"].astype("Int64").astype("string")
        + "|"
        + comparison_base["sex_raw"].astype("Int64").astype("string")
        + "|"
        + comparison_base["race_ethnicity_3cat"].fillna("missing")
    )
    comparison_base["stratum_key_birth_year"] = comparison_base["birth_year"].astype("Int64").astype("string")

    treated_candidates = comparison_base.loc[
        (comparison_base["primary_treatment_nlsy97"] == 1) & comparison_base["first_absent_year"].notna()
    ].copy()
    treated_candidates["comparison_role"] = "treated_localized_exit"
    control_history_type_used = "none_available"
    stable_present_controls = pd.DataFrame(columns=comparison_base.columns.tolist() + ["comparison_role"])
    for history_type in CONTROL_HISTORY_PRIORITY:
        candidate_controls = comparison_base.loc[
            (comparison_base["primary_treatment_nlsy97"] == 0)
            & (comparison_base["childhood_history_type"] == history_type)
        ].copy()
        if not candidate_controls.empty:
            stable_present_controls = candidate_controls
            control_history_type_used = history_type
            break
    stable_present_controls["comparison_role"] = "control_baseline_present"

    treated_exact_support = (
        treated_candidates.groupby(["birth_year", "sex_raw", "race_ethnicity_3cat"], dropna=False)
        .agg(
            treated_rows=("respondent_id", "size"),
            treated_respondents=("respondent_id", "nunique"),
            treated_first_absent_year_median=("first_absent_year", "median"),
        )
        .reset_index()
    )
    control_exact_support = (
        stable_present_controls.groupby(["birth_year", "sex_raw", "race_ethnicity_3cat"], dropna=False)
        .agg(
            control_rows=("respondent_id", "size"),
            control_respondents=("respondent_id", "nunique"),
        )
        .reset_index()
    )
    comparison_support = treated_exact_support.merge(
        control_exact_support,
        on=["birth_year", "sex_raw", "race_ethnicity_3cat"],
        how="outer",
    )
    comparison_support["treated_rows"] = comparison_support["treated_rows"].fillna(0).astype(int)
    comparison_support["treated_respondents"] = comparison_support["treated_respondents"].fillna(0).astype(int)
    comparison_support["control_rows"] = comparison_support["control_rows"].fillna(0).astype(int)
    comparison_support["control_respondents"] = comparison_support["control_respondents"].fillna(0).astype(int)
    comparison_support["exact_overlap"] = (
        (comparison_support["treated_respondents"] > 0) & (comparison_support["control_respondents"] > 0)
    ).astype(int)
    event_time_comparison_support_path = output_dir / "nlsy97_event_time_comparison_support.csv"
    comparison_support.to_csv(event_time_comparison_support_path, index=False)
    exact_overlap_keys = set(
        comparison_support.loc[comparison_support["exact_overlap"] == 1, "birth_year"].astype("Int64").astype("string")
        + "|"
        + comparison_support.loc[comparison_support["exact_overlap"] == 1, "sex_raw"].astype("Int64").astype("string")
        + "|"
        + comparison_support.loc[comparison_support["exact_overlap"] == 1, "race_ethnicity_3cat"].astype("string").fillna("missing")
    )

    exact_anchor = treated_candidates.groupby("stratum_key_exact", dropna=False)["first_absent_year"].median()
    birth_year_anchor = treated_candidates.groupby("stratum_key_birth_year", dropna=False)["first_absent_year"].median()
    overall_anchor = float(treated_candidates["first_absent_year"].median()) if not treated_candidates.empty else np.nan

    stable_present_controls["event_anchor_year"] = stable_present_controls["stratum_key_exact"].map(exact_anchor)
    stable_present_controls["anchor_strategy"] = "exact_birth_year_sex_race"
    stable_present_controls["exact_overlap_available"] = stable_present_controls["stratum_key_exact"].isin(exact_overlap_keys).astype(int)
    birth_year_only_mask = stable_present_controls["event_anchor_year"].isna()
    stable_present_controls.loc[birth_year_only_mask, "event_anchor_year"] = (
        stable_present_controls.loc[birth_year_only_mask, "stratum_key_birth_year"].map(birth_year_anchor)
    )
    stable_present_controls.loc[
        birth_year_only_mask & stable_present_controls["event_anchor_year"].notna(), "anchor_strategy"
    ] = "birth_year_only"
    stable_present_controls.loc[stable_present_controls["event_anchor_year"].isna(), "event_anchor_year"] = overall_anchor
    stable_present_controls.loc[
        stable_present_controls["anchor_strategy"].ne("exact_birth_year_sex_race")
        & stable_present_controls["event_anchor_year"].notna()
        & stable_present_controls["anchor_strategy"].ne("birth_year_only"),
        "anchor_strategy",
    ] = "overall_median"

    treated_candidates["event_anchor_year"] = treated_candidates["first_absent_year"]
    treated_candidates["anchor_strategy"] = "observed_exit_year"
    treated_candidates["exact_overlap_available"] = treated_candidates["stratum_key_exact"].isin(exact_overlap_keys).astype(int)

    comparison_candidates = pd.concat(
        [
            treated_candidates,
            stable_present_controls,
        ],
        ignore_index=True,
    )[
        [
            "respondent_id",
            "comparison_role",
            "birth_year",
            "sex_raw",
            "race_ethnicity_3cat",
            "primary_treatment_nlsy97",
            "primary_treatment_label_nlsy97",
            "childhood_history_type",
            "first_absent_year",
            "event_anchor_year",
            "anchor_strategy",
            "exact_overlap_available",
            "stratum_key_exact",
            "stratum_key_birth_year",
        ]
    ].copy()
    event_time_comparison_candidates_path = processed_root / "nlsy97_event_time_comparison_candidates.parquet"
    comparison_candidates.to_parquet(event_time_comparison_candidates_path, index=False)

    treated_with_exact_control_overlap = int(
        comparison_candidates.loc[
            (comparison_candidates["comparison_role"] == "treated_localized_exit")
            & (comparison_candidates["exact_overlap_available"] == 1),
            "respondent_id",
        ].nunique()
    )
    stable_present_control_candidates_n = int(
        comparison_candidates.loc[
            comparison_candidates["comparison_role"] == "control_baseline_present", "respondent_id"
        ].nunique()
    )
    exact_overlap_strata = int(comparison_support["exact_overlap"].sum()) if not comparison_support.empty else 0
    comparison_strategy_path = output_dir / "nlsy97_event_time_strategy.md"
    strategy_lines = [
        "# NLSY97 Event-Time Comparison Strategy",
        "",
        "## Recommended comparison set",
        "",
        "- Treated group: localized-exit respondents with observed `first_absent_year`.",
        f"- Control group: respondents coded `{control_history_type_used}` in the childhood-history scaffold.",
        "- Matching/anchor rule: exact `birth_year x sex_raw x race_ethnicity_3cat` strata where available; otherwise birth-year fallback; otherwise overall treated median exit year.",
        "- Interpretation: this supports age/calendar-aligned post-event comparisons only. It does not recover treated pretrends.",
        "",
        "## Live support",
        "",
        f"- Treated localized-exit respondents: {int(treated_candidates['respondent_id'].nunique())}",
        f"- Baseline-present control candidates: {stable_present_control_candidates_n}",
        f"- Treated respondents with exact-overlap controls: {treated_with_exact_control_overlap}",
        f"- Exact-overlap strata: {exact_overlap_strata}",
        f"- Supported near-term windows in the treated design: {supported_windows}",
        "",
        "## Constraint",
        "",
        "The current public-use scaffold supports only post-event treated windows (`post_1`, `post_2`, `post_3plus`).",
        "Any causal estimator would therefore need a post-only comparison design rather than a standard pre/post event-study with treated-unit leads.",
    ]
    comparison_strategy_path.write_text("\n".join(strategy_lines) + "\n", encoding="utf-8")

    treated_post_only = near_term_panel.loc[
        near_term_panel["event_time_window"].isin(POST_ONLY_EVENT_TIME_WINDOWS)
    ].copy()
    treated_post_only = treated_post_only.merge(
        treated_candidates[
            [
                "respondent_id",
                "event_anchor_year",
                "anchor_strategy",
                "exact_overlap_available",
                "stratum_key_exact",
                "stratum_key_birth_year",
            ]
        ],
        on="respondent_id",
        how="left",
        validate="many_to_one",
    )
    treated_post_only["comparison_group"] = "treated"

    primary_controls = _assign_control_anchors(
        stable_present_controls,
        anchor_rule="exact_stratum_median",
        exact_anchor=exact_anchor,
        birth_year_anchor=birth_year_anchor,
        overall_anchor=overall_anchor,
        exact_overlap_keys=exact_overlap_keys,
    )
    control_post_only = _build_control_post_only_rows(
        nlsy97_panel=nlsy97_panel,
        controls=primary_controls,
    )

    post_only_design_columns = [
        "respondent_id",
        "comparison_group",
        "panel_year",
        "event_anchor_year",
        "event_time_window",
        "anchor_strategy",
        "exact_overlap_available",
        "stratum_key_exact",
        "stratum_key_birth_year",
        "k12_enrolled_months",
        "college_enrolled_months",
        "schooling_engagement_months",
        "schooling_observed",
        "arrest_any",
        "arrest_months",
        "incarceration_any",
        "incarceration_months",
        "justice_observed",
        "bkrpt_weeks",
        "bkrpt_hours",
        "bkrpt_observed",
    ]
    post_only_design = pd.concat(
        [
            treated_post_only.reindex(columns=post_only_design_columns),
            control_post_only.reindex(columns=post_only_design_columns),
        ],
        ignore_index=True,
    )
    post_only_design = post_only_design.loc[post_only_design["exact_overlap_available"] == 1].copy()
    event_time_post_only_design_path = processed_root / "nlsy97_event_time_post_only_comparison_design.parquet"
    post_only_design.to_parquet(event_time_post_only_design_path, index=False)

    post_only_outcomes = (
        ("schooling_engagement_months", "months"),
        ("k12_enrolled_months", "months"),
        ("college_enrolled_months", "months"),
        ("arrest_any", "share"),
        ("incarceration_any", "share"),
        ("bkrpt_weeks", "level"),
        ("bkrpt_hours", "level"),
    )
    post_only_rows: list[dict[str, object]] = []
    post_only_robustness_rows: list[dict[str, object]] = []
    for window in POST_ONLY_EVENT_TIME_WINDOWS:
        window_frame = post_only_design.loc[post_only_design["event_time_window"] == window].copy()
        for outcome, scale in post_only_outcomes:
            treated_frame = window_frame.loc[window_frame["comparison_group"] == "treated", ["stratum_key_exact", outcome]].dropna()
            control_frame = window_frame.loc[window_frame["comparison_group"] == "control", ["stratum_key_exact", outcome]].dropna()
            combined = pd.concat(
                [
                    treated_frame.assign(comparison_group="treated"),
                    control_frame.assign(comparison_group="control"),
                ],
                ignore_index=True,
            )
            strata_att, overlap_strata_n = _strata_att_difference(combined, outcome=outcome)
            respondent_collapsed = _respondent_collapsed_contrast(window_frame, outcome=outcome)
            adjusted = _adjusted_treatment_effect(
                window_frame,
                outcome=outcome,
                fixed_effects=("stratum_key_exact", "panel_year"),
            )
            post_only_rows.append(
                {
                    "event_time_window": window,
                    "outcome": outcome,
                    "scale": scale,
                    "treated_rows": int(len(treated_frame.index)),
                    "treated_respondents": int(window_frame.loc[window_frame["comparison_group"] == "treated", "respondent_id"].nunique()),
                    "control_rows": int(len(control_frame.index)),
                    "control_respondents": int(window_frame.loc[window_frame["comparison_group"] == "control", "respondent_id"].nunique()),
                    "treated_mean": _safe_float_mean(treated_frame[outcome]) if not treated_frame.empty else np.nan,
                    "control_mean": _safe_float_mean(control_frame[outcome]) if not control_frame.empty else np.nan,
                    "raw_difference": (
                        _safe_float_mean(treated_frame[outcome]) - _safe_float_mean(control_frame[outcome])
                        if not treated_frame.empty and not control_frame.empty
                        else np.nan
                    ),
                    "strata_att_difference": strata_att,
                    "overlap_strata_n": overlap_strata_n,
                    **respondent_collapsed,
                    **adjusted,
                }
            )
            post_only_robustness_rows.extend(
                [
                    {
                        "event_time_window": window,
                        "outcome": outcome,
                        "scale": scale,
                        "spec": "row_weighted_strata_att",
                        "estimate": strata_att,
                        "std_error": np.nan,
                        "n_rows": int(len(combined.index)),
                        "n_treated": int(treated_frame["stratum_key_exact"].size),
                        "n_control": int(control_frame["stratum_key_exact"].size),
                        "n_strata": overlap_strata_n,
                        "status": "estimated" if not pd.isna(strata_att) else "insufficient_overlap",
                    },
                    {
                        "event_time_window": window,
                        "outcome": outcome,
                        "scale": scale,
                        "spec": "respondent_collapsed_strata_att",
                        "estimate": respondent_collapsed["respondent_collapsed_strata_att_difference"],
                        "std_error": np.nan,
                        "n_rows": respondent_collapsed["respondent_collapsed_rows"],
                        "n_treated": respondent_collapsed["respondent_collapsed_treated_respondents"],
                        "n_control": respondent_collapsed["respondent_collapsed_control_respondents"],
                        "n_strata": respondent_collapsed["respondent_collapsed_overlap_strata_n"],
                        "status": "estimated"
                        if not pd.isna(respondent_collapsed["respondent_collapsed_strata_att_difference"])
                        else "insufficient_overlap",
                    },
                    {
                        "event_time_window": window,
                        "outcome": outcome,
                        "scale": scale,
                        "spec": "ols_stratum_panel_year_fe_hc1",
                        "estimate": adjusted["adjusted_treatment_coef"],
                        "std_error": adjusted["adjusted_treatment_se_hc1"],
                        "n_rows": adjusted["adjusted_n_rows"],
                        "n_treated": adjusted["adjusted_n_treated_rows"],
                        "n_control": adjusted["adjusted_n_control_rows"],
                        "n_strata": adjusted["adjusted_n_strata"],
                        "status": adjusted["adjusted_status"],
                    },
                ]
            )
    post_only_summary = pd.DataFrame(post_only_rows)
    event_time_post_only_summary_path = output_dir / "nlsy97_event_time_post_only_summary.csv"
    post_only_summary.to_csv(event_time_post_only_summary_path, index=False)
    post_only_robustness = pd.DataFrame(post_only_robustness_rows)
    event_time_post_only_robustness_path = output_dir / "nlsy97_event_time_post_only_robustness.csv"
    post_only_robustness.to_csv(event_time_post_only_robustness_path, index=False)
    post_only_sensitivity_rows: list[dict[str, object]] = []
    for history_type in CONTROL_HISTORY_PRIORITY:
        control_pool = comparison_base.loc[
            (comparison_base["primary_treatment_nlsy97"] == 0)
            & (comparison_base["childhood_history_type"] == history_type)
        ].copy()
        for anchor_rule in SENSITIVITY_ANCHOR_RULES:
            if control_pool.empty:
                for window in POST_ONLY_EVENT_TIME_WINDOWS:
                    for outcome, scale in post_only_outcomes:
                        post_only_sensitivity_rows.append(
                            {
                                "control_history_type": history_type,
                                "anchor_rule": anchor_rule,
                                "event_time_window": window,
                                "outcome": outcome,
                                "scale": scale,
                                "design_rows": 0,
                                "treated_rows": 0,
                                "control_rows": 0,
                                "treated_respondents": 0,
                                "control_respondents": 0,
                                "overlap_strata_n": 0,
                                "row_weighted_strata_att": np.nan,
                                "respondent_collapsed_strata_att": np.nan,
                                "adjusted_treatment_coef": np.nan,
                                "adjusted_treatment_se_hc1": np.nan,
                                "adjusted_status": "no_controls_available",
                            }
                        )
                continue
            sensitivity_controls = _assign_control_anchors(
                control_pool,
                anchor_rule=anchor_rule,
                exact_anchor=exact_anchor,
                birth_year_anchor=birth_year_anchor,
                overall_anchor=overall_anchor,
                exact_overlap_keys=exact_overlap_keys,
            )
            sensitivity_control_post_only = _build_control_post_only_rows(
                nlsy97_panel=nlsy97_panel,
                controls=sensitivity_controls,
            )
            sensitivity_design = pd.concat(
                [
                    treated_post_only.reindex(columns=post_only_design_columns),
                    sensitivity_control_post_only.reindex(columns=post_only_design_columns),
                ],
                ignore_index=True,
            )
            sensitivity_design = sensitivity_design.loc[sensitivity_design["exact_overlap_available"] == 1].copy()
            for window in POST_ONLY_EVENT_TIME_WINDOWS:
                window_frame = sensitivity_design.loc[sensitivity_design["event_time_window"] == window].copy()
                for outcome, scale in post_only_outcomes:
                    treated_frame = window_frame.loc[
                        window_frame["comparison_group"] == "treated", ["stratum_key_exact", outcome]
                    ].dropna()
                    control_frame = window_frame.loc[
                        window_frame["comparison_group"] == "control", ["stratum_key_exact", outcome]
                    ].dropna()
                    combined = pd.concat(
                        [
                            treated_frame.assign(comparison_group="treated"),
                            control_frame.assign(comparison_group="control"),
                        ],
                        ignore_index=True,
                    )
                    strata_att, overlap_strata_n = _strata_att_difference(combined, outcome=outcome)
                    respondent_collapsed = _respondent_collapsed_contrast(window_frame, outcome=outcome)
                    adjusted = _adjusted_treatment_effect(
                        window_frame,
                        outcome=outcome,
                        fixed_effects=("stratum_key_exact", "panel_year"),
                    )
                    post_only_sensitivity_rows.append(
                        {
                            "control_history_type": history_type,
                            "anchor_rule": anchor_rule,
                            "event_time_window": window,
                            "outcome": outcome,
                            "scale": scale,
                            "design_rows": int(len(window_frame.index)),
                            "treated_rows": int(len(treated_frame.index)),
                            "control_rows": int(len(control_frame.index)),
                            "treated_respondents": int(
                                window_frame.loc[window_frame["comparison_group"] == "treated", "respondent_id"].nunique()
                            ),
                            "control_respondents": int(
                                window_frame.loc[window_frame["comparison_group"] == "control", "respondent_id"].nunique()
                            ),
                            "overlap_strata_n": overlap_strata_n,
                            "row_weighted_strata_att": strata_att,
                            "respondent_collapsed_strata_att": respondent_collapsed["respondent_collapsed_strata_att_difference"],
                            "adjusted_treatment_coef": adjusted["adjusted_treatment_coef"],
                            "adjusted_treatment_se_hc1": adjusted["adjusted_treatment_se_hc1"],
                            "adjusted_status": adjusted["adjusted_status"],
                        }
                    )
    post_only_sensitivity = pd.DataFrame(post_only_sensitivity_rows)
    event_time_post_only_sensitivity_path = output_dir / "nlsy97_event_time_post_only_sensitivity.csv"
    post_only_sensitivity.to_csv(event_time_post_only_sensitivity_path, index=False)
    event_time_post_only_sensitivity_report_path = output_dir / "nlsy97_event_time_post_only_sensitivity.md"
    sensitivity_focus = post_only_sensitivity.loc[
        post_only_sensitivity["event_time_window"].eq("post_3plus")
        & post_only_sensitivity["outcome"].isin(["schooling_engagement_months", "arrest_any"])
    ].copy()
    sensitivity_report_lines = [
        "# NLSY97 Post-Only Sensitivity Report",
        "",
        "Alternative baseline-present control pools and anchor rules are compared below for the best-supported `post_3plus` window.",
        "The adjusted columns use HC1 OLS with exact-stratum and panel-year fixed effects on overlap-supported strata only.",
        "",
        "| control_history_type | anchor_rule | outcome | treated_rows | control_rows | overlap_strata | row_att | respondent_att | adjusted_coef | adjusted_se | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in sensitivity_focus.iterrows():
        sensitivity_report_lines.append(
            f"| {row['control_history_type']} | {row['anchor_rule']} | {row['outcome']} | "
            f"{int(row['treated_rows'])} | {int(row['control_rows'])} | {int(row['overlap_strata_n'])} | "
            f"{'' if pd.isna(row['row_weighted_strata_att']) else round(float(row['row_weighted_strata_att']), 4)} | "
            f"{'' if pd.isna(row['respondent_collapsed_strata_att']) else round(float(row['respondent_collapsed_strata_att']), 4)} | "
            f"{'' if pd.isna(row['adjusted_treatment_coef']) else round(float(row['adjusted_treatment_coef']), 4)} | "
            f"{'' if pd.isna(row['adjusted_treatment_se_hc1']) else round(float(row['adjusted_treatment_se_hc1']), 4)} | "
            f"{row['adjusted_status']} |"
        )
    event_time_post_only_sensitivity_report_path.write_text("\n".join(sensitivity_report_lines) + "\n", encoding="utf-8")
    preferred_post_only = post_only_sensitivity.loc[
        post_only_sensitivity["control_history_type"].eq(PRIMARY_POST_ONLY_CONTROL_HISTORY)
        & post_only_sensitivity["anchor_rule"].eq(PRIMARY_POST_ONLY_ANCHOR_RULE)
        & post_only_sensitivity["event_time_window"].eq(PRIMARY_POST_ONLY_WINDOW)
    ].copy()
    preferred_post_only["headline_interpretation"] = np.where(
        preferred_post_only["adjusted_status"].eq("estimated"),
        "headline_supported",
        "diagnostic_only",
    )
    event_time_post_only_preferred_summary_path = output_dir / "nlsy97_event_time_post_only_preferred_summary.csv"
    preferred_post_only.to_csv(event_time_post_only_preferred_summary_path, index=False)
    event_time_post_only_preferred_report_path = output_dir / "nlsy97_event_time_post_only_preferred.md"
    preferred_report_lines = [
        "# NLSY97 Preferred Post-Only Specification",
        "",
        f"- Preferred control pool: `{PRIMARY_POST_ONLY_CONTROL_HISTORY}`",
        f"- Preferred anchor rule: `{PRIMARY_POST_ONLY_ANCHOR_RULE}`",
        f"- Preferred headline window: `{PRIMARY_POST_ONLY_WINDOW}`",
        "- Interpretation rule: only rows marked `headline_supported` belong in first-pass headline writeups; all other windows/specifications stay diagnostic.",
        "",
        "| outcome | control_rows | overlap_strata | row_att | respondent_att | adjusted_coef | adjusted_se | status | interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in preferred_post_only.iterrows():
        preferred_report_lines.append(
            f"| {row['outcome']} | {int(row['control_rows'])} | {int(row['overlap_strata_n'])} | "
            f"{'' if pd.isna(row['row_weighted_strata_att']) else round(float(row['row_weighted_strata_att']), 4)} | "
            f"{'' if pd.isna(row['respondent_collapsed_strata_att']) else round(float(row['respondent_collapsed_strata_att']), 4)} | "
            f"{'' if pd.isna(row['adjusted_treatment_coef']) else round(float(row['adjusted_treatment_coef']), 4)} | "
            f"{'' if pd.isna(row['adjusted_treatment_se_hc1']) else round(float(row['adjusted_treatment_se_hc1']), 4)} | "
            f"{row['adjusted_status']} | {row['headline_interpretation']} |"
        )
    event_time_post_only_preferred_report_path.write_text("\n".join(preferred_report_lines) + "\n", encoding="utf-8")
    event_time_post_only_report_path = output_dir / "nlsy97_event_time_post_only_report.md"
    report_lines = [
        "# NLSY97 Post-Only Near-Term Comparison Report",
        "",
        f"- Treated localized-exit rows in post-only design: {int((post_only_design['comparison_group'] == 'treated').sum()) if not post_only_design.empty else 0}",
        f"- Baseline-present control rows in post-only design: {int((post_only_design['comparison_group'] == 'control').sum()) if not post_only_design.empty else 0}",
        f"- Supported post-only windows: {', '.join(POST_ONLY_EVENT_TIME_WINDOWS)}",
        "",
        "The estimates below are descriptive post-only treated-versus-baseline-present contrasts, not causal event-study estimates.",
        "Strata-adjusted differences use exact `birth_year x sex_raw x race_ethnicity_3cat` overlap strata and treated-row weights.",
        "Robustness columns add respondent-collapsed exact-strata contrasts and HC1 OLS treatment coefficients with exact-stratum and panel-year fixed effects.",
        f"Headline interpretation is now restricted to the preferred supported spec in `{PRIMARY_POST_ONLY_WINDOW}` with `{PRIMARY_POST_ONLY_CONTROL_HISTORY}` controls and `{PRIMARY_POST_ONLY_ANCHOR_RULE}` anchors; thinner windows and unsupported sensitivity combinations are diagnostic only.",
        "",
        "| window | outcome | treated_mean | control_mean | raw_diff | strata_att | respondent_att | adjusted_coef | adjusted_se | overlap_strata |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in post_only_summary.iterrows():
        report_lines.append(
            f"| {row['event_time_window']} | {row['outcome']} | "
            f"{'' if pd.isna(row['treated_mean']) else round(float(row['treated_mean']), 4)} | "
            f"{'' if pd.isna(row['control_mean']) else round(float(row['control_mean']), 4)} | "
            f"{'' if pd.isna(row['raw_difference']) else round(float(row['raw_difference']), 4)} | "
            f"{'' if pd.isna(row['strata_att_difference']) else round(float(row['strata_att_difference']), 4)} | "
            f"{'' if pd.isna(row['respondent_collapsed_strata_att_difference']) else round(float(row['respondent_collapsed_strata_att_difference']), 4)} | "
            f"{'' if pd.isna(row['adjusted_treatment_coef']) else round(float(row['adjusted_treatment_coef']), 4)} | "
            f"{'' if pd.isna(row['adjusted_treatment_se_hc1']) else round(float(row['adjusted_treatment_se_hc1']), 4)} | "
            f"{int(row['overlap_strata_n'])} |"
        )
    event_time_post_only_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    near_term_design_rows = int(len(near_term_panel.index))
    near_term_design_respondents = int(near_term_panel["respondent_id"].nunique()) if not near_term_panel.empty else 0
    near_term_schooling_rows = int(near_term_panel["schooling_observed"].sum()) if not near_term_panel.empty else 0
    near_term_bkrpt_rows = int(near_term_panel["bkrpt_observed"].sum()) if not near_term_panel.empty else 0
    near_term_justice_rows = int(near_term_panel["justice_observed"].sum()) if not near_term_panel.empty else 0
    post_only_design_rows = int(len(post_only_design.index)) if not post_only_design.empty else 0
    post_only_treated_rows = int((post_only_design["comparison_group"] == "treated").sum()) if not post_only_design.empty else 0
    post_only_control_rows = int((post_only_design["comparison_group"] == "control").sum()) if not post_only_design.empty else 0
    post_only_sensitivity_specs = int(
        post_only_sensitivity[["control_history_type", "anchor_rule"]].drop_duplicates().shape[0]
    ) if not post_only_sensitivity.empty else 0
    preferred_supported_outcomes = int(
        preferred_post_only["headline_interpretation"].eq("headline_supported").sum()
    ) if not preferred_post_only.empty else 0
    event_time = pd.DataFrame(
        [
            {
                "metric": "absent_rows",
                "count": int(len(absent.index)),
                "share": 1.0 if len(absent.index) else 0.0,
                "note": "Rows with locked primary treatment equal to resident father absence in 1997.",
            },
            {
                "metric": "absent_rows_with_last_year_lived_with_father",
                "count": int(absent["last_year_lived_with_bio_father_clean"].notna().sum()),
                "share": float(absent["last_year_lived_with_bio_father_clean"].notna().mean()) if len(absent.index) else 0.0,
                "note": "Potential timing variable for approximate exit-age derivation.",
            },
            {
                "metric": "absent_rows_with_derived_age_at_last_coresidence",
                "count": int(absent["age_at_last_coresidence"].notna().sum()),
                "share": float(absent["age_at_last_coresidence"].notna().mean()) if len(absent.index) else 0.0,
                "note": "Rows with a cleaned last-coresidence year that can support coarse exit-age derivation.",
            },
            {
                "metric": "childhood_history_rows_available",
                "count": history_rows,
                "share": 1.0 if history_rows else 0.0,
                "note": "Childhood person-year exposure rows assembled from the 1997 retrospective father-history items.",
            },
            {
                "metric": "history_respondents_available",
                "count": history_respondents,
                "share": 1.0 if history_respondents else 0.0,
                "note": "Distinct respondents represented in the childhood exposure-history scaffold.",
            },
            {
                "metric": "localized_exit_respondents",
                "count": localized_exit_respondents,
                "share": float(localized_exit_respondents / absent["respondent_id"].nunique()) if absent["respondent_id"].nunique() else 0.0,
                "note": "Father-absent respondents whose first absent year can be localized from the last-coresidence item.",
            },
            {
                "metric": "respondents_with_pre_post_childhood_rows",
                "count": pre_post_respondents,
                "share": float(pre_post_respondents / absent["respondent_id"].nunique()) if absent["respondent_id"].nunique() else 0.0,
                "note": "Respondents with at least one observed childhood father-present row and one observed father-absent row.",
            },
            {
                "metric": "panel_rows_available",
                "count": panel_row_count,
                "share": 1.0 if panel_row_count else 0.0,
                "note": "Repeated adult-outcome panel rows currently assembled across the available waves.",
            },
            {
                "metric": "panel_waves_available",
                "count": panel_wave_count,
                "share": 1.0 if panel_wave_count >= 2 else 0.0,
                "note": "Distinct repeated-outcome waves in the current panel scaffold.",
            },
            {
                "metric": "near_term_design_rows_available",
                "count": near_term_design_rows,
                "share": float(near_term_design_rows / panel_row_count) if panel_row_count else 0.0,
                "note": "Localized-exit NLSY97 respondent-year rows in the 1998-2005 near-term design scaffold.",
            },
            {
                "metric": "near_term_design_respondents",
                "count": near_term_design_respondents,
                "share": float(near_term_design_respondents / localized_exit_respondents) if localized_exit_respondents else 0.0,
                "note": "Localized-exit respondents represented in the near-term event-time design scaffold.",
            },
            {
                "metric": "near_term_windows_with_support",
                "count": windows_with_support,
                "share": float(windows_with_support / len(EVENT_TIME_WINDOW_ORDER)),
                "note": f"Event-time windows with at least one observed row in the localized-exit near-term design: {supported_windows}.",
            },
            {
                "metric": "near_term_schooling_rows_available",
                "count": near_term_schooling_rows,
                "share": float(near_term_schooling_rows / near_term_design_rows) if near_term_design_rows else 0.0,
                "note": "Near-term design rows with schooling-month observations.",
            },
            {
                "metric": "near_term_broken_report_work_rows_available",
                "count": near_term_bkrpt_rows,
                "share": float(near_term_bkrpt_rows / near_term_design_rows) if near_term_design_rows else 0.0,
                "note": "Near-term design rows with broken-report work intensity measures.",
            },
            {
                "metric": "near_term_justice_rows_available",
                "count": near_term_justice_rows,
                "share": float(near_term_justice_rows / near_term_design_rows) if near_term_design_rows else 0.0,
                "note": "Near-term design rows with arrest or incarceration observations.",
            },
            {
                "metric": "baseline_present_control_candidates",
                "count": stable_present_control_candidates_n,
                "share": float(stable_present_control_candidates_n / nlsy97["respondent_id"].nunique()) if nlsy97["respondent_id"].nunique() else 0.0,
                "note": f"Control candidates drawn from the strongest available baseline-present childhood-history pool: {control_history_type_used}.",
            },
            {
                "metric": "treated_localized_exit_with_exact_overlap_controls",
                "count": treated_with_exact_control_overlap,
                "share": float(treated_with_exact_control_overlap / localized_exit_respondents) if localized_exit_respondents else 0.0,
                "note": "Localized-exit treated respondents whose exact birth-year/sex/race stratum also contains a baseline-present control.",
            },
            {
                "metric": "exact_overlap_strata",
                "count": exact_overlap_strata,
                "share": float(exact_overlap_strata / len(comparison_support.index)) if len(comparison_support.index) else 0.0,
                "note": "Exact birth-year/sex/race strata containing both treated localized exits and baseline-present controls.",
            },
            {
                "metric": "post_only_comparison_design_rows",
                "count": post_only_design_rows,
                "share": float(post_only_design_rows / (post_only_treated_rows + post_only_control_rows)) if (post_only_treated_rows + post_only_control_rows) else 0.0,
                "note": "Rows retained in the exact-overlap post-only treated-versus-control comparison design.",
            },
            {
                "metric": "post_only_treated_rows",
                "count": post_only_treated_rows,
                "share": float(post_only_treated_rows / post_only_design_rows) if post_only_design_rows else 0.0,
                "note": "Treated localized-exit rows retained in the post-only comparison design.",
            },
            {
                "metric": "post_only_control_rows",
                "count": post_only_control_rows,
                "share": float(post_only_control_rows / post_only_design_rows) if post_only_design_rows else 0.0,
                "note": "Baseline-present control rows retained in the post-only comparison design.",
            },
            {
                "metric": "post_only_sensitivity_specs",
                "count": post_only_sensitivity_specs,
                "share": float(post_only_sensitivity_specs / (len(CONTROL_HISTORY_PRIORITY) * len(SENSITIVITY_ANCHOR_RULES))),
                "note": "Control-pool x anchor-rule sensitivity specifications evaluated for the post-only design.",
            },
            {
                "metric": "post_only_preferred_supported_outcomes",
                "count": preferred_supported_outcomes,
                "share": float(preferred_supported_outcomes / len(post_only_outcomes)) if len(post_only_outcomes) else 0.0,
                "note": f"Preferred first-pass headline outcomes supported in `{PRIMARY_POST_ONLY_WINDOW}` under `{PRIMARY_POST_ONLY_CONTROL_HISTORY}` and `{PRIMARY_POST_ONLY_ANCHOR_RULE}`.",
            },
            {
                "metric": "event_time_model_ready",
                "count": 0,
                "share": 0.0,
                "note": "Still not ready for a causal event-study estimator: localized-exit near-term windows now exist, but the scaffold still lacks a defensible comparison design and richer repeated outcomes around each childhood exit.",
            },
        ]
    )
    event_time_path = output_dir / "nlsy97_event_time_readiness.csv"
    event_time.to_csv(event_time_path, index=False)

    readiness = pd.DataFrame(
        [
            {
                "design": "cnlsy_sibling_fe",
                "rows": int(len(sibling_design.index)),
                "groups": int(sibling_design["mother_id"].nunique()),
                "groups_with_2plus": int((family_sizes >= 2).sum()),
                "groups_with_treatment_variation": int((treatment_var >= 2).sum()),
                "status": "not_ready" if int((treatment_var >= 2).sum()) < MIN_DISCORDANT_FAMILIES else "ready",
                "note": "Current CNLSY restricted subset has too little within-mother treatment variation for a defensible sibling FE estimate.",
            },
            {
                "design": "nlsy97_event_time",
                "rows": int(len(absent.index)),
                "groups": int(nlsy97["respondent_id"].nunique()),
                "groups_with_2plus": near_term_design_respondents,
                "groups_with_treatment_variation": localized_exit_respondents,
                "status": "not_ready",
                "note": "A childhood exposure history, dense 1998-2005 near-term design scaffold, and multiwave adult panel now exist, but the current public-use setup still does not support a defensible causal event-time estimator.",
            },
        ]
    )
    readiness_path = output_dir / "quasi_causal_readiness.csv"
    readiness.to_csv(readiness_path, index=False)

    summary_path = output_dir / "quasi_causal_summary.md"
    summary_lines = [
        "# Quasi-Causal Scaffold Summary",
        "",
        "## CNLSY sibling fixed effects",
        "",
        f"- Restricted sibling-design rows: {int(len(sibling_design.index))}",
        f"- Mothers in design: {int(sibling_design['mother_id'].nunique())}",
        f"- Mothers with 2+ children in design: {int((family_sizes >= 2).sum())}",
        f"- Mothers with treatment variation: {int((treatment_var >= 2).sum())}",
        "",
        f"Status: `{readiness.loc[readiness['design'] == 'cnlsy_sibling_fe', 'status'].iloc[0]}`",
        "",
        "## NLSY97 event-time",
        "",
        f"- Father-absent rows: {int(len(absent.index))}",
        f"- Rows with derived age-at-last-coresidence candidate: {int(absent['age_at_last_coresidence'].notna().sum())}",
        f"- Childhood history rows assembled: {history_rows}",
        f"- Respondents with localized first-absence year: {localized_exit_respondents}",
        f"- Respondents with observed pre/post childhood rows: {pre_post_respondents}",
        f"- Panel waves assembled: {panel_wave_count}",
        f"- Panel rows assembled: {panel_row_count}",
        f"- Near-term design rows assembled: {near_term_design_rows}",
        f"- Near-term design respondents: {near_term_design_respondents}",
        f"- Near-term windows with support: {supported_windows}",
        f"- Near-term schooling rows: {near_term_schooling_rows}",
        f"- Near-term broken-report work rows: {near_term_bkrpt_rows}",
        f"- Near-term justice rows: {near_term_justice_rows}",
        f"- Baseline-present control candidates: {stable_present_control_candidates_n} ({control_history_type_used})",
        f"- Treated localized exits with exact-overlap controls: {treated_with_exact_control_overlap}",
        f"- Exact-overlap strata: {exact_overlap_strata}",
        f"- Post-only comparison design rows: {post_only_design_rows}",
        f"- Post-only treated rows: {post_only_treated_rows}",
        f"- Post-only control rows: {post_only_control_rows}",
        f"- Post-only sensitivity specs evaluated: {post_only_sensitivity_specs}",
        f"- Preferred supported headline outcomes: {preferred_supported_outcomes}",
        "",
        "Status: `not_ready`",
        "",
        f"Interpretation: the quasi-causal layer is now explicit and reproducible. A childhood exposure history, a localized-exit 1998-2005 near-term design scaffold, a baseline-present comparison pool, a post-only treated-versus-control comparison design, a control-pool/anchor sensitivity layer, a locked first-pass headline spec (`{PRIMARY_POST_ONLY_CONTROL_HISTORY}` x `{PRIMARY_POST_ONLY_ANCHOR_RULE}` x `{PRIMARY_POST_ONLY_WINDOW}`), and a multiwave NLSY97 outcome panel now exist, but the current public-use scaffold still does not support a credible causal event-time design because the treated side is post-event only and comparison identification still needs stronger assumptions.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return QuasiCausalBuildResult(
        sibling_design_path=sibling_design_path,
        sibling_fe_path=sibling_fe_path,
        event_time_path=event_time_path,
        event_time_design_path=event_time_design_path,
        event_time_window_summary_path=event_time_window_summary_path,
        event_time_comparison_candidates_path=event_time_comparison_candidates_path,
        event_time_comparison_support_path=event_time_comparison_support_path,
        event_time_strategy_path=comparison_strategy_path,
        event_time_post_only_design_path=event_time_post_only_design_path,
        event_time_post_only_summary_path=event_time_post_only_summary_path,
        event_time_post_only_robustness_path=event_time_post_only_robustness_path,
        event_time_post_only_sensitivity_path=event_time_post_only_sensitivity_path,
        event_time_post_only_sensitivity_report_path=event_time_post_only_sensitivity_report_path,
        event_time_post_only_preferred_summary_path=event_time_post_only_preferred_summary_path,
        event_time_post_only_preferred_report_path=event_time_post_only_preferred_report_path,
        event_time_post_only_report_path=event_time_post_only_report_path,
        readiness_path=readiness_path,
        summary_path=summary_path,
    )
