from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from father_longrun.pipelines.contracts import (
    build_canonical_results_payload,
    relative_to_root,
    validate_canonical_results_payload,
)


@dataclass(frozen=True)
class ResultsAppendixResult:
    manifest_path: Path
    results_json_path: Path
    handoff_path: Path
    synthesis_path: Path
    nlsy_prevalence_table_path: Path
    nlsy_predictor_table_path: Path
    nlsy_outcome_gap_table_path: Path
    nlsy_race_gap_table_path: Path
    nlsy_cognitive_table_path: Path
    nlsy_cognitive_subgroup_table_path: Path
    nlsy_near_term_effects_table_path: Path
    nlsy_near_term_robustness_table_path: Path
    nlsy_health_table_path: Path
    nlsy_mental_health_table_path: Path
    nlsy_family_formation_table_path: Path
    nlsy_occupation_summary_table_path: Path
    nlsy_occupation_effect_table_path: Path
    nlsy_effect_heterogeneity_table_path: Path
    nlsy_group_residual_table_path: Path
    acs_child_context_table_path: Path
    benchmark_context_table_path: Path


SPECIAL_MISSING_CODES = {-1, -2, -3, -4, -5}
NLSY97_G_PROXY_PAIR_SPECS = [
    ("GS", "R9705200", "R9706400"),
    ("AR", "R9705300", "R9706500"),
    ("WK", "R9705400", "R9706600"),
    ("PC", "R9705500", "R9706700"),
    ("NO", "R9705600", "R9706800"),
    ("CS", "R9705700", "R9706900"),
    ("AUTO", "R9705800", "R9707000"),
    ("SHOP", "R9705900", "R9707100"),
    ("MK", "R9706000", "R9707200"),
    ("MC", "R9706100", "R9707300"),
    ("EI", "R9706200", "R9707400"),
]
NLSY97_G_PROXY_SUBTESTS = ["GS", "AR", "WK", "PC", "NO", "CS", "AS", "MK", "MC", "EI"]
NLSY97_OCCUPATION_WAVES = (2021, 2019, 2017, 2015, 2013, 2011)
NLSY97_OCCUPATION_MAJOR_GROUPS = (
    ("management_professional_related", "Management/professional/related", 10, 3540),
    ("service", "Service", 3600, 4650),
    ("sales_office", "Sales/office", 4700, 4965),
    ("farming_fishing_forestry", "Farming/fishing/forestry", 5000, 5940),
    ("construction_maintenance", "Construction/extraction/maintenance", 6000, 7630),
    ("production_transport", "Production/transport/material moving", 7700, 9750),
)


def _required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required reporting input does not exist: {path}")
    return pd.read_csv(path)


def _safe_percent(value: float | None) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value) * 100, 2)


def _standardize_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(skipna=True, ddof=1)
    if pd.isna(sd) or float(sd) <= 0.0:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    return (values - values.mean(skipna=True)) / sd


def _clean_special_missing(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.isin(SPECIAL_MISSING_CODES))


def _residualize_quadratic(y: pd.Series, x: pd.Series) -> pd.Series:
    work = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "x": pd.to_numeric(x, errors="coerce")}).dropna()
    residuals = pd.Series(np.nan, index=y.index, dtype="float64")
    if work.empty:
        return residuals
    xx = work["x"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(work)), xx, xx**2])
    target = work["y"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals.loc[work.index] = target - (design @ beta)
    return residuals


def _sex_label(series: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out = out.mask(pd.to_numeric(series, errors="coerce") == 1, "male")
    out = out.mask(pd.to_numeric(series, errors="coerce") == 2, "female")
    return out


def _ols_hc1_treatment_effect(
    data: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    numeric_covariates: list[str],
    categorical_covariates: list[str],
) -> tuple[float | None, float | None, int]:
    work = data[[outcome, treatment, *numeric_covariates, *categorical_covariates]].copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work[treatment] = pd.to_numeric(work[treatment], errors="coerce")
    for column in numeric_covariates:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[outcome, treatment, *numeric_covariates, *categorical_covariates])
    if work.empty:
        return None, None, 0

    design = pd.DataFrame(index=work.index)
    design["intercept"] = 1.0
    design[treatment] = work[treatment].astype(float)
    for column in numeric_covariates:
        design[column] = work[column].astype(float)
    for column in categorical_covariates:
        dummies = pd.get_dummies(work[column].astype("string"), prefix=column, drop_first=True, dtype=float)
        if not dummies.empty:
            design = pd.concat([design, dummies], axis=1)

    design = design.replace([np.inf, -np.inf], np.nan).dropna()
    work = work.loc[design.index]
    if design.empty:
        return None, None, 0

    x = design.to_numpy(dtype=float)
    y = work[outcome].to_numpy(dtype=float)
    n, p = x.shape
    if n <= p:
        return None, None, int(n)

    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - (x @ beta)
    dof = n - p
    if dof <= 0:
        return None, None, int(n)

    meat = x.T @ ((resid[:, None] ** 2) * x)
    hc1_scale = n / dof
    var_beta = hc1_scale * (xtx_inv @ meat @ xtx_inv)
    treatment_idx = int(design.columns.get_loc(treatment))
    coef = float(beta[treatment_idx])
    se = float(np.sqrt(max(var_beta[treatment_idx, treatment_idx], 0.0)))
    return coef, se, int(n)


def _build_nlsy97_g_proxy(project_root: Path) -> pd.DataFrame:
    panel_extract_path = project_root / "data" / "interim" / "nlsy_refresh" / "nlsy97" / "panel_extract.csv"
    if not panel_extract_path.exists():
        raise FileNotFoundError(f"Required NLSY97 raw panel extract does not exist: {panel_extract_path}")

    usecols = ["R0000100", "R0536402"] + [col for _, pos_col, neg_col in NLSY97_G_PROXY_PAIR_SPECS for col in (pos_col, neg_col)]
    raw = pd.read_csv(panel_extract_path, usecols=usecols, low_memory=False).rename(
        columns={"R0000100": "respondent_id", "R0536402": "birth_year"}
    )

    for output, pos_col, neg_col in NLSY97_G_PROXY_PAIR_SPECS:
        pos_raw = pd.to_numeric(raw[pos_col], errors="coerce")
        neg_raw = pd.to_numeric(raw[neg_col], errors="coerce")
        pos_valid = pos_raw.notna() & ~pos_raw.isin(SPECIAL_MISSING_CODES)
        neg_valid = neg_raw.notna() & ~neg_raw.isin(SPECIAL_MISSING_CODES)
        neg_signed = neg_raw.where(neg_raw <= 0.0, -neg_raw)
        raw[output] = np.where(pos_valid, pos_raw, np.where(neg_valid, neg_signed, np.nan))

    auto_z = _standardize_series(raw["AUTO"])
    shop_z = _standardize_series(raw["SHOP"])
    raw["AS"] = pd.concat([auto_z, shop_z], axis=1).mean(axis=1, skipna=True)
    raw["observed_subtests"] = raw[NLSY97_G_PROXY_SUBTESTS].notna().sum(axis=1)
    work = raw.loc[raw["observed_subtests"] >= len(NLSY97_G_PROXY_SUBTESTS)].copy()

    for subtest in NLSY97_G_PROXY_SUBTESTS:
        work[subtest] = _standardize_series(_residualize_quadratic(work[subtest], work["birth_year"]))

    work["g_proxy_1997"] = work[NLSY97_G_PROXY_SUBTESTS].mean(axis=1, skipna=False)
    return work[["respondent_id", "birth_year", "observed_subtests", "g_proxy_1997"]]


def _build_cognitive_and_milestone_tables(project_root: Path, tables_root: Path) -> tuple[Path, Path]:
    analysis_path = project_root / "data" / "processed" / "nlsy" / "nlsy97_analysis_ready.parquet"
    panel_path = project_root / "data" / "processed" / "nlsy" / "nlsy97_longitudinal_outcome_panel.parquet"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Required NLSY97 analysis-ready parquet does not exist: {analysis_path}")
    if not panel_path.exists():
        raise FileNotFoundError(f"Required NLSY97 longitudinal panel parquet does not exist: {panel_path}")

    g_proxy = _build_nlsy97_g_proxy(project_root)
    analysis = pd.read_parquet(
        analysis_path,
        columns=["respondent_id", "resident_bio_father_absent_1997", "sex_raw", "race_ethnicity_3cat", "parent_education"],
    )
    analysis["resident_bio_father_absent_1997"] = pd.to_numeric(
        analysis["resident_bio_father_absent_1997"], errors="coerce"
    )
    analysis["parent_education_clean"] = _clean_special_missing(analysis["parent_education"])
    analysis["sex"] = _sex_label(analysis["sex_raw"])

    panel = pd.read_parquet(
        panel_path,
        columns=[
            "respondent_id",
            "panel_year",
            "education_years_snapshot",
            "sat_math_bin",
            "sat_verbal_bin",
            "act_bin",
        ],
    )
    milestones_2007 = panel.loc[panel["panel_year"] == 2007, ["respondent_id", "education_years_snapshot", "sat_math_bin", "sat_verbal_bin", "act_bin"]]

    merged = analysis.merge(g_proxy, on="respondent_id", how="left").merge(milestones_2007, on="respondent_id", how="left")

    outcome_specs = [
        ("cognition", "g_proxy_1997", "NLSY97 g_proxy (1997 CAT-ASVAB composite)", 1997, "sd_units"),
        ("education", "education_years_snapshot", "Education years snapshot", 2007, "years"),
        ("education", "sat_math_bin", "SAT math bin", 2007, "bin"),
        ("education", "sat_verbal_bin", "SAT verbal bin", 2007, "bin"),
        ("education", "act_bin", "ACT bin", 2007, "bin"),
    ]
    main_rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []

    for outcome_family, outcome, outcome_label, outcome_year, scale in outcome_specs:
        work = merged.dropna(subset=[outcome, "resident_bio_father_absent_1997"]).copy()
        if work.empty:
            continue
        present = work.loc[work["resident_bio_father_absent_1997"] == 0.0, outcome]
        absent = work.loc[work["resident_bio_father_absent_1997"] == 1.0, outcome]
        coef, se, n_used = _ols_hc1_treatment_effect(
            work,
            outcome=outcome,
            treatment="resident_bio_father_absent_1997",
            numeric_covariates=["birth_year", "parent_education_clean"],
            categorical_covariates=["sex", "race_ethnicity_3cat"],
        )
        main_rows.append(
            {
                "outcome_family": outcome_family,
                "outcome": outcome,
                "outcome_label": outcome_label,
                "outcome_year": outcome_year,
                "scale": scale,
                "row_count": int(len(work)),
                "father_present_n": int(present.notna().sum()),
                "father_absent_n": int(absent.notna().sum()),
                "father_present_mean": round(float(present.mean()), 6),
                "father_absent_mean": round(float(absent.mean()), 6),
                "absent_minus_present_gap": round(float(absent.mean() - present.mean()), 6),
                "adjusted_absent_coef": None if coef is None else round(float(coef), 6),
                "adjusted_absent_se_hc1": None if se is None else round(float(se), 6),
                "adjusted_n_used": int(n_used),
                "interpretation_tier": "descriptive_cross_sectional",
            }
        )

        for group_type in ("sex", "race_ethnicity_3cat"):
            for group_value, group_df in work.groupby(group_type):
                present_group = group_df.loc[group_df["resident_bio_father_absent_1997"] == 0.0, outcome]
                absent_group = group_df.loc[group_df["resident_bio_father_absent_1997"] == 1.0, outcome]
                subgroup_rows.append(
                    {
                        "group_type": group_type,
                        "group_value": str(group_value),
                        "outcome": outcome,
                        "outcome_label": outcome_label,
                        "outcome_year": outcome_year,
                        "row_count": int(len(group_df)),
                        "father_present_n": int(present_group.notna().sum()),
                        "father_absent_n": int(absent_group.notna().sum()),
                        "father_present_mean": round(float(present_group.mean()), 6) if present_group.notna().any() else None,
                        "father_absent_mean": round(float(absent_group.mean()), 6) if absent_group.notna().any() else None,
                        "absent_minus_present_gap": (
                            round(float(absent_group.mean() - present_group.mean()), 6)
                            if present_group.notna().any() and absent_group.notna().any()
                            else None
                        ),
                    }
                )

    cognitive_table_path = tables_root / "table_nlsy97_cognitive_education_milestones.csv"
    pd.DataFrame(main_rows).to_csv(cognitive_table_path, index=False)
    cognitive_subgroup_table_path = tables_root / "table_nlsy97_cognitive_education_subgroups.csv"
    pd.DataFrame(subgroup_rows).to_csv(cognitive_subgroup_table_path, index=False)
    return cognitive_table_path, cognitive_subgroup_table_path


def _build_near_term_effect_tables(outputs_root: Path, tables_root: Path) -> tuple[Path, Path]:
    models_root = outputs_root / "models"
    preferred = _required_csv(models_root / "nlsy97_event_time_post_only_preferred_summary.csv")
    robustness = _required_csv(models_root / "nlsy97_event_time_post_only_robustness.csv")

    family_map = {
        "schooling_engagement_months": ("education", "Schooling engagement months"),
        "k12_enrolled_months": ("education", "K-12 enrolled months"),
        "college_enrolled_months": ("education", "College enrolled months"),
        "arrest_any": ("crime", "Any arrest"),
        "incarceration_any": ("crime", "Any incarceration"),
        "bkrpt_weeks": ("work", "Broken-report weeks"),
        "bkrpt_hours": ("work", "Broken-report hours"),
    }

    preferred = preferred.loc[preferred["headline_interpretation"] == "headline_supported"].copy()
    preferred["outcome_family"] = preferred["outcome"].map(lambda x: family_map.get(x, ("other", str(x)))[0])
    preferred["outcome_label"] = preferred["outcome"].map(lambda x: family_map.get(x, ("other", str(x)))[1])
    near_term_effects_table_path = tables_root / "table_nlsy97_near_term_education_crime_work_effects.csv"
    preferred.to_csv(near_term_effects_table_path, index=False)

    robustness = robustness.loc[
        (robustness["event_time_window"] == "post_3plus")
        & (robustness["outcome"].isin(family_map))
    ].copy()
    robustness["outcome_family"] = robustness["outcome"].map(lambda x: family_map.get(x, ("other", str(x)))[0])
    robustness["outcome_label"] = robustness["outcome"].map(lambda x: family_map.get(x, ("other", str(x)))[1])
    near_term_robustness_table_path = tables_root / "table_nlsy97_near_term_effects_robustness.csv"
    robustness.to_csv(near_term_robustness_table_path, index=False)
    return near_term_effects_table_path, near_term_robustness_table_path


def _education_band(series: pd.Series) -> pd.Series:
    cleaned = _clean_special_missing(series)
    result = pd.Series("missing", index=series.index, dtype="string")
    observed = cleaned.dropna()
    if observed.empty:
        return result
    if observed.nunique() >= 4:
        ranked = observed.rank(method="first")
        labels = ["q1_low", "q2", "q3", "q4_high"]
        bands = pd.qcut(ranked, 4, labels=labels)
        result.loc[bands.index] = bands.astype("string")
        return result
    median = observed.median()
    result.loc[cleaned.notna() & cleaned.le(median)] = "lower_or_equal_median"
    result.loc[cleaned.notna() & cleaned.gt(median)] = "above_median"
    return result


def _scaled_occupation_code(series: pd.Series) -> pd.Series:
    codes = pd.to_numeric(series, errors="coerce")
    return codes.where(codes >= 1000, codes * 10.0)


def _assign_latest_occupation_group(series: pd.Series) -> pd.Series:
    scaled = _scaled_occupation_code(series)
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    for key, _label, low, high in NLSY97_OCCUPATION_MAJOR_GROUPS:
        mask = scaled.ge(float(low)) & scaled.le(float(high))
        out = out.mask(mask, key)
    return out


def _select_latest_occupation(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["occupation_code_latest"] = np.nan
    out["occupation_age_latest"] = np.nan
    out["occupation_source_wave"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for year in NLSY97_OCCUPATION_WAVES:
        occ_col = f"occupation_code_{year}"
        age_col = f"age_at_wave_{year}"
        if occ_col not in frame.columns:
            continue
        occ = pd.to_numeric(frame[occ_col], errors="coerce")
        age = pd.to_numeric(frame[age_col], errors="coerce") if age_col in frame.columns else np.nan
        mask = out["occupation_code_latest"].isna() & occ.notna()
        out.loc[mask, "occupation_code_latest"] = occ.loc[mask]
        if isinstance(age, pd.Series):
            out.loc[mask, "occupation_age_latest"] = age.loc[mask]
        out.loc[mask, "occupation_source_wave"] = str(year)
    out["occupation_group_latest"] = _assign_latest_occupation_group(out["occupation_code_latest"])
    out["high_skill_occupation_latest"] = out["occupation_group_latest"].eq("management_professional_related").astype("float64")
    out.loc[out["occupation_group_latest"].isna(), "high_skill_occupation_latest"] = np.nan
    return out


def _merge_panel_wave(
    panel: pd.DataFrame,
    *,
    year: int,
    columns: list[str],
) -> pd.DataFrame:
    existing = [column for column in columns if column in panel.columns]
    if not existing:
        return pd.DataFrame(columns=["respondent_id"])
    subset = panel.loc[panel["panel_year"] == year, ["respondent_id", *existing]].copy()
    return subset.rename(columns={column: f"{column}_{year}" for column in existing})


def _build_nlsy97_reporting_frame(project_root: Path) -> pd.DataFrame:
    analysis_path = project_root / "data" / "processed" / "nlsy" / "nlsy97_analysis_ready.parquet"
    panel_path = project_root / "data" / "processed" / "nlsy" / "nlsy97_longitudinal_outcome_panel.parquet"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Required NLSY97 analysis-ready parquet does not exist: {analysis_path}")
    if not panel_path.exists():
        raise FileNotFoundError(f"Required NLSY97 longitudinal panel parquet does not exist: {panel_path}")

    analysis = pd.read_parquet(
        analysis_path,
        columns=[
            "respondent_id",
            "resident_bio_father_absent_1997",
            "sex_raw",
            "birth_year",
            "race_ethnicity_3cat",
            "mother_education",
            "father_education",
            "parent_education",
        ],
    ).copy()
    analysis["resident_bio_father_absent_1997"] = pd.to_numeric(
        analysis["resident_bio_father_absent_1997"], errors="coerce"
    )
    analysis["sex"] = _sex_label(analysis["sex_raw"])
    analysis["parent_education_clean"] = _clean_special_missing(analysis["parent_education"])
    analysis["mother_education_clean"] = _clean_special_missing(analysis["mother_education"])
    analysis["father_education_clean"] = _clean_special_missing(analysis["father_education"])
    analysis["parent_education_band"] = _education_band(analysis["parent_education"])

    g_proxy = _build_nlsy97_g_proxy(project_root)[["respondent_id", "g_proxy_1997"]]
    frame = analysis.merge(g_proxy, on="respondent_id", how="left")

    panel = pd.read_parquet(
        panel_path,
        columns=[
            "respondent_id",
            "panel_year",
            "age_at_wave",
            "occupation_code",
            "education_years_snapshot",
            "sat_math_bin",
            "sat_verbal_bin",
            "act_bin",
            "annual_earnings",
            "household_income",
            "govt_program_income",
            "health_status",
            "smoking_days_30",
            "alcohol_days_30",
            "binge_days_30",
            "marijuana_days_30",
            "bmi",
            "cesd_score",
            "first_marriage_year",
            "first_marriage_end",
            "total_bio_children",
            "total_marriages",
            "marital_status_collapsed",
            "household_type_40",
        ],
    )

    for year, columns in {
        2007: ["education_years_snapshot", "sat_math_bin", "sat_verbal_bin", "act_bin", "age_at_wave"],
        2011: ["bmi", "occupation_code", "age_at_wave"],
        2015: ["marijuana_days_30", "occupation_code", "age_at_wave"],
        2017: ["occupation_code", "age_at_wave"],
        2019: ["occupation_code", "annual_earnings", "household_income", "govt_program_income", "age_at_wave"],
        2021: ["occupation_code", "annual_earnings", "household_income", "first_marriage_year", "first_marriage_end", "total_bio_children", "total_marriages", "marital_status_collapsed", "household_type_40", "age_at_wave"],
        2023: ["health_status", "smoking_days_30", "alcohol_days_30", "binge_days_30", "cesd_score", "age_at_wave"],
    }.items():
        frame = frame.merge(_merge_panel_wave(panel, year=year, columns=columns), on="respondent_id", how="left")

    latest_occ = _select_latest_occupation(frame)
    frame = pd.concat([frame, latest_occ], axis=1)

    frame["bmi_2011_clean"] = pd.to_numeric(frame.get("bmi_2011"), errors="coerce").where(
        pd.to_numeric(frame.get("bmi_2011"), errors="coerce").between(10.0, 80.0)
    )
    frame["poor_health_2023"] = pd.to_numeric(frame.get("health_status_2023"), errors="coerce").ge(4).astype("float64")
    frame.loc[pd.to_numeric(frame.get("health_status_2023"), errors="coerce").isna(), "poor_health_2023"] = np.nan
    frame["any_smoking_2023"] = pd.to_numeric(frame.get("smoking_days_30_2023"), errors="coerce").gt(0).astype("float64")
    frame.loc[pd.to_numeric(frame.get("smoking_days_30_2023"), errors="coerce").isna(), "any_smoking_2023"] = np.nan
    frame["any_binge_2023"] = pd.to_numeric(frame.get("binge_days_30_2023"), errors="coerce").gt(0).astype("float64")
    frame.loc[pd.to_numeric(frame.get("binge_days_30_2023"), errors="coerce").isna(), "any_binge_2023"] = np.nan
    frame["any_marijuana_2015"] = pd.to_numeric(frame.get("marijuana_days_30_2015"), errors="coerce").gt(0).astype("float64")
    frame.loc[pd.to_numeric(frame.get("marijuana_days_30_2015"), errors="coerce").isna(), "any_marijuana_2015"] = np.nan
    frame["obesity_2011"] = pd.to_numeric(frame["bmi_2011_clean"], errors="coerce").ge(30).astype("float64")
    frame.loc[pd.to_numeric(frame["bmi_2011_clean"], errors="coerce").isna(), "obesity_2011"] = np.nan
    frame["elevated_cesd_2023"] = pd.to_numeric(frame.get("cesd_score_2023"), errors="coerce").ge(16).astype("float64")
    frame.loc[pd.to_numeric(frame.get("cesd_score_2023"), errors="coerce").isna(), "elevated_cesd_2023"] = np.nan
    frame["ever_married_2021"] = pd.to_numeric(frame.get("total_marriages_2021"), errors="coerce").gt(0).astype("float64")
    frame.loc[pd.to_numeric(frame.get("total_marriages_2021"), errors="coerce").isna(), "ever_married_2021"] = np.nan
    frame["currently_married_2021"] = pd.to_numeric(frame.get("marital_status_collapsed_2021"), errors="coerce").eq(1).astype("float64")
    frame.loc[pd.to_numeric(frame.get("marital_status_collapsed_2021"), errors="coerce").isna(), "currently_married_2021"] = np.nan
    frame["any_children_2021"] = pd.to_numeric(frame.get("total_bio_children_2021"), errors="coerce").gt(0).astype("float64")
    frame.loc[pd.to_numeric(frame.get("total_bio_children_2021"), errors="coerce").isna(), "any_children_2021"] = np.nan
    first_marriage_year = pd.to_numeric(frame.get("first_marriage_year_2021"), errors="coerce")
    birth_year = pd.to_numeric(frame["birth_year"], errors="coerce")
    frame["age_first_marriage_2021"] = (first_marriage_year - birth_year).where(first_marriage_year.notna() & birth_year.notna())
    frame["govt_program_income_any_2019"] = pd.to_numeric(frame.get("govt_program_income_2019"), errors="coerce")

    return frame


def _effect_summary_rows(
    frame: pd.DataFrame,
    *,
    outcome_specs: list[dict[str, object]],
) -> pd.DataFrame:
    columns = [
        "outcome_family",
        "outcome",
        "outcome_label",
        "outcome_year",
        "scale",
        "row_count",
        "father_present_n",
        "father_absent_n",
        "father_present_mean",
        "father_absent_mean",
        "absent_minus_present_gap",
        "adjusted_absent_coef",
        "adjusted_absent_se_hc1",
        "adjusted_n_used",
        "interpretation_tier",
    ]
    rows: list[dict[str, object]] = []
    for spec in outcome_specs:
        outcome = str(spec["outcome"])
        work = frame.dropna(subset=[outcome, "resident_bio_father_absent_1997"]).copy()
        if work.empty:
            continue
        present = work.loc[work["resident_bio_father_absent_1997"] == 0.0, outcome]
        absent = work.loc[work["resident_bio_father_absent_1997"] == 1.0, outcome]
        coef, se, n_used = _ols_hc1_treatment_effect(
            work,
            outcome=outcome,
            treatment="resident_bio_father_absent_1997",
            numeric_covariates=["birth_year", "parent_education_clean"],
            categorical_covariates=["sex", "race_ethnicity_3cat"],
        )
        rows.append(
            {
                "outcome_family": spec["outcome_family"],
                "outcome": outcome,
                "outcome_label": spec["outcome_label"],
                "outcome_year": spec["outcome_year"],
                "scale": spec["scale"],
                "row_count": int(len(work)),
                "father_present_n": int(present.notna().sum()),
                "father_absent_n": int(absent.notna().sum()),
                "father_present_mean": round(float(present.mean()), 6),
                "father_absent_mean": round(float(absent.mean()), 6),
                "absent_minus_present_gap": round(float(absent.mean() - present.mean()), 6),
                "adjusted_absent_coef": None if coef is None else round(float(coef), 6),
                "adjusted_absent_se_hc1": None if se is None else round(float(se), 6),
                "adjusted_n_used": int(n_used),
                "interpretation_tier": spec.get("interpretation_tier", "descriptive_cross_sectional"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _heterogeneity_rows(
    frame: pd.DataFrame,
    *,
    outcome_specs: list[dict[str, object]],
) -> pd.DataFrame:
    columns = [
        "group_type",
        "group_value",
        "outcome",
        "outcome_label",
        "outcome_family",
        "row_count",
        "father_present_n",
        "father_absent_n",
        "father_present_mean",
        "father_absent_mean",
        "absent_minus_present_gap",
        "adjusted_absent_coef",
        "adjusted_absent_se_hc1",
        "adjusted_n_used",
    ]
    rows: list[dict[str, object]] = []
    for spec in outcome_specs:
        outcome = str(spec["outcome"])
        for group_type in ("sex", "race_ethnicity_3cat", "parent_education_band"):
            if group_type not in frame.columns:
                continue
            for group_value, group_df in frame.groupby(group_type, dropna=False):
                work = group_df.dropna(subset=[outcome, "resident_bio_father_absent_1997"]).copy()
                if work.empty:
                    continue
                present = work.loc[work["resident_bio_father_absent_1997"] == 0.0, outcome]
                absent = work.loc[work["resident_bio_father_absent_1997"] == 1.0, outcome]
                cat_covariates = [col for col in ("sex", "race_ethnicity_3cat") if col in work.columns and col != group_type]
                coef, se, n_used = _ols_hc1_treatment_effect(
                    work,
                    outcome=outcome,
                    treatment="resident_bio_father_absent_1997",
                    numeric_covariates=["birth_year", "parent_education_clean"],
                    categorical_covariates=cat_covariates,
                )
                rows.append(
                    {
                        "group_type": group_type,
                        "group_value": str(group_value),
                        "outcome": outcome,
                        "outcome_label": spec["outcome_label"],
                        "outcome_family": spec["outcome_family"],
                        "row_count": int(len(work)),
                        "father_present_n": int(present.notna().sum()),
                        "father_absent_n": int(absent.notna().sum()),
                        "father_present_mean": round(float(present.mean()), 6),
                        "father_absent_mean": round(float(absent.mean()), 6),
                        "absent_minus_present_gap": round(float(absent.mean() - present.mean()), 6),
                        "adjusted_absent_coef": None if coef is None else round(float(coef), 6),
                        "adjusted_absent_se_hc1": None if se is None else round(float(se), 6),
                        "adjusted_n_used": int(n_used),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _group_residual_rows(
    frame: pd.DataFrame,
    *,
    outcome_specs: list[dict[str, object]],
    min_group_n: int = 20,
) -> pd.DataFrame:
    columns = [
        "outcome",
        "outcome_label",
        "group_type",
        "group_value",
        "n_pooled",
        "n_group",
        "mean_actual",
        "mean_predicted",
        "mean_residual",
        "se_residual",
        "pct_over_under",
        "mean_g_proxy",
        "pooled_beta_g_proxy",
    ]
    rows: list[dict[str, object]] = []
    for spec in outcome_specs:
        outcome = str(spec["outcome"])
        work = frame[[outcome, "g_proxy_1997", "resident_bio_father_absent_1997", "birth_year", "sex", "race_ethnicity_3cat", "parent_education_band"]].copy()
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
        work["g_proxy_1997"] = pd.to_numeric(work["g_proxy_1997"], errors="coerce")
        work["resident_bio_father_absent_1997"] = pd.to_numeric(work["resident_bio_father_absent_1997"], errors="coerce")
        work["birth_year"] = pd.to_numeric(work["birth_year"], errors="coerce")
        work = work.dropna(subset=[outcome, "g_proxy_1997", "resident_bio_father_absent_1997", "birth_year"])
        if len(work) < 30:
            continue
        design = pd.DataFrame(
            {
                "intercept": 1.0,
                "g_proxy_1997": work["g_proxy_1997"],
                "fatherlessness": work["resident_bio_father_absent_1997"],
                "birth_year": work["birth_year"],
            },
            index=work.index,
        )
        x = design.to_numpy(dtype=float)
        y = work[outcome].to_numpy(dtype=float)
        beta = np.linalg.pinv(x.T @ x) @ x.T @ y
        predicted = x @ beta
        work["predicted"] = predicted
        work["residual"] = y - predicted
        for group_type in ("sex", "race_ethnicity_3cat", "parent_education_band"):
            if group_type not in work.columns:
                continue
            for group_value, group_df in work.groupby(group_type, dropna=False):
                if len(group_df) < min_group_n:
                    continue
                mean_predicted = float(group_df["predicted"].mean())
                mean_residual = float(group_df["residual"].mean())
                se_residual = float(group_df["residual"].std(ddof=1) / np.sqrt(len(group_df))) if len(group_df) > 1 else np.nan
                pct = (mean_residual / abs(mean_predicted) * 100.0) if abs(mean_predicted) > 1e-9 else np.nan
                rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": spec["outcome_label"],
                        "group_type": group_type,
                        "group_value": str(group_value),
                        "n_pooled": int(len(work)),
                        "n_group": int(len(group_df)),
                        "mean_actual": round(float(group_df[outcome].mean()), 6),
                        "mean_predicted": round(mean_predicted, 6),
                        "mean_residual": round(mean_residual, 6),
                        "se_residual": round(se_residual, 6) if pd.notna(se_residual) else None,
                        "pct_over_under": round(float(pct), 4) if pd.notna(pct) else None,
                        "mean_g_proxy": round(float(group_df["g_proxy_1997"].mean()), 6),
                        "pooled_beta_g_proxy": round(float(beta[1]), 6),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _build_health_family_occupation_tables(project_root: Path, tables_root: Path) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    frame = _build_nlsy97_reporting_frame(project_root)

    health_specs = [
        {"outcome_family": "health", "outcome": "poor_health_2023", "outcome_label": "Poor self-rated health (2023, 4-5 vs 1-3)", "outcome_year": 2023, "scale": "share"},
        {"outcome_family": "health", "outcome": "health_status_2023", "outcome_label": "Self-rated health status (2023)", "outcome_year": 2023, "scale": "ordered_score"},
        {"outcome_family": "health", "outcome": "obesity_2011", "outcome_label": "Obesity (2011, BMI >= 30)", "outcome_year": 2011, "scale": "share"},
        {"outcome_family": "health", "outcome": "bmi_2011_clean", "outcome_label": "Body mass index (2011)", "outcome_year": 2011, "scale": "bmi"},
        {"outcome_family": "substance", "outcome": "any_smoking_2023", "outcome_label": "Any smoking past 30 days (2023)", "outcome_year": 2023, "scale": "share"},
        {"outcome_family": "substance", "outcome": "smoking_days_30_2023", "outcome_label": "Smoking days past 30 days (2023)", "outcome_year": 2023, "scale": "days"},
        {"outcome_family": "substance", "outcome": "alcohol_days_30_2023", "outcome_label": "Alcohol days past 30 days (2023)", "outcome_year": 2023, "scale": "days"},
        {"outcome_family": "substance", "outcome": "any_binge_2023", "outcome_label": "Any binge drinking past 30 days (2023)", "outcome_year": 2023, "scale": "share"},
        {"outcome_family": "substance", "outcome": "binge_days_30_2023", "outcome_label": "Binge drinking days past 30 days (2023)", "outcome_year": 2023, "scale": "days"},
        {"outcome_family": "substance", "outcome": "any_marijuana_2015", "outcome_label": "Any marijuana use past 30 days (2015)", "outcome_year": 2015, "scale": "share"},
        {"outcome_family": "substance", "outcome": "marijuana_days_30_2015", "outcome_label": "Marijuana days past 30 days (2015)", "outcome_year": 2015, "scale": "days"},
    ]
    mental_health_specs = [
        {"outcome_family": "mental_health", "outcome": "cesd_score_2023", "outcome_label": "CES-D score (2023)", "outcome_year": 2023, "scale": "score"},
        {"outcome_family": "mental_health", "outcome": "elevated_cesd_2023", "outcome_label": "Elevated CES-D score (2023, >=16)", "outcome_year": 2023, "scale": "share"},
    ]
    family_specs = [
        {"outcome_family": "family_formation", "outcome": "ever_married_2021", "outcome_label": "Ever married by 2021", "outcome_year": 2021, "scale": "share"},
        {"outcome_family": "family_formation", "outcome": "currently_married_2021", "outcome_label": "Currently married in 2021", "outcome_year": 2021, "scale": "share"},
        {"outcome_family": "family_formation", "outcome": "any_children_2021", "outcome_label": "Any biological children by 2021", "outcome_year": 2021, "scale": "share"},
        {"outcome_family": "family_formation", "outcome": "total_bio_children_2021", "outcome_label": "Total biological children (2021)", "outcome_year": 2021, "scale": "count"},
        {"outcome_family": "family_formation", "outcome": "total_marriages_2021", "outcome_label": "Total marriages (2021)", "outcome_year": 2021, "scale": "count"},
        {"outcome_family": "family_formation", "outcome": "age_first_marriage_2021", "outcome_label": "Age at first marriage (2021 cumulative)", "outcome_year": 2021, "scale": "years"},
    ]

    health_table = _effect_summary_rows(frame, outcome_specs=health_specs)
    health_table_path = tables_root / "table_nlsy97_health_substance_effects.csv"
    health_table.to_csv(health_table_path, index=False)

    mental_health_table = _effect_summary_rows(frame, outcome_specs=mental_health_specs)
    mental_health_table_path = tables_root / "table_nlsy97_mental_health_effects.csv"
    mental_health_table.to_csv(mental_health_table_path, index=False)

    family_table = _effect_summary_rows(frame, outcome_specs=family_specs)
    family_table_path = tables_root / "table_nlsy97_family_formation_effects.csv"
    family_table.to_csv(family_table_path, index=False)

    occupation = frame.loc[frame["occupation_group_latest"].notna()].copy()
    occ_rows: list[dict[str, object]] = []
    for group_key, group_label, _low, _high in NLSY97_OCCUPATION_MAJOR_GROUPS:
        for status_value, status_label in ((0.0, "father_present"), (1.0, "father_absent")):
            subset = occupation.loc[
                (occupation["occupation_group_latest"] == group_key)
                & (occupation["resident_bio_father_absent_1997"] == status_value)
            ].copy()
            occ_rows.append(
                {
                    "occupation_group": group_key,
                    "occupation_group_label": group_label,
                    "fatherlessness_status": status_label,
                    "row_count": int(len(subset)),
                    "share_within_latest_occupation_sample": round(float(len(subset) / len(occupation)), 6) if len(occupation) else None,
                    "mean_g_proxy": round(float(pd.to_numeric(subset["g_proxy_1997"], errors="coerce").mean()), 6) if not subset.empty else None,
                    "mean_annual_earnings_2021": round(float(pd.to_numeric(subset["annual_earnings_2021"], errors="coerce").mean()), 6) if not subset.empty else None,
                    "top_source_wave": subset["occupation_source_wave"].mode().iloc[0] if not subset.empty and subset["occupation_source_wave"].notna().any() else None,
                }
            )
    occupation_summary_path = tables_root / "table_nlsy97_occupation_group_summary.csv"
    pd.DataFrame(occ_rows).to_csv(occupation_summary_path, index=False)

    occupation_effect_spec = [
        {"outcome_family": "occupation", "outcome": "high_skill_occupation_latest", "outcome_label": "Latest adult occupation in management/professional/related", "outcome_year": 2021, "scale": "share"},
    ]
    occupation_effect_path = tables_root / "table_nlsy97_occupation_high_skill_effect.csv"
    _effect_summary_rows(frame, outcome_specs=occupation_effect_spec).to_csv(occupation_effect_path, index=False)

    heterogeneity_specs = [
        {"outcome_family": "cognition", "outcome": "g_proxy_1997", "outcome_label": "NLSY97 g_proxy (1997 CAT-ASVAB composite)"},
        {"outcome_family": "education", "outcome": "education_years_snapshot_2007", "outcome_label": "Education years snapshot (2007)"},
        {"outcome_family": "economics", "outcome": "annual_earnings_2021", "outcome_label": "Annual earnings (2021)"},
        {"outcome_family": "health", "outcome": "poor_health_2023", "outcome_label": "Poor self-rated health (2023)"},
        {"outcome_family": "mental_health", "outcome": "cesd_score_2023", "outcome_label": "CES-D score (2023)"},
        {"outcome_family": "family_formation", "outcome": "ever_married_2021", "outcome_label": "Ever married by 2021"},
        {"outcome_family": "occupation", "outcome": "high_skill_occupation_latest", "outcome_label": "Latest high-skill occupation"},
    ]
    heterogeneity_path = tables_root / "table_nlsy97_fatherlessness_effect_heterogeneity.csv"
    _heterogeneity_rows(frame, outcome_specs=heterogeneity_specs).to_csv(heterogeneity_path, index=False)

    residual_specs = [
        {"outcome": "annual_earnings_2021", "outcome_label": "Annual earnings (2021)"},
        {"outcome": "household_income_2021", "outcome_label": "Household income (2021)"},
        {"outcome": "education_years_snapshot_2007", "outcome_label": "Education years snapshot (2007)"},
        {"outcome": "cesd_score_2023", "outcome_label": "CES-D score (2023)"},
    ]
    residual_path = tables_root / "table_nlsy97_group_residual_gaps.csv"
    _group_residual_rows(frame, outcome_specs=residual_specs).to_csv(residual_path, index=False)

    return (
        health_table_path,
        mental_health_table_path,
        family_table_path,
        occupation_summary_path,
        occupation_effect_path,
        heterogeneity_path,
        residual_path,
    )


def build_results_appendix(*, outputs_root: Path, project_root: Path | None = None) -> ResultsAppendixResult:
    project_root = outputs_root.parent if project_root is None else project_root
    manifests_root = outputs_root / "manifests"
    tables_root = outputs_root / "tables"
    manifests_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    fatherlessness = _required_csv(manifests_root / "nlsy97_fatherlessness_group_summary.csv")
    predictors = _required_csv(manifests_root / "nlsy97_fatherlessness_predictors.csv")
    cross_summary = _required_csv(manifests_root / "cross_cohort_benchmark_summary.csv")
    cross_subgroup = _required_csv(manifests_root / "cross_cohort_benchmark_subgroup_summary.csv")
    acs_child = _required_csv(manifests_root / "acs_child_father_presence_summary.csv")
    public_summary = _required_csv(manifests_root / "public_benchmark_profile_summary.csv")
    nlsy_cognitive_table_path, nlsy_cognitive_subgroup_table_path = _build_cognitive_and_milestone_tables(
        project_root,
        tables_root,
    )
    nlsy_near_term_effects_table_path, nlsy_near_term_robustness_table_path = _build_near_term_effect_tables(
        outputs_root,
        tables_root,
    )
    (
        nlsy_health_table_path,
        nlsy_mental_health_table_path,
        nlsy_family_formation_table_path,
        nlsy_occupation_summary_table_path,
        nlsy_occupation_effect_table_path,
        nlsy_effect_heterogeneity_table_path,
        nlsy_group_residual_table_path,
    ) = _build_health_family_occupation_tables(project_root, tables_root)

    nlsy_prevalence = fatherlessness.loc[
        fatherlessness["group_type"].isin(
            ["overall", "sex", "race_ethnicity_3cat", "parent_education_band", "mother_education_band", "father_education_band"]
        )
    ].copy()
    nlsy_prevalence["fatherlessness_pct"] = nlsy_prevalence["fatherlessness_rate"].map(_safe_percent)
    nlsy_prevalence["father_present_pct"] = nlsy_prevalence["father_present_rate"].map(_safe_percent)
    nlsy_prevalence_table_path = tables_root / "table_nlsy97_fatherlessness_prevalence.csv"
    nlsy_prevalence.to_csv(nlsy_prevalence_table_path, index=False)

    nlsy_predictors = predictors.loc[predictors["term"] != "const"].copy()
    nlsy_predictors["odds_ratio"] = nlsy_predictors["odds_ratio"].round(3)
    nlsy_predictors["coefficient"] = nlsy_predictors["coefficient"].round(4)
    nlsy_predictors["std_error"] = nlsy_predictors["std_error"].round(4)
    nlsy_predictor_table_path = tables_root / "table_nlsy97_fatherlessness_predictors.csv"
    nlsy_predictors.to_csv(nlsy_predictor_table_path, index=False)

    nlsy_outcome_gap = cross_summary.loc[
        cross_summary["source_group"].isin(
            [
                "overall",
                "resident_bio_father_present",
                "resident_bio_father_absent",
                "acs_pums_2024_context",
                "cps_asec_2023_2025_pooled",
                "sipp_2023_monthly_context",
            ]
        )
    ].copy()
    nlsy_outcome_gap_table_path = tables_root / "table_nlsy97_outcome_gaps_vs_public_context.csv"
    nlsy_outcome_gap.to_csv(nlsy_outcome_gap_table_path, index=False)

    nlsy_race_gap = cross_subgroup.loc[
        (cross_subgroup["source"] == "nlsy97")
        & (
            cross_subgroup["source_group"].isin(
                ["overall", "resident_bio_father_present", "resident_bio_father_absent"]
            )
        )
    ].copy()
    nlsy_race_gap_table_path = tables_root / "table_nlsy97_race_sex_outcome_gaps.csv"
    nlsy_race_gap.to_csv(nlsy_race_gap_table_path, index=False)

    acs_child_context = acs_child.loc[
        acs_child["group_type"].isin(["overall", "race_ethnicity_3cat", "poverty_band", "household_income_band"])
    ].copy()
    acs_child_context["father_absent_pct"] = acs_child_context["father_absent_share"].map(_safe_percent)
    acs_child_context["father_present_pct"] = acs_child_context["father_present_share"].map(_safe_percent)
    acs_child_context_table_path = tables_root / "table_acs_child_father_presence_context.csv"
    acs_child_context.to_csv(acs_child_context_table_path, index=False)

    benchmark_context = public_summary.copy()
    benchmark_context["weighted_employment_pct"] = benchmark_context["weighted_employment_share"].map(_safe_percent)
    benchmark_context["weighted_poverty_pct"] = benchmark_context["weighted_poverty_share"].map(_safe_percent)
    benchmark_context_table_path = tables_root / "table_public_benchmark_context.csv"
    benchmark_context.to_csv(benchmark_context_table_path, index=False)

    overall_rate = float(
        fatherlessness.loc[fatherlessness["group_type"] == "overall", "fatherlessness_rate"].iloc[0]
    )
    race_rows = fatherlessness.loc[fatherlessness["group_type"] == "race_ethnicity_3cat"].sort_values(
        "fatherlessness_rate", ascending=False
    )
    top_race = race_rows.iloc[0]
    low_race = race_rows.iloc[-1]
    predictor_rows = nlsy_predictors.sort_values("p_value", na_position="last").head(3)
    overall_cross = cross_summary.loc[cross_summary["source_group"] == "overall"].iloc[0]
    present_cross = cross_summary.loc[cross_summary["source_group"] == "resident_bio_father_present"].iloc[0]
    absent_cross = cross_summary.loc[cross_summary["source_group"] == "resident_bio_father_absent"].iloc[0]
    acs_context = cross_summary.loc[cross_summary["source_group"] == "acs_pums_2024_context"].iloc[0]
    cps_context = cross_summary.loc[cross_summary["source_group"] == "cps_asec_2023_2025_pooled"].iloc[0]
    sipp_context = cross_summary.loc[cross_summary["source_group"] == "sipp_2023_monthly_context"].iloc[0]
    acs_child_overall = acs_child.loc[acs_child["group_type"] == "overall"].iloc[0]
    acs_child_race = acs_child.loc[acs_child["group_type"] == "race_ethnicity_3cat"].sort_values(
        "father_absent_share", ascending=False
    )
    acs_child_poverty = acs_child.loc[acs_child["group_type"] == "poverty_band"].sort_values(
        "father_absent_share", ascending=False
    )
    cognitive = pd.read_csv(nlsy_cognitive_table_path)
    g_proxy_row = cognitive.loc[cognitive["outcome"] == "g_proxy_1997"].iloc[0]
    education_2007_row = cognitive.loc[cognitive["outcome"] == "education_years_snapshot"].iloc[0]
    near_term = pd.read_csv(nlsy_near_term_effects_table_path)
    schooling_row = near_term.loc[near_term["outcome"] == "schooling_engagement_months"].iloc[0]
    arrest_row = near_term.loc[near_term["outcome"] == "arrest_any"].iloc[0]
    health = pd.read_csv(nlsy_health_table_path)
    mental_health = pd.read_csv(nlsy_mental_health_table_path)
    family_formation = pd.read_csv(nlsy_family_formation_table_path)
    occupation_effect = pd.read_csv(nlsy_occupation_effect_table_path)
    heterogeneity = pd.read_csv(nlsy_effect_heterogeneity_table_path)
    residual_gaps = pd.read_csv(nlsy_group_residual_table_path)
    poor_health_row = health.loc[health["outcome"] == "poor_health_2023"].iloc[0]
    cesd_row = mental_health.loc[mental_health["outcome"] == "cesd_score_2023"].iloc[0]
    ever_married_row = family_formation.loc[family_formation["outcome"] == "ever_married_2021"].iloc[0]
    high_skill_row = occupation_effect.loc[occupation_effect["outcome"] == "high_skill_occupation_latest"].iloc[0]
    earnings_heterogeneity = heterogeneity.loc[
        (heterogeneity["outcome"] == "annual_earnings_2021")
        & (heterogeneity["group_type"] == "race_ethnicity_3cat")
    ].sort_values("absent_minus_present_gap")
    largest_earnings_heterogeneity = earnings_heterogeneity.iloc[0] if not earnings_heterogeneity.empty else None
    residual_earnings = residual_gaps.loc[residual_gaps["outcome"] == "annual_earnings_2021"].sort_values(
        "mean_residual"
    )
    largest_negative_residual = residual_earnings.iloc[0] if not residual_earnings.empty else None

    synthesis_lines = [
        "# Results Synthesis",
        "",
        "This memo is the stable interpretation layer for the current public-data build. It is descriptive and ready for frontend/doc drafting.",
        "",
        "## Headline NLSY97 Findings",
        f"- Overall NLSY97 fatherlessness rate: {overall_rate:.4f} ({overall_rate * 100:.2f}%).",
        f"- Highest observed race-specific fatherlessness rate: {top_race['group_value']} at {top_race['fatherlessness_rate']:.4f}.",
        f"- Lowest observed race-specific fatherlessness rate: {low_race['group_value']} at {low_race['fatherlessness_rate']:.4f}.",
        f"- NLSY97 mean 2021 earnings: {overall_cross['mean_earnings']:.2f}.",
        f"- Father-present mean 2021 earnings: {present_cross['mean_earnings']:.2f}.",
        f"- Father-absent mean 2021 earnings: {absent_cross['mean_earnings']:.2f}.",
        f"- Present-minus-absent earnings gap: {(present_cross['mean_earnings'] - absent_cross['mean_earnings']):.2f}.",
        "",
        "## NLSY97 Cognitive And Milestone Gaps",
        f"- Father-absent minus father-present g_proxy gap: {g_proxy_row['absent_minus_present_gap']:.4f}.",
        f"- Adjusted father-absence coefficient for g_proxy: {g_proxy_row['adjusted_absent_coef']:.4f}.",
        f"- Father-absent minus father-present 2007 education-years gap: {education_2007_row['absent_minus_present_gap']:.4f}.",
        "",
        "## NLSY97 Near-Term Post-Only Effects",
        f"- Post_3plus adjusted schooling-engagement coefficient: {schooling_row['adjusted_treatment_coef']:.4f}.",
        f"- Post_3plus adjusted arrest coefficient: {arrest_row['adjusted_treatment_coef']:.4f}.",
        "- These near-term education/crime/work estimates remain descriptive post-only contrasts, not causal event-study estimates.",
        "",
        "## NLSY97 Health, Mental Health, Family, And Occupation",
        f"- Adjusted father-absence coefficient for poor self-rated health (2023): {poor_health_row['adjusted_absent_coef']:.4f}.",
        f"- Adjusted father-absence coefficient for CES-D score (2023): {cesd_row['adjusted_absent_coef']:.4f}.",
        f"- Adjusted father-absence coefficient for ever married by 2021: {ever_married_row['adjusted_absent_coef']:.4f}.",
        f"- Adjusted father-absence coefficient for latest high-skill occupation: {high_skill_row['adjusted_absent_coef']:.4f}.",
        "",
        "## NLSY97 Heterogeneity And Residualized Gaps",
    ]
    if largest_earnings_heterogeneity is not None:
        synthesis_lines.append(
            f"- Largest race-specific annual-earnings gap appears in {largest_earnings_heterogeneity['group_value']}: {largest_earnings_heterogeneity['absent_minus_present_gap']:.2f}."
        )
    if largest_negative_residual is not None:
        synthesis_lines.append(
            f"- Most negative residualized annual-earnings gap appears in {largest_negative_residual['group_type']}={largest_negative_residual['group_value']}: {largest_negative_residual['mean_residual']:.2f}."
        )
    synthesis_lines.extend(
        [
            "- Residualized subgroup tables are net of g_proxy, birth year, and fatherlessness in pooled models; they are descriptive diagnostics, not structural decompositions.",
            "",
        "## Predictors of Fatherlessness",
        ]
    )
    for _, row in predictor_rows.iterrows():
        synthesis_lines.append(
            f"- {row['term']}: odds ratio {row['odds_ratio']:.3f}, p={row['p_value']:.4g}."
        )
    synthesis_lines.extend(
        [
            "",
            "## Public Benchmark Context",
            f"- ACS PUMS 2024 weighted mean annual earnings: {acs_context['mean_earnings']:.2f}.",
            f"- CPS ASEC 2023-2025 pooled weighted mean annual earnings: {cps_context['mean_earnings']:.2f}.",
            f"- SIPP 2023 weighted mean monthly earnings: {sipp_context['mean_earnings']:.2f}.",
            "",
            "## ACS Child Father-Presence Proxy",
            f"- Weighted father-present proxy share: {acs_child_overall['father_present_share']:.4f}.",
            f"- Weighted father-absent proxy share: {acs_child_overall['father_absent_share']:.4f}.",
            f"- Highest race-specific father-absent proxy share: {acs_child_race.iloc[0]['group_value']} at {acs_child_race.iloc[0]['father_absent_share']:.4f}.",
            f"- Highest poverty-band father-absent proxy share: {acs_child_poverty.iloc[0]['group_value']} at {acs_child_poverty.iloc[0]['father_absent_share']:.4f}.",
            "",
            "## Boundaries",
            "- NLSY97 fatherlessness is the locked first-pass treatment definition.",
            "- The NLSY97 g_proxy table uses the same signed-merge, Auto/Shop composite, birth-year residualization, and z-scored unit-weighted construction pattern documented in the `sexg` project, but it remains a descriptive observed composite rather than a causal estimand.",
            "- ACS child father-presence uses the documented `ESP` universe only; it is a public-use proxy, not a full family-history measure.",
            "- Near-term education/crime/work effects come from the preferred post-only localized-exit comparison design and should be narrated as descriptive post-event contrasts, not as causal pretrend-validated event studies.",
            "- SIPP remains monthly context and should not be narrated as annual earnings context.",
        ]
    )
    synthesis_path = manifests_root / "results_synthesis.md"
    synthesis_path.write_text("\n".join(synthesis_lines) + "\n", encoding="utf-8")

    manifest_rows = [
        {"artifact": "nlsy_prevalence_table", "path": nlsy_prevalence_table_path.name, "purpose": "NLSY97 fatherlessness prevalence by total, race, sex, and parental education."},
        {"artifact": "nlsy_predictor_table", "path": nlsy_predictor_table_path.name, "purpose": "Descriptive predictor model for NLSY97 fatherlessness."},
        {"artifact": "nlsy_outcome_gap_table", "path": nlsy_outcome_gap_table_path.name, "purpose": "Adult outcome gaps and public benchmark context."},
        {"artifact": "nlsy_race_gap_table", "path": nlsy_race_gap_table_path.name, "purpose": "NLSY97 race-by-sex adult outcome splits by father-presence status."},
        {"artifact": "nlsy_cognitive_table", "path": nlsy_cognitive_table_path.name, "purpose": "NLSY97 g_proxy and educational milestone gaps by fatherlessness status."},
        {"artifact": "nlsy_cognitive_subgroup_table", "path": nlsy_cognitive_subgroup_table_path.name, "purpose": "NLSY97 g_proxy and educational milestone gaps by sex and race/ethnicity group."},
        {"artifact": "nlsy_near_term_effects_table", "path": nlsy_near_term_effects_table_path.name, "purpose": "Preferred post-only NLSY97 near-term education, crime, and work effects."},
        {"artifact": "nlsy_near_term_robustness_table", "path": nlsy_near_term_robustness_table_path.name, "purpose": "Post_3plus robustness rows for the near-term NLSY97 effect table."},
        {"artifact": "nlsy_health_table", "path": nlsy_health_table_path.name, "purpose": "NLSY97 health and substance-use gaps by fatherlessness status."},
        {"artifact": "nlsy_mental_health_table", "path": nlsy_mental_health_table_path.name, "purpose": "NLSY97 mental-health gaps by fatherlessness status."},
        {"artifact": "nlsy_family_formation_table", "path": nlsy_family_formation_table_path.name, "purpose": "NLSY97 marriage and fertility gaps by fatherlessness status."},
        {"artifact": "nlsy_occupation_summary_table", "path": nlsy_occupation_summary_table_path.name, "purpose": "NLSY97 latest occupation-group composition by fatherlessness status."},
        {"artifact": "nlsy_occupation_effect_table", "path": nlsy_occupation_effect_table_path.name, "purpose": "NLSY97 high-skill occupation gap by fatherlessness status."},
        {"artifact": "nlsy_effect_heterogeneity_table", "path": nlsy_effect_heterogeneity_table_path.name, "purpose": "NLSY97 fatherlessness effects by race, sex, and parent-education band."},
        {"artifact": "nlsy_group_residual_table", "path": nlsy_group_residual_table_path.name, "purpose": "Residualized subgroup gaps net of g_proxy and baseline covariates."},
        {"artifact": "acs_child_context_table", "path": acs_child_context_table_path.name, "purpose": "ACS child father-presence proxy by race, poverty, and income."},
        {"artifact": "benchmark_context_table", "path": benchmark_context_table_path.name, "purpose": "ACS, CPS, and SIPP weighted context table."},
        {"artifact": "results_synthesis", "path": synthesis_path.name, "purpose": "Narrative synthesis for documentation and frontend copy."},
    ]
    manifest_path = manifests_root / "results_appendix_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    results_json_path = manifests_root / "results.json"
    canonical_payload = build_canonical_results_payload(
        artifacts=manifest_rows,
        source_manifest=manifest_path.name,
        synthesis_artifacts=[relative_to_root(synthesis_path, project_root)],
    )
    canonical_errors = validate_canonical_results_payload(canonical_payload)
    if canonical_errors:
        raise ValueError(f"Invalid canonical results payload: {'; '.join(canonical_errors)}")
    results_json_path.write_text(
        json.dumps(canonical_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    handoff_lines = [
        "# Frontend / Doc Handoff",
        "",
        "Status: ready for frontend and documentation work.",
        "",
        "Use these as primary sources:",
        f"- [{nlsy_prevalence_table_path.name}]({nlsy_prevalence_table_path})",
        f"- [{nlsy_predictor_table_path.name}]({nlsy_predictor_table_path})",
        f"- [{nlsy_outcome_gap_table_path.name}]({nlsy_outcome_gap_table_path})",
        f"- [{nlsy_race_gap_table_path.name}]({nlsy_race_gap_table_path})",
        f"- [{nlsy_cognitive_table_path.name}]({nlsy_cognitive_table_path})",
        f"- [{nlsy_cognitive_subgroup_table_path.name}]({nlsy_cognitive_subgroup_table_path})",
        f"- [{nlsy_near_term_effects_table_path.name}]({nlsy_near_term_effects_table_path})",
        f"- [{nlsy_near_term_robustness_table_path.name}]({nlsy_near_term_robustness_table_path})",
        f"- [{nlsy_health_table_path.name}]({nlsy_health_table_path})",
        f"- [{nlsy_mental_health_table_path.name}]({nlsy_mental_health_table_path})",
        f"- [{nlsy_family_formation_table_path.name}]({nlsy_family_formation_table_path})",
        f"- [{nlsy_occupation_summary_table_path.name}]({nlsy_occupation_summary_table_path})",
        f"- [{nlsy_occupation_effect_table_path.name}]({nlsy_occupation_effect_table_path})",
        f"- [{nlsy_effect_heterogeneity_table_path.name}]({nlsy_effect_heterogeneity_table_path})",
        f"- [{nlsy_group_residual_table_path.name}]({nlsy_group_residual_table_path})",
        f"- [{acs_child_context_table_path.name}]({acs_child_context_table_path})",
        f"- [{benchmark_context_table_path.name}]({benchmark_context_table_path})",
        f"- [{synthesis_path.name}]({synthesis_path})",
        "",
        "What is stable enough to narrate:",
        "- NLSY97 fatherlessness prevalence and subgroup differences.",
        "- NLSY97 g_proxy and educational milestone gaps by fatherlessness status, including sex/race subgroup splits where available.",
        "- Preferred NLSY97 post-only near-term education, crime, and work contrasts from the localized-exit design.",
        "- NLSY97 health, substance-use, mental-health, family-formation, and occupation-status differences by fatherlessness status.",
        "- NLSY97 heterogeneity tables by race, sex, and parent education, plus residualized subgroup-gap diagnostics.",
        "- NLSY97 adult earnings and employment gaps by father-presence status.",
        "- ACS child father-presence proxy gradients by race and poverty.",
        "- ACS/CPS/SIPP benchmark context for external comparison.",
        "",
        "What should stay caveated in copy:",
        "- The NLSY97 g_proxy artifact is an observed composite and should not be described as latent g.",
        "- The NLSY97 near-term education/crime/work tables are descriptive post-only contrasts, not causal event-study estimates.",
        "- ACS child father-presence is an `ESP`-based proxy, not a full causal family-history measure.",
        "- NLSY97 predictor coefficients are descriptive associations, not causal estimates.",
        "- SIPP is monthly context, not directly annual-comparable to NLSY97 or CPS annual earnings.",
    ]
    handoff_path = manifests_root / "frontend_doc_handoff.md"
    handoff_path.write_text("\n".join(handoff_lines) + "\n", encoding="utf-8")

    return ResultsAppendixResult(
        manifest_path=manifest_path,
        results_json_path=results_json_path,
        handoff_path=handoff_path,
        synthesis_path=synthesis_path,
        nlsy_prevalence_table_path=nlsy_prevalence_table_path,
        nlsy_predictor_table_path=nlsy_predictor_table_path,
        nlsy_outcome_gap_table_path=nlsy_outcome_gap_table_path,
        nlsy_race_gap_table_path=nlsy_race_gap_table_path,
        nlsy_cognitive_table_path=nlsy_cognitive_table_path,
        nlsy_cognitive_subgroup_table_path=nlsy_cognitive_subgroup_table_path,
        nlsy_near_term_effects_table_path=nlsy_near_term_effects_table_path,
        nlsy_near_term_robustness_table_path=nlsy_near_term_robustness_table_path,
        nlsy_health_table_path=nlsy_health_table_path,
        nlsy_mental_health_table_path=nlsy_mental_health_table_path,
        nlsy_family_formation_table_path=nlsy_family_formation_table_path,
        nlsy_occupation_summary_table_path=nlsy_occupation_summary_table_path,
        nlsy_occupation_effect_table_path=nlsy_occupation_effect_table_path,
        nlsy_effect_heterogeneity_table_path=nlsy_effect_heterogeneity_table_path,
        nlsy_group_residual_table_path=nlsy_group_residual_table_path,
        acs_child_context_table_path=acs_child_context_table_path,
        benchmark_context_table_path=benchmark_context_table_path,
    )
