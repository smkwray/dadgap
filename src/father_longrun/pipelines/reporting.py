from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ResultsAppendixResult:
    manifest_path: Path
    handoff_path: Path
    synthesis_path: Path
    nlsy_prevalence_table_path: Path
    nlsy_predictor_table_path: Path
    nlsy_outcome_gap_table_path: Path
    nlsy_race_gap_table_path: Path
    acs_child_context_table_path: Path
    benchmark_context_table_path: Path


def _required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required reporting input does not exist: {path}")
    return pd.read_csv(path)


def _safe_percent(value: float | None) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value) * 100, 2)


def build_results_appendix(*, outputs_root: Path) -> ResultsAppendixResult:
    manifests_root = outputs_root / "manifests"
    tables_root = outputs_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    fatherlessness = _required_csv(manifests_root / "nlsy97_fatherlessness_group_summary.csv")
    predictors = _required_csv(manifests_root / "nlsy97_fatherlessness_predictors.csv")
    cross_summary = _required_csv(manifests_root / "cross_cohort_benchmark_summary.csv")
    cross_subgroup = _required_csv(manifests_root / "cross_cohort_benchmark_subgroup_summary.csv")
    acs_child = _required_csv(manifests_root / "acs_child_father_presence_summary.csv")
    public_summary = _required_csv(manifests_root / "public_benchmark_profile_summary.csv")

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
        "## Predictors of Fatherlessness",
    ]
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
            "- ACS child father-presence uses the documented `ESP` universe only; it is a public-use proxy, not a full family-history measure.",
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
        {"artifact": "acs_child_context_table", "path": acs_child_context_table_path.name, "purpose": "ACS child father-presence proxy by race, poverty, and income."},
        {"artifact": "benchmark_context_table", "path": benchmark_context_table_path.name, "purpose": "ACS, CPS, and SIPP weighted context table."},
        {"artifact": "results_synthesis", "path": synthesis_path.name, "purpose": "Narrative synthesis for documentation and frontend copy."},
    ]
    manifest_path = manifests_root / "results_appendix_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

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
        f"- [{acs_child_context_table_path.name}]({acs_child_context_table_path})",
        f"- [{benchmark_context_table_path.name}]({benchmark_context_table_path})",
        f"- [{synthesis_path.name}]({synthesis_path})",
        "",
        "What is stable enough to narrate:",
        "- NLSY97 fatherlessness prevalence and subgroup differences.",
        "- NLSY97 adult earnings and employment gaps by father-presence status.",
        "- ACS child father-presence proxy gradients by race and poverty.",
        "- ACS/CPS/SIPP benchmark context for external comparison.",
        "",
        "What should stay caveated in copy:",
        "- ACS child father-presence is an `ESP`-based proxy, not a full causal family-history measure.",
        "- NLSY97 predictor coefficients are descriptive associations, not causal estimates.",
        "- SIPP is monthly context, not directly annual-comparable to NLSY97 or CPS annual earnings.",
    ]
    handoff_path = manifests_root / "frontend_doc_handoff.md"
    handoff_path.write_text("\n".join(handoff_lines) + "\n", encoding="utf-8")

    return ResultsAppendixResult(
        manifest_path=manifest_path,
        handoff_path=handoff_path,
        synthesis_path=synthesis_path,
        nlsy_prevalence_table_path=nlsy_prevalence_table_path,
        nlsy_predictor_table_path=nlsy_predictor_table_path,
        nlsy_outcome_gap_table_path=nlsy_outcome_gap_table_path,
        nlsy_race_gap_table_path=nlsy_race_gap_table_path,
        acs_child_context_table_path=acs_child_context_table_path,
        benchmark_context_table_path=benchmark_context_table_path,
    )
