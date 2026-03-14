from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.csv as pa_csv


COHORT_DISCOVERY_ORDER: tuple[str, ...] = ("nlsy79", "cnlsy", "nlsy97")
MANIFEST_TEMPLATE_MAP: dict[str, str] = {
    "nlsy79": "nlsy79_manifest_template.csv",
    "nlsy97": "nlsy97_manifest_template.csv",
}
SOURCE_FILE_TO_COHORT: dict[str, str] = {
    "nlsy79_main": "nlsy79",
    "nlsy79_child": "cnlsy",
    "nlsy79_ya": "cnlsy",
    "nlsy97": "nlsy97",
}
LIKELY_ID_HINTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("nlsy79_main", "respondent_id"): ("R0000100",),
    ("nlsy79_child", "child_id"): ("C0000100",),
    ("nlsy79_child", "mother_id"): ("C0000200",),
    ("nlsy79_ya", "child_id"): ("C0000100",),
    ("nlsy97", "respondent_id"): ("R0000100",),
}
CNLSY_2014_ATTAINMENT_LABELS: dict[str, dict[int, str]] = {
    "education_attainment_code_2014": {
        1: "8th grade or less",
        2: "Some high school",
        3: "High school graduate",
        4: "Some vocational/technical training (after high school)",
        5: "Completed vocational/technical training (after high school)",
        6: "Some college",
        7: "Completed college (Associate's degree)",
        8: "Completed college (Bachelor's degree)",
        9: "Some graduate school",
        10: "Completed Master's degree",
        11: "Some graduate school beyond Master's degree",
        12: "Completed doctoral degree (PhD)",
        13: "Some post-baccalaureate professional education",
        14: "Completed post-baccalaureate professional education",
    },
    "degree_attainment_code_2014": {
        0: "No degree",
        1: "GED",
        2: "HS diploma",
        3: "Associate's degree",
        4: "Bachelor of Arts",
        5: "Bachelor of Science",
        6: "Master's degree",
        7: "PhD",
        8: "Professional degree",
    },
}
REVIEWED_MAPPING_ROWS: tuple[dict[str, str], ...] = (
    {
        "dataset": "nlsy79_main",
        "canonical_name": "respondent_id",
        "raw_column": "R0000100",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy79_main",
        "canonical_name": "mother_race_ethnicity_raw",
        "raw_column": "R0214700",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy79_main",
        "canonical_name": "mother_sex_raw",
        "raw_column": "R0214800",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy79_main",
        "canonical_name": "mother_birth_year",
        "raw_column": "R0000500",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy79_main",
        "canonical_name": "mother_education_baseline",
        "raw_column": "R0006500",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy79_main",
        "canonical_name": "father_education_baseline",
        "raw_column": "R0007900",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy79_main",
        "canonical_name": "mother_annual_earnings_2000",
        "raw_column": "R3279401",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy79_main",
        "canonical_name": "mother_household_income_2000",
        "raw_column": "R7006500",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy79_main",
        "canonical_name": "mother_education_years",
        "raw_column": "T9900000",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy79_main",
        "canonical_name": "mother_net_worth",
        "raw_column": "R6940103",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "child_id",
        "raw_column": "C0000100",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "mother_id",
        "raw_column": "C0000200",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "child_race_ethnicity_raw",
        "raw_column": "C0005300",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "child_sex_raw",
        "raw_column": "C0005400",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "child_birth_year",
        "raw_column": "C0005700",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "mother_education",
        "raw_column": "C0053500",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "education_years",
        "raw_column": "Y1211300",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "employment_2014",
        "raw_column": "Y3066000",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "annual_earnings_2014",
        "raw_column": "Y3112400",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "wage_income_2014_best_est",
        "raw_column": "Y3291500",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "family_income_2014_best_est",
        "raw_column": "Y3299900",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "age_2014",
        "raw_column": "Y3331900",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "education_years_2014",
        "raw_column": "Y3332100",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "cnlsy",
        "canonical_name": "degree_2014",
        "raw_column": "Y3332200",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "respondent_id",
        "raw_column": "R0000100",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "sex_raw",
        "raw_column": "R0536300",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "birth_year",
        "raw_column": "R0536402",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "race_ethnicity_raw",
        "raw_column": "R1482600",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "mother_education",
        "raw_column": "R0554500",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "father_education",
        "raw_column": "R0554800",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "education_years",
        "raw_column": "Z9083800",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "household_income_2010",
        "raw_column": "T5206900",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "annual_earnings_2019",
        "raw_column": "U4282300",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "household_income_2019",
        "raw_column": "U3444000",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "employment_2019",
        "raw_column": "U3455100",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "annual_earnings_2021",
        "raw_column": "U5753500",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "household_income_2021",
        "raw_column": "U4949700",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "employment_2021",
        "raw_column": "U4958300",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "degree_2021",
        "raw_column": "U5072600",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config",
    },
    {
        "dataset": "nlsy97",
        "canonical_name": "net_worth",
        "raw_column": "Z9121900",
        "transform": "identity",
        "status": "reviewed_from_prior_project",
        "evidence": "prior project config and public documentation",
    },
)
REQUIRED_EXPOSURE_COLUMNS: tuple[dict[str, str], ...] = (
    {"dataset": "nlsy79_child", "canonical_name": "father_present_household"},
    {"dataset": "nlsy79_child", "canonical_name": "father_alive"},
    {"dataset": "nlsy79_child", "canonical_name": "nonresident_contact"},
    {"dataset": "nlsy79_child", "canonical_name": "stepfather_present"},
    {"dataset": "nlsy97", "canonical_name": "household_roster_father_presence"},
)
CURATED_REFRESH_CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "cohort": "nlsy97",
        "source_file": "nlsy97_all_1997-2023",
        "refnum": "R03356.00",
        "canonical_target": "resident_bio_father_present_1997",
        "priority": "high",
        "title": "CHK R LIVES WITH BIO FATHER",
        "reason": "Core treatment anchor for whether youth lives with biological father at baseline.",
        "evidence": "local codebook search in nlsy97_all_1997-2023.cdb",
    },
    {
        "cohort": "nlsy97",
        "source_file": "nlsy97_all_1997-2023",
        "refnum": "R03357.00",
        "canonical_target": "bio_father_contact_ever_1997",
        "priority": "high",
        "title": "R HAD CONTACT WITH BIO FATHER?",
        "reason": "Separates nonresident no-contact from nonresident contact cases.",
        "evidence": "local codebook search in nlsy97_all_1997-2023.cdb",
    },
    {
        "cohort": "nlsy97",
        "source_file": "nlsy97_all_1997-2023",
        "refnum": "R03358.00",
        "canonical_target": "bio_father_alive_1997",
        "priority": "high",
        "title": "BIO FATHER LIVING OR DECEASED?",
        "reason": "Separates paternal death from other absence reasons.",
        "evidence": "local codebook search in nlsy97_all_1997-2023.cdb",
    },
    {
        "cohort": "nlsy97",
        "source_file": "nlsy97_all_1997-2023",
        "refnum": "R03360.00",
        "canonical_target": "lived_apart_from_bio_father_gt12m_1997",
        "priority": "high",
        "title": "R LIVE APART FROM BIO FATHER >12 MONTHS?",
        "reason": "Distinguishes short disruptions from sustained absence.",
        "evidence": "local codebook search in nlsy97_all_1997-2023.cdb",
    },
    {
        "cohort": "nlsy97",
        "source_file": "nlsy97_all_1997-2023",
        "refnum": "R03361.00",
        "canonical_target": "ever_lived_with_bio_father_1997",
        "priority": "high",
        "title": "R EVER LIVE WITH BIO FATHER?",
        "reason": "Separates absent-from-birth from later exit cases.",
        "evidence": "local codebook search in nlsy97_all_1997-2023.cdb",
    },
    {
        "cohort": "nlsy97",
        "source_file": "nlsy97_all_1997-2023",
        "refnum": "R03362.00",
        "canonical_target": "last_year_lived_with_bio_father",
        "priority": "high",
        "title": "WHAT YEAR DID R LAST LIVE WITH BIO FATHER?",
        "reason": "Supports treatment timing and event-time coding.",
        "evidence": "local codebook search in nlsy97_all_1997-2023.cdb",
    },
    {
        "cohort": "nlsy97",
        "source_file": "nlsy97_all_1997-2023",
        "refnum": "R03363.00",
        "canonical_target": "last_month_lived_with_bio_father",
        "priority": "medium",
        "title": "WHAT WAS THE LAST MONTH R LIVED WITH BIO FATHER?",
        "reason": "Refines treatment timing when year alone is too coarse.",
        "evidence": "local codebook search in nlsy97_all_1997-2023.cdb",
    },
    {
        "cohort": "nlsy97",
        "source_file": "nlsy97_all_1997-2023",
        "refnum": "R08854.00-R08858.00",
        "canonical_target": "nonresident_father_distance_1997",
        "priority": "medium",
        "title": "HOW FAR NHH FATHER LIVE FROM YOUTH 01-05?",
        "reason": "Captures nonresident father proximity as a moderator.",
        "evidence": "local codebook search in nlsy97_all_1997-2023.cdb",
    },
    {
        "cohort": "cnlsy",
        "source_file": "nlscya_all_1979-2020",
        "refnum": "C09533.00",
        "canonical_target": "child_ever_sees_father_figure_1990",
        "priority": "high",
        "title": "HOME PART A (0-2 YRS): DOES CHILD EVER SEE FATHER (-FIGURE)?",
        "reason": "Direct early-childhood father-figure contact indicator.",
        "evidence": "local codebook search in nlscya_all_1979-2020.cdb",
    },
    {
        "cohort": "cnlsy",
        "source_file": "nlscya_all_1979-2020",
        "refnum": "C09534.00",
        "canonical_target": "father_figure_type_1990",
        "priority": "high",
        "title": "IS FATHER BIOLOGICAL, STEP-, OR FATHER-FIGURE",
        "reason": "Distinguishes biological father from stepfather/father-figure.",
        "evidence": "local codebook search in nlscya_all_1979-2020.cdb",
    },
    {
        "cohort": "cnlsy",
        "source_file": "nlscya_all_1979-2020",
        "refnum": "C09535.00",
        "canonical_target": "father_figure_relationship_to_mother_1990",
        "priority": "high",
        "title": "FATHER (-FIGURE) RELATIONSHIP TO MOTHER",
        "reason": "Captures union context and separation from residence.",
        "evidence": "local codebook search in nlscya_all_1979-2020.cdb",
    },
    {
        "cohort": "cnlsy",
        "source_file": "nlscya_all_1979-2020",
        "refnum": "C09536.00",
        "canonical_target": "child_sees_father_figure_daily_1990",
        "priority": "medium",
        "title": "DOES CHILD SEE FATHER (-FIGURE) DAILY?",
        "reason": "Adds intensity of exposure, not just presence.",
        "evidence": "local codebook search in nlscya_all_1979-2020.cdb",
    },
    {
        "cohort": "cnlsy",
        "source_file": "nlscya_all_1979-2020",
        "refnum": "C09537.00",
        "canonical_target": "child_eats_with_both_mom_and_dad_1990",
        "priority": "medium",
        "title": "HOW OFTEN CHILD EATS WITH BOTH MOM AND DAD",
        "reason": "Secondary family-routine indicator for father presence.",
        "evidence": "local codebook search in nlscya_all_1979-2020.cdb",
    },
    {
        "cohort": "nlsy79",
        "source_file": "nlsy79_all_1979-2022",
        "refnum": "H00016.00",
        "canonical_target": "biological_father_alive_h40",
        "priority": "medium",
        "title": "R'S BIOLOGICAL FATHER LIVING?",
        "reason": "Needed to separate death from other absence in the maternal cohort.",
        "evidence": "local codebook search in nlsy79_all_1979-2022.cdb",
    },
    {
        "cohort": "nlsy79",
        "source_file": "nlsy79_all_1979-2022",
        "refnum": "H00136.00",
        "canonical_target": "biological_father_alive_h50",
        "priority": "medium",
        "title": "R'S BIOLOGICAL FATHER LIVING?",
        "reason": "Repeat health-module father survival status for later wave consistency.",
        "evidence": "local codebook search in nlsy79_all_1979-2022.cdb",
    },
    {
        "cohort": "nlsy79",
        "source_file": "nlsy79_all_1979-2022",
        "refnum": "H00463.00",
        "canonical_target": "biological_father_alive_h60",
        "priority": "medium",
        "title": "R'S BIOLOGICAL FATHER LIVING?",
        "reason": "Repeat health-module father survival status for later wave consistency.",
        "evidence": "local codebook search in nlsy79_all_1979-2022.cdb",
    },
)


@dataclass(frozen=True)
class CohortExtract:
    cohort: str
    panel_extract_path: Path
    manifest_path: Path | None
    varmap_path: Path | None
    row_count: int | None
    column_count: int
    source_path: str | None
    columns: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedCohort:
    cohort: str
    parquet_path: Path
    dictionary_path: Path
    row_count: int
    column_count: int


@dataclass(frozen=True)
class PilotBuildResult:
    artifacts: tuple[MaterializedCohort, ...]
    inventory_markdown_path: Path
    inventory_json_path: Path


@dataclass(frozen=True)
class ManifestArtifact:
    dataset_key: str
    manifest_path: Path
    row_count: int


@dataclass(frozen=True)
class KeyDiagnostic:
    cohort: str
    key_name: str
    column: str
    row_count: int
    duplicate_rows: int
    null_rows: int


@dataclass(frozen=True)
class ManifestBuildResult:
    manifests: tuple[ManifestArtifact, ...]
    diagnostics_csv_path: Path
    diagnostics_markdown_path: Path


@dataclass(frozen=True)
class MergeContractResult:
    report_path: Path
    json_path: Path


@dataclass(frozen=True)
class BackboneBuildResult:
    parquet_path: Path
    report_path: Path
    row_count: int


@dataclass(frozen=True)
class ReviewedLayerResult:
    mapping_csv_path: Path
    availability_markdown_path: Path
    exposure_gap_markdown_path: Path
    backbone_parquet_path: Path
    nlsy97_parquet_path: Path


@dataclass(frozen=True)
class RefreshSpecResult:
    csv_path: Path
    markdown_path: Path
    yaml_path: Path


@dataclass(frozen=True)
class RefreshExtractArtifact:
    cohort: str
    panel_extract_path: Path
    varmap_path: Path
    manifest_path: Path
    added_columns: tuple[str, ...]
    row_count: int


@dataclass(frozen=True)
class RefreshExtractBuildResult:
    refreshed_root: Path
    report_path: Path
    artifacts: tuple[RefreshExtractArtifact, ...]


@dataclass(frozen=True)
class TreatmentLayerBuildResult:
    backbone_path: Path
    nlsy97_path: Path
    mapping_path: Path
    value_counts_path: Path


@dataclass(frozen=True)
class AnalysisReadyTreatmentResult:
    backbone_path: Path
    nlsy97_path: Path
    nlsy97_baseline_path: Path
    nlsy97_primary_baseline_path: Path
    cnlsy_subset_path: Path
    cnlsy_baseline_path: Path
    cnlsy_primary_baseline_path: Path
    cnlsy_readiness_path: Path
    cnlsy_outcome_tiering_path: Path
    cnlsy_education_validation_path: Path
    cnlsy_education_crosstab_path: Path
    cnlsy_attainment_codebook_path: Path
    coding_rules_path: Path
    summary_path: Path


@dataclass(frozen=True)
class FatherlessnessProfileResult:
    group_summary_path: Path
    predictor_path: Path
    report_path: Path


@dataclass(frozen=True)
class NLSY97LongitudinalPanelResult:
    panel_path: Path
    childhood_history_path: Path
    summary_path: Path
    availability_path: Path
    childhood_availability_path: Path


def _read_manifest(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    return tuple(header)


def _refnum_to_raw_column(refnum: str) -> str:
    return refnum.replace(".", "")


def _expand_refnum_spec(refnum_spec: str) -> tuple[str, ...]:
    if "-" not in refnum_spec:
        return (_refnum_to_raw_column(refnum_spec),)

    start_refnum, end_refnum = refnum_spec.split("-", 1)
    start_match = re.match(r"^([A-Z]+)(\d+)\.00$", start_refnum)
    end_match = re.match(r"^([A-Z]+)(\d+)\.00$", end_refnum)
    if start_match is None or end_match is None:
        raise ValueError(f"Unsupported refnum range syntax: {refnum_spec}")
    if start_match.group(1) != end_match.group(1):
        raise ValueError(f"Refnum range crosses prefixes: {refnum_spec}")

    prefix = start_match.group(1)
    start_num = int(start_match.group(2))
    end_num = int(end_match.group(2))
    width = len(start_match.group(2))
    return tuple(f"{prefix}{value:0{width}d}00" for value in range(start_num, end_num + 1))


def _expanded_refresh_candidates() -> tuple[dict[str, str], ...]:
    expanded: list[dict[str, str]] = []
    for row in CURATED_REFRESH_CANDIDATES:
        raw_columns = _expand_refnum_spec(row["refnum"])
        if len(raw_columns) == 1:
            expanded.append({**row, "raw_column": raw_columns[0], "canonical_name": row["canonical_target"]})
            continue
        for index, raw_column in enumerate(raw_columns, start=1):
            expanded.append(
                {
                    **row,
                    "raw_column": raw_column,
                    "canonical_name": f"{row['canonical_target']}_{index:02d}",
                }
            )
    return tuple(expanded)


def discover_cohort_extracts(interim_root: Path) -> tuple[CohortExtract, ...]:
    extracts: list[CohortExtract] = []
    for cohort in COHORT_DISCOVERY_ORDER:
        cohort_dir = interim_root / cohort
        panel_extract_path = cohort_dir / "panel_extract.csv"
        if not panel_extract_path.exists():
            continue
        manifest_path = cohort_dir / "panel_extract.manifest.json"
        varmap_path = cohort_dir / "varmap.csv"
        manifest = _read_manifest(manifest_path if manifest_path.exists() else None)
        columns = _read_csv_header(panel_extract_path)
        extracts.append(
            CohortExtract(
                cohort=cohort,
                panel_extract_path=panel_extract_path,
                manifest_path=manifest_path if manifest_path.exists() else None,
                varmap_path=varmap_path if varmap_path.exists() else None,
                row_count=manifest.get("n_rows"),
                column_count=manifest.get("n_columns") or len(columns),
                source_path=manifest.get("source_path"),
                columns=columns,
            )
        )
    return tuple(extracts)


def _sanitize_public_path(value: str | Path | None, *, interim_root: Path | None = None) -> str:
    if value in (None, ""):
        return ""

    raw = str(value)
    try:
        path = Path(raw).expanduser()
    except (TypeError, ValueError):
        return raw

    if not path.is_absolute():
        return raw

    if interim_root is not None:
        try:
            relative_to_interim = path.resolve().relative_to(interim_root.resolve())
            return f"<nlsy_interim_root>/{relative_to_interim.as_posix()}"
        except ValueError:
            pass

    try:
        relative_to_repo = path.resolve().relative_to(Path(__file__).resolve().parents[3])
        return relative_to_repo.as_posix()
    except ValueError:
        pass

    parts = path.parts[-3:] if len(path.parts) >= 3 else path.parts
    return f"<external>/{'/'.join(parts)}"


def _inventory_rows(extracts: tuple[CohortExtract, ...], *, interim_root: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for extract in extracts:
        rows.append(
            {
                "cohort": extract.cohort,
                "panel_extract_path": _sanitize_public_path(extract.panel_extract_path, interim_root=interim_root),
                "manifest_path": _sanitize_public_path(extract.manifest_path, interim_root=interim_root),
                "varmap_path": _sanitize_public_path(extract.varmap_path, interim_root=interim_root),
                "row_count": extract.row_count,
                "column_count": extract.column_count,
                "source_path": _sanitize_public_path(extract.source_path, interim_root=interim_root),
                "sample_columns": list(extract.columns[:10]),
            }
        )
    return rows


def write_inventory_report(
    extracts: tuple[CohortExtract, ...],
    *,
    report_dir: Path,
    interim_root: Path | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (generated_at or datetime.now(timezone.utc)).isoformat()
    inventory_rows = _inventory_rows(extracts, interim_root=interim_root)

    markdown_path = report_dir / "nlsy_inventory.md"
    json_path = report_dir / "nlsy_inventory.json"

    markdown_lines = [
        "# NLSY Inventory",
        "",
        f"Generated at: {timestamp}",
        "",
        "| cohort | rows | columns | panel extract | manifest | varmap |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in inventory_rows:
        markdown_lines.append(
            f"| {row['cohort']} | {row['row_count'] if row['row_count'] is not None else '?'} "
            f"| {row['column_count']} | {row['panel_extract_path']} | {row['manifest_path'] or '-'} "
            f"| {row['varmap_path'] or '-'} |"
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps({"generated_at": timestamp, "extracts": inventory_rows}, indent=2),
        encoding="utf-8",
    )
    return {"markdown": markdown_path, "json": json_path}


def _build_dictionary_frame(extract: CohortExtract) -> pd.DataFrame:
    header_frame = pd.DataFrame({"refnum": list(extract.columns)})
    if extract.varmap_path is None or not extract.varmap_path.exists():
        header_frame["question_name"] = header_frame["refnum"]
        header_frame["title"] = header_frame["refnum"]
        header_frame["survey_year"] = ""
        header_frame["type"] = "unknown"
        return header_frame

    varmap = pd.read_csv(extract.varmap_path)
    dictionary = header_frame.merge(varmap, on="refnum", how="left")
    dictionary["in_panel_extract"] = True
    return dictionary


def _template_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "templates"


def _read_template_rows(template_name: str) -> list[dict[str, str]]:
    path = _template_dir() / template_name
    frame = pd.read_csv(path, dtype=str).fillna("")
    return frame.to_dict(orient="records")


def _suggest_candidates(dataset: str, logical_name: str, extract: CohortExtract) -> tuple[str, ...]:
    hints = LIKELY_ID_HINTS.get((dataset, logical_name), ())
    suggestions = [column for column in hints if column in extract.columns]
    return tuple(suggestions)


def _manifest_status(required: str, candidate_variable: str) -> str:
    if candidate_variable:
        return "suggested"
    if required.lower() == "yes":
        return "required_manual_review"
    return "optional_manual_review"


def generate_draft_manifests(
    *,
    extracts: tuple[CohortExtract, ...],
    output_dir: Path,
) -> tuple[ManifestArtifact, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_map = {extract.cohort: extract for extract in extracts}
    artifacts: list[ManifestArtifact] = []

    for dataset_key, template_name in MANIFEST_TEMPLATE_MAP.items():
        rows = _read_template_rows(template_name)
        draft_rows: list[dict[str, str]] = []
        for row in rows:
            source_file = row["source_file"]
            mapped_cohort = SOURCE_FILE_TO_COHORT.get(source_file)
            extract = extract_map.get(mapped_cohort) if mapped_cohort else None
            suggestions = _suggest_candidates(row["dataset"], row["logical_name"], extract) if extract else ()
            draft_rows.append(
                {
                    **row,
                    "candidate_variable": "|".join(suggestions),
                    "status": _manifest_status(row["required"], "|".join(suggestions)),
                    "available_cohort": mapped_cohort or "",
                    "available_column_count": str(len(extract.columns)) if extract else "",
                    "suggestion_basis": "id_hint" if suggestions else "manual_review",
                }
            )

        manifest_path = output_dir / f"{dataset_key}_draft_manifest.csv"
        pd.DataFrame(draft_rows).to_csv(manifest_path, index=False)
        artifacts.append(ManifestArtifact(dataset_key=dataset_key, manifest_path=manifest_path, row_count=len(draft_rows)))
    return tuple(artifacts)


def _diagnose_key(frame: pd.DataFrame, cohort: str, key_name: str, column: str) -> KeyDiagnostic:
    series = frame[column]
    duplicate_rows = int(series.duplicated(keep=False).sum())
    null_rows = int(series.isna().sum())
    return KeyDiagnostic(
        cohort=cohort,
        key_name=key_name,
        column=column,
        row_count=len(frame.index),
        duplicate_rows=duplicate_rows,
        null_rows=null_rows,
    )


def run_key_diagnostics(
    *,
    extracts: tuple[CohortExtract, ...],
    output_dir: Path,
) -> tuple[KeyDiagnostic, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rules: dict[str, dict[str, str]] = {
        "nlsy79": {"respondent_id": "R0000100"},
        "cnlsy": {"child_id": "C0000100", "mother_id": "C0000200"},
        "nlsy97": {"respondent_id": "R0000100"},
    }
    diagnostics: list[KeyDiagnostic] = []
    for extract in extracts:
        cohort_rules = rules.get(extract.cohort, {})
        rule_columns = set(cohort_rules.values())
        frame = pd.read_csv(extract.panel_extract_path, usecols=lambda c: c in rule_columns)
        for key_name, column in cohort_rules.items():
            if column in frame.columns:
                diagnostics.append(_diagnose_key(frame, extract.cohort, key_name, column))

    csv_path = output_dir / "nlsy_key_diagnostics.csv"
    markdown_path = output_dir / "nlsy_key_diagnostics.md"
    diagnostics_frame = pd.DataFrame([diagnostic.__dict__ for diagnostic in diagnostics])
    diagnostics_frame.to_csv(csv_path, index=False)

    lines = [
        "# NLSY Key Diagnostics",
        "",
        "| cohort | key_name | column | rows | duplicate_rows | null_rows |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in diagnostics:
        lines.append(
            f"| {item.cohort} | {item.key_name} | {item.column} | {item.row_count} | "
            f"{item.duplicate_rows} | {item.null_rows} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tuple(diagnostics)


def build_phase0_artifacts(
    *,
    interim_root: Path,
    output_dir: Path,
) -> ManifestBuildResult:
    extracts = discover_cohort_extracts(interim_root)
    if not extracts:
        raise FileNotFoundError(f"No NLSY cohort extracts found under {interim_root}")
    manifests = generate_draft_manifests(extracts=extracts, output_dir=output_dir)
    run_key_diagnostics(extracts=extracts, output_dir=output_dir)
    return ManifestBuildResult(
        manifests=manifests,
        diagnostics_csv_path=output_dir / "nlsy_key_diagnostics.csv",
        diagnostics_markdown_path=output_dir / "nlsy_key_diagnostics.md",
    )


def build_merge_contract_report(
    *,
    interim_root: Path,
    output_dir: Path,
) -> MergeContractResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    nlsy79 = pd.read_csv(interim_root / "nlsy79" / "panel_extract.csv", usecols=["R0000100"])
    cnlsy = pd.read_csv(interim_root / "cnlsy" / "panel_extract.csv", usecols=["C0000100", "C0000200"])
    nlsy97 = pd.read_csv(interim_root / "nlsy97" / "panel_extract.csv", usecols=["R0000100"])

    matched = cnlsy["C0000200"].isin(set(nlsy79["R0000100"]))
    payload = {
        "nlsy79": {
            "row_count": int(len(nlsy79.index)),
            "respondent_id_unique": int(nlsy79["R0000100"].nunique()),
        },
        "cnlsy": {
            "row_count": int(len(cnlsy.index)),
            "child_id_unique": int(cnlsy["C0000100"].nunique()),
            "mother_id_unique": int(cnlsy["C0000200"].nunique()),
            "child_rows_with_matching_mother": int(matched.sum()),
            "child_rows_without_matching_mother": int((~matched).sum()),
        },
        "nlsy97": {
            "row_count": int(len(nlsy97.index)),
            "respondent_id_unique": int(nlsy97["R0000100"].nunique()),
        },
        "contract": {
            "nlsy79_to_cnlsy_join": "nlsy79.R0000100 = cnlsy.C0000200",
            "expected_cardinality": "one_to_many",
            "cnlsy_child_id_key": "C0000100",
            "status": "ready_for_backbone_join_scaffold" if int((~matched).sum()) == 0 else "needs_linkage_review",
        },
    }

    report_path = output_dir / "nlsy_merge_contract.md"
    json_path = output_dir / "nlsy_merge_contract.json"
    lines = [
        "# NLSY Merge Contract",
        "",
        "## Backbone keys",
        "",
        "- `nlsy79.R0000100` is the unique respondent key in the NLSY79 extract.",
        "- `cnlsy.C0000100` is the unique child key in the CNLSY extract.",
        "- `cnlsy.C0000200` links each CNLSY child row back to the NLSY79 mother/respondent.",
        "- `nlsy97.R0000100` is the standalone respondent key for the replication cohort.",
        "",
        "## Join feasibility",
        "",
        f"- NLSY79 rows: {payload['nlsy79']['row_count']}",
        f"- CNLSY rows: {payload['cnlsy']['row_count']}",
        f"- CNLSY unique children: {payload['cnlsy']['child_id_unique']}",
        f"- CNLSY unique mothers: {payload['cnlsy']['mother_id_unique']}",
        f"- CNLSY child rows with matching NLSY79 mother ID: {payload['cnlsy']['child_rows_with_matching_mother']}",
        f"- CNLSY child rows without matching NLSY79 mother ID: {payload['cnlsy']['child_rows_without_matching_mother']}",
        "",
        "## Contract",
        "",
        f"- Join rule: `{payload['contract']['nlsy79_to_cnlsy_join']}`",
        f"- Expected cardinality: `{payload['contract']['expected_cardinality']}`",
        f"- Status: `{payload['contract']['status']}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return MergeContractResult(report_path=report_path, json_path=json_path)


def build_backbone_scaffold(
    *,
    interim_root: Path,
    processed_root: Path,
    output_dir: Path,
) -> BackboneBuildResult:
    processed_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.read_csv(interim_root / "nlsy79" / "panel_extract.csv").add_prefix("nlsy79__")
    cnlsy = pd.read_csv(interim_root / "cnlsy" / "panel_extract.csv").add_prefix("cnlsy__")
    merged = cnlsy.merge(
        nlsy79,
        left_on="cnlsy__C0000200",
        right_on="nlsy79__R0000100",
        how="left",
        validate="many_to_one",
    )
    unmatched_rows = int(merged["nlsy79__R0000100"].isna().sum())
    if unmatched_rows:
        raise ValueError(f"Backbone scaffold has {unmatched_rows} unmatched CNLSY rows")

    merged.insert(0, "mother_id", merged["cnlsy__C0000200"])
    merged.insert(0, "child_id", merged["cnlsy__C0000100"])
    merged.insert(0, "respondent_id", merged["nlsy79__R0000100"])

    parquet_path = processed_root / "nlsy79_cnlsy_backbone_scaffold.parquet"
    report_path = output_dir / "nlsy_backbone_scaffold.md"
    merged.to_parquet(parquet_path, index=False)

    lines = [
        "# NLSY Backbone Scaffold",
        "",
        "Structural scaffold only. Raw columns are preserved with cohort prefixes.",
        "",
        f"- Output parquet: `{parquet_path.name}`",
        f"- Rows: {len(merged.index)}",
        f"- Columns: {len(merged.columns)}",
        "- Join: `cnlsy.C0000200 -> nlsy79.R0000100`",
        "- Cardinality validated as `many_to_one` from CNLSY child rows to NLSY79 respondent rows.",
        "- Added convenience keys: `respondent_id`, `mother_id`, `child_id`.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BackboneBuildResult(parquet_path=parquet_path, report_path=report_path, row_count=int(len(merged.index)))


def _harmonize_race_ethnicity_3cat(cohort: str, raw: pd.Series) -> pd.Series:
    raw_num = pd.to_numeric(raw, errors="coerce")
    cohort_key = cohort.lower()
    if cohort_key in {"nlsy79", "cnlsy"}:
        mapping = {1: "HISPANIC", 2: "BLACK", 3: "NON-BLACK, NON-HISPANIC"}
    elif cohort_key == "nlsy97":
        mapping = {1: "BLACK", 2: "HISPANIC", 3: "NON-BLACK, NON-HISPANIC", 4: "NON-BLACK, NON-HISPANIC"}
    else:
        return pd.Series(pd.NA, index=raw.index, dtype="string")
    return raw_num.map(mapping).astype("string")


def _compute_parent_education(mother: pd.Series, father: pd.Series | None = None) -> pd.Series:
    mother_num = pd.to_numeric(mother, errors="coerce")
    if father is None:
        return mother_num
    father_num = pd.to_numeric(father, errors="coerce")
    return pd.concat([mother_num, father_num], axis=1).mean(axis=1, skipna=True)


def _mapping_frame() -> pd.DataFrame:
    return pd.DataFrame(REVIEWED_MAPPING_ROWS)


def _availability_rows(frame: pd.DataFrame, *, dataset: str) -> list[dict[str, object]]:
    mapping = _mapping_frame()
    subset = mapping.loc[mapping["dataset"] == dataset]
    rows: list[dict[str, object]] = []
    for _, item in subset.iterrows():
        column = str(item["canonical_name"])
        present = column in frame.columns
        non_null = int(frame[column].notna().sum()) if present else 0
        rows.append(
            {
                "dataset": dataset,
                "canonical_name": column,
                "status": item["status"],
                "present_in_mapped_file": "yes" if present else "no",
                "non_null_rows": non_null,
            }
        )
    return rows


def build_reviewed_layers(
    *,
    interim_root: Path,
    processed_root: Path,
    output_dir: Path,
) -> ReviewedLayerResult:
    processed_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    nlsy79 = pd.read_csv(interim_root / "nlsy79" / "panel_extract.csv")
    cnlsy = pd.read_csv(interim_root / "cnlsy" / "panel_extract.csv")
    nlsy97 = pd.read_csv(interim_root / "nlsy97" / "panel_extract.csv")

    backbone = cnlsy.merge(
        nlsy79,
        left_on="C0000200",
        right_on="R0000100",
        how="left",
        validate="many_to_one",
        suffixes=("_cnlsy", "_nlsy79"),
    )
    if backbone["R0000100"].isna().any():
        raise ValueError("Backbone reviewed layer found unmatched CNLSY mother IDs")

    reviewed_backbone = pd.DataFrame(
        {
            "respondent_id": backbone["R0000100"],
            "mother_id": backbone["C0000200"],
            "child_id": backbone["C0000100"],
            "mother_race_ethnicity_raw": backbone["R0214700"],
            "mother_race_ethnicity_3cat": _harmonize_race_ethnicity_3cat("nlsy79", backbone["R0214700"]),
            "mother_sex_raw": backbone["R0214800"],
            "mother_birth_year": pd.to_numeric(backbone["R0000500"], errors="coerce"),
            "mother_education_baseline": pd.to_numeric(backbone["R0006500"], errors="coerce"),
            "father_education_baseline": pd.to_numeric(backbone["R0007900"], errors="coerce"),
            "child_race_ethnicity_raw": backbone["C0005300"],
            "child_race_ethnicity_3cat": _harmonize_race_ethnicity_3cat("cnlsy", backbone["C0005300"]),
            "child_sex_raw": backbone["C0005400"],
            "child_birth_year": pd.to_numeric(backbone["C0005700"], errors="coerce"),
            "mother_education": pd.to_numeric(backbone["C0053500"], errors="coerce"),
            "parent_education": _compute_parent_education(backbone["C0053500"]),
            "education_years": pd.to_numeric(backbone["Y1211300"], errors="coerce"),
            "employment_2014": pd.to_numeric(backbone["Y3066000"], errors="coerce"),
            "annual_earnings_2014": pd.to_numeric(backbone["Y3112400"], errors="coerce"),
            "wage_income_2014_best_est": pd.to_numeric(backbone["Y3291500"], errors="coerce"),
            "family_income_2014_best_est": pd.to_numeric(backbone["Y3299900"], errors="coerce"),
            "age_2014": pd.to_numeric(backbone["Y3331900"], errors="coerce"),
            "education_years_2014": pd.to_numeric(backbone["Y3332100"], errors="coerce"),
            "degree_2014": pd.to_numeric(backbone["Y3332200"], errors="coerce"),
            "mother_education_years": pd.to_numeric(backbone["T9900000"], errors="coerce"),
            "mother_annual_earnings_2000": pd.to_numeric(backbone["R3279401"], errors="coerce"),
            "mother_household_income_2000": pd.to_numeric(backbone["R7006500"], errors="coerce"),
            "mother_net_worth": pd.to_numeric(backbone["R6940103"], errors="coerce"),
        }
    )

    reviewed_nlsy97 = pd.DataFrame(
        {
            "respondent_id": nlsy97["R0000100"],
            "sex_raw": nlsy97["R0536300"],
            "birth_year": pd.to_numeric(nlsy97["R0536402"], errors="coerce"),
            "race_ethnicity_raw": nlsy97["R1482600"],
            "race_ethnicity_3cat": _harmonize_race_ethnicity_3cat("nlsy97", nlsy97["R1482600"]),
            "mother_education": pd.to_numeric(nlsy97["R0554500"], errors="coerce"),
            "father_education": pd.to_numeric(nlsy97["R0554800"], errors="coerce"),
            "parent_education": _compute_parent_education(nlsy97["R0554500"], nlsy97["R0554800"]),
            "education_years": pd.to_numeric(nlsy97["Z9083800"], errors="coerce"),
            "household_income_2010": pd.to_numeric(nlsy97["T5206900"], errors="coerce"),
            "annual_earnings_2019": pd.to_numeric(nlsy97["U4282300"], errors="coerce"),
            "household_income_2019": pd.to_numeric(nlsy97["U3444000"], errors="coerce"),
            "employment_2019": pd.to_numeric(nlsy97["U3455100"], errors="coerce"),
            "annual_earnings_2021": pd.to_numeric(nlsy97["U5753500"], errors="coerce"),
            "household_income_2021": pd.to_numeric(nlsy97["U4949700"], errors="coerce"),
            "employment_2021": pd.to_numeric(nlsy97["U4958300"], errors="coerce"),
            "degree_2021": pd.to_numeric(nlsy97["U5072600"], errors="coerce"),
            "net_worth": pd.to_numeric(nlsy97["Z9121900"], errors="coerce"),
        }
    )

    backbone_parquet_path = processed_root / "nlsy79_cnlsy_backbone_reviewed.parquet"
    nlsy97_parquet_path = processed_root / "nlsy97_reviewed.parquet"
    reviewed_backbone.to_parquet(backbone_parquet_path, index=False)
    reviewed_nlsy97.to_parquet(nlsy97_parquet_path, index=False)

    mapping_csv_path = output_dir / "nlsy_reviewed_mappings.csv"
    _mapping_frame().to_csv(mapping_csv_path, index=False)

    availability_rows = _availability_rows(reviewed_backbone, dataset="cnlsy")
    availability_rows.extend(_availability_rows(reviewed_nlsy97, dataset="nlsy97"))
    availability_rows.extend(_availability_rows(reviewed_backbone, dataset="nlsy79_main"))
    availability_frame = pd.DataFrame(availability_rows)
    availability_markdown_path = output_dir / "nlsy_reviewed_availability.md"
    availability_lines = [
        "# NLSY Reviewed Availability",
        "",
        "| dataset | canonical_name | status | present_in_mapped_file | non_null_rows |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for _, row in availability_frame.iterrows():
        availability_lines.append(
            f"| {row['dataset']} | {row['canonical_name']} | {row['status']} | "
            f"{row['present_in_mapped_file']} | {row['non_null_rows']} |"
        )
    availability_markdown_path.write_text("\n".join(availability_lines) + "\n", encoding="utf-8")

    exposure_gap_markdown_path = output_dir / "nlsy_exposure_gap.md"
    exposure_lines = [
        "# NLSY Exposure Gap",
        "",
        "The current fallback extracts support reviewed baseline covariates and adult outcome variables,",
        "but they do not include the father-residence/contact variables required for the main treatment definition.",
        "",
        "| dataset | canonical_name | status |",
        "| --- | --- | --- |",
    ]
    for item in REQUIRED_EXPOSURE_COLUMNS:
        exposure_lines.append(f"| {item['dataset']} | {item['canonical_name']} | missing_from_current_extract |")
    exposure_lines.append("")
    exposure_lines.append("Next action: refresh the NLSY extracts to include father-presence, father-contact, and related family-structure variables.")
    exposure_gap_markdown_path.write_text("\n".join(exposure_lines) + "\n", encoding="utf-8")

    return ReviewedLayerResult(
        mapping_csv_path=mapping_csv_path,
        availability_markdown_path=availability_markdown_path,
        exposure_gap_markdown_path=exposure_gap_markdown_path,
        backbone_parquet_path=backbone_parquet_path,
        nlsy97_parquet_path=nlsy97_parquet_path,
    )


def build_refresh_spec(*, output_dir: Path) -> RefreshSpecResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(CURATED_REFRESH_CANDIDATES)

    csv_path = output_dir / "nlsy_treatment_refresh_candidates.csv"
    markdown_path = output_dir / "nlsy_treatment_refresh_spec.md"
    yaml_path = output_dir / "nlsy_treatment_refresh_request.yaml"

    frame.to_csv(csv_path, index=False)

    markdown_lines = [
        "# NLSY Treatment Refresh Spec",
        "",
        "These candidates were recovered from prior project codebook assets and should be added to the next NLSY extract refresh.",
        "",
        "| cohort | source_file | refnum | canonical_target | priority | title |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in frame.iterrows():
        markdown_lines.append(
            f"| {row['cohort']} | {row['source_file']} | {row['refnum']} | "
            f"{row['canonical_target']} | {row['priority']} | {row['title']} |"
        )
    markdown_lines.append("")
    markdown_lines.append("Notes:")
    markdown_lines.append("- Refnums are cohort-specific, not globally unique. Request by cohort/source file, not by refnum alone.")
    markdown_lines.append("- `nlsy97` has the strongest immediately usable father-residence/contact variables in the local codebook search.")
    markdown_lines.append("- `cnlsy` contributes early-childhood father-figure exposure variables from the HOME module.")
    markdown_lines.append("- `nlsy79` currently contributes father-survival markers; a second pass is still needed for maternal marital/cohabitation and household-partner structure variables.")
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    yaml_lines = [
        "request_name: nlsy_treatment_refresh",
        "notes:",
        "  - Add father-residence/contact and family-structure variables to the next public-use NLSY extracts.",
        "  - Keep existing reviewed outcome/covariate columns in place.",
        "  - Submit requests by cohort/source file because some refnums collide across cohorts.",
        "cohorts:",
    ]
    for cohort in ("nlsy79", "cnlsy", "nlsy97"):
        yaml_lines.append(f"  {cohort}:")
        source_files = sorted(set(frame.loc[frame["cohort"] == cohort, "source_file"]))
        yaml_lines.append("    source_files:")
        for source_file in source_files:
            yaml_lines.append(f"      - {source_file}")
        yaml_lines.append("    include_variables:")
        for _, row in frame.loc[frame["cohort"] == cohort].iterrows():
            yaml_lines.append(f"      - refnum: {row['refnum']}")
            yaml_lines.append(f"        canonical_target: {row['canonical_target']}")
            yaml_lines.append(f"        priority: {row['priority']}")
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return RefreshSpecResult(csv_path=csv_path, markdown_path=markdown_path, yaml_path=yaml_path)


def build_treatment_refresh_extracts(
    *,
    interim_root: Path,
    refreshed_root: Path,
    output_dir: Path,
) -> RefreshExtractBuildResult:
    refreshed_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _expanded_refresh_candidates()
    artifacts: list[RefreshExtractArtifact] = []
    report_lines = [
        "# NLSY Treatment Refresh Build",
        "",
        "| cohort | rows | panel_extract | added_columns |",
        "| --- | ---: | --- | --- |",
    ]

    for cohort in COHORT_DISCOVERY_ORDER:
        source_panel = interim_root / cohort / "panel_extract.csv"
        source_varmap = interim_root / cohort / "varmap.csv"
        source_raw_candidates = sorted((interim_root / cohort / "raw_files").glob("*.csv"))
        if not source_raw_candidates:
            raise FileNotFoundError(f"No raw CSV found under {interim_root / cohort / 'raw_files'}")
        source_raw = source_raw_candidates[0]
        current_columns = list(_read_csv_header(source_panel))
        treatment_columns = [row["raw_column"] for row in candidates if row["cohort"] == cohort]
        added_columns = [column for column in treatment_columns if column not in current_columns]
        selected_columns = list(dict.fromkeys(current_columns + added_columns))

        frame = pa_csv.read_csv(
            source_raw,
            read_options=pa_csv.ReadOptions(block_size=1 << 26),
            convert_options=pa_csv.ConvertOptions(include_columns=selected_columns),
        ).to_pandas()
        cohort_output_dir = refreshed_root / cohort
        cohort_output_dir.mkdir(parents=True, exist_ok=True)
        panel_path = cohort_output_dir / "panel_extract.csv"
        varmap_path = cohort_output_dir / "varmap.csv"
        manifest_path = cohort_output_dir / "panel_extract.manifest.json"
        frame.to_csv(panel_path, index=False)

        varmap = pd.read_csv(source_varmap)
        varmap[varmap["refnum"].isin(selected_columns)].to_csv(varmap_path, index=False)
        manifest_payload = {
            "cohort": cohort,
            "source_path": f"<nlsy_interim_root>/{cohort}/raw_files/{source_raw.name}",
            "output_path": f"data/interim/nlsy_refresh/{cohort}/panel_extract.csv",
            "selected_columns": selected_columns,
            "added_treatment_columns": added_columns,
            "n_rows": int(len(frame.index)),
            "n_columns": int(len(frame.columns)),
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

        artifacts.append(
            RefreshExtractArtifact(
                cohort=cohort,
                panel_extract_path=panel_path,
                varmap_path=varmap_path,
                manifest_path=manifest_path,
                added_columns=tuple(added_columns),
                row_count=int(len(frame.index)),
            )
        )
        report_lines.append(
            f"| {cohort} | {len(frame.index)} | {panel_path.name} | "
            f"{', '.join(added_columns) if added_columns else '-'} |"
        )

    report_path = output_dir / "nlsy_treatment_refresh_build.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return RefreshExtractBuildResult(
        refreshed_root=refreshed_root,
        report_path=report_path,
        artifacts=tuple(artifacts),
    )


def build_treatment_candidate_layers(
    *,
    refreshed_root: Path,
    processed_root: Path,
    output_dir: Path,
) -> TreatmentLayerBuildResult:
    processed_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _expanded_refresh_candidates()
    nlsy79_map = {row["raw_column"]: row["canonical_name"] for row in candidates if row["cohort"] == "nlsy79"}
    cnlsy_map = {row["raw_column"]: row["canonical_name"] for row in candidates if row["cohort"] == "cnlsy"}
    nlsy97_map = {row["raw_column"]: row["canonical_name"] for row in candidates if row["cohort"] == "nlsy97"}

    reviewed_backbone_path = processed_root / "nlsy79_cnlsy_backbone_reviewed.parquet"
    scaffold_backbone_path = processed_root / "nlsy79_cnlsy_backbone_scaffold.parquet"
    base_backbone = pd.read_parquet(reviewed_backbone_path if reviewed_backbone_path.exists() else scaffold_backbone_path)

    nlsy79_usecols = ["R0000100", *nlsy79_map.keys()]
    nlsy79_frame = pd.read_csv(refreshed_root / "nlsy79" / "panel_extract.csv", usecols=nlsy79_usecols).rename(
        columns={"R0000100": "respondent_id", **nlsy79_map}
    )
    cnlsy_usecols = ["C0000100", "C0000200", *cnlsy_map.keys()]
    cnlsy_frame = pd.read_csv(refreshed_root / "cnlsy" / "panel_extract.csv", usecols=cnlsy_usecols).rename(
        columns={"C0000100": "child_id", "C0000200": "mother_id", **cnlsy_map}
    )

    backbone = base_backbone.merge(nlsy79_frame, on="respondent_id", how="left", validate="many_to_one")
    backbone = backbone.merge(cnlsy_frame, on=["child_id", "mother_id"], how="left", validate="one_to_one")
    backbone_path = processed_root / "nlsy79_cnlsy_backbone_treatment_candidates.parquet"
    backbone.to_parquet(backbone_path, index=False)

    nlsy97_usecols = ["R0000100", *nlsy97_map.keys()]
    nlsy97_base_path = processed_root / "nlsy97_reviewed.parquet"
    base_nlsy97 = pd.read_parquet(nlsy97_base_path) if nlsy97_base_path.exists() else pd.DataFrame()
    nlsy97_frame = pd.read_csv(refreshed_root / "nlsy97" / "panel_extract.csv", usecols=nlsy97_usecols).rename(
        columns={"R0000100": "respondent_id", **nlsy97_map}
    )
    nlsy97 = (
        base_nlsy97.merge(nlsy97_frame, on="respondent_id", how="left", validate="one_to_one")
        if not base_nlsy97.empty
        else nlsy97_frame
    )
    nlsy97_path = processed_root / "nlsy97_treatment_candidates.parquet"
    nlsy97.to_parquet(nlsy97_path, index=False)

    mapping_rows = [
        {
            "cohort": row["cohort"],
            "source_file": row["source_file"],
            "refnum": row["refnum"],
            "raw_column": row["raw_column"],
            "canonical_name": row["canonical_name"],
        }
        for row in candidates
    ]
    mapping_path = output_dir / "nlsy_treatment_candidate_mapping.csv"
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    value_count_rows: list[dict[str, Any]] = []
    for frame_name, frame in (("backbone", backbone), ("nlsy97", nlsy97)):
        for row in candidates:
            canonical_name = row["canonical_name"]
            if canonical_name not in frame.columns:
                continue
            counts = frame[canonical_name].value_counts(dropna=False).head(10)
            for value, count in counts.items():
                value_count_rows.append(
                    {
                        "frame": frame_name,
                        "canonical_name": canonical_name,
                        "value": "" if pd.isna(value) else value,
                        "count": int(count),
                    }
                )
    value_counts_path = output_dir / "nlsy_treatment_candidate_value_counts.csv"
    pd.DataFrame(value_count_rows).to_csv(value_counts_path, index=False)

    return TreatmentLayerBuildResult(
        backbone_path=backbone_path,
        nlsy97_path=nlsy97_path,
        mapping_path=mapping_path,
        value_counts_path=value_counts_path,
    )


SPECIAL_MISSING_CODES: frozenset[int] = frozenset({-1, -2, -3, -4, -5, -7})
NLSY97_NEAR_EXIT_YEARS: tuple[int, ...] = tuple(range(1998, 2006))


def _clean_special_missing(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(~numeric.isin(SPECIAL_MISSING_CODES))


def _nlsy97_year_code(year: int) -> str:
    return f"{year - 1980:02d}"


def _nlsy97_annual_refnum(prefix: str, year: int) -> str:
    return f"{prefix}{_nlsy97_year_code(year)}00"


def _nlsy97_monthly_refnums(prefix: str, year: int) -> list[str]:
    year_code = _nlsy97_year_code(year)
    return [f"{prefix}{year_code}{month:02d}" for month in range(1, 13)]


def build_analysis_ready_treatment_layers(
    *,
    processed_root: Path,
    output_dir: Path,
) -> AnalysisReadyTreatmentResult:
    processed_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone = pd.read_parquet(processed_root / "nlsy79_cnlsy_backbone_treatment_candidates.parquet").copy()
    nlsy97 = pd.read_parquet(processed_root / "nlsy97_treatment_candidates.parquet").copy()

    nlsy97["resident_bio_father_present_1997_clean"] = _clean_special_missing(nlsy97["resident_bio_father_present_1997"])
    nlsy97["bio_father_contact_ever_1997_clean"] = _clean_special_missing(nlsy97["bio_father_contact_ever_1997"])
    nlsy97["bio_father_alive_1997_clean"] = _clean_special_missing(nlsy97["bio_father_alive_1997"])
    nlsy97["resident_bio_father_absent_1997"] = nlsy97["resident_bio_father_present_1997_clean"].map({1.0: 0, 0.0: 1})
    nlsy97["bio_father_deceased_1997"] = nlsy97["bio_father_alive_1997_clean"].map({1.0: 0, 2.0: 1})
    nlsy97["primary_treatment_nlsy97"] = nlsy97["resident_bio_father_absent_1997"]
    nlsy97["primary_treatment_observed_nlsy97"] = nlsy97["primary_treatment_nlsy97"].notna().astype(int)
    nlsy97["primary_treatment_label_nlsy97"] = nlsy97["primary_treatment_nlsy97"].map(
        {0.0: "resident_bio_father_present", 1.0: "resident_bio_father_absent"}
    )

    absence_type = pd.Series(pd.NA, index=nlsy97.index, dtype="object")
    resident_present = nlsy97["resident_bio_father_present_1997_clean"]
    contact = nlsy97["bio_father_contact_ever_1997_clean"]
    alive = nlsy97["bio_father_alive_1997_clean"]
    absence_type.loc[resident_present == 1] = "resident_bio_father_present"
    absence_type.loc[(resident_present == 0) & (alive == 2)] = "absent_deceased"
    absence_type.loc[(resident_present == 0) & (alive == 1) & (contact == 1)] = "absent_alive_contact"
    absence_type.loc[(resident_present == 0) & (alive == 1) & (contact == 0)] = "absent_alive_no_contact"
    absence_type.loc[(resident_present == 0) & absence_type.isna()] = "absent_other_or_missing_detail"
    nlsy97["father_absence_type_1997"] = absence_type

    backbone["child_age_1990"] = 1990 - pd.to_numeric(backbone["child_birth_year"], errors="coerce")
    backbone["home_0_2_eligible_1990"] = backbone["child_age_1990"].between(0, 2, inclusive="both")
    backbone["child_ever_sees_father_figure_1990_clean"] = _clean_special_missing(backbone["child_ever_sees_father_figure_1990"])
    backbone["father_figure_type_1990_clean"] = _clean_special_missing(backbone["father_figure_type_1990"])
    backbone["child_sees_father_figure_daily_1990_clean"] = _clean_special_missing(backbone["child_sees_father_figure_daily_1990"])

    presence_type = pd.Series(pd.NA, index=backbone.index, dtype="object")
    eligible = backbone["home_0_2_eligible_1990"] == True
    figure_type = backbone["father_figure_type_1990_clean"]
    seen_any = backbone["child_ever_sees_father_figure_1990_clean"]
    presence_type.loc[eligible & (figure_type == 1)] = "biological_father"
    presence_type.loc[eligible & (figure_type == 2)] = "step_father"
    presence_type.loc[eligible & (figure_type == 3)] = "father_figure"
    presence_type.loc[eligible & ((figure_type == 4) | (seen_any == 0))] = "none"
    backbone["early_father_figure_presence_type_1990"] = presence_type
    backbone["early_father_figure_present_1990"] = presence_type.map(
        {
            "biological_father": 1,
            "step_father": 1,
            "father_figure": 1,
            "none": 0,
        }
    )
    backbone["primary_treatment_cnlsy_1990"] = backbone["early_father_figure_present_1990"].map({1.0: 0, 0.0: 1})
    backbone["primary_treatment_label_cnlsy_1990"] = backbone["primary_treatment_cnlsy_1990"].map(
        {0.0: "father_figure_present", 1.0: "no_father_figure_present"}
    )
    backbone["early_father_figure_daily_1990"] = backbone["child_sees_father_figure_daily_1990_clean"].map(
        {1.0: 1, 0.0: 0, 2.0: 0}
    )

    latest_alive = _clean_special_missing(backbone["biological_father_alive_h60"])
    latest_alive = latest_alive.fillna(_clean_special_missing(backbone["biological_father_alive_h50"]))
    latest_alive = latest_alive.fillna(_clean_special_missing(backbone["biological_father_alive_h40"]))
    backbone["maternal_biological_father_alive_latest"] = latest_alive.map({1.0: 1, 0.0: 0})

    for column in ("education_years", "annual_earnings_2021", "household_income_2021", "net_worth"):
        if column in nlsy97.columns:
            nlsy97[f"{column}_clean"] = _clean_special_missing(nlsy97[column])
    for column in (
        "age_2014",
        "education_years_2014",
        "annual_earnings_2014",
        "family_income_2014_best_est",
        "wage_income_2014_best_est",
        "family_income_2014",
        "wage_income_2014",
        "degree_2014",
    ):
        if column in backbone.columns:
            backbone[f"{column}_clean"] = _clean_special_missing(backbone[column])
    if "wage_income_2014_clean" not in backbone.columns:
        if "wage_income_2014_best_est_clean" in backbone.columns:
            backbone["wage_income_2014_clean"] = backbone["wage_income_2014_best_est_clean"]
        elif "wage_income_2014_best_est" in backbone.columns:
            backbone["wage_income_2014_clean"] = _clean_special_missing(backbone["wage_income_2014_best_est"])
    if "family_income_2014_clean" not in backbone.columns:
        if "family_income_2014_best_est_clean" in backbone.columns:
            backbone["family_income_2014_clean"] = backbone["family_income_2014_best_est_clean"]
        elif "family_income_2014_best_est" in backbone.columns:
            backbone["family_income_2014_clean"] = _clean_special_missing(backbone["family_income_2014_best_est"])
    if "employment_2014" in backbone.columns:
        employment = pd.to_numeric(backbone["employment_2014"], errors="coerce")
        employment = employment.mask(employment.isin([1, 2]), 1).mask(employment == 3, 0)
        backbone["employment_2014_clean"] = _clean_special_missing(employment)
    if "education_years_2014_clean" in backbone.columns:
        backbone["education_attainment_code_2014"] = backbone["education_years_2014_clean"]
        backbone["education_attainment_label_2014"] = (
            pd.to_numeric(backbone["education_attainment_code_2014"], errors="coerce")
            .astype("Int64")
            .map(CNLSY_2014_ATTAINMENT_LABELS["education_attainment_code_2014"])
            .astype("string")
        )
    if "degree_2014_clean" in backbone.columns:
        backbone["degree_attainment_code_2014"] = backbone["degree_2014_clean"]
        backbone["degree_attainment_label_2014"] = (
            pd.to_numeric(backbone["degree_attainment_code_2014"], errors="coerce")
            .astype("Int64")
            .map(CNLSY_2014_ATTAINMENT_LABELS["degree_attainment_code_2014"])
            .astype("string")
        )
    if "age_2014_clean" in backbone.columns:
        backbone["adult_outcome_eligible_2014"] = backbone["age_2014_clean"].ge(24).fillna(False)
    else:
        backbone["adult_outcome_eligible_2014"] = False

    analysis_backbone_path = processed_root / "nlsy79_cnlsy_backbone_analysis_ready.parquet"
    analysis_nlsy97_path = processed_root / "nlsy97_analysis_ready.parquet"
    backbone.to_parquet(analysis_backbone_path, index=False)
    nlsy97.to_parquet(analysis_nlsy97_path, index=False)

    nlsy97_baseline = (
        nlsy97.groupby("father_absence_type_1997", dropna=False)
        .agg(
            n=("respondent_id", "count"),
            education_years_mean=("education_years_clean", "mean"),
            annual_earnings_2021_mean=("annual_earnings_2021_clean", "mean"),
            household_income_2021_mean=("household_income_2021_clean", "mean"),
            net_worth_mean=("net_worth_clean", "mean"),
        )
        .reset_index()
    )
    nlsy97_baseline_path = output_dir / "nlsy97_father_absence_baseline.csv"
    nlsy97_baseline.to_csv(nlsy97_baseline_path, index=False)

    nlsy97_primary = (
        nlsy97.loc[nlsy97["primary_treatment_observed_nlsy97"] == 1]
        .groupby("primary_treatment_label_nlsy97", dropna=False)
        .agg(
            n=("respondent_id", "count"),
            education_years_mean=("education_years_clean", "mean"),
            annual_earnings_2021_mean=("annual_earnings_2021_clean", "mean"),
            household_income_2021_mean=("household_income_2021_clean", "mean"),
            net_worth_mean=("net_worth_clean", "mean"),
        )
        .reset_index()
    )
    nlsy97_primary_baseline_path = output_dir / "nlsy97_primary_treatment_baseline.csv"
    nlsy97_primary.to_csv(nlsy97_primary_baseline_path, index=False)

    cnlsy_baseline_frame = backbone.loc[
        (backbone["home_0_2_eligible_1990"] == True)
        & (backbone["adult_outcome_eligible_2014"] == True)
        & (backbone["early_father_figure_presence_type_1990"].notna())
    ].copy()
    cnlsy_subset_path = processed_root / "cnlsy_early_childhood_adult_2014_subset.parquet"
    cnlsy_baseline_frame.to_parquet(cnlsy_subset_path, index=False)

    def _cnlsy_group_summary(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
        return (
            frame.groupby(group_col, dropna=False)
            .agg(
                n=("child_id", "count"),
                age_2014_mean=("age_2014_clean", "mean"),
                employment_2014_rate=("employment_2014_clean", "mean"),
                annual_earnings_2014_mean=("annual_earnings_2014_clean", "mean"),
                wage_income_2014_mean=("wage_income_2014_clean", "mean"),
                family_income_2014_mean=("family_income_2014_clean", "mean"),
                n_employment_2014_obs=("employment_2014_clean", lambda s: int(s.notna().sum())),
                n_annual_earnings_2014_obs=("annual_earnings_2014_clean", lambda s: int(s.notna().sum())),
                n_wage_income_2014_obs=("wage_income_2014_clean", lambda s: int(s.notna().sum())),
                n_family_income_2014_obs=("family_income_2014_clean", lambda s: int(s.notna().sum())),
            )
            .reset_index()
        )

    cnlsy_baseline = _cnlsy_group_summary(cnlsy_baseline_frame, "early_father_figure_presence_type_1990")
    cnlsy_baseline_path = output_dir / "cnlsy_father_figure_baseline.csv"
    cnlsy_baseline.to_csv(cnlsy_baseline_path, index=False)

    cnlsy_primary = _cnlsy_group_summary(
        cnlsy_baseline_frame.loc[cnlsy_baseline_frame["primary_treatment_label_cnlsy_1990"].notna()],
        "primary_treatment_label_cnlsy_1990",
    )
    cnlsy_primary_baseline_path = output_dir / "cnlsy_primary_treatment_baseline.csv"
    cnlsy_primary.to_csv(cnlsy_primary_baseline_path, index=False)

    baseline_n = len(cnlsy_baseline_frame.index)
    def _observed_rows(frame: pd.DataFrame, column: str) -> int:
        return int(frame[column].notna().sum()) if column in frame.columns else 0

    def _coverage_rate(frame: pd.DataFrame, column: str) -> float:
        if baseline_n == 0 or column not in frame.columns:
            return 0.0
        return float(frame[column].notna().mean())

    cnlsy_readiness = pd.DataFrame(
        [
            {
                "metric": "eligible_home_0_2_rows",
                "count": int((backbone["home_0_2_eligible_1990"] == True).sum()),
            },
            {
                "metric": "adult_outcome_eligible_rows",
                "count": int(((backbone["home_0_2_eligible_1990"] == True) & (backbone["adult_outcome_eligible_2014"] == True)).sum()),
            },
            {
                "metric": "adult_outcome_eligible_rows_with_observed_treatment",
                "count": int(
                    (
                        (backbone["home_0_2_eligible_1990"] == True)
                        & (backbone["adult_outcome_eligible_2014"] == True)
                        & (backbone["primary_treatment_label_cnlsy_1990"].notna())
                    ).sum()
                ),
            },
            {
                "metric": "baseline_rows_used",
                "count": int(baseline_n),
            },
            {
                "metric": "employment_2014_observed_rows",
                "count": _observed_rows(cnlsy_baseline_frame, "employment_2014_clean"),
            },
            {
                "metric": "annual_earnings_2014_observed_rows",
                "count": _observed_rows(cnlsy_baseline_frame, "annual_earnings_2014_clean"),
            },
            {
                "metric": "education_years_2014_observed_rows",
                "count": _observed_rows(cnlsy_baseline_frame, "education_years_2014_clean"),
            },
            {
                "metric": "wage_income_2014_observed_rows",
                "count": _observed_rows(cnlsy_baseline_frame, "wage_income_2014_clean"),
            },
            {
                "metric": "family_income_2014_observed_rows",
                "count": _observed_rows(cnlsy_baseline_frame, "family_income_2014_clean"),
            },
        ]
    )
    cnlsy_readiness_path = output_dir / "cnlsy_treatment_outcome_readiness.csv"
    cnlsy_readiness.to_csv(cnlsy_readiness_path, index=False)

    cnlsy_outcome_tiering = pd.DataFrame(
        [
            {
                "metric": "employment_2014",
                "observed_rows": _observed_rows(cnlsy_baseline_frame, "employment_2014_clean"),
                "coverage_rate": _coverage_rate(cnlsy_baseline_frame, "employment_2014_clean"),
                "tier": "primary_baseline",
                "note": "Adult-status outcome retained despite thinner coverage because it is substantively central.",
            },
            {
                "metric": "annual_earnings_2014",
                "observed_rows": _observed_rows(cnlsy_baseline_frame, "annual_earnings_2014_clean"),
                "coverage_rate": _coverage_rate(cnlsy_baseline_frame, "annual_earnings_2014_clean"),
                "tier": "primary_baseline",
                "note": "Retained as the first-pass earnings outcome with explicit observed-row counts.",
            },
            {
                "metric": "education_years_2014",
                "observed_rows": _observed_rows(cnlsy_baseline_frame, "education_years_2014_clean"),
                "coverage_rate": _coverage_rate(cnlsy_baseline_frame, "education_years_2014_clean"),
                "tier": "officially_labeled_attainment_code",
                "note": "Fully observed 1-14 attainment code range with official 2014 CNLSY code labels attached; do not treat as literal years.",
            },
            {
                "metric": "degree_2014",
                "observed_rows": _observed_rows(cnlsy_baseline_frame, "degree_2014_clean"),
                "coverage_rate": _coverage_rate(cnlsy_baseline_frame, "degree_2014_clean"),
                "tier": "officially_labeled_attainment_code",
                "note": "Fully observed 0-8 attainment code range with official 2014 CNLSY degree labels attached; keep as a code rather than a baseline mean.",
            },
            {
                "metric": "wage_income_2014",
                "observed_rows": _observed_rows(cnlsy_baseline_frame, "wage_income_2014_clean"),
                "coverage_rate": _coverage_rate(cnlsy_baseline_frame, "wage_income_2014_clean"),
                "tier": "supplementary_sparse",
                "note": "Sparse coverage; keep visible for diagnostics but not as a primary outcome.",
            },
            {
                "metric": "family_income_2014",
                "observed_rows": _observed_rows(cnlsy_baseline_frame, "family_income_2014_clean"),
                "coverage_rate": _coverage_rate(cnlsy_baseline_frame, "family_income_2014_clean"),
                "tier": "supplementary_sparse",
                "note": "Sparse coverage; useful for sensitivity checks only.",
            },
        ]
    )
    cnlsy_outcome_tiering_path = output_dir / "cnlsy_outcome_tiering.csv"
    cnlsy_outcome_tiering.to_csv(cnlsy_outcome_tiering_path, index=False)

    education_code = cnlsy_baseline_frame["education_attainment_code_2014"] if "education_attainment_code_2014" in cnlsy_baseline_frame.columns else pd.Series(dtype="float64")
    degree_code = cnlsy_baseline_frame["degree_attainment_code_2014"] if "degree_attainment_code_2014" in cnlsy_baseline_frame.columns else pd.Series(dtype="float64")
    monotone_pairs = pd.DataFrame({"education_code": education_code, "degree_code": degree_code}).dropna()
    monotone_share = float(
        (
            monotone_pairs.sort_values(["education_code", "degree_code"])
            .groupby("education_code")["degree_code"]
            .agg(["min", "max"])
            .assign(monotone=lambda x: x["max"].cummax().eq(x["max"]))
            ["monotone"]
            .mean()
        )
    ) if not monotone_pairs.empty else np.nan
    cnlsy_education_validation = pd.DataFrame(
        [
            {
                "metric": "education_code_observed_rows",
                "value": int(education_code.notna().sum()) if not education_code.empty else 0,
                "note": "Observed rows for Y3332100-derived education code in the restricted CNLSY subset.",
            },
            {
                "metric": "education_code_min",
                "value": float(education_code.min()) if not education_code.empty else np.nan,
                "note": "Observed minimum code; literal adult years-of-schooling would not usually bottom out at 1 in this adult-eligible subset.",
            },
            {
                "metric": "education_code_max",
                "value": float(education_code.max()) if not education_code.empty else np.nan,
                "note": "Observed maximum code in the current restricted subset.",
            },
            {
                "metric": "education_code_n_unique",
                "value": int(education_code.nunique(dropna=True)) if not education_code.empty else 0,
                "note": "Distinct observed values for the Y3332100-derived field.",
            },
            {
                "metric": "degree_code_observed_rows",
                "value": int(degree_code.notna().sum()) if not degree_code.empty else 0,
                "note": "Observed rows for Y3332200-derived degree code in the restricted CNLSY subset.",
            },
            {
                "metric": "degree_code_min",
                "value": float(degree_code.min()) if not degree_code.empty else np.nan,
                "note": "Observed minimum degree code.",
            },
            {
                "metric": "degree_code_max",
                "value": float(degree_code.max()) if not degree_code.empty else np.nan,
                "note": "Observed maximum degree code.",
            },
            {
                "metric": "education_degree_monotone_share",
                "value": monotone_share,
                "note": "Share of education-code strata whose paired degree-code maxima follow a non-decreasing ordered pattern; supports ordered-code interpretation rather than literal-year interpretation.",
            },
            {
                "metric": "prior_project_evidence",
                "value": np.nan,
                "note": "Local prior-project config/tests also treated Y3332100 and Y3332200 as paired adult attainment fields, but downstream summaries reported means around 2.07 for the education field, reinforcing that it behaves like a code scale rather than literal schooling years.",
            },
        ]
    )
    cnlsy_education_validation_path = output_dir / "cnlsy_education_code_validation.csv"
    cnlsy_education_validation.to_csv(cnlsy_education_validation_path, index=False)

    cnlsy_education_crosstab = pd.crosstab(
        education_code,
        degree_code,
        dropna=False,
    )
    cnlsy_education_crosstab_path = output_dir / "cnlsy_education_degree_crosstab.csv"
    cnlsy_education_crosstab.to_csv(cnlsy_education_crosstab_path)

    cnlsy_attainment_codebook_rows: list[dict[str, object]] = []
    for field_name, label_map in CNLSY_2014_ATTAINMENT_LABELS.items():
        raw_column = "Y3332100" if field_name == "education_attainment_code_2014" else "Y3332200"
        question_label = (
            "HIGHEST GRADE COMPLETED AS OF 2014 2014"
            if field_name == "education_attainment_code_2014"
            else "HIGHEST ACADEMIC DEGREE RECEIVED AS OF 2014 2014"
        )
        format_name = "vx74499f" if field_name == "education_attainment_code_2014" else "vx74500f"
        for code, label in label_map.items():
            cnlsy_attainment_codebook_rows.append(
                {
                    "field_name": field_name,
                    "raw_column": raw_column,
                    "question_label": question_label,
                    "format_name": format_name,
                    "code": code,
                    "label": label,
                    "source_note": "Recovered from the local CNLSY SAS export spec.",
                }
            )
    cnlsy_attainment_codebook = pd.DataFrame(cnlsy_attainment_codebook_rows)
    cnlsy_attainment_codebook_path = output_dir / "cnlsy_attainment_codebook_2014.csv"
    cnlsy_attainment_codebook.to_csv(cnlsy_attainment_codebook_path, index=False)

    summary_path = output_dir / "nlsy_treatment_summary.md"
    summary_lines = [
        "# NLSY Treatment Summary",
        "",
        "## Primary NLSY97 treatment",
        "",
        f"- Observed treatment rows: {int(nlsy97['primary_treatment_observed_nlsy97'].sum())}",
        f"- Resident biological father present: {int((nlsy97['primary_treatment_nlsy97'] == 0).sum())}",
        f"- Resident biological father absent: {int((nlsy97['primary_treatment_nlsy97'] == 1).sum())}",
        "",
        "The first-pass primary treatment is locked as `resident_bio_father_absent_1997` from `R03356.00`.",
        "",
        "## CNLSY early-childhood subset",
        "",
        f"- HOME 0-2 eligible rows: {int((backbone['home_0_2_eligible_1990'] == True).sum())}",
        f"- Adult-outcome eligible rows in 2014 (age >= 24): {int(((backbone['home_0_2_eligible_1990'] == True) & (backbone['adult_outcome_eligible_2014'] == True)).sum())}",
        f"- Adult-outcome eligible rows with observed treatment: {int(cnlsy_baseline_frame['primary_treatment_label_cnlsy_1990'].notna().sum())}",
        f"- Rows used in CNLSY baseline table: {int(baseline_n)}",
        "",
        "CNLSY baseline tables are restricted to the 1990 age-0-to-2 exposure subset with adult-eligible 2014 outcomes.",
        "Employment and annual earnings remain the first-pass baseline outcomes.",
        "Education-like fields are fully observed and are now attached to official 2014 CNLSY code labels in `cnlsy_attainment_codebook_2014.csv`, but they remain ordered attainment codes rather than literal years-of-schooling; see `cnlsy_education_code_validation.csv` and `cnlsy_education_degree_crosstab.csv`.",
        "Wage-income and family-income style fields remain supplementary and are tiered separately in `cnlsy_outcome_tiering.csv` because their observed coverage is thin.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    coding_rules_path = output_dir / "nlsy_treatment_coding_rules.md"
    coding_lines = [
        "# NLSY Treatment Coding Rules",
        "",
        "## NLSY97",
        "- `resident_bio_father_absent_1997 = 1` when `resident_bio_father_present_1997 == 0`; `0` when `== 1`.",
        "- `primary_treatment_nlsy97` is locked to `resident_bio_father_absent_1997` for the first-pass treatment definition.",
        "- `bio_father_deceased_1997 = 1` when `bio_father_alive_1997 == 2`; `0` when `== 1`.",
        "- `father_absence_type_1997` prioritizes: resident present, absent deceased, absent alive with contact, absent alive with no contact, then residual missing-detail.",
        "",
        "## CNLSY",
        "- `home_0_2_eligible_1990 = 1` when `1990 - child_birth_year` is between 0 and 2 inclusive.",
        "- `adult_outcome_eligible_2014 = 1` when cleaned `age_2014 >= 24`.",
        "- `early_father_figure_presence_type_1990` maps `father_figure_type_1990`: `1=biological_father`, `2=step_father`, `3=father_figure`, `4=none`.",
        "- `early_father_figure_present_1990 = 1` for biological/step/father-figure, `0` for none.",
        "- `employment_2014_clean` follows the prior project recode: `1/2 -> employed`, `3 -> not employed`.",
        "- CNLSY baseline tables use adult-eligible rows only and keep `employment_2014` plus `annual_earnings_2014` as the first-pass interpretable outcomes.",
        "- `education_years_2014` and `degree_2014` are aliased as `education_attainment_code_2014` and `degree_attainment_code_2014`, with official code meanings attached in `cnlsy_attainment_codebook_2014.csv` and row-level labels materialized as `education_attainment_label_2014` and `degree_attainment_label_2014`.",
        "- `wage_income_2014` and `family_income_2014` remain supplementary sparse outcomes and are explicitly tiered in `cnlsy_outcome_tiering.csv`.",
    ]
    coding_rules_path.write_text("\n".join(coding_lines) + "\n", encoding="utf-8")

    return AnalysisReadyTreatmentResult(
        backbone_path=analysis_backbone_path,
        nlsy97_path=analysis_nlsy97_path,
        nlsy97_baseline_path=nlsy97_baseline_path,
        nlsy97_primary_baseline_path=nlsy97_primary_baseline_path,
        cnlsy_subset_path=cnlsy_subset_path,
        cnlsy_baseline_path=cnlsy_baseline_path,
        cnlsy_primary_baseline_path=cnlsy_primary_baseline_path,
        cnlsy_readiness_path=cnlsy_readiness_path,
        cnlsy_outcome_tiering_path=cnlsy_outcome_tiering_path,
        cnlsy_education_validation_path=cnlsy_education_validation_path,
        cnlsy_education_crosstab_path=cnlsy_education_crosstab_path,
        cnlsy_attainment_codebook_path=cnlsy_attainment_codebook_path,
        coding_rules_path=coding_rules_path,
        summary_path=summary_path,
    )


def _nlsy97_sex_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map({1.0: "male", 2.0: "female"}).astype("string")


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


def build_nlsy97_fatherlessness_profiles(
    *,
    processed_root: Path,
    output_dir: Path,
) -> FatherlessnessProfileResult:
    processed_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(processed_root / "nlsy97_analysis_ready.parquet").copy()
    frame = frame.loc[frame["primary_treatment_nlsy97"].notna()].copy()
    frame["sex"] = _nlsy97_sex_label(frame["sex_raw"])
    frame["parent_education_clean"] = _clean_special_missing(frame["parent_education"])
    frame["mother_education_clean"] = _clean_special_missing(frame["mother_education"])
    frame["father_education_clean"] = _clean_special_missing(frame["father_education"])
    frame["parent_education_band"] = _education_band(frame["parent_education"])
    frame["mother_education_band"] = _education_band(frame["mother_education"])
    frame["father_education_band"] = _education_band(frame["father_education"])
    frame["birth_year_clean"] = _clean_special_missing(frame["birth_year"])
    frame["sex_x_race_ethnicity"] = (
        frame["sex"].fillna("missing") + " | " + frame["race_ethnicity_3cat"].astype("string").fillna("missing")
    )
    frame["primary_treatment_nlsy97"] = pd.to_numeric(frame["primary_treatment_nlsy97"], errors="coerce")

    def _group_summary(group_col: str, *, value_label: str | None = None) -> pd.DataFrame:
        label = value_label or group_col
        grouped = (
            frame.groupby(group_col, dropna=False)
            .agg(
                n=("respondent_id", "count"),
                fatherlessness_rate=("primary_treatment_nlsy97", "mean"),
                mother_education_mean=("mother_education_clean", "mean"),
                father_education_mean=("father_education_clean", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "group_type", label)
        grouped = grouped.rename(columns={group_col: "group_value"})
        grouped["father_present_rate"] = 1 - grouped["fatherlessness_rate"]
        return grouped

    summary_frames = [
        pd.DataFrame(
            [
                {
                    "group_type": "overall",
                    "group_value": "overall",
                    "n": int(len(frame.index)),
                    "fatherlessness_rate": float(frame["primary_treatment_nlsy97"].mean()),
                    "mother_education_mean": float(frame["mother_education_clean"].mean()),
                    "father_education_mean": float(frame["father_education_clean"].mean()),
                    "father_present_rate": float(1 - frame["primary_treatment_nlsy97"].mean()),
                }
            ]
        ),
        _group_summary("sex"),
        _group_summary("race_ethnicity_3cat"),
        _group_summary("sex_x_race_ethnicity"),
        _group_summary("parent_education_band"),
        _group_summary("mother_education_band"),
        _group_summary("father_education_band"),
        _group_summary("birth_year_clean", value_label="birth_year"),
    ]
    group_summary = pd.concat(summary_frames, ignore_index=True)
    group_summary_path = output_dir / "nlsy97_fatherlessness_group_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)

    predictor_path = output_dir / "nlsy97_fatherlessness_predictors.csv"
    predictor_note = "Logit unavailable"
    predictor_frame = pd.DataFrame(
        columns=["term", "coefficient", "std_error", "p_value", "odds_ratio", "model", "n"]
    )
    try:
        import statsmodels.api as sm

        model_frame = frame[
            [
                "primary_treatment_nlsy97",
                "sex",
                "race_ethnicity_3cat",
                "mother_education_clean",
                "father_education_clean",
                "birth_year_clean",
            ]
        ].copy()
        model_frame["mother_education_missing"] = model_frame["mother_education_clean"].isna().astype(int)
        model_frame["father_education_missing"] = model_frame["father_education_clean"].isna().astype(int)
        mother_median = float(model_frame["mother_education_clean"].median())
        father_median = float(model_frame["father_education_clean"].median())
        birth_year_mean = float(model_frame["birth_year_clean"].mean())
        model_frame["mother_education_filled"] = model_frame["mother_education_clean"].fillna(mother_median)
        model_frame["father_education_filled"] = model_frame["father_education_clean"].fillna(father_median)
        model_frame["birth_year_centered"] = model_frame["birth_year_clean"] - birth_year_mean
        design = pd.get_dummies(
            model_frame[["sex", "race_ethnicity_3cat"]],
            prefix=["sex", "race"],
            drop_first=True,
            dtype=float,
        )
        design["mother_education_filled"] = model_frame["mother_education_filled"].astype(float)
        design["father_education_filled"] = model_frame["father_education_filled"].astype(float)
        design["mother_education_missing"] = model_frame["mother_education_missing"].astype(float)
        design["father_education_missing"] = model_frame["father_education_missing"].astype(float)
        design["birth_year_centered"] = model_frame["birth_year_centered"].astype(float)
        design = sm.add_constant(design, has_constant="add")
        fit = sm.Logit(model_frame["primary_treatment_nlsy97"].astype(float), design).fit(disp=False, cov_type="HC1")
        predictor_frame = pd.DataFrame(
            {
                "term": fit.params.index,
                "coefficient": fit.params.values,
                "std_error": fit.bse.values,
                "p_value": fit.pvalues.values,
                "odds_ratio": np.exp(fit.params.values),
                "model": "logit_hc1",
                "n": int(model_frame.shape[0]),
            }
        )
        predictor_note = (
            "Descriptive logit with sex, race/ethnicity, mother education, father education, and birth year."
        )
    except Exception as exc:  # pragma: no cover - fallback only
        predictor_note = f"Predictor model not estimated: {exc}"
    predictor_frame.to_csv(predictor_path, index=False)

    race_rows = group_summary.loc[group_summary["group_type"] == "race_ethnicity_3cat"].copy()
    race_rows = race_rows.sort_values("fatherlessness_rate", ascending=False)
    top_race = race_rows.iloc[0] if not race_rows.empty else None
    low_race = race_rows.iloc[-1] if len(race_rows.index) > 1 else None
    sex_rows = group_summary.loc[group_summary["group_type"] == "sex"].copy()
    edu_rows = group_summary.loc[group_summary["group_type"] == "parent_education_band"].copy()
    edu_rows = edu_rows.loc[edu_rows["group_value"] != "missing"].sort_values("fatherlessness_rate", ascending=False)
    top_edu = edu_rows.iloc[0] if not edu_rows.empty else None
    mother_rows = group_summary.loc[group_summary["group_type"] == "mother_education_band"].copy()
    mother_rows = mother_rows.loc[mother_rows["group_value"] != "missing"].sort_values(
        "fatherlessness_rate", ascending=False
    )
    top_mother = mother_rows.iloc[0] if not mother_rows.empty else None
    father_rows = group_summary.loc[group_summary["group_type"] == "father_education_band"].copy()
    father_rows = father_rows.loc[father_rows["group_value"] != "missing"].sort_values(
        "fatherlessness_rate", ascending=False
    )
    top_father = father_rows.iloc[0] if not father_rows.empty else None

    report_lines = [
        "# NLSY97 Fatherlessness Profiles",
        "",
        "These outputs are descriptive profiles of the locked first-pass `resident_bio_father_absent_1997` treatment, not causal estimates.",
        "",
        f"- Observed treatment rows: {len(frame.index)}",
        f"- Overall fatherlessness rate: {frame['primary_treatment_nlsy97'].mean():.4f}",
        f"- Female fatherlessness rate: {sex_rows.loc[sex_rows['group_value'] == 'female', 'fatherlessness_rate'].iloc[0]:.4f}" if "female" in set(sex_rows["group_value"]) else "- Female fatherlessness rate: unavailable",
        f"- Male fatherlessness rate: {sex_rows.loc[sex_rows['group_value'] == 'male', 'fatherlessness_rate'].iloc[0]:.4f}" if "male" in set(sex_rows["group_value"]) else "- Male fatherlessness rate: unavailable",
        f"- Highest race/ethnicity fatherlessness rate: {top_race['group_value']} = {top_race['fatherlessness_rate']:.4f}" if top_race is not None else "- Highest race/ethnicity fatherlessness rate: unavailable",
        f"- Lowest race/ethnicity fatherlessness rate: {low_race['group_value']} = {low_race['fatherlessness_rate']:.4f}" if low_race is not None else "- Lowest race/ethnicity fatherlessness rate: unavailable",
        f"- Highest observed parent-education-band fatherlessness rate: {top_edu['group_value']} = {top_edu['fatherlessness_rate']:.4f}" if top_edu is not None else "- Parent education band rate: unavailable",
        f"- Highest observed mother-education-band fatherlessness rate: {top_mother['group_value']} = {top_mother['fatherlessness_rate']:.4f}" if top_mother is not None else "- Mother education band rate: unavailable",
        f"- Highest observed father-education-band fatherlessness rate: {top_father['group_value']} = {top_father['fatherlessness_rate']:.4f}" if top_father is not None else "- Father education band rate: unavailable",
        "",
        "Predictor model:",
        f"- {predictor_note}",
        f"- Group summary: `{group_summary_path.name}`",
        f"- Predictor CSV: `{predictor_path.name}`",
    ]
    report_path = output_dir / "nlsy97_fatherlessness_profiles.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return FatherlessnessProfileResult(
        group_summary_path=group_summary_path,
        predictor_path=predictor_path,
        report_path=report_path,
    )


def build_nlsy97_longitudinal_panel_scaffold(
    *,
    interim_root: Path,
    processed_root: Path,
    output_dir: Path,
) -> NLSY97LongitudinalPanelResult:
    processed_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_ready = pd.read_parquet(processed_root / "nlsy97_analysis_ready.parquet").copy()
    panel_extract_path = interim_root / "nlsy97" / "panel_extract.csv"
    base_usecols = [
        "E8013912",
        "E8023912",
        "R0000100",
        "R0360900",
        "R0361000",
        "R0361100",
        "R0361300",
        "R0361400",
        "R9705200",
        "R9705300",
        "R9705400",
        "R9705500",
        "R9705600",
        "R9705700",
        "R9705800",
        "R9705900",
        "R9706000",
        "R9706100",
        "R9706200",
        "R9706400",
        "R9706500",
        "R9706600",
        "R9706700",
        "R9706800",
        "R9706900",
        "R9707000",
        "R9707100",
        "R9707200",
        "R9707300",
        "R9707400",
        "R9708601",
        "R9708602",
        "T5206900",
        "T5229101",
        "T5229102",
        "T6680901",
        "T6680902",
        "T7295800",
        "T7311500",
        "T7311600",
        "T7311700",
        "T7311800",
        "T7311900",
        "T7635600",
        "T7635700",
        "T7635800",
        "T8154001",
        "T8154002",
        "T8821300",
        "T8821400",
        "T8821500",
        "T8821600",
        "T8821700",
        "T8821800",
        "U0036301",
        "U0036302",
        "U1032300",
        "U0741900",
        "U0742000",
        "U0742100",
        "U0742200",
        "U0742300",
        "U0742400",
        "U0742500",
        "U1876601",
        "U1876602",
        "U2679300",
        "U2679400",
        "U2679500",
        "U2679600",
        "U2679700",
        "U2679800",
        "U2679900",
        "U3438101",
        "U3444000",
        "U3455100",
        "U3475201",
        "U3475202",
        "U4114400",
        "U4114500",
        "U4114600",
        "U4114700",
        "U4114800",
        "U4114900",
        "U4115000",
        "U4282300",
        "U4285700",
        "U4949700",
        "U4958300",
        "U4976701",
        "U4976702",
        "U5072600",
        "U5591200",
        "U5591300",
        "U5591400",
        "U5591500",
        "U5591600",
        "U5753500",
        "U6365300",
        "U7239100",
        "U7239400",
        "U7239600",
        "U7239800",
        "U7239900",
        "Z9033700",
        "Z9033900",
        "Z9034100",
        "Z9073201",
        "Z9073400",
        "Z9074610",
        "Z9074612",
        "Z9083800",
        "Z9083410",
        "Z9083412",
        "Z9085400",
        "Z9123000",
        "Z9149100",
        "Z9165100",
    ]
    supplemental_usecols = ["R0000100"]
    for year in NLSY97_NEAR_EXIT_YEARS:
        supplemental_usecols.extend(_nlsy97_monthly_refnums("E501", year))
        supplemental_usecols.extend(_nlsy97_monthly_refnums("E511", year))
        supplemental_usecols.extend(_nlsy97_monthly_refnums("E801", year))
        supplemental_usecols.extend(_nlsy97_monthly_refnums("E802", year))
        if year >= 2000:
            supplemental_usecols.append(_nlsy97_annual_refnum("E026", year))
            supplemental_usecols.append(_nlsy97_annual_refnum("E028", year))
    available_usecols = pd.read_csv(panel_extract_path, nrows=0).columns.tolist()
    raw = pd.read_csv(
        panel_extract_path,
        usecols=[column for column in base_usecols if column in available_usecols],
    ).rename(columns={"R0000100": "respondent_id"})
    missing_supplemental = [column for column in supplemental_usecols if column not in raw.columns]
    nlsy97_full_raw_path = interim_root / "nlsy97" / "raw_files" / "nlsy97_all_1997-2023.csv"
    if nlsy97_full_raw_path.exists() and missing_supplemental:
        with nlsy97_full_raw_path.open(newline="", encoding="utf-8") as handle:
            full_available_usecols = next(csv.reader(handle))
        supplemental_columns = [column for column in missing_supplemental if column in full_available_usecols]
        if supplemental_columns:
            supplemental_raw = pa_csv.read_csv(
                nlsy97_full_raw_path,
                read_options=pa_csv.ReadOptions(block_size=1024 * 1024 * 128),
                convert_options=pa_csv.ConvertOptions(include_columns=supplemental_columns),
            ).to_pandas().rename(columns={"R0000100": "respondent_id"})
            raw = raw.merge(supplemental_raw, on="respondent_id", how="left", validate="one_to_one")

    analysis_cols = [
        column
        for column in [
            "respondent_id",
            "birth_year",
            "primary_treatment_nlsy97",
            "primary_treatment_label_nlsy97",
            "father_absence_type_1997",
            "resident_bio_father_absent_1997",
            "resident_bio_father_present_1997",
            "resident_bio_father_present_1997_clean",
            "bio_father_contact_ever_1997",
            "bio_father_contact_ever_1997_clean",
            "bio_father_alive_1997",
            "bio_father_alive_1997_clean",
            "lived_apart_from_bio_father_gt12m_1997",
            "ever_lived_with_bio_father_1997",
            "last_year_lived_with_bio_father",
            "last_month_lived_with_bio_father",
        ]
        if column in analysis_ready.columns
    ]
    merged = raw.merge(
        analysis_ready[analysis_cols],
        on="respondent_id",
        how="left",
        validate="one_to_one",
    )
    def _series_or_nan(column: str) -> pd.Series:
        if column in merged.columns:
            return merged[column]
        return pd.Series(np.nan, index=merged.index, dtype="float64")

    merged["birth_year_clean"] = pd.to_numeric(merged.get("birth_year"), errors="coerce")
    if "resident_bio_father_present_1997_clean" not in merged.columns:
        merged["resident_bio_father_present_1997_clean"] = _clean_special_missing(_series_or_nan("resident_bio_father_present_1997"))
    if "bio_father_contact_ever_1997_clean" not in merged.columns:
        merged["bio_father_contact_ever_1997_clean"] = _clean_special_missing(_series_or_nan("bio_father_contact_ever_1997"))
    if "bio_father_alive_1997_clean" not in merged.columns:
        merged["bio_father_alive_1997_clean"] = _clean_special_missing(_series_or_nan("bio_father_alive_1997"))
    merged["lived_apart_from_bio_father_gt12m_1997_clean"] = _clean_special_missing(_series_or_nan("lived_apart_from_bio_father_gt12m_1997"))
    merged["ever_lived_with_bio_father_1997_clean"] = _clean_special_missing(_series_or_nan("ever_lived_with_bio_father_1997"))
    merged["last_year_lived_with_bio_father_clean"] = _clean_special_missing(_series_or_nan("last_year_lived_with_bio_father"))
    merged["last_month_lived_with_bio_father_clean"] = _clean_special_missing(_series_or_nan("last_month_lived_with_bio_father"))

    history_rows: list[dict[str, object]] = []
    respondent_summaries: list[dict[str, object]] = []
    baseline_year = 1997
    for row in merged.itertuples(index=False):
        birth_year = pd.to_numeric(getattr(row, "birth_year_clean"), errors="coerce")
        resident_present = pd.to_numeric(getattr(row, "resident_bio_father_present_1997_clean"), errors="coerce")
        apart_gt12m = pd.to_numeric(getattr(row, "lived_apart_from_bio_father_gt12m_1997_clean"), errors="coerce")
        ever_lived = pd.to_numeric(getattr(row, "ever_lived_with_bio_father_1997_clean"), errors="coerce")
        last_year = pd.to_numeric(getattr(row, "last_year_lived_with_bio_father_clean"), errors="coerce")
        last_month = pd.to_numeric(getattr(row, "last_month_lived_with_bio_father_clean"), errors="coerce")
        alive = pd.to_numeric(getattr(row, "bio_father_alive_1997_clean"), errors="coerce")
        contact = pd.to_numeric(getattr(row, "bio_father_contact_ever_1997_clean"), errors="coerce")

        childhood_history_type = "missing_birth_year"
        first_absent_year = np.nan
        first_absent_month = np.nan
        if not pd.isna(birth_year):
            if resident_present == 0:
                if ever_lived == 0:
                    childhood_history_type = "absent_from_birth"
                elif not pd.isna(last_year):
                    first_absent_year = float(last_year if last_year == baseline_year else last_year + 1)
                    first_absent_month = float(last_month) if not pd.isna(last_month) else np.nan
                    childhood_history_type = "localized_exit_in_1997" if first_absent_year == baseline_year else "localized_exit_before_1997"
                elif ever_lived == 1:
                    childhood_history_type = "absent_with_unknown_exit_timing"
                else:
                    childhood_history_type = "absent_missing_retrospective_detail"
            elif resident_present == 1:
                if apart_gt12m == 0:
                    childhood_history_type = "stable_present_until_1997"
                elif apart_gt12m == 1:
                    childhood_history_type = "present_but_prior_gap_unlocalized"
                else:
                    childhood_history_type = "present_no_history_detail"
            else:
                childhood_history_type = "missing_baseline_presence"

        if pd.isna(birth_year):
            respondent_summaries.append(
                {
                    "respondent_id": getattr(row, "respondent_id"),
                    "birth_year_clean": birth_year,
                    "baseline_age_1997": np.nan,
                    "childhood_history_type": childhood_history_type,
                    "first_absent_year": first_absent_year,
                    "first_absent_month": first_absent_month,
                    "localized_exit_year_available": 0,
                    "childhood_rows": 0,
                    "childhood_observed_rows": 0,
                    "has_pre_post_childhood_observation": 0,
                }
            )
            continue

        observed_end_year = min(baseline_year, int(birth_year) + 17)
        baseline_age = observed_end_year - int(birth_year)
        observed_rows = 0
        observed_values: set[int] = set()
        for childhood_year in range(int(birth_year), observed_end_year + 1):
            childhood_age = childhood_year - int(birth_year)
            father_presence = np.nan
            source = "history_not_localized"
            if childhood_history_type == "stable_present_until_1997":
                father_presence = 1.0
                source = "baseline_present_and_no_gt12m_apart"
            elif childhood_history_type == "absent_from_birth":
                father_presence = 0.0
                source = "baseline_absent_never_lived_with_father"
            elif childhood_history_type == "localized_exit_before_1997":
                father_presence = 1.0 if childhood_year < first_absent_year else 0.0
                source = "localized_exit_year_from_last_coresidence"
            elif childhood_history_type == "localized_exit_in_1997":
                father_presence = 1.0 if childhood_year < baseline_year else 0.0
                source = "localized_exit_in_baseline_year"
            elif childhood_history_type == "absent_with_unknown_exit_timing":
                if childhood_year == baseline_year:
                    father_presence = 0.0
                    source = "baseline_absent_only"
            elif childhood_history_type in {"present_but_prior_gap_unlocalized", "present_no_history_detail"}:
                if childhood_year == baseline_year:
                    father_presence = 1.0
                    source = "baseline_present_only"

            if not pd.isna(father_presence):
                observed_rows += 1
                observed_values.add(int(father_presence))

            history_rows.append(
                {
                    "respondent_id": getattr(row, "respondent_id"),
                    "birth_year_clean": float(birth_year),
                    "baseline_age_1997": float(baseline_age),
                    "childhood_year": childhood_year,
                    "childhood_age": childhood_age,
                    "childhood_history_type": childhood_history_type,
                    "father_presence_imputed": father_presence,
                    "father_absence_imputed": 1.0 - father_presence if not pd.isna(father_presence) else np.nan,
                    "father_presence_observed": 0 if pd.isna(father_presence) else 1,
                    "childhood_history_source": source,
                    "resident_bio_father_absent_1997": getattr(row, "resident_bio_father_absent_1997"),
                    "father_absence_type_1997": getattr(row, "father_absence_type_1997"),
                    "bio_father_contact_ever_1997_clean": contact,
                    "bio_father_alive_1997_clean": alive,
                    "first_absent_year": first_absent_year,
                    "first_absent_month": first_absent_month,
                    "localized_exit_year_available": 1 if childhood_history_type.startswith("localized_exit_") else 0,
                    "absent_from_birth_flag": 1 if childhood_history_type == "absent_from_birth" else 0,
                    "event_time_from_first_absent_year": childhood_year - first_absent_year if not pd.isna(first_absent_year) else np.nan,
                }
            )

        respondent_summaries.append(
            {
                "respondent_id": getattr(row, "respondent_id"),
                "birth_year_clean": float(birth_year),
                "baseline_age_1997": float(baseline_age),
                "childhood_history_type": childhood_history_type,
                "first_absent_year": first_absent_year,
                "first_absent_month": first_absent_month,
                "localized_exit_year_available": 1 if childhood_history_type.startswith("localized_exit_") else 0,
                "childhood_rows": max(observed_end_year - int(birth_year) + 1, 0),
                "childhood_observed_rows": observed_rows,
                "has_pre_post_childhood_observation": 1 if observed_values == {0, 1} else 0,
            }
        )

    childhood_history = pd.DataFrame(history_rows)
    childhood_history_path = processed_root / "nlsy97_childhood_exposure_history.parquet"
    childhood_history.to_parquet(childhood_history_path, index=False)
    respondent_history = pd.DataFrame(respondent_summaries)
    respondent_history_index = respondent_history.set_index("respondent_id")

    merged["childhood_history_type"] = merged["respondent_id"].map(respondent_history_index["childhood_history_type"])
    merged["first_absent_year"] = merged["respondent_id"].map(respondent_history_index["first_absent_year"])
    merged["first_absent_month"] = merged["respondent_id"].map(respondent_history_index["first_absent_month"])
    merged["localized_exit_year_available"] = merged["respondent_id"].map(respondent_history_index["localized_exit_year_available"]).fillna(0)
    merged["event_time_candidate_year"] = merged["first_absent_year"].fillna(merged["last_year_lived_with_bio_father_clean"])

    panel_frames: list[pd.DataFrame] = []
    def _coalesce_clean_numeric(columns: list[str]) -> pd.Series:
        if not columns:
            return pd.Series(np.nan, index=merged.index, dtype="float64")
        result = pd.Series(np.nan, index=merged.index, dtype="float64")
        for column in columns:
            if column not in merged.columns:
                continue
            cleaned = _clean_special_missing(merged[column])
            result = result.fillna(cleaned)
        return result

    near_exit_specs = [
        {
            "panel_year": year,
            "annual_earnings_col": None,
            "household_income_col": None,
            "employment_col": None,
            "interview_month_col": None,
            "interview_year_col": None,
            "degree_col": None,
            "occupation_cols": [],
            "ui_spells_col": None,
            "ui_amount_col": None,
            "govt_program_income_col": None,
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": None,
            "weight_pounds_col": None,
            "cesd_score_col": None,
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "delinquency_cols": {},
            "k12_status_cols": _nlsy97_monthly_refnums("E501", year),
            "college_status_cols": _nlsy97_monthly_refnums("E511", year),
            "arrest_month_cols": _nlsy97_monthly_refnums("E801", year),
            "incarc_month_cols": _nlsy97_monthly_refnums("E802", year),
            "bkrpt_weeks_col": _nlsy97_annual_refnum("E026", year) if year >= 2000 else None,
            "bkrpt_hours_col": _nlsy97_annual_refnum("E028", year) if year >= 2000 else None,
        }
        for year in NLSY97_NEAR_EXIT_YEARS
    ]

    panel_specs = (
        {
            "panel_year": 1997,
            "annual_earnings_col": None,
            "household_income_col": None,
            "employment_col": None,
            "interview_month_col": None,
            "interview_year_col": None,
            "degree_col": None,
            "occupation_cols": [],
            "ui_spells_col": None,
            "ui_amount_col": None,
            "govt_program_income_col": None,
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": None,
            "weight_pounds_col": None,
            "cesd_score_col": None,
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "asvab_test_month_col": "R9708601",
            "asvab_test_year_col": "R9708602",
            "asvab_pos_cols": {
                "gs": "R9705200",
                "ar": "R9705300",
                "wk": "R9705400",
                "pc": "R9705500",
                "no": "R9705600",
                "cs": "R9705700",
                "ai": "R9705800",
                "si": "R9705900",
                "mk": "R9706000",
                "mc": "R9706100",
                "ei": "R9706200",
            },
            "asvab_neg_cols": {
                "gs": "R9706400",
                "ar": "R9706500",
                "wk": "R9706600",
                "pc": "R9706700",
                "no": "R9706800",
                "cs": "R9706900",
                "ai": "R9707000",
                "si": "R9707100",
                "mk": "R9707200",
                "mc": "R9707300",
                "ei": "R9707400",
            },
            "delinquency_cols": {
                "destroyed_property": "R0360900",
                "theft_under50": "R0361000",
                "theft_over50": "R0361100",
                "attacked": "R0361300",
                "sold_drugs": "R0361400",
            },
        },
        *near_exit_specs,
        {
            "panel_year": 2007,
            "annual_earnings_col": None,
            "household_income_col": None,
            "employment_col": None,
            "interview_month_col": None,
            "interview_year_col": None,
            "degree_col": None,
            "occupation_cols": [],
            "ui_spells_col": None,
            "ui_amount_col": None,
            "govt_program_income_col": None,
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": None,
            "weight_pounds_col": None,
            "cesd_score_col": None,
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": "Z9083800",
            "sat_math_col": "Z9033700",
            "sat_verbal_col": "Z9033900",
            "act_col": "Z9034100",
            "delinquency_cols": {},
        },
        {
            "panel_year": 2010,
            "annual_earnings_col": None,
            "household_income_col": "T5206900",
            "employment_col": None,
            "interview_month_col": "T5229101",
            "interview_year_col": "T5229102",
            "degree_col": None,
            "occupation_cols": [],
            "ui_spells_col": None,
            "ui_amount_col": None,
            "govt_program_income_col": None,
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": None,
            "weight_pounds_col": None,
            "cesd_score_col": None,
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "delinquency_cols": {},
        },
        {
            "panel_year": 2011,
            "annual_earnings_col": None,
            "household_income_col": None,
            "employment_col": "T7295800",
            "interview_month_col": "T6680901",
            "interview_year_col": "T6680902",
            "degree_col": None,
            "occupation_cols": ["T7311500", "T7311600", "T7311700", "T7311800", "T7311900"],
            "ui_spells_col": None,
            "ui_amount_col": None,
            "govt_program_income_col": None,
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": None,
            "weight_pounds_col": "T7635800",
            "height_feet_col": "T7635600",
            "height_inches_col": "T7635700",
            "cesd_score_col": None,
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "delinquency_cols": {},
        },
        {
            "panel_year": 2015,
            "annual_earnings_col": None,
            "household_income_col": None,
            "employment_col": None,
            "interview_month_col": "U0036301",
            "interview_year_col": "U0036302",
            "degree_col": None,
            "occupation_cols": ["U0741900", "U0742000", "U0742100", "U0742200", "U0742300", "U0742400", "U0742500"],
            "ui_spells_col": None,
            "ui_amount_col": None,
            "govt_program_income_col": None,
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": "U1032300",
            "weight_pounds_col": None,
            "cesd_score_col": None,
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "delinquency_cols": {},
        },
        {
            "panel_year": 2013,
            "annual_earnings_col": None,
            "household_income_col": None,
            "employment_col": None,
            "interview_month_col": "T8154001",
            "interview_year_col": "T8154002",
            "degree_col": None,
            "occupation_cols": ["T8821300", "T8821400", "T8821500", "T8821600", "T8821700", "T8821800"],
            "ui_spells_col": None,
            "ui_amount_col": None,
            "govt_program_income_col": None,
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": None,
            "weight_pounds_col": None,
            "cesd_score_col": None,
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "delinquency_cols": {},
        },
        {
            "panel_year": 2017,
            "annual_earnings_col": None,
            "household_income_col": None,
            "employment_col": None,
            "interview_month_col": "U1876601",
            "interview_year_col": "U1876602",
            "degree_col": None,
            "occupation_cols": ["U2679300", "U2679400", "U2679500", "U2679600", "U2679700", "U2679800", "U2679900"],
            "ui_spells_col": None,
            "ui_amount_col": None,
            "govt_program_income_col": None,
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": None,
            "weight_pounds_col": None,
            "cesd_score_col": None,
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "delinquency_cols": {},
        },
        {
            "panel_year": 2019,
            "annual_earnings_col": "U4282300",
            "household_income_col": "U3444000",
            "employment_col": "U3455100",
            "interview_month_col": "U3475201",
            "interview_year_col": "U3475202",
            "degree_col": None,
            "occupation_cols": ["U4114400", "U4114500", "U4114600", "U4114700", "U4114800", "U4114900", "U4115000"],
            "ui_spells_col": "Z9074610",
            "ui_amount_col": "Z9083410",
            "govt_program_income_col": "U4285700",
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": None,
            "weight_pounds_col": None,
            "cesd_score_col": None,
            "arrest_status_col": "E8013912",
            "incarc_status_col": "E8023912",
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "delinquency_cols": {},
        },
        {
            "panel_year": 2021,
            "annual_earnings_col": "U5753500",
            "household_income_col": "U4949700",
            "employment_col": "U4958300",
            "interview_month_col": "U4976701",
            "interview_year_col": "U4976702",
            "degree_col": "U5072600",
            "occupation_cols": ["U5591200", "U5591300", "U5591400", "U5591500", "U5591600"],
            "ui_spells_col": "Z9074612",
            "ui_amount_col": "Z9083412",
            "govt_program_income_col": None,
            "first_marriage_year_col": "Z9073201",
            "first_marriage_end_col": "Z9073400",
            "total_bio_children_col": "Z9085400",
            "total_marriages_col": "Z9123000",
            "marital_status_col": "Z9149100",
            "household_type_col": "Z9165100",
            "health_status_col": None,
            "smoking_days_col": None,
            "alcohol_days_col": None,
            "binge_days_col": None,
            "marijuana_days_col": None,
            "weight_pounds_col": None,
            "cesd_score_col": None,
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "delinquency_cols": {},
        },
        {
            "panel_year": 2023,
            "annual_earnings_col": None,
            "household_income_col": None,
            "employment_col": None,
            "interview_month_col": None,
            "interview_year_col": None,
            "degree_col": None,
            "occupation_cols": [],
            "ui_spells_col": None,
            "ui_amount_col": None,
            "govt_program_income_col": None,
            "health_status_col": "U7239800",
            "smoking_days_col": "U7239100",
            "alcohol_days_col": "U7239400",
            "binge_days_col": "U7239600",
            "marijuana_days_col": None,
            "weight_pounds_col": "U7239900",
            "cesd_score_col": "U6365300",
            "arrest_status_col": None,
            "incarc_status_col": None,
            "education_years_col": None,
            "sat_math_col": None,
            "sat_verbal_col": None,
            "act_col": None,
            "delinquency_cols": {},
        },
    )
    def _raw_or_nan(column: str | None) -> pd.Series:
        if column is None or column not in merged.columns:
            return pd.Series(np.nan, index=merged.index, dtype="float64")
        return merged[column]

    def _count_observed(frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(np.nan, index=merged.index, dtype="float64")
        count = frame.notna().sum(axis=1).astype("float64")
        return count.where(count > 0)

    def _sum_observed(frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(np.nan, index=merged.index, dtype="float64")
        total = frame.sum(axis=1, min_count=1).astype("float64")
        return total

    def _clean_status_frame(columns: list[str]) -> pd.DataFrame:
        if not columns:
            return pd.DataFrame(index=merged.index)
        return pd.DataFrame({column: _clean_special_missing(_raw_or_nan(column)) for column in columns})

    def _count_matching_codes(frame: pd.DataFrame, codes: tuple[int, ...]) -> pd.Series:
        if frame.empty:
            return pd.Series(np.nan, index=merged.index, dtype="float64")
        observed = frame.notna().sum(axis=1)
        counts = frame.isin(codes).sum(axis=1).astype("float64")
        return counts.where(observed > 0)

    numeric_panel_cols = (
        "annual_earnings",
        "household_income",
        "employment_raw",
        "interview_month",
        "interview_year",
        "degree_code",
        "occupation_code",
        "ui_spells",
        "ui_amount",
        "govt_program_income",
        "health_status",
        "smoking_days_30",
        "alcohol_days_30",
        "binge_days_30",
        "marijuana_days_30",
        "asvab_test_month",
        "asvab_test_year",
        "asvab_pos_subtests_observed",
        "asvab_neg_subtests_observed",
        "asvab_pos_score_sum",
        "asvab_neg_score_sum",
        "weight_pounds",
        "height_feet",
        "height_inches",
        "height_total_inches",
        "bmi",
        "cesd_score",
        "k12_enrolled_months",
        "k12_vacation_months",
        "k12_disciplinary_or_other_months",
        "college_enrolled_months",
        "college_4yrplus_months",
        "arrest_months",
        "incarceration_months",
        "bkrpt_weeks",
        "bkrpt_hours",
        "first_marriage_year",
        "first_marriage_end",
        "total_bio_children",
        "total_marriages",
        "marital_status_collapsed",
        "household_type_40",
        "arrest_status",
        "incarc_status",
        "education_years_snapshot",
        "sat_math_bin",
        "sat_verbal_bin",
        "act_bin",
        "ever_destroyed_property",
        "ever_theft_under50",
        "ever_theft_over50",
        "ever_attacked",
        "ever_sold_drugs",
        "primary_treatment_nlsy97",
        "resident_bio_father_absent_1997",
        "first_absent_year",
        "first_absent_month",
        "localized_exit_year_available",
        "event_time_candidate_year",
        "age_at_wave",
    )
    for spec in panel_specs:
        panel_year = spec["panel_year"]
        wave_source_cols = [
            spec.get(key)
            for key in (
                "annual_earnings_col",
                "household_income_col",
                "employment_col",
                "interview_month_col",
                "interview_year_col",
                "degree_col",
                "ui_spells_col",
                "ui_amount_col",
                "govt_program_income_col",
                "asvab_test_month_col",
                "asvab_test_year_col",
                "health_status_col",
                "smoking_days_col",
                "alcohol_days_col",
                "binge_days_col",
                "marijuana_days_col",
                "weight_pounds_col",
                "height_feet_col",
                "height_inches_col",
                "cesd_score_col",
                "first_marriage_year_col",
                "first_marriage_end_col",
                "total_bio_children_col",
                "total_marriages_col",
                "marital_status_col",
                "household_type_col",
                "arrest_status_col",
                "incarc_status_col",
                "bkrpt_weeks_col",
                "bkrpt_hours_col",
                "education_years_col",
                "sat_math_col",
                "sat_verbal_col",
                "act_col",
            )
            if spec.get(key) is not None
        ]
        wave_source_cols.extend(spec.get("occupation_cols", []))
        wave_source_cols.extend(spec.get("delinquency_cols", {}).values())
        wave_source_cols.extend(spec.get("asvab_pos_cols", {}).values())
        wave_source_cols.extend(spec.get("asvab_neg_cols", {}).values())
        wave_source_cols.extend(spec.get("k12_status_cols", []))
        wave_source_cols.extend(spec.get("college_status_cols", []))
        wave_source_cols.extend(spec.get("arrest_month_cols", []))
        wave_source_cols.extend(spec.get("incarc_month_cols", []))
        if not any(column in merged.columns for column in wave_source_cols):
            continue
        asvab_pos = pd.DataFrame(
            {
                name: _clean_special_missing(_raw_or_nan(column))
                for name, column in spec.get("asvab_pos_cols", {}).items()
            }
        )
        asvab_neg = pd.DataFrame(
            {
                name: _clean_special_missing(_raw_or_nan(column))
                for name, column in spec.get("asvab_neg_cols", {}).items()
            }
        )
        height_feet = _clean_special_missing(_raw_or_nan(spec.get("height_feet_col")))
        height_inches = _clean_special_missing(_raw_or_nan(spec.get("height_inches_col")))
        height_total_inches = (height_feet * 12.0) + height_inches
        height_total_inches = height_total_inches.where(height_total_inches > 0)
        weight_pounds = _clean_special_missing(_raw_or_nan(spec.get("weight_pounds_col")))
        bmi = (weight_pounds / (height_total_inches * height_total_inches) * 703.0).where(
            height_total_inches.notna() & weight_pounds.notna()
        )
        k12_status = _clean_status_frame(spec.get("k12_status_cols", []))
        college_status = _clean_status_frame(spec.get("college_status_cols", []))
        arrest_month_status = _clean_status_frame(spec.get("arrest_month_cols", []))
        incarc_month_status = _clean_status_frame(spec.get("incarc_month_cols", []))
        frame = pd.DataFrame(
            {
                "respondent_id": merged["respondent_id"],
                "panel_year": pd.Series(int(panel_year), index=merged.index, dtype="Int64"),
                "annual_earnings": _clean_special_missing(_raw_or_nan(spec["annual_earnings_col"])),
                "household_income": _clean_special_missing(_raw_or_nan(spec["household_income_col"])),
                "employment_raw": pd.to_numeric(_raw_or_nan(spec["employment_col"]), errors="coerce"),
                "interview_month": _clean_special_missing(_raw_or_nan(spec["interview_month_col"])),
                "interview_year": _clean_special_missing(_raw_or_nan(spec["interview_year_col"])),
                "degree_code": (
                    _clean_special_missing(_raw_or_nan(spec["degree_col"]))
                ),
                "occupation_code": _coalesce_clean_numeric(spec.get("occupation_cols", [])),
                "ui_spells": _clean_special_missing(_raw_or_nan(spec["ui_spells_col"])),
                "ui_amount": _clean_special_missing(_raw_or_nan(spec["ui_amount_col"])),
                "govt_program_income": _clean_special_missing(_raw_or_nan(spec["govt_program_income_col"])),
                "asvab_test_month": _clean_special_missing(_raw_or_nan(spec.get("asvab_test_month_col"))),
                "asvab_test_year": _clean_special_missing(_raw_or_nan(spec.get("asvab_test_year_col"))),
                "asvab_pos_subtests_observed": _count_observed(asvab_pos),
                "asvab_neg_subtests_observed": _count_observed(asvab_neg),
                "asvab_pos_score_sum": _sum_observed(asvab_pos),
                "asvab_neg_score_sum": _sum_observed(asvab_neg),
                "health_status": _clean_special_missing(_raw_or_nan(spec["health_status_col"])),
                "smoking_days_30": _clean_special_missing(_raw_or_nan(spec["smoking_days_col"])),
                "alcohol_days_30": _clean_special_missing(_raw_or_nan(spec["alcohol_days_col"])),
                "binge_days_30": _clean_special_missing(_raw_or_nan(spec["binge_days_col"])),
                "marijuana_days_30": _clean_special_missing(_raw_or_nan(spec["marijuana_days_col"])),
                "weight_pounds": weight_pounds,
                "height_feet": height_feet,
                "height_inches": height_inches,
                "height_total_inches": height_total_inches,
                "bmi": bmi,
                "cesd_score": _clean_special_missing(_raw_or_nan(spec["cesd_score_col"])),
                "k12_enrolled_months": _count_matching_codes(k12_status, (2,)),
                "k12_vacation_months": _count_matching_codes(k12_status, (4,)),
                "k12_disciplinary_or_other_months": _count_matching_codes(k12_status, (5, 6)),
                "college_enrolled_months": _count_matching_codes(college_status, (2, 3, 4)),
                "college_4yrplus_months": _count_matching_codes(college_status, (3, 4)),
                "arrest_months": _count_matching_codes(arrest_month_status, (1,)),
                "incarceration_months": _count_matching_codes(incarc_month_status, (1,)),
                "bkrpt_weeks": _clean_special_missing(_raw_or_nan(spec.get("bkrpt_weeks_col"))),
                "bkrpt_hours": _clean_special_missing(_raw_or_nan(spec.get("bkrpt_hours_col"))),
                "first_marriage_year": _clean_special_missing(_raw_or_nan(spec.get("first_marriage_year_col"))),
                "first_marriage_end": _clean_special_missing(_raw_or_nan(spec.get("first_marriage_end_col"))),
                "total_bio_children": _clean_special_missing(_raw_or_nan(spec.get("total_bio_children_col"))),
                "total_marriages": _clean_special_missing(_raw_or_nan(spec.get("total_marriages_col"))),
                "marital_status_collapsed": _clean_special_missing(_raw_or_nan(spec.get("marital_status_col"))),
                "household_type_40": _clean_special_missing(_raw_or_nan(spec.get("household_type_col"))),
                "arrest_status": _clean_special_missing(_raw_or_nan(spec["arrest_status_col"])),
                "incarc_status": _clean_special_missing(_raw_or_nan(spec["incarc_status_col"])),
                "education_years_snapshot": _clean_special_missing(_raw_or_nan(spec["education_years_col"])),
                "sat_math_bin": _clean_special_missing(_raw_or_nan(spec["sat_math_col"])),
                "sat_verbal_bin": _clean_special_missing(_raw_or_nan(spec["sat_verbal_col"])),
                "act_bin": _clean_special_missing(_raw_or_nan(spec["act_col"])),
                "ever_destroyed_property": _clean_special_missing(_raw_or_nan(spec.get("delinquency_cols", {}).get("destroyed_property"))),
                "ever_theft_under50": _clean_special_missing(_raw_or_nan(spec.get("delinquency_cols", {}).get("theft_under50"))),
                "ever_theft_over50": _clean_special_missing(_raw_or_nan(spec.get("delinquency_cols", {}).get("theft_over50"))),
                "ever_attacked": _clean_special_missing(_raw_or_nan(spec.get("delinquency_cols", {}).get("attacked"))),
                "ever_sold_drugs": _clean_special_missing(_raw_or_nan(spec.get("delinquency_cols", {}).get("sold_drugs"))),
                "primary_treatment_nlsy97": merged["primary_treatment_nlsy97"],
                "primary_treatment_label_nlsy97": merged["primary_treatment_label_nlsy97"],
                "father_absence_type_1997": merged["father_absence_type_1997"],
                "resident_bio_father_absent_1997": merged["resident_bio_father_absent_1997"],
                "childhood_history_type": merged["childhood_history_type"],
                "first_absent_year": merged["first_absent_year"],
                "first_absent_month": merged["first_absent_month"],
                "localized_exit_year_available": merged["localized_exit_year_available"],
                "event_time_candidate_year": merged["event_time_candidate_year"],
                "age_at_wave": pd.Series(float(panel_year), index=merged.index) - merged["birth_year_clean"],
            }
        )
        for col in numeric_panel_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Float64")
        frame["primary_treatment_label_nlsy97"] = frame["primary_treatment_label_nlsy97"].astype("string")
        frame["father_absence_type_1997"] = frame["father_absence_type_1997"].astype("string")
        frame["childhood_history_type"] = frame["childhood_history_type"].astype("string")
        frame["ever_theft_any"] = (
            frame[["ever_theft_under50", "ever_theft_over50"]].max(axis=1, skipna=True)
        ).astype("Float64")
        frame["delinquency_any"] = (
            frame[["ever_destroyed_property", "ever_theft_any", "ever_attacked", "ever_sold_drugs"]].max(axis=1, skipna=True)
        ).astype("Float64")
        employment_clean = pd.Series(np.nan, index=frame.index, dtype="float64")
        employment_clean.loc[frame["employment_raw"].isin([1, 2])] = 1.0
        employment_clean.loc[frame["employment_raw"].isin([0, 3])] = 0.0
        frame["employment_clean"] = pd.to_numeric(employment_clean, errors="coerce").astype("Float64")
        frame["event_time_from_first_absent_year"] = (
            frame["panel_year"].astype("Float64") - pd.to_numeric(frame["first_absent_year"], errors="coerce")
        ).astype("Float64")
        frame["event_time_from_last_coresidence"] = (
            frame["panel_year"].astype("Float64") - frame["event_time_candidate_year"]
        ).astype("Float64")
        panel_frames.append(frame)

    panel = pd.concat(panel_frames, ignore_index=True)
    panel_path = processed_root / "nlsy97_longitudinal_outcome_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    availability = pd.DataFrame(
        [
            {
                "panel_year": year,
                "rows": int((panel["panel_year"] == year).sum()),
                "annual_earnings_observed_rows": int(((panel["panel_year"] == year) & panel["annual_earnings"].notna()).sum()),
                "household_income_observed_rows": int(((panel["panel_year"] == year) & panel["household_income"].notna()).sum()),
                "employment_observed_rows": int(((panel["panel_year"] == year) & panel["employment_clean"].notna()).sum()),
                "interview_date_observed_rows": int(((panel["panel_year"] == year) & panel["interview_year"].notna() & panel["interview_month"].notna()).sum()),
                "occupation_observed_rows": int(((panel["panel_year"] == year) & panel["occupation_code"].notna()).sum()),
                "schooling_observed_rows": int(((panel["panel_year"] == year) & (panel["k12_enrolled_months"].notna() | panel["k12_vacation_months"].notna() | panel["k12_disciplinary_or_other_months"].notna() | panel["college_enrolled_months"].notna() | panel["college_4yrplus_months"].notna())).sum()),
                "broken_report_work_observed_rows": int(((panel["panel_year"] == year) & (panel["bkrpt_weeks"].notna() | panel["bkrpt_hours"].notna())).sum()),
                "ui_observed_rows": int(((panel["panel_year"] == year) & (panel["ui_spells"].notna() | panel["ui_amount"].notna())).sum()),
                "health_observed_rows": int(((panel["panel_year"] == year) & (panel["health_status"].notna() | panel["cesd_score"].notna())).sum()),
                "anthropometric_observed_rows": int(((panel["panel_year"] == year) & (panel["height_total_inches"].notna() | panel["weight_pounds"].notna() | panel["bmi"].notna())).sum()),
                "substance_observed_rows": int(((panel["panel_year"] == year) & (panel["smoking_days_30"].notna() | panel["alcohol_days_30"].notna() | panel["binge_days_30"].notna() | panel["marijuana_days_30"].notna())).sum()),
                "milestone_observed_rows": int(((panel["panel_year"] == year) & (panel["education_years_snapshot"].notna() | panel["sat_math_bin"].notna() | panel["sat_verbal_bin"].notna() | panel["act_bin"].notna() | panel["delinquency_any"].notna() | panel["asvab_pos_subtests_observed"].notna() | panel["asvab_neg_subtests_observed"].notna())).sum()),
                "family_formation_observed_rows": int(((panel["panel_year"] == year) & (panel["first_marriage_year"].notna() | panel["first_marriage_end"].notna() | panel["total_bio_children"].notna() | panel["total_marriages"].notna() | panel["marital_status_collapsed"].notna() | panel["household_type_40"].notna())).sum()),
                "monthly_justice_observed_rows": int(((panel["panel_year"] == year) & (panel["arrest_months"].notna() | panel["incarceration_months"].notna())).sum()),
                "justice_observed_rows": int(((panel["panel_year"] == year) & (panel["arrest_status"].notna() | panel["incarc_status"].notna() | panel["arrest_months"].notna() | panel["incarceration_months"].notna())).sum()),
            }
            for year in sorted(panel["panel_year"].dropna().unique())
        ]
    )
    availability_path = output_dir / "nlsy97_longitudinal_panel_availability.csv"
    availability.to_csv(availability_path, index=False)

    childhood_availability = pd.DataFrame(
        [
            {
                "metric": "respondents_in_analysis_ready",
                "count": int(merged["respondent_id"].nunique()),
                "share": 1.0,
                "note": "Distinct respondents available for the NLSY97 childhood-history scaffold.",
            },
            {
                "metric": "childhood_history_rows",
                "count": int(len(childhood_history.index)),
                "share": float(len(childhood_history.index) / merged["respondent_id"].nunique()) if merged["respondent_id"].nunique() else 0.0,
                "note": "Observed-by-1997 childhood person-year rows between birth and age 17.",
            },
            {
                "metric": "childhood_history_observed_presence_rows",
                "count": int(childhood_history["father_presence_observed"].sum()) if not childhood_history.empty else 0,
                "share": float(childhood_history["father_presence_observed"].mean()) if not childhood_history.empty else 0.0,
                "note": "Rows with an explicit present/absent coding rather than an unresolved retrospective gap.",
            },
            {
                "metric": "respondents_with_localized_exit_year",
                "count": int(respondent_history["localized_exit_year_available"].sum()),
                "share": float(respondent_history["localized_exit_year_available"].mean()) if not respondent_history.empty else 0.0,
                "note": "Respondents whose first absent year can be localized from the last-coresidence item.",
            },
            {
                "metric": "respondents_with_pre_post_childhood_observation",
                "count": int(respondent_history["has_pre_post_childhood_observation"].sum()),
                "share": float(respondent_history["has_pre_post_childhood_observation"].mean()) if not respondent_history.empty else 0.0,
                "note": "Respondents with at least one observed present row and one observed absent row in childhood.",
            },
            {
                "metric": "respondents_absent_from_birth",
                "count": int((respondent_history["childhood_history_type"] == "absent_from_birth").sum()),
                "share": float((respondent_history["childhood_history_type"] == "absent_from_birth").mean()) if not respondent_history.empty else 0.0,
                "note": "Respondents coded absent from birth using the 1997 retrospective item.",
            },
            {
                "metric": "respondents_stable_present_until_1997",
                "count": int((respondent_history["childhood_history_type"] == "stable_present_until_1997").sum()),
                "share": float((respondent_history["childhood_history_type"] == "stable_present_until_1997").mean()) if not respondent_history.empty else 0.0,
                "note": "Respondents coded present through baseline with no >12-month separation reported.",
            },
        ]
    )
    childhood_availability_path = output_dir / "nlsy97_childhood_history_availability.csv"
    childhood_availability.to_csv(childhood_availability_path, index=False)

    summary_path = output_dir / "nlsy97_longitudinal_panel_summary.md"
    summary_lines = [
        "# NLSY97 Longitudinal Panel Summary",
        "",
        f"- Respondents in panel scaffold: {int(panel['respondent_id'].nunique())}",
        f"- Waves included: {', '.join(str(int(v)) for v in sorted(panel['panel_year'].dropna().unique()))}",
        f"- Adult-panel rows with event-time candidate year: {int(panel['event_time_candidate_year'].notna().sum())}",
        f"- Employment-observed rows: {int(panel['employment_clean'].notna().sum())}",
        f"- Earnings-observed rows: {int(panel['annual_earnings'].notna().sum())}",
        f"- Near-exit schooling-observed rows: {int((panel['k12_enrolled_months'].notna() | panel['k12_vacation_months'].notna() | panel['k12_disciplinary_or_other_months'].notna() | panel['college_enrolled_months'].notna() | panel['college_4yrplus_months'].notna()).sum())}",
        f"- Near-exit broken-report work rows: {int((panel['bkrpt_weeks'].notna() | panel['bkrpt_hours'].notna()).sum())}",
        f"- Near-exit monthly justice rows: {int((panel['arrest_months'].notna() | panel['incarceration_months'].notna()).sum())}",
        f"- Health/substance-observed rows: {int((panel['health_status'].notna() | panel['cesd_score'].notna() | panel['smoking_days_30'].notna() | panel['alcohol_days_30'].notna() | panel['binge_days_30'].notna() | panel['marijuana_days_30'].notna()).sum())}",
        f"- Anthropometric-observed rows: {int((panel['height_total_inches'].notna() | panel['weight_pounds'].notna() | panel['bmi'].notna()).sum())}",
        f"- Family-formation-observed rows: {int((panel['first_marriage_year'].notna() | panel['first_marriage_end'].notna() | panel['total_bio_children'].notna() | panel['total_marriages'].notna() | panel['marital_status_collapsed'].notna() | panel['household_type_40'].notna()).sum())}",
        f"- Adolescent/young-adult milestone rows: {int((panel['education_years_snapshot'].notna() | panel['sat_math_bin'].notna() | panel['sat_verbal_bin'].notna() | panel['act_bin'].notna() | panel['delinquency_any'].notna() | panel['asvab_pos_subtests_observed'].notna() | panel['asvab_neg_subtests_observed'].notna()).sum())}",
        f"- Childhood history rows: {int(len(childhood_history.index))}",
        f"- Respondents with localized first-absence year: {int(respondent_history['localized_exit_year_available'].sum())}",
        f"- Respondents with observed pre/post childhood rows: {int(respondent_history['has_pre_post_childhood_observation'].sum())}",
        "",
        "This scaffold now carries 1997 delinquency plus ASVAB/cognitive testing, 1998-2005 monthly school/college/arrest/incarceration summaries with 2000-2005 broken-report work intensity, 2007 test/education milestones, 2010 household-income timing, 2011 employment and anthropometrics, 2013/2015/2017 occupation and interview timing, 2015 substance use, 2019/2021 economic and UI outcomes, 2021 family-formation summaries, 2023 health/substance outcomes, plus a childhood person-year exposure history recovered from the locked 1997 father-treatment items.",
        "It is materially stronger for timing diagnostics than the prior adult-follow-up-heavy panel because it now includes denser near-exit school, work, and justice observations, but it is still not a full event-study panel because outcome support around each childhood exit remains partial rather than continuous.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return NLSY97LongitudinalPanelResult(
        panel_path=panel_path,
        childhood_history_path=childhood_history_path,
        summary_path=summary_path,
        availability_path=availability_path,
        childhood_availability_path=childhood_availability_path,
    )


def build_nlsy_pilot(
    *,
    interim_root: Path,
    processed_root: Path,
    outputs_root: Path,
    overwrite: bool = False,
    generated_at: datetime | None = None,
) -> PilotBuildResult:
    extracts = discover_cohort_extracts(interim_root)
    if not extracts:
        raise FileNotFoundError(f"No NLSY cohort extracts found under {interim_root}")

    processed_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    inventory_paths = write_inventory_report(
        extracts,
        report_dir=outputs_root,
        interim_root=interim_root,
        generated_at=generated_at,
    )

    artifacts: list[MaterializedCohort] = []
    for extract in extracts:
        frame = pd.read_csv(extract.panel_extract_path)
        parquet_path = processed_root / f"{extract.cohort}_panel_extract.parquet"
        dictionary_path = outputs_root / f"{extract.cohort}_column_dictionary.csv"
        if overwrite or not parquet_path.exists():
            frame.to_parquet(parquet_path, index=False)
        dictionary = _build_dictionary_frame(extract)
        dictionary.to_csv(dictionary_path, index=False)
        artifacts.append(
            MaterializedCohort(
                cohort=extract.cohort,
                parquet_path=parquet_path,
                dictionary_path=dictionary_path,
                row_count=len(frame.index),
                column_count=len(frame.columns),
            )
        )

    return PilotBuildResult(
        artifacts=tuple(artifacts),
        inventory_markdown_path=inventory_paths["markdown"],
        inventory_json_path=inventory_paths["json"],
    )
