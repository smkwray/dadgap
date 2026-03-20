from __future__ import annotations

import json
from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd

from father_longrun.pipelines.contracts import (
    build_site_results_payload,
    relative_to_root,
    validate_canonical_results_payload,
    validate_site_results_payload,
)
from father_longrun.pipelines.harmonize import to_serializable_record


SUMMARY_COLUMNS = [
    "artifact",
    "purpose",
    "topic",
    "source",
    "source_group",
    "group_type",
    "group_value",
    "metric",
    "value",
    "value_kind",
    "reference_year",
    "sample_size",
    "weighting_method",
    "notes",
]

FOREST_COLUMNS = [
    "artifact",
    "purpose",
    "topic",
    "source",
    "source_group",
    "group_type",
    "group_value",
    "label",
    "metric",
    "scale",
    "estimate",
    "std_error",
    "lower_ci",
    "upper_ci",
    "reference_year",
    "sample_size",
    "weighting_method",
    "notes",
]

PERCENT_METRICS = {
    "fatherlessness_rate",
    "father_present_rate",
    "father_absent_share",
    "father_present_share",
    "two_parent_share",
    "father_only_share",
    "mother_only_share",
    "weighted_female_share",
    "weighted_employment_share",
    "weighted_poverty_share",
    "employment_rate",
    "poverty_share",
}

SUMMARY_TABLE_SPECS = [
    ("table_nlsy97_fatherlessness_prevalence.csv", "nlsy_fatherlessness", "prevalence"),
    ("table_nlsy97_fatherlessness_predictors.csv", "nlsy_fatherlessness_predictors", "predictors"),
    ("table_nlsy97_outcome_gaps_vs_public_context.csv", "cross_cohort_context", "benchmark_context"),
    ("table_public_benchmark_context.csv", "public_benchmark_context", "benchmark_context"),
    ("table_acs_child_father_presence_context.csv", "acs_child_context", "benchmark_context"),
    ("table_nlsy97_race_sex_outcome_gaps.csv", "nlsy_race_sex_gaps", "subgroup_gaps"),
]


@dataclass(frozen=True)
class SynthesisResult:
    summary_path: Path
    forest_ready_path: Path
    memo_path: Path
    site_payload_path: Path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required synthesis artifact does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_manifest_rows(manifests_root: Path) -> list[dict[str, str]]:
    results_json_path = manifests_root / "results.json"
    if results_json_path.exists():
        payload = json.loads(results_json_path.read_text(encoding="utf-8"))
        errors = validate_canonical_results_payload(payload)
        if errors:
            raise ValueError(f"Invalid canonical results payload at {results_json_path}: {'; '.join(errors)}")
        return [dict(row) for row in payload.get("artifacts", [])]
    return _read_csv_rows(manifests_root / "results_appendix_manifest.csv")


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"na", "nan", "none"}:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _artifact_meta_by_path(manifest_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        path = (row.get("path") or "").strip()
        if path:
            meta[path] = row
    return meta


def _meta_for_file(meta_by_path: dict[str, dict[str, str]], filename: str) -> tuple[str, str]:
    meta = meta_by_path.get(filename, {})
    artifact = meta.get("artifact") or Path(filename).stem
    purpose = meta.get("purpose") or ""
    return artifact, purpose


def _is_percent_metric(metric: str) -> bool:
    return metric in PERCENT_METRICS or metric.endswith("_share") or metric.endswith("_rate")


def _group_label(source: str, source_group: str, group_type: str, group_value: str, metric: str) -> str:
    parts = [source]
    if source_group and source_group not in {source, "overall"}:
        parts.append(source_group)
    elif group_value and group_value not in {source, "overall"}:
        parts.append(group_value)
    parts.append(metric)
    return " | ".join(parts)


def _append_summary_row(
    rows: list[dict[str, Any]],
    *,
    artifact: str,
    purpose: str,
    topic: str,
    source: str,
    source_group: str,
    group_type: str,
    group_value: str,
    metric: str,
    value: float | int | str | None,
    value_kind: str,
    reference_year: int | None,
    sample_size: int | None,
    weighting_method: str,
    notes: str,
) -> None:
    if value is None:
        return
    rows.append(
        {
            "artifact": artifact,
            "purpose": purpose,
            "topic": topic,
            "source": source,
            "source_group": source_group,
            "group_type": group_type,
            "group_value": group_value,
            "metric": metric,
            "value": value,
            "value_kind": value_kind,
            "reference_year": reference_year,
            "sample_size": sample_size,
            "weighting_method": weighting_method,
            "notes": notes,
        }
    )


def _append_forest_row(
    rows: list[dict[str, Any]],
    *,
    artifact: str,
    purpose: str,
    topic: str,
    source: str,
    source_group: str,
    group_type: str,
    group_value: str,
    label: str,
    metric: str,
    scale: str,
    estimate: float,
    std_error: float,
    reference_year: int | None,
    sample_size: int | None,
    weighting_method: str,
    notes: str,
) -> None:
    if estimate is None or std_error is None:
        return
    rows.append(
        {
            "artifact": artifact,
            "purpose": purpose,
            "topic": topic,
            "source": source,
            "source_group": source_group,
            "group_type": group_type,
            "group_value": group_value,
            "label": label,
            "metric": metric,
            "scale": scale,
            "estimate": estimate,
            "std_error": std_error,
            "lower_ci": estimate - (1.96 * std_error),
            "upper_ci": estimate + (1.96 * std_error),
            "reference_year": reference_year,
            "sample_size": sample_size,
            "weighting_method": weighting_method,
            "notes": notes,
        }
    )


def _binomial_se(value: float, sample_size: int) -> float | None:
    if sample_size <= 0:
        return None
    if value < 0.0 or value > 1.0:
        return None
    return math.sqrt(value * (1.0 - value) / sample_size)


def _append_rate_rows(
    summary_rows: list[dict[str, Any]],
    forest_rows: list[dict[str, Any]],
    *,
    artifact: str,
    purpose: str,
    topic: str,
    source: str,
    row: dict[str, str],
    metric: str,
    group_type: str,
    group_value: str,
    source_group: str,
    sample_key: str,
    notes: str,
) -> None:
    value = _to_float(row.get(metric))
    if value is None:
        return
    reference_year = _to_int(row.get("reference_year"))
    sample_size = _to_int(row.get(sample_key))
    weighting_method = row.get("weighting_method") or ""

    _append_summary_row(
        summary_rows,
        artifact=artifact,
        purpose=purpose,
        topic=topic,
        source=source,
        source_group=source_group,
        group_type=group_type,
        group_value=group_value,
        metric=metric,
        value=value,
        value_kind="rate" if _is_percent_metric(metric) else "amount",
        reference_year=reference_year,
        sample_size=sample_size,
        weighting_method=weighting_method,
        notes=notes,
    )

    if sample_size is None:
        return
    std_error = _binomial_se(value, sample_size)
    if std_error is None:
        return
    _append_forest_row(
        forest_rows,
        artifact=artifact,
        purpose=purpose,
        topic=topic,
        source=source,
        source_group=source_group,
        group_type=group_type,
        group_value=group_value,
        label=_group_label(source, source_group, group_type, group_value, metric),
        metric=metric,
        scale="rate",
        estimate=value,
        std_error=std_error,
        reference_year=reference_year,
        sample_size=sample_size,
        weighting_method=weighting_method,
        notes=notes,
    )


def _append_amount_row(
    summary_rows: list[dict[str, Any]],
    *,
    artifact: str,
    purpose: str,
    topic: str,
    source: str,
    row: dict[str, str],
    metric: str,
    group_type: str,
    group_value: str,
    source_group: str,
    notes: str,
) -> None:
    value = _to_float(row.get(metric))
    if value is None:
        return
    _append_summary_row(
        summary_rows,
        artifact=artifact,
        purpose=purpose,
        topic=topic,
        source=source,
        source_group=source_group,
        group_type=group_type,
        group_value=group_value,
        metric=metric,
        value=value,
        value_kind="amount",
        reference_year=_to_int(row.get("reference_year")),
        sample_size=_to_int(row.get("row_count") or row.get("n")),
        weighting_method=row.get("weighting_method") or "",
        notes=notes,
    )


def _append_predictor_rows(
    summary_rows: list[dict[str, Any]],
    forest_rows: list[dict[str, Any]],
    *,
    artifact: str,
    purpose: str,
    topic: str,
    source: str,
    row: dict[str, str],
    group_type: str,
    group_value: str,
    notes: str,
) -> None:
    term = row.get("term") or ""
    if not term or term == "const":
        return
    coefficient = _to_float(row.get("coefficient"))
    std_error = _to_float(row.get("std_error"))
    odds_ratio = _to_float(row.get("odds_ratio"))
    p_value = _to_float(row.get("p_value"))
    sample_size = _to_int(row.get("n"))
    weighting_method = row.get("weighting_method") or ""
    reference_year = _to_int(row.get("reference_year"))

    _append_summary_row(
        summary_rows,
        artifact=artifact,
        purpose=purpose,
        topic=topic,
        source=source,
        source_group=term,
        group_type=group_type,
        group_value=group_value or term,
        metric="coefficient",
        value=coefficient,
        value_kind="log_odds",
        reference_year=reference_year,
        sample_size=sample_size,
        weighting_method=weighting_method,
        notes=notes,
    )
    _append_summary_row(
        summary_rows,
        artifact=artifact,
        purpose=purpose,
        topic=topic,
        source=source,
        source_group=term,
        group_type=group_type,
        group_value=group_value or term,
        metric="odds_ratio",
        value=odds_ratio,
        value_kind="odds_ratio",
        reference_year=reference_year,
        sample_size=sample_size,
        weighting_method=weighting_method,
        notes=notes,
    )
    _append_summary_row(
        summary_rows,
        artifact=artifact,
        purpose=purpose,
        topic=topic,
        source=source,
        source_group=term,
        group_type=group_type,
        group_value=group_value or term,
        metric="p_value",
        value=p_value,
        value_kind="probability",
        reference_year=reference_year,
        sample_size=sample_size,
        weighting_method=weighting_method,
        notes=notes,
    )

    if coefficient is None or std_error is None:
        return
    _append_forest_row(
        forest_rows,
        artifact=artifact,
        purpose=purpose,
        topic=topic,
        source=source,
        source_group=term,
        group_type=group_type,
        group_value=group_value or term,
        label=_group_label(source, term, group_type, group_value or term, "coefficient"),
        metric="coefficient",
        scale="log_odds",
        estimate=coefficient,
        std_error=std_error,
        reference_year=reference_year,
        sample_size=sample_size,
        weighting_method=weighting_method,
        notes=notes,
    )


def _sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("artifact", "")),
        str(row.get("source", "")),
        str(row.get("group_type", "")),
        str(row.get("group_value", "")),
        str(row.get("metric", "")),
    )


def _summarize_inventory(manifest_rows: list[dict[str, str]], summary_rows: list[dict[str, Any]]) -> None:
    _append_summary_row(
        summary_rows,
        artifact="results_appendix_manifest",
        purpose="Inventory of appendix outputs used by synthesis.",
        topic="inventory",
        source="manifest",
        source_group="results_appendix_manifest",
        group_type="artifact_inventory",
        group_value="all",
        metric="artifact_count",
        value=len(manifest_rows),
        value_kind="count",
        reference_year=None,
        sample_size=None,
        weighting_method="",
        notes="Artifact inventory derived from the committed appendix manifest.",
    )


def _headline_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    headlines: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["topic"] == "inventory":
            headlines.append(row)
        elif row["artifact"] == "nlsy_prevalence_table" and row["metric"] == "fatherlessness_rate" and row["group_value"] == "overall":
            headlines.append(row)
        elif row["artifact"] == "benchmark_context_table" and row["metric"] in {"weighted_employment_share", "weighted_poverty_share"} and row["source"] in {"acs_pums", "cps_asec"}:
            headlines.append(row)
        elif row["artifact"] == "acs_child_context_table" and row["metric"] == "father_absent_share" and row["group_value"] == "overall":
            headlines.append(row)
        elif row["artifact"] == "nlsy_predictor_table" and row["metric"] == "odds_ratio" and row["source_group"] in {"race_HISPANIC", "race_NON-BLACK, NON-HISPANIC", "sex_male"}:
            headlines.append(row)
    return headlines


def _build_memo(summary_rows: list[dict[str, Any]], forest_rows: list[dict[str, Any]], manifest_rows: list[dict[str, str]]) -> str:
    headlines = _headline_rows(summary_rows)
    lines = [
        "# Cross-Cohort Synthesis",
        "",
        "This memo is built only from committed appendix and benchmark artifacts.",
        "",
        "## Inputs",
        f"- Appendix manifest artifacts: {len(manifest_rows)}",
        f"- Normalized summary rows: {len(summary_rows)}",
        f"- Forest-ready rows: {len(forest_rows)}",
        "",
        "## Headline Rows",
    ]
    for row in headlines[:8]:
        value = row["value"]
        if isinstance(value, float):
            if row["value_kind"] == "count":
                rendered = f"{value:.0f}"
            elif row["value_kind"] in {"rate", "probability", "odds_ratio"}:
                rendered = f"{value:.4f}"
            else:
                rendered = f"{value:.2f}"
        else:
            rendered = str(value)
        lines.append(
            f"- {row['artifact']} / {row['source']} / {row['metric']}: {rendered}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- The summary table keeps amount and share rows together so downstream consumers can filter by `value_kind`.",
            "- The forest-ready table only includes rows with an interval estimate, which keeps it usable for plotting without raw-data recalculation.",
            "- Predictor terms remain on the log-odds scale in the forest-ready layer.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_currency(value: float | None, *, digits: int = 0) -> str:
    if value is None:
        return "unavailable"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{digits}f}"


def _format_pct(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        return "unavailable"
    return f"{value * 100:.{digits}f}%"


def _format_signed(value: float | None, *, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    text = f"{value:+.{digits}f}"
    return f"{text}{suffix}"


def _significance_label(coef: float | None, std_error: float | None) -> str:
    if coef is None or std_error is None or std_error <= 0:
        return "n.s."
    z_score = abs(coef / std_error)
    if z_score >= 3.291:
        return "p < 0.001"
    if z_score >= 2.576:
        return "p < 0.01"
    if z_score >= 1.96:
        return "p < 0.05"
    return "n.s."


def _artifact_path_by_name(manifest_rows: list[dict[str, str]]) -> dict[str, str]:
    return {row.get("artifact", ""): row.get("path", "") for row in manifest_rows if row.get("artifact")}


def _read_optional_artifact_table(
    manifest_rows: list[dict[str, str]],
    *,
    tables_root: Path,
    artifact: str,
) -> pd.DataFrame:
    path_by_name = _artifact_path_by_name(manifest_rows)
    filename = path_by_name.get(artifact)
    if not filename:
        return pd.DataFrame()
    path = tables_root / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _subset(frame: pd.DataFrame, *, columns: list[str] | None = None, **filters: Any) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns or [])
    work = frame
    for key, expected in filters.items():
        if key not in work.columns:
            return pd.DataFrame(columns=columns or [])
        work = work.loc[work[key] == expected]
    if columns is None:
        return work.copy()
    missing = [column for column in columns if column not in work.columns]
    if missing:
        return pd.DataFrame(columns=columns)
    return work.loc[:, columns].copy()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [to_serializable_record(row) for row in frame.to_dict(orient="records")]


def _column_values(frame: pd.DataFrame, column: str) -> list[float]:
    if frame.empty or column not in frame.columns:
        return []
    return frame[column].astype(float).tolist()


def _value(frame: pd.DataFrame, *, column: str, **filters: Any) -> float | None:
    if frame.empty:
        return None
    work = frame
    for key, expected in filters.items():
        if key not in work.columns:
            return None
        work = work.loc[work[key] == expected]
    if work.empty or column not in work.columns:
        return None
    return _to_float(work.iloc[0][column])


def _labelize_group(value: str) -> str:
    mapping = {
        "BLACK": "Black",
        "HISPANIC": "Hispanic",
        "NON-BLACK, NON-HISPANIC": "Non-Black non-Hispanic",
        "female": "Female",
        "male": "Male",
        "q1_low": "Q1 (low)",
        "q2": "Q2",
        "q3": "Q3",
        "q4_high": "Q4 (high)",
        "below_100_pct": "Below 100% poverty",
        "100_124_pct": "100–124%",
        "125_149_pct": "125–149%",
        "150_plus_pct": "150%+",
    }
    return mapping.get(value, str(value).replace("_", " ").title())


def _format_predictor_label(term: str) -> str:
    mapping = {
        "sex_male": "Male",
        "race_HISPANIC": "Hispanic (vs Black)",
        "race_NON-BLACK, NON-HISPANIC": "Non-Black non-Hispanic (vs Black)",
        "mother_education_filled": "Mother education (filled)",
        "father_education_filled": "Father education (filled)",
        "mother_education_missing": "Mother education missing",
        "father_education_missing": "Father education missing",
        "birth_year_centered": "Birth year (centered)",
    }
    return mapping.get(term, term)


def _race_sex_earnings_gaps(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required_columns = {"sex", "race_ethnicity_3cat", "source_group", "mean_earnings"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return [], []
    rows: list[dict[str, Any]] = []
    chart_rows: list[dict[str, Any]] = []
    for race in ["BLACK", "HISPANIC", "NON-BLACK, NON-HISPANIC"]:
        race_slice = frame.loc[frame["race_ethnicity_3cat"] == race].copy()
        overall = {}
        for source_group in ["resident_bio_father_present", "resident_bio_father_absent"]:
            subset = race_slice.loc[race_slice["source_group"] == source_group]
            if subset.empty:
                continue
            overall[source_group] = _to_float(subset.iloc[0]["mean_earnings"])
        if len(overall) == 2:
            chart_rows.append(
                {
                    "label": _labelize_group(race),
                    "value": overall["resident_bio_father_absent"] - overall["resident_bio_father_present"],
                }
            )
    for sex in ["FEMALE", "MALE"]:
        for race in ["BLACK", "HISPANIC", "NON-BLACK, NON-HISPANIC"]:
            present = frame.loc[
                (frame["sex"] == sex)
                & (frame["race_ethnicity_3cat"] == race)
                & (frame["source_group"] == "resident_bio_father_present")
            ]
            absent = frame.loc[
                (frame["sex"] == sex)
                & (frame["race_ethnicity_3cat"] == race)
                & (frame["source_group"] == "resident_bio_father_absent")
            ]
            if present.empty or absent.empty:
                continue
            present_mean = _to_float(present.iloc[0]["mean_earnings"])
            absent_mean = _to_float(absent.iloc[0]["mean_earnings"])
            if present_mean is None or absent_mean is None:
                continue
            rows.append(
                {
                    "group": f"{'Black' if race == 'BLACK' else 'Hispanic' if race == 'HISPANIC' else 'NBNH'} {'Female' if sex == 'FEMALE' else 'Male'}",
                    "father_present": _format_currency(present_mean),
                    "father_absent": _format_currency(absent_mean),
                    "gap": _format_currency(present_mean - absent_mean),
                }
            )
    return rows, chart_rows


def _build_site_payload(
    *,
    project_root: Path,
    manifest_rows: list[dict[str, str]],
    tables_root: Path,
    summary_path: Path,
    forest_ready_path: Path,
    memo_path: Path,
) -> Path:
    docs_root = project_root / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    site_payload_path = docs_root / "results.json"

    prevalence = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_prevalence_table")
    predictors = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_predictor_table")
    outcome_gaps = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_outcome_gap_table")
    race_sex = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_race_gap_table")
    acs_child = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="acs_child_context_table")
    benchmarks = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="benchmark_context_table")
    cognitive = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_cognitive_table")
    cognitive_subgroups = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_cognitive_subgroup_table")
    health = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_health_table")
    mental = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_mental_health_table")
    family = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_family_formation_table")
    occupation = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_occupation_effect_table")
    heterogeneity = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_effect_heterogeneity_table")
    near_term = _read_optional_artifact_table(manifest_rows, tables_root=tables_root, artifact="nlsy_near_term_effects_table")
    summary_rows = _read_optional_csv(summary_path)
    forest_rows = _read_optional_csv(forest_ready_path)

    overall_prevalence = _value(prevalence, column="fatherlessness_rate", group_type="overall")
    overall_n = _value(prevalence, column="n", group_type="overall")
    present_earnings = _value(outcome_gaps, column="mean_earnings", source_group="resident_bio_father_present")
    absent_earnings = _value(outcome_gaps, column="mean_earnings", source_group="resident_bio_father_absent")
    overall_earnings = _value(outcome_gaps, column="mean_earnings", source_group="overall")
    acs_earnings = _value(benchmarks, column="weighted_mean_earnings", source="acs_pums")
    cps_benchmarks = _subset(benchmarks, source="cps_asec")
    if not cps_benchmarks.empty and "reference_year" in cps_benchmarks.columns:
        cps_benchmarks = cps_benchmarks.sort_values("reference_year")
    cps_earnings = _value(cps_benchmarks, column="weighted_mean_earnings", source="cps_asec")
    acs_child_overall = _value(acs_child, column="father_absent_share", group_type="overall")
    g_gap = _value(cognitive, column="adjusted_absent_coef", outcome="g_proxy_1997")
    ed_gap = _value(cognitive, column="adjusted_absent_coef", outcome="education_years_snapshot")
    poor_health_gap = _value(health, column="adjusted_absent_coef", outcome="poor_health_2023")
    cesd_gap = _value(mental, column="adjusted_absent_coef", outcome="cesd_score_2023")
    married_gap = _value(family, column="adjusted_absent_coef", outcome="ever_married_2021")

    race_chart = _subset(
        prevalence,
        columns=["group_value", "fatherlessness_pct", "n"],
        group_type="race_ethnicity_3cat",
    )
    if not prevalence.empty and {"group_type", "group_value", "fatherlessness_pct", "n"}.issubset(prevalence.columns):
        overall_row = _subset(prevalence, columns=["group_value", "fatherlessness_pct", "n"], group_type="overall")
        race_chart = pd.concat([race_chart, overall_row], ignore_index=True)

    edu_chart = _subset(
        prevalence,
        columns=["group_value", "fatherlessness_pct"],
        group_type="parent_education_band",
    )
    if not edu_chart.empty and "group_value" in edu_chart.columns:
        edu_chart = edu_chart.loc[edu_chart["group_value"] != "missing"].copy()
    edu_order = ["q1_low", "q2", "q3", "q4_high"]
    if not edu_chart.empty:
        edu_chart["group_value"] = pd.Categorical(edu_chart["group_value"], categories=edu_order, ordered=True)
        edu_chart = edu_chart.sort_values("group_value")

    acs_race = _subset(acs_child, columns=["group_value", "father_absent_pct"], group_type="race_ethnicity_3cat")
    acs_poverty = _subset(acs_child, columns=["group_value", "father_absent_pct"], group_type="poverty_band")
    acs_income = _subset(acs_child, columns=["group_value", "father_absent_pct"], group_type="household_income_band")
    income_order = ["q1_low", "q2", "q3", "q4_high"]
    if not acs_income.empty:
        acs_income["group_value"] = pd.Categorical(acs_income["group_value"], categories=income_order, ordered=True)
        acs_income = acs_income.sort_values("group_value")

    predictor_rows: list[dict[str, Any]] = []
    if not predictors.empty and {"term", "p_value", "odds_ratio"}.issubset(predictors.columns):
        top_predictors = predictors.loc[predictors["term"] != "const"].copy().sort_values("p_value").head(5)
        for _, row in top_predictors.iterrows():
            predictor_rows.append(
                {
                    "predictor": _format_predictor_label(str(row["term"])),
                    "odds_ratio": f"{float(row['odds_ratio']):.3f}" if pd.notna(row["odds_ratio"]) else "unavailable",
                    "p_value": f"{float(row['p_value']):.3g}" if pd.notna(row["p_value"]) else "unavailable",
                }
            )

    heterogeneity_rows: list[dict[str, Any]] = []
    if not heterogeneity.empty and {"group_type", "group_value", "outcome", "row_count", "absent_minus_present_gap", "adjusted_absent_coef"}.issubset(heterogeneity.columns):
        wanted = [
            ("sex", "female", "g_proxy_1997"),
            ("sex", "male", "g_proxy_1997"),
            ("race_ethnicity_3cat", "BLACK", "g_proxy_1997"),
            ("race_ethnicity_3cat", "HISPANIC", "g_proxy_1997"),
            ("race_ethnicity_3cat", "NON-BLACK, NON-HISPANIC", "g_proxy_1997"),
            ("race_ethnicity_3cat", "BLACK", "annual_earnings_2021"),
            ("race_ethnicity_3cat", "HISPANIC", "annual_earnings_2021"),
            ("race_ethnicity_3cat", "NON-BLACK, NON-HISPANIC", "annual_earnings_2021"),
        ]
        for group_type, group_value, outcome in wanted:
            row = heterogeneity.loc[
                (heterogeneity["group_type"] == group_type)
                & (heterogeneity["group_value"] == group_value)
                & (heterogeneity["outcome"] == outcome)
            ]
            if row.empty:
                continue
            first = row.iloc[0]
            heterogeneity_rows.append(
                {
                    "subgroup": _labelize_group(group_value),
                    "outcome": "g_proxy" if outcome == "g_proxy_1997" else "Earnings",
                    "n": int(first["row_count"]),
                    "raw_gap": f"{float(first['absent_minus_present_gap']):+.3f}" if outcome == "g_proxy_1997" else _format_currency(float(first["absent_minus_present_gap"]), digits=0),
                    "adjusted": f"{float(first['adjusted_absent_coef']):+.3f}" if outcome == "g_proxy_1997" else _format_currency(float(first["adjusted_absent_coef"]), digits=0),
                }
            )

    outcome_cognitive = _subset(cognitive)
    if not outcome_cognitive.empty and "outcome" in outcome_cognitive.columns:
        outcome_cognitive = outcome_cognitive.loc[outcome_cognitive["outcome"].isin(["g_proxy_1997", "education_years_snapshot", "sat_math_bin", "act_bin"])].copy()
    outcome_health = _subset(health)
    if not outcome_health.empty and "outcome" in outcome_health.columns:
        outcome_health = outcome_health.loc[outcome_health["outcome"].isin(["poor_health_2023", "bmi_2011_clean", "obesity_2011", "smoking_days_30_2023", "marijuana_days_30_2015", "alcohol_days_30_2023", "any_binge_2023"])].copy()
    outcome_mental = mental.copy()
    outcome_family = _subset(family)
    if not outcome_family.empty and "outcome" in outcome_family.columns:
        outcome_family = outcome_family.loc[outcome_family["outcome"].isin(["ever_married_2021", "currently_married_2021", "total_marriages_2021", "total_bio_children_2021", "age_first_marriage_2021"])].copy()
    outcome_near_term = _subset(near_term)
    if not outcome_near_term.empty and "outcome" in outcome_near_term.columns:
        outcome_near_term = outcome_near_term.loc[outcome_near_term["outcome"].isin(["schooling_engagement_months", "k12_enrolled_months", "college_enrolled_months", "arrest_any", "incarceration_any"])].copy()
    benchmark_chart = [
        {"label": "NLSY97 Overall", "value": overall_earnings},
        {"label": "ACS PUMS 2024", "value": acs_earnings},
    ]
    if not benchmarks.empty and {"source", "reference_year", "weighted_mean_earnings"}.issubset(benchmarks.columns):
        for year in [2023, 2024, 2025]:
            row = benchmarks.loc[(benchmarks["source"] == "cps_asec") & (benchmarks["reference_year"] == year)]
            if not row.empty:
                benchmark_chart.append({"label": f"CPS ASEC {year}", "value": _to_float(row.iloc[0]["weighted_mean_earnings"])})
    race_sex_table, earnings_race_chart = _race_sex_earnings_gaps(race_sex)
    largest_earnings_gap = max(earnings_race_chart, key=lambda row: abs(row.get("value") or 0.0)) if earnings_race_chart else None

    payload = build_site_results_payload(
        artifacts=manifest_rows,
        pages={
            "home": {
                "stats": {
                    "respondents": f"{int(overall_n):,}" if overall_n is not None else "unavailable",
                    "father_absent_pct": _format_pct(overall_prevalence),
                    "earnings_gap": _format_currency((present_earnings - absent_earnings) if present_earnings is not None and absent_earnings is not None else None),
                    "adjusted_g_gap": _format_signed(g_gap, digits=2, suffix=" SD"),
                },
                "key_findings": [
                    {
                        "text": (
                            f"Father-absent respondents score {_format_signed(_value(cognitive, column='absent_minus_present_gap', outcome='g_proxy_1997'), digits=2, suffix=' SD')} lower on g_proxy; "
                            f"{_format_signed(g_gap, digits=2, suffix=' SD')} persists after covariate adjustment. "
                            f"Education gap: {_format_signed(_value(cognitive, column='absent_minus_present_gap', outcome='education_years_snapshot'), digits=1)} years raw, "
                            f"{_format_signed(ed_gap, digits=1)} years adjusted."
                        ),
                    },
                    {
                        "text": (
                            f"Overall {_format_currency((present_earnings - absent_earnings) if present_earnings is not None and absent_earnings is not None else None)} gap. "
                            f"Largest race-adjusted earnings gap is {_format_currency(largest_earnings_gap['value'] if largest_earnings_gap else None)} for "
                            f"{largest_earnings_gap['label'] if largest_earnings_gap else 'unavailable'}."
                        ),
                    },
                    {
                        "text": (
                            f"ACS 2024: {_format_pct(_value(acs_child, column='father_absent_share', group_type='poverty_band', group_value='below_100_pct'))} of children below poverty "
                            f"lack a resident father vs {_format_pct(_value(acs_child, column='father_absent_share', group_type='household_income_band', group_value='q4_high'))} in the top income quartile."
                        ),
                    },
                ],
                "glance_chart": {
                    "labels": ["g_proxy (SD)", "Education (years)", "Poor health (share)", "CES-D (score)", "Ever married (share)"],
                    "values": [g_gap, ed_gap, poor_health_gap, cesd_gap, married_gap],
                },
            },
            "prevalence": {
                "sex_cards": {
                    "female": _format_pct(_value(prevalence, column="fatherlessness_rate", group_type="sex", group_value="female")),
                    "male": _format_pct(_value(prevalence, column="fatherlessness_rate", group_type="sex", group_value="male")),
                },
                "race_chart": {
                    "labels": [
                        f"{_labelize_group(str(row['group_value']))}\\n(n={int(row['n']):,})" for _, row in race_chart.iterrows()
                    ],
                    "values": [float(row["fatherlessness_pct"]) for _, row in race_chart.iterrows()],
                },
                "education_chart": {
                    "labels": [_labelize_group(str(value)) for value in edu_chart["group_value"].astype(str).tolist()],
                    "values": edu_chart["fatherlessness_pct"].astype(float).tolist(),
                },
                "acs_race_chart": {
                    "labels": [_labelize_group(str(value)) for value in acs_race["group_value"].tolist()],
                    "values": acs_race["father_absent_pct"].astype(float).tolist(),
                },
                "acs_poverty_chart": {
                    "labels": [_labelize_group(str(value)) for value in acs_poverty["group_value"].tolist()],
                    "values": acs_poverty["father_absent_pct"].astype(float).tolist(),
                },
                "acs_income_chart": {
                    "labels": [_labelize_group(str(value)) for value in acs_income["group_value"].astype(str).tolist()],
                    "values": acs_income["father_absent_pct"].astype(float).tolist(),
                },
                "predictors": predictor_rows,
                "earnings_race_chart": {
                    "labels": [row["label"] for row in earnings_race_chart],
                    "values": [row["value"] for row in earnings_race_chart],
                },
                "g_race_chart": {
                    "labels": [
                        _labelize_group(str(value))
                        for value in _subset(
                            heterogeneity,
                            columns=["group_value"],
                            group_type="race_ethnicity_3cat",
                            outcome="g_proxy_1997",
                        )["group_value"].tolist()
                    ],
                    "values": _subset(
                        heterogeneity,
                        columns=["adjusted_absent_coef"],
                        group_type="race_ethnicity_3cat",
                        outcome="g_proxy_1997",
                    )["adjusted_absent_coef"].astype(float).tolist(),
                },
                "heterogeneity_table": heterogeneity_rows,
                "insight": (
                    f"The adjusted earnings gap is largest in {largest_earnings_gap['label'] if largest_earnings_gap else 'the largest observed group'} "
                    f"({_format_currency(largest_earnings_gap['value'] if largest_earnings_gap else None)}). "
                    f"The cognitive gap remains negative across race groups after adjustment."
                ),
            },
            "outcomes": {
                "cognition_table": [
                    {
                        "measure": str(row["outcome_label"]).replace("NLSY97 ", "").replace(" (1997 CAT-ASVAB composite)", ""),
                        "n": f"{int(row['row_count']):,}",
                        "present_mean": f"{float(row['father_present_mean']):.3f}",
                        "absent_mean": f"{float(row['father_absent_mean']):.3f}",
                        "raw_gap": f"{float(row['absent_minus_present_gap']):+.3f}",
                        "adjusted": f"{float(row['adjusted_absent_coef']):+.3f}",
                        "se": f"{float(row['adjusted_absent_se_hc1']):.3f}",
                    }
                    for _, row in outcome_cognitive.iterrows()
                ],
                "cognition_chart": {
                    "labels": [str(row["outcome_label"]).replace("NLSY97 ", "").replace(" (1997 CAT-ASVAB composite)", "") for _, row in outcome_cognitive.iterrows()],
                    "raw": _column_values(outcome_cognitive, "absent_minus_present_gap"),
                    "adjusted": _column_values(outcome_cognitive, "adjusted_absent_coef"),
                },
                "cognition_insight": (
                    f"Father-absent respondents score {_format_signed(_value(cognitive, column='absent_minus_present_gap', outcome='g_proxy_1997'), digits=2, suffix=' SD')} lower on cognitive composite "
                    f"and complete {_format_signed(_value(cognitive, column='absent_minus_present_gap', outcome='education_years_snapshot'), digits=1)} fewer years of education. "
                    f"After adjustment, the g gap is {_format_signed(g_gap, digits=2, suffix=' SD')} and the education gap is {_format_signed(ed_gap, digits=2)} years."
                ),
                "earnings_chart": {
                    "labels": [row["group"] for row in race_sex_table],
                    "values": [
                        float(row["gap"].replace("$", "").replace(",", ""))
                        for row in race_sex_table
                    ],
                },
                "earnings_subgroup_table": race_sex_table,
                "overall_gap_table": [
                    {
                        "outcome": "Mean earnings",
                        "father_present": _format_currency(present_earnings),
                        "father_absent": _format_currency(absent_earnings),
                        "gap": _format_currency((present_earnings - absent_earnings) if present_earnings is not None and absent_earnings is not None else None),
                    },
                    {
                        "outcome": "Employment rate",
                        "father_present": _format_pct(_value(outcome_gaps, column="employment_rate", source_group="resident_bio_father_present")),
                        "father_absent": _format_pct(_value(outcome_gaps, column="employment_rate", source_group="resident_bio_father_absent")),
                        "gap": f"{((_value(outcome_gaps, column='employment_rate', source_group='resident_bio_father_present') or 0) - (_value(outcome_gaps, column='employment_rate', source_group='resident_bio_father_absent') or 0)) * 100:.1f} pp",
                    },
                    {
                        "outcome": "Mean HH income",
                        "father_present": _format_currency(_value(outcome_gaps, column="mean_household_income", source_group="resident_bio_father_present")),
                        "father_absent": _format_currency(_value(outcome_gaps, column="mean_household_income", source_group="resident_bio_father_absent")),
                        "gap": _format_currency(((_value(outcome_gaps, column="mean_household_income", source_group="resident_bio_father_present") or 0) - (_value(outcome_gaps, column="mean_household_income", source_group="resident_bio_father_absent") or 0))),
                    },
                ],
                "health_table": [
                    {
                        "outcome": str(row["outcome_label"]).split(" (")[0],
                        "adjusted": f"{float(row['adjusted_absent_coef']):+.3f}",
                        "unit": row["scale"],
                        "significance": _significance_label(_to_float(row["adjusted_absent_coef"]), _to_float(row["adjusted_absent_se_hc1"])),
                    }
                    for _, row in outcome_health.iterrows()
                ],
                "health_chart": {
                    "labels": [str(row["outcome_label"]).split(" (")[0] for _, row in outcome_health.iterrows()],
                    "values": _column_values(outcome_health, "adjusted_absent_coef"),
                },
                "mental_table": [
                    {
                        "outcome": str(row["outcome_label"]).split(" (")[0],
                        "adjusted": f"{float(row['adjusted_absent_coef']):+.3f}",
                        "unit": row["scale"],
                        "significance": _significance_label(_to_float(row["adjusted_absent_coef"]), _to_float(row["adjusted_absent_se_hc1"])),
                    }
                    for _, row in outcome_mental.iterrows()
                ],
                "mental_chart": {
                    "labels": [str(row["outcome_label"]).split(" (")[0] for _, row in outcome_mental.iterrows()],
                    "values": _column_values(outcome_mental, "adjusted_absent_coef"),
                },
                "family_table": [
                    {
                        "outcome": str(row["outcome_label"]).split(" (")[0],
                        "adjusted": f"{float(row['adjusted_absent_coef']):+.3f}",
                        "unit": row["scale"],
                    }
                    for _, row in outcome_family.iterrows()
                ],
                "family_chart": {
                    "labels": [str(row["outcome_label"]).split(" (")[0] for _, row in outcome_family.iterrows()],
                    "values": _column_values(outcome_family, "adjusted_absent_coef"),
                },
                "event_time_table": [
                    {
                        "outcome": str(row["outcome_label"]),
                        "coefficient": f"{float(row['adjusted_treatment_coef']):+.3f}",
                        "unit": row["scale"],
                    }
                    for _, row in outcome_near_term.iterrows()
                ],
                "event_time_chart": {
                    "labels": [str(row["outcome_label"]) for _, row in outcome_near_term.iterrows()],
                    "values": _column_values(outcome_near_term, "adjusted_treatment_coef"),
                },
                "benchmark_chart": {
                    "labels": [row["label"] for row in benchmark_chart if row["value"] is not None],
                    "values": [row["value"] for row in benchmark_chart if row["value"] is not None],
                },
            },
            "faq": {
                "prevalence_answer": (
                    f"{_format_pct(overall_prevalence)} of respondents in the NLSY97 cohort (born 1980–84) experienced father absence by this definition. "
                    f"In ACS 2024 data on current children under 18, {_format_pct(acs_child_overall)} lack a resident father."
                ),
                "earnings_answer": (
                    f"Father-present adults earn {_format_currency(present_earnings)} on average versus {_format_currency(absent_earnings)} for father-absent adults — "
                    f"a {_format_currency((present_earnings - absent_earnings) if present_earnings is not None and absent_earnings is not None else None)} gap."
                ),
                "national_answer": (
                    f"NLSY97 mean earnings ({_format_currency(overall_earnings)}) are higher than ACS ({_format_currency(acs_earnings)}) "
                    f"and CPS ({_format_currency(cps_earnings)}) estimates for similar ages."
                ),
            },
        },
        tables={
            "summary": _records(summary_rows),
            "forest_ready": _records(forest_rows),
            "cognitive_subgroups": _records(cognitive_subgroups),
            "occupation_effect": _records(occupation),
        },
        memos={
            "cross_cohort_synthesis": memo_path.read_text(encoding="utf-8") if memo_path.exists() else "",
        },
        source_manifest=relative_to_root(project_root / "outputs" / "manifests" / "results.json", project_root),
        synthesis_artifacts=[
            relative_to_root(summary_path, project_root),
            relative_to_root(forest_ready_path, project_root),
            relative_to_root(memo_path, project_root),
        ],
    )
    site_errors = validate_site_results_payload(payload)
    if site_errors:
        raise ValueError(f"Invalid site results payload for {site_payload_path}: {'; '.join(site_errors)}")
    site_payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return site_payload_path


def build_synthesis(*, outputs_root: Path, project_root: Path | None = None) -> SynthesisResult:
    outputs_root = Path(outputs_root)
    project_root = outputs_root.parent if project_root is None else Path(project_root)
    manifests_root = outputs_root / "manifests"
    tables_root = outputs_root / "tables"

    manifest_rows = _load_manifest_rows(manifests_root)
    meta_by_path = _artifact_meta_by_path(manifest_rows)

    summary_rows: list[dict[str, Any]] = []
    forest_rows: list[dict[str, Any]] = []
    _summarize_inventory(manifest_rows, summary_rows)

    for filename, topic, notes in SUMMARY_TABLE_SPECS:
        artifact, purpose = _meta_for_file(meta_by_path, filename)
        table_rows = _read_csv_rows(tables_root / filename)
        for row in table_rows:
            source = row.get("source") or artifact
            source_group = row.get("source_group") or row.get("group_value") or ""
            group_type = row.get("group_type") or topic
            group_value = row.get("group_value") or source_group or source

            if filename == "table_nlsy97_fatherlessness_predictors.csv":
                _append_predictor_rows(
                    summary_rows,
                    forest_rows,
                    artifact=artifact,
                    purpose=purpose,
                    topic=topic,
                    source=source,
                    row=row,
                    group_type=group_type,
                    group_value=group_value,
                    notes=notes,
                )
                continue

            if filename in {
                "table_nlsy97_fatherlessness_prevalence.csv",
                "table_acs_child_father_presence_context.csv",
                "table_public_benchmark_context.csv",
                "table_nlsy97_outcome_gaps_vs_public_context.csv",
                "table_nlsy97_race_sex_outcome_gaps.csv",
            }:
                sample_key = "n" if filename == "table_nlsy97_fatherlessness_prevalence.csv" else "row_count"
                for metric in [
                    column
                    for column in row.keys()
                    if column
                    in {
                        "fatherlessness_rate",
                        "father_present_rate",
                        "mother_education_mean",
                        "father_education_mean",
                        "mean_earnings",
                        "mean_person_income",
                        "mean_household_income",
                        "employment_rate",
                        "poverty_share",
                        "weighted_female_share",
                        "weighted_employment_share",
                        "weighted_mean_earnings",
                        "weighted_mean_person_income",
                        "weighted_poverty_share",
                        "father_present_share",
                        "father_absent_share",
                        "two_parent_share",
                        "father_only_share",
                        "mother_only_share",
                    }
                ]:
                    value = _to_float(row.get(metric))
                    if value is None:
                        continue
                    if metric in {"mother_education_mean", "father_education_mean", "mean_earnings", "mean_person_income", "mean_household_income", "weighted_mean_earnings", "weighted_mean_person_income"}:
                        _append_amount_row(
                            summary_rows,
                            artifact=artifact,
                            purpose=purpose,
                            topic=topic,
                            source=source,
                            row=row,
                            metric=metric,
                            group_type=group_type,
                            group_value=group_value,
                            source_group=source_group,
                            notes=notes,
                        )
                        continue
                    _append_rate_rows(
                        summary_rows,
                        forest_rows,
                        artifact=artifact,
                        purpose=purpose,
                        topic=topic,
                        source=source,
                        row=row,
                        metric=metric,
                        group_type=group_type,
                        group_value=group_value,
                        source_group=source_group,
                        sample_key=sample_key,
                        notes=notes,
                    )

    summary_rows.sort(key=_sort_key)
    forest_rows.sort(key=_sort_key)

    summary_path = manifests_root / "cross_cohort_synthesis_summary.csv"
    forest_ready_path = manifests_root / "cross_cohort_synthesis_forest_ready.csv"
    memo_path = manifests_root / "cross_cohort_synthesis.md"

    _write_csv_rows(summary_path, summary_rows, SUMMARY_COLUMNS)
    _write_csv_rows(forest_ready_path, forest_rows, FOREST_COLUMNS)
    memo_path.write_text(_build_memo(summary_rows, forest_rows, manifest_rows), encoding="utf-8")
    site_payload_path = _build_site_payload(
        project_root=project_root,
        manifest_rows=manifest_rows,
        tables_root=tables_root,
        summary_path=summary_path,
        forest_ready_path=forest_ready_path,
        memo_path=memo_path,
    )

    return SynthesisResult(
        summary_path=summary_path,
        forest_ready_path=forest_ready_path,
        memo_path=memo_path,
        site_payload_path=site_payload_path,
    )
