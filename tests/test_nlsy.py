from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from father_longrun.pipelines.nlsy import (
    build_analysis_ready_treatment_layers,
    build_backbone_scaffold,
    build_nlsy97_fatherlessness_profiles,
    build_merge_contract_report,
    build_nlsy97_longitudinal_panel_scaffold,
    build_nlsy_pilot,
    build_phase0_artifacts,
    build_refresh_spec,
    build_treatment_candidate_layers,
    build_treatment_refresh_extracts,
    build_reviewed_layers,
    discover_cohort_extracts,
)


def _write_fixture_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(header)]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fixture_varmap(path: Path, header: list[str]) -> None:
    lines = ["refnum,question_name,title,survey_year,type"]
    for column in header:
        lines.append(f"{column},{column},{column},,unknown")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fixture_manifest(path: Path, cohort: str, header: list[str], rows: list[list[str]]) -> None:
    payload = {
        "cohort": cohort,
        "n_rows": len(rows),
        "n_columns": len(header),
        "source_path": f"data/interim/{cohort}/raw_files/{cohort}.csv",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_cohort_extracts(tmp_path: Path) -> None:
    for cohort in ("nlsy79", "cnlsy", "nlsy97"):
        cohort_dir = tmp_path / cohort
        cohort_dir.mkdir(parents=True)
        header = ["ID", "VALUE"]
        rows = [["1", "10"], ["2", "20"]]
        _write_fixture_csv(cohort_dir / "panel_extract.csv", header, rows)
        _write_fixture_varmap(cohort_dir / "varmap.csv", header)
        _write_fixture_manifest(cohort_dir / "panel_extract.manifest.json", cohort, header, rows)

    extracts = discover_cohort_extracts(tmp_path)
    assert [extract.cohort for extract in extracts] == ["nlsy79", "cnlsy", "nlsy97"]
    assert all(extract.column_count == 2 for extract in extracts)


def test_build_nlsy_pilot(tmp_path: Path) -> None:
    interim_root = tmp_path / "interim"
    processed_root = tmp_path / "processed"
    outputs_root = tmp_path / "outputs"
    for cohort in ("nlsy79", "cnlsy", "nlsy97"):
        cohort_dir = interim_root / cohort
        cohort_dir.mkdir(parents=True)
        header = ["ID", "VALUE"]
        rows = [["1", "10"], ["2", "20"]]
        _write_fixture_csv(cohort_dir / "panel_extract.csv", header, rows)
        _write_fixture_varmap(cohort_dir / "varmap.csv", header)
        _write_fixture_manifest(cohort_dir / "panel_extract.manifest.json", cohort, header, rows)

    result = build_nlsy_pilot(
        interim_root=interim_root,
        processed_root=processed_root,
        outputs_root=outputs_root,
        generated_at=datetime(2026, 3, 13, tzinfo=timezone.utc),
    )

    assert len(result.artifacts) == 3
    assert result.inventory_markdown_path.exists()
    assert result.inventory_json_path.exists()
    assert str(interim_root) not in result.inventory_markdown_path.read_text(encoding="utf-8")
    assert str(interim_root) not in result.inventory_json_path.read_text(encoding="utf-8")
    assert "<nlsy_interim_root>/nlsy79/panel_extract.csv" in result.inventory_markdown_path.read_text(encoding="utf-8")
    for artifact in result.artifacts:
        assert artifact.parquet_path.exists()
        assert artifact.dictionary_path.exists()


def test_build_phase0_artifacts(tmp_path: Path) -> None:
    interim_root = tmp_path / "interim"
    output_dir = tmp_path / "outputs"
    fixtures = {
        "nlsy79": ["R0000100", "VALUE"],
        "cnlsy": ["C0000100", "C0000200", "VALUE"],
        "nlsy97": ["R0000100", "VALUE"],
    }
    for cohort, header in fixtures.items():
        cohort_dir = interim_root / cohort
        cohort_dir.mkdir(parents=True)
        rows = [["1" for _ in header], ["2" for _ in header]]
        _write_fixture_csv(cohort_dir / "panel_extract.csv", header, rows)
        _write_fixture_varmap(cohort_dir / "varmap.csv", header)
        _write_fixture_manifest(cohort_dir / "panel_extract.manifest.json", cohort, header, rows)

    result = build_phase0_artifacts(interim_root=interim_root, output_dir=output_dir)

    assert result.diagnostics_csv_path.exists()
    assert result.diagnostics_markdown_path.exists()
    assert any(item.dataset_key == "nlsy79" for item in result.manifests)


def test_build_merge_contract_report(tmp_path: Path) -> None:
    interim_root = tmp_path / "interim"
    output_dir = tmp_path / "outputs"

    fixtures = {
        "nlsy79": (["R0000100", "VALUE"], [["10", "1"], ["20", "2"]]),
        "cnlsy": (["C0000100", "C0000200", "VALUE"], [["100", "10", "1"], ["101", "20", "2"]]),
        "nlsy97": (["R0000100", "VALUE"], [["1", "1"], ["2", "2"]]),
    }
    for cohort, (header, rows) in fixtures.items():
        cohort_dir = interim_root / cohort
        cohort_dir.mkdir(parents=True)
        _write_fixture_csv(cohort_dir / "panel_extract.csv", header, rows)
        _write_fixture_varmap(cohort_dir / "varmap.csv", header)
        _write_fixture_manifest(cohort_dir / "panel_extract.manifest.json", cohort, header, rows)

    result = build_merge_contract_report(interim_root=interim_root, output_dir=output_dir)
    assert result.report_path.exists()
    assert result.json_path.exists()


def test_build_backbone_scaffold(tmp_path: Path) -> None:
    interim_root = tmp_path / "interim"
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "outputs"

    fixtures = {
        "nlsy79": (["R0000100", "VALUE"], [["10", "1"], ["20", "2"]]),
        "cnlsy": (["C0000100", "C0000200", "VALUE"], [["100", "10", "1"], ["101", "20", "2"]]),
    }
    for cohort, (header, rows) in fixtures.items():
        cohort_dir = interim_root / cohort
        cohort_dir.mkdir(parents=True)
        _write_fixture_csv(cohort_dir / "panel_extract.csv", header, rows)

    result = build_backbone_scaffold(
        interim_root=interim_root,
        processed_root=processed_root,
        output_dir=output_dir,
    )
    assert result.parquet_path.exists()
    assert result.report_path.exists()


def test_build_reviewed_layers(tmp_path: Path) -> None:
    interim_root = tmp_path / "interim"
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "outputs"

    nlsy79_dir = interim_root / "nlsy79"
    cnlsy_dir = interim_root / "cnlsy"
    nlsy97_dir = interim_root / "nlsy97"
    nlsy79_dir.mkdir(parents=True)
    cnlsy_dir.mkdir(parents=True)
    nlsy97_dir.mkdir(parents=True)

    _write_fixture_csv(
        nlsy79_dir / "panel_extract.csv",
        ["R0000100", "R0214700", "R0214800", "R0000500", "R0006500", "R0007900", "R3279401", "R7006500", "T9900000", "R6940103"],
        [["10", "2", "2", "1960", "12", "11", "30000", "50000", "16", "10000"]],
    )
    _write_fixture_csv(
        cnlsy_dir / "panel_extract.csv",
        ["C0000100", "C0000200", "C0005300", "C0005400", "C0005700", "C0053500", "Y1211300", "Y3066000", "Y3112400", "Y3291500", "Y3299900", "Y3331900", "Y3332100", "Y3332200"],
        [["100", "10", "2", "2", "1990", "14", "13", "1", "12000", "10000", "40000", "24", "14", "4"]],
    )
    _write_fixture_csv(
        nlsy97_dir / "panel_extract.csv",
        ["R0000100", "R0536300", "R0536402", "R1482600", "R0554500", "R0554800", "Z9083800", "T5206900", "U4282300", "U3444000", "U3455100", "U5753500", "U4949700", "U4958300", "U5072600", "Z9121900"],
        [["1", "1", "1981", "2", "12", "12", "16", "40000", "50000", "60000", "1", "55000", "65000", "1", "4", "100000"]],
    )

    result = build_reviewed_layers(interim_root=interim_root, processed_root=processed_root, output_dir=output_dir)
    assert result.mapping_csv_path.exists()
    assert result.availability_markdown_path.exists()
    assert result.exposure_gap_markdown_path.exists()
    assert result.backbone_parquet_path.exists()
    assert result.nlsy97_parquet_path.exists()


def test_build_refresh_spec(tmp_path: Path) -> None:
    result = build_refresh_spec(output_dir=tmp_path)
    assert result.csv_path.exists()
    assert result.markdown_path.exists()
    assert result.yaml_path.exists()
    markdown = result.markdown_path.read_text(encoding="utf-8")
    yaml_payload = result.yaml_path.read_text(encoding="utf-8")
    assert "Refnums are cohort-specific" in markdown
    assert "source_file" in markdown
    assert "source_files:" in yaml_payload
    assert "include_variables:" in yaml_payload


def test_build_treatment_refresh_extracts(tmp_path: Path) -> None:
    interim_root = tmp_path / "interim"
    refreshed_root = tmp_path / "refreshed"
    output_dir = tmp_path / "outputs"
    fixtures = {
        "nlsy79": (["R0000100", "H0001600", "H0013600", "H0046300"], [["10", "1", "1", "1"]]),
        "cnlsy": (["C0000100", "C0000200", "C0953300", "C0953400", "C0953500", "C0953600", "C0953700"], [["100", "10", "1", "1", "1", "1", "1"]]),
        "nlsy97": (["R0000100", "R0335600", "R0335700", "R0335800", "R0336000", "R0336100", "R0336200", "R0336300", "R0885400", "R0885500", "R0885600", "R0885700", "R0885800"], [["1", "1", "1", "1", "0", "1", "2000", "6", "1", "2", "3", "4", "5"]]),
    }
    current_extracts = {
        "nlsy79": ["R0000100"],
        "cnlsy": ["C0000100", "C0000200"],
        "nlsy97": ["R0000100"],
    }
    for cohort, (header, rows) in fixtures.items():
        cohort_dir = interim_root / cohort
        raw_dir = cohort_dir / "raw_files"
        raw_dir.mkdir(parents=True)
        _write_fixture_csv(raw_dir / f"{cohort}.csv", header, rows)
        _write_fixture_csv(cohort_dir / "panel_extract.csv", current_extracts[cohort], [row[: len(current_extracts[cohort])] for row in rows])
        _write_fixture_varmap(cohort_dir / "varmap.csv", header)

    result = build_treatment_refresh_extracts(
        interim_root=interim_root,
        refreshed_root=refreshed_root,
        output_dir=output_dir,
    )
    assert result.report_path.exists()
    assert len(result.artifacts) == 3
    nlsy97_header = _read_fixture_header(refreshed_root / "nlsy97" / "panel_extract.csv")
    assert "R0335600" in nlsy97_header
    assert "R0885800" in nlsy97_header


def test_build_treatment_candidate_layers(tmp_path: Path) -> None:
    refreshed_root = tmp_path / "refreshed"
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    for cohort in ("nlsy79", "cnlsy", "nlsy97"):
        (refreshed_root / cohort).mkdir(parents=True)

    _write_fixture_csv(
        refreshed_root / "nlsy79" / "panel_extract.csv",
        ["R0000100", "H0001600", "H0013600", "H0046300"],
        [["10", "1", "1", "1"]],
    )
    _write_fixture_csv(
        refreshed_root / "cnlsy" / "panel_extract.csv",
        ["C0000100", "C0000200", "C0953300", "C0953400", "C0953500", "C0953600", "C0953700"],
        [["100", "10", "1", "2", "3", "4", "5"]],
    )
    _write_fixture_csv(
        refreshed_root / "nlsy97" / "panel_extract.csv",
        ["R0000100", "R0335600", "R0335700", "R0335800", "R0336000", "R0336100", "R0336200", "R0336300", "R0885400", "R0885500", "R0885600", "R0885700", "R0885800"],
        [["1", "1", "1", "1", "0", "1", "2000", "6", "1", "2", "3", "4", "5"]],
    )

    processed_root.mkdir(parents=True)
    pd.DataFrame([{"respondent_id": 10, "mother_id": 10, "child_id": 100}]).to_parquet(
        processed_root / "nlsy79_cnlsy_backbone_reviewed.parquet",
        index=False,
    )
    pd.DataFrame([{"respondent_id": 1}]).to_parquet(processed_root / "nlsy97_reviewed.parquet", index=False)

    result = build_treatment_candidate_layers(
        refreshed_root=refreshed_root,
        processed_root=processed_root,
        output_dir=output_dir,
    )
    assert result.backbone_path.exists()
    assert result.nlsy97_path.exists()
    assert result.mapping_path.exists()
    assert result.value_counts_path.exists()
    backbone = pd.read_parquet(result.backbone_path)
    nlsy97 = pd.read_parquet(result.nlsy97_path)
    assert "child_ever_sees_father_figure_1990" in backbone.columns
    assert "resident_bio_father_present_1997" in nlsy97.columns


def test_build_analysis_ready_treatment_layers(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    processed_root.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "respondent_id": 10,
                "mother_id": 10,
                "child_id": 100,
                "child_birth_year": 1990,
                "age_2014": 24,
                "education_years_2014": 14,
                "degree_2014": 4,
                "employment_2014": 1,
                "annual_earnings_2014": 12000,
                "wage_income_2014_best_est": 11000,
                "family_income_2014_best_est": 40000,
                "child_ever_sees_father_figure_1990": 1,
                "father_figure_type_1990": 1,
                "child_sees_father_figure_daily_1990": 1,
                "biological_father_alive_h40": 1,
                "biological_father_alive_h50": -4,
                "biological_father_alive_h60": -4,
            }
        ]
    ).to_parquet(processed_root / "nlsy79_cnlsy_backbone_treatment_candidates.parquet", index=False)
    pd.DataFrame(
        [
            {
                "respondent_id": 1,
                "education_years": 16,
                "annual_earnings_2021": 55000,
                "household_income_2021": 65000,
                "net_worth": 100000,
                "resident_bio_father_present_1997": 0,
                "bio_father_contact_ever_1997": 1,
                "bio_father_alive_1997": 1,
            }
        ]
    ).to_parquet(processed_root / "nlsy97_treatment_candidates.parquet", index=False)

    result = build_analysis_ready_treatment_layers(processed_root=processed_root, output_dir=output_dir)
    assert result.backbone_path.exists()
    assert result.nlsy97_path.exists()
    assert result.nlsy97_baseline_path.exists()
    assert result.nlsy97_primary_baseline_path.exists()
    assert result.cnlsy_subset_path.exists()
    assert result.cnlsy_baseline_path.exists()
    assert result.cnlsy_primary_baseline_path.exists()
    assert result.cnlsy_readiness_path.exists()
    assert result.cnlsy_outcome_tiering_path.exists()
    assert result.cnlsy_education_validation_path.exists()
    assert result.cnlsy_education_crosstab_path.exists()
    assert result.cnlsy_attainment_codebook_path.exists()
    assert result.coding_rules_path.exists()
    assert result.summary_path.exists()
    nlsy97 = pd.read_parquet(result.nlsy97_path)
    backbone = pd.read_parquet(result.backbone_path)
    cnlsy_primary = pd.read_csv(result.cnlsy_primary_baseline_path)
    cnlsy_tiering = pd.read_csv(result.cnlsy_outcome_tiering_path)
    cnlsy_validation = pd.read_csv(result.cnlsy_education_validation_path)
    cnlsy_codebook = pd.read_csv(result.cnlsy_attainment_codebook_path)
    assert nlsy97.loc[0, "father_absence_type_1997"] == "absent_alive_contact"
    assert nlsy97.loc[0, "primary_treatment_label_nlsy97"] == "resident_bio_father_absent"
    assert backbone.loc[0, "early_father_figure_presence_type_1990"] == "biological_father"
    assert backbone.loc[0, "education_attainment_code_2014"] == 14
    assert backbone.loc[0, "education_attainment_label_2014"] == "Completed post-baccalaureate professional education"
    assert backbone.loc[0, "degree_attainment_label_2014"] == "Bachelor of Arts"
    assert cnlsy_primary.loc[0, "annual_earnings_2014_mean"] == 12000
    assert "supplementary_sparse" in set(cnlsy_tiering["tier"])
    assert "officially_labeled_attainment_code" in set(cnlsy_tiering["tier"])


def test_build_nlsy97_fatherlessness_profiles(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    processed_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "respondent_id": 1,
                "primary_treatment_nlsy97": 1,
                "sex_raw": 2,
                "race_ethnicity_3cat": "BLACK",
                "mother_education": 2,
                "father_education": 1,
                "parent_education": 1.5,
                "birth_year": 1982,
            },
            {
                "respondent_id": 2,
                "primary_treatment_nlsy97": 0,
                "sex_raw": 1,
                "race_ethnicity_3cat": "BLACK",
                "mother_education": 3,
                "father_education": 3,
                "parent_education": 3,
                "birth_year": 1982,
            },
            {
                "respondent_id": 3,
                "primary_treatment_nlsy97": 1,
                "sex_raw": 2,
                "race_ethnicity_3cat": "HISPANIC",
                "mother_education": 1,
                "father_education": 0,
                "parent_education": 0.5,
                "birth_year": 1981,
            },
            {
                "respondent_id": 4,
                "primary_treatment_nlsy97": 0,
                "sex_raw": 1,
                "race_ethnicity_3cat": "HISPANIC",
                "mother_education": 4,
                "father_education": 4,
                "parent_education": 4,
                "birth_year": 1983,
            },
            {
                "respondent_id": 5,
                "primary_treatment_nlsy97": 1,
                "sex_raw": 2,
                "race_ethnicity_3cat": "NON-BLACK, NON-HISPANIC",
                "mother_education": 1,
                "father_education": -4,
                "parent_education": 1,
                "birth_year": 1984,
            },
            {
                "respondent_id": 6,
                "primary_treatment_nlsy97": 0,
                "sex_raw": 1,
                "race_ethnicity_3cat": "NON-BLACK, NON-HISPANIC",
                "mother_education": 5,
                "father_education": 5,
                "parent_education": 5,
                "birth_year": 1980,
            },
            {
                "respondent_id": 7,
                "primary_treatment_nlsy97": 1,
                "sex_raw": 2,
                "race_ethnicity_3cat": "BLACK",
                "mother_education": 2,
                "father_education": 2,
                "parent_education": 2,
                "birth_year": 1981,
            },
            {
                "respondent_id": 8,
                "primary_treatment_nlsy97": 0,
                "sex_raw": 1,
                "race_ethnicity_3cat": "NON-BLACK, NON-HISPANIC",
                "mother_education": 4,
                "father_education": 4,
                "parent_education": 4,
                "birth_year": 1983,
            },
        ]
    ).to_parquet(processed_root / "nlsy97_analysis_ready.parquet", index=False)

    result = build_nlsy97_fatherlessness_profiles(processed_root=processed_root, output_dir=output_dir)
    assert result.group_summary_path.exists()
    assert result.predictor_path.exists()
    assert result.report_path.exists()

    group_summary = pd.read_csv(result.group_summary_path)
    predictors = pd.read_csv(result.predictor_path)
    report = result.report_path.read_text(encoding="utf-8")
    assert "race_ethnicity_3cat" in set(group_summary["group_type"])
    assert "sex_x_race_ethnicity" in set(group_summary["group_type"])
    assert "parent_education_band" in set(group_summary["group_type"])
    assert "mother_education_band" in set(group_summary["group_type"])
    assert "father_education_band" in set(group_summary["group_type"])
    assert "overall fatherlessness rate" in report.lower()
    assert "const" in set(predictors["term"]) or predictors.empty


def test_build_nlsy97_longitudinal_panel_scaffold(tmp_path: Path) -> None:
    interim_root = tmp_path / "interim"
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "outputs"
    nlsy97_dir = interim_root / "nlsy97"
    raw_dir = nlsy97_dir / "raw_files"
    nlsy97_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    processed_root.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "R0000100": 1,
                "E8013912": 1,
                "E8023912": 0,
                "R0360900": 1,
                "R0361000": 0,
                "R0361100": 1,
                "R0361300": 0,
                "R0361400": 1,
                "R9705200": 100,
                "R9705300": 200,
                "R9706400": 90,
                "R9706500": 180,
                "R9708601": 2,
                "R9708602": 1998,
                "T5206900": 55000,
                "T5229101": 5,
                "T5229102": 2010,
                "T6680901": 6,
                "T6680902": 2011,
                "T7295800": 1,
                "T7311500": 1020,
                "T7311600": -4,
                "T7635600": 5,
                "T7635700": 7,
                "T7635800": 180,
                "T8154001": 6,
                "T8154002": 2013,
                "T8821300": 1100,
                "T8821400": -4,
                "U0036301": 7,
                "U0036302": 2015,
                "U1032300": 2,
                "U0741900": -4,
                "U0742000": 3600,
                "U1876601": 8,
                "U1876602": 2017,
                "U2679300": 4210,
                "U2679400": -4,
                "U4282300": 40000,
                "U4285700": 1200,
                "U3444000": 70000,
                "U3455100": 1,
                "U3475201": 10,
                "U3475202": 2019,
                "U4114400": 5000,
                "U4114500": -4,
                "Z9074610": 1,
                "Z9083410": 900,
                "U5753500": 50000,
                "U4949700": 80000,
                "U4958300": 1,
                "U4976701": 11,
                "U4976702": 2021,
                "U5072600": 4,
                "U5591200": 7000,
                "U5591300": -4,
                "Z9073201": 2018,
                "Z9073400": -4,
                "Z9074612": 2,
                "Z9083412": 1200,
                "Z9085400": 2,
                "Z9123000": 1,
                "Z9149100": 1,
                "Z9165100": 3,
                "U6365300": 8,
                "U7239100": 3,
                "U7239400": 1,
                "U7239600": 0,
                "U7239800": 2,
                "U7239900": 185,
                "Z9033700": 4,
                "Z9033900": 5,
                "Z9034100": 3,
                "Z9083800": 13,
            },
            {
                "R0000100": 2,
                "E8013912": 0,
                "E8023912": 99,
                "R0360900": 0,
                "R0361000": 1,
                "R0361100": 0,
                "R0361300": 1,
                "R0361400": 0,
                "R9705200": 110,
                "R9705300": 210,
                "R9706400": 100,
                "R9706500": 190,
                "R9708601": 6,
                "R9708602": 1997,
                "T5206900": 45000,
                "T5229101": 6,
                "T5229102": 2010,
                "T6680901": 7,
                "T6680902": 2011,
                "T7295800": 0,
                "T7311500": -4,
                "T7311600": 2200,
                "T7635600": 5,
                "T7635700": 7,
                "T7635800": 200,
                "T8154001": 9,
                "T8154002": 2013,
                "T8821300": -4,
                "T8821400": 3300,
                "U0036301": 8,
                "U0036302": 2015,
                "U1032300": 0,
                "U0741900": 4400,
                "U0742000": -4,
                "U1876601": 10,
                "U1876602": 2017,
                "U2679300": -4,
                "U2679400": 5500,
                "U4282300": 30000,
                "U4285700": 500,
                "U3444000": 60000,
                "U3455100": 1,
                "U3475201": 9,
                "U3475202": 2019,
                "U4114400": -4,
                "U4114500": 6100,
                "Z9074610": 0,
                "Z9083410": 0,
                "U5753500": 35000,
                "U4949700": 65000,
                "U4958300": 1,
                "U4976701": 10,
                "U4976702": 2021,
                "U5072600": 3,
                "U5591200": -4,
                "U5591300": 7200,
                "Z9073201": 2015,
                "Z9073400": 2020,
                "Z9074612": 0,
                "Z9083412": 0,
                "Z9085400": 1,
                "Z9123000": 2,
                "Z9149100": 2,
                "Z9165100": 4,
                "U6365300": 12,
                "U7239100": 0,
                "U7239400": 2,
                "U7239600": 1,
                "U7239800": 3,
                "U7239900": 205,
                "Z9033700": 3,
                "Z9033900": 4,
                "Z9034100": 2,
                "Z9083800": 12,
            },
        ]
    ).to_csv(nlsy97_dir / "panel_extract.csv", index=False)
    raw_rows: list[dict[str, int]] = []
    for respondent_id in (1, 2):
        row: dict[str, int] = {"R0000100": respondent_id}
        for year in range(1998, 2006):
            year_code = f"{year - 1980:02d}"
            for month in range(1, 13):
                month_code = f"{month:02d}"
                k12_col = f"E501{year_code}{month_code}"
                college_col = f"E511{year_code}{month_code}"
                arrest_col = f"E801{year_code}{month_code}"
                incarc_col = f"E802{year_code}{month_code}"
                if respondent_id == 1 and year == 1998:
                    row[k12_col] = 2 if month <= 10 else 4
                    row[college_col] = 1
                    row[arrest_col] = 1 if month == 6 else 0
                    row[incarc_col] = 0
                elif respondent_id == 1 and year == 2000:
                    row[k12_col] = 1
                    row[college_col] = 3
                    row[arrest_col] = 1 if month in {1, 2} else 0
                    row[incarc_col] = 1 if month == 2 else 0
                elif respondent_id == 2 and year == 1999:
                    row[k12_col] = 2
                    row[college_col] = 1
                    row[arrest_col] = 0
                    row[incarc_col] = 0
                elif respondent_id == 2 and year == 2001:
                    row[k12_col] = 1
                    row[college_col] = 1
                    row[arrest_col] = 0
                    row[incarc_col] = 1 if month in {3, 4} else 0
                else:
                    row[k12_col] = 1
                    row[college_col] = 2 if respondent_id == 2 and year >= 2000 else 1
                    row[arrest_col] = 1 if respondent_id == 2 and year == 2000 and month == 5 else 0
                    row[incarc_col] = 0
            if year >= 2000:
                row[f"E026{year_code}00"] = 8 if respondent_id == 1 and year == 2000 else 10 + (year - 2000) if respondent_id == 1 else 6 + (year - 2000)
                row[f"E028{year_code}00"] = 20 if respondent_id == 1 and year == 2000 else 25 + (year - 2000) if respondent_id == 1 else 10 + (year - 2000)
        raw_rows.append(row)
    pd.DataFrame(raw_rows).to_csv(raw_dir / "nlsy97_all_1997-2023.csv", index=False)
    pd.DataFrame(
        [
            {
                "respondent_id": 1,
                "birth_year": 1982,
                "primary_treatment_nlsy97": 1,
                "primary_treatment_label_nlsy97": "resident_bio_father_absent",
                "father_absence_type_1997": "absent_alive_contact",
                "resident_bio_father_absent_1997": 1,
                "resident_bio_father_present_1997": 0,
                "bio_father_contact_ever_1997": 1,
                "bio_father_alive_1997": 1,
                "lived_apart_from_bio_father_gt12m_1997": 1,
                "ever_lived_with_bio_father_1997": 1,
                "last_year_lived_with_bio_father": 1996,
                "last_month_lived_with_bio_father": 8,
            },
            {
                "respondent_id": 2,
                "birth_year": 1981,
                "primary_treatment_nlsy97": 0,
                "primary_treatment_label_nlsy97": "resident_bio_father_present",
                "father_absence_type_1997": "resident_bio_father_present",
                "resident_bio_father_absent_1997": 0,
                "resident_bio_father_present_1997": 1,
                "bio_father_contact_ever_1997": -4,
                "bio_father_alive_1997": -4,
                "lived_apart_from_bio_father_gt12m_1997": 0,
                "ever_lived_with_bio_father_1997": -4,
                "last_year_lived_with_bio_father": -4,
                "last_month_lived_with_bio_father": -4,
            },
        ]
    ).to_parquet(processed_root / "nlsy97_analysis_ready.parquet", index=False)

    result = build_nlsy97_longitudinal_panel_scaffold(
        interim_root=interim_root,
        processed_root=processed_root,
        output_dir=output_dir,
    )

    assert result.panel_path.exists()
    assert result.childhood_history_path.exists()
    assert result.availability_path.exists()
    assert result.childhood_availability_path.exists()
    assert result.summary_path.exists()
    panel = pd.read_parquet(result.panel_path)
    availability = pd.read_csv(result.availability_path)
    childhood_history = pd.read_parquet(result.childhood_history_path)
    childhood_availability = pd.read_csv(result.childhood_availability_path)
    assert set(panel["panel_year"]) == {1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2007, 2010, 2011, 2013, 2015, 2017, 2019, 2021, 2023}
    assert availability["interview_date_observed_rows"].sum() == 14
    assert availability.loc[availability["panel_year"] == 1997, "milestone_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 1998, "schooling_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 1998, "monthly_justice_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 2000, "broken_report_work_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 1997, "family_formation_observed_rows"].iloc[0] == 0
    assert availability.loc[availability["panel_year"] == 2007, "milestone_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 2011, "anthropometric_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 2013, "occupation_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 2019, "justice_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 2019, "ui_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 2021, "family_formation_observed_rows"].iloc[0] == 2
    assert availability.loc[availability["panel_year"] == 2023, "health_observed_rows"].iloc[0] == 2
    assert panel.loc[(panel["respondent_id"] == 1) & (panel["panel_year"] == 1997), "delinquency_any"].iloc[0] == 1
    assert panel.loc[(panel["respondent_id"] == 1) & (panel["panel_year"] == 1997), "asvab_pos_score_sum"].iloc[0] == 300
    assert panel.loc[(panel["respondent_id"] == 1) & (panel["panel_year"] == 1998), "k12_enrolled_months"].iloc[0] == 10
    assert panel.loc[(panel["respondent_id"] == 1) & (panel["panel_year"] == 1998), "k12_vacation_months"].iloc[0] == 2
    assert panel.loc[(panel["respondent_id"] == 1) & (panel["panel_year"] == 2000), "college_4yrplus_months"].iloc[0] == 12
    assert panel.loc[(panel["respondent_id"] == 1) & (panel["panel_year"] == 2000), "bkrpt_weeks"].iloc[0] == 8
    assert panel.loc[(panel["respondent_id"] == 2) & (panel["panel_year"] == 1998), "k12_enrolled_months"].iloc[0] == 0
    assert panel.loc[(panel["respondent_id"] == 2) & (panel["panel_year"] == 2001), "incarceration_months"].iloc[0] == 2
    assert panel.loc[(panel["respondent_id"] == 2) & (panel["panel_year"] == 2007), "sat_verbal_bin"].iloc[0] == 4
    assert panel.loc[(panel["respondent_id"] == 1) & (panel["panel_year"] == 2010), "household_income"].iloc[0] == 55000
    assert panel.loc[(panel["respondent_id"] == 1) & (panel["panel_year"] == 2011), "employment_clean"].iloc[0] == 1
    assert round(panel.loc[(panel["respondent_id"] == 2) & (panel["panel_year"] == 2011), "bmi"].iloc[0], 2) == 31.32
    assert panel.loc[(panel["respondent_id"] == 2) & (panel["panel_year"] == 2017), "occupation_code"].iloc[0] == 5500
    assert panel.loc[(panel["respondent_id"] == 1) & (panel["panel_year"] == 2021), "total_bio_children"].iloc[0] == 2
    assert panel.loc[(panel["respondent_id"] == 2) & (panel["panel_year"] == 2023), "health_status"].iloc[0] == 3
    assert "localized_exit_in_1997" in set(childhood_history["childhood_history_type"])
    assert childhood_history.loc[
        (childhood_history["respondent_id"] == 1) & (childhood_history["childhood_year"] == 1997),
        "father_presence_imputed",
    ].iloc[0] == 0
    assert childhood_history.loc[
        (childhood_history["respondent_id"] == 2) & (childhood_history["childhood_year"] == 1997),
        "father_presence_imputed",
    ].iloc[0] == 1
    assert childhood_availability.loc[
        childhood_availability["metric"] == "respondents_with_localized_exit_year",
        "count",
    ].iloc[0] == 1


def _read_fixture_header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[0].split(",")
