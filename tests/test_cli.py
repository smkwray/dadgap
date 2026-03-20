from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from father_longrun.cli import app
from father_longrun.pipelines.contracts import (
    RESULTS_SCHEMA_VERSION,
    SITE_PAYLOAD_VERSION,
    build_canonical_results_payload,
    build_site_results_payload,
)


runner = CliRunner()


def test_cli_help_smoke() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output
    assert "runtime-info" in result.output


def test_cli_runtime_info_smoke() -> None:
    result = runner.invoke(app, ["runtime-info"])

    assert result.exit_code == 0
    assert "Resolved runtime paths" in result.output
    assert "venv_path" in result.output


def test_cli_source_status_smoke() -> None:
    result = runner.invoke(app, ["source-status"])

    assert result.exit_code == 0
    assert "External public-data source status" in result.output
    assert "Configured" in result.output


def test_cli_check_config_smoke(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    nlsy_root = tmp_path / "nlsy"
    add_health_dir = tmp_path / "add_health"
    outputs_root.mkdir()
    nlsy_root.mkdir()
    add_health_dir.mkdir()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  outputs_root: {outputs_root}",
                "nlsy:",
                f"  fallback_interim_root: {nlsy_root}",
                "add_health:",
                f"  add_health_dir: {add_health_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["check-config", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Path validation" in result.output
    assert "All checked paths exist." in result.output


def _write_doctor_config(tmp_path: Path, *, outputs_root: Path, nlsy_root: Path, add_health_dir: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  outputs_root: {outputs_root}",
                "nlsy:",
                f"  fallback_interim_root: {nlsy_root}",
                "add_health:",
                f"  add_health_dir: {add_health_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_doctor_artifacts(tmp_path: Path, outputs_root: Path) -> None:
    manifests_root = outputs_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (manifests_root / "results_appendix_manifest.csv").write_text("artifact,path,purpose\n", encoding="utf-8")
    (manifests_root / "results.json").write_text(
        json.dumps(
            build_canonical_results_payload(
                artifacts=[],
                source_manifest="results_appendix_manifest.csv",
                synthesis_artifacts=["outputs/manifests/cross_cohort_synthesis.md"],
                generated_at="2026-03-20T00:00:00Z",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (manifests_root / "cross_cohort_synthesis_summary.csv").write_text("artifact\n", encoding="utf-8")
    (manifests_root / "cross_cohort_synthesis_forest_ready.csv").write_text("artifact\n", encoding="utf-8")
    (manifests_root / "cross_cohort_synthesis.md").write_text("# memo\n", encoding="utf-8")
    (tmp_path / "docs" / "results.json").write_text(
        json.dumps(
            build_site_results_payload(
                artifacts=[],
                pages={"home": {}, "prevalence": {}, "outcomes": {}, "faq": {}},
                tables={},
                memos={},
                source_manifest="outputs/manifests/results.json",
                synthesis_artifacts=[
                    "outputs/manifests/cross_cohort_synthesis_summary.csv",
                    "outputs/manifests/cross_cohort_synthesis_forest_ready.csv",
                    "outputs/manifests/cross_cohort_synthesis.md",
                ],
                generated_at="2026-03-20T00:00:00Z",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_cli_doctor_passes_when_config_and_artifacts_exist(tmp_path: Path, monkeypatch) -> None:
    outputs_root = tmp_path / "outputs"
    nlsy_root = tmp_path / "nlsy"
    add_health_dir = tmp_path / "add_health"
    outputs_root.mkdir()
    nlsy_root.mkdir()
    add_health_dir.mkdir()
    config_path = _write_doctor_config(tmp_path, outputs_root=outputs_root, nlsy_root=nlsy_root, add_health_dir=add_health_dir)
    _write_doctor_artifacts(tmp_path, outputs_root)

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "dadgap doctor" in result.output
    assert "All required checks passed." in result.output


def test_cli_doctor_fails_when_config_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", str(tmp_path / "missing.yaml")])

    assert result.exit_code == 1
    assert "config_file" in result.output


def test_cli_doctor_fails_when_site_payload_missing(tmp_path: Path, monkeypatch) -> None:
    outputs_root = tmp_path / "outputs"
    nlsy_root = tmp_path / "nlsy"
    add_health_dir = tmp_path / "add_health"
    outputs_root.mkdir()
    nlsy_root.mkdir()
    add_health_dir.mkdir()
    config_path = _write_doctor_config(tmp_path, outputs_root=outputs_root, nlsy_root=nlsy_root, add_health_dir=add_health_dir)
    _write_doctor_artifacts(tmp_path, outputs_root)
    (tmp_path / "docs" / "results.json").unlink()

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "docs_results_json" in result.output


def test_cli_doctor_fails_when_canonical_results_schema_is_invalid(tmp_path: Path, monkeypatch) -> None:
    outputs_root = tmp_path / "outputs"
    nlsy_root = tmp_path / "nlsy"
    add_health_dir = tmp_path / "add_health"
    outputs_root.mkdir()
    nlsy_root.mkdir()
    add_health_dir.mkdir()
    config_path = _write_doctor_config(tmp_path, outputs_root=outputs_root, nlsy_root=nlsy_root, add_health_dir=add_health_dir)
    _write_doctor_artifacts(tmp_path, outputs_root)
    manifests_root = outputs_root / "manifests"
    bad_payload = json.loads((manifests_root / "results.json").read_text(encoding="utf-8"))
    bad_payload["schema_version"] = "broken"
    (manifests_root / "results.json").write_text(json.dumps(bad_payload) + "\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "appendix_results_json" in result.output
    assert RESULTS_SCHEMA_VERSION in result.output


def test_cli_doctor_fails_when_site_payload_schema_is_invalid(tmp_path: Path, monkeypatch) -> None:
    outputs_root = tmp_path / "outputs"
    nlsy_root = tmp_path / "nlsy"
    add_health_dir = tmp_path / "add_health"
    outputs_root.mkdir()
    nlsy_root.mkdir()
    add_health_dir.mkdir()
    config_path = _write_doctor_config(tmp_path, outputs_root=outputs_root, nlsy_root=nlsy_root, add_health_dir=add_health_dir)
    _write_doctor_artifacts(tmp_path, outputs_root)
    docs_results = tmp_path / "docs" / "results.json"
    bad_payload = json.loads(docs_results.read_text(encoding="utf-8"))
    bad_payload["site_payload_version"] = "broken"
    docs_results.write_text(json.dumps(bad_payload) + "\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "docs_results_json" in result.output
    assert SITE_PAYLOAD_VERSION in result.output
