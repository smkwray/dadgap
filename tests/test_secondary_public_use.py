from pathlib import Path

import pandas as pd

from father_longrun.pipelines.add_health import build_add_health_intake_artifacts
from father_longrun.pipelines.ffcws import build_ffcws_intake_artifacts


def test_build_add_health_intake_artifacts(tmp_path: Path) -> None:
    add_health_dir = tmp_path / "add_health_public"
    add_health_dir.mkdir(parents=True)
    (add_health_dir / "wave1_public.dta").write_text("stub", encoding="utf-8")
    (add_health_dir / "wave4_weights.csv").write_text("stub", encoding="utf-8")
    (add_health_dir / "wave6_public.zip").write_text("stub", encoding="utf-8")

    config = {
        "add_health": {
            "use_public_data": True,
            "add_health_dir": str(add_health_dir),
            "wave_manifest_notes": "public-use waves only",
        }
    }

    result = build_add_health_intake_artifacts(config=config, output_dir=tmp_path / "outputs")

    assert result.markdown_path.exists()
    assert result.yaml_path.exists()
    assert result.manifest_path.exists()
    markdown = result.markdown_path.read_text(encoding="utf-8")
    yaml_payload = result.yaml_path.read_text(encoding="utf-8")
    manifest = pd.read_csv(result.manifest_path)

    assert str(tmp_path) not in markdown
    assert "<local_path>/" in markdown
    assert "Wave I" in markdown
    assert "Wave IV" in markdown
    assert "Wave VI" in markdown
    assert "Waves I-VI" in yaml_payload
    assert {"source_config_key", "source_ready", "status", "detected_waves"} <= set(manifest.columns)
    assert set(manifest["source_ready"]) == {"yes"}


def test_build_ffcws_intake_artifacts(tmp_path: Path) -> None:
    ffcws_dir = tmp_path / "ffcws_public"
    ffcws_dir.mkdir(parents=True)
    (ffcws_dir / "baseline_core.dta").write_text("stub", encoding="utf-8")
    (ffcws_dir / "year9_child.csv").write_text("stub", encoding="utf-8")
    (ffcws_dir / "year22_adult.zip").write_text("stub", encoding="utf-8")

    config = {
        "ffcws": {
            "use_public_data": True,
            "ffcws_dir": str(ffcws_dir),
        }
    }

    result = build_ffcws_intake_artifacts(config=config, output_dir=tmp_path / "outputs")

    assert result.markdown_path.exists()
    assert result.yaml_path.exists()
    assert result.manifest_path.exists()
    markdown = result.markdown_path.read_text(encoding="utf-8")
    yaml_payload = result.yaml_path.read_text(encoding="utf-8")
    manifest = pd.read_csv(result.manifest_path)

    assert str(tmp_path) not in markdown
    assert "<local_path>/" in markdown
    assert "Baseline" in markdown
    assert "Year 9" in markdown
    assert "Year 22" in markdown
    assert "Baseline, Years 1, 3, 5, 9, 15, and 22" in yaml_payload
    assert {"source_config_key", "source_ready", "status", "detected_waves"} <= set(manifest.columns)
    assert set(manifest["source_ready"]) == {"yes"}
