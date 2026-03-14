from pathlib import Path

import pandas as pd

from father_longrun.pipelines.psid import build_psid_intake_artifacts


def test_build_psid_intake_artifacts(tmp_path: Path) -> None:
    main_dir = tmp_path / "psid" / "main"
    main_dir.mkdir(parents=True)
    shelf_path = tmp_path / "psid" / "psid_shelf.dta"
    shelf_path.write_text("stub", encoding="utf-8")

    config = {
        "psid": {
            "download_or_register_now": True,
            "registration_complete": False,
            "psid_main_dir": str(main_dir),
            "psid_shelf_path": str(shelf_path),
            "parent_identification_path": "/ABSOLUTE/PATH/TO/PSID/parent_identification.dta",
            "childbirth_adoption_history_path": "/ABSOLUTE/PATH/TO/PSID/childbirth_history.dta",
            "marriage_history_path": "/ABSOLUTE/PATH/TO/PSID/marriage_history.dta",
        }
    }

    result = build_psid_intake_artifacts(config=config, output_dir=tmp_path / "outputs")

    assert result.markdown_path.exists()
    assert result.yaml_path.exists()
    assert result.manifest_path.exists()
    markdown = result.markdown_path.read_text(encoding="utf-8")
    yaml_payload = result.yaml_path.read_text(encoding="utf-8")
    manifest = pd.read_csv(result.manifest_path)

    assert str(tmp_path) not in markdown
    assert "<local_path>/" in markdown
    assert "needs_local_path" in yaml_payload
    assert {"source_config_key", "source_ready", "status"} <= set(manifest.columns)
