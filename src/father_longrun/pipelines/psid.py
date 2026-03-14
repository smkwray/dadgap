from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from father_longrun.config import normalize_path


PSID_ASSET_SPECS: tuple[dict[str, str], ...] = (
    {"key": "psid_main_dir", "label": "PSID main directory", "required": "yes"},
    {"key": "psid_shelf_path", "label": "PSID-SHELF file", "required": "yes"},
    {"key": "parent_identification_path", "label": "Parent Identification file", "required": "yes"},
    {"key": "childbirth_adoption_history_path", "label": "Childbirth/Adoption History file", "required": "yes"},
    {"key": "marriage_history_path", "label": "Marriage History file", "required": "yes"},
    {"key": "cds_path", "label": "Child Development Supplement", "required": "no"},
    {"key": "tas_path", "label": "Transition into Adulthood Supplement", "required": "no"},
)

PSID_SOURCE_TO_CONFIG_KEY: dict[str, str] = {
    "psid_main": "psid_main_dir",
    "psid_shelf": "psid_shelf_path",
    "parent_identification": "parent_identification_path",
    "childbirth_history": "childbirth_adoption_history_path",
    "marriage_history": "marriage_history_path",
}


@dataclass(frozen=True)
class PSIDAssetStatus:
    key: str
    label: str
    required: bool
    configured: bool
    exists: bool
    placeholder: bool
    public_value: str


@dataclass(frozen=True)
class PSIDIntakeResult:
    markdown_path: Path
    yaml_path: Path
    manifest_path: Path


def _public_value(raw_value: str | None) -> str:
    if not raw_value:
        return "-"
    if raw_value.startswith("/ABSOLUTE/PATH/TO") or raw_value.startswith("/OPTIONAL/PATH/TO"):
        return raw_value
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        return raw_value
    suffix = path.parts[-2:] if len(path.parts) >= 2 else path.parts
    return f"<local_path>/{'/'.join(suffix)}"


def _asset_statuses(config: dict[str, Any]) -> tuple[PSIDAssetStatus, ...]:
    psid = config.get("psid", {}) if isinstance(config.get("psid", {}), dict) else {}
    statuses: list[PSIDAssetStatus] = []
    for spec in PSID_ASSET_SPECS:
        raw_value = psid.get(spec["key"])
        configured = isinstance(raw_value, str) and bool(raw_value)
        placeholder = configured and (
            raw_value.startswith("/ABSOLUTE/PATH/TO") or raw_value.startswith("/OPTIONAL/PATH/TO")
        )
        exists = False
        if configured and not placeholder:
            exists = normalize_path(raw_value).exists()
        statuses.append(
            PSIDAssetStatus(
                key=spec["key"],
                label=spec["label"],
                required=spec["required"] == "yes",
                configured=configured,
                exists=exists,
                placeholder=placeholder,
                public_value=_public_value(raw_value if isinstance(raw_value, str) else None),
            )
        )
    return tuple(statuses)


def _template_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "templates"


def build_psid_intake_artifacts(*, config: dict[str, Any], output_dir: Path) -> PSIDIntakeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = _asset_statuses(config)
    status_by_key = {item.key: item for item in statuses}
    psid = config.get("psid", {}) if isinstance(config.get("psid", {}), dict) else {}

    markdown_path = output_dir / "psid_intake.md"
    yaml_path = output_dir / "psid_extract_request.yaml"
    manifest_path = output_dir / "psid_manifest_draft.csv"

    lines = [
        "# PSID Intake",
        "",
        f"- `download_or_register_now`: `{psid.get('download_or_register_now', 'unknown')}`",
        f"- `registration_complete`: `{psid.get('registration_complete', 'unknown')}`",
        "",
        "| config_key | required | configured | exists | placeholder | value |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in statuses:
        lines.append(
            f"| {item.key} | {'yes' if item.required else 'no'} | "
            f"{'yes' if item.configured else 'no'} | {'yes' if item.exists else 'no'} | "
            f"{'yes' if item.placeholder else 'no'} | {item.public_value} |"
        )
    lines.append("")
    lines.append("Next action: populate missing PSID paths locally, then start cohort-specific manifest review before any cross-cohort synthesis.")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_lines = [
        "request_name: psid_intake_scaffold",
        f"registration_complete: {str(psid.get('registration_complete', False)).lower()}",
        "required_assets:",
    ]
    for item in statuses:
        if not item.required:
            continue
        yaml_lines.append(f"  - config_key: {item.key}")
        yaml_lines.append(f"    label: {item.label}")
        yaml_lines.append(f"    status: {'ready' if item.exists else 'needs_local_path'}")
    yaml_lines.append("optional_assets:")
    for item in statuses:
        if item.required:
            continue
        yaml_lines.append(f"  - config_key: {item.key}")
        yaml_lines.append(f"    label: {item.label}")
        yaml_lines.append(f"    status: {'ready' if item.exists else 'optional_not_ready'}")
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    template = pd.read_csv(_template_dir() / "psid_manifest_template.csv", dtype=str).fillna("")
    template["source_config_key"] = template["source_file"].map(PSID_SOURCE_TO_CONFIG_KEY).fillna("")
    template["source_ready"] = template["source_config_key"].map(
        lambda key: "yes" if key and status_by_key.get(key) and status_by_key[key].exists else "no"
    )
    template["status"] = template["source_ready"].map(lambda value: "ready_for_mapping" if value == "yes" else "awaiting_local_asset")
    template.to_csv(manifest_path, index=False)
    return PSIDIntakeResult(markdown_path=markdown_path, yaml_path=yaml_path, manifest_path=manifest_path)
