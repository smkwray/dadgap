from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from father_longrun.config import normalize_path


ADD_HEALTH_ASSET_SPECS: tuple[dict[str, str], ...] = (
    {"key": "add_health_dir", "label": "Add Health public-use directory", "required": "yes"},
)

ADD_HEALTH_PUBLIC_SCOPE = {
    "source": "ICPSR study 21600",
    "verified_on": "2026-03-13",
    "public_use_scope": "Waves I-VI",
    "latest_release_date": "2026-03-03",
    "limitation": "Public-use is reduced-sample and does not support restricted-use friend or sibling linkage.",
}

ADD_HEALTH_WAVE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Wave I", r"(?i)(?:^|[^a-z])(?:wave[\s_-]*1|wave[\s_-]*i|w1)(?:[^a-z]|$)"),
    ("Wave II", r"(?i)(?:^|[^a-z])(?:wave[\s_-]*2|wave[\s_-]*ii|w2)(?:[^a-z]|$)"),
    ("Wave III", r"(?i)(?:^|[^a-z])(?:wave[\s_-]*3|wave[\s_-]*iii|w3)(?:[^a-z]|$)"),
    ("Wave IV", r"(?i)(?:^|[^a-z])(?:wave[\s_-]*4|wave[\s_-]*iv|w4)(?:[^a-z]|$)"),
    ("Wave V", r"(?i)(?:^|[^a-z])(?:wave[\s_-]*5|wave[\s_-]*v|w5)(?:[^a-z]|$)"),
    ("Wave VI", r"(?i)(?:^|[^a-z])(?:wave[\s_-]*6|wave[\s_-]*vi|w6)(?:[^a-z]|$)"),
)


@dataclass(frozen=True)
class AddHealthAssetStatus:
    key: str
    label: str
    required: bool
    configured: bool
    exists: bool
    placeholder: bool
    public_value: str


@dataclass(frozen=True)
class AddHealthIntakeResult:
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


def _asset_statuses(config: dict[str, Any]) -> tuple[AddHealthAssetStatus, ...]:
    add_health = config.get("add_health", {}) if isinstance(config.get("add_health", {}), dict) else {}
    statuses: list[AddHealthAssetStatus] = []
    for spec in ADD_HEALTH_ASSET_SPECS:
        raw_value = add_health.get(spec["key"])
        configured = isinstance(raw_value, str) and bool(raw_value)
        placeholder = configured and (
            raw_value.startswith("/ABSOLUTE/PATH/TO") or raw_value.startswith("/OPTIONAL/PATH/TO")
        )
        exists = False
        if configured and not placeholder:
            exists = normalize_path(raw_value).exists()
        statuses.append(
            AddHealthAssetStatus(
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


def _iter_public_use_files(root: Path) -> list[Path]:
    allowed_suffixes = {".csv", ".dta", ".sav", ".sas7bdat", ".xpt", ".por", ".txt", ".zip"}
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )


def _detect_waves(paths: list[Path]) -> list[str]:
    detected: list[str] = []
    names = [path.name for path in paths]
    for label, pattern in ADD_HEALTH_WAVE_PATTERNS:
        if any(re.search(pattern, name) for name in names):
            detected.append(label)
    return detected


def build_add_health_intake_artifacts(*, config: dict[str, Any], output_dir: Path) -> AddHealthIntakeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = _asset_statuses(config)
    status_by_key = {item.key: item for item in statuses}
    add_health = config.get("add_health", {}) if isinstance(config.get("add_health", {}), dict) else {}

    markdown_path = output_dir / "add_health_intake.md"
    yaml_path = output_dir / "add_health_extract_request.yaml"
    manifest_path = output_dir / "add_health_manifest_draft.csv"

    root_value = add_health.get("add_health_dir")
    file_count = 0
    detected_waves: list[str] = []
    if isinstance(root_value, str) and root_value and not root_value.startswith("/ABSOLUTE/PATH/TO"):
        root_path = normalize_path(root_value)
        if root_path.exists():
            files = _iter_public_use_files(root_path)
            file_count = len(files)
            detected_waves = _detect_waves(files)

    lines = [
        "# Add Health Intake",
        "",
        f"- `use_public_data`: `{add_health.get('use_public_data', 'unknown')}`",
        f"- `wave_manifest_notes`: `{add_health.get('wave_manifest_notes', '-')}`",
        f"- official_public_use_scope: `{ADD_HEALTH_PUBLIC_SCOPE['public_use_scope']}`",
        f"- latest_official_release_date: `{ADD_HEALTH_PUBLIC_SCOPE['latest_release_date']}`",
        f"- official_source: `{ADD_HEALTH_PUBLIC_SCOPE['source']}`",
        f"- public_use_limitation: `{ADD_HEALTH_PUBLIC_SCOPE['limitation']}`",
        f"- local_file_count: `{file_count}`",
        f"- detected_waves: `{', '.join(detected_waves) if detected_waves else 'none_detected'}`",
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
    lines.extend(
        [
            "",
            "Best role: adult replication of employment, income-adjacent, education, and family-formation outcomes under public-use constraints.",
            "Next action: point `add_health.add_health_dir` to the downloaded public-use files, confirm locally detected waves, then review the draft manifest before any harmonization.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_lines = [
        "request_name: add_health_public_intake_scaffold",
        f"use_public_data: {str(add_health.get('use_public_data', True)).lower()}",
        "official_scope:",
        f"  source: {ADD_HEALTH_PUBLIC_SCOPE['source']}",
        f"  verified_on: {ADD_HEALTH_PUBLIC_SCOPE['verified_on']}",
        f"  public_use_scope: {ADD_HEALTH_PUBLIC_SCOPE['public_use_scope']}",
        f"  latest_release_date: {ADD_HEALTH_PUBLIC_SCOPE['latest_release_date']}",
        "required_assets:",
    ]
    for item in statuses:
        yaml_lines.append(f"  - config_key: {item.key}")
        yaml_lines.append(f"    label: {item.label}")
        yaml_lines.append(f"    status: {'ready' if item.exists else 'needs_local_path'}")
    yaml_lines.append("local_inventory:")
    yaml_lines.append(f"  file_count: {file_count}")
    yaml_lines.append(f"  detected_waves: [{', '.join(detected_waves)}]" if detected_waves else "  detected_waves: []")
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    template = pd.read_csv(_template_dir() / "add_health_manifest_template.csv", dtype=str).fillna("")
    template["source_config_key"] = "add_health_dir"
    template["source_ready"] = "yes" if status_by_key["add_health_dir"].exists else "no"
    template["status"] = template["source_ready"].map(
        lambda value: "ready_for_mapping" if value == "yes" else "awaiting_local_asset"
    )
    template["detected_waves"] = ", ".join(detected_waves)
    template.to_csv(manifest_path, index=False)
    return AddHealthIntakeResult(markdown_path=markdown_path, yaml_path=yaml_path, manifest_path=manifest_path)
