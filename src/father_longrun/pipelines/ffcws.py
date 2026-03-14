from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from father_longrun.config import normalize_path


FFCWS_ASSET_SPECS: tuple[dict[str, str], ...] = (
    {"key": "ffcws_dir", "label": "FFCWS public-use directory", "required": "yes"},
)

FFCWS_PUBLIC_SCOPE = {
    "source": "FFCWS public data documentation",
    "verified_on": "2026-03-13",
    "public_use_scope": "Baseline, Years 1, 3, 5, 9, 15, and 22",
    "latest_public_release_date": "2024-07-31",
    "latest_datafile_update": "2025-03-27",
    "limitation": "Treat FFCWS as early-adult or near-adult rather than mature-adult earnings evidence.",
}

FFCWS_WAVE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Baseline", r"(?i)(?:^|[^a-z])(?:baseline|birth|wave[\s_-]*0|core[_-]?0)(?:[^a-z]|$)"),
    ("Year 1", r"(?i)(?:^|[^a-z])(?:year[\s_-]*1|wave[\s_-]*1|age[\s_-]*1)(?:[^a-z]|$)"),
    ("Year 3", r"(?i)(?:^|[^a-z])(?:year[\s_-]*3|wave[\s_-]*3|age[\s_-]*3)(?:[^a-z]|$)"),
    ("Year 5", r"(?i)(?:^|[^a-z])(?:year[\s_-]*5|wave[\s_-]*5|age[\s_-]*5)(?:[^a-z]|$)"),
    ("Year 9", r"(?i)(?:^|[^a-z])(?:year[\s_-]*9|wave[\s_-]*9|age[\s_-]*9)(?:[^a-z]|$)"),
    ("Year 15", r"(?i)(?:^|[^a-z])(?:year[\s_-]*15|wave[\s_-]*15|age[\s_-]*15)(?:[^a-z]|$)"),
    ("Year 22", r"(?i)(?:^|[^a-z])(?:year[\s_-]*22|wave[\s_-]*22|age[\s_-]*22)(?:[^a-z]|$)"),
)


@dataclass(frozen=True)
class FFCWSAssetStatus:
    key: str
    label: str
    required: bool
    configured: bool
    exists: bool
    placeholder: bool
    public_value: str


@dataclass(frozen=True)
class FFCWSIntakeResult:
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


def _asset_statuses(config: dict[str, Any]) -> tuple[FFCWSAssetStatus, ...]:
    ffcws = config.get("ffcws", {}) if isinstance(config.get("ffcws", {}), dict) else {}
    statuses: list[FFCWSAssetStatus] = []
    for spec in FFCWS_ASSET_SPECS:
        raw_value = ffcws.get(spec["key"])
        configured = isinstance(raw_value, str) and bool(raw_value)
        placeholder = configured and (
            raw_value.startswith("/ABSOLUTE/PATH/TO") or raw_value.startswith("/OPTIONAL/PATH/TO")
        )
        exists = False
        if configured and not placeholder:
            exists = normalize_path(raw_value).exists()
        statuses.append(
            FFCWSAssetStatus(
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
    for label, pattern in FFCWS_WAVE_PATTERNS:
        if any(re.search(pattern, name) for name in names):
            detected.append(label)
    return detected


def build_ffcws_intake_artifacts(*, config: dict[str, Any], output_dir: Path) -> FFCWSIntakeResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = _asset_statuses(config)
    status_by_key = {item.key: item for item in statuses}
    ffcws = config.get("ffcws", {}) if isinstance(config.get("ffcws", {}), dict) else {}

    markdown_path = output_dir / "ffcws_intake.md"
    yaml_path = output_dir / "ffcws_extract_request.yaml"
    manifest_path = output_dir / "ffcws_manifest_draft.csv"

    root_value = ffcws.get("ffcws_dir")
    file_count = 0
    detected_waves: list[str] = []
    if isinstance(root_value, str) and root_value and not root_value.startswith("/ABSOLUTE/PATH/TO"):
        root_path = normalize_path(root_value)
        if root_path.exists():
            files = _iter_public_use_files(root_path)
            file_count = len(files)
            detected_waves = _detect_waves(files)

    lines = [
        "# FFCWS Intake",
        "",
        f"- `use_public_data`: `{ffcws.get('use_public_data', 'unknown')}`",
        f"- official_public_use_scope: `{FFCWS_PUBLIC_SCOPE['public_use_scope']}`",
        f"- latest_public_release_date: `{FFCWS_PUBLIC_SCOPE['latest_public_release_date']}`",
        f"- latest_datafile_update: `{FFCWS_PUBLIC_SCOPE['latest_datafile_update']}`",
        f"- official_source: `{FFCWS_PUBLIC_SCOPE['source']}`",
        f"- usage_limitation: `{FFCWS_PUBLIC_SCOPE['limitation']}`",
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
            "Best role: early-adult / near-adult replication for father involvement, contact, union instability, and transition-to-adulthood outcomes.",
            "Next action: point `ffcws.ffcws_dir` to the downloaded public-use files, confirm the local wave coverage, then review the draft manifest before any harmonized treatment coding.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml_lines = [
        "request_name: ffcws_public_intake_scaffold",
        f"use_public_data: {str(ffcws.get('use_public_data', True)).lower()}",
        "official_scope:",
        f"  source: {FFCWS_PUBLIC_SCOPE['source']}",
        f"  verified_on: {FFCWS_PUBLIC_SCOPE['verified_on']}",
        f"  public_use_scope: {FFCWS_PUBLIC_SCOPE['public_use_scope']}",
        f"  latest_public_release_date: {FFCWS_PUBLIC_SCOPE['latest_public_release_date']}",
        f"  latest_datafile_update: {FFCWS_PUBLIC_SCOPE['latest_datafile_update']}",
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

    template = pd.read_csv(_template_dir() / "ffcws_manifest_template.csv", dtype=str).fillna("")
    template["source_config_key"] = "ffcws_dir"
    template["source_ready"] = "yes" if status_by_key["ffcws_dir"].exists else "no"
    template["status"] = template["source_ready"].map(
        lambda value: "ready_for_mapping" if value == "yes" else "awaiting_local_asset"
    )
    template["detected_waves"] = ", ".join(detected_waves)
    template.to_csv(manifest_path, index=False)
    return FFCWSIntakeResult(markdown_path=markdown_path, yaml_path=yaml_path, manifest_path=manifest_path)
