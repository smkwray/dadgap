from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULTS_SCHEMA_VERSION = "1.0"
SITE_PAYLOAD_VERSION = "1.0"
REQUIRED_MANIFEST_COLUMNS = ("artifact", "path", "purpose")
REQUIRED_SITE_PAGES = ("home", "prevalence", "outcomes", "faq")


def generated_at_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_canonical_results_payload(
    *,
    artifacts: list[dict[str, Any]],
    source_manifest: str,
    synthesis_artifacts: list[str],
    generated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_at_utc": generated_at or generated_at_utc(),
        "source_manifest": source_manifest,
        "synthesis_artifacts": synthesis_artifacts,
        "artifacts": artifacts,
    }


def build_site_results_payload(
    *,
    artifacts: list[dict[str, Any]],
    pages: dict[str, Any],
    tables: dict[str, Any],
    memos: dict[str, Any],
    source_manifest: str,
    synthesis_artifacts: list[str],
    generated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "site_payload_version": SITE_PAYLOAD_VERSION,
        "generated_at_utc": generated_at or generated_at_utc(),
        "source_manifest": source_manifest,
        "synthesis_artifacts": synthesis_artifacts,
        "artifacts": artifacts,
        "pages": pages,
        "tables": tables,
        "memos": memos,
    }


def _validate_common_metadata(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != RESULTS_SCHEMA_VERSION:
        errors.append(f"schema_version must equal {RESULTS_SCHEMA_VERSION!r}")
    generated = payload.get("generated_at_utc")
    if not isinstance(generated, str) or not generated.strip():
        errors.append("generated_at_utc must be a non-empty string")
    source_manifest = payload.get("source_manifest")
    if not isinstance(source_manifest, str) or not source_manifest.strip():
        errors.append("source_manifest must be a non-empty string")
    synthesis_artifacts = payload.get("synthesis_artifacts")
    if not isinstance(synthesis_artifacts, list) or not all(isinstance(item, str) and item.strip() for item in synthesis_artifacts):
        errors.append("synthesis_artifacts must be a list of non-empty strings")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifacts must be a list")
    else:
        for index, row in enumerate(artifacts):
            if not isinstance(row, dict):
                errors.append(f"artifacts[{index}] must be an object")
                continue
            missing = [column for column in REQUIRED_MANIFEST_COLUMNS if not isinstance(row.get(column), str) or not row.get(column, "").strip()]
            if missing:
                errors.append(f"artifacts[{index}] is missing required string fields: {', '.join(missing)}")
    return errors


def validate_canonical_results_payload(payload: dict[str, Any]) -> list[str]:
    return _validate_common_metadata(payload)


def validate_site_results_payload(payload: dict[str, Any]) -> list[str]:
    errors = _validate_common_metadata(payload)
    if payload.get("site_payload_version") != SITE_PAYLOAD_VERSION:
        errors.append(f"site_payload_version must equal {SITE_PAYLOAD_VERSION!r}")
    pages = payload.get("pages")
    if not isinstance(pages, dict):
        errors.append("pages must be an object")
    else:
        for page in REQUIRED_SITE_PAGES:
            if not isinstance(pages.get(page), dict):
                errors.append(f"pages.{page} must be an object")
    if not isinstance(payload.get("tables"), dict):
        errors.append("tables must be an object")
    if not isinstance(payload.get("memos"), dict):
        errors.append("memos must be an object")
    return errors


def validate_manifest_frame_columns(columns: list[str] | tuple[str, ...]) -> list[str]:
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in columns]
    if not missing:
        return []
    return [f"manifest is missing required columns: {', '.join(missing)}"]


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
