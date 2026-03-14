from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from father_longrun.utils import repo_root


ENV_NLSY_INTERIM_ROOT = "DADGAP_NLSY_INTERIM_ROOT"
ENV_VENV_PATH = "DADGAP_VENV_PATH"
DEFAULT_ENV_FILE = ".env.local"


@dataclass(frozen=True)
class PathCheck:
    key: str
    raw_value: str
    exists: bool
    kind: str


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected top-level mapping in {path}, got {type(data)!r}")
    return data


def user_cache_root(app_name: str = "dadgap") -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / app_name
    if os.name == "nt":
        local_app_data = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        return local_app_data / app_name / "Cache"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / app_name
    return home / ".cache" / app_name


def normalize_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    anchor = base_dir or repo_root()
    return (anchor / path).resolve()


def resolve_runtime_paths(config: dict[str, Any] | None = None) -> dict[str, Path]:
    root = repo_root()
    configured = (config or {}).get("paths", {})
    defaults = {
        "data_root": root / "data",
        "raw_root": root / "data" / "raw",
        "external_root": root / "data" / "external",
        "interim_root": root / "data" / "interim",
        "processed_root": root / "data" / "processed",
        "outputs_root": root / "outputs",
        "cache_root": user_cache_root(),
    }

    resolved: dict[str, Path] = {}
    for key, default in defaults.items():
        raw_value = configured.get(key)
        if isinstance(raw_value, str) and raw_value and not raw_value.startswith("/ABSOLUTE/PATH/TO"):
            resolved[key] = normalize_path(raw_value, base_dir=root)
        else:
            resolved[key] = default
    return resolved


def resolve_nlsy_interim_root(config: dict[str, Any] | None = None) -> Path | None:
    env_value = os.environ.get(ENV_NLSY_INTERIM_ROOT)
    if env_value:
        return normalize_path(env_value)

    nlsy_config = (config or {}).get("nlsy", {})
    raw_value = nlsy_config.get("fallback_interim_root")
    if isinstance(raw_value, str) and raw_value and not raw_value.startswith("/ABSOLUTE/PATH/TO"):
        return normalize_path(raw_value)
    return None


def resolve_project_venv_path() -> Path:
    env_value = os.environ.get(ENV_VENV_PATH)
    if env_value:
        return normalize_path(env_value)
    return normalize_path("~/venvs/dadgap")


def load_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            loaded[key] = value
    return loaded


def apply_env_overrides(env_path: str | Path | None = None) -> dict[str, str]:
    path = Path(env_path) if env_path is not None else repo_root() / DEFAULT_ENV_FILE
    loaded = load_env_file(path)
    for key, value in loaded.items():
        os.environ.setdefault(key, value)
    return loaded


def _iter_path_like_values(mapping: dict[str, Any], prefix: str = "") -> Iterable[tuple[str, str]]:
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _iter_path_like_values(value, prefix=full_key)
        elif isinstance(value, str) and (
            key.endswith("_path")
            or key.endswith("_dir")
            or key.endswith("_root")
        ):
            yield full_key, value


def validate_paths(config: dict[str, Any]) -> list[PathCheck]:
    checks: list[PathCheck] = []
    root = repo_root()
    for key, raw_value in _iter_path_like_values(config):
        if not raw_value or raw_value.startswith("/ABSOLUTE/PATH/TO"):
            checks.append(PathCheck(key=key, raw_value=raw_value, exists=False, kind="placeholder"))
            continue
        path = normalize_path(raw_value, base_dir=root)
        kind = "directory" if path.suffix == "" else "file"
        checks.append(PathCheck(key=key, raw_value=raw_value, exists=path.exists(), kind=kind))
    return checks
