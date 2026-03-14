from pathlib import Path

from father_longrun.config import load_yaml, resolve_runtime_paths, validate_paths


def test_example_config_loads() -> None:
    config = load_yaml(Path("config/user_inputs.example.yaml"))
    assert "nlsy" in config
    assert "analysis" in config


def test_example_config_has_path_checks() -> None:
    config = load_yaml(Path("config/user_inputs.example.yaml"))
    checks = validate_paths(config)
    assert checks
    assert any(item.key.endswith("nlsy79_main_path") for item in checks)


def test_runtime_paths_use_external_cache() -> None:
    paths = resolve_runtime_paths({})
    assert paths["cache_root"].name == "dadgap"
