from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from father_longrun.config import (
    apply_env_overrides,
    load_yaml,
    resolve_nlsy_interim_root,
    resolve_project_venv_path,
    resolve_runtime_paths,
    validate_paths,
)
from father_longrun.models.ml import build_ml_benchmarks
from father_longrun.models.quasi_causal import build_quasi_causal_scaffold
from father_longrun.pipelines.nlsy import (
    build_backbone_scaffold,
    build_nlsy_pilot,
    build_phase0_artifacts,
    build_merge_contract_report,
    build_refresh_spec,
    build_analysis_ready_treatment_layers,
    build_nlsy97_fatherlessness_profiles,
    build_nlsy97_longitudinal_panel_scaffold,
    build_treatment_candidate_layers,
    build_treatment_refresh_extracts,
    build_reviewed_layers,
    discover_cohort_extracts,
    write_inventory_report,
)
from father_longrun.pipelines.add_health import build_add_health_intake_artifacts
from father_longrun.pipelines.ffcws import build_ffcws_intake_artifacts
from father_longrun.pipelines.psid import build_psid_intake_artifacts
from father_longrun.pipelines.public_benchmarks import (
    build_ipums_extract_workflow,
    build_cross_cohort_benchmark_comparison,
    build_public_benchmark_profiles,
    build_public_microdata_artifacts,
    build_public_benchmark_snapshot,
    source_statuses,
)
from father_longrun.pipelines.reporting import build_results_appendix
from father_longrun.questions import render_questions_markdown

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
apply_env_overrides()


@app.command("print-questions")
def print_questions() -> None:
    """Print the one-shot intake questionnaire for the user."""
    console.print(render_questions_markdown())


@app.command("runtime-info")
def runtime_info() -> None:
    """Print resolved runtime paths and environment locations."""
    config_path = Path("config/user_inputs.local.yaml")
    data = load_yaml(config_path) if config_path.exists() else {}
    runtime_paths = resolve_runtime_paths(data)

    table = Table(title="Resolved runtime paths")
    table.add_column("Key")
    table.add_column("Value")
    for key, value in runtime_paths.items():
        table.add_row(key, str(value))
    table.add_row("venv_path", str(resolve_project_venv_path()))

    interim_root = resolve_nlsy_interim_root(data)
    if interim_root is not None:
        table.add_row("nlsy_fallback_interim_root", str(interim_root))
    console.print(table)


@app.command("source-status")
def source_status() -> None:
    """Show whether external-source credentials are configured locally."""
    table = Table(title="External public-data source status")
    table.add_column("Source")
    table.add_column("Role")
    table.add_column("Category")
    table.add_column("Env var")
    table.add_column("Configured")

    for item in source_statuses():
        table.add_row(
            item.label,
            item.role,
            item.category,
            item.env_var,
            "yes" if item.configured else "no",
        )
    console.print(table)


@app.command("check-config")
def check_config(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    """Validate path-like values in a YAML config file."""
    data = load_yaml(config)
    checks = validate_paths(data)

    table = Table(title=f"Path validation: {config}")
    table.add_column("Key")
    table.add_column("Kind")
    table.add_column("Exists")
    table.add_column("Value")

    missing = 0
    for item in checks:
        table.add_row(item.key, item.kind, "yes" if item.exists else "no", item.raw_value)
        if not item.exists:
            missing += 1

    console.print(table)
    if missing:
        console.print(f"[yellow]{missing} path(s) are missing or placeholders.[/yellow]")
        raise typer.Exit(code=1)
    console.print("[green]All checked paths exist.[/green]")


def _load_optional_config(config: Path | None) -> dict[str, object]:
    if config is None:
        return {}
    if not config.exists():
        raise typer.BadParameter(f"Config file does not exist: {config}")
    return load_yaml(config)


@app.command("inspect-nlsy")
def inspect_nlsy(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
    write_report: bool = typer.Option(True, "--write-report/--no-write-report"),
) -> None:
    """Inspect NLSY cohort extracts discovered from the configured fallback directory."""
    data = _load_optional_config(config)
    interim_root = resolve_nlsy_interim_root(data)
    if interim_root is None:
        console.print("[red]No NLSY fallback directory configured.[/red]")
        raise typer.Exit(code=1)

    extracts = discover_cohort_extracts(interim_root)
    if not extracts:
        console.print(f"[red]No cohort extracts found under {interim_root}.[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"NLSY extract inventory: {interim_root}")
    table.add_column("Cohort")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    table.add_column("Panel extract")
    for extract in extracts:
        table.add_row(
            extract.cohort,
            str(extract.row_count) if extract.row_count is not None else "?",
            str(extract.column_count),
            str(extract.panel_extract_path),
        )
    console.print(table)

    if write_report:
        runtime_paths = resolve_runtime_paths(data)
        report_dir = runtime_paths["outputs_root"] / "manifests"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_paths = write_inventory_report(
            extracts,
            report_dir=report_dir,
            interim_root=interim_root,
            generated_at=datetime.now(timezone.utc),
        )
        console.print(f"[green]Wrote inventory reports to {report_paths['markdown']} and {report_paths['json']}.[/green]")


@app.command("build-nlsy-pilot")
def build_nlsy_pilot_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Materialize discovered NLSY panel extracts into parquet and write inventory artifacts."""
    data = _load_optional_config(config)
    interim_root = resolve_nlsy_interim_root(data)
    if interim_root is None:
        console.print("[red]No NLSY fallback directory configured.[/red]")
        raise typer.Exit(code=1)

    runtime_paths = resolve_runtime_paths(data)
    result = build_nlsy_pilot(
        interim_root=interim_root,
        processed_root=runtime_paths["processed_root"] / "nlsy",
        outputs_root=runtime_paths["outputs_root"] / "manifests",
        overwrite=overwrite,
        generated_at=datetime.now(timezone.utc),
    )

    table = Table(title="NLSY pilot build outputs")
    table.add_column("Cohort")
    table.add_column("Parquet")
    table.add_column("Dictionary")
    for artifact in result.artifacts:
        table.add_row(artifact.cohort, str(artifact.parquet_path), str(artifact.dictionary_path))
    console.print(table)
    console.print(f"[green]Inventory: {result.inventory_markdown_path}[/green]")


@app.command("build-phase0")
def build_phase0_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Generate draft manifests and key diagnostics for Phase 0 review."""
    data = _load_optional_config(config)
    interim_root = resolve_nlsy_interim_root(data)
    if interim_root is None:
        console.print("[red]No NLSY fallback directory configured.[/red]")
        raise typer.Exit(code=1)

    runtime_paths = resolve_runtime_paths(data)
    output_dir = runtime_paths["outputs_root"] / "manifests"
    result = build_phase0_artifacts(interim_root=interim_root, output_dir=output_dir)

    table = Table(title="Phase 0 manifest artifacts")
    table.add_column("Dataset")
    table.add_column("Manifest")
    table.add_column("Rows", justify="right")
    for artifact in result.manifests:
        table.add_row(artifact.dataset_key, str(artifact.manifest_path), str(artifact.row_count))
    console.print(table)
    console.print(f"[green]Diagnostics: {result.diagnostics_markdown_path}[/green]")


@app.command("build-merge-contract")
def build_merge_contract_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build the structural NLSY backbone join contract from confirmed key columns."""
    data = _load_optional_config(config)
    interim_root = resolve_nlsy_interim_root(data)
    if interim_root is None:
        console.print("[red]No NLSY fallback directory configured.[/red]")
        raise typer.Exit(code=1)

    runtime_paths = resolve_runtime_paths(data)
    result = build_merge_contract_report(
        interim_root=interim_root,
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    console.print(f"[green]Merge contract: {result.report_path}[/green]")


@app.command("build-backbone-scaffold")
def build_backbone_scaffold_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build the structural NLSY79 -> CNLSY backbone scaffold."""
    data = _load_optional_config(config)
    interim_root = resolve_nlsy_interim_root(data)
    if interim_root is None:
        console.print("[red]No NLSY fallback directory configured.[/red]")
        raise typer.Exit(code=1)

    runtime_paths = resolve_runtime_paths(data)
    result = build_backbone_scaffold(
        interim_root=interim_root,
        processed_root=runtime_paths["processed_root"] / "nlsy",
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    console.print(f"[green]Backbone scaffold: {result.parquet_path}[/green]")


@app.command("build-reviewed-layers")
def build_reviewed_layers_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build reviewed canonical NLSY layers from the existing fallback extracts."""
    data = _load_optional_config(config)
    interim_root = resolve_nlsy_interim_root(data)
    if interim_root is None:
        console.print("[red]No NLSY fallback directory configured.[/red]")
        raise typer.Exit(code=1)

    runtime_paths = resolve_runtime_paths(data)
    result = build_reviewed_layers(
        interim_root=interim_root,
        processed_root=runtime_paths["processed_root"] / "nlsy",
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    table = Table(title="Reviewed NLSY layers")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("mapping_csv", str(result.mapping_csv_path))
    table.add_row("availability_md", str(result.availability_markdown_path))
    table.add_row("exposure_gap_md", str(result.exposure_gap_markdown_path))
    table.add_row("backbone_parquet", str(result.backbone_parquet_path))
    table.add_row("nlsy97_parquet", str(result.nlsy97_parquet_path))
    console.print(table)


@app.command("build-refresh-spec")
def build_refresh_spec_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build the next-pass NLSY extract refresh spec for father-treatment variables."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_refresh_spec(output_dir=runtime_paths["outputs_root"] / "manifests")
    table = Table(title="NLSY treatment refresh spec")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("csv", str(result.csv_path))
    table.add_row("markdown", str(result.markdown_path))
    table.add_row("yaml", str(result.yaml_path))
    console.print(table)


@app.command("refresh-nlsy-treatment-extracts")
def refresh_nlsy_treatment_extracts_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build refreshed NLSY panel extracts with the requested treatment columns added."""
    data = _load_optional_config(config)
    interim_root = resolve_nlsy_interim_root(data)
    if interim_root is None:
        console.print("[red]No NLSY fallback directory configured.[/red]")
        raise typer.Exit(code=1)

    runtime_paths = resolve_runtime_paths(data)
    result = build_treatment_refresh_extracts(
        interim_root=interim_root,
        refreshed_root=runtime_paths["interim_root"] / "nlsy_refresh",
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    table = Table(title="NLSY treatment refresh extracts")
    table.add_column("Cohort")
    table.add_column("Rows", justify="right")
    table.add_column("Added columns")
    for item in result.artifacts:
        table.add_row(item.cohort, str(item.row_count), str(len(item.added_columns)))
    console.print(table)
    console.print(f"[green]Report: {result.report_path}[/green]")


@app.command("build-treatment-candidate-layers")
def build_treatment_candidate_layers_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Attach refreshed treatment candidate columns to the reviewed NLSY layers."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_treatment_candidate_layers(
        refreshed_root=runtime_paths["interim_root"] / "nlsy_refresh",
        processed_root=runtime_paths["processed_root"] / "nlsy",
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    table = Table(title="NLSY treatment candidate layers")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("backbone", str(result.backbone_path))
    table.add_row("nlsy97", str(result.nlsy97_path))
    table.add_row("mapping_csv", str(result.mapping_path))
    table.add_row("value_counts_csv", str(result.value_counts_path))
    console.print(table)


@app.command("build-analysis-ready-treatments")
def build_analysis_ready_treatments_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Code the first analysis-ready NLSY treatment measures and baseline tables."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_analysis_ready_treatment_layers(
        processed_root=runtime_paths["processed_root"] / "nlsy",
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    table = Table(title="Analysis-ready NLSY treatment outputs")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("backbone", str(result.backbone_path))
    table.add_row("nlsy97", str(result.nlsy97_path))
    table.add_row("nlsy97_baseline", str(result.nlsy97_baseline_path))
    table.add_row("nlsy97_primary_baseline", str(result.nlsy97_primary_baseline_path))
    table.add_row("cnlsy_subset", str(result.cnlsy_subset_path))
    table.add_row("cnlsy_baseline", str(result.cnlsy_baseline_path))
    table.add_row("cnlsy_primary_baseline", str(result.cnlsy_primary_baseline_path))
    table.add_row("cnlsy_readiness", str(result.cnlsy_readiness_path))
    table.add_row("cnlsy_outcome_tiering", str(result.cnlsy_outcome_tiering_path))
    table.add_row("cnlsy_education_validation", str(result.cnlsy_education_validation_path))
    table.add_row("cnlsy_education_crosstab", str(result.cnlsy_education_crosstab_path))
    table.add_row("cnlsy_attainment_codebook", str(result.cnlsy_attainment_codebook_path))
    table.add_row("coding_rules", str(result.coding_rules_path))
    table.add_row("summary", str(result.summary_path))
    console.print(table)


@app.command("build-fatherlessness-profiles")
def build_fatherlessness_profiles_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build descriptive NLSY97 fatherlessness profiles by race, sex, and socioeconomic background."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_nlsy97_fatherlessness_profiles(
        processed_root=runtime_paths["processed_root"] / "nlsy",
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    table = Table(title="NLSY97 fatherlessness profiles")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("group_summary", str(result.group_summary_path))
    table.add_row("predictors", str(result.predictor_path))
    table.add_row("report", str(result.report_path))
    console.print(table)


@app.command("build-nlsy97-longitudinal-panel")
def build_nlsy97_longitudinal_panel_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build the multiwave NLSY97 panel and childhood-history scaffold for panel/event-time follow-on work."""
    data = _load_optional_config(config)
    interim_root = resolve_nlsy_interim_root(data)
    if interim_root is None:
        console.print("[red]No NLSY fallback directory configured.[/red]")
        raise typer.Exit(code=1)

    runtime_paths = resolve_runtime_paths(data)
    result = build_nlsy97_longitudinal_panel_scaffold(
        interim_root=interim_root,
        processed_root=runtime_paths["processed_root"] / "nlsy",
        output_dir=runtime_paths["outputs_root"] / "models",
    )
    table = Table(title="NLSY97 longitudinal panel scaffold")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("panel", str(result.panel_path))
    table.add_row("childhood_history", str(result.childhood_history_path))
    table.add_row("availability", str(result.availability_path))
    table.add_row("childhood_availability", str(result.childhood_availability_path))
    table.add_row("summary", str(result.summary_path))
    console.print(table)


@app.command("build-quasi-causal-scaffold")
def build_quasi_causal_scaffold_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build quasi-causal readiness artifacts and the first sibling-FE workbench."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_quasi_causal_scaffold(
        processed_root=runtime_paths["processed_root"] / "nlsy",
        output_dir=runtime_paths["outputs_root"] / "models",
    )
    table = Table(title="Quasi-causal scaffold artifacts")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("sibling_design", str(result.sibling_design_path))
    table.add_row("sibling_fe", str(result.sibling_fe_path))
    table.add_row("event_time", str(result.event_time_path))
    table.add_row("event_time_design", str(result.event_time_design_path))
    table.add_row("event_time_window_summary", str(result.event_time_window_summary_path))
    table.add_row("event_time_comparison_candidates", str(result.event_time_comparison_candidates_path))
    table.add_row("event_time_comparison_support", str(result.event_time_comparison_support_path))
    table.add_row("event_time_strategy", str(result.event_time_strategy_path))
    table.add_row("event_time_post_only_design", str(result.event_time_post_only_design_path))
    table.add_row("event_time_post_only_summary", str(result.event_time_post_only_summary_path))
    table.add_row("event_time_post_only_robustness", str(result.event_time_post_only_robustness_path))
    table.add_row("event_time_post_only_sensitivity", str(result.event_time_post_only_sensitivity_path))
    table.add_row("event_time_post_only_sensitivity_report", str(result.event_time_post_only_sensitivity_report_path))
    table.add_row("event_time_post_only_preferred_summary", str(result.event_time_post_only_preferred_summary_path))
    table.add_row("event_time_post_only_preferred_report", str(result.event_time_post_only_preferred_report_path))
    table.add_row("event_time_post_only_report", str(result.event_time_post_only_report_path))
    table.add_row("readiness", str(result.readiness_path))
    table.add_row("summary", str(result.summary_path))
    console.print(table)


@app.command("build-ml-benchmarks")
def build_ml_benchmarks_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build exploratory ML prediction benchmarks from the analysis-ready NLSY layer."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_ml_benchmarks(
        processed_root=runtime_paths["processed_root"] / "nlsy",
        output_dir=runtime_paths["outputs_root"] / "models",
    )
    table = Table(title="ML benchmark artifacts")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("readiness", str(result.readiness_path))
    table.add_row("metrics", str(result.metrics_path))
    table.add_row("feature_importance", str(result.feature_importance_path))
    table.add_row("predictions", str(result.predictions_path))
    table.add_row("summary", str(result.summary_path))
    console.print(table)


@app.command("build-psid-intake")
def build_psid_intake_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build the PSID intake checklist and draft manifest artifacts."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_psid_intake_artifacts(config=data, output_dir=runtime_paths["outputs_root"] / "manifests")
    table = Table(title="PSID intake artifacts")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("markdown", str(result.markdown_path))
    table.add_row("yaml", str(result.yaml_path))
    table.add_row("manifest_csv", str(result.manifest_path))
    console.print(table)


@app.command("build-add-health-intake")
def build_add_health_intake_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build the Add Health public-use intake checklist and draft manifest artifacts."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_add_health_intake_artifacts(config=data, output_dir=runtime_paths["outputs_root"] / "manifests")
    table = Table(title="Add Health intake artifacts")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("markdown", str(result.markdown_path))
    table.add_row("yaml", str(result.yaml_path))
    table.add_row("manifest_csv", str(result.manifest_path))
    console.print(table)


@app.command("build-ffcws-intake")
def build_ffcws_intake_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build the FFCWS public-use intake checklist and draft manifest artifacts."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_ffcws_intake_artifacts(config=data, output_dir=runtime_paths["outputs_root"] / "manifests")
    table = Table(title="FFCWS intake artifacts")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("markdown", str(result.markdown_path))
    table.add_row("yaml", str(result.yaml_path))
    table.add_row("manifest_csv", str(result.manifest_path))
    console.print(table)


@app.command("build-benchmarks")
def build_benchmarks_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
    sources: str = typer.Option("fred,bea,bls,census", "--sources"),
) -> None:
    """Fetch and normalize the first-pass public benchmark sources."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    selected_sources = tuple(item.strip() for item in sources.split(",") if item.strip())
    result = build_public_benchmark_snapshot(
        config=data,
        raw_root=runtime_paths["external_root"] / "public_benchmarks" / "raw",
        processed_root=runtime_paths["processed_root"] / "public_benchmarks",
        output_dir=runtime_paths["outputs_root"] / "manifests",
        sources=selected_sources,
    )
    table = Table(title="Public benchmark artifacts")
    table.add_column("Source")
    table.add_column("Rows", justify="right")
    table.add_column("Observations")
    table.add_column("Metadata")
    for item in result.results:
        table.add_row(item.source, str(item.row_count), str(item.observations_path), str(item.metadata_path))
    console.print(table)
    console.print(f"[green]Manifest: {result.manifest_path}[/green]")


@app.command("build-ipums-workflow")
def build_ipums_workflow_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Build IPUMS extract request artifacts and poll current extract status."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_ipums_extract_workflow(
        config=data,
        raw_root=runtime_paths["external_root"] / "ipums" / "raw",
        processed_root=runtime_paths["processed_root"] / "ipums",
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    table = Table(title="IPUMS extract workflow artifacts")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("request_json", str(result.request_path))
    table.add_row("status_md", str(result.status_path))
    table.add_row("extracts_parquet", str(result.extracts_path))
    table.add_row("metadata_csv", str(result.metadata_path))
    table.add_row("raw_snapshot", str(result.raw_json_path))
    table.add_row("live_extract_rows", str(result.row_count))
    table.add_row("submission_attempted", "yes" if result.submitted else "no")
    console.print(table)


@app.command("build-public-microdata")
def build_public_microdata_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Materialize selected local public microdata subsets for ACS PUMS, SIPP, and CPS ASEC."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_public_microdata_artifacts(
        config=data,
        processed_root=runtime_paths["processed_root"] / "public_microdata",
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    table = Table(title="Public microdata artifacts")
    table.add_column("Source")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    table.add_column("Parquet")
    table.add_column("Metadata")
    for item in result.artifacts:
        table.add_row(
            item.source,
            str(item.row_count),
            str(item.column_count),
            str(item.parquet_path),
            str(item.metadata_path),
        )
    console.print(table)
    console.print(f"[green]Manifest: {result.manifest_path}[/green]")


@app.command("build-public-benchmark-profiles")
def build_public_benchmark_profiles_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Harmonize local ACS PUMS, SIPP, and CPS ASEC public microdata into benchmark-ready profiles."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_public_benchmark_profiles(
        config=data,
        processed_root=runtime_paths["processed_root"],
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    table = Table(title="Public benchmark profile artifacts")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("profiles_parquet", str(result.profiles_path))
    table.add_row("mapping_csv", str(result.mapping_path))
    table.add_row("summary_csv", str(result.summary_path))
    table.add_row("subgroup_summary_csv", str(result.subgroup_summary_path))
    table.add_row("sipp_employment_codebook_csv", str(result.sipp_employment_codebook_path))
    table.add_row("acs_child_context_parquet", str(result.acs_child_context_path))
    table.add_row("acs_child_summary_csv", str(result.acs_child_summary_path))
    table.add_row("acs_child_report_md", str(result.acs_child_report_path))
    table.add_row("row_count", str(result.row_count))
    console.print(table)


@app.command("build-cross-cohort-benchmarks")
def build_cross_cohort_benchmarks_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Compare the NLSY97 adult window to the public benchmark profile layer."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_cross_cohort_benchmark_comparison(
        config=data,
        processed_root=runtime_paths["processed_root"],
        output_dir=runtime_paths["outputs_root"] / "manifests",
    )
    table = Table(title="Cross-cohort benchmark artifacts")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("profiles_parquet", str(result.profiles_path))
    table.add_row("summary_csv", str(result.summary_path))
    table.add_row("subgroup_summary_csv", str(result.subgroup_summary_path))
    table.add_row("report_md", str(result.report_path))
    table.add_row("row_count", str(result.row_count))
    console.print(table)


@app.command("build-results-appendix")
def build_results_appendix_command(
    config: Path | None = typer.Option(Path("config/user_inputs.local.yaml"), "--config"),
) -> None:
    """Materialize stable appendix tables and a frontend/doc handoff memo from existing outputs."""
    data = _load_optional_config(config)
    runtime_paths = resolve_runtime_paths(data)
    result = build_results_appendix(outputs_root=runtime_paths["outputs_root"])
    table = Table(title="Results appendix artifacts")
    table.add_column("Artifact")
    table.add_column("Path")
    table.add_row("manifest_csv", str(result.manifest_path))
    table.add_row("frontend_doc_handoff_md", str(result.handoff_path))
    table.add_row("results_synthesis_md", str(result.synthesis_path))
    table.add_row("nlsy_prevalence_table", str(result.nlsy_prevalence_table_path))
    table.add_row("nlsy_predictor_table", str(result.nlsy_predictor_table_path))
    table.add_row("nlsy_outcome_gap_table", str(result.nlsy_outcome_gap_table_path))
    table.add_row("nlsy_race_gap_table", str(result.nlsy_race_gap_table_path))
    table.add_row("acs_child_context_table", str(result.acs_child_context_table_path))
    table.add_row("benchmark_context_table", str(result.benchmark_context_table_path))
    console.print(table)


def main() -> None:
    app()
