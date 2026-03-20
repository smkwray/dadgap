from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from father_longrun.models.associations import (
    fit_adjusted_glm,
    fit_adjusted_ols,
    fit_subgroup_interactions,
    summarize_mean_gap,
    summarize_prevalence,
)


def test_summarize_prevalence_and_mean_gap_with_weights() -> None:
    frame = pd.DataFrame(
        {
            "treated": [1, 0, 1, 1, 0],
            "outcome": [10.0, 8.0, 12.0, 15.0, 7.0],
            "group": ["a", "a", "a", "b", "b"],
            "weight": [1.0, 1.0, 2.0, 1.0, 3.0],
        }
    )

    prevalence = summarize_prevalence(frame, treatment="treated", group_by="group", weight_col="weight")
    overall = prevalence.loc[prevalence["group_type"] == "overall"].iloc[0]
    group_a = prevalence.loc[prevalence["group_value"] == "a"].iloc[0]
    group_b = prevalence.loc[prevalence["group_value"] == "b"].iloc[0]
    assert overall["binary_mean"] == pytest.approx(0.5)
    assert overall["binary_complement"] == pytest.approx(0.5)
    assert group_a["binary_mean"] == pytest.approx(0.75)
    assert group_b["binary_mean"] == pytest.approx(0.25)

    mean_gaps = summarize_mean_gap(
        frame,
        outcome="outcome",
        treatment="treated",
        group_by="group",
        weight_col="weight",
    )
    overall_gap = mean_gaps.loc[mean_gaps["group_type"] == "overall"].iloc[0]
    group_a_gap = mean_gaps.loc[mean_gaps["group_value"] == "a"].iloc[0]
    assert overall_gap["treated_mean"] == pytest.approx(12.25)
    assert overall_gap["control_mean"] == pytest.approx(7.25)
    assert overall_gap["mean_gap"] == pytest.approx(5.0)
    assert group_a_gap["treated_mean"] == pytest.approx(11.3333333333)
    assert group_a_gap["control_mean"] == pytest.approx(8.0)


def test_fit_adjusted_ols_returns_treatment_effect() -> None:
    rng = np.random.default_rng(0)
    n = 250
    exposed = rng.binomial(1, 0.5, size=n)
    x = rng.normal(size=n)
    outcome = 1.5 * exposed + 0.25 * x + rng.normal(scale=0.05, size=n)
    frame = pd.DataFrame({"outcome": outcome, "exposed": exposed, "x": x})

    table = fit_adjusted_ols(frame, outcome="outcome", treatment="exposed", covariates=["x"])
    row = table.loc[table["term"] == "exposed"].iloc[0]

    assert row["model"] == "ols_hc1"
    assert row["family"] == "gaussian"
    assert row["coefficient"] == pytest.approx(1.5, abs=0.08)
    assert row["std_error"] > 0
    assert 0 <= row["p_value"] <= 1


def test_fit_adjusted_glm_binomial_returns_odds_ratio() -> None:
    rng = np.random.default_rng(1)
    n = 300
    exposed = rng.binomial(1, 0.5, size=n)
    x = rng.normal(size=n)
    linear_predictor = -0.4 + 1.1 * exposed + 0.3 * x
    probability = 1.0 / (1.0 + np.exp(-linear_predictor))
    outcome = rng.binomial(1, probability, size=n)
    frame = pd.DataFrame({"outcome": outcome, "exposed": exposed, "x": x})

    table = fit_adjusted_glm(frame, outcome="outcome", treatment="exposed", family="binomial", covariates=["x"])
    row = table.loc[table["term"] == "exposed"].iloc[0]

    assert row["model"] == "glm_binomial_hc1"
    assert row["family"] == "binomial"
    assert row["coefficient"] > 0
    assert row["odds_ratio"] > 1
    assert row["std_error"] > 0


def test_fit_subgroup_interactions_returns_level_specific_effects() -> None:
    rng = np.random.default_rng(2)
    rows: list[dict[str, float | int | str]] = []
    effects = {"a": 1.0, "b": 2.0}
    for subgroup in ("a", "b"):
        for exposed in (0, 1):
            for _ in range(120):
                x = rng.normal()
                outcome = 3.0 + effects[subgroup] * exposed + 0.2 * x + rng.normal(scale=0.05)
                rows.append(
                    {
                        "outcome": outcome,
                        "exposed": exposed,
                        "subgroup": subgroup,
                        "x": x,
                    }
                )
    frame = pd.DataFrame(rows)

    table = fit_subgroup_interactions(
        frame,
        outcome="outcome",
        treatment="exposed",
        subgroup="subgroup",
        covariates=["x"],
        family="gaussian",
        reference_level="a",
    )

    assert set(table["subgroup_level"]) == {"a", "b"}
    effect_a = table.loc[table["subgroup_level"] == "a", "estimate"].iloc[0]
    effect_b = table.loc[table["subgroup_level"] == "b", "estimate"].iloc[0]
    assert effect_a == pytest.approx(1.0, abs=0.08)
    assert effect_b == pytest.approx(2.0, abs=0.08)
    assert set(table["family"]) == {"gaussian"}
