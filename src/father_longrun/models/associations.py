from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "summarize_prevalence",
    "summarize_mean_gap",
    "fit_adjusted_ols",
    "fit_adjusted_glm",
    "fit_subgroup_interactions",
]


@dataclass(frozen=True)
class _FittedModel:
    params: pd.Series
    cov_params: pd.DataFrame
    n_obs: int
    model: str
    family: str


def _as_tuple(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, tuple):
        return values
    return tuple(values)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _group_columns(group_by: str | Sequence[str] | None) -> list[str]:
    if group_by is None:
        return []
    if isinstance(group_by, str):
        return [group_by]
    return list(group_by)


def _label_group_value(value: object) -> str:
    if isinstance(value, tuple):
        return " | ".join("missing" if pd.isna(item) else str(item) for item in value)
    return "missing" if pd.isna(value) else str(value)


def _weight_series(frame: pd.DataFrame, weight_col: str | None) -> pd.Series | None:
    if weight_col is None:
        return None
    weights = pd.to_numeric(frame[weight_col], errors="coerce")
    return weights.where(weights > 0)


def _weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if weights is None:
        valid = numeric.dropna()
        return float(valid.mean()) if not valid.empty else float("nan")

    aligned = pd.concat([numeric.rename("value"), weights.rename("weight")], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    total_weight = float(aligned["weight"].sum())
    if total_weight <= 0:
        return float("nan")
    return float(np.average(aligned["value"].to_numpy(dtype=float), weights=aligned["weight"].to_numpy(dtype=float)))


def _filtered_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    weight_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    required = list(dict.fromkeys([*columns, *( [weight_col] if weight_col is not None else [] )]))
    data = frame.loc[:, required].copy()
    if weight_col is not None:
        data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")
        data = data.loc[data[weight_col] > 0].copy()
    return data, _weight_series(data, weight_col)


def summarize_prevalence(
    frame: pd.DataFrame,
    *,
    treatment: str,
    group_by: str | Sequence[str] | None = None,
    weight_col: str | None = None,
    overall_label: str = "overall",
) -> pd.DataFrame:
    """Summarize a binary treatment or exposure by subgroup."""

    group_cols = _group_columns(group_by)
    data, weights = _filtered_frame(frame, columns=[treatment, *group_cols], weight_col=weight_col)
    data[treatment] = _safe_numeric(data[treatment])
    data = data.dropna(subset=[treatment]).copy()
    if weights is not None:
        weights = weights.loc[data.index]

    rows: list[dict[str, object]] = []
    overall_rate = _weighted_mean(data[treatment], weights)
    rows.append(
        {
            "group_type": overall_label,
            "group_value": overall_label,
            "n": int(data.shape[0]),
            "weight_total": float(weights.sum()) if weights is not None else float("nan"),
            "binary_mean": overall_rate,
            "binary_complement": float("nan") if pd.isna(overall_rate) else 1.0 - overall_rate,
        }
    )

    if group_cols:
        for group_value, group in data.groupby(group_cols, dropna=False):
            group_weights = weights.loc[group.index] if weights is not None else None
            rate = _weighted_mean(group[treatment], group_weights)
            rows.append(
                {
                    "group_type": " | ".join(group_cols),
                    "group_value": _label_group_value(group_value),
                    "n": int(group.shape[0]),
                    "weight_total": float(group_weights.sum()) if group_weights is not None else float("nan"),
                    "binary_mean": rate,
                    "binary_complement": float("nan") if pd.isna(rate) else 1.0 - rate,
                }
            )

    return pd.DataFrame(rows)


def summarize_mean_gap(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    group_by: str | Sequence[str] | None = None,
    weight_col: str | None = None,
    overall_label: str = "overall",
) -> pd.DataFrame:
    """Summarize treated and control means and the between-group gap."""

    group_cols = _group_columns(group_by)
    data, weights = _filtered_frame(frame, columns=[outcome, treatment, *group_cols], weight_col=weight_col)
    data[outcome] = _safe_numeric(data[outcome])
    data[treatment] = _safe_numeric(data[treatment])
    data = data.dropna(subset=[outcome, treatment]).copy()
    if weights is not None:
        weights = weights.loc[data.index]

    def _summary(subframe: pd.DataFrame, label_type: str, label_value: str) -> dict[str, object]:
        sub_weights = weights.loc[subframe.index] if weights is not None else None
        treated = subframe.loc[subframe[treatment] == 1, outcome]
        control = subframe.loc[subframe[treatment] == 0, outcome]
        treated_weights = sub_weights.loc[treated.index] if sub_weights is not None else None
        control_weights = sub_weights.loc[control.index] if sub_weights is not None else None
        treated_mean = _weighted_mean(treated, treated_weights)
        control_mean = _weighted_mean(control, control_weights)
        return {
            "group_type": label_type,
            "group_value": label_value,
            "n": int(subframe.shape[0]),
            "weight_total": float(sub_weights.sum()) if sub_weights is not None else float("nan"),
            "treated_n": int(treated.shape[0]),
            "control_n": int(control.shape[0]),
            "treated_mean": treated_mean,
            "control_mean": control_mean,
            "mean_gap": float("nan") if pd.isna(treated_mean) or pd.isna(control_mean) else treated_mean - control_mean,
        }

    rows = [_summary(data, overall_label, overall_label)]
    if group_cols:
        for group_value, group in data.groupby(group_cols, dropna=False):
            rows.append(_summary(group, " | ".join(group_cols), _label_group_value(group_value)))
    return pd.DataFrame(rows)


def _build_main_effect_design(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: Sequence[str] = (),
    categorical_covariates: Sequence[str] = (),
    weight_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    required = [outcome, treatment, *covariates, *categorical_covariates]
    if weight_col is not None:
        required.append(weight_col)
    data, weights = _filtered_frame(frame, columns=required, weight_col=weight_col)
    data[outcome] = _safe_numeric(data[outcome])
    data[treatment] = _safe_numeric(data[treatment])
    for column in covariates:
        data[column] = _safe_numeric(data[column])
    data = data.dropna(subset=[outcome, treatment, *covariates, *categorical_covariates]).copy()
    if weights is not None:
        weights = weights.loc[data.index]

    design = pd.DataFrame(index=data.index)
    design["intercept"] = 1.0
    design[treatment] = data[treatment].astype(float)
    for column in covariates:
        design[column] = data[column].astype(float)
    for column in categorical_covariates:
        dummies = pd.get_dummies(data[column].astype("string"), prefix=column, drop_first=True, dtype=float)
        if not dummies.empty:
            design = pd.concat([design, dummies], axis=1)
    design = design.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[design.index]
    weights = weights.loc[design.index] if weights is not None else None
    return design, data[outcome].astype(float), weights


def _fit_gaussian(design: pd.DataFrame, target: pd.Series, weights: pd.Series | None) -> _FittedModel:
    x = design.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    w = np.ones_like(y, dtype=float) if weights is None else weights.to_numpy(dtype=float)
    sw = np.sqrt(w)
    xw = x * sw[:, None]
    yw = y * sw
    beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    resid = y - x @ beta
    resid_w = resid * sw
    xtwx = xw.T @ xw
    xtwx_inv = np.linalg.pinv(xtwx)
    n_obs, p = x.shape
    if n_obs > p:
        meat = xw.T @ ((resid_w[:, None] ** 2) * xw)
        cov = (n_obs / (n_obs - p)) * xtwx_inv @ meat @ xtwx_inv
    else:
        cov = np.full((p, p), np.nan)
    return _FittedModel(
        params=pd.Series(beta, index=design.columns, dtype="float64"),
        cov_params=pd.DataFrame(cov, index=design.columns, columns=design.columns),
        n_obs=int(n_obs),
        model="ols_hc1",
        family="gaussian",
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_binomial(design: pd.DataFrame, target: pd.Series, weights: pd.Series | None) -> _FittedModel:
    x = design.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    prior = np.ones_like(y, dtype=float) if weights is None else weights.to_numpy(dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)

    for _ in range(100):
        eta = x @ beta
        mu = _sigmoid(eta)
        variance = np.clip(mu * (1.0 - mu), 1e-8, None)
        working_weights = prior * variance
        z = eta + (y - mu) / variance
        sw = np.sqrt(working_weights)
        xw = x * sw[:, None]
        zw = z * sw
        beta_new, *_ = np.linalg.lstsq(xw, zw, rcond=None)
        if float(np.max(np.abs(beta_new - beta))) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    eta = x @ beta
    mu = _sigmoid(eta)
    variance = np.clip(mu * (1.0 - mu), 1e-8, None)
    working_weights = prior * variance
    xw = x * np.sqrt(working_weights)[:, None]
    xtwx_inv = np.linalg.pinv(xw.T @ xw)
    return _FittedModel(
        params=pd.Series(beta, index=design.columns, dtype="float64"),
        cov_params=pd.DataFrame(xtwx_inv, index=design.columns, columns=design.columns),
        n_obs=int(x.shape[0]),
        model="glm_binomial_hc1",
        family="binomial",
    )


def _fit_model(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: Sequence[str] = (),
    categorical_covariates: Sequence[str] = (),
    weight_col: str | None = None,
    family: str = "gaussian",
) -> _FittedModel:
    design, target, weights = _build_main_effect_design(
        frame,
        outcome=outcome,
        treatment=treatment,
        covariates=covariates,
        categorical_covariates=categorical_covariates,
        weight_col=weight_col,
    )
    if design.empty:
        return _FittedModel(
            params=pd.Series(dtype="float64"),
            cov_params=pd.DataFrame(dtype="float64"),
            n_obs=0,
            model="empty",
            family=family.lower(),
        )
    family_name = family.lower()
    if family_name in {"gaussian", "normal"}:
        return _fit_gaussian(design, target, weights)
    if family_name in {"binomial", "logit"}:
        return _fit_binomial(design, target, weights)
    raise ValueError(f"Unsupported family: {family}")


def _coefficient_table(fitted: _FittedModel) -> pd.DataFrame:
    if fitted.params.empty:
        return pd.DataFrame(
            columns=["term", "coefficient", "std_error", "p_value", "odds_ratio", "model", "family", "n"]
        )
    se = np.sqrt(np.clip(np.diag(fitted.cov_params.to_numpy(dtype=float)), 0.0, None))
    coef = fitted.params.to_numpy(dtype=float)
    z = np.divide(coef, se, out=np.full_like(coef, np.nan), where=se > 0)
    p = np.array([math.erfc(abs(value) / math.sqrt(2.0)) if not math.isnan(value) else np.nan for value in z])
    odds_ratio = np.exp(coef) if fitted.family == "binomial" else np.full_like(coef, np.nan)
    return pd.DataFrame(
        {
            "term": fitted.params.index,
            "coefficient": coef,
            "std_error": se,
            "p_value": p,
            "odds_ratio": odds_ratio,
            "model": fitted.model,
            "family": fitted.family,
            "n": fitted.n_obs,
        }
    )


def fit_adjusted_ols(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: Sequence[str] | None = None,
    categorical_covariates: Sequence[str] | None = None,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Fit a weighted or unweighted HC1 OLS model and return tidy coefficients."""

    fitted = _fit_model(
        frame,
        outcome=outcome,
        treatment=treatment,
        covariates=_as_tuple(covariates),
        categorical_covariates=_as_tuple(categorical_covariates),
        weight_col=weight_col,
        family="gaussian",
    )
    return _coefficient_table(fitted)


def fit_adjusted_glm(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    family: str = "binomial",
    covariates: Sequence[str] | None = None,
    categorical_covariates: Sequence[str] | None = None,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Fit a weighted or unweighted GLM and return tidy coefficients."""

    fitted = _fit_model(
        frame,
        outcome=outcome,
        treatment=treatment,
        covariates=_as_tuple(covariates),
        categorical_covariates=_as_tuple(categorical_covariates),
        weight_col=weight_col,
        family=family,
    )
    return _coefficient_table(fitted)


def _contrast_row(
    fitted: _FittedModel,
    *,
    outcome: str,
    treatment: str,
    subgroup: str,
    subgroup_level: str,
    reference_level: str,
    design_columns: Sequence[str],
    contrast: dict[str, float],
) -> dict[str, object]:
    vector = np.array([contrast.get(column, 0.0) for column in design_columns], dtype=float)
    params = fitted.params.to_numpy(dtype=float)
    cov = fitted.cov_params.to_numpy(dtype=float)
    estimate = float(vector @ params)
    variance = float(vector @ cov @ vector)
    std_error = float(math.sqrt(max(variance, 0.0)))
    z_score = float(estimate / std_error) if std_error > 0 else float("nan")
    p_value = float(math.erfc(abs(z_score) / math.sqrt(2.0))) if not math.isnan(z_score) else float("nan")
    odds_ratio = float(np.exp(estimate)) if fitted.family == "binomial" else float("nan")
    return {
        "outcome": outcome,
        "treatment": treatment,
        "subgroup": subgroup,
        "subgroup_level": subgroup_level,
        "reference_level": reference_level,
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "model": fitted.model,
        "family": fitted.family,
        "n": fitted.n_obs,
    }


def fit_subgroup_interactions(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    subgroup: str,
    covariates: Sequence[str] | None = None,
    categorical_covariates: Sequence[str] | None = None,
    weight_col: str | None = None,
    family: str = "gaussian",
    reference_level: str | None = None,
) -> pd.DataFrame:
    """Fit a pooled interaction model and return subgroup-specific treatment contrasts."""

    covariate_list = _as_tuple(covariates)
    categorical_list = _as_tuple(categorical_covariates)
    required = [outcome, treatment, subgroup, *covariate_list, *categorical_list]
    if weight_col is not None:
        required.append(weight_col)
    data, weights = _filtered_frame(frame, columns=required, weight_col=weight_col)
    data[outcome] = _safe_numeric(data[outcome])
    data[treatment] = _safe_numeric(data[treatment])
    data[subgroup] = data[subgroup].astype("string")
    for column in covariate_list:
        data[column] = _safe_numeric(data[column])
    data = data.dropna(subset=[outcome, treatment, subgroup, *covariate_list, *categorical_list]).copy()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "outcome",
                "treatment",
                "subgroup",
                "subgroup_level",
                "reference_level",
                "estimate",
                "std_error",
                "p_value",
                "odds_ratio",
                "model",
                "family",
                "n",
            ]
        )
    if weights is not None:
        weights = weights.loc[data.index]

    observed_levels = list(pd.unique(data[subgroup]))
    reference = observed_levels[0] if reference_level is None else reference_level
    if reference not in observed_levels:
        raise ValueError(f"reference_level {reference!r} is not present in subgroup {subgroup!r}")
    ordered_levels = [reference, *[level for level in observed_levels if level != reference]]

    design = pd.DataFrame(index=data.index)
    design["intercept"] = 1.0
    design[treatment] = data[treatment].astype(float)
    subgroup_dummies = pd.get_dummies(data[subgroup], prefix=subgroup, drop_first=False, dtype=float)
    for level in ordered_levels[1:]:
        column = f"{subgroup}_{level}"
        if column not in subgroup_dummies.columns:
            subgroup_dummies[column] = 0.0
        design[column] = subgroup_dummies[column].astype(float)
        design[f"{treatment}:{column}"] = design[treatment] * design[column]
    for column in covariate_list:
        design[column] = data[column].astype(float)
    for column in categorical_list:
        dummies = pd.get_dummies(data[column].astype("string"), prefix=column, drop_first=True, dtype=float)
        if not dummies.empty:
            design = pd.concat([design, dummies], axis=1)
    design = design.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[design.index]
    weights = weights.loc[design.index] if weights is not None else None

    family_name = family.lower()
    if family_name in {"gaussian", "normal"}:
        fitted = _fit_gaussian(design, data[outcome].astype(float), weights)
    elif family_name in {"binomial", "logit"}:
        fitted = _fit_binomial(design, data[outcome].astype(float), weights)
    else:
        raise ValueError(f"Unsupported family: {family}")

    rows: list[dict[str, object]] = []
    rows.append(
        _contrast_row(
            fitted,
            outcome=outcome,
            treatment=treatment,
            subgroup=subgroup,
            subgroup_level=reference,
            reference_level=reference,
            design_columns=design.columns,
            contrast={treatment: 1.0},
        )
    )
    for level in ordered_levels[1:]:
        rows.append(
            _contrast_row(
                fitted,
                outcome=outcome,
                treatment=treatment,
                subgroup=subgroup,
                subgroup_level=level,
                reference_level=reference,
                design_columns=design.columns,
                contrast={treatment: 1.0, f"{treatment}:{subgroup}_{level}": 1.0},
            )
        )
    return pd.DataFrame(rows)
