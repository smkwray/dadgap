from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNetCV, LogisticRegression
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency fallback
    SKLEARN_AVAILABLE = False


ML_RANDOM_STATE = 97


@dataclass(frozen=True)
class MLBenchmarkResult:
    readiness_path: Path
    metrics_path: Path
    feature_importance_path: Path
    predictions_path: Path
    summary_path: Path


def _clean_binary_employment(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.mask(numeric < 0)
    numeric = numeric.mask(numeric == 0, 0)
    numeric = numeric.mask(numeric.isin([1, 2]), 1)
    return numeric


def _regression_pipeline(*, numeric_features: list[str], categorical_features: list[str], model: object) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", SimpleImputer(strategy="median"), numeric_features),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical_features,
                        ),
                    ]
                ),
            ),
            ("model", model),
        ]
    )


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def build_ml_benchmarks(*, processed_root: Path, output_dir: Path) -> MLBenchmarkResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    nlsy97 = pd.read_parquet(processed_root / "nlsy97_analysis_ready.parquet").copy()
    nlsy97["employment_2021_clean"] = _clean_binary_employment(nlsy97["employment_2021"])
    nlsy97["employment_2019_clean"] = _clean_binary_employment(nlsy97["employment_2019"])

    readiness_rows = [
        {
            "dataset": "nlsy97",
            "target": "annual_earnings_2021_clean",
            "observed_rows": int(nlsy97["annual_earnings_2021_clean"].notna().sum()),
            "status": "ready" if SKLEARN_AVAILABLE else "sklearn_unavailable",
            "note": "Regression target for predictive benchmark.",
        },
        {
            "dataset": "nlsy97",
            "target": "employment_2021_clean",
            "observed_rows": int(nlsy97["employment_2021_clean"].notna().sum()),
            "status": "ready" if SKLEARN_AVAILABLE else "sklearn_unavailable",
            "note": "Binary employment benchmark derived from 2021 employment status.",
        },
    ]
    readiness = pd.DataFrame(readiness_rows)
    readiness_path = output_dir / "ml_readiness.csv"
    readiness.to_csv(readiness_path, index=False)

    metrics_path = output_dir / "ml_benchmark_metrics.csv"
    feature_importance_path = output_dir / "ml_feature_importance.csv"
    predictions_path = output_dir / "ml_holdout_predictions.parquet"
    summary_path = output_dir / "ml_benchmark_summary.md"

    if not SKLEARN_AVAILABLE:
        pd.DataFrame(columns=["task", "feature_set", "model", "metric", "value"]).to_csv(metrics_path, index=False)
        pd.DataFrame(columns=["task", "feature_set", "model", "feature", "importance"]).to_csv(feature_importance_path, index=False)
        pd.DataFrame(columns=["respondent_id"]).to_parquet(predictions_path, index=False)
        summary_path.write_text(
            "# ML Benchmark Summary\n\n`scikit-learn` is not available in the current environment, so only readiness artifacts were written.\n",
            encoding="utf-8",
        )
        return MLBenchmarkResult(
            readiness_path=readiness_path,
            metrics_path=metrics_path,
            feature_importance_path=feature_importance_path,
            predictions_path=predictions_path,
            summary_path=summary_path,
        )

    base_numeric = [
        "birth_year",
        "parent_education",
        "household_income_2010",
        "annual_earnings_2019",
        "household_income_2019",
        "employment_2019_clean",
    ]
    treatment_numeric = ["primary_treatment_nlsy97"]
    categorical = ["sex_raw", "race_ethnicity_3cat"]

    metrics_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    def _regression_task(feature_set: str, include_treatment: bool) -> None:
        numeric = [*base_numeric, *(treatment_numeric if include_treatment else [])]
        cols = ["respondent_id", *numeric, *categorical, "annual_earnings_2021_clean"]
        frame = nlsy97.loc[nlsy97["annual_earnings_2021_clean"].notna(), cols].copy()
        if include_treatment:
            frame = frame.loc[nlsy97["primary_treatment_observed_nlsy97"] == 1].copy()
        x = frame[numeric + categorical]
        y = frame["annual_earnings_2021_clean"]
        train_idx, test_idx = train_test_split(frame.index, test_size=0.25, random_state=ML_RANDOM_STATE)
        x_train, x_test = x.loc[train_idx], x.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        models = {
            "elastic_net": ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], random_state=ML_RANDOM_STATE, cv=5),
            "random_forest": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=ML_RANDOM_STATE,
                n_jobs=1,
            ),
        }
        for model_name, model in models.items():
            pipe = _regression_pipeline(numeric_features=numeric, categorical_features=categorical, model=model)
            pipe.fit(x_train, y_train)
            pred = pipe.predict(x_test)
            metrics_rows.extend(
                [
                    {
                        "task": "annual_earnings_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "metric": "mae",
                        "value": float(mean_absolute_error(y_test, pred)),
                    },
                    {
                        "task": "annual_earnings_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "metric": "r2",
                        "value": float(r2_score(y_test, pred)),
                    },
                    {
                        "task": "annual_earnings_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "metric": "n_train",
                        "value": int(len(x_train.index)),
                    },
                    {
                        "task": "annual_earnings_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "metric": "n_test",
                        "value": int(len(x_test.index)),
                    },
                ]
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "respondent_id": frame.loc[test_idx, "respondent_id"].to_numpy(),
                        "task": "annual_earnings_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "actual": y_test.to_numpy(),
                        "predicted": pred,
                    }
                )
            )

            prep = pipe.named_steps["prep"]
            feature_names = prep.get_feature_names_out()
            model_obj = pipe.named_steps["model"]
            if model_name == "elastic_net":
                values = np.abs(model_obj.coef_)
            else:
                values = model_obj.feature_importances_
            top_idx = np.argsort(values)[::-1][:20]
            for idx in top_idx:
                importance_rows.append(
                    {
                        "task": "annual_earnings_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "feature": str(feature_names[idx]),
                        "importance": float(values[idx]),
                    }
                )

    def _classification_task(feature_set: str, include_treatment: bool) -> None:
        numeric = [*base_numeric, *(treatment_numeric if include_treatment else [])]
        cols = ["respondent_id", *numeric, *categorical, "employment_2021_clean"]
        frame = nlsy97.loc[nlsy97["employment_2021_clean"].notna(), cols].copy()
        if include_treatment:
            frame = frame.loc[nlsy97["primary_treatment_observed_nlsy97"] == 1].copy()
        x = frame[numeric + categorical]
        y = frame["employment_2021_clean"].astype(int)
        if y.nunique(dropna=True) < 2:
            return
        class_counts = y.value_counts()
        stratify = y if int(class_counts.min()) >= 2 else None
        train_idx, test_idx = train_test_split(
            frame.index,
            test_size=0.25,
            random_state=ML_RANDOM_STATE,
            stratify=stratify,
        )
        x_train, x_test = x.loc[train_idx], x.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        models = {
            "logit": LogisticRegression(max_iter=1000, random_state=ML_RANDOM_STATE),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=ML_RANDOM_STATE,
                n_jobs=1,
            ),
        }
        for model_name, model in models.items():
            pipe = _regression_pipeline(numeric_features=numeric, categorical_features=categorical, model=model)
            pipe.fit(x_train, y_train)
            pred = pipe.predict(x_test)
            score = pipe.predict_proba(x_test)[:, 1]
            metrics_rows.extend(
                [
                    {
                        "task": "employment_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "metric": "accuracy",
                        "value": float(accuracy_score(y_test, pred)),
                    },
                    {
                        "task": "employment_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "metric": "roc_auc",
                        "value": _safe_auc(y_test, score),
                    },
                    {
                        "task": "employment_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "metric": "n_train",
                        "value": int(len(x_train.index)),
                    },
                    {
                        "task": "employment_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "metric": "n_test",
                        "value": int(len(x_test.index)),
                    },
                ]
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "respondent_id": frame.loc[test_idx, "respondent_id"].to_numpy(),
                        "task": "employment_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "actual": y_test.to_numpy(),
                        "predicted": pred,
                        "score": score,
                    }
                )
            )

            prep = pipe.named_steps["prep"]
            feature_names = prep.get_feature_names_out()
            model_obj = pipe.named_steps["model"]
            if model_name == "logit":
                values = np.abs(model_obj.coef_[0])
            else:
                values = model_obj.feature_importances_
            top_idx = np.argsort(values)[::-1][:20]
            for idx in top_idx:
                importance_rows.append(
                    {
                        "task": "employment_2021",
                        "feature_set": feature_set,
                        "model": model_name,
                        "feature": str(feature_names[idx]),
                        "importance": float(values[idx]),
                    }
                )

    _regression_task("base", include_treatment=False)
    _regression_task("base_plus_treatment", include_treatment=True)
    _classification_task("base", include_treatment=False)
    _classification_task("base_plus_treatment", include_treatment=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(metrics_path, index=False)
    feature_importance = pd.DataFrame(importance_rows)
    feature_importance.to_csv(feature_importance_path, index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(predictions_path, index=False)

    def _metric(task: str, feature_set: str, model: str, metric: str) -> float:
        subset = metrics.loc[
            (metrics["task"] == task)
            & (metrics["feature_set"] == feature_set)
            & (metrics["model"] == model)
            & (metrics["metric"] == metric),
            "value",
        ]
        return float(subset.iloc[0]) if not subset.empty else float("nan")

    summary_lines = [
        "# ML Benchmark Summary",
        "",
        "These are exploratory prediction benchmarks only. They are not causal estimates.",
        "",
        "## Annual earnings 2021",
        "",
        f"- Elastic Net R^2 without treatment: {_metric('annual_earnings_2021', 'base', 'elastic_net', 'r2'):.4f}",
        f"- Elastic Net R^2 with treatment: {_metric('annual_earnings_2021', 'base_plus_treatment', 'elastic_net', 'r2'):.4f}",
        f"- Random Forest R^2 without treatment: {_metric('annual_earnings_2021', 'base', 'random_forest', 'r2'):.4f}",
        f"- Random Forest R^2 with treatment: {_metric('annual_earnings_2021', 'base_plus_treatment', 'random_forest', 'r2'):.4f}",
        "",
        "## Employment 2021",
        "",
        f"- Logit ROC AUC without treatment: {_metric('employment_2021', 'base', 'logit', 'roc_auc'):.4f}",
        f"- Logit ROC AUC with treatment: {_metric('employment_2021', 'base_plus_treatment', 'logit', 'roc_auc'):.4f}",
        f"- Random Forest ROC AUC without treatment: {_metric('employment_2021', 'base', 'random_forest', 'roc_auc'):.4f}",
        f"- Random Forest ROC AUC with treatment: {_metric('employment_2021', 'base_plus_treatment', 'random_forest', 'roc_auc'):.4f}",
        "",
        "Feature importances are written separately so treatment and baseline covariates can be compared explicitly.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return MLBenchmarkResult(
        readiness_path=readiness_path,
        metrics_path=metrics_path,
        feature_importance_path=feature_importance_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
    )
