"""Training, evaluation and batch prediction entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

from .config import load_config, resolve_project_path
from .data import load_training_frame, read_csv
from .features import model_matrix, prepare_features


def _one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _column_types(X):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    return numeric, categorical


def _preprocessor(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric, categorical = _column_types(X)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def _classifier(config: dict):
    from sklearn.linear_model import LogisticRegression

    modeling = config.get("modeling", {})
    class_weight = modeling.get("class_weight", "balanced")
    return LogisticRegression(
        max_iter=int(modeling.get("max_iter", 1000)),
        class_weight=class_weight,
        n_jobs=-1,
        random_state=int(modeling.get("random_state", 42)),
    )


def _pipeline(X, estimator):
    from sklearn.pipeline import Pipeline

    return Pipeline(steps=[("preprocess", _preprocessor(X)), ("model", estimator)])


def _classification_metrics(y_true, y_pred, y_proba, config: dict):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = sorted(set(y_true.dropna().tolist()))
    average = "binary" if len(labels) == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_proba is not None and len(labels) == 2:
        positive = config.get("data", {}).get("positive_label", labels[-1])
        classes = list(getattr(y_proba, "classes_", []))
        scores = y_proba
        if hasattr(y_proba, "shape") and len(y_proba.shape) == 2:
            pos_idx = 1
            scores = y_proba[:, pos_idx]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            metrics["average_precision"] = float(average_precision_score(y_true, scores))
        except ValueError:
            pass
        for k in config.get("evaluation", {}).get("top_k", []):
            k = min(int(k), len(scores))
            if k <= 0:
                continue
            order = np.argsort(scores)[::-1][:k]
            positive_mask = (y_true.reset_index(drop=True).iloc[order] == positive).astype(int)
            base_rate = float((y_true == positive).mean())
            precision_at_k = float(positive_mask.mean())
            metrics[f"precision_at_{k}"] = precision_at_k
            metrics[f"lift_at_{k}"] = precision_at_k / base_rate if base_rate else None
    return metrics


def _try_log_mlflow(config: dict, model, metrics: dict, model_path) -> str | None:
    if not config.get("modeling", {}).get("use_mlflow", False):
        return None
    try:
        import mlflow
    except ImportError:
        return "MLflow is not installed. Run `pip install mlflow` to enable experiment tracking."

    experiment = config.get("modeling", {}).get("mlflow_experiment", config.get("project", {}).get("slug", "kaggle_project"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=config.get("project", {}).get("slug", "training")):
        mlflow.log_params(
            {
                "problem_type": config.get("project", {}).get("problem_type"),
                "family": config.get("project", {}).get("family"),
                "target": config.get("data", {}).get("target", ""),
            }
        )
        mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
        mlflow.log_artifact(str(model_path))
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception:
            pass
    return None


def train(config_path: str | Path | None = None):
    """Train the fraud classifier, evaluate it on a held-out split and persist artifacts."""
    import joblib
    from sklearn.model_selection import train_test_split

    config = load_config(config_path)
    df = load_training_frame(config)
    X, y, _ = model_matrix(df, config, training=True)
    if y is None:
        raise ValueError("Target column is not available in the training data.")

    modeling = config.get("modeling", {})
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=float(modeling.get("test_size", 0.2)),
        random_state=int(modeling.get("random_state", 42)),
        stratify=y,
    )

    model = _pipeline(X_train, _classifier(config))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)
    metrics = _classification_metrics(y_valid, y_pred, y_proba, config)

    model_path = resolve_project_path(config, modeling.get("model_file", "models/model.joblib"))
    metrics_path = resolve_project_path(config, "reports/metrics.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    mlflow_note = _try_log_mlflow(config, model, metrics, model_path)
    if mlflow_note:
        metrics["mlflow_note"] = mlflow_note
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "metrics_path": str(metrics_path), "metrics": metrics}


def predict(config_path: str | Path | None, input_path: str | Path, output_path: str | Path | None = None):
    import joblib
    import pandas as pd

    config = load_config(config_path)
    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    model = joblib.load(model_path)
    raw = read_csv(input_path)
    prepared = prepare_features(raw, config, training=False)
    X, _, _ = model_matrix(prepared, config, training=False)
    result = raw.copy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            result["score"] = proba[:, 1]
        else:
            for idx, klass in enumerate(model.classes_):
                result[f"score_{klass}"] = proba[:, idx]
    result["prediction"] = model.predict(X)
    output = Path(output_path) if output_path else resolve_project_path(config, "data/processed/predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
