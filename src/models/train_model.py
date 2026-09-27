"""Train Logistic Regression and Random Forest baselines on a time split."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import build_features

ROOT = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = [
    "amount", "amount_deviation", "transactions_last_1h", "transactions_last_6h",
    "transactions_last_24h", "new_beneficiary", "new_device", "location_deviation",
    "time_deviation_hours", "transaction_type", "merchant_category",
]


def train_from_csv(
    data_path: Path = ROOT / "data/raw/transactions.csv",
    model_path: Path = ROOT / "models/fraud_model.joblib",
    metrics_path: Path = ROOT / "models/metrics.json",
    test_fraction: float = 0.20,
) -> dict:
    """Fit baselines, select by PR-AUC on a chronological holdout, save artifact."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    data = pd.read_csv(data_path, parse_dates=["timestamp"])
    featured = build_features(data).sort_values("timestamp").reset_index(drop=True)
    split = int(len(featured) * (1 - test_fraction))
    train, test = featured.iloc[:split], featured.iloc[split:]
    if train["is_fraud"].nunique() < 2 or test["is_fraud"].nunique() < 2:
        raise ValueError("Both chronological train and holdout sets must contain both classes")

    numeric = [column for column in FEATURE_COLUMNS if column not in {"transaction_type", "merchant_category"}]
    categorical = ["transaction_type", "merchant_category"]
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        ),
    }

    outcomes = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipeline.fit(train[FEATURE_COLUMNS], train["is_fraud"])
        probabilities = pipeline.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        outcomes[name] = {
            "pipeline": pipeline,
            "pr_auc": float(average_precision_score(test["is_fraud"], probabilities)),
            "roc_auc": float(roc_auc_score(test["is_fraud"], probabilities)),
            "report": classification_report(test["is_fraud"], probabilities >= 0.5, output_dict=True, zero_division=0),
        }

    best_name = max(outcomes, key=lambda name: outcomes[name]["pr_auc"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": outcomes[best_name]["pipeline"],
        "feature_columns": FEATURE_COLUMNS,
        "model_name": best_name,
        "threshold": 0.5,
        "test_fraction": test_fraction,
    }
    joblib.dump(artifact, model_path)

    metrics = {
        "selected_model": best_name,
        "train_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "train_fraud_rate": float(train["is_fraud"].mean()),
        "holdout_fraud_rate": float(test["is_fraud"].mean()),
        "models": {
            name: {"pr_auc": result["pr_auc"], "roc_auc": result["roc_auc"], "classification_report": result["report"]}
            for name, result in outcomes.items()
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def evaluate_saved_model(
    data_path: Path = ROOT / "data/raw/transactions.csv",
    model_path: Path = ROOT / "models/fraud_model.joblib",
    metrics_path: Path = ROOT / "models/metrics.json",
) -> dict:
    """Create holdout metrics for an existing notebook-trained model artifact."""
    artifact = joblib.load(model_path)
    data = pd.read_csv(data_path, parse_dates=["timestamp"])
    featured = build_features(data).sort_values("timestamp").reset_index(drop=True)
    fraction = float(artifact.get("test_fraction", 0.20))
    split = int(len(featured) * (1 - fraction))
    holdout = featured.iloc[split:]
    probabilities = artifact["pipeline"].predict_proba(holdout[artifact["feature_columns"]])[:, 1]
    metrics = {
        "selected_model": artifact.get("model_name", "Saved model"),
        "train_rows": int(split),
        "holdout_rows": int(len(holdout)),
        "train_fraud_rate": float(featured.iloc[:split]["is_fraud"].mean()),
        "holdout_fraud_rate": float(holdout["is_fraud"].mean()),
        "models": {
            artifact.get("model_name", "Saved model"): {
                "pr_auc": float(average_precision_score(holdout["is_fraud"], probabilities)),
                "roc_auc": float(roc_auc_score(holdout["is_fraud"], probabilities)),
                "classification_report": classification_report(
                    holdout["is_fraud"], probabilities >= artifact.get("threshold", 0.5),
                    output_dict=True, zero_division=0,
                ),
            }
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
