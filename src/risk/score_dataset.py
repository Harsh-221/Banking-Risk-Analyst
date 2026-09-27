"""Apply the saved fraud model and risk layer to a chronological holdout set."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import build_features
from src.risk.scoring import score_transactions

ROOT = Path(__file__).resolve().parents[2]


def score_file(
    data_path: Path = ROOT / "data/raw/transactions.csv",
    model_path: Path = ROOT / "models/fraud_model.joblib",
    output_path: Path = ROOT / "data/processed/test_risk_scores.csv",
    test_fraction: float = 0.20,
) -> pd.DataFrame:
    """Score a chronological holdout, save detail rows, and return the result."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    raw = pd.read_csv(data_path, parse_dates=["timestamp"])
    featured = build_features(raw).sort_values("timestamp").reset_index(drop=True)
    artifact = joblib.load(model_path)
    cut = int(len(featured) * (1 - test_fraction))
    holdout = featured.iloc[cut:].copy()
    probabilities = artifact["pipeline"].predict_proba(holdout[artifact["feature_columns"]])[:, 1]
    scores = score_transactions(probabilities, holdout[artifact["feature_columns"]])
    detail_columns = [
        "transaction_id", "customer_id", "amount", "timestamp", "transaction_type",
        "merchant_category", "location", "device_id", "beneficiary_id", "is_fraud",
        "average_transaction_amount", "amount_deviation", "transactions_last_1h",
        "transactions_last_6h", "transactions_last_24h", "new_beneficiary", "new_device",
        "location_deviation", "time_deviation_hours",
    ]
    report = holdout[detail_columns].join(scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/transactions.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models/fraud_model.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/test_risk_scores.csv")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()
    try:
        report = score_file(args.data, args.model, args.output, args.test_fraction)
    except (FileNotFoundError, KeyError) as exc:
        parser.error(f"{exc}. Run `python -m src.pipeline` first to create the model and dataset.")

    print(f"Scored holdout transactions: {len(report):,}")
    print(f"Holdout synthetic fraud rate: {report['is_fraud'].mean():.2%}")
    print("Risk band summary (synthetic labels):")
    print(
        report.groupby("risk_level", observed=False)
        .agg(transactions=("transaction_id", "count"), fraud_rate=("is_fraud", "mean"),
             average_score=("transaction_risk_score", "mean"))
        .reindex(["HIGH", "MEDIUM", "LOW"])
        .to_string(float_format=lambda value: f"{value:.3f}")
    )
    print(f"Saved scores and reason codes to {args.output}")
    print("Model outputs are prototype signals, not calibrated real-world fraud probabilities.")


if __name__ == "__main__":
    main()
