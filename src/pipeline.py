"""Prepare the synthetic data, model, risk scores, and application database."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from src.data.generate_synthetic import generate_transactions  # noqa: E402
from src.models.train_model import evaluate_saved_model, train_from_csv  # noqa: E402
from src.risk.score_dataset import score_file  # noqa: E402
from src.storage.database import init_db, upsert_scored_transactions  # noqa: E402


def run(force_train: bool = False) -> None:
    raw_path = ROOT / "data/raw/transactions.csv"
    model_path = ROOT / "models/fraud_model.joblib"
    metrics_path = ROOT / "models/metrics.json"
    scores_path = ROOT / "data/processed/test_risk_scores.csv"

    if not raw_path.exists():
        print("Generating synthetic transactions...")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        generate_transactions().to_csv(raw_path, index=False)
    if force_train or not model_path.exists():
        print("Training and selecting the model by holdout PR-AUC...")
        metrics = train_from_csv(raw_path, model_path, metrics_path)
        print(f"Selected model: {metrics['selected_model']}")
    else:
        print(f"Using saved model: {model_path}")
        if not metrics_path.exists():
            print("Creating holdout metrics for the saved model...")
            metrics = evaluate_saved_model(raw_path, model_path, metrics_path)
            print(f"Saved model holdout PR-AUC: {metrics['models'][metrics['selected_model']]['pr_auc']:.4f}")

    print("Scoring chronological holdout transactions...")
    scored = score_file(raw_path, model_path, scores_path)
    print("Initializing database and loading scored transactions...")
    init_db()
    loaded = upsert_scored_transactions(scored)
    print(f"Loaded or refreshed {loaded:,} scored transactions.")
    print(f"Dashboard data: {scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train", action="store_true", help="retrain and replace the saved model")
    args = parser.parse_args()
    run(force_train=args.force_train)


if __name__ == "__main__":
    main()
