"""Generate a reproducible, behavior-based synthetic transaction dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def generate_transactions(
    customer_count: int = 5_000,
    transactions_per_customer: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Create histories and occasional anomalous labeled transactions.

    The first three rows per customer are a warm-up history and are always
    normal. Remaining labels are injected with approximately 5% prevalence.
    Fraud examples are intentionally correlated with behavior changes to make
    the dataset useful for learning, not to represent real-world prevalence.
    """
    if customer_count < 1 or transactions_per_customer < 4:
        raise ValueError("Use at least one customer and four transactions per customer.")

    rng = np.random.default_rng(seed)
    cities = np.array(["Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad", "Pune"])
    categories = np.array(["grocery", "fuel", "utilities", "dining", "travel", "electronics", "cash"])
    txn_types = np.array(["card", "upi", "bank_transfer", "atm"])
    rows: list[dict] = []
    start = pd.Timestamp("2025-01-01", tz="UTC")

    for c in range(customer_count):
        customer_id = f"C{c + 1:05d}"
        home_city = str(rng.choice(cities))
        device = f"D{c + 1:05d}-1"
        beneficiary = f"B{c + 1:05d}-1"
        usual_hour = int(rng.integers(8, 21))
        typical_amount = float(np.exp(rng.uniform(np.log(500), np.log(12_000))))
        customer_start = start + pd.to_timedelta(int(rng.integers(0, 60)), unit="D")
        elapsed = 0.0

        for i in range(transactions_per_customer):
            elapsed += float(rng.uniform(2.0, 36.0))
            fraud = i >= 3 and bool(rng.random() < 0.05)
            anomalous = fraud or (i >= 3 and bool(rng.random() < 0.035))

            if anomalous:
                amount = float(np.clip(typical_amount * rng.uniform(8, 35), 300, 500_000))
                hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
                location = str(rng.choice(cities[cities != home_city]))
                txn_device = f"D{c + 1:05d}-X{i}"
                txn_beneficiary = f"B{c + 1:05d}-X{i}"
            else:
                amount = float(np.clip(rng.lognormal(np.log(typical_amount), 0.5), 100, 50_000))
                hour = int(np.clip(rng.normal(usual_hour, 2.5), 6, 23))
                location = home_city if rng.random() < 0.94 else str(rng.choice(cities))
                txn_device = device if rng.random() < 0.97 else f"D{c + 1:05d}-2"
                txn_beneficiary = beneficiary if rng.random() < 0.90 else f"B{c + 1:05d}-2"

            timestamp = customer_start + pd.to_timedelta(elapsed, unit="h")
            timestamp = timestamp.normalize() + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(int(rng.integers(0, 60)), unit="m")
            rows.append(
                {
                    "transaction_id": f"TX{c * transactions_per_customer + i + 1:07d}",
                    "customer_id": customer_id,
                    "amount": round(amount, 2),
                    "transaction_type": str(rng.choice(txn_types, p=[0.42, 0.35, 0.18, 0.05])),
                    "merchant_category": str(rng.choice(categories)),
                    "timestamp": timestamp,
                    "location": location,
                    "device_id": txn_device,
                    "beneficiary_id": txn_beneficiary,
                    "is_fraud": int(fraud),
                }
            )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["timestamp", "transaction_id"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=5_000)
    parser.add_argument("--transactions-per-customer", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/transactions.csv")
    args = parser.parse_args()

    frame = generate_transactions(args.customers, args.transactions_per_customer, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} rows to {args.output}")
    print(f"Fraud rate: {frame['is_fraud'].mean():.2%}")


if __name__ == "__main__":
    main()
