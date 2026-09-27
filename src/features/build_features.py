"""Build transaction features using only each customer's prior activity."""

from __future__ import annotations

import pandas as pd


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with behavior features; labels and future rows are never read.

    Expected raw columns: customer_id, timestamp, amount, location, device_id,
    beneficiary_id. The returned frame retains its input columns and row order.
    New-entity flags are zero for the first observed item, since no prior
    history exists. Amount baselines likewise use prior transactions only.
    """
    required = {"customer_id", "timestamp", "amount", "location", "device_id", "beneficiary_id"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    result = transactions.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["amount"] = pd.to_numeric(result["amount"], errors="raise")
    result["original_order_marker"] = range(len(result))
    ordered = result.sort_values(["customer_id", "timestamp", "original_order_marker"])
    feature_rows: list[dict] = []

    for customer_id, group in ordered.groupby("customer_id", sort=False):
        prior: list[tuple[pd.Timestamp, float, str, str, str]] = []
        for row in group.itertuples(index=False):
            ts = row.timestamp
            # tuples: timestamp, amount, location, device, beneficiary
            recent_1h = sum(t >= ts - pd.Timedelta(hours=1) for t, *_ in prior)
            recent_6h = sum(t >= ts - pd.Timedelta(hours=6) for t, *_ in prior)
            recent_24h = sum(t >= ts - pd.Timedelta(hours=24) for t, *_ in prior)
            amounts = [a for _, a, *_ in prior]
            seen_locations = {loc for _, _, loc, _, _ in prior}
            seen_devices = {dev for _, _, _, dev, _ in prior}
            seen_beneficiaries = {ben for _, _, _, _, ben in prior}

            baseline = sum(amounts) / len(amounts) if amounts else float(row.amount)
            previous_hours = [t.hour for t, *_ in prior]
            # Circular clock distance avoids treating 23:00 and 00:00 as far apart.
            if previous_hours:
                time_deviation = min(min(abs(row.timestamp.hour - h), 24 - abs(row.timestamp.hour - h)) for h in previous_hours)
            else:
                time_deviation = 0

            feature_rows.append(
                {
                    "original_order_marker": row.original_order_marker,
                    "average_transaction_amount": baseline,
                    "amount_deviation": float(row.amount) / max(baseline, 1.0),
                    "transactions_last_1h": recent_1h,
                    "transactions_last_6h": recent_6h,
                    "transactions_last_24h": recent_24h,
                    "new_beneficiary": int(row.beneficiary_id not in seen_beneficiaries and bool(prior)),
                    "new_device": int(row.device_id not in seen_devices and bool(prior)),
                    "location_deviation": int(bool(prior) and row.location not in seen_locations),
                    "time_deviation_hours": time_deviation,
                }
            )
            prior.append((ts, float(row.amount), row.location, row.device_id, row.beneficiary_id))

    features = pd.DataFrame(feature_rows)
    result = result.merge(features, on="original_order_marker", how="left", validate="one_to_one")
    return result.sort_values("original_order_marker").drop(columns="original_order_marker").reset_index(drop=True)
