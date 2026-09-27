"""Transparent prototype risk scoring on top of a fraud model.

The model signal and behavior anomaly signal are kept visible separately. The
combined 0-100 score is an illustrative investigation-priority score, not a
calibrated fraud probability or a production banking decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd

_WEIGHTS = {
    "amount": 0.25,
    "new_beneficiary": 0.20,
    "new_device": 0.15,
    "location": 0.15,
    "time": 0.15,
    "velocity": 0.10,
}


def risk_level(score: float) -> str:
    """Map a 0-100 transaction risk score to a prototype risk band."""
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    if score < 30:
        return "LOW"
    if score < 70:
        return "MEDIUM"
    return "HIGH"


def score_transaction(model_probability: float, features: Mapping) -> dict:
    """Combine a model output and behavior signals into an explainable score.

    `model_probability` is the model's positive-class output. With the current
    class-weighted baseline it should be treated as a model signal, not a
    calibrated real-world fraud probability.
    """
    if not 0.0 <= model_probability <= 1.0:
        raise ValueError("model_probability must be between 0 and 1")

    amount_ratio = max(float(features.get("amount_deviation", 1.0)), 0.0)
    amount_signal = min(math.log(max(amount_ratio, 1.0), 20), 1.0)
    new_beneficiary = int(bool(features.get("new_beneficiary", 0)))
    new_device = int(bool(features.get("new_device", 0)))
    location_deviation = int(bool(features.get("location_deviation", 0)))
    time_hours = max(float(features.get("time_deviation_hours", 0)), 0.0)
    time_signal = min(time_hours / 12.0, 1.0)
    tx_1h = max(float(features.get("transactions_last_1h", 0)), 0.0)
    tx_24h = max(float(features.get("transactions_last_24h", 0)), 0.0)
    velocity_signal = min(max(tx_1h / 5.0, tx_24h / 15.0), 1.0)

    signals = {
        "amount": amount_signal,
        "new_beneficiary": float(new_beneficiary),
        "new_device": float(new_device),
        "location": float(location_deviation),
        "time": time_signal,
        "velocity": velocity_signal,
    }
    anomaly_score = sum(_WEIGHTS[name] * value for name, value in signals.items())
    combined_score = 100 * (0.70 * model_probability + 0.30 * anomaly_score)

    reasons = []
    if amount_ratio >= 3:
        reasons.append(f"Amount is {amount_ratio:.1f}x the customer's prior average")
    if new_beneficiary:
        reasons.append("Beneficiary has not appeared in this customer's prior transactions")
    if new_device:
        reasons.append("Device has not appeared in this customer's prior transactions")
    if location_deviation:
        reasons.append("Location has not appeared in this customer's prior transactions")
    if time_hours >= 6:
        reasons.append(f"Transaction time differs by {time_hours:.0f} hours from prior activity")
    if tx_1h >= 4 or tx_24h >= 12:
        reasons.append(f"Elevated transaction frequency ({tx_1h:.0f} in 1h, {tx_24h:.0f} in 24h)")
    if not reasons and model_probability >= 0.5:
        reasons.append("Model score is elevated; review the transaction and its history")

    return {
        "model_probability": float(model_probability),
        "behavior_anomaly_score": round(anomaly_score, 4),
        "transaction_risk_score": round(combined_score, 2),
        "risk_level": risk_level(combined_score),
        "risk_factors": reasons,
    }


def score_transactions(probabilities: Sequence[float], feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Score aligned model outputs and feature rows, preserving their indexes."""
    if len(probabilities) != len(feature_frame):
        raise ValueError("probabilities and feature_frame must have the same number of rows")
    scored = [score_transaction(float(probability), row) for probability, row in zip(probabilities, feature_frame.to_dict("records"))]
    return pd.DataFrame(scored, index=feature_frame.index)
