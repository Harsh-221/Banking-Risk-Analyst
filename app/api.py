"""FastAPI service for scored transaction cases and policy-assisted review."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from src.rag.assistant import answer_question  # noqa: E402
from src.storage.database import (  # noqa: E402
    dashboard_summary,
    dashboard_activity,
    get_investigation,
    get_transaction,
    init_db,
    list_transactions,
    save_investigation,
    customer_summary,
)

logger = logging.getLogger("fraud_api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Banking Risk & Fraud Learning API",
    version="1.0.0",
    description="Synthetic-data fraud risk scores and policy-grounded investigation support.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    transaction_id: str | None = None


class InvestigationUpdate(BaseModel):
    status: Literal["Open", "In Progress", "Closed"] = "Open"
    analyst_notes: str = Field(default="", max_length=4000)


@app.get("/api/health")
def health():
    try:
        summary = dashboard_summary()
        return {"status": "ok", "scored_transactions": summary["scored_transactions"]}
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc


@app.get("/api/dashboard")
def dashboard():
    return dashboard_summary()


@app.get("/api/dashboard/activity")
def dashboard_activity_route():
    return dashboard_activity()


@app.get("/api/transactions")
def transactions(
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] | None = None,
    query: str | None = Query(default=None, max_length=100),
    limit: int = Query(default=100, ge=1, le=5000),
):
    return list_transactions(risk_level=risk_level, query=query, limit=limit)


@app.get("/api/transactions/{transaction_id}")
def transaction_detail(transaction_id: str):
    record = get_transaction(transaction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Transaction not found in the scored holdout window")
    record["investigation"] = get_investigation(transaction_id)
    return record


@app.get("/api/customers/{customer_id}")
def customer_detail(customer_id: str):
    record = customer_summary(customer_id)
    if not record:
        raise HTTPException(status_code=404, detail="Customer not found in the scored holdout window")
    return record


@app.get("/api/alerts")
def alerts(
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH",
    limit: int = Query(default=100, ge=1, le=5000),
):
    return list_transactions(risk_level=risk_level, limit=limit)


@app.post("/api/assistant/ask")
def assistant_ask(request: AskRequest):
    transaction = get_transaction(request.transaction_id) if request.transaction_id else None
    if request.transaction_id and transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found in the scored holdout window")
    try:
        return answer_question(request.question, transaction)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/transactions/{transaction_id}/report")
def investigation_report(transaction_id: str):
    transaction = get_transaction(transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found in the scored holdout window")
    question = (
        "Prepare a concise investigation report: summarize why this transaction differs from prior customer behavior, "
        "identify policy-supported checks for an analyst, and state what evidence should be recorded. Do not conclude that fraud is proven."
    )
    try:
        guidance = answer_question(question, transaction)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "transaction": transaction,
        "model_assessment": {
            "model_probability_signal": transaction["model_probability"],
            "behavior_anomaly_score": transaction["behavior_anomaly_score"],
            "transaction_risk_score": transaction["transaction_risk_score"],
            "risk_level": transaction["risk_level"],
            "risk_factors": transaction["risk_factors"],
        },
        "policy_guidance": guidance["answer"],
        "sources": guidance["sources"],
        "generation_mode": guidance["mode"],
    }


@app.put("/api/transactions/{transaction_id}/investigation")
def update_investigation(transaction_id: str, update: InvestigationUpdate):
    result = save_investigation(transaction_id, update.status, update.analyst_notes)
    if result is None:
        raise HTTPException(status_code=404, detail="Transaction not found in the scored holdout window")
    return result


if os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG":
    logging.basicConfig(level=logging.DEBUG)
