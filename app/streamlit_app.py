"""Streamlit analyst console backed by the FastAPI service."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

try:
    _secret_api_url = st.secrets.get("API_BASE_URL", "")
    for _setting_name in ("DATABASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "POLICY_DOCUMENTS_DIR"):
        _secret_value = st.secrets.get(_setting_name, "")
        if _secret_value:
            os.environ[_setting_name] = str(_secret_value)
except Exception:
    _secret_api_url = ""
API = str(_secret_api_url or os.getenv("API_BASE_URL", "")).strip().rstrip("/")
TIMEOUT = 5

st.set_page_config(page_title="Banking Risk Analyst", page_icon="🏦", layout="wide")
st.title("🏦 Banking Risk & Fraud Analyst")
st.caption("Synthetic learning prototype · Scores prioritize investigation and do not establish fraud.")


class BackendNotFound(Exception):
    """Requested transaction or customer is absent from the scored window."""


@st.cache_data(ttl=20, show_spinner=False)
def _remote_api_is_ready(base_url: str) -> bool:
    if not base_url:
        return False
    try:
        response = requests.get(f"{base_url}/api/health", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner="Preparing the local fraud pipeline for this deployment...")
def _start_embedded_backend():
    from src.pipeline import run

    run()
    return True


def _local_get(path: str, params: dict | None = None):
    from src.storage.database import (
        customer_summary,
        dashboard_activity,
        dashboard_summary,
        get_investigation,
        get_transaction,
        list_transactions,
    )

    params = params or {}
    if path == "/api/dashboard":
        return dashboard_summary()
    if path == "/api/dashboard/activity":
        return dashboard_activity()
    if path == "/api/transactions":
        return list_transactions(params.get("risk_level"), params.get("query"), int(params.get("limit", 100)))
    if path == "/api/alerts":
        return list_transactions(params.get("risk_level", "HIGH"), limit=int(params.get("limit", 100)))
    if path.startswith("/api/transactions/"):
        transaction_id = path.removeprefix("/api/transactions/").split("/", 1)[0]
        record = get_transaction(transaction_id)
        if record is None:
            raise BackendNotFound(transaction_id)
        record["investigation"] = get_investigation(transaction_id)
        return record
    if path.startswith("/api/customers/"):
        customer_id = path.removeprefix("/api/customers/")
        result = customer_summary(customer_id)
        if result is None:
            raise BackendNotFound(customer_id)
        return result
    raise ValueError(f"Unsupported local endpoint: {path}")


def _local_post(path: str, payload: dict):
    from src.rag.assistant import answer_question
    from src.storage.database import get_transaction

    if path == "/api/assistant/ask":
        transaction = get_transaction(payload["transaction_id"]) if payload.get("transaction_id") else None
        if payload.get("transaction_id") and transaction is None:
            raise BackendNotFound(payload["transaction_id"])
        return answer_question(payload["question"], transaction)
    if path.startswith("/api/transactions/") and path.endswith("/report"):
        transaction_id = path.removeprefix("/api/transactions/").removesuffix("/report").strip("/")
        transaction = get_transaction(transaction_id)
        if transaction is None:
            raise BackendNotFound(transaction_id)
        guidance = answer_question(
            "Prepare a concise investigation report: summarize why this transaction differs from prior customer behavior, "
            "identify policy-supported checks, and state what evidence should be recorded. Do not conclude that fraud is proven.",
            transaction,
        )
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
    raise ValueError(f"Unsupported local endpoint: {path}")


def _use_remote_api() -> bool:
    if _remote_api_is_ready(API):
        return True
    _start_embedded_backend()
    return False


def _http_request(method: str, path: str, *, params=None, payload=None):
    try:
        response = requests.request(method, f"{API}{path}", params=params, json=payload, timeout=TIMEOUT)
        if response.status_code == 404:
            raise BackendNotFound(path)
        response.raise_for_status()
        return response.json()
    except BackendNotFound:
        raise
    except requests.RequestException as exc:
        st.error(f"Could not reach configured API at {API}: {exc}")
        st.stop()


def api_get(path: str, params: dict | None = None):
    if _use_remote_api():
        return _http_request("GET", path, params=params)
    return _local_get(path, params)


def api_post(path: str, payload: dict):
    if _use_remote_api():
        return _http_request("POST", path, payload=payload)
    return _local_post(path, payload)


def api_put(path: str, payload: dict):
    if _use_remote_api():
        return _http_request("PUT", path, payload=payload)
    from src.storage.database import save_investigation

    parts = path.strip("/").split("/")
    if len(parts) == 4 and parts[0:2] == ["api", "transactions"] and parts[3] == "investigation":
        saved = save_investigation(parts[2], payload["status"], payload["analyst_notes"])
        if saved is None:
            raise BackendNotFound(parts[2])
        return saved
    raise ValueError(f"Unsupported local endpoint: {path}")


if API and _remote_api_is_ready(API):
    BACKEND_MODE = "remote API"
    st.sidebar.success(f"Connected to API: {API}")
else:
    BACKEND_MODE = "embedded local mode"
    try:
        _start_embedded_backend()
        if API:
            st.sidebar.warning(f"External API at {API} is unreachable; standalone mode is active.")
        else:
            st.sidebar.success("Standalone mode active. This app is using its embedded pipeline and local database.")
    except Exception as exc:
        st.error(f"Could not start the embedded project pipeline: {exc}")
        st.stop()


def money(value) -> str:
    return f"₹{float(value or 0):,.0f}"


def render_transaction(record: dict):
    left, right = st.columns(2)
    left.metric("Transaction risk", f"{record['transaction_risk_score']:.1f}/100", record["risk_level"])
    right.metric("Model signal", f"{record['model_probability']:.1%}")
    st.write(f"**Amount:** {money(record['amount'])} · **Customer:** {record['customer_id']} · **When:** {record['timestamp']}")
    st.write(f"**Location:** {record['location']} · **Device:** {record['device_id']} · **Beneficiary:** {record['beneficiary_id']}")
    st.write("**Why flagged**")
    if record.get("risk_factors"):
        for reason in record["risk_factors"]:
            st.markdown(f"- {reason}")
    else:
        st.write("No explicit behavior reason crossed its display threshold; the model still contributes to the review score.")
    st.caption(f"Behavior anomaly score: {record['behavior_anomaly_score']:.1%}. Model output is not calibrated as a real-world fraud probability.")


def render_investigation_editor(record: dict):
    current = record.get("investigation") or {}
    with st.form(f"investigation_{record['transaction_id']}"):
        status = st.selectbox("Case status", ["Open", "In Progress", "Closed"], index=["Open", "In Progress", "Closed"].index(current.get("status", "Open")))
        notes = st.text_area("Analyst notes", value=current.get("analyst_notes", ""), max_chars=4000)
        saved = st.form_submit_button("Save investigation")
    if saved:
        try:
            api_put(f"/api/transactions/{record['transaction_id']}/investigation", {"status": status, "analyst_notes": notes})
            st.success("Investigation saved.")
        except (requests.RequestException, BackendNotFound) as exc:
            st.error(f"Could not save investigation: {exc}")


page = st.sidebar.radio("Workspace", ["Dashboard", "Transaction Explorer", "Fraud Alerts", "Customer Risk", "AI Investigation Assistant", "Investigation Report"])

if page == "Dashboard":
    stats = api_get("/api/dashboard")
    bands = stats.get("risk_bands", {})
    metric_cols = st.columns(4)
    metric_cols[0].metric("Scored transactions", f"{stats.get('scored_transactions', 0):,}")
    metric_cols[1].metric("High risk", f"{bands.get('HIGH', 0):,}")
    metric_cols[2].metric("High risk amount", money(stats.get("high_risk_amount")))
    metric_cols[3].metric("Synthetic fraud labels", f"{stats.get('synthetic_labeled_fraud', 0):,}")

    activity = api_get("/api/dashboard/activity")
    if activity.get("daily"):
        daily_frame = pd.DataFrame(activity["daily"]).set_index("date")
        location_frame = pd.DataFrame(activity["locations"]).set_index("location")
        chart_cols = st.columns(2)
        chart_cols[0].subheader("Scored holdout activity by day")
        chart_cols[0].bar_chart(daily_frame)
        chart_cols[1].subheader("Scored transactions by location")
        chart_cols[1].bar_chart(location_frame)
        frame = pd.DataFrame(api_get("/api/transactions", {"limit": 20}))
        st.subheader("Highest risk transactions")
        st.dataframe(frame[["transaction_id", "customer_id", "amount", "timestamp", "risk_level", "transaction_risk_score"]].head(20), width="stretch", hide_index=True)
    st.info("Fraud labels and metrics in this application come from generated synthetic data, not actual bank cases.")

elif page == "Transaction Explorer":
    query = st.text_input("Transaction ID or customer ID", placeholder="TX0092381 or C04620")
    if query:
        exact = None
        if query.upper().startswith("TX"):
            try:
                exact = api_get(f"/api/transactions/{query.upper()}")
            except BackendNotFound:
                exact = None
        if exact:
            render_transaction(exact)
            render_investigation_editor(exact)
        else:
            matches = api_get("/api/transactions", {"query": query, "limit": 100})
            if matches:
                st.dataframe(pd.DataFrame(matches), width="stretch", hide_index=True)
            else:
                st.info("No scored transaction matched that search in the holdout window.")

elif page == "Fraud Alerts":
    level = st.selectbox("Risk level", ["HIGH", "MEDIUM", "LOW"], index=0)
    alerts = api_get("/api/alerts", {"risk_level": level, "limit": 500})
    if alerts:
        frame = pd.DataFrame(alerts)
        st.dataframe(frame[["transaction_id", "customer_id", "amount", "timestamp", "transaction_risk_score", "risk_factors"]], width="stretch", hide_index=True)
        chosen = st.selectbox("Open transaction", frame["transaction_id"].tolist())
        detail = api_get(f"/api/transactions/{chosen}")
        render_transaction(detail)
        render_investigation_editor(detail)
    else:
        st.info(f"No {level.lower()} risk transactions in the scored window.")

elif page == "Customer Risk":
    customer_id = st.text_input("Customer ID", placeholder="C04620").upper()
    if customer_id:
        try:
            customer = api_get(f"/api/customers/{customer_id}")
            cols = st.columns(4)
            cols[0].metric("Transactions in scored window", customer["transactions_in_scored_window"])
            cols[1].metric("Average amount", money(customer["average_amount"]))
            cols[2].metric("High risk alerts", customer["high_risk_alerts"])
            cols[3].metric("Synthetic fraud labels", customer["synthetic_labeled_fraud"])
            st.dataframe(pd.DataFrame(customer["transactions"]), width="stretch", hide_index=True)
        except BackendNotFound:
            st.info("Customer not found in the scored holdout window.")
        except requests.RequestException as exc:
            st.error(f"Could not load customer: {exc}")

elif page == "AI Investigation Assistant":
    st.write("Ask about investigation steps. Answers retrieve evidence from the project’s synthetic policy documents and can optionally use OpenAI for grounded generation.")
    high_alerts = api_get("/api/alerts", {"risk_level": "HIGH", "limit": 100})
    options = ["General policy question"] + [row["transaction_id"] for row in high_alerts]
    selected = st.selectbox("Add transaction context (optional)", options)
    question = st.text_area("Question", value="Why was this transaction flagged, and what should I investigate?")
    if st.button("Ask policy assistant", type="primary"):
        with st.spinner("Retrieving policy evidence..."):
            try:
                result = api_post("/api/assistant/ask", {"question": question, "transaction_id": None if selected == options[0] else selected})
                st.markdown(result["answer"])
                st.caption(f"Answer mode: {result['mode']}")
                with st.expander("Retrieved policy sources"):
                    for source in result["sources"]:
                        st.markdown(f"**{source['citation']}** · relevance {source['relevance']}")
                        st.write(source["text"])
            except requests.RequestException as exc:
                st.error(f"Assistant request failed: {exc}")

else:
    st.write("Generate a structured investigation summary from the selected transaction and retrieved policy evidence.")
    transaction_id = st.text_input("Transaction ID", placeholder="TX0052086").strip().upper()
    if st.button("Generate investigation report", type="primary", disabled=not transaction_id):
        with st.spinner("Preparing report from transaction evidence and policy documents..."):
            try:
                report = api_post(f"/api/transactions/{transaction_id}/report", {})
                record = report["transaction"]
                assessment = report["model_assessment"]
                st.subheader("Investigation Summary")
                st.markdown(report["policy_guidance"])
                st.subheader("Transaction Details")
                st.write(f"**ID:** {record['transaction_id']} · **Customer:** {record['customer_id']} · **Amount:** {money(record['amount'])} · **Time:** {record['timestamp']}")
                st.subheader("Model Assessment")
                st.write(f"**Model signal:** {assessment['model_probability_signal']:.1%} · **Risk score:** {assessment['transaction_risk_score']:.1f}/100 · **Risk level:** {assessment['risk_level']}")
                for factor in assessment["risk_factors"]:
                    st.markdown(f"- {factor}")
                st.subheader("Relevant Policy Sources")
                for source in report["sources"]:
                    st.markdown(f"- **{source['citation']}** — {source['text']}")
                st.caption(f"Generated with: {report['generation_mode']}. Analyst review remains required.")
            except (requests.RequestException, BackendNotFound) as exc:
                st.error(f"Could not generate report: {exc}")

st.sidebar.caption("Local synthetic prototype · Do not use for actual transaction decisions.")
