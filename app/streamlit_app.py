"""Streamlit analyst console backed by the FastAPI service."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 10

st.set_page_config(page_title="Banking Risk Analyst", page_icon="🏦", layout="wide")
st.title("🏦 Banking Risk & Fraud Analyst")
st.caption("Synthetic learning prototype · Scores prioritize investigation and do not establish fraud.")


def api_get(path: str, params: dict | None = None):
    try:
        response = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API unavailable at {API}. Start the API server and run `python -m src.pipeline` first. Details: {exc}")
        st.stop()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


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
            response = requests.put(
                f"{API}/api/transactions/{record['transaction_id']}/investigation",
                json={"status": status, "analyst_notes": notes}, timeout=TIMEOUT,
            )
            response.raise_for_status()
            st.success("Investigation saved.")
        except requests.RequestException as exc:
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
        st.dataframe(frame[["transaction_id", "customer_id", "amount", "timestamp", "risk_level", "transaction_risk_score"]].head(20), use_container_width=True, hide_index=True)
    st.info("Fraud labels and metrics in this application come from generated synthetic data, not actual bank cases.")

elif page == "Transaction Explorer":
    query = st.text_input("Transaction ID or customer ID", placeholder="TX0092381 or C04620")
    if query:
        exact = None
        if query.upper().startswith("TX"):
            try:
                response = requests.get(f"{API}/api/transactions/{query.upper()}", timeout=TIMEOUT)
                if response.ok:
                    exact = response.json()
            except requests.RequestException:
                pass
        if exact:
            render_transaction(exact)
            render_investigation_editor(exact)
        else:
            matches = api_get("/api/transactions", {"query": query, "limit": 100})
            if matches:
                st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
            else:
                st.info("No scored transaction matched that search in the holdout window.")

elif page == "Fraud Alerts":
    level = st.selectbox("Risk level", ["HIGH", "MEDIUM", "LOW"], index=0)
    alerts = api_get("/api/alerts", {"risk_level": level, "limit": 500})
    if alerts:
        frame = pd.DataFrame(alerts)
        st.dataframe(frame[["transaction_id", "customer_id", "amount", "timestamp", "transaction_risk_score", "risk_factors"]], use_container_width=True, hide_index=True)
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
            response = requests.get(f"{API}/api/customers/{customer_id}", timeout=TIMEOUT)
            if response.status_code == 404:
                st.info("Customer not found in the scored holdout window.")
            else:
                response.raise_for_status()
                customer = response.json()
                cols = st.columns(4)
                cols[0].metric("Transactions in scored window", customer["transactions_in_scored_window"])
                cols[1].metric("Average amount", money(customer["average_amount"]))
                cols[2].metric("High risk alerts", customer["high_risk_alerts"])
                cols[3].metric("Synthetic fraud labels", customer["synthetic_labeled_fraud"])
                st.dataframe(pd.DataFrame(customer["transactions"]), use_container_width=True, hide_index=True)
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
                report_response = requests.post(f"{API}/api/transactions/{transaction_id}/report", timeout=60)
                report_response.raise_for_status()
                report = report_response.json()
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
            except requests.RequestException as exc:
                st.error(f"Could not generate report: {exc}")

st.sidebar.caption("Local synthetic prototype · Do not use for actual transaction decisions.")
