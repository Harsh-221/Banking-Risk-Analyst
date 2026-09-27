"""SQLAlchemy persistence with SQLite by default and PostgreSQL when configured."""

from __future__ import annotations

import ast
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, create_engine, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parents[2]


class Base(DeclarativeBase):
    pass


class TransactionRecord(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    amount: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    transaction_type: Mapped[str] = mapped_column(String(40))
    merchant_category: Mapped[str] = mapped_column(String(60))
    location: Mapped[str] = mapped_column(String(80))
    device_id: Mapped[str] = mapped_column(String(40))
    beneficiary_id: Mapped[str] = mapped_column(String(40))
    is_fraud: Mapped[int] = mapped_column(Integer, default=0)
    average_transaction_amount: Mapped[float] = mapped_column(Float, default=0)
    amount_deviation: Mapped[float] = mapped_column(Float, default=1)
    transactions_last_1h: Mapped[int] = mapped_column(Integer, default=0)
    transactions_last_6h: Mapped[int] = mapped_column(Integer, default=0)
    transactions_last_24h: Mapped[int] = mapped_column(Integer, default=0)
    new_beneficiary: Mapped[int] = mapped_column(Integer, default=0)
    new_device: Mapped[int] = mapped_column(Integer, default=0)
    location_deviation: Mapped[int] = mapped_column(Integer, default=0)
    time_deviation_hours: Mapped[float] = mapped_column(Float, default=0)
    model_probability: Mapped[float] = mapped_column(Float)
    behavior_anomaly_score: Mapped[float] = mapped_column(Float)
    transaction_risk_score: Mapped[float] = mapped_column(Float, index=True)
    risk_level: Mapped[str] = mapped_column(String(10), index=True)
    risk_factors: Mapped[list] = mapped_column(JSON, default=list)


class Investigation(Base):
    __tablename__ = "investigations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.transaction_id"), index=True)
    status: Mapped[str] = mapped_column(String(30), default="Open")
    analyst_notes: Mapped[str] = mapped_column(String(4000), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


@lru_cache(maxsize=1)
def get_engine():
    url = os.getenv("DATABASE_URL", f"sqlite:///{(ROOT / 'data/banking_fraud.db').as_posix()}")
    if url.startswith("sqlite:///"):
        db_path = url.removeprefix("sqlite:///")
        if db_path != ":memory:":
            path = Path(db_path)
            if not path.is_absolute():
                path = ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


def get_session_factory():
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def _clean_reasons(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        decoded = ast.literal_eval(str(value))
        return [str(item) for item in decoded] if isinstance(decoded, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def upsert_scored_transactions(frame: pd.DataFrame, batch_size: int = 1000) -> int:
    """Insert or refresh score rows from a scoring DataFrame or CSV frame."""
    if frame.empty:
        return 0
    init_db()
    required = {"transaction_id", "customer_id", "timestamp", "transaction_risk_score", "risk_level"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing database columns: {', '.join(sorted(missing))}")
    engine = get_engine()
    table = TransactionRecord.__table__
    dialect = engine.dialect.name
    insert = sqlite_insert if dialect == "sqlite" else pg_insert if dialect == "postgresql" else None
    if insert is None:
        raise ValueError(f"Unsupported database dialect: {dialect}")
    update_columns = [column.name for column in table.columns if column.name != "transaction_id"]
    count = 0
    current_ids = {str(value) for value in frame["transaction_id"].tolist()}

    def convert(value):
        if value is None or isinstance(value, (list, dict)):
            return value
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if hasattr(value, "item"):
            return value.item()
        return value

    with engine.begin() as connection:
        stored_ids = set(connection.execute(select(table.c.transaction_id)).scalars())
        stale_ids = stored_ids - current_ids
        stale_list = list(stale_ids)
        for start in range(0, len(stale_list), 500):
            batch_ids = stale_list[start : start + 500]
            connection.execute(delete(Investigation.__table__).where(Investigation.transaction_id.in_(batch_ids)))
            connection.execute(delete(table).where(table.c.transaction_id.in_(batch_ids)))

        records = frame.to_dict("records")
        for start in range(0, len(records), batch_size):
            batch = []
            for source in records[start : start + batch_size]:
                row = {key: convert(source.get(key)) for key in table.columns.keys() if key in source}
                row["risk_factors"] = _clean_reasons(source.get("risk_factors"))
                batch.append(row)
            statement = insert(table).values(batch)
            excluded = statement.excluded
            statement = statement.on_conflict_do_update(
                index_elements=["transaction_id"],
                set_={column: getattr(excluded, column) for column in update_columns},
            )
            connection.execute(statement)
            count += len(batch)
    return count


def transaction_to_dict(record: TransactionRecord) -> dict:
    return {
        column.name: getattr(record, column.name)
        for column in TransactionRecord.__table__.columns
    } | {"timestamp": record.timestamp.isoformat() if record.timestamp else None}


def list_transactions(risk_level: str | None = None, query: str | None = None, limit: int = 100) -> list[dict]:
    Session = get_session_factory()
    with Session() as session:
        statement = select(TransactionRecord)
        if risk_level:
            statement = statement.where(TransactionRecord.risk_level == risk_level.upper())
        if query:
            statement = statement.where(
                (TransactionRecord.transaction_id.ilike(f"%{query}%"))
                | (TransactionRecord.customer_id.ilike(f"%{query}%"))
            )
        statement = statement.order_by(TransactionRecord.transaction_risk_score.desc()).limit(limit)
        return [transaction_to_dict(record) for record in session.scalars(statement)]


def get_transaction(transaction_id: str) -> dict | None:
    Session = get_session_factory()
    with Session() as session:
        record = session.get(TransactionRecord, transaction_id)
        return transaction_to_dict(record) if record else None


def dashboard_summary() -> dict:
    Session = get_session_factory()
    with Session() as session:
        total = session.scalar(select(func.count()).select_from(TransactionRecord)) or 0
        fraud = session.scalar(select(func.count()).select_from(TransactionRecord).where(TransactionRecord.is_fraud == 1)) or 0
        bands = {
            band: count for band, count in session.execute(
                select(TransactionRecord.risk_level, func.count()).group_by(TransactionRecord.risk_level)
            ).all()
        }
        high_amount = session.scalar(
            select(func.coalesce(func.sum(TransactionRecord.amount), 0)).where(TransactionRecord.risk_level == "HIGH")
        ) or 0
        return {"scored_transactions": total, "synthetic_labeled_fraud": fraud, "risk_bands": bands, "high_risk_amount": float(high_amount)}


def dashboard_activity() -> dict:
    Session = get_session_factory()
    engine = get_engine()
    if engine.dialect.name == "postgresql":
        date_bucket = func.date_trunc("day", TransactionRecord.timestamp)
    else:
        date_bucket = func.date(TransactionRecord.timestamp)
    with Session() as session:
        daily_rows = session.execute(
            select(date_bucket, func.count()).group_by(date_bucket).order_by(date_bucket)
        ).all()
        location_rows = session.execute(
            select(TransactionRecord.location, func.count())
            .group_by(TransactionRecord.location)
            .order_by(func.count().desc())
            .limit(8)
        ).all()
    daily = []
    for date_value, count in daily_rows:
        if hasattr(date_value, "isoformat"):
            date_value = date_value.date().isoformat() if isinstance(date_value, datetime) else date_value.isoformat()
        daily.append({"date": str(date_value), "transactions": int(count)})
    return {"daily": daily, "locations": [{"location": location, "transactions": int(count)} for location, count in location_rows]}


def customer_summary(customer_id: str) -> dict | None:
    Session = get_session_factory()
    with Session() as session:
        rows = session.scalars(
            select(TransactionRecord).where(TransactionRecord.customer_id == customer_id)
            .order_by(TransactionRecord.timestamp.desc()).limit(200)
        ).all()
        if not rows:
            return None
        amounts = [row.amount for row in rows]
        return {
            "customer_id": customer_id,
            "transactions_in_scored_window": len(rows),
            "average_amount": sum(amounts) / len(amounts),
            "synthetic_labeled_fraud": sum(row.is_fraud for row in rows),
            "high_risk_alerts": sum(row.risk_level == "HIGH" for row in rows),
            "transactions": [transaction_to_dict(row) for row in rows],
        }


def save_investigation(transaction_id: str, status: str, analyst_notes: str) -> dict | None:
    Session = get_session_factory()
    with Session.begin() as session:
        if session.get(TransactionRecord, transaction_id) is None:
            return None
        entry = session.scalar(
            select(Investigation).where(Investigation.transaction_id == transaction_id)
        )
        if entry is None:
            entry = Investigation(transaction_id=transaction_id)
            session.add(entry)
        entry.status = status
        entry.analyst_notes = analyst_notes
        entry.updated_at = datetime.now(timezone.utc)
        session.flush()
        return {"transaction_id": transaction_id, "status": entry.status, "analyst_notes": entry.analyst_notes, "updated_at": entry.updated_at.isoformat()}


def get_investigation(transaction_id: str) -> dict | None:
    Session = get_session_factory()
    with Session() as session:
        entry = session.scalar(select(Investigation).where(Investigation.transaction_id == transaction_id))
        if entry is None:
            return None
        return {"transaction_id": entry.transaction_id, "status": entry.status, "analyst_notes": entry.analyst_notes, "updated_at": entry.updated_at.isoformat()}
