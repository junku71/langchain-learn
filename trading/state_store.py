from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from trading.models import ProtectionState


class TradingStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS trading_sessions (
                    session_id TEXT PRIMARY KEY,
                    trade_date TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    buy_enabled INTEGER NOT NULL DEFAULT 1,
                    kill_switch TEXT NOT NULL DEFAULT 'NORMAL',
                    payload TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS ux_session_strategy
                    ON trading_sessions(trade_date, strategy_version);
                CREATE TABLE IF NOT EXISTS job_runs (
                    job_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS order_intents (
                    idempotency_key TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    broker_order_id TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS position_protection (
                    ticker TEXT PRIMARY KEY,
                    stop_loss REAL,
                    take_profit REAL,
                    trailing_stop_pct REAL,
                    trailing_stop REAL,
                    highest_price REAL NOT NULL,
                    strategy TEXT NOT NULL DEFAULT 'LEGACY',
                    atr REAL,
                    atr_multiple REAL,
                    donchian_period INTEGER,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    entity_key TEXT,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS system_controls (
                    control_key TEXT PRIMARY KEY,
                    control_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS notification_events (
                    event_key TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS daily_recommendations (
                    trade_date TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(trade_date, strategy_version)
                );
                CREATE TABLE IF NOT EXISTS scheduled_orders (
                    reservation_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    execute_on TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    broker_order_id TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_scheduled_orders_due
                    ON scheduled_orders(status, execute_on);
                """
            )
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(position_protection)")
            }
            migrations = {
                "strategy": "TEXT NOT NULL DEFAULT 'LEGACY'",
                "atr": "REAL",
                "atr_multiple": "REAL",
                "donchian_period": "INTEGER",
            }
            for name, definition in migrations.items():
                if name not in columns:
                    db.execute(
                        f"ALTER TABLE position_protection ADD COLUMN {name} {definition}"
                    )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def claim_job(self, job_key: str, payload: dict | None = None) -> bool:
        try:
            with self._connection() as db:
                db.execute(
                    "INSERT INTO job_runs(job_key,status,payload,started_at) VALUES(?,?,?,?)",
                    (job_key, "RUNNING", json.dumps(payload or {}), self._now()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get_control(self, key: str, default=None):
        with self._connection() as db:
            row = db.execute(
                "SELECT control_value FROM system_controls WHERE control_key=?",
                (key,),
            ).fetchone()
        return default if row is None else json.loads(row["control_value"])

    def set_control(self, key: str, value) -> None:
        with self._connection() as db:
            db.execute(
                """
                INSERT INTO system_controls(control_key,control_value,updated_at)
                VALUES(?,?,?)
                ON CONFLICT(control_key) DO UPDATE SET
                    control_value=excluded.control_value,
                    updated_at=excluded.updated_at
                """,
                (key, json.dumps(value), self._now()),
            )

    def enqueue_scheduled_order(
        self,
        reservation_id: str,
        proposal_id: str,
        execute_on: str,
        ticker: str,
        side: str,
        quantity: int,
        payload: dict,
    ) -> bool:
        try:
            with self._connection() as db:
                db.execute(
                    """
                    INSERT INTO scheduled_orders(
                        reservation_id,proposal_id,execute_on,ticker,side,
                        quantity,status,payload,updated_at
                    ) VALUES(?,?,?,?,?,?,'QUEUED',?,?)
                    """,
                    (
                        reservation_id, proposal_id, execute_on, ticker, side,
                        quantity, json.dumps(payload, default=str), self._now(),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def list_scheduled_orders(
        self, *, statuses: tuple[str, ...] | None = None, limit: int = 100
    ) -> list[dict]:
        query = "SELECT * FROM scheduled_orders"
        values: list[object] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            values.extend(statuses)
        query += " ORDER BY execute_on ASC, CASE side WHEN 'SELL' THEN 0 ELSE 1 END, updated_at ASC LIMIT ?"
        values.append(max(1, limit))
        with self._connection() as db:
            rows = db.execute(query, values).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def claim_scheduled_order(self, reservation_id: str, execute_on: str) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                """
                UPDATE scheduled_orders SET status='EXECUTING',updated_at=?
                WHERE reservation_id=? AND status='QUEUED' AND execute_on<=?
                """,
                (self._now(), reservation_id, execute_on),
            )
        return cursor.rowcount == 1

    def cancel_scheduled_order(self, reservation_id: str) -> bool:
        with self._connection() as db:
            cursor = db.execute(
                """
                UPDATE scheduled_orders SET status='CANCELLED',updated_at=?
                WHERE reservation_id=? AND status='QUEUED'
                """,
                (self._now(), reservation_id),
            )
        return cursor.rowcount == 1

    def amend_scheduled_order_quantity(
        self, reservation_id: str, quantity: int
    ) -> bool:
        with self._connection() as db:
            row = db.execute(
                """
                SELECT payload FROM scheduled_orders
                WHERE reservation_id=? AND status='QUEUED'
                """,
                (reservation_id,),
            ).fetchone()
            if row is None:
                return False
            payload = json.loads(row["payload"])
            payload["quantity"] = quantity
            cursor = db.execute(
                """
                UPDATE scheduled_orders
                SET quantity=?,payload=?,updated_at=?
                WHERE reservation_id=? AND status='QUEUED'
                """,
                (
                    quantity, json.dumps(payload, default=str), self._now(),
                    reservation_id,
                ),
            )
        return cursor.rowcount == 1

    def finish_scheduled_order(
        self,
        reservation_id: str,
        status: str,
        *,
        broker_order_id: str = "",
        payload: dict | None = None,
    ) -> None:
        with self._connection() as db:
            existing = db.execute(
                "SELECT payload FROM scheduled_orders WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            merged = json.loads(existing["payload"]) if existing else {}
            merged.update(payload or {})
            db.execute(
                """
                UPDATE scheduled_orders
                SET status=?,broker_order_id=?,payload=?,updated_at=?
                WHERE reservation_id=?
                """,
                (
                    status, broker_order_id, json.dumps(merged, default=str),
                    self._now(), reservation_id,
                ),
            )

    def get_bool_control(self, key: str, default: bool) -> bool:
        return bool(self.get_control(key, default))

    def finish_job(self, job_key: str, status: str, payload: dict | None = None) -> None:
        with self._connection() as db:
            db.execute(
                "UPDATE job_runs SET status=?,payload=?,finished_at=? WHERE job_key=?",
                (status, json.dumps(payload or {}, default=str), self._now(), job_key),
            )

    def upsert_session(
        self,
        session_id: str,
        trade_date: str,
        strategy_version: str,
        status: str,
        *,
        buy_enabled: bool,
        kill_switch: str = "NORMAL",
        payload: dict | None = None,
    ) -> None:
        with self._connection() as db:
            db.execute(
                """
                INSERT INTO trading_sessions(
                    session_id,trade_date,strategy_version,status,buy_enabled,
                    kill_switch,payload,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status=excluded.status,
                    buy_enabled=excluded.buy_enabled,
                    kill_switch=excluded.kill_switch,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    session_id, trade_date, strategy_version, status,
                    int(buy_enabled), kill_switch,
                    json.dumps(payload or {}, default=str), self._now(),
                ),
            )

    def get_session(self, session_id: str) -> dict | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM trading_sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["buy_enabled"] = bool(result["buy_enabled"])
        result["payload"] = json.loads(result["payload"])
        return result

    def get_latest_session(self) -> dict | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM trading_sessions ORDER BY trade_date DESC, updated_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["buy_enabled"] = bool(result["buy_enabled"])
        result["payload"] = json.loads(result["payload"])
        return result

    def set_session_controls(
        self,
        session_id: str,
        *,
        buy_enabled: bool | None = None,
        kill_switch: str | None = None,
    ) -> bool:
        assignments: list[str] = []
        values: list[object] = []
        if buy_enabled is not None:
            assignments.append("buy_enabled=?")
            values.append(int(buy_enabled))
        if kill_switch is not None:
            assignments.append("kill_switch=?")
            values.append(kill_switch)
        if not assignments:
            return False
        assignments.append("updated_at=?")
        values.append(self._now())
        values.append(session_id)
        with self._connection() as db:
            cursor = db.execute(
                f"UPDATE trading_sessions SET {', '.join(assignments)} WHERE session_id=?",
                values,
            )
        return cursor.rowcount > 0

    def create_order_intent(
        self,
        idempotency_key: str,
        session_id: str,
        ticker: str,
        side: str,
        payload: dict,
    ) -> bool:
        try:
            with self._connection() as db:
                db.execute(
                    """
                    INSERT INTO order_intents(
                        idempotency_key,session_id,ticker,side,status,payload,updated_at
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        idempotency_key, session_id, ticker, side, "INTENT",
                        json.dumps(payload, default=str), self._now(),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def update_order_intent(
        self,
        idempotency_key: str,
        status: str,
        *,
        broker_order_id: str = "",
        payload: dict | None = None,
    ) -> None:
        with self._connection() as db:
            existing = db.execute(
                "SELECT payload FROM order_intents WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            merged_payload = {}
            if existing is not None:
                merged_payload.update(json.loads(existing["payload"]))
            merged_payload.update(payload or {})
            db.execute(
                """
                UPDATE order_intents
                SET status=?,broker_order_id=?,payload=?,updated_at=?
                WHERE idempotency_key=?
                """,
                (
                    status, broker_order_id, json.dumps(merged_payload, default=str),
                    self._now(), idempotency_key,
                ),
            )

    def import_broker_execution(self, execution: object, updated_at: str) -> bool:
        """Insert a broker-ledger order that was not created by this process."""
        order_date = str(getattr(execution, "order_date", "")).replace("-", "")
        order_id = str(getattr(execution, "order_id", ""))
        key = f"KIS:{order_date}:{order_id}"
        payload = {
            "price": (
                getattr(execution, "average_fill_price", 0)
                or getattr(execution, "order_price", 0)
            ),
            "quantity": getattr(execution, "ordered_quantity", 0),
            "filled_quantity": getattr(execution, "filled_quantity", 0),
            "remaining_quantity": getattr(execution, "remaining_quantity", 0),
            "source": "kis_daily_ledger",
        }
        try:
            with self._connection() as db:
                existing = db.execute(
                    "SELECT 1 FROM order_intents WHERE broker_order_id=? AND session_id LIKE ?",
                    (order_id, f"{order_date[:4]}-{order_date[4:6]}-{order_date[6:8]}%"),
                ).fetchone()
                if existing is not None:
                    return False
                db.execute(
                    """
                    INSERT INTO order_intents(
                        idempotency_key,session_id,ticker,side,status,payload,
                        broker_order_id,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        key, f"{order_date[:4]}-{order_date[4:6]}-{order_date[6:8]}:KIS",
                        getattr(execution, "ticker"), getattr(execution, "side"),
                        getattr(execution, "status"), json.dumps(payload, default=str),
                        order_id, updated_at,
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def count_orders(self, session_id: str) -> int:
        with self._connection() as db:
            row = db.execute(
                "SELECT COUNT(*) AS count FROM order_intents WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return int(row["count"])

    def list_order_intents(
        self, *, status: str | None = None, limit: int = 50
    ) -> list[dict]:
        query = "SELECT * FROM order_intents"
        values: list[object] = []
        if status is not None:
            query += " WHERE status=?"
            values.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        values.append(max(1, limit))
        with self._connection() as db:
            rows = db.execute(query, values).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def list_reconcilable_orders(self, limit: int = 100) -> list[dict]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT * FROM order_intents
                WHERE status IN ('SUBMITTED','PARTIALLY_FILLED')
                  AND COALESCE(broker_order_id, '') <> ''
                ORDER BY updated_at ASC LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result

    def begin_notification(self, event_key: str, channel: str, payload: dict) -> bool:
        with self._connection() as db:
            row = db.execute(
                "SELECT status FROM notification_events WHERE event_key=?",
                (event_key,),
            ).fetchone()
            if row is not None and row["status"] in {"SENDING", "SENT"}:
                return False
            db.execute(
                """
                INSERT INTO notification_events(
                    event_key,channel,status,payload,last_error,updated_at
                ) VALUES(?,?,?,?,?,?)
                ON CONFLICT(event_key) DO UPDATE SET
                    status=excluded.status,
                    payload=excluded.payload,
                    last_error=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    event_key, channel, "SENDING",
                    json.dumps(payload, default=str), None, self._now(),
                ),
            )
        return True

    def finish_notification(
        self, event_key: str, *, sent: bool, error: str = ""
    ) -> None:
        with self._connection() as db:
            db.execute(
                """
                UPDATE notification_events
                SET status=?,last_error=?,updated_at=? WHERE event_key=?
                """,
                ("SENT" if sent else "FAILED", error or None, self._now(), event_key),
            )

    def save_recommendations(
        self, trade_date: str, strategy_version: str, payload: list[dict]
    ) -> None:
        with self._connection() as db:
            db.execute(
                """
                INSERT INTO daily_recommendations(
                    trade_date,strategy_version,payload,updated_at
                ) VALUES(?,?,?,?)
                ON CONFLICT(trade_date,strategy_version) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    trade_date, strategy_version,
                    json.dumps(payload, ensure_ascii=False, default=str), self._now(),
                ),
            )

    def get_recommendations(
        self, trade_date: str, strategy_version: str
    ) -> list[dict] | None:
        with self._connection() as db:
            row = db.execute(
                """
                SELECT payload FROM daily_recommendations
                WHERE trade_date=? AND strategy_version=?
                """,
                (trade_date, strategy_version),
            ).fetchone()
        return None if row is None else json.loads(row["payload"])

    def save_protection(self, state: ProtectionState) -> None:
        with self._connection() as db:
            db.execute(
                """
                INSERT INTO position_protection(
                    ticker,stop_loss,take_profit,trailing_stop_pct,
                    trailing_stop,highest_price,strategy,atr,atr_multiple,
                    donchian_period,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET
                    stop_loss=excluded.stop_loss,
                    take_profit=excluded.take_profit,
                    trailing_stop_pct=excluded.trailing_stop_pct,
                    trailing_stop=excluded.trailing_stop,
                    highest_price=excluded.highest_price,
                    strategy=excluded.strategy,
                    atr=excluded.atr,
                    atr_multiple=excluded.atr_multiple,
                    donchian_period=excluded.donchian_period,
                    updated_at=excluded.updated_at
                """,
                (
                    state.ticker, state.stop_loss, state.take_profit,
                    state.trailing_stop_pct, state.trailing_stop,
                    state.highest_price, state.strategy, state.atr,
                    state.atr_multiple, state.donchian_period,
                    state.updated_at.isoformat(),
                ),
            )

    def get_protection(self, ticker: str) -> ProtectionState | None:
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM position_protection WHERE ticker=?", (ticker,)
            ).fetchone()
        if row is None:
            return None
        return ProtectionState(
            ticker=row["ticker"], stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            trailing_stop_pct=row["trailing_stop_pct"],
            trailing_stop=row["trailing_stop"], highest_price=row["highest_price"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
            strategy=row["strategy"], atr=row["atr"],
            atr_multiple=row["atr_multiple"],
            donchian_period=row["donchian_period"],
        )

    def delete_protection(self, ticker: str) -> None:
        with self._connection() as db:
            db.execute("DELETE FROM position_protection WHERE ticker=?", (ticker,))

    def audit(self, event_type: str, entity_key: str, payload: dict) -> None:
        with self._connection() as db:
            db.execute(
                "INSERT INTO audit_events(event_type,entity_key,payload,created_at) VALUES(?,?,?,?)",
                (event_type, entity_key, json.dumps(payload, default=str), self._now()),
            )

    def list_audit_events(self, limit: int = 50) -> list[dict]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            result.append(item)
        return result
