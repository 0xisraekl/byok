"""
spend_tracker.py — Local Spend + Routing Log

Stores every routing decision and API call in a local SQLite database.
This gives you:
  - Per-model monthly spend tracking (to enforce limits)
  - Full history of what BYOK decided and why
  - Cost savings reports vs. always using your most expensive model

Uses Python's built-in sqlite3 — no extra dependencies.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional


@dataclass
class UsageRecord:
    id: int
    timestamp: str
    model_name: str
    provider: str
    task_type: str
    difficulty: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    routing_reason: str


class SpendTracker:
    """
    Lightweight SQLite-backed tracker for model usage and spend.

    The database is created automatically at the given path.
    Default: ./byok.db in your project directory.

    Note: pass ":memory:" for an in-memory database (used in tests).
    We hold an open connection in that case because sqlite3 creates a
    brand-new empty database on every connect(":memory:") call.
    """

    def __init__(self, db_path: str | Path = "byok.db"):
        self.db_path = str(db_path)
        # For :memory: databases keep one persistent connection open.
        # For file-based databases open/close per query (safe for multiprocess).
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:")
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Return the connection to use for a query."""
        if self._mem_conn is not None:
            return self._mem_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create the database and table if they don't exist yet."""
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                model_name      TEXT    NOT NULL,
                provider        TEXT    NOT NULL,
                task_type       TEXT    NOT NULL,
                difficulty      TEXT    NOT NULL DEFAULT 'unknown',
                input_tokens    INTEGER NOT NULL DEFAULT 0,
                output_tokens   INTEGER NOT NULL DEFAULT 0,
                cost_usd        REAL    NOT NULL DEFAULT 0.0,
                routing_reason  TEXT    NOT NULL DEFAULT ''
            )
        """)
        conn.commit()

    def log(
        self,
        model_name: str,
        provider: str,
        task_type: str,
        difficulty: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        routing_reason: str = "",
    ) -> None:
        """Record one API call."""
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO usage_log
                (timestamp, model_name, provider, task_type, difficulty,
                 input_tokens, output_tokens, cost_usd, routing_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                model_name,
                provider,
                task_type,
                difficulty,
                input_tokens,
                output_tokens,
                cost_usd,
                routing_reason,
            ),
        )
        conn.commit()

    def get_monthly_spend(self, model_name: str) -> float:
        """
        How much has been spent on this model in the current calendar month?
        Used by the router to enforce spend limits.
        """
        today = date.today()
        month_start = date(today.year, today.month, 1).isoformat()
        conn = self._connect()
        row = conn.execute(
            """
            SELECT COALESCE(SUM(cost_usd), 0.0)
            FROM usage_log
            WHERE model_name = ? AND timestamp >= ?
            """,
            (model_name, month_start),
        ).fetchone()
        return row[0] if row else 0.0

    def get_all_monthly_spend(self) -> dict[str, float]:
        """Monthly spend for every model — used in CLI reports."""
        today = date.today()
        month_start = date(today.year, today.month, 1).isoformat()
        conn = self._connect()
        rows = conn.execute(
            """
            SELECT model_name, COALESCE(SUM(cost_usd), 0.0)
            FROM usage_log
            WHERE timestamp >= ?
            GROUP BY model_name
            """,
            (month_start,),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_recent(self, limit: int = 20) -> list[UsageRecord]:
        """Fetch the most recent routing decisions."""
        conn = self._connect()
        rows = conn.execute(
                """
                SELECT id, timestamp, model_name, provider, task_type,
                       difficulty, input_tokens, output_tokens, cost_usd, routing_reason
                FROM usage_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            UsageRecord(
                id=r[0],
                timestamp=r[1],
                model_name=r[2],
                provider=r[3],
                task_type=r[4],
                difficulty=r[5],
                input_tokens=r[6],
                output_tokens=r[7],
                cost_usd=r[8],
                routing_reason=r[9],
            )
            for r in rows
        ]

    def total_spent(self) -> float:
        """All-time total spend across all models."""
        conn = self._connect()
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM usage_log"
        ).fetchone()
        return row[0] if row else 0.0

    def total_requests(self) -> int:
        """All-time total number of routed requests."""
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM usage_log").fetchone()
        return row[0] if row else 0
