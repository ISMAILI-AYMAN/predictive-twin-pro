from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List


def _ensure_db_parent(db_path: str) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect(db_path: str) -> sqlite3.Connection:
    path = _ensure_db_parent(db_path)
    return sqlite3.connect(path, timeout=10)


def init_db(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time REAL NOT NULL,
                asset_id TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                rul_hours REAL NOT NULL,
                drift_severity TEXT NOT NULL,
                drift_score REAL NOT NULL,
                model_backend TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_events_event_time
            ON metrics_events(event_time DESC)
            """
        )
        conn.commit()


def insert_event(
    db_path: str,
    *,
    event_time: float,
    asset_id: str,
    anomaly_score: float,
    rul_hours: float,
    drift_severity: str,
    drift_score: float,
    model_backend: str,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO metrics_events (
                event_time,
                asset_id,
                anomaly_score,
                rul_hours,
                drift_severity,
                drift_score,
                model_backend
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_time,
                asset_id,
                anomaly_score,
                rul_hours,
                drift_severity,
                drift_score,
                model_backend,
            ),
        )
        conn.commit()


def fetch_recent_events(db_path: str, limit: int = 200) -> List[Dict[str, object]]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id,
                event_time,
                asset_id,
                anomaly_score,
                rul_hours,
                drift_severity,
                drift_score,
                model_backend
            FROM metrics_events
            ORDER BY event_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
