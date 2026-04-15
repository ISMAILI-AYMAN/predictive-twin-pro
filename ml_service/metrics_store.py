from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, List


def _ensure_db_parent(db_path: str) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect(db_path: str) -> sqlite3.Connection:
    path = _ensure_db_parent(db_path)
    return sqlite3.connect(path, timeout=10)


def _is_corruption_error(exc: sqlite3.DatabaseError) -> bool:
    lowered = str(exc).lower()
    return "malformed" in lowered or "disk image is malformed" in lowered


def _reset_corrupted_db(db_path: str) -> None:
    path = _ensure_db_parent(db_path)
    if path.exists():
        backup = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
        path.replace(backup)


def _with_db_retry(operation, db_path: str, retries: int = 2):
    last_exc: sqlite3.OperationalError | None = None
    for attempt in range(retries + 1):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            last_exc = exc
            # Recover from transient mount/path readiness issues in containers.
            init_db(db_path)
            if attempt < retries:
                time.sleep(0.2 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected SQLite retry failure without exception")


def init_db(db_path: str) -> None:
    try:
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
    except sqlite3.DatabaseError as exc:
        if not _is_corruption_error(exc):
            raise
        # Preserve the corrupt DB for inspection, then recreate a healthy one.
        _reset_corrupted_db(db_path)
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
    def _insert() -> None:
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

    _with_db_retry(_insert, db_path)


def fetch_recent_events(db_path: str, limit: int = 200) -> List[Dict[str, object]]:
    def _fetch() -> List[Dict[str, object]]:
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

    return _with_db_retry(_fetch, db_path)
