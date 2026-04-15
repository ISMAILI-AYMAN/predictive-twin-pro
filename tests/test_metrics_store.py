from pathlib import Path

from ml_service.metrics_store import fetch_recent_events, init_db, insert_event


def test_metrics_store_init_insert_fetch(tmp_path: Path) -> None:
    db_path = tmp_path / "metrics.db"
    init_db(str(db_path))

    insert_event(
        str(db_path),
        event_time=1_700_000_000.0,
        asset_id="A1",
        anomaly_score=0.42,
        rul_hours=900.0,
        drift_severity="warn",
        drift_score=0.02,
        model_backend="dl_trained",
    )
    events = fetch_recent_events(str(db_path), limit=10)
    assert len(events) == 1
    assert events[0]["asset_id"] == "A1"
    assert events[0]["drift_severity"] == "warn"
