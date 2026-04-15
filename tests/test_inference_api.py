from fastapi.testclient import TestClient

from ml_service.inference_api import app


def test_infer_endpoint_contract() -> None:
    client = TestClient(app)
    payload = {
        "timestamp": 1_700_000_000.0,
        "asset_id": "A1",
        "temp": 78.0,
        "vibration": 0.03,
        "pressure": 99.0,
        "health_index": 95.0,
        "fault_active": 0.0,
    }
    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "anomaly_score" in body
    assert "rul_hours" in body
    assert "drift" in body


def test_metrics_history_endpoint() -> None:
    client = TestClient(app)
    payload = {
        "timestamp": 1_700_000_123.0,
        "asset_id": "A2",
        "temp": 79.0,
        "vibration": 0.04,
        "pressure": 98.0,
        "health_index": 94.0,
        "fault_active": 1.0,
    }
    infer_resp = client.post("/infer", json=payload)
    assert infer_resp.status_code == 200

    history_resp = client.get("/metrics/history", params={"limit": 5})
    assert history_resp.status_code == 200
    body = history_resp.json()
    assert "events" in body
    assert "count" in body
    assert body["count"] >= 1
