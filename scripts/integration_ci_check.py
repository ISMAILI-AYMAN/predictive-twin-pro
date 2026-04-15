from __future__ import annotations

import subprocess
import sys
import time
from typing import Dict

import requests


def run_command(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, shell=True, check=True, capture_output=True, text=True)


def wait_for_health(url: str, retries: int = 40, sleep_s: float = 2.0) -> Dict[str, str]:
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=4)
            if response.ok:
                return response.json()
        except requests.RequestException:
            pass
        time.sleep(sleep_s)
    raise RuntimeError("API health endpoint never became ready")


def run_producer_with_retries(retries: int = 6, sleep_s: float = 3.0) -> subprocess.CompletedProcess[str]:
    command = f"\"{sys.executable}\" data_simulator/kafka_producer.py --duration 8 --inject-fault"
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            result = run_command(command)
            if "connected_to_kafka" in result.stdout:
                return result
        except Exception as exc:
            last_error = exc
        time.sleep(sleep_s)
    raise RuntimeError(f"producer failed after retries: {last_error}")


def main() -> None:
    compose_ps = run_command("docker compose ps")
    required_services = ("kafka", "spark-master", "spark-worker", "inference_api", "dashboard")
    for service_name in required_services:
        if service_name not in compose_ps.stdout:
            raise RuntimeError(f"missing_service_in_compose_ps: {service_name}")

    health = wait_for_health("http://localhost:8000/health")
    print(f"api_health={health}")

    producer_run = run_producer_with_retries()
    if "producer_shutdown_ok" not in producer_run.stdout:
        raise RuntimeError("producer did not shutdown cleanly in integration check")

    payload = {
        "timestamp": time.time(),
        "asset_id": "CI_ASSET_01",
        "temp": 88.0,
        "vibration": 0.09,
        "pressure": 95.0,
        "health_index": 90.0,
        "fault_active": 1.0,
    }
    infer_response = requests.post("http://localhost:8000/infer", json=payload, timeout=5)
    infer_response.raise_for_status()
    body = infer_response.json()
    assert "anomaly_score" in body and "rul_hours" in body and "drift" in body
    print(f"infer_response={body}")

    metrics_response = requests.get("http://localhost:8000/metrics", timeout=5)
    metrics_response.raise_for_status()
    metrics = metrics_response.json()
    if metrics["requests_total"] < 1:
        raise RuntimeError("metrics did not increment requests_total")
    print(f"metrics={metrics}")


if __name__ == "__main__":
    main()
