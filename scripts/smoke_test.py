from __future__ import annotations

import time

import requests


def main() -> None:
    base_url = "http://localhost:8000"
    payload = {
        "timestamp": time.time(),
        "asset_id": "SMOKE_ASSET_01",
        "temp": 76.0,
        "vibration": 0.025,
        "pressure": 101.0,
        "health_index": 97.0,
        "fault_active": 0.0,
    }

    health = requests.get(f"{base_url}/health", timeout=5)
    health.raise_for_status()
    print("health:", health.json())

    infer = requests.post(f"{base_url}/infer", json=payload, timeout=5)
    infer.raise_for_status()
    print("infer:", infer.json())

    metrics = requests.get(f"{base_url}/metrics", timeout=5)
    metrics.raise_for_status()
    print("metrics:", metrics.json())


if __name__ == "__main__":
    main()
