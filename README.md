# Predictive Twin Pro

End-to-end Industry 4.0 predictive maintenance platform for real-time monitoring, anomaly detection, drift monitoring, and Remaining Useful Life (RUL) estimation.

`Status: Active` `Backend: DL + Fallback` `Stack: Kafka/Spark/FastAPI/Streamlit/MLflow` `Storage: SQLite`

## Why this project matters
- Shows end-to-end delivery across data, ML, API, and monitoring.
- Applies practical MLOps: artifacts, fallback logic, drift signaling, MLflow.
- Reflects industrial constraints: noisy telemetry, faults, near-real-time responses.
- Includes portfolio evidence: tests, Docker stack, live dashboard.

## Quick start
```bash
docker compose up --build
```

Dashboard: `http://localhost:8501`  
Inference API docs: `http://localhost:8000/docs`

## What it does
- Simulates machine telemetry with degradation and fault injection.
- Streams data through Kafka and Spark Structured Streaming.
- Serves anomaly score + RUL through FastAPI inference.
- Tracks drift severity and retraining policy signals.
- Persists events in SQLite and exposes `GET /metrics/history`.
- Visualizes KPIs, trends, and drift in Streamlit.
- Defaults to LSTM autoencoder inference with fallback backends.

## Current status
- Docker stack is up: `kafka`, `spark-master`, `spark-worker`, `inference_api`, `dashboard`, `mlflow`.
- Active inference backend: `dl_trained`.
- API endpoints validated: `GET /health`, `POST /infer`, `GET /metrics`, `GET /metrics/history`.
- Local validations passed: unit tests, smoke test, integration check.

## Architecture
Compact flow:
`Simulator` -> `Kafka` -> `Spark Structured Streaming` -> `FastAPI Inference (DL + fallback)` -> `SQLite metrics store` -> `Streamlit Dashboard`

Control loop:
`Inference + Drift Detector` -> `Retraining policy signal` -> `MLflow training/registry workflow`

- `data_simulator/generator.py`: synthetic machine behavior and optional fault injection.
- `data_simulator/kafka_producer.py`: stream publisher to Kafka topic.
- `streaming/spark_processor.py`: rolling feature engineering on streaming data.
- `ml_service/inference_api.py`: online inference and metrics endpoint.
- `ml_service/drift_detector.py`: drift signal + retraining policy logic.
- `dashboard/app.py`: operations dashboard.

See also:
- `docs/architecture.md`
- `docs/runbook.md`
- `docs/retraining_policy.md`
- `docs/demo_script.md`
- `docs/whitepaper.md`

## Local development
```bash
# API
venv\Scripts\python.exe -m uvicorn ml_service.inference_api:app --reload

# Dashboard
venv\Scripts\python.exe -m streamlit run dashboard/app.py

# Produce events
venv\Scripts\python.exe data_simulator/kafka_producer.py --duration 300
venv\Scripts\python.exe data_simulator/kafka_producer.py --duration 120 --inject-fault
```

## Quality gates
- Unit tests: `venv\Scripts\python.exe -m pytest -q`
- Lint: `venv\Scripts\python.exe -m ruff check .`
- Smoke test (API path): `venv\Scripts\python.exe scripts/smoke_test.py`
- Integration check: `venv\Scripts\python.exe scripts/integration_ci_check.py`
- CI: `.github/workflows/ci.yml`

## Project scope (MVP done definition)
- End-to-end path exists: simulator -> Kafka -> Spark -> inference -> dashboard.
- Drift warnings and critical alerts are emitted from live-like vibration shifts.
- One-command boot with Docker Compose and documented troubleshooting.
- Tests and CI cover simulator, drift logic, and inference contract.

## Next iterations
- Add model promotion rules and stage transitions in MLflow Registry.
- Add richer Spark-to-inference handoff assertions and contract tests.
- Optimize image size/startup time for ML and dashboard containers.
- Package optional demo video when needed.

## Release Artifacts
- Dashboard proof snapshot: `assets/c__Users_sonic_AppData_Roaming_Cursor_User_workspaceStorage_18c0db82ecb2130b8f3922a58e54a9ba_images_Capture_d__cran_2026-04-15_093658-f7b5cfd0-e50f-4f94-b8a4-049bc233e8d0.png`
- Whitepaper/report: `docs/whitepaper.md`
- Demo script: `docs/demo_script.md`

