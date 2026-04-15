# Runbook

## One-command startup
```bash
docker compose up --build
```

## Local dev mode
1. Start infra:
   - `docker compose up kafka spark-master spark-worker`
2. Start API:
   - `venv\Scripts\python.exe -m uvicorn ml_service.inference_api:app --reload`
3. Start dashboard:
   - `venv\Scripts\python.exe -m streamlit run dashboard/app.py`
4. Run simulator producer:
   - `venv\Scripts\python.exe data_simulator/kafka_producer.py --duration 300`
   - `venv\Scripts\python.exe data_simulator/kafka_producer.py --duration 120 --inject-fault`

## Smoke check
```bash
venv\Scripts\python.exe scripts/smoke_test.py
```

## Train and Register Model (MLflow)
1. Train artifact and log run:
   - `venv\Scripts\python.exe experiments/train_model.py`
   - Produces:
     - `experiments/artifacts/dl_model.pt`
     - `experiments/artifacts/dl_model_metadata.json`
     - `experiments/artifacts/trained_model.json` (statistical fallback)
2. Optional MLflow registry setup:
   - Set `MLFLOW_TRACKING_URI`
   - Set `MLFLOW_REGISTERED_MODEL_NAME`
3. Inference runtime from MLflow artifact:
   - Set `MODEL_SOURCE=mlflow`
   - Set `MLFLOW_MODEL_URI` (or use registry indirection)
   - Restart API service

## Troubleshooting
- Kafka connection failure: verify `KAFKA_BROKER` and Kafka healthcheck.
- Dashboard empty: ensure `INFERENCE_API_URL` resolves and API `/metrics` returns JSON.
- Spark not consuming: verify topic name and bootstrap server (`kafka:29092` in containers).
- Inference falls back to heuristic model: ensure `experiments/artifacts/trained_model.json` exists or MLflow URI resolves.
- DL backend not active: ensure `DL_MODEL_PATH` and `DL_METADATA_PATH` point to generated artifacts.
