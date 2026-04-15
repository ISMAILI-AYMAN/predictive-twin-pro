# Project Status - Predictive Digital Twin

## Pipeline
1. **Data Generation (Simulator)**
   - Simulate high-frequency industrial sensor signals (vibration, temperature, pressure).
   - Inject controlled anomalies/faults to emulate asset degradation and failure patterns.

2. **Streaming Ingestion (Kafka)**
   - Publish simulator events to Kafka topic `industrial_sensors`.
   - Ensure schema consistency and timestamp integrity for downstream processing.

3. **Stream Processing (Spark Structured Streaming)**
   - Consume Kafka stream and compute rolling features (mean, std, trend windows).
   - Prepare model-ready features and route outputs to inference + monitoring channels.

4. **Model Inference (LSTM Autoencoder / RUL)**
   - Run online anomaly scoring from reconstruction error.
   - Estimate Remaining Useful Life (RUL) from learned temporal degradation patterns.

5. **Drift Detection**
   - Compare live feature distributions against baseline training distributions.
   - Trigger statistical drift alerts (e.g., K-S based) every configured sample window.

6. **Monitoring & Visualization**
   - Expose operational metrics (stream health, anomaly score, drift alerts, RUL).
   - Display system status in dashboard layer (Grafana/Streamlit).


## Tasks Done
- MVP scope frozen with concrete done criteria in `README.md`.
- SimPy generator implemented with configurable degradation and fault injection.
- Kafka producer implemented with configurable broker/topic and delivery retries.
- Spark streaming consumer implemented with rolling feature engineering (`mean/std` windows).
- Inference API implemented (`/health`, `/infer`, `/metrics`) with anomaly + RUL scoring.
- Drift detector implemented with severity levels and retraining trigger policy.
- Streamlit dashboard implemented for system/model/asset health metrics.
- Docker setup updated for Kafka, Spark, inference API, and dashboard services.
- CI workflow added for lint + test (`.github/workflows/ci.yml`).
- Unit tests added for simulator, drift logic, and inference API contract.
- Delivery docs added: `docs/architecture.md`, `docs/runbook.md`, `docs/retraining_policy.md`, `docs/demo_script.md`.
- Environment template and smoke test added (`.env.example`, `scripts/smoke_test.py`).
- Runtime validation completed on local stack:
  - clean reproducibility pass completed (`docker compose down -v` then `up --build`)
  - producer shutdown now clean (`producer_shutdown_ok`)
  - fault injection stream verified (`fault=1` event path)
  - smoke test verified (`/health`, `/infer`, `/metrics`)
  - dashboard metrics verified with drift simulation (`critical_total` increments observed)
- Trained model workflow added:
  - training artifact pipeline in `experiments/train_model.py`
  - runtime trained-model loading with heuristic fallback in `ml_service/inference_api.py`
  - DL artifacts added (`dl_model.pt` + `dl_model_metadata.json`) from LSTM autoencoder training
- MLflow wiring added:
  - run logging in training workflow
  - runtime artifact resolution via `ml_service/model_registry.py`
- CI integration stage added in `.github/workflows/ci.yml` with deterministic `scripts/integration_ci_check.py`.
- Whitepaper/report artifact added in `docs/whitepaper.md`.


## Tasks Left
- Capture and publish final demo video using `docs/demo_script.md`.
- Validate MLflow-backed model registration flow on a live tracking backend.

## Definition of Done (MVP)
- End-to-end architecture and runnable services are documented and executable.
- Fault injection impacts anomaly/RUL and can be demonstrated.
- Drift severity (`info`, `warn`, `critical`) is exposed with retraining policy.
- Dashboard renders live metrics from inference service.
- CI runs lint and tests successfully.

## Latest Validation Snapshot
- Dashboard requests observed: `61`
- Latest anomaly score observed: `0.7667`
- Latest RUL observed: `900.00`
- Drift alert counters observed: `warn_total=0`, `critical_total=12`
