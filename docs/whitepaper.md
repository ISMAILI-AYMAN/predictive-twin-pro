# Predictive Twin Pro Whitepaper

## Abstract
Predictive Twin Pro is an Industrial 4.0 predictive-maintenance prototype that combines synthetic telemetry generation, stream transport, online inference, drift detection, and dashboard monitoring. The system demonstrates how model-aware observability can be used to detect degradation and trigger retraining decisions.

## Problem Statement
Industrial assets fail progressively, but static threshold monitoring often misses early warning signals. We need an online architecture that can:
- ingest continuous telemetry,
- estimate anomaly and residual life,
- detect distribution drift,
- and expose operations-ready metrics.

## Methodology
1. **Synthetic data simulation** using configurable degradation and fault injection.
2. **Streaming ingestion** through Kafka topic `industrial_sensors`.
3. **Feature engineering** via Spark Structured Streaming rolling windows.
4. **Inference** through FastAPI with trained-stat model backend (heuristic fallback).
5. **Drift detection** on vibration windows with severity mapping (`info`, `warn`, `critical`).
6. **Observability** through `/metrics` and Streamlit dashboard.

## Experiments and Validation
- Clean reproducibility run: `docker compose down -v` then `docker compose up --build`.
- Fault-injection stream verified (`fault=1` path).
- Smoke API contract validated (`/health`, `/infer`, `/metrics`).
- Dashboard observed runtime evidence:
  - requests: `61`
  - anomaly score: `0.7667`
  - RUL: `900.00`
  - critical drift count: `12`

## Limitations
- Current trained model is a lightweight statistical artifact and not yet a deep sequence model.
- CI integration check validates service-level behavior, but not full Spark-to-API materialization semantics.
- No automated retraining scheduler is deployed yet.

## Future Work
- Replace statistical artifact with LSTM/autoencoder model package.
- Automate model promotion with stricter MLflow governance.
- Add production-grade alerting backend and long-term metric retention.
