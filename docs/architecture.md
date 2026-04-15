# Predictive Twin Pro Architecture

## Components
- `data_simulator`: SimPy-based industrial asset signal generation + optional fault injection.
- `streaming`: Spark Structured Streaming consumer with rolling feature engineering.
- `ml_service`: FastAPI inference endpoint for anomaly score, RUL, drift, and retraining advice.
- `dashboard`: Streamlit dashboard that renders runtime health metrics.
- `docker-compose.yml`: local multi-service orchestration.

## Data Flow
1. Simulator emits JSON sensor events.
2. Kafka ingests events (`industrial_sensors` topic).
3. Spark consumes events and computes rolling means/std.
4. Inference API scores events and updates drift/health metrics.
5. Dashboard pulls `/metrics` and shows system/model/asset state.
