# Demo Script

## Narrative (5-7 minutes)
1. Introduce the industrial asset failure prediction problem.
2. Show architecture and pipeline from simulator to dashboard.
3. Run normal stream and observe low anomaly / no drift.
4. Inject fault and show anomaly spike + RUL degradation.
5. Introduce drifted vibration pattern and show drift alert severity.
6. Explain retraining policy and CI/test quality gates.
7. Conclude with portfolio takeaways and extension ideas.

## Live commands
- `docker compose up --build`
- `venv\Scripts\python.exe data_simulator/kafka_producer.py --duration 120`
- `venv\Scripts\python.exe data_simulator/kafka_producer.py --duration 120 --inject-fault`
- `venv\Scripts\python.exe scripts/smoke_test.py`
- `venv\Scripts\python.exe -c "import time,requests; url='http://localhost:8000/infer'; [requests.post(url,json={'timestamp':time.time(),'asset_id':'DEMO_DRIFT','temp':85.0,'vibration':0.09,'pressure':94.0,'health_index':90.0,'fault_active':1.0},timeout=5) for _ in range(60)]"`
