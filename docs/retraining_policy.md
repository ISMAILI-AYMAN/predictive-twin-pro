# Drift and Retraining Policy

## Drift scoring
- Drift signal is computed from live vibration mean delta versus baseline mean.
- Severity thresholds:
  - `info`: score < 0.015
  - `warn`: 0.015 <= score < 0.03
  - `critical`: score >= 0.03

## Retraining trigger
- Retraining is triggered only on `critical` drift.
- Required minimum window:
  - critical: 500 recent events
  - non-critical: 1000 event observation before reconsideration

## Promotion checks
- Candidate model MAE must beat current baseline.
- False positive alert rate must remain within agreed threshold.
- API contract and smoke tests must pass before promotion.
