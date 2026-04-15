from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Dict


@dataclass
class DriftResult:
    drift_detected: bool
    severity: str
    score: float


class DriftDetector:
    def __init__(self, baseline_vibration_mean: float = 0.02, window_size: int = 50) -> None:
        self.baseline_vibration_mean = baseline_vibration_mean
        self.window_size = window_size
        self.samples: Deque[float] = deque(maxlen=window_size)

    def update(self, vibration: float) -> DriftResult:
        self.samples.append(vibration)
        if len(self.samples) < self.window_size:
            return DriftResult(False, "info", 0.0)

        live_mean = mean(self.samples)
        score = abs(live_mean - self.baseline_vibration_mean)
        severity = "info"
        if score >= 0.03:
            severity = "critical"
        elif score >= 0.015:
            severity = "warn"

        return DriftResult(drift_detected=severity != "info", severity=severity, score=round(score, 5))

    def retraining_policy(self, drift_result: DriftResult) -> Dict[str, object]:
        return {
            "trigger_retraining": drift_result.severity == "critical",
            "required_window_size": 500 if drift_result.severity == "critical" else 1000,
            "validation_checks": ["mae_below_threshold", "false_positive_rate_check"],
        }
