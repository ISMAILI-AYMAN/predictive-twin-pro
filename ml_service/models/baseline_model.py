from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineModel:
    baseline_temp: float = 75.0
    baseline_vibration: float = 0.02
    baseline_pressure: float = 100.0
    max_expected_wear_hours: float = 1000.0

    def anomaly_score(self, temp: float, vibration: float, pressure: float) -> float:
        temp_delta = abs(temp - self.baseline_temp) / 20.0
        vib_delta = abs(vibration - self.baseline_vibration) / 0.05
        pressure_delta = abs(pressure - self.baseline_pressure) / 15.0
        return round((temp_delta + vib_delta + pressure_delta) / 3.0, 4)

    def estimate_rul(self, health_index: float) -> float:
        bounded_health = max(0.0, min(100.0, health_index))
        return round((bounded_health / 100.0) * self.max_expected_wear_hours, 2)
