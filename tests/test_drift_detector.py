from ml_service.drift_detector import DriftDetector


def test_no_drift_on_baseline() -> None:
    detector = DriftDetector(baseline_vibration_mean=0.02, window_size=5)
    result = None
    for _ in range(5):
        result = detector.update(0.021)
    assert result is not None
    assert result.drift_detected is False


def test_critical_drift_on_shift() -> None:
    detector = DriftDetector(baseline_vibration_mean=0.02, window_size=5)
    result = None
    for _ in range(5):
        result = detector.update(0.08)
    assert result is not None
    assert result.severity == "critical"
