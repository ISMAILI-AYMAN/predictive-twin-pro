import simpy

from data_simulator.generator import IndustrialAsset


def test_fault_injection_increases_vibration() -> None:
    env = simpy.Environment()
    asset = IndustrialAsset(env, "A1", random_seed=42)
    normal = asset.calculate_metrics()
    asset.inject_fault()
    faulty = asset.calculate_metrics()
    assert faulty["vibration"] > normal["vibration"]


def test_health_degrades_after_threshold() -> None:
    env = simpy.Environment()
    asset = IndustrialAsset(env, "A1", degradation_start=1, degradation_rate=5.0)
    simulation = asset.run_simulation()
    processed = 0
    while processed < 8:
        item = next(simulation)
        if isinstance(item, dict):
            processed += 1
        else:
            env.run(until=item)
    assert asset.health < 100.0
