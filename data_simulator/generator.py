from __future__ import annotations

import argparse
import random
import time
from typing import Dict, Iterator, Optional, Union

import simpy
from simpy.events import Event


class IndustrialAsset:
    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        *,
        degradation_start: int = 500,
        degradation_rate: float = 0.1,
        random_seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.name = name
        self.health = 100.0
        self.is_running = True
        self.degradation_start = degradation_start
        self.degradation_rate = degradation_rate
        self.fault_active = False
        self.fault_intensity = 1.0
        self._rng = random.Random(random_seed)

    def inject_fault(self, intensity: float = 2.0) -> None:
        self.fault_active = True
        self.fault_intensity = max(1.0, intensity)

    def clear_fault(self) -> None:
        self.fault_active = False
        self.fault_intensity = 1.0

    def calculate_metrics(self) -> Dict[str, Union[str, float]]:
        effective_wear = 100 - self.health
        temp = 75 + effective_wear * 0.5 + self._rng.uniform(-1, 1)
        vibration = 0.02 + effective_wear * 0.005 + self._rng.uniform(-0.005, 0.005)
        pressure = 100 + self._rng.uniform(-2, 2)

        if self.fault_active:
            temp += 8.0 * self.fault_intensity
            vibration += 0.03 * self.fault_intensity
            pressure -= 4.0 * self.fault_intensity

        return {
            "timestamp": time.time(),
            "asset_id": self.name,
            "temp": round(temp, 2),
            "vibration": round(vibration, 4),
            "pressure": round(pressure, 2),
            "health_index": round(self.health, 2),
            "fault_active": float(1 if self.fault_active else 0),
        }

    def run_simulation(self) -> Iterator[Union[Dict[str, Union[str, float]], Event]]:
        while self.is_running:
            if self.env.now > self.degradation_start:
                self.health -= self.degradation_rate

            if self.health <= 0:
                self.is_running = False
                break

            yield self.calculate_metrics()
            yield self.env.timeout(1)


def _local_test_runner(
    env: simpy.Environment, asset: IndustrialAsset, max_events: int
) -> Iterator[Event]:
    simulation = asset.run_simulation()
    count = 0
    for item in simulation:
        if isinstance(item, dict):
            print(item)
            count += 1
            if count >= max_events:
                break
        else:
            yield item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial asset simulator")
    parser.add_argument("--asset-id", default="CNC_Router_01")
    parser.add_argument("--events", type=int, default=20)
    parser.add_argument("--inject-fault", action="store_true")
    args = parser.parse_args()

    env = simpy.Environment()
    machine = IndustrialAsset(env, args.asset_id)
    if args.inject_fault:
        machine.inject_fault()
    env.process(_local_test_runner(env, machine, args.events))
    env.run(until=args.events + 5)