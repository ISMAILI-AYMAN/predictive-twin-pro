from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterator, Optional, Union

import simpy
from kafka import KafkaProducer
from simpy.events import Event

from generator import IndustrialAsset

KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "industrial_sensors")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")


def json_serializer(data: Dict[str, Union[str, float]]) -> bytes:
    return json.dumps(data).encode("utf-8")


def create_producer() -> Optional[KafkaProducer]:
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=json_serializer,
            acks="all",
            retries=3,
            linger_ms=0,
            max_block_ms=10000,
            request_timeout_ms=15000,
        )
        print(f"connected_to_kafka broker={KAFKA_BROKER} topic={KAFKA_TOPIC}")
        return producer
    except Exception as exc:
        print(f"kafka_connection_failed error={exc}")
        return None


def stream_to_kafka(
    env: simpy.Environment,
    machine: IndustrialAsset,
    producer: Optional[KafkaProducer],
) -> Iterator[Event]:
    simulation = machine.run_simulation()
    for item in simulation:
        if isinstance(item, dict):
            if producer:
                producer.send(KAFKA_TOPIC, item)
                print(
                    "event_sent "
                    f"asset={item['asset_id']} "
                    f"health={item['health_index']} "
                    f"fault={int(item['fault_active'])}"
                )
            else:
                print(f"offline_event {item}")
        else:
            yield item


def shutdown_producer(producer: Optional[KafkaProducer]) -> None:
    if producer is None:
        return
    try:
        producer.flush(timeout=15)
        producer.close(timeout=15)
        print("producer_shutdown_ok")
    except Exception as exc:
        print(f"producer_shutdown_warning error={exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish simulator events to Kafka")
    parser.add_argument("--asset-id", default="CNC_Router_01")
    parser.add_argument("--duration", type=int, default=1000)
    parser.add_argument("--inject-fault", action="store_true")
    args = parser.parse_args()

    env = simpy.Environment()
    machine = IndustrialAsset(env, args.asset_id)
    if args.inject_fault:
        machine.inject_fault()

    producer = create_producer()
    try:
        env.process(stream_to_kafka(env, machine, producer))
        env.run(until=args.duration)
    finally:
        shutdown_producer(producer)