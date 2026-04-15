from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, from_json, stddev
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.sql.window import Window

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "industrial_sensors")
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoint")

SENSOR_SCHEMA = StructType(
    [
        StructField("timestamp", DoubleType()),
        StructField("asset_id", StringType()),
        StructField("temp", DoubleType()),
        StructField("vibration", DoubleType()),
        StructField("pressure", DoubleType()),
        StructField("health_index", DoubleType()),
        StructField("fault_active", DoubleType()),
    ]
)


def main() -> None:
    spark = (
        SparkSession.builder.appName("PredictiveTwinStreaming")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        raw_stream.selectExpr("CAST(value AS STRING)")
        .select(from_json(col("value"), SENSOR_SCHEMA).alias("event"))
        .select("event.*")
    )

    window_spec = Window.partitionBy("asset_id").orderBy("timestamp").rowsBetween(-20, 0)
    featured = (
        parsed.withColumn("temp_mean_20", avg("temp").over(window_spec))
        .withColumn("vibration_mean_20", avg("vibration").over(window_spec))
        .withColumn("vibration_std_20", stddev("vibration").over(window_spec))
        .fillna(0.0, subset=["vibration_std_20"])
    )

    query = (
        featured.writeStream.outputMode("append")
        .format("console")
        .option("truncate", False)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .start()
    )

    print(f"spark_streaming_started topic={KAFKA_TOPIC} bootstrap={KAFKA_BOOTSTRAP}")
    query.awaitTermination()


if __name__ == "__main__":
    main()