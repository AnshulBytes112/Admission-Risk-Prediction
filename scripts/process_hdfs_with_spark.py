#!/usr/bin/env python3
"""Process Phase 3 Kafka JSONL files in HDFS and write clean Parquet features."""
import argparse
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window

EVENT_SCHEMA = T.StructType([
    T.StructField("event_id", T.StringType(), True),
    T.StructField("patient_id", T.StringType(), True),
    T.StructField("observation_timestamp", T.StringType(), True),
    T.StructField("age", T.DoubleType(), True),
    T.StructField("heart_rate", T.DoubleType(), True),
    T.StructField("systolic_bp", T.DoubleType(), True),
    T.StructField("diastolic_bp", T.DoubleType(), True),
    T.StructField("spo2", T.DoubleType(), True),
    T.StructField("respiratory_rate", T.DoubleType(), True),
    T.StructField("temperature_c", T.DoubleType(), True),
    T.StructField("previous_admissions", T.DoubleType(), True),
    T.StructField("chronic_conditions_count", T.DoubleType(), True),
    T.StructField("lactate", T.DoubleType(), True),
    T.StructField("admission_within_24h", T.DoubleType(), True),
    T.StructField("synthetic_data", T.BooleanType(), True),
    T.StructField("_stream", T.StructType([
        T.StructField("event_key", T.StringType(), True),
        T.StructField("published_at_utc", T.StringType(), True),
        T.StructField("schema_version", T.StringType(), True),
        T.StructField("data_origin", T.StringType(), True),
    ]), True),
])

BOUNDS = {
    "age": (18, 100),
    "heart_rate": (25, 240),
    "systolic_bp": (40, 300),
    "diastolic_bp": (20, 200),
    "spo2": (50, 100),
    "respiratory_rate": (3, 80),
    "temperature_c": (30, 44),
    "previous_admissions": (0, 100),
    "chronic_conditions_count": (0, 50),
    "lactate": (0, 30),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="hdfs://namenode:9000/emergency-admission-bda/raw/kafka-events")
    parser.add_argument("--output", default="hdfs://namenode:9000/emergency-admission-bda/processed/patient-features")
    parser.add_argument("--quarantine", default="hdfs://namenode:9000/emergency-admission-bda/processed/quarantine")
    parser.add_argument("--master", default="local[*]", help="Use local[*] for the Compose Spark container.")
    args = parser.parse_args()

    spark = (SparkSession.builder
             .appName("EmergencyAdmissionBDA-FeatureEngineering")
             .master(args.master)
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        # JSONL lines written by Phase 2 have an outer envelope with event nested inside.
        raw = spark.read.json(args.input)
        if "event" not in raw.columns:
            raise ValueError("Input does not contain an 'event' field. Check Phase 2 JSONL output.")

        events = raw.select(
            F.col("event.*"),
            F.col("kafka_topic"),
            F.col("kafka_partition"),
            F.col("kafka_offset"),
            F.col("consumed_at_utc")
        )

        # Convert typed fields robustly and parse timestamps.
        for col_name in BOUNDS:
            if col_name in events.columns:
                events = events.withColumn(col_name, F.col(col_name).cast("double"))
        events = events.withColumn(
            "observation_ts",
            F.to_timestamp("observation_timestamp")
        )

        # Keep malformed/missing identifiers out of the model feature set.
        invalid = events.filter(
            F.col("patient_id").isNull()
            | (F.trim(F.col("patient_id")) == "")
            | F.col("event_id").isNull()
            | F.col("observation_ts").isNull()
        )
        if invalid.limit(1).count() > 0:
            (invalid.withColumn("quarantine_reason", F.lit("missing_identifier_or_invalid_timestamp"))
             .write.mode("append").json(args.quarantine))

        valid = events.filter(
            F.col("patient_id").isNotNull()
            & (F.trim(F.col("patient_id")) != "")
            & F.col("event_id").isNotNull()
            & F.col("observation_ts").isNotNull()
        )

        # Remove duplicate event IDs, keeping the first by Kafka offset where possible.
        order_cols = []
        if "kafka_partition" in valid.columns:
            order_cols.append(F.col("kafka_partition").asc_nulls_last())
        if "kafka_offset" in valid.columns:
            order_cols.append(F.col("kafka_offset").asc_nulls_last())
        if not order_cols:
            order_cols = [F.col("observation_ts").asc()]
        window = Window.partitionBy("event_id").orderBy(*order_cols)
        dedup = (valid.withColumn("_duplicate_rank", F.row_number().over(window))
                 .filter(F.col("_duplicate_rank") == 1)
                 .drop("_duplicate_rank"))

        # Values outside plausible data-quality bounds are nulled rather than silently clipped.
        for col_name, (low, high) in BOUNDS.items():
            if col_name in dedup.columns:
                dedup = dedup.withColumn(
                    col_name,
                    F.when(F.col(col_name).between(low, high), F.col(col_name)).otherwise(F.lit(None).cast("double"))
                )

        # Simple median imputation for selected numeric fields, computed on this input batch.
        # In Phase 5, imputation parameters will be fitted only on training data to prevent leakage.
        impute_cols = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2",
                       "respiratory_rate", "temperature_c", "lactate"]
        medians = dedup.approxQuantile(impute_cols, [0.5], 0.01)
        for idx, col_name in enumerate(impute_cols):
            median_value = medians[idx][0] if medians[idx] else None
            if median_value is not None:
                dedup = dedup.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(float(median_value))))

        # Add time features and simple patient-level history aggregates.
        featured = (dedup
            .withColumn("observation_date", F.to_date("observation_ts"))
            .withColumn("observation_hour", F.hour("observation_ts"))
            .withColumn("is_synthetic", F.coalesce(F.col("synthetic_data"), F.lit(True)))
            .withColumn("data_origin", F.coalesce(F.col("_stream.data_origin"), F.lit("synthetic")))
        )

        # Keep one row per event, with previous/current observed history available as columns.
        # This does not create a clinical prediction; it prepares a modelling table.
        patient_window = Window.partitionBy("patient_id").orderBy("observation_ts")
        featured = featured.withColumn(
            "prior_observation_count",
            F.row_number().over(patient_window) - F.lit(1)
        )

        output_cols = [
            "event_id", "patient_id", "observation_ts", "observation_date", "observation_hour",
            "age", "heart_rate", "systolic_bp", "diastolic_bp", "spo2", "respiratory_rate",
            "temperature_c", "previous_admissions", "chronic_conditions_count", "lactate",
            "admission_within_24h", "is_synthetic", "data_origin", "prior_observation_count",
            "kafka_topic", "kafka_partition", "kafka_offset"
        ]
        output_cols = [c for c in output_cols if c in featured.columns]
        final_df = featured.select(*output_cols)

        final_df.write.mode("overwrite").partitionBy("observation_date").parquet(args.output)

        print("=== Spark processing summary ===")
        print(f"Raw events: {raw.count():,}")
        print(f"Valid deduplicated events written: {final_df.count():,}")
        print(f"Output: {args.output}")
        print(f"Quarantine path (if needed): {args.quarantine}")
        print("Output format: Parquet partitioned by observation_date")
        print("Reminder: source data is synthetic and not for clinical use.")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
