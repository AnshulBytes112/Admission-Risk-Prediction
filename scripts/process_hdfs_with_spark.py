#!/usr/bin/env python3
"""Phase 4: Apache Spark Distributed Data Processing & Feature Engineering.

Reads raw Kafka JSONL event archives from HDFS, performs data quality validation,
bounds enforcement, deduplication, missing-value imputation, feature engineering,
invalid-record quarantine, and writes partitioned Parquet feature tables to HDFS.

Data Lake Layout:
- Input:      hdfs://namenode:9000/emergency-admission-bda/raw/kafka-events
- Output:     hdfs://namenode:9000/emergency-admission-bda/processed/patient-features
- Quarantine: hdfs://namenode:9000/emergency-admission-bda/quarantine/spark-processing
- Manifest:   hdfs://namenode:9000/emergency-admission-bda/metadata/spark-processing

Disclaimer: All data processed by this script is strictly synthetic for academic BDA demonstration.
It is not clinically validated and must not be used for medical decisions.
"""
import argparse
import datetime
import json
import logging
import os
import sys
import uuid
from typing import Any

try:
    from pyspark.sql import SparkSession, functions as F, types as T
    from pyspark.sql.window import Window
    HAS_PYSPARK = True
except ImportError:
    SparkSession = None
    F = None
    T = None
    Window = None
    HAS_PYSPARK = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("spark_feature_engineering")

# -----------------------------------------------------------------------------
# 1. Schemas & Data Quality Bounds
# -----------------------------------------------------------------------------

if HAS_PYSPARK:
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

    ENVELOPE_SCHEMA = T.StructType([
        T.StructField("kafka_topic", T.StringType(), True),
        T.StructField("kafka_partition", T.IntegerType(), True),
        T.StructField("kafka_offset", T.LongType(), True),
        T.StructField("kafka_key", T.StringType(), True),
        T.StructField("consumed_at_utc", T.StringType(), True),
        T.StructField("event", EVENT_SCHEMA, True),
        T.StructField("metadata", T.MapType(T.StringType(), T.StringType()), True),
        T.StructField("_corrupt_record", T.StringType(), True),
    ])
else:
    EVENT_SCHEMA = None
    ENVELOPE_SCHEMA = None

BOUNDS = {
    "age": (0.0, 120.0),
    "heart_rate": (25.0, 240.0),
    "systolic_bp": (40.0, 300.0),
    "diastolic_bp": (20.0, 200.0),
    "spo2": (50.0, 100.0),
    "respiratory_rate": (3.0, 80.0),
    "temperature_c": (30.0, 45.0),
    "previous_admissions": (0.0, 100.0),
    "chronic_conditions_count": (0.0, 50.0),
    "lactate": (0.0, 30.0),
}

NUMERIC_IMPUTE_COLS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature_c",
    "lactate",
]

OUTPUT_COLUMNS = [
    "event_id",
    "patient_id",
    "observation_ts",
    "observation_date",
    "observation_hour",
    "age",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "respiratory_rate",
    "temperature_c",
    "previous_admissions",
    "chronic_conditions_count",
    "lactate",
    "admission_within_24h",
    "is_synthetic",
    "data_origin",
    "prior_observation_count",
    "kafka_topic",
    "kafka_partition",
    "kafka_offset",
]


# -----------------------------------------------------------------------------
# 2. Main Spark Processing Workflow
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.getenv("HDFS_RAW_URI", "hdfs://namenode:9000/emergency-admission-bda/raw/kafka-events"),
        help="HDFS input URI for raw Kafka JSONL events.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("HDFS_PROCESSED_URI", "hdfs://namenode:9000/emergency-admission-bda/processed/patient-features"),
        help="HDFS output URI for partitioned Parquet features.",
    )
    parser.add_argument(
        "--quarantine",
        default=os.getenv("HDFS_QUARANTINE_URI", "hdfs://namenode:9000/emergency-admission-bda/quarantine/spark-processing"),
        help="HDFS output URI for quarantined records.",
    )
    parser.add_argument(
        "--manifest",
        default=os.getenv("HDFS_MANIFEST_URI", "hdfs://namenode:9000/emergency-admission-bda/metadata/spark-processing"),
        help="HDFS output URI for processing manifests.",
    )
    parser.add_argument(
        "--master",
        default=os.getenv("SPARK_MASTER_URL", "local[*]"),
        help="Spark master URL (default: local[*]).",
    )
    parser.add_argument(
        "--write-mode",
        default=os.getenv("WRITE_MODE", "overwrite"),
        choices=["overwrite", "append"],
        help="Parquet output write mode.",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "2")),
        help="Number of Spark shuffle partitions.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("SPARK_LOG_LEVEL", "WARN"),
        help="Spark log level (INFO, WARN, ERROR).",
    )
    return parser.parse_args()


def run_pipeline(spark: SparkSession, args: argparse.Namespace) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    logger.info("Starting Phase 4 Spark Processing Pipeline [run_id=%s]...", run_id)
    logger.info("Reading raw JSONL from: %s", args.input)

    # 1. Read Raw JSONL with Schema & Corrupt Record Capture
    raw_df = (
        spark.read
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .schema(ENVELOPE_SCHEMA)
        .json(args.input)
    )

    total_raw_count = raw_df.count()
    logger.info("Total raw records read: %d", total_raw_count)
    if total_raw_count == 0:
        raise ValueError(f"No records found in HDFS input path '{args.input}'. Ingest data first in Phase 3.")

    # 2. Extract Fields and Classify Malformed / Invalid Records
    extracted_df = raw_df.select(
        F.col("event.*"),
        F.col("kafka_topic"),
        F.col("kafka_partition"),
        F.col("kafka_offset"),
        F.col("consumed_at_utc"),
        F.col("_corrupt_record"),
    )

    # Parse Observation Timestamp
    extracted_df = extracted_df.withColumn(
        "observation_ts",
        F.to_timestamp(F.col("observation_timestamp")),
    )

    # Casting numeric fields explicitly
    for col_name in BOUNDS:
        if col_name in extracted_df.columns:
            extracted_df = extracted_df.withColumn(col_name, F.col(col_name).cast(T.DoubleType()))

    extracted_df = extracted_df.withColumn("admission_within_24h", F.col("admission_within_24h").cast(T.IntegerType()))

    # Categorize Quarantine Reasons
    quarantine_condition = (
        F.col("_corrupt_record").isNotNull()
        | F.col("patient_id").isNull()
        | (F.trim(F.col("patient_id")) == "")
        | F.col("event_id").isNull()
        | (F.trim(F.col("event_id")) == "")
        | F.col("observation_ts").isNull()
        | F.col("admission_within_24h").isNull()
        | (~F.col("admission_within_24h").isin([0, 1]))
    )

    quarantined_df = extracted_df.filter(quarantine_condition).withColumn(
        "error_reason",
        F.when(F.col("_corrupt_record").isNotNull(), F.lit("malformed_json_syntax"))
        .when(F.col("patient_id").isNull() | (F.trim(F.col("patient_id")) == ""), F.lit("missing_patient_id"))
        .when(F.col("event_id").isNull() | (F.trim(F.col("event_id")) == ""), F.lit("missing_event_id"))
        .when(F.col("observation_ts").isNull(), F.lit("invalid_or_unparseable_timestamp"))
        .when(F.col("admission_within_24h").isNull() | (~F.col("admission_within_24h").isin([0, 1])), F.lit("missing_or_non_binary_target"))
        .otherwise(F.lit("unspecified_schema_validation_failure"))
    ).withColumn("error_category", F.lit("SCHEMA_VIOLATION")) \
     .withColumn("run_id", F.lit(run_id)) \
     .withColumn("processing_timestamp_utc", F.lit(now_utc))

    quarantined_count = quarantined_df.count()

    if quarantined_count > 0:
        logger.warning("Quarantining %d malformed/invalid records to %s...", quarantined_count, args.quarantine)
        quarantine_dest = f"{args.quarantine}/run_date={run_date}/run_id={run_id}"
        quarantined_df.write.mode("overwrite").json(quarantine_dest)
        logger.info("Quarantined records written to: %s", quarantine_dest)

    # 3. Filter Valid Records for Modeling
    valid_df = extracted_df.filter(~quarantine_condition)
    valid_count = valid_df.count()
    logger.info("Valid records matching core schema contract: %d", valid_count)

    # 4. Deduplicate on event_id with Deterministic Tie-Breaker
    # Tie-breaker: kafka_partition ASC, kafka_offset ASC, observation_ts ASC
    dedup_window = Window.partitionBy("event_id").orderBy(
        F.col("kafka_partition").asc_nulls_last(),
        F.col("kafka_offset").asc_nulls_last(),
        F.col("observation_ts").asc_nulls_last(),
    )

    deduped_df = (
        valid_df
        .withColumn("_dup_rank", F.row_number().over(dedup_window))
        .filter(F.col("_dup_rank") == 1)
        .drop("_dup_rank")
    )

    deduped_count = deduped_df.count()
    duplicate_count = valid_count - deduped_count
    logger.info("Deduplication complete: %d unique events (%d duplicates removed)", deduped_count, duplicate_count)

    # 5. Data Quality Bounds: Convert out-of-bounds measurements to NULL for imputation
    bounded_df = deduped_df
    for col_name, (low, high) in BOUNDS.items():
        if col_name in bounded_df.columns:
            bounded_df = bounded_df.withColumn(
                col_name,
                F.when(F.col(col_name).between(low, high), F.col(col_name)).otherwise(F.lit(None).cast(T.DoubleType())),
            )

    # 6. Missing-Value Handling: Batch Median Imputation
    imputation_summary: dict[str, dict[str, Any]] = {}
    null_counts = bounded_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in NUMERIC_IMPUTE_COLS]).collect()[0]

    # Calculate median quantiles
    quantiles = bounded_df.approxQuantile(NUMERIC_IMPUTE_COLS, [0.5], 0.01)

    imputed_df = bounded_df
    for idx, col_name in enumerate(NUMERIC_IMPUTE_COLS):
        null_count = null_counts[col_name]
        median_val = quantiles[idx][0] if quantiles[idx] else None
        if median_val is not None and null_count > 0:
            imputed_df = imputed_df.withColumn(
                col_name,
                F.coalesce(F.col(col_name), F.lit(float(median_val))),
            )
            logger.info("Imputed %d nulls in '%s' with median value %.2f", null_count, col_name, median_val)
        imputation_summary[col_name] = {
            "pre_impute_null_count": int(null_count),
            "median_applied": float(median_val) if median_val is not None else None,
        }

    # 7. Feature Engineering
    # - observation_date: Date derived from timestamp for partitioning
    # - observation_hour: Hour of observation (0-23)
    # - prior_observation_count: Strict historical observation count for patient, excluding current
    patient_window = Window.partitionBy("patient_id").orderBy(
        F.col("observation_ts").asc(),
        F.col("event_id").asc(),
    )

    featured_df = (
        imputed_df
        .withColumn("observation_date", F.to_date(F.col("observation_ts")))
        .withColumn("observation_hour", F.hour(F.col("observation_ts")).cast(T.IntegerType()))
        .withColumn("prior_observation_count", (F.row_number().over(patient_window) - F.lit(1)).cast(T.IntegerType()))
        .withColumn("is_synthetic", F.coalesce(F.col("synthetic_data"), F.lit(True)))
        .withColumn("data_origin", F.coalesce(F.col("_stream.data_origin"), F.lit("synthetic")))
    )

    final_df = featured_df.select([c for c in OUTPUT_COLUMNS if c in featured_df.columns])
    output_record_count = final_df.count()

    # 8. Safe Staged Output Publication
    staging_path = f"{args.output}_staging_{run_id}"
    logger.info("Writing staged Parquet dataset to: %s...", staging_path)

    (
        final_df.write
        .mode("overwrite")
        .partitionBy("observation_date")
        .parquet(staging_path)
    )

    # Verify staging output
    staged_read = spark.read.parquet(staging_path)
    staged_count = staged_read.count()
    if staged_count != output_record_count:
        raise RuntimeError(
            f"Staging verification failed: staged count ({staged_count}) does not match expected ({output_record_count})"
        )

    # Final publication
    logger.info("Publishing finalized Parquet dataset to: %s...", args.output)
    (
        final_df.write
        .mode(args.write_mode)
        .partitionBy("observation_date")
        .parquet(args.output)
    )

    # Optional local export for immediate Phase 5 modeling access
    if os.path.exists("/project-data"):
        try:
            local_features_path = "/project-data/processed/features.parquet"
            final_df.write.mode(args.write_mode).partitionBy("observation_date").parquet(local_features_path)
            logger.info("Exported local copy for Phase 5 modeling to: %s", local_features_path)
        except Exception as exc:
            logger.warning("Could not export local copy to /project-data/processed: %s", exc)

    # 9. Machine-Readable Processing Manifest
    manifest_data = {
        "manifest_version": "1.0",
        "project": "emergency-admission-risk-prediction",
        "data_origin": "synthetic",
        "run_id": run_id,
        "processing_timestamp_utc": now_utc,
        "spark_version": spark.version,
        "input_path": args.input,
        "output_path": args.output,
        "raw_record_count": total_raw_count,
        "valid_record_count": valid_count,
        "quarantined_record_count": quarantined_count,
        "duplicate_count": duplicate_count,
        "output_record_count": output_record_count,
        "imputation_summary": imputation_summary,
        "output_format": "parquet",
        "partition_column": "observation_date",
        "status": "SUCCESS",
    }

    # Save manifest to HDFS
    manifest_hdfs_path = f"{args.manifest}/run_id={run_id}/processing_manifest.json"
    manifest_json_str = json.dumps(manifest_data, indent=2)
    manifest_rdd = spark.sparkContext.parallelize([manifest_json_str])
    manifest_rdd.saveAsTextFile(manifest_hdfs_path)
    logger.info("Manifest published to HDFS at: %s", manifest_hdfs_path)

    # Also save manifest locally in ./data volume for host accessibility
    local_manifest_dir = "/project-data/metadata/spark_manifests"
    try:
        os.makedirs(f"{local_manifest_dir}/run_id={run_id}", exist_ok=True)
        with open(f"{local_manifest_dir}/run_id={run_id}/processing_manifest.json", "w", encoding="utf-8") as mf:
            mf.write(manifest_json_str)
        logger.info("Local manifest copy saved to: %s/run_id=%s/processing_manifest.json", local_manifest_dir, run_id)
    except Exception as exc:
        logger.warning("Could not write local manifest copy: %s", exc)

    logger.info("=== Phase 4 Processing Summary ===")
    logger.info("  Raw records processed:    %d", total_raw_count)
    logger.info("  Valid schema records:     %d", valid_count)
    logger.info("  Duplicates removed:       %d", duplicate_count)
    logger.info("  Quarantined records:      %d", quarantined_count)
    logger.info("  Final Parquet rows:       %d", output_record_count)
    logger.info("  Output HDFS destination:  %s", args.output)
    logger.info("  Partition scheme:         observation_date=YYYY-MM-DD")
    logger.info("  Execution Status:         SUCCESS")

    return manifest_data


def main():
    args = parse_args()
    spark = (
        SparkSession.builder
        .appName(os.getenv("SPARK_APP_NAME", "EmergencyAdmissionBDA-FeatureEngineering"))
        .master(args.master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(args.log_level)

    try:
        run_pipeline(spark, args)
    except Exception as exc:
        logger.error("Spark processing job failed: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()
        logger.info("SparkSession stopped cleanly.")


if __name__ == "__main__":
    main()
