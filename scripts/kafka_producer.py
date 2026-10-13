"""Kafka Event Producer for Synthetic Emergency Observations.

Reads synthetic clinical observation CSV records and streams them to a Kafka topic:
- Validates CSV file existence and required schema columns.
- Preserves patient-grouped partition ordering using patient_id as the message key.
- Casts strings to appropriate types: floats, ints, booleans, and nulls.
- Envelopes events with provenance metadata (schema version, UTC timestamps, synthetic marker).
- Uses acks='all' and delivery acknowledgment verification.
- Supports rate-limiting (--delay-ms), batch limit (--limit), and repeat streaming (--repeat).
- Structured logging without exposing sensitive patient details.
"""
import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from kafka import KafkaProducer

try:
    from scripts.kafka_topic_manager import ensure_topic
except ModuleNotFoundError:
    from kafka_topic_manager import ensure_topic

logger = logging.getLogger("kafka_producer")

REQUIRED_COLUMNS = [
    "event_id",
    "patient_id",
    "observation_timestamp",
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
    "synthetic_data",
]


def parse_csv_value(raw: str) -> Any:
    """Parse a CSV string into its appropriate Python typed representation."""
    if raw is None or raw == "":
        return None
    val = raw.strip()
    lower = val.lower()
    if lower in ("nan", "null", "none", "na", ""):
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    # Attempt numeric parsing
    try:
        if any(c in val for c in ".eE"):
            return float(val)
        return int(val)
    except ValueError:
        return val


def create_event_payload(row: dict[str, Any], schema_version: str = "1.0") -> dict[str, Any]:
    """Transform a raw CSV row dictionary into a standardized Kafka event envelope."""
    parsed_event = {k: parse_csv_value(v) for k, v in row.items()}
    event_id = str(parsed_event.get("event_id") or uuid.uuid4())
    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    stream_meta = {
        "event_key": event_id,
        "published_at_utc": now_utc,
        "schema_version": schema_version,
        "data_origin": "synthetic",
    }
    # Embed stream metadata both inside event._stream (for PySpark) and at root metadata
    parsed_event["_stream"] = stream_meta

    return {
        "event": parsed_event,
        "metadata": stream_meta,
    }


def validate_csv_header(header: list[str]) -> None:
    """Verify that all required clinical schema columns exist in the CSV header."""
    missing = [col for col in REQUIRED_COLUMNS if col not in header]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}. Required columns: {REQUIRED_COLUMNS}")


class GracefulProducerStopper:
    def __init__(self):
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info("Termination signal (%s) received. Initiating graceful shutdown...", signum)
        self.stop_requested = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish synthetic observation records to Kafka")
    parser.add_argument(
        "--input",
        default=os.getenv("INPUT_CSV", "data/raw/patient_observations.csv"),
        help="Path to synthetic CSV file",
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_TOPIC", "patient-observations"),
        help="Target Kafka topic",
    )
    parser.add_argument(
        "--bootstrap",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap server address",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=int(os.getenv("KAFKA_PUBLISH_DELAY_MS", "25")),
        help="Delay in milliseconds between streamed events",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("STREAM_MAX_EVENTS", "0")),
        help="Maximum events to publish (0 for all rows)",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Repeat publishing CSV cyclically for continuous streaming demonstrations",
    )
    parser.add_argument(
        "--schema-version",
        default=os.getenv("STREAM_SCHEMA_VERSION", "1.0"),
        help="Schema version header",
    )
    args = parser.parse_args()

    # Configure structured logging
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    csv_path = Path(args.input)
    if not csv_path.exists():
        logger.error("Input CSV not found at '%s'. Generate synthetic data first.", csv_path)
        sys.exit(1)

    if args.delay_ms < 0:
        logger.error("--delay-ms must be non-negative, got %d", args.delay_ms)
        sys.exit(1)

    # 1. Ensure Kafka topic exists before attempting to produce
    try:
        partitions = int(os.getenv("KAFKA_PARTITIONS", "3"))
        rep_factor = int(os.getenv("KAFKA_REPLICATION_FACTOR", "1"))
        ensure_topic(args.bootstrap, args.topic, num_partitions=partitions, replication_factor=rep_factor)
    except Exception as exc:
        logger.error("Topic verification failed: %s", exc)
        sys.exit(1)

    # 2. Initialize Kafka Producer with safety and delivery guarantees
    logger.info("Initializing KafkaProducer targeting %s (topic: %s)...", args.bootstrap, args.topic)
    try:
        producer = KafkaProducer(
            bootstrap_servers=args.bootstrap,
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            value_serializer=lambda v: json.dumps(v, separators=(",", ":")).encode("utf-8"),
            acks="all",
            retries=int(os.getenv("KAFKA_RETRY_ATTEMPTS", "10")),
            retry_backoff_ms=int(float(os.getenv("KAFKA_RETRY_BACKOFF_SECONDS", "3")) * 1000),
            linger_ms=10,
            batch_size=int(os.getenv("KAFKA_BATCH_SIZE", "16384")),
            request_timeout_ms=int(os.getenv("KAFKA_REQUEST_TIMEOUT_MS", "30000")),
        )
    except Exception as exc:
        logger.error("Failed to connect producer to Kafka at %s: %s", args.bootstrap, exc)
        sys.exit(1)

    stopper = GracefulProducerStopper()
    published_count = 0
    failed_count = 0
    start_time = time.time()

    logger.info("Starting stream transmission from %s to topic '%s'...", csv_path, args.topic)

    try:
        while not stopper.stop_requested:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    logger.error("Input CSV file is empty or corrupted: %s", csv_path)
                    sys.exit(1)
                validate_csv_header(reader.fieldnames)

                for row in reader:
                    if stopper.stop_requested:
                        break

                    payload = create_event_payload(row, schema_version=args.schema_version)
                    patient_id = str(payload["event"].get("patient_id") or "UNKNOWN")

                    try:
                        # Send record keyed on patient_id for partition ordering
                        future = producer.send(args.topic, key=patient_id, value=payload)
                        # Wait for broker acknowledgment
                        future.get(timeout=30)
                        published_count += 1

                        if published_count % 500 == 0:
                            elapsed = time.time() - start_time
                            rate = published_count / elapsed if elapsed > 0 else 0
                            logger.info("Published %d events (rate: %.1f events/s)...", published_count, rate)

                        if args.delay_ms > 0:
                            time.sleep(args.delay_ms / 1000.0)

                        if args.limit > 0 and published_count >= args.limit:
                            logger.info("Reached configured event limit of %d.", args.limit)
                            break

                    except Exception as send_err:
                        failed_count += 1
                        logger.error("Failed to publish event: %s", send_err)
                        if failed_count > 50 and published_count == 0:
                            logger.critical("Excessive consecutive publish failures. Aborting.")
                            sys.exit(1)

            producer.flush()

            if args.limit > 0 and published_count >= args.limit:
                break
            if not args.repeat:
                break
            logger.info("Finished pass over CSV. --repeat enabled; looping...")

    except KeyboardInterrupt:
        logger.info("Streaming interrupted by user (KeyboardInterrupt).")
    finally:
        logger.info("Flushing pending records and closing producer...")
        try:
            producer.flush(timeout=10)
            producer.close(timeout=10)
        except Exception:
            pass

    elapsed_total = time.time() - start_time
    print(
        f"\nProducer completed: {published_count:,} events published to '{args.topic}', "
        f"{failed_count} failures in {elapsed_total:.2f}s."
    )
    if published_count == 0 and not stopper.stop_requested:
        sys.exit(1)


if __name__ == "__main__":
    main()
