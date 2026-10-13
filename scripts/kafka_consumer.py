"""Kafka Event Consumer for Emergency Observations Archive.

Consumes streamed clinical events from Kafka and writes an append-safe local JSONL stream archive:
- Connects to Kafka with manual offset management (enable_auto_commit=False).
- Enforces at-least-once delivery guarantees: commits offsets ONLY after local disk write and flush.
- Validates JSON format and required event identifiers (event_id, patient_id).
- Quarantines malformed, invalid, or corrupted messages to a separate quarantine file.
- Outputs clean UTF-8 JSON Lines (one line per event) compatible with Phase 3 HDFS ingestion and Phase 4 PySpark.
- Graceful shutdown on SIGINT / SIGTERM with safe file closure and offset commit.
- Structured logging without exposing sensitive patient-level details.
"""
import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer

try:
    from scripts.kafka_topic_manager import wait_for_kafka
except ModuleNotFoundError:
    from kafka_topic_manager import wait_for_kafka

logger = logging.getLogger("kafka_consumer")


def validate_event_message(data: Any) -> tuple[bool, str, dict[str, Any], dict[str, Any]]:
    """Validate incoming message structure.

    Returns:
        (is_valid, error_reason, event_dict, metadata_dict)
    """
    if not isinstance(data, dict):
        return False, "Payload is not a JSON object", {}, {}

    # Extract event and metadata
    event = data.get("event")
    metadata = data.get("metadata", {})

    if not isinstance(event, dict):
        return False, "Missing or non-object 'event' field in envelope", {}, {}

    event_id = event.get("event_id")
    patient_id = event.get("patient_id")

    if not event_id or str(event_id).strip() == "":
        return False, "Event missing required 'event_id' identifier", event, metadata
    if not patient_id or str(patient_id).strip() == "":
        return False, "Event missing required 'patient_id' identifier", event, metadata

    return True, "", event, metadata


def format_archived_record(
    topic: str,
    partition: int,
    offset: int,
    key: str | None,
    event: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Format a valid event into a single-line JSON string with transport metadata."""
    record = {
        "kafka_topic": topic,
        "kafka_partition": partition,
        "kafka_offset": offset,
        "kafka_key": key,
        "consumed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        "metadata": metadata,
    }
    return json.dumps(record, separators=(",", ":")) + "\n"


def format_quarantine_record(
    topic: str,
    partition: int,
    offset: int,
    key: str | None,
    reason: str,
    raw_preview: str,
) -> str:
    """Format a malformed event into a quarantine log entry."""
    record = {
        "kafka_topic": topic,
        "kafka_partition": partition,
        "kafka_offset": offset,
        "kafka_key": key,
        "quarantined_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error_reason": reason,
        "raw_preview": raw_preview[:500],
    }
    return json.dumps(record, separators=(",", ":")) + "\n"


class ConsumerContext:
    def __init__(self, output_path: Path, quarantine_path: Path, flush_interval: int = 250):
        self.output_path = output_path
        self.quarantine_path = quarantine_path
        self.flush_interval = flush_interval

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)

        self.out_file = self.output_path.open("a", encoding="utf-8")
        self.quarantine_file = self.quarantine_path.open("a", encoding="utf-8")

        self.consumed_count = 0
        self.archived_count = 0
        self.quarantined_count = 0
        self.uncommitted_count = 0

    def write_archived(self, line: str) -> None:
        self.out_file.write(line)
        self.archived_count += 1
        self.uncommitted_count += 1

    def write_quarantined(self, line: str) -> None:
        self.quarantine_file.write(line)
        self.quarantined_count += 1
        self.uncommitted_count += 1

    def flush(self) -> None:
        if self.uncommitted_count > 0:
            self.out_file.flush()
            os.fsync(self.out_file.fileno())
            self.quarantine_file.flush()
            os.fsync(self.quarantine_file.fileno())
            self.uncommitted_count = 0

    def close(self) -> None:
        try:
            self.flush()
            self.out_file.close()
            self.quarantine_file.close()
        except Exception as exc:
            logger.error("Error during output file closure: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume Kafka emergency observation events to JSONL archive")
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_TOPIC", "patient-observations"),
        help="Kafka topic to consume from",
    )
    parser.add_argument(
        "--bootstrap",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap server address",
    )
    parser.add_argument(
        "--group",
        default=os.getenv("KAFKA_CONSUMER_GROUP", "emergency-admission-bda-consumer"),
        help="Kafka consumer group ID",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("STREAM_OUTPUT_PATH", os.getenv("CONSUMER_OUTPUT", "data/streamed/patient_events.jsonl")),
        help="Target local JSONL archive path (Phase 3 hand-off artifact)",
    )
    parser.add_argument(
        "--quarantine",
        default=os.getenv("STREAM_ERROR_PATH", "data/streamed/quarantine_events.jsonl"),
        help="Target local JSONL quarantine path for invalid messages",
    )
    parser.add_argument(
        "--auto-offset-reset",
        default=os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
        choices=["earliest", "latest", "none"],
        help="Offset reset policy",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=int(os.getenv("STREAM_MAX_EVENTS", "0")),
        help="Stop after consuming N messages (0 for continuous consumption)",
    )
    parser.add_argument(
        "--poll-timeout-ms",
        type=int,
        default=int(os.getenv("KAFKA_POLL_TIMEOUT_MS", "1000")),
        help="Poll timeout in milliseconds",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=int(os.getenv("STREAM_FLUSH_INTERVAL", "250")),
        help="Number of records between file flushes and offset commits",
    )
    args = parser.parse_args()

    # Configure structured logging
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_path = Path(args.output)
    quarantine_path = Path(args.quarantine)

    logger.info("Verifying Kafka broker connectivity at %s...", args.bootstrap)
    try:
        wait_for_kafka(args.bootstrap, timeout_seconds=60)
    except Exception as exc:
        logger.error("Kafka connectivity check failed: %s", exc)
        sys.exit(1)

    ctx = ConsumerContext(output_path, quarantine_path, flush_interval=args.flush_interval)

    # Signal handling for graceful shutdown
    running = True

    def _shutdown(signum, frame):
        nonlocal running
        logger.info("Received termination signal (%s). Initiating graceful shutdown...", signum)
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(
        "Initializing KafkaConsumer (topic=%s, group=%s, offset_reset=%s)...",
        args.topic,
        args.group,
        args.auto_offset_reset,
    )
    try:
        consumer = KafkaConsumer(
            args.topic,
            bootstrap_servers=args.bootstrap,
            group_id=args.group,
            enable_auto_commit=False,
            auto_offset_reset=args.auto_offset_reset,
            value_deserializer=None,  # Deserialize manually for safe error handling
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            max_poll_records=args.flush_interval,
            consumer_timeout_ms=args.poll_timeout_ms * 3,
        )
    except Exception as exc:
        logger.error("Failed to initialize Kafka consumer: %s", exc)
        ctx.close()
        sys.exit(1)

    logger.info("Consumer active. Writing stream archive to '%s'", output_path.resolve())

    try:
        while running:
            # Poll for messages
            msg_batch = consumer.poll(timeout_ms=args.poll_timeout_ms, max_records=args.flush_interval)

            if not msg_batch:
                continue

            for topic_partition, records in msg_batch.items():
                for record in records:
                    ctx.consumed_count += 1
                    raw_val = record.value

                    # 1. Parse JSON safely
                    try:
                        if raw_val is None:
                            raise ValueError("Received null payload")
                        decoded_str = raw_val.decode("utf-8")
                        payload = json.loads(decoded_str)
                    except Exception as json_err:
                        logger.warning(
                            "Corrupted message on partition %d offset %d: %s. Quarantining.",
                            record.partition,
                            record.offset,
                            json_err,
                        )
                        q_line = format_quarantine_record(
                            record.topic,
                            record.partition,
                            record.offset,
                            record.key,
                            f"JSON decode failure: {json_err}",
                            repr(raw_val)[:300],
                        )
                        ctx.write_quarantined(q_line)
                        continue

                    # 2. Validate envelope and identifiers
                    is_valid, reason, event_dict, metadata_dict = validate_event_message(payload)
                    if not is_valid:
                        logger.warning(
                            "Invalid message envelope on partition %d offset %d: %s. Quarantining.",
                            record.partition,
                            record.offset,
                            reason,
                        )
                        q_line = format_quarantine_record(
                            record.topic,
                            record.partition,
                            record.offset,
                            record.key,
                            reason,
                            json.dumps(payload)[:300],
                        )
                        ctx.write_quarantined(q_line)
                        continue

                    # 3. Archive valid record
                    out_line = format_archived_record(
                        record.topic,
                        record.partition,
                        record.offset,
                        record.key,
                        event_dict,
                        metadata_dict,
                    )
                    ctx.write_archived(out_line)

                    if ctx.archived_count % 500 == 0:
                        logger.info("Archived %d events to %s", ctx.archived_count, output_path.name)

                    if args.max_messages > 0 and ctx.consumed_count >= args.max_messages:
                        logger.info("Reached configured limit of %d messages.", args.max_messages)
                        running = False
                        break

                if not running:
                    break

            # 4. Flush files and commit offsets only after data is written to disk
            try:
                ctx.flush()
                consumer.commit()
            except Exception as flush_err:
                logger.error("Failed to flush archive to disk or commit offsets: %s", flush_err)
                # Do NOT commit offsets on write failure to guarantee at-least-once processing
                break

    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    finally:
        logger.info("Committing final offsets and closing consumer...")
        try:
            ctx.flush()
            consumer.commit()
            consumer.close(timeout=5)
            ctx.close()
        except Exception:
            pass

    print(
        f"\nConsumer finished: {ctx.consumed_count:,} consumed, "
        f"{ctx.archived_count:,} archived to '{output_path}', "
        f"{ctx.quarantined_count:,} quarantined to '{quarantine_path}'."
    )


if __name__ == "__main__":
    main()
