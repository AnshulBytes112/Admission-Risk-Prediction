"""Kafka topic initialization and management utility.

Provides reusable, idempotent topic creation with safety checks:
- Verifies broker connectivity with bounded retries.
- Creates configured topic if not existing.
- Gracefully handles topic creation race conditions (TopicAlreadyExistsError).
- Warns if topic exists with a different partition count (does not destroy existing data).
- Validates that replication factor does not exceed broker count.
- Structured logging without exposing sensitive connection credentials.
"""
import argparse
import logging
import os
import sys
import time
from typing import Any

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, KafkaError

logger = logging.getLogger("kafka_topic_manager")


def wait_for_kafka(bootstrap_servers: str, timeout_seconds: int = 60, retry_interval: float = 3.0) -> bool:
    """Poll Kafka broker until reachable or timeout expires."""
    deadline = time.time() + timeout_seconds
    last_err = None
    logger.info("Checking Kafka broker connectivity at %s (timeout: %ds)...", bootstrap_servers, timeout_seconds)
    while time.time() < deadline:
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=bootstrap_servers,
                request_timeout_ms=4000,
            )
            admin.close()
            logger.info("Successfully connected to Kafka broker at %s", bootstrap_servers)
            return True
        except Exception as exc:
            last_err = exc
            logger.debug("Kafka broker not ready yet (%s). Retrying in %.1fs...", exc, retry_interval)
            time.sleep(retry_interval)
    raise TimeoutError(f"Kafka broker at {bootstrap_servers} remained unavailable after {timeout_seconds}s: {last_err}")


def ensure_topic(
    bootstrap_servers: str,
    topic: str,
    num_partitions: int = 3,
    replication_factor: int = 1,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Ensure that the specified topic exists with configured partitions and replication.

    Idempotent and safe: never deletes or resets an existing topic.
    """
    if num_partitions < 1:
        raise ValueError(f"Partition count must be >= 1, got {num_partitions}")
    if replication_factor < 1:
        raise ValueError(f"Replication factor must be >= 1, got {replication_factor}")

    wait_for_kafka(bootstrap_servers, timeout_seconds=timeout_seconds)

    admin = KafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        request_timeout_ms=10000,
    )
    try:
        existing_topics = admin.list_topics()
        if topic in existing_topics:
            logger.info("Topic '%s' already exists in Kafka cluster.", topic)
            try:
                # Describe topic to verify partition count
                cluster_metadata = admin.describe_topics([topic])
                if cluster_metadata and len(cluster_metadata) > 0:
                    partitions = cluster_metadata[0].get("partitions", [])
                    actual_count = len(partitions)
                    if actual_count != num_partitions:
                        logger.warning(
                            "Topic '%s' partition mismatch: cluster has %d partitions, configured for %d. "
                            "Retaining existing partitions without destructive changes.",
                            topic,
                            actual_count,
                            num_partitions,
                        )
                    return {"status": "exists", "topic": topic, "partitions": actual_count}
            except Exception as desc_err:
                logger.debug("Could not describe topic '%s': %s", topic, desc_err)
            return {"status": "exists", "topic": topic, "partitions": num_partitions}

        # Topic does not exist; create it
        new_topic = NewTopic(
            name=topic,
            num_partitions=num_partitions,
            replication_factor=replication_factor,
        )
        logger.info(
            "Creating Kafka topic '%s' (partitions=%d, replication_factor=%d)...",
            topic,
            num_partitions,
            replication_factor,
        )
        admin.create_topics(new_topics=[new_topic], validate_only=False)
        logger.info("Successfully created topic '%s'.", topic)
        return {"status": "created", "topic": topic, "partitions": num_partitions}

    except TopicAlreadyExistsError:
        logger.info("Topic '%s' was created concurrently by another process.", topic)
        return {"status": "exists", "topic": topic, "partitions": num_partitions}
    except KafkaError as exc:
        if "InvalidReplicationFactor" in str(exc):
            raise RuntimeError(
                f"Cannot create topic with replication_factor={replication_factor}: "
                f"exceeds available Kafka broker count. For local development use replication_factor=1."
            ) from exc
        raise
    finally:
        try:
            admin.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize and verify Kafka topic")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--topic", default=os.getenv("KAFKA_TOPIC", "patient-observations"))
    parser.add_argument("--partitions", type=int, default=int(os.getenv("KAFKA_PARTITIONS", "3")))
    parser.add_argument("--replication-factor", type=int, default=int(os.getenv("KAFKA_REPLICATION_FACTOR", "1")))
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
    try:
        result = ensure_topic(
            bootstrap_servers=args.bootstrap,
            topic=args.topic,
            num_partitions=args.partitions,
            replication_factor=args.replication_factor,
            timeout_seconds=args.timeout,
        )
        print(f"Topic '{args.topic}' verified: {result['status']} ({result.get('partitions')} partitions).")
    except Exception as exc:
        print(f"Error initializing topic '{args.topic}': {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
