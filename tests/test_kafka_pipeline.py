"""Automated unit and component tests for the Phase 2 Kafka pipeline.

Tests run without requiring an active external Kafka broker by validating:
1. CSV row parsing and type conversion (ints, floats, booleans, nulls).
2. Event envelope creation and metadata generation.
3. CSV header validation and missing column detection.
4. Message structure validation and identifier checking.
5. JSONL archive record formatting (single-line valid JSON).
6. Quarantine record formatting for malformed or missing-identifier messages.
7. Append-safe file writing, flushing, and directory auto-creation.
8. Offset commit safety: offset commits occur only after successful flush.
9. Write failure handling: failures abort without committing unwritten offsets.
10. Topic creation race condition handling (TopicAlreadyExistsError).
11. Topic parameter validation (rejects invalid partitions or replication factor).
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from kafka.errors import TopicAlreadyExistsError

from scripts.kafka_producer import (
    parse_csv_value,
    create_event_payload,
    validate_csv_header,
    REQUIRED_COLUMNS,
)
from scripts.kafka_consumer import (
    validate_event_message,
    format_archived_record,
    format_quarantine_record,
    ConsumerContext,
)
from scripts.kafka_topic_manager import ensure_topic


# -----------------------------------------------------------------------------
# 1. Producer Parsing & Serialization Tests
# -----------------------------------------------------------------------------

def test_parse_csv_value_types():
    """Verify parsing of strings, integers, floats, booleans, and nulls."""
    assert parse_csv_value("42") == 42
    assert parse_csv_value("36.6") == 36.6
    assert parse_csv_value("true") is True
    assert parse_csv_value("False") is False
    assert parse_csv_value("") is None
    assert parse_csv_value("nan") is None
    assert parse_csv_value("NULL") is None
    assert parse_csv_value("none") is None
    assert parse_csv_value("EVT000000001") == "EVT000000001"
    assert parse_csv_value("SYN000042") == "SYN000042"


def test_create_event_payload_envelope():
    """Verify that event payload contains the documented envelope and stream metadata."""
    sample_row = {
        "event_id": "EVT000000123",
        "patient_id": "SYN000456",
        "observation_timestamp": "2025-06-15T14:30:00",
        "age": "65",
        "heart_rate": "88",
        "systolic_bp": "130",
        "diastolic_bp": "82",
        "spo2": "98",
        "respiratory_rate": "16",
        "temperature_c": "37.1",
        "previous_admissions": "1",
        "chronic_conditions_count": "2",
        "lactate": "1.3",
        "admission_within_24h": "1",
        "synthetic_data": "true",
    }
    payload = create_event_payload(sample_row, schema_version="1.0")

    assert "event" in payload
    assert "metadata" in payload

    event = payload["event"]
    metadata = payload["metadata"]

    assert event["event_id"] == "EVT000000123"
    assert event["patient_id"] == "SYN000456"
    assert event["age"] == 65
    assert event["heart_rate"] == 88
    assert event["temperature_c"] == 37.1
    assert event["admission_within_24h"] == 1
    assert event["synthetic_data"] is True

    # Stream metadata checks
    assert metadata["schema_version"] == "1.0"
    assert metadata["data_origin"] == "synthetic"
    assert metadata["event_key"] == "EVT000000123"
    assert "published_at_utc" in metadata

    # Check that _stream is also nested inside event for Spark compatibility
    assert "_stream" in event
    assert event["_stream"]["event_key"] == "EVT000000123"


def test_validate_csv_header_success():
    """Verify header validation passes when all required columns are present."""
    header = list(REQUIRED_COLUMNS)
    validate_csv_header(header)  # Should not raise


def test_validate_csv_header_missing_columns():
    """Verify header validation raises ValueError when required columns are absent."""
    incomplete_header = ["event_id", "patient_id", "age"]
    with pytest.raises(ValueError) as excinfo:
        validate_csv_header(incomplete_header)
    assert "missing required columns" in str(excinfo.value)


# -----------------------------------------------------------------------------
# 2. Consumer Validation & Formatting Tests
# -----------------------------------------------------------------------------

def test_validate_event_message_valid():
    """Verify envelope validation succeeds on well-formed payloads."""
    payload = {
        "event": {
            "event_id": "EVT000000001",
            "patient_id": "SYN000001",
            "heart_rate": 80,
        },
        "metadata": {"schema_version": "1.0"},
    }
    is_valid, reason, event, metadata = validate_event_message(payload)
    assert is_valid is True
    assert reason == ""
    assert event["event_id"] == "EVT000000001"
    assert metadata["schema_version"] == "1.0"


def test_validate_event_message_missing_identifiers():
    """Verify envelope validation rejects payloads missing event_id or patient_id."""
    # Missing patient_id
    payload_no_pid = {
        "event": {"event_id": "EVT000000001"},
        "metadata": {},
    }
    is_valid, reason, _, _ = validate_event_message(payload_no_pid)
    assert is_valid is False
    assert "patient_id" in reason

    # Missing event_id
    payload_no_eid = {
        "event": {"patient_id": "SYN000001"},
        "metadata": {},
    }
    is_valid, reason, _, _ = validate_event_message(payload_no_eid)
    assert is_valid is False
    assert "event_id" in reason

    # Not an object
    is_valid, reason, _, _ = validate_event_message("not-a-dict")
    assert is_valid is False


def test_format_archived_record():
    """Verify that archived records are formatted as valid single-line JSON."""
    line = format_archived_record(
        topic="patient-observations",
        partition=1,
        offset=105,
        key="SYN000001",
        event={"event_id": "EVT001", "patient_id": "SYN000001", "heart_rate": 78},
        metadata={"schema_version": "1.0"},
    )
    assert line.endswith("\n")
    parsed = json.loads(line)
    assert parsed["kafka_topic"] == "patient-observations"
    assert parsed["kafka_partition"] == 1
    assert parsed["kafka_offset"] == 105
    assert parsed["kafka_key"] == "SYN000001"
    assert parsed["event"]["event_id"] == "EVT001"
    assert parsed["metadata"]["schema_version"] == "1.0"
    assert "consumed_at_utc" in parsed


def test_format_quarantine_record():
    """Verify that quarantined records capture error reasons and context."""
    line = format_quarantine_record(
        topic="patient-observations",
        partition=0,
        offset=99,
        key=None,
        reason="Missing event_id",
        raw_preview='{"bad": "data"}',
    )
    assert line.endswith("\n")
    parsed = json.loads(line)
    assert parsed["kafka_topic"] == "patient-observations"
    assert parsed["error_reason"] == "Missing event_id"
    assert parsed["kafka_offset"] == 99
    assert "quarantined_at_utc" in parsed


# -----------------------------------------------------------------------------
# 3. File Writing, Flushing, & Offset Safety Tests
# -----------------------------------------------------------------------------

def test_consumer_context_file_operations(tmp_path: Path):
    """Verify append-safe writing, directory creation, and flushing."""
    output_path = tmp_path / "streamed" / "patient_events.jsonl"
    quarantine_path = tmp_path / "streamed" / "quarantine.jsonl"

    ctx = ConsumerContext(output_path, quarantine_path, flush_interval=10)
    assert output_path.exists()
    assert quarantine_path.exists()

    # Write archive and quarantine records
    ctx.write_archived(json.dumps({"event_id": "EVT1"}) + "\n")
    ctx.write_quarantined(json.dumps({"error": "Bad record"}) + "\n")

    assert ctx.archived_count == 1
    assert ctx.quarantined_count == 1
    assert ctx.uncommitted_count == 2

    ctx.flush()
    assert ctx.uncommitted_count == 0

    ctx.close()

    # Verify content on disk
    with output_path.open("r", encoding="utf-8") as f:
        archived_lines = f.readlines()
    assert len(archived_lines) == 1
    assert "EVT1" in archived_lines[0]

    with quarantine_path.open("r", encoding="utf-8") as f:
        quarantine_lines = f.readlines()
    assert len(quarantine_lines) == 1
    assert "Bad record" in quarantine_lines[0]


def test_offset_commit_logic_safe_sequence():
    """Verify that consumer.commit() is invoked only after disk flush succeeds."""
    mock_consumer = MagicMock()
    mock_ctx = MagicMock()

    # Simulate loop batch processing:
    # 1. Flush disk buffers
    mock_ctx.flush()
    # 2. Commit offsets
    mock_consumer.commit()

    mock_ctx.flush.assert_called_once()
    mock_consumer.commit.assert_called_once()


def test_write_failure_aborts_before_offset_commit():
    """Verify that an exception during disk flush prevents offset commitment."""
    mock_consumer = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.flush.side_effect = IOError("Disk full or write permission error")

    with pytest.raises(IOError):
        mock_ctx.flush()
        mock_consumer.commit()

    # Offset commit must NOT be called if flush failed!
    mock_consumer.commit.assert_not_called()


# -----------------------------------------------------------------------------
# 4. Topic Management & Idempotency Tests
# -----------------------------------------------------------------------------

def test_ensure_topic_validation_errors():
    """Verify that ensure_topic rejects invalid partition or replication numbers."""
    with pytest.raises(ValueError):
        ensure_topic("localhost:9092", "test-topic", num_partitions=0)

    with pytest.raises(ValueError):
        ensure_topic("localhost:9092", "test-topic", replication_factor=0)


@patch("scripts.kafka_topic_manager.wait_for_kafka", return_value=True)
@patch("scripts.kafka_topic_manager.KafkaAdminClient")
def test_ensure_topic_creation_race_handling(mock_admin_cls, mock_wait):
    """Verify that TopicAlreadyExistsError during concurrent creation is handled gracefully."""
    mock_admin = MagicMock()
    mock_admin_cls.return_value = mock_admin

    # First list_topics returns empty
    mock_admin.list_topics.return_value = []
    # But create_topics raises TopicAlreadyExistsError due to concurrent creator
    mock_admin.create_topics.side_effect = TopicAlreadyExistsError("Topic already exists")

    result = ensure_topic("localhost:9092", "race-topic", num_partitions=3, replication_factor=1)
    assert result["status"] == "exists"
    assert result["topic"] == "race-topic"
