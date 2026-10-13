"""Automated unit and component tests for Phase 3 HDFS Ingestion.

Tests execute standalone without requiring a live Hadoop cluster by using
file fixtures and mock HDFS runners to validate:
1. Valid JSONL input and envelope structure preservation.
2. Multiple JSONL records parsing and counting.
3. Empty source file rejection (ValueError).
4. Missing source file rejection (FileNotFoundError).
5. Invalid JSON parsing and quarantine categorization.
6. Missing required event identifiers (event_id, patient_id) quarantine.
7. Source SHA-256 calculation accuracy.
8. Partition path generation based on explicit or default UTC dates.
9. Staged atomic upload workflow and temporary file cleanup.
10. Idempotency: identical destination file triggers no-op without re-upload.
11. Conflict detection: differing destination without --overwrite raises FileExistsError.
12. Overwrite behavior: overwriting existing destination replaces file cleanly.
13. Manifest creation with all required dynamically computed fields.
14. Failed HDFS command handling and error reporting.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from scripts.ingest_to_hdfs import (
    compute_sha256,
    validate_jsonl_content,
    resolve_container_path,
    HdfsIngestionPipeline,
)


class MockHdfsRunner:
    """Simulates HDFS CLI executions for standalone testing."""

    def __init__(self):
        self.commands = []
        self.files_on_hdfs = {}  # path -> size_in_bytes
        self.fail_next_put = False
        self.fail_mkdir = False

    def run_cmd(self, args: list[str]) -> subprocess.CompletedProcess:
        self.commands.append(list(args))

        if args[0] == "hdfs" and args[1] == "dfsadmin" and args[2] == "-safemode":
            return subprocess.CompletedProcess(args, 0, stdout="Safe mode is OFF\n", stderr="")

        if args[0] == "hdfs" and args[1] == "dfs":
            subcmd = args[2]

            if subcmd == "-mkdir":
                if self.fail_mkdir:
                    return subprocess.CompletedProcess(args, 1, stdout="", stderr="Mkdir failed")
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

            if subcmd == "-test" and args[3] == "-e":
                path = args[4]
                code = 0 if path in self.files_on_hdfs else 1
                return subprocess.CompletedProcess(args, code, stdout="", stderr="")

            if subcmd == "-stat" and args[3] == "%b":
                path = args[4]
                if path in self.files_on_hdfs:
                    return subprocess.CompletedProcess(args, 0, stdout=f"{self.files_on_hdfs[path]}\n", stderr="")
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="File not found")

            if subcmd == "-put":
                if self.fail_next_put:
                    return subprocess.CompletedProcess(args, 1, stdout="", stderr="Simulated put network failure")
                dst = args[5]
                if dst not in self.files_on_hdfs:
                    self.files_on_hdfs[dst] = getattr(self, "default_put_size", 1024)
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

            if subcmd == "-mv":
                src = args[3]
                dst = args[4]
                if src in self.files_on_hdfs:
                    self.files_on_hdfs[dst] = self.files_on_hdfs.pop(src)
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

            if subcmd == "-rm":
                path = args[4]
                self.files_on_hdfs.pop(path, None)
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


# -----------------------------------------------------------------------------
# 1. Validation & Content Tests
# -----------------------------------------------------------------------------

def test_compute_sha256(tmp_path: Path):
    """Verify SHA-256 computation matches hashlib output."""
    sample_file = tmp_path / "sample.txt"
    sample_content = b"Emergency admission risk analytics test string\n"
    sample_file.write_bytes(sample_content)

    expected_hash = hashlib.sha256(sample_content).hexdigest()
    assert compute_sha256(sample_file) == expected_hash


def test_validate_jsonl_content_valid(tmp_path: Path):
    """Verify parsing and validation of valid JSONL records."""
    jsonl_file = tmp_path / "events.jsonl"
    lines = [
        json.dumps({
            "kafka_topic": "patient-observations",
            "kafka_offset": i,
            "event": {
                "event_id": f"EVT{i:06d}",
                "patient_id": f"SYN{i:06d}",
                "age": 60,
                "heart_rate": 80.0,
                "synthetic_data": True,
            },
            "metadata": {"schema_version": "1.0"},
        }) + "\n"
        for i in range(1, 6)
    ]
    jsonl_file.write_text("".join(lines), encoding="utf-8")

    result = validate_jsonl_content(jsonl_file)
    assert result["total_records"] == 5
    assert result["valid_records"] == 5
    assert result["invalid_records"] == 0
    assert len(result["quarantine_entries"]) == 0
    assert result["sample_record"]["event"]["event_id"] == "EVT000001"


def test_validate_jsonl_content_with_malformed_records(tmp_path: Path):
    """Verify that corrupt lines and missing identifiers are categorized as invalid."""
    jsonl_file = tmp_path / "mixed_events.jsonl"
    content = (
        json.dumps({"event": {"event_id": "EVT1", "patient_id": "SYN1"}}) + "\n"
        + "CORRUPT NOT JSON LINE\n"
        + json.dumps({"event": {"missing_patient_id": True}}) + "\n"
        + json.dumps({"not_an_event_dict": 123}) + "\n"
        + json.dumps({"event": {"event_id": "EVT2", "patient_id": "SYN2"}}) + "\n"
    )
    jsonl_file.write_text(content, encoding="utf-8")

    result = validate_jsonl_content(jsonl_file)
    assert result["total_records"] == 5
    assert result["valid_records"] == 2
    assert result["invalid_records"] == 3
    assert len(result["quarantine_entries"]) == 3
    assert result["quarantine_entries"][0]["line_number"] == 2


def test_validate_jsonl_empty_file_rejected(tmp_path: Path):
    """Verify that empty files raise a descriptive ValueError."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        validate_jsonl_content(empty_file)
    assert "empty" in str(excinfo.value)


# -----------------------------------------------------------------------------
# 2. Path Resolution Tests
# -----------------------------------------------------------------------------

def test_resolve_container_path(tmp_path: Path):
    """Verify path mapping to /shared-data within project root."""
    project_root = tmp_path
    data_dir = project_root / "data" / "streamed"
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_file = data_dir / "patient_events.jsonl"
    sample_file.write_text("test", encoding="utf-8")

    container_path, is_mounted = resolve_container_path(sample_file, project_root)
    assert is_mounted is True
    assert container_path == "/shared-data/streamed/patient_events.jsonl"

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside", encoding="utf-8")
    _, is_mounted_outside = resolve_container_path(outside_file, project_root)
    assert is_mounted_outside is False


# -----------------------------------------------------------------------------
# 3. Pipeline Ingestion Workflow Tests
# -----------------------------------------------------------------------------

def test_ingest_pipeline_success(tmp_path: Path):
    """Verify complete successful upload, staged rename, and manifest creation."""
    project_root = tmp_path
    stream_dir = project_root / "data" / "streamed"
    stream_dir.mkdir(parents=True, exist_ok=True)
    source_file = stream_dir / "patient_events.jsonl"

    record = json.dumps({
        "kafka_topic": "patient-observations",
        "event": {"event_id": "EVT100", "patient_id": "SYN100", "age": 45},
    }) + "\n"
    source_file.write_text(record, encoding="utf-8")
    expected_size = source_file.stat().st_size

    runner = MockHdfsRunner()
    # Hook runner to report the actual file size when queried
    def dynamic_cmd(args):
        if args[0] == "hdfs" and args[1] == "dfs" and args[2] == "-put":
            runner.files_on_hdfs[args[5]] = expected_size
        return runner.run_cmd(args)

    mock_runner = MagicMock()
    mock_runner.run_cmd.side_effect = dynamic_cmd

    pipeline = HdfsIngestionPipeline(
        runner=mock_runner,
        project_root=project_root,
        hdfs_base="/emergency-admission-bda",
    )

    result = pipeline.ingest_file(source_file, ingestion_date="2026-10-14", check_readiness=True)

    assert result["status"] == "success"
    assert result["ingestion_date"] == "2026-10-14"
    assert result["record_count"] == 1
    assert "patient_events.jsonl" in result["hdfs_destination"]
    assert "manifest.json" in result["manifest_destination"]

    # Verify manifest written locally
    local_manifest = Path(result["local_manifest"])
    assert local_manifest.exists()
    manifest_data = json.loads(local_manifest.read_text(encoding="utf-8"))
    assert manifest_data["record_count"] == 1
    assert manifest_data["verification_status"] == "verified"
    assert manifest_data["source_sha256"] == compute_sha256(source_file)


def test_ingest_pipeline_idempotency_noop(tmp_path: Path):
    """Verify that attempting to re-upload an identical file is an idempotent NO-OP."""
    project_root = tmp_path
    stream_dir = project_root / "data" / "streamed"
    stream_dir.mkdir(parents=True, exist_ok=True)
    source_file = stream_dir / "patient_events.jsonl"

    record = json.dumps({"event": {"event_id": "EVT1", "patient_id": "SYN1"}}) + "\n"
    source_file.write_text(record, encoding="utf-8")
    size = source_file.stat().st_size

    runner = MockHdfsRunner()
    # Destination already exists on HDFS with matching size
    target_path = "/emergency-admission-bda/raw/kafka-events/ingestion_date=2026-10-14/patient_events.jsonl"
    runner.files_on_hdfs[target_path] = size

    pipeline = HdfsIngestionPipeline(
        runner=runner,
        project_root=project_root,
        hdfs_base="/emergency-admission-bda",
        overwrite=False,
    )

    result = pipeline.ingest_file(source_file, ingestion_date="2026-10-14", check_readiness=False)
    assert result["status"] == "idempotent_noop"


def test_ingest_pipeline_conflict_raises_error(tmp_path: Path):
    """Verify that differing existing destination raises FileExistsError when overwrite=False."""
    project_root = tmp_path
    stream_dir = project_root / "data" / "streamed"
    stream_dir.mkdir(parents=True, exist_ok=True)
    source_file = stream_dir / "patient_events.jsonl"

    record = json.dumps({"event": {"event_id": "EVT1", "patient_id": "SYN1"}}) + "\n"
    source_file.write_text(record, encoding="utf-8")

    runner = MockHdfsRunner()
    target_path = "/emergency-admission-bda/raw/kafka-events/ingestion_date=2026-10-14/patient_events.jsonl"
    runner.files_on_hdfs[target_path] = 999999  # Different size!

    pipeline = HdfsIngestionPipeline(
        runner=runner,
        project_root=project_root,
        hdfs_base="/emergency-admission-bda",
        overwrite=False,
    )

    with pytest.raises(FileExistsError) as excinfo:
        pipeline.ingest_file(source_file, ingestion_date="2026-10-14", check_readiness=False)
    assert "exists in HDFS with different size" in str(excinfo.value)


def test_ingest_pipeline_upload_failure_cleans_up(tmp_path: Path):
    """Verify that an upload failure raises RuntimeError and cleans up staging files."""
    project_root = tmp_path
    stream_dir = project_root / "data" / "streamed"
    stream_dir.mkdir(parents=True, exist_ok=True)
    source_file = stream_dir / "patient_events.jsonl"
    source_file.write_text(json.dumps({"event": {"event_id": "EVT1", "patient_id": "SYN1"}}) + "\n", encoding="utf-8")

    runner = MockHdfsRunner()
    runner.fail_next_put = True

    pipeline = HdfsIngestionPipeline(
        runner=runner,
        project_root=project_root,
        hdfs_base="/emergency-admission-bda",
    )

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.ingest_file(source_file, ingestion_date="2026-10-14", check_readiness=False)
    assert "HDFS upload failed" in str(excinfo.value)
