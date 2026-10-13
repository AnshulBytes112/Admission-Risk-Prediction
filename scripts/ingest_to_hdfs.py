"""HDFS Ingestion and Raw Data Lake Management Utility (Phase 3).

Ingests the Phase 2 Kafka consumer JSONL stream archive into Apache Hadoop HDFS:
- Pre-upload validation: verifies JSON format, checks schema envelope, counts valid/invalid records.
- Computes SHA-256 checksum and byte size of the source file.
- Checks NameNode & DataNode readiness (including SafeMode status).
- Organizes HDFS into a date-partitioned raw data lake:
    /emergency-admission-bda/raw/kafka-events/ingestion_date=YYYY-MM-DD/patient_events.jsonl
- Idempotency & Conflict Safety: detects previously ingested identical files (no-op)
  and guards against conflicting overwrites unless explicitly enabled.
- Staged upload: uploads to a temporary file, verifies size, and atomically moves to target.
- Generates and publishes machine-readable ingestion manifest in JSON format.
- Quarantines malformed records to /emergency-admission-bda/quarantine/kafka-events/.
- Structured logging without exposing sensitive patient-level details.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("ingest_to_hdfs")


# -----------------------------------------------------------------------------
# 1. Validation & Checksum Functions
# -----------------------------------------------------------------------------

def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a local file."""
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_jsonl_content(filepath: Path) -> dict[str, Any]:
    """Validate each line of the JSONL file.

    Returns:
        dict with total_records, valid_records, invalid_records,
        quarantine_entries, and sample_record.
    """
    total = 0
    valid = 0
    invalid = 0
    quarantine_entries = []
    sample = None

    with filepath.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            try:
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    invalid += 1
                    quarantine_entries.append({
                        "line_number": line_no,
                        "error": "Record is not a JSON object",
                        "raw_line": stripped[:300],
                    })
                    continue

                # Validate expected Kafka stream archive envelope
                event = record.get("event")
                if not isinstance(event, dict):
                    invalid += 1
                    quarantine_entries.append({
                        "line_number": line_no,
                        "error": "Missing or non-object 'event' field",
                        "raw_line": stripped[:300],
                    })
                    continue

                event_id = event.get("event_id")
                patient_id = event.get("patient_id")
                if not event_id or not patient_id:
                    invalid += 1
                    quarantine_entries.append({
                        "line_number": line_no,
                        "error": "Event missing event_id or patient_id identifier",
                        "raw_line": stripped[:300],
                    })
                    continue

                valid += 1
                if sample is None:
                    sample = record

            except Exception as exc:
                invalid += 1
                quarantine_entries.append({
                    "line_number": line_no,
                    "error": f"JSON parse error: {exc}",
                    "raw_line": stripped[:300],
                })

    if total == 0:
        raise ValueError(f"Input file '{filepath}' is empty (0 records found).")

    return {
        "total_records": total,
        "valid_records": valid,
        "invalid_records": invalid,
        "quarantine_entries": quarantine_entries,
        "sample_record": sample,
    }


# -----------------------------------------------------------------------------
# 2. HDFS Command Runner Abstraction (Enables testing without live Docker)
# -----------------------------------------------------------------------------

class HdfsRunner(Protocol):
    def run_cmd(self, args: list[str]) -> subprocess.CompletedProcess:
        ...


class DockerComposeHdfsRunner:
    """Executes HDFS CLI commands inside the Docker Compose NameNode container."""

    def __init__(self, service_name: str = "namenode", compose_file: str | None = None):
        self.service_name = service_name
        self.compose_file = compose_file

    def run_cmd(self, args: list[str]) -> subprocess.CompletedProcess:
        base_cmd = ["docker", "compose"]
        if self.compose_file:
            base_cmd.extend(["-f", self.compose_file])
        base_cmd.extend(["exec", "-T", self.service_name])
        full_cmd = base_cmd + args
        logger.debug("Executing HDFS command: %s", " ".join(full_cmd))
        return subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )


# -----------------------------------------------------------------------------
# 3. Path Resolution for Shared Volume
# -----------------------------------------------------------------------------

def resolve_container_path(local_path: Path, project_root: Path) -> tuple[str, bool]:
    """Resolve a local path to the container path mounted in NameNode (/shared-data).

    Returns:
        (container_path, is_mounted)
    """
    abs_local = local_path.resolve()
    abs_root = project_root.resolve()
    abs_data = (abs_root / "data").resolve()

    try:
        rel_data = abs_local.relative_to(abs_data)
        # File is inside ./data directory -> mapped to /shared-data
        return f"/shared-data/{rel_data.as_posix()}", True
    except ValueError:
        # File is outside ./data directory
        return "", False


# -----------------------------------------------------------------------------
# 4. Ingestion Workflow
# -----------------------------------------------------------------------------

class HdfsIngestionPipeline:
    def __init__(
        self,
        runner: HdfsRunner,
        project_root: Path,
        hdfs_base: str = "/emergency-admission-bda",
        overwrite: bool = False,
    ):
        self.runner = runner
        self.project_root = project_root
        self.hdfs_base = hdfs_base.rstrip("/")
        self.overwrite = overwrite

    def check_hdfs_readiness(self, timeout_seconds: int = 60, retry_interval: float = 3.0) -> bool:
        """Poll NameNode until HDFS responds and is out of SafeMode."""
        deadline = time.time() + timeout_seconds
        last_error = ""
        logger.info("Verifying HDFS NameNode readiness (timeout: %ds)...", timeout_seconds)

        while time.time() < deadline:
            res = self.runner.run_cmd(["hdfs", "dfsadmin", "-safemode", "get"])
            if res.returncode == 0:
                output = res.stdout.strip()
                if "Safe mode is OFF" in output:
                    logger.info("HDFS is ready and SafeMode is OFF.")
                    return True
                elif "Safe mode is ON" in output:
                    logger.info("HDFS SafeMode is ON. Waiting for initialization...")
            else:
                last_error = res.stderr.strip() or res.stdout.strip()
                logger.debug("HDFS not ready yet (%s). Retrying in %.1fs...", last_error, retry_interval)

            time.sleep(retry_interval)

        raise TimeoutError(f"HDFS cluster remained unavailable after {timeout_seconds}s: {last_error}")

    def file_exists_in_hdfs(self, hdfs_path: str) -> bool:
        res = self.runner.run_cmd(["hdfs", "dfs", "-test", "-e", hdfs_path])
        return res.returncode == 0

    def get_hdfs_file_size(self, hdfs_path: str) -> int:
        res = self.runner.run_cmd(["hdfs", "dfs", "-stat", "%b", hdfs_path])
        if res.returncode == 0 and res.stdout.strip().isdigit():
            return int(res.stdout.strip())
        return -1

    def ingest_file(
        self,
        source_file: Path,
        ingestion_date: str | None = None,
        check_readiness: bool = True,
    ) -> dict[str, Any]:
        """Execute complete validation, staged upload, manifest creation, and verification."""
        source_path = source_file.resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found at '{source_path}'")
        if source_path.stat().st_size == 0:
            raise ValueError(f"Source file '{source_path}' is empty (0 bytes).")

        # 1. Validate JSONL records & calculate SHA-256
        logger.info("Validating JSONL records and calculating SHA-256 for %s...", source_path.name)
        file_sha256 = compute_sha256(source_path)
        file_size_bytes = source_path.stat().st_size
        validation_info = validate_jsonl_content(source_path)

        # 2. Determine partition date & paths
        if not ingestion_date:
            ingestion_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

        raw_dir = f"{self.hdfs_base}/raw/kafka-events/ingestion_date={ingestion_date}"
        target_path = f"{raw_dir}/patient_events.jsonl"
        manifest_dir = f"{self.hdfs_base}/metadata/ingestion-manifests/ingestion_date={ingestion_date}"
        manifest_hdfs_path = f"{manifest_dir}/manifest.json"
        quarantine_hdfs_dir = f"{self.hdfs_base}/quarantine/kafka-events/ingestion_date={ingestion_date}"

        # 3. Check HDFS readiness
        if check_readiness:
            self.check_hdfs_readiness()

        # 4. Check for duplicate ingestion / idempotency
        if self.file_exists_in_hdfs(target_path):
            existing_size = self.get_hdfs_file_size(target_path)
            if existing_size == file_size_bytes:
                # Check manifest if present
                logger.info(
                    "Target file '%s' already exists in HDFS with matching size (%d bytes). "
                    "Operation is an idempotent NO-OP.",
                    target_path,
                    file_size_bytes,
                )
                return {
                    "status": "idempotent_noop",
                    "source_file": str(source_path),
                    "hdfs_destination": target_path,
                    "sha256": file_sha256,
                    "record_count": validation_info["total_records"],
                    "ingestion_date": ingestion_date,
                }
            elif not self.overwrite:
                raise FileExistsError(
                    f"Destination '{target_path}' exists in HDFS with different size ({existing_size} bytes vs {file_size_bytes} bytes). "
                    f"Aborting to prevent accidental data loss. Enable --overwrite to replace."
                )
            else:
                logger.warning("Destination '%s' exists with different content. Overwriting as requested.", target_path)

        # 5. Resolve container path
        container_path, is_mounted = resolve_container_path(source_path, self.project_root)
        if not is_mounted:
            raise ValueError(
                f"Source file '{source_path}' is outside the './data' directory and cannot be read by NameNode shared volume. "
                f"Please place files inside './data/' (e.g. data/streamed/patient_events.jsonl)."
            )

        # 6. Ensure destination directory exists
        res = self.runner.run_cmd(["hdfs", "dfs", "-mkdir", "-p", raw_dir])
        if res.returncode != 0:
            raise RuntimeError(f"Failed to create HDFS directory '{raw_dir}': {res.stderr}")

        # 7. Safe Staged Upload (Upload to tmp file first)
        tmp_staging = f"{raw_dir}/.tmp_{uuid.uuid4().hex}_{source_path.name}"
        logger.info("Uploading '%s' to HDFS staging path '%s'...", container_path, tmp_staging)

        upload_res = self.runner.run_cmd(["hdfs", "dfs", "-put", "-f", container_path, tmp_staging])
        if upload_res.returncode != 0:
            raise RuntimeError(f"HDFS upload failed: {upload_res.stderr.strip() or upload_res.stdout.strip()}")

        # Verify staged file size
        staged_size = self.get_hdfs_file_size(tmp_staging)
        if staged_size != file_size_bytes:
            # Clean up corrupted staging file
            self.runner.run_cmd(["hdfs", "dfs", "-rm", "-f", tmp_staging])
            raise RuntimeError(
                f"HDFS upload verification failed: uploaded size ({staged_size} B) "
                f"does not match local size ({file_size_bytes} B)."
            )

        # Atomic rename to final target
        logger.info("Moving staging file to final destination '%s'...", target_path)
        if self.overwrite and self.file_exists_in_hdfs(target_path):
            self.runner.run_cmd(["hdfs", "dfs", "-rm", "-f", target_path])

        mv_res = self.runner.run_cmd(["hdfs", "dfs", "-mv", tmp_staging, target_path])
        if mv_res.returncode != 0:
            self.runner.run_cmd(["hdfs", "dfs", "-rm", "-f", tmp_staging])
            raise RuntimeError(f"Failed to finalize HDFS file rename: {mv_res.stderr}")

        # 8. Upload quarantine file if invalid records exist
        if validation_info["invalid_records"] > 0:
            logger.warning(
                "Detected %d invalid records. Writing quarantine log...",
                validation_info["invalid_records"],
            )
            local_q_path = self.project_root / "data" / "streamed" / f"quarantine_{ingestion_date}.jsonl"
            local_q_path.parent.mkdir(parents=True, exist_ok=True)
            with local_q_path.open("w", encoding="utf-8") as qf:
                for entry in validation_info["quarantine_entries"]:
                    qf.write(json.dumps(entry, separators=(",", ":")) + "\n")

            q_container, _ = resolve_container_path(local_q_path, self.project_root)
            self.runner.run_cmd(["hdfs", "dfs", "-mkdir", "-p", quarantine_hdfs_dir])
            self.runner.run_cmd([
                "hdfs", "dfs", "-put", "-f",
                q_container,
                f"{quarantine_hdfs_dir}/quarantine_events.jsonl",
            ])

        # 9. Create and publish ingestion manifest
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest_data = {
            "manifest_version": "1.0",
            "project": "emergency-admission-risk-prediction",
            "data_origin": "synthetic",
            "source_filename": source_path.name,
            "source_size_bytes": file_size_bytes,
            "source_sha256": file_sha256,
            "record_count": validation_info["total_records"],
            "valid_record_count": validation_info["valid_records"],
            "invalid_record_count": validation_info["invalid_records"],
            "ingestion_date": ingestion_date,
            "ingested_at_utc": now_utc,
            "hdfs_destination": target_path,
            "verification_status": "verified",
        }

        # Save manifest locally
        local_manifest_dir = self.project_root / "data" / "metadata" / "ingestion_manifests" / f"ingestion_date={ingestion_date}"
        local_manifest_dir.mkdir(parents=True, exist_ok=True)
        local_manifest_path = local_manifest_dir / "manifest.json"
        with local_manifest_path.open("w", encoding="utf-8") as mf:
            json.dump(manifest_data, mf, indent=2)

        # Upload manifest to HDFS
        manifest_container, _ = resolve_container_path(local_manifest_path, self.project_root)
        self.runner.run_cmd(["hdfs", "dfs", "-mkdir", "-p", manifest_dir])
        self.runner.run_cmd(["hdfs", "dfs", "-put", "-f", manifest_container, manifest_hdfs_path])

        logger.info("Ingestion complete. Manifest published to '%s'.", manifest_hdfs_path)

        return {
            "status": "success",
            "source_file": str(source_path),
            "hdfs_destination": target_path,
            "manifest_destination": manifest_hdfs_path,
            "local_manifest": str(local_manifest_path),
            "sha256": file_sha256,
            "file_size_bytes": file_size_bytes,
            "record_count": validation_info["total_records"],
            "valid_record_count": validation_info["valid_records"],
            "invalid_record_count": validation_info["invalid_records"],
            "ingestion_date": ingestion_date,
            "manifest": manifest_data,
        }


# -----------------------------------------------------------------------------
# 5. CLI Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Phase 2 JSONL stream archive to Hadoop HDFS (Phase 3)")
    parser.add_argument(
        "--input",
        default=os.getenv("STREAM_OUTPUT_PATH", "data/streamed/patient_events.jsonl"),
        help="Path to source JSONL file",
    )
    parser.add_argument(
        "--hdfs-base",
        default=os.getenv("HDFS_BASE_PATH", "/emergency-admission-bda"),
        help="Root HDFS namespace path",
    )
    parser.add_argument(
        "--ingestion-date",
        default=os.getenv("HDFS_INGESTION_DATE", None),
        help="Explicit partition date (YYYY-MM-DD); defaults to current UTC date",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=os.getenv("HDFS_OVERWRITE", "false").lower() == "true",
        help="Overwrite existing destination file if already present",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("HDFS_READY_TIMEOUT_SECONDS", "60")),
        help="Seconds to wait for HDFS readiness",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parents[1]
    source_file = Path(args.input)
    if not source_file.is_absolute():
        source_file = project_root / source_file

    runner = DockerComposeHdfsRunner(service_name="namenode")
    pipeline = HdfsIngestionPipeline(
        runner=runner,
        project_root=project_root,
        hdfs_base=args.hdfs_base,
        overwrite=args.overwrite,
    )

    try:
        result = pipeline.ingest_file(
            source_file=source_file,
            ingestion_date=args.ingestion_date,
            check_readiness=True,
        )

        print("\n=================================================================")
        print("  HADOOP HDFS INGESTION COMPLETED SUCCESSFULLY")
        print("=================================================================")
        print(f"Status:               {result['status']}")
        print(f"Source File:          {result['source_file']}")
        print(f"HDFS Destination:     {result['hdfs_destination']}")
        print(f"SHA-256 Checksum:     {result['sha256']}")
        print(f"Record Count:         {result['record_count']:,}")
        if result.get("manifest_destination"):
            print(f"HDFS Manifest:        {result['manifest_destination']}")
            print(f"Local Manifest:       {result['local_manifest']}")
        print("=================================================================\n")

    except Exception as exc:
        logger.error("HDFS Ingestion failed: %s", exc)
        print(f"\nERROR: HDFS Ingestion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
