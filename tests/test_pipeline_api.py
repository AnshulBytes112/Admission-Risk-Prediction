"""Unit tests for Phase 8 / Big Data Pipeline diagnostic endpoints."""
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_pipeline_status():
    response = client.get("/pipeline/status")
    assert response.status_code == 200
    data = response.json()
    assert "components" in data
    assert "kafka" in data["components"]
    assert "hdfs" in data["components"]
    assert "spark" in data["components"]
    assert "parquet" in data["components"]
    assert "model" in data["components"]
    assert "fastapi" in data["components"]
    assert data["components"]["fastapi"]["status"] == "online"
    assert "streaming_storage" in data


def test_pipeline_kafka():
    response = client.get("/pipeline/kafka")
    assert response.status_code == 200
    data = response.json()
    assert data["topic"] == "patient-observations"
    assert data["consumer_group"] == "emergency-admission-bda-consumer"
    assert "broker_probe" in data
    assert "streamed_events_archive" in data


def test_pipeline_hdfs():
    response = client.get("/pipeline/hdfs")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "hadoop-hdfs"
    assert "configured_paths" in data
    assert "/emergency-admission-bda/raw/kafka-events" in data["configured_paths"]["raw"]


def test_pipeline_spark_latest():
    response = client.get("/pipeline/spark/latest")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert "manifest" in data
        assert data["service"] == "pyspark-batch-processing"


def test_pipeline_parquet():
    response = client.get("/pipeline/parquet")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert "partition_column" in data
        assert data["partition_column"] == "observation_date"


def test_pipeline_manifests():
    response = client.get("/pipeline/manifests")
    assert response.status_code == 200
    data = response.json()
    assert "ingestion_manifests" in data
    assert "spark_manifests" in data
