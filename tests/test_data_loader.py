"""Unit tests for metadata loader, schema validation, and task working set filtering."""

import csv
from pathlib import Path
import pytest
from pydantic import ValidationError

from coughvid.data.loader import (
    MetadataAudit,
    MetadataRecord,
    audit_metadata,
    load_raw_metadata,
)
from coughvid.data.tasks import WorkingSet, build_working_set


@pytest.fixture
def synthetic_metadata_csv(tmp_path) -> Path:
    """Create a synthetic metadata CSV with known distributions and edge cases."""
    rows = [
        {"": "0", "uuid": "u1", "cough_detected": "0.95", "status": "healthy", "age": "30", "gender": "male", "latitude": "45.0", "longitude": "8.0", "diagnosis_1": "healthy_cough", "cough_type_1": "dry"},
        {"": "1", "uuid": "u2", "cough_detected": "0.85", "status": "COVID-19", "age": "45", "gender": "female", "latitude": "45.0", "longitude": "8.0", "diagnosis_1": "COVID-19", "cough_type_1": "wet"},
        {"": "2", "uuid": "u3", "cough_detected": "0.90", "status": "symptomatic", "age": "28", "gender": "male", "latitude": "40.0", "longitude": "2.0", "diagnosis_1": "", "cough_type_1": "dry"},
        {"": "3", "uuid": "u4", "cough_detected": "0.30", "status": "COVID-19", "age": "50", "gender": "male", "latitude": "40.0", "longitude": "2.0", "diagnosis_1": "COVID-19", "cough_type_1": "wet"},  # Low cough score
        {"": "4", "uuid": "u5", "cough_detected": "0.88", "status": "", "age": "60", "gender": "female", "latitude": "42.0", "longitude": "1.0", "diagnosis_1": "", "cough_type_1": ""},  # Missing status
        {"": "5", "uuid": "u6", "cough_detected": "0.82", "status": "healthy", "age": "35", "gender": "male", "latitude": "45.0", "longitude": "8.0", "diagnosis_1": "healthy_cough", "cough_type_1": "dry"},
        {"": "6", "uuid": "u7", "cough_detected": "0.91", "status": "COVID-19", "age": "40", "gender": "female", "latitude": "38.0", "longitude": "15.0", "diagnosis_1": "COVID-19", "cough_type_1": "wet"},
    ]
    csv_file = tmp_path / "test_metadata.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_file


def test_metadata_record_valid_schema():
    """Verify MetadataRecord parses valid fields and converts types correctly."""
    rec = MetadataRecord(
        uuid="test-uuid-1",
        cough_detected="0.875",
        age="32.0",
        respiratory_condition="True",
        status="healthy",
    )
    assert rec.uuid == "test-uuid-1"
    assert rec.cough_detected == 0.875
    assert rec.age == 32.0
    assert rec.respiratory_condition is True
    assert rec.status == "healthy"


def test_metadata_record_rejects_invalid_bounds():
    """Verify MetadataRecord rejects cough_detected > 1.0 or age > 120."""
    with pytest.raises(ValidationError):
        MetadataRecord(uuid="bad-cough", cough_detected=1.5)

    with pytest.raises(ValidationError):
        MetadataRecord(uuid="bad-age", age=150.0)


def test_load_raw_metadata(synthetic_metadata_csv):
    """Verify load_raw_metadata correctly parses rows and ignores index column."""
    rows = load_raw_metadata(synthetic_metadata_csv)
    assert len(rows) == 7
    assert rows[0]["uuid"] == "u1"
    assert "" not in rows[0]  # Index column removed


def test_audit_metadata(synthetic_metadata_csv):
    """Verify audit_metadata accurately summarizes counts, missing values, and status distribution."""
    rows = load_raw_metadata(synthetic_metadata_csv)
    audit = audit_metadata(rows)

    assert isinstance(audit, MetadataAudit)
    assert audit.total_records == 7
    assert audit.unique_uuids == 7
    assert audit.duplicate_uuids == 0
    assert audit.missing_status_count == 1
    assert audit.status_distribution["healthy"] == 2
    assert audit.status_distribution["COVID-19"] == 3
    assert audit.status_distribution["symptomatic"] == 1
    assert audit.status_distribution["NaN"] == 1
    assert audit.cough_detected_stats["ge_0_8_count"] == 6


def test_build_working_set_covid_binary(synthetic_metadata_csv):
    """Verify build_working_set filters for high cough confidence and strictly binary status."""
    rows = load_raw_metadata(synthetic_metadata_csv)
    # Synthetic records do not have actual audio files, so require_audio=False for unit test
    ws = build_working_set(rows, task="covid_binary", cough_threshold=0.8, require_audio=False)

    assert isinstance(ws, WorkingSet)
    assert ws.task_name == "covid_binary"

    # In synthetic data:
    # u1: healthy, 0.95 -> KEPT (label=0)
    # u2: COVID-19, 0.85 -> KEPT (label=1)
    # u3: symptomatic -> DROPPED (ambiguous)
    # u4: low score 0.30 -> DROPPED
    # u5: missing status -> DROPPED
    # u6: healthy, 0.82 -> KEPT (label=0)
    # u7: COVID-19, 0.91 -> KEPT (label=1)
    assert ws.num_samples == 4
    assert ws.class_counts == {0: 2, 1: 2}

    # Verify attrition log
    log = ws.attrition_log
    assert log["dropped_low_cough_score"] == 1  # u4
    assert log["dropped_symptomatic_ambiguous"] == 1  # u3
    assert log["dropped_missing_label"] == 1  # u5
    assert log["final_working_set"] == 4


def test_build_working_set_never_maps_missing_to_healthy(synthetic_metadata_csv):
    """Verify that records with missing or symptomatic status are NEVER mapped to healthy."""
    rows = load_raw_metadata(synthetic_metadata_csv)
    ws = build_working_set(rows, task="covid_binary", cough_threshold=0.0, require_audio=False)

    uuids = ws.get_uuids()
    assert "u3" not in uuids  # symptomatic must NOT be included
    assert "u5" not in uuids  # missing status must NOT be included


def test_build_working_set_unknown_task_raises(synthetic_metadata_csv):
    """Verify requesting an unsupported task raises ValueError."""
    rows = load_raw_metadata(synthetic_metadata_csv)
    with pytest.raises(ValueError, match="Unknown classification task"):
        build_working_set(rows, task="unknown_prediction_task", require_audio=False)
