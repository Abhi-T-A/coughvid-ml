"""Metadata ingestion, schema validation, audio attachment, and data quality auditing.

Parses COUGHVID metadata CSV, validates records against typed schemas,
resolves audio paths on disk, and generates comprehensive data-quality reports.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class MetadataRecord(BaseModel):
    """Pydantic validated record for a single COUGHVID metadata entry."""
    uuid: str = Field(description="Primary UUID identifier for the recording.")
    datetime: Optional[str] = Field(default=None, description="ISO timestamp of recording.")
    cough_detected: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="XGBoost cough confidence score."
    )
    latitude: Optional[float] = Field(default=None, description="Rounded latitude.")
    longitude: Optional[float] = Field(default=None, description="Rounded longitude.")
    age: Optional[float] = Field(default=None, ge=0.0, le=120.0, description="Self-reported age.")
    gender: Optional[str] = Field(default=None, description="Self-reported gender.")
    respiratory_condition: Optional[bool] = Field(default=None, description="Pre-existing condition flag.")
    fever_muscle_pain: Optional[bool] = Field(default=None, description="Fever/muscle pain flag.")
    status: Optional[str] = Field(default=None, description="Self-reported COVID status.")
    status_SSL: Optional[str] = Field(default=None, description="Semi-supervised v3 label.")
    diagnosis_1: Optional[str] = Field(default=None, description="Expert physician 1 diagnosis.")
    quality_1: Optional[str] = Field(default=None, description="Expert physician 1 audio quality.")
    cough_type_1: Optional[str] = Field(default=None, description="Expert physician 1 cough type.")
    audio_path: Optional[str] = Field(default=None, description="Path to audio file on disk.")

    @field_validator("cough_detected", mode="before")
    @classmethod
    def parse_float_fields(cls, v: Any) -> Optional[float]:
        if v is None or v == "" or str(v).strip().lower() == "nan":
            return None
        return float(v)

    @field_validator("latitude", "longitude", "age", mode="before")
    @classmethod
    def parse_optional_floats(cls, v: Any) -> Optional[float]:
        if v is None or v == "" or str(v).strip().lower() == "nan":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    @field_validator("respiratory_condition", "fever_muscle_pain", mode="before")
    @classmethod
    def parse_optional_bools(cls, v: Any) -> Optional[bool]:
        if v is None or v == "" or str(v).strip().lower() in ("nan", "none"):
            return None
        s = str(v).strip().lower()
        if s in ("true", "1", "t", "yes"):
            return True
        if s in ("false", "0", "f", "no"):
            return False
        return None

    @field_validator("status", "status_SSL", "diagnosis_1", "quality_1", "cough_type_1", mode="before")
    @classmethod
    def parse_optional_strings(cls, v: Any) -> Optional[str]:
        if v is None or v == "" or str(v).strip().lower() == "nan":
            return None
        return str(v).strip()


@dataclass
class MetadataAudit:
    """Structured audit metrics for raw or filtered metadata."""
    total_records: int
    unique_uuids: int
    duplicate_uuids: int
    missing_status_count: int
    missing_status_pct: float
    missing_cough_detected_count: int
    missing_cough_detected_pct: float
    missing_audio_count: int
    status_distribution: Dict[str, int]
    cough_detected_stats: Dict[str, float]
    quality_columns_present: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_audio_path_for_uuid(uuid_stem: str, data_dir: Path) -> Optional[Path]:
    """Search for the audio file corresponding to uuid_stem (.webm, .wav, .ogg)."""
    for ext in ("webm", "wav", "ogg"):
        p = data_dir / f"{uuid_stem}.{ext}"
        if p.exists():
            return p
    return None


def load_raw_metadata(
    csv_path: str | Path,
    raw_data_dir: Optional[str | Path] = None,
) -> list[dict]:
    """Read metadata CSV and return a list of parsed dictionaries.

    Parameters
    ----------
    csv_path : Path or str
        Path to metadata CSV (e.g. metadata_compiled.csv).
    raw_data_dir : Path or str, optional
        Directory where audio files reside. If provided, resolves and attaches
        'audio_path' and 'has_audio' to each row.

    Returns
    -------
    list[dict]
        Parsed rows with cleaned fields and audio paths attached.
    """
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    search_dir = Path(raw_data_dir).resolve() if raw_data_dir else path.parent

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            # Clean empty string index column if present
            row = {k: v for k, v in raw_row.items() if k != ""}
            uuid_str = row.get("uuid", "").strip()
            if not uuid_str:
                continue

            # Attach audio path if search directory exists
            if search_dir and search_dir.exists():
                audio_p = resolve_audio_path_for_uuid(uuid_str, search_dir)
                row["audio_path"] = str(audio_p) if audio_p else None
                row["has_audio"] = audio_p is not None
            else:
                row["audio_path"] = None
                row["has_audio"] = False

            rows.append(row)

    return rows


def audit_metadata(rows: list[dict]) -> MetadataAudit:
    """Generate a comprehensive data-quality audit report from parsed metadata rows.

    Parameters
    ----------
    rows : list[dict]
        Parsed metadata rows.

    Returns
    -------
    MetadataAudit
        Structured audit report.
    """
    total = len(rows)
    uuids = [r.get("uuid", "") for r in rows]
    unique_uuids = len(set(uuids))
    duplicates = total - unique_uuids

    status_dist: Dict[str, int] = {}
    missing_status = 0
    missing_cough = 0
    missing_audio = 0
    cough_scores: list[float] = []

    for r in rows:
        st = r.get("status")
        if not st or st == "" or str(st).lower() == "nan":
            missing_status += 1
            status_dist["NaN"] = status_dist.get("NaN", 0) + 1
        else:
            status_dist[st] = status_dist.get(st, 0) + 1

        cd = r.get("cough_detected")
        if cd is None or cd == "" or str(cd).lower() == "nan":
            missing_cough += 1
        else:
            try:
                score = float(cd)
                cough_scores.append(score)
            except ValueError:
                missing_cough += 1

        if not r.get("has_audio", False):
            missing_audio += 1

    cough_stats = {
        "valid_count": float(len(cough_scores)),
        "mean": float(sum(cough_scores) / len(cough_scores)) if cough_scores else 0.0,
        "min": float(min(cough_scores)) if cough_scores else 0.0,
        "max": float(max(cough_scores)) if cough_scores else 0.0,
        "ge_0_8_count": float(sum(1 for s in cough_scores if s >= 0.8)),
    }

    sample_keys = list(rows[0].keys()) if rows else []

    return MetadataAudit(
        total_records=total,
        unique_uuids=unique_uuids,
        duplicate_uuids=duplicates,
        missing_status_count=missing_status,
        missing_status_pct=(missing_status / total * 100.0) if total > 0 else 0.0,
        missing_cough_detected_count=missing_cough,
        missing_cough_detected_pct=(missing_cough / total * 100.0) if total > 0 else 0.0,
        missing_audio_count=missing_audio,
        status_distribution=status_dist,
        cough_detected_stats=cough_stats,
        quality_columns_present=[k for k in sample_keys if "quality" in k or "diagnosis" in k],
    )
