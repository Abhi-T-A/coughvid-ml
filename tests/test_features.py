"""Unit tests for acoustic feature extraction, Log-Mel spectrograms, and Parquet caching.

Validates:
- Valid waveform feature extraction.
- Short and silent segments.
- Empty and non-finite inputs.
- Consistent feature names and dimensions.
- Deterministic feature extraction.
- Fixed-size Log-Mel output and normalization.
- Feature extraction failure and exclusion handling.
- UUID and label preservation.
- Recording-level aggregation without leakage.
- Parquet write/read round trips using PyArrow.
- Resumable extraction without duplicate records.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.io import wavfile

from coughvid.config import AppConfig, FeaturesConfig
from coughvid.features.extractor import (
    BatchFeatureExtractor,
    ConfigurationMismatchError,
    extract_recording_features,
    load_spectrogram,
)
from coughvid.features.schema import (
    ACOUSTIC_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    RECORDING_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    SEGMENTATION_FEATURE_COLUMNS,
    SIGNAL_FEATURE_COLUMNS,
    compute_config_fingerprint,
    validate_features_dataframe,
)
from coughvid.features.spectrogram import compute_log_mel_spectrogram
from coughvid.features.tabular import (
    build_mel_filterbank,
    compute_deltas,
    compute_mfcc_features,
    compute_psd_band_features,
    compute_spectral_features,
    compute_time_domain_features,
    extract_tabular_features,
)


@pytest.fixture
def synthetic_cough_audio() -> tuple[np.ndarray, int]:
    """Generate a 1-second synthetic acoustic signal with cough-like spectral distribution."""
    sr = 12000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # Fundamental vocal tone + harmonics + shaped noise
    tone = 0.4 * np.sin(2 * np.pi * 320 * t) + 0.2 * np.sin(2 * np.pi * 640 * t)
    np.random.seed(42)
    noise = 0.2 * np.random.randn(sr).astype(np.float32)
    # Exponential decay envelope characteristic of an expiratory cough burst
    envelope = np.exp(-4.0 * t).astype(np.float32)
    waveform = (tone + noise) * envelope
    return waveform, sr


def test_tabular_features_valid_waveform(synthetic_cough_audio):
    """Verify feature extraction produces complete suite of finite acoustic features."""
    y, sr = synthetic_cough_audio
    feats = extract_tabular_features(y, sr)

    assert isinstance(feats, dict)
    assert len(feats) >= 190  # 197 acoustic features

    # Verify all feature values are strictly finite numbers
    for k, v in feats.items():
        assert isinstance(v, float)
        assert np.isfinite(v), f"Feature '{k}' is non-finite: {v}"

    # Verify presence of all required feature categories
    assert "mfcc_1_mean" in feats
    assert "mfcc_20_std" in feats
    assert "delta_mfcc_1_mean" in feats
    assert "delta2_mfcc_1_mean" in feats
    assert "spectral_centroid_mean" in feats
    assert "spectral_bandwidth_mean" in feats
    assert "spectral_rolloff_mean" in feats
    assert "spectral_slope_mean" in feats
    assert "zcr_mean" in feats
    assert "rms_mean" in feats
    assert "rms_peak" in feats
    assert "psd_band_0_500_energy" in feats
    assert "psd_band_5000_6000_rel" in feats


def test_tabular_features_short_and_silent_segments():
    """Verify safe handling of short segments and absolute silence."""
    sr = 12000

    # 1. Short segment (50 samples, much shorter than n_fft=512)
    y_short = np.array([0.1, -0.2, 0.3, -0.1] * 12, dtype=np.float32)
    feats_short = extract_tabular_features(y_short, sr)
    assert len(feats_short) >= 190
    assert all(np.isfinite(v) for v in feats_short.values())

    # 2. Complete silence (all zeros)
    y_silent = np.zeros(2400, dtype=np.float32)
    feats_silent = extract_tabular_features(y_silent, sr)
    assert len(feats_silent) >= 190
    assert all(np.isfinite(v) for v in feats_silent.values())
    assert feats_silent["rms_mean"] == 0.0
    assert feats_silent["rms_peak"] == 0.0


def test_tabular_features_empty_and_non_finite_inputs():
    """Verify explicit error handling for invalid or corrupt inputs."""
    sr = 12000

    # Non-finite values
    y_nan = np.array([0.1, np.nan, 0.2], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        extract_tabular_features(y_nan, sr)

    y_inf = np.array([0.1, np.inf, 0.2], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        extract_tabular_features(y_inf, sr)

    # Invalid dimension
    y_2d = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="1D"):
        extract_tabular_features(y_2d, sr)

    # Invalid sample rate
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        extract_tabular_features(np.ones(100, dtype=np.float32), sr=0)


def test_tabular_features_deterministic(synthetic_cough_audio):
    """Verify identical inputs yield 100% identical feature outputs."""
    y, sr = synthetic_cough_audio
    feats_1 = extract_tabular_features(y, sr)
    feats_2 = extract_tabular_features(y, sr)

    assert set(feats_1.keys()) == set(feats_2.keys())
    for k in feats_1:
        assert feats_1[k] == feats_2[k], f"Feature {k} differed between runs."


def test_tabular_features_consistent_names_and_dimensions():
    """Verify feature names and dimensionality remain identical regardless of audio length."""
    sr = 12000
    y_short = np.sin(np.linspace(0, 10, 600)).astype(np.float32)
    y_long = np.sin(np.linspace(0, 50, 18000)).astype(np.float32)

    feats_short = extract_tabular_features(y_short, sr)
    feats_long = extract_tabular_features(y_long, sr)

    assert list(feats_short.keys()) == list(feats_long.keys())
    assert len(feats_short) == len(feats_long)


def test_log_mel_spectrogram_fixed_shape_and_dtype(synthetic_cough_audio):
    """Verify fixed-shape 2D tensor output and configurable parameters."""
    y, sr = synthetic_cough_audio

    # Standard configuration
    spec = compute_log_mel_spectrogram(
        y, sr=sr, n_mels=64, n_fft=512, hop_length=128, target_frames=128, normalize="db"
    )
    assert spec.shape == (64, 128)
    assert spec.dtype == np.float32
    assert np.all(np.isfinite(spec))
    # Normalized dB values should be bounded in [-1.0, 1.0]
    assert float(spec.min()) >= -1.0 - 1e-5
    assert float(spec.max()) <= 1.0 + 1e-5

    # Short audio (padded to target frames)
    y_short = y[:400]
    spec_short = compute_log_mel_spectrogram(
        y_short, sr=sr, n_mels=32, target_frames=64, normalize="standard"
    )
    assert spec_short.shape == (32, 64)
    assert np.all(np.isfinite(spec_short))

    # Long audio (cropped to target frames)
    y_long = np.tile(y, 4)
    spec_long = compute_log_mel_spectrogram(
        y_long, sr=sr, n_mels=40, target_frames=96, crop_mode="center"
    )
    assert spec_long.shape == (40, 96)
    assert np.all(np.isfinite(spec_long))


def test_log_mel_spectrogram_silent_and_invalid():
    """Verify graceful handling of silence and rejection of empty/non-finite signals."""
    sr = 12000

    # Silent audio returns finite floor without crashing or division-by-zero
    y_silent = np.zeros(6000, dtype=np.float32)
    spec_silent = compute_log_mel_spectrogram(y_silent, sr=sr, n_mels=64, target_frames=128)
    assert spec_silent.shape == (64, 128)
    assert np.all(np.isfinite(spec_silent))
    assert np.all(spec_silent == -1.0)

    # Empty audio
    with pytest.raises(ValueError, match="empty"):
        compute_log_mel_spectrogram(np.array([], dtype=np.float32), sr=sr)

    # Non-finite audio
    with pytest.raises(ValueError, match="non-finite"):
        compute_log_mel_spectrogram(np.array([0.1, np.nan, 0.3], dtype=np.float32), sr=sr)


def _create_mock_manifest_and_audio(
    tmp_path: Path,
    records: List[Dict[str, Any]],
) -> tuple[Path, Path]:
    """Helper to write mock WAV files and a CSV manifest for end-to-end testing."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "test_manifest.csv"

    sr = 12000
    manifest_rows = []

    for r in records:
        uuid = r["uuid"]
        wav_path = audio_dir / f"{uuid}.wav"

        # Generate audio based on record spec
        if r.get("is_silent"):
            audio = np.zeros(sr * 2, dtype=np.float32)
        elif r.get("has_cough"):
            t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
            audio = np.zeros_like(t, dtype=np.float32)
            # Create 2 distinct energy bursts at t=0.2s and t=0.8s
            burst1_mask = (t >= 0.2) & (t <= 0.6)
            burst2_mask = (t >= 0.8) & (t <= 1.2)
            audio[burst1_mask] = 0.8 * np.sin(2 * np.pi * 400 * t[burst1_mask])
            audio[burst2_mask] = 0.7 * np.sin(2 * np.pi * 500 * t[burst2_mask])
        else:
            audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr)).astype(np.float32)

        int16_audio = (audio * 32767.0).astype(np.int16)
        wavfile.write(str(wav_path), sr, int16_audio)

        row = {
            "uuid": uuid,
            "label": r.get("label", 0),
            "status": r.get("status", "healthy"),
            "cough_detected": r.get("cough_detected", 0.95),
            "age": r.get("age", 35.0),
            "gender": r.get("gender", "male"),
            "respiratory_condition": r.get("respiratory_condition", False),
            "fever_muscle_pain": r.get("fever_muscle_pain", False),
            "audio_path": str(wav_path),
            "has_audio": True,
        }
        manifest_rows.append(row)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(manifest_path.open("w"), fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path, audio_dir


def test_batch_extractor_uuid_and_label_preservation(tmp_path):
    """Verify that UUIDs and classification labels are preserved through extraction."""
    mock_records = [
        {"uuid": "rec-001", "label": 0, "status": "healthy", "has_cough": True},
        {"uuid": "rec-002", "label": 1, "status": "COVID-19", "has_cough": True},
    ]
    manifest_p, audio_dir = _create_mock_manifest_and_audio(tmp_path, mock_records)

    cfg = AppConfig()
    extractor = BatchFeatureExtractor(
        config=cfg,
        output_dir=tmp_path / "features",
        reports_dir=tmp_path / "reports",
    )

    summary = extractor.process_manifest(
        manifest_path=manifest_p,
        partition_name="unit_test",
        overwrite=True,
    )

    assert summary.success_count == 2
    assert summary.exclusion_count == 0

    # Inspect parquet table
    pq_path = tmp_path / "features" / "unit_test_features.parquet"
    assert pq_path.exists()
    table = pq.read_table(pq_path)
    df = table.to_pandas()

    assert len(df) == 2
    assert set(df["uuid"]) == {"rec-001", "rec-002"}
    assert df.loc[df["uuid"] == "rec-001", "label"].iloc[0] == 0
    assert df.loc[df["uuid"] == "rec-002", "label"].iloc[0] == 1


def test_batch_extractor_recording_level_aggregation(tmp_path):
    """Verify that multiple candidate bursts are aggregated into a single recording-level row."""
    mock_records = [
        {"uuid": "multi-burst-uuid", "label": 1, "status": "COVID-19", "has_cough": True},
    ]
    manifest_p, _ = _create_mock_manifest_and_audio(tmp_path, mock_records)

    extractor = BatchFeatureExtractor(
        output_dir=tmp_path / "features",
        reports_dir=tmp_path / "reports",
    )

    summary = extractor.process_manifest(
        manifest_path=manifest_p,
        partition_name="agg_test",
        overwrite=True,
    )

    assert summary.success_count == 1

    df = pq.read_table(tmp_path / "features" / "agg_test_features.parquet").to_pandas()
    assert len(df) == 1
    assert df["uuid"].iloc[0] == "multi-burst-uuid"
    assert df["num_cough_segments"].iloc[0] >= 1
    assert df["total_cough_duration"].iloc[0] > 0.0

    # Verify separate spectrogram storage
    spec_file = tmp_path / "features" / "agg_test_spectrograms.npz"
    assert spec_file.exists()
    with np.load(spec_file) as npz:
        assert "multi-burst-uuid" in npz
        assert npz["multi-burst-uuid"].shape == (64, 128)


def test_batch_extractor_handles_no_segments_safely(tmp_path):
    """Verify that silent recordings with no detected coughs are excluded and logged."""
    mock_records = [
        {"uuid": "silent-record", "label": 1, "status": "COVID-19", "is_silent": True},
    ]
    manifest_p, _ = _create_mock_manifest_and_audio(tmp_path, mock_records)

    extractor = BatchFeatureExtractor(
        output_dir=tmp_path / "features",
        reports_dir=tmp_path / "reports",
    )

    summary = extractor.process_manifest(
        manifest_path=manifest_p,
        partition_name="silent_test",
        overwrite=True,
    )

    assert summary.success_count == 0
    assert summary.exclusion_count == 1
    assert "NoSegmentsDetectedError" in summary.failure_reasons

    # Check error manifest
    err_json = tmp_path / "reports" / "silent_test_extraction_errors.json"
    assert err_json.exists()
    with open(err_json, "r", encoding="utf-8") as f:
        errors = json.load(f)
    assert len(errors) == 1
    assert errors[0]["uuid"] == "silent-record"
    assert errors[0]["error_type"] == "NoSegmentsDetectedError"
    # Verify label was preserved, not overwritten to healthy
    assert errors[0]["label"] == 1


def test_parquet_write_read_roundtrip(tmp_path, synthetic_cough_audio):
    """Verify PyArrow Parquet serialization and deserialization integrity."""
    y, sr = synthetic_cough_audio
    feats = extract_tabular_features(y, sr)

    row = {
        "uuid": "roundtrip-uuid",
        "label": 1,
        "status": "COVID-19",
        "num_cough_segments": 1,
        "total_cough_duration": 0.5,
    }
    row.update(feats)

    df_orig = pd.DataFrame([row])
    table = pa.Table.from_pandas(df_orig, preserve_index=False)

    pq_path = tmp_path / "roundtrip.parquet"
    pq.write_table(table, pq_path)

    table_read = pq.read_table(pq_path)
    df_read = table_read.to_pandas()

    assert len(df_read) == 1
    assert df_read["uuid"].iloc[0] == "roundtrip-uuid"
    assert df_read["label"].iloc[0] == 1
    for k in feats:
        np.testing.assert_allclose(df_read[k].iloc[0], feats[k], rtol=1e-5)


def test_resumable_extraction_without_duplicates(tmp_path):
    """Verify that running extraction multiple times skips existing rows and prevents duplicates."""
    initial_records = [
        {"uuid": "res-001", "label": 0, "has_cough": True},
        {"uuid": "res-002", "label": 1, "has_cough": True},
    ]
    manifest_p, audio_dir = _create_mock_manifest_and_audio(tmp_path, initial_records)

    extractor = BatchFeatureExtractor(
        output_dir=tmp_path / "features",
        reports_dir=tmp_path / "reports",
    )

    # Initial extraction
    summary_1 = extractor.process_manifest(
        manifest_path=manifest_p,
        partition_name="resume_test",
        overwrite=False,
    )
    assert summary_1.success_count == 2

    # Add 3rd record to manifest
    sr = 12000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    audio = np.zeros_like(t, dtype=np.float32)
    burst_mask = (t >= 0.2) & (t <= 0.6)
    audio[burst_mask] = 0.8 * np.sin(2 * np.pi * 400 * t[burst_mask])
    wavfile.write(str(audio_dir / "res-003.wav"), sr, (audio * 32767.0).astype(np.int16))

    with open(manifest_p, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["res-003", 0, "healthy", 0.9, 40.0, "female", False, False, str(audio_dir / "res-003.wav"), True])

    # Second extraction without overwrite (resumes)
    summary_2 = extractor.process_manifest(
        manifest_path=manifest_p,
        partition_name="resume_test",
        overwrite=False,
    )

    assert summary_2.success_count == 3
    df = pq.read_table(tmp_path / "features" / "resume_test_features.parquet").to_pandas()
    assert len(df) == 3
    assert set(df["uuid"]) == {"res-001", "res-002", "res-003"}
    # Verify no duplicates
    assert len(df["uuid"].unique()) == len(df)


def test_authoritative_feature_schema_counts_and_ordering(synthetic_cough_audio):
    """Verify exact versioned schema column counts, composition, and signal feature match."""
    assert SCHEMA_VERSION == "1.0.0"
    assert len(METADATA_COLUMNS) == 9
    assert len(SEGMENTATION_FEATURE_COLUMNS) == 11
    assert len(SIGNAL_FEATURE_COLUMNS) == 197
    assert len(ACOUSTIC_FEATURE_COLUMNS) == 208
    assert len(RECORDING_FEATURE_COLUMNS) == 217

    # Verify that extract_tabular_features output keys strictly match SIGNAL_FEATURE_COLUMNS
    y, sr = synthetic_cough_audio
    feats = extract_tabular_features(y, sr)
    assert list(feats.keys()) == SIGNAL_FEATURE_COLUMNS


def test_validate_features_dataframe_schema_guard():
    """Verify schema validator accepts valid DataFrames and rejects missing or misordered columns."""
    valid_data = {col: [1.0] if col not in ("uuid", "status", "gender", "audio_path") else ["mock"] for col in RECORDING_FEATURE_COLUMNS}
    valid_df = pd.DataFrame(valid_data)
    # Must pass without error
    validate_features_dataframe(valid_df)

    # Missing column
    invalid_missing = valid_df.drop(columns=["spectral_centroid_mean"])
    with pytest.raises(ValueError, match="Schema mismatch"):
        validate_features_dataframe(invalid_missing)

    # Misordered columns
    cols = list(valid_df.columns)
    cols[0], cols[1] = cols[1], cols[0]
    misordered_df = valid_df[cols]
    with pytest.raises(ValueError, match="out of order"):
        validate_features_dataframe(misordered_df)


def test_config_fingerprint_generation_and_mutation():
    """Verify deterministic fingerprinting and mutation sensitivity."""
    cfg = AppConfig()
    fp_base = compute_config_fingerprint(cfg)
    assert isinstance(fp_base, str)
    assert len(fp_base) == 16

    # Repeated calls are identical
    assert compute_config_fingerprint(cfg) == fp_base

    # Mutation of audio parameters changes fingerprint
    cfg_audio = AppConfig()
    cfg_audio.audio.lowpass_hz = 5000
    assert compute_config_fingerprint(cfg_audio) != fp_base

    # Mutation of segmentation parameters changes fingerprint
    cfg_seg = AppConfig()
    cfg_seg.segmentation.min_burst_duration = 0.25
    assert compute_config_fingerprint(cfg_seg) != fp_base

    # Mutation of feature parameters changes fingerprint
    cfg_feat = AppConfig()
    cfg_feat.features.n_mfcc = 13
    assert compute_config_fingerprint(cfg_feat) != fp_base


def test_batch_extractor_enforces_config_fingerprint_on_resume(tmp_path):
    """Verify that attempting to resume with an altered config raises ConfigurationMismatchError."""
    records = [{"uuid": "fp-001", "label": 0, "has_cough": True}]
    manifest_p, _ = _create_mock_manifest_and_audio(tmp_path, records)

    cfg1 = AppConfig()
    extractor1 = BatchFeatureExtractor(config=cfg1, output_dir=tmp_path / "features", reports_dir=tmp_path / "reports")
    extractor1.process_manifest(manifest_p, partition_name="fp_test", overwrite=True)

    # Attempt to resume with altered lowpass cutoff
    cfg2 = AppConfig()
    cfg2.audio.lowpass_hz = 5200
    extractor2 = BatchFeatureExtractor(config=cfg2, output_dir=tmp_path / "features", reports_dir=tmp_path / "reports")

    with pytest.raises(ConfigurationMismatchError, match="Existing cache was generated with configuration fingerprint"):
        extractor2.process_manifest(manifest_p, partition_name="fp_test", overwrite=False)


def test_load_spectrogram_by_uuid(tmp_path):
    """Verify direct UUID-based spectrogram retrieval from NPZ archive."""
    records = [{"uuid": "spec-uuid-1", "label": 1, "has_cough": True}]
    manifest_p, _ = _create_mock_manifest_and_audio(tmp_path, records)

    extractor = BatchFeatureExtractor(output_dir=tmp_path / "features", reports_dir=tmp_path / "reports")
    summary = extractor.process_manifest(manifest_p, partition_name="spec_test", overwrite=True)

    spec_archive = Path(summary.spectrogram_path)
    spec = load_spectrogram(spec_archive, "spec-uuid-1")
    assert spec.shape == (64, 128)
    assert spec.dtype == np.float32

    # Non-existent UUID raises KeyError
    with pytest.raises(KeyError, match="not found"):
        load_spectrogram(spec_archive, "non-existent-uuid")

    # Non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_spectrogram(tmp_path / "missing.npz", "spec-uuid-1")


def test_summary_class_accounting_and_feature_counts(tmp_path):
    """Verify extraction summary accurately accounts for class-specific success/failure rates."""
    records = [
        {"uuid": "acc-h1", "label": 0, "status": "healthy", "has_cough": True},
        {"uuid": "acc-h2", "label": 0, "status": "healthy", "is_silent": True},  # will fail segmentation
        {"uuid": "acc-c1", "label": 1, "status": "COVID-19", "has_cough": True},
    ]
    manifest_p, _ = _create_mock_manifest_and_audio(tmp_path, records)

    extractor = BatchFeatureExtractor(output_dir=tmp_path / "features", reports_dir=tmp_path / "reports")
    summary = extractor.process_manifest(manifest_p, partition_name="summary_test", overwrite=True)

    assert summary.processed_records == 3
    assert summary.success_count == 2
    assert summary.exclusion_count == 1
    assert summary.num_tabular_features == 208
    assert summary.total_columns == 217
    assert summary.class_distribution_input == {"0": 2, "1": 1}
    assert summary.class_distribution_success == {"0": 1, "1": 1}
    assert summary.class_distribution_excluded == {"0": 1, "1": 0}
    assert summary.class_success_rates == {"0": 0.5, "1": 1.0}

