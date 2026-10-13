"""Authoritative schema definition and configuration fingerprinting for acoustic features.

Defines the exact, versioned column names, ordering, and data types for the
recording-level and segment-level tabular feature stores.

SCHEMA BREAKDOWN:
- METADATA_COLUMNS: 9 columns (uuid, label, status, demographics, audio_path)
- SEGMENTATION_FEATURE_COLUMNS: 11 columns (burst count, duration statistics, energy thresholds)
- SIGNAL_FEATURE_COLUMNS: 197 columns (MFCCs, Deltas, spectral moments, slope, ZCR, RMS, PSD bands)
- TOTAL ACOUSTIC FEATURES: 11 + 197 = 208 features
- TOTAL RECORDING TABLE WIDTH: 9 + 208 = 217 columns
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
import pyarrow as pa

SCHEMA_VERSION = "1.0.0"

# 1. Authoritative Metadata Columns (9 columns)
METADATA_COLUMNS: List[str] = [
    "uuid",
    "label",
    "status",
    "cough_detected",
    "age",
    "gender",
    "respiratory_condition",
    "fever_muscle_pain",
    "audio_path",
]

# 2. Authoritative Segmentation / Burst Summary Columns (11 columns)
SEGMENTATION_FEATURE_COLUMNS: List[str] = [
    "num_cough_segments",
    "total_cough_duration",
    "mean_cough_duration",
    "std_cough_duration",
    "min_cough_duration",
    "max_cough_duration",
    "peak_burst_energy",
    "mean_burst_energy",
    "segmenter_noise_floor",
    "segmenter_onset_threshold",
    "segmenter_offset_threshold",
]

DEFAULT_PSD_BANDS: List[tuple[int, int]] = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 5000),
    (5000, 6000),
]


def generate_signal_feature_names(
    n_mfcc: int = 20,
    psd_bands: Optional[Sequence[tuple[int, int]]] = None,
) -> List[str]:
    """Generate the exact ordered list of 197 signal-level acoustic feature names."""
    bands = psd_bands or DEFAULT_PSD_BANDS
    names: List[str] = []

    # 1. MFCCs and Deltas (8 stats * n_mfcc)
    for k in range(1, n_mfcc + 1):
        names.extend([
            f"mfcc_{k}_mean",
            f"mfcc_{k}_std",
            f"mfcc_{k}_min",
            f"mfcc_{k}_max",
            f"delta_mfcc_{k}_mean",
            f"delta_mfcc_{k}_std",
            f"delta2_mfcc_{k}_mean",
            f"delta2_mfcc_{k}_std",
        ])

    # 2. Spectral features (14 features)
    names.extend([
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_centroid_min",
        "spectral_centroid_max",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_bandwidth_min",
        "spectral_bandwidth_max",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "spectral_rolloff_min",
        "spectral_rolloff_max",
        "spectral_slope_mean",
        "spectral_slope_std",
    ])

    # 3. Time domain features (9 features)
    names.extend([
        "zcr_mean",
        "zcr_std",
        "zcr_min",
        "zcr_max",
        "rms_mean",
        "rms_std",
        "rms_min",
        "rms_max",
        "rms_peak",
    ])

    # 4. PSD Band energies (2 * len(bands))
    for f_low, f_high in bands:
        names.extend([
            f"psd_band_{f_low}_{f_high}_energy",
            f"psd_band_{f_low}_{f_high}_rel",
        ])

    return names


# Authoritative 197 signal features
SIGNAL_FEATURE_COLUMNS: List[str] = generate_signal_feature_names()

# Authoritative 208 acoustic features (11 segmentation + 197 signal)
ACOUSTIC_FEATURE_COLUMNS: List[str] = SEGMENTATION_FEATURE_COLUMNS + SIGNAL_FEATURE_COLUMNS

# Authoritative 217 recording-level columns (9 metadata + 208 acoustic)
RECORDING_FEATURE_COLUMNS: List[str] = METADATA_COLUMNS + ACOUSTIC_FEATURE_COLUMNS


def compute_config_fingerprint(config: Any) -> str:
    """Compute a deterministic SHA-256 fingerprint of pipeline feature configuration.

    Guarantees that any change in DSP filter cutoff, sample rate, burst segmentation
    thresholds, MFCC settings, or PSD bands produces a different fingerprint,
    preventing silent retention of stale features when resuming.
    """
    spec: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target_sr": getattr(config.audio, "target_sr", 12000),
        "lowpass_hz": getattr(config.audio, "lowpass_hz", 6000),
        "filter_order": getattr(config.audio, "filter_order", 5),
        "normalize_peak": getattr(config.audio, "normalize_peak", True),
        "min_burst_duration": getattr(config.segmentation, "min_burst_duration", 0.2),
        "max_burst_duration": getattr(config.segmentation, "max_burst_duration", 1.5),
        "power_threshold_factor": getattr(config.segmentation, "power_threshold_factor", 0.2),
        "pad_context_duration": getattr(config.segmentation, "pad_context_duration", 0.05),
        "n_mfcc": getattr(config.features, "n_mfcc", 20),
        "n_fft": getattr(config.features, "n_fft", 512),
        "hop_length": getattr(config.features, "hop_length", 128),
        "n_mels": getattr(config.features, "n_mels", 64),
        "spectrogram_frames": getattr(config.features, "spectrogram_frames", 128),
        "psd_bands": [list(b) for b in getattr(config.features, "psd_bands", DEFAULT_PSD_BANDS)],
    }
    encoded = json.dumps(spec, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def validate_features_dataframe(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms strictly to the authoritative recording schema.

    Raises
    ------
    ValueError
        If columns, ordering, or counts do not match the authoritative schema.
    """
    if len(df.columns) != len(RECORDING_FEATURE_COLUMNS):
        raise ValueError(
            f"Schema mismatch: expected {len(RECORDING_FEATURE_COLUMNS)} columns, got {len(df.columns)}."
        )

    expected_cols = list(RECORDING_FEATURE_COLUMNS)
    actual_cols = list(df.columns)

    if actual_cols != expected_cols:
        missing = set(expected_cols) - set(actual_cols)
        unexpected = set(actual_cols) - set(expected_cols)
        order_diff = [i for i, (e, a) in enumerate(zip(expected_cols, actual_cols)) if e != a]
        raise ValueError(
            f"Schema column mismatch or out of order. Missing: {missing}, Unexpected: {unexpected}, "
            f"First mismatched index: {order_diff[0] if order_diff else 'None'}"
        )


def build_pyarrow_schema() -> pa.Schema:
    """Build the strict PyArrow schema for recording-level features."""
    fields = [
        pa.field("uuid", pa.string(), nullable=False),
        pa.field("label", pa.int64(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("cough_detected", pa.float64(), nullable=True),
        pa.field("age", pa.float64(), nullable=True),
        pa.field("gender", pa.string(), nullable=True),
        pa.field("respiratory_condition", pa.bool_(), nullable=True),
        pa.field("fever_muscle_pain", pa.bool_(), nullable=True),
        pa.field("audio_path", pa.string(), nullable=False),
    ]

    # Segmentation burst summary features
    for col in SEGMENTATION_FEATURE_COLUMNS:
        if col == "num_cough_segments":
            fields.append(pa.field(col, pa.int32(), nullable=False))
        else:
            fields.append(pa.field(col, pa.float32(), nullable=False))

    # Signal acoustic features
    for col in SIGNAL_FEATURE_COLUMNS:
        fields.append(pa.field(col, pa.float32(), nullable=False))

    return pa.schema(fields)
