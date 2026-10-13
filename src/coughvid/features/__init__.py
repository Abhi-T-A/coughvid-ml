"""Acoustic feature extraction (segmented handcrafted features, spectrograms, schema, and batch extraction) module."""

from coughvid.features.extractor import (
    BatchFeatureExtractor,
    ConfigurationMismatchError,
    ExtractionErrorRecord,
    ExtractionSummary,
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
    compute_mfcc_features,
    compute_psd_band_features,
    compute_spectral_features,
    compute_time_domain_features,
    extract_tabular_features,
)

__all__ = [
    "BatchFeatureExtractor",
    "ConfigurationMismatchError",
    "ExtractionErrorRecord",
    "ExtractionSummary",
    "extract_recording_features",
    "load_spectrogram",
    "compute_log_mel_spectrogram",
    "extract_tabular_features",
    "compute_mfcc_features",
    "compute_spectral_features",
    "compute_time_domain_features",
    "compute_psd_band_features",
    "SCHEMA_VERSION",
    "METADATA_COLUMNS",
    "SEGMENTATION_FEATURE_COLUMNS",
    "SIGNAL_FEATURE_COLUMNS",
    "ACOUSTIC_FEATURE_COLUMNS",
    "RECORDING_FEATURE_COLUMNS",
    "compute_config_fingerprint",
    "validate_features_dataframe",
]
