"""Pilot real-data validation runner for Milestone 4 acoustic feature extraction.

Executes feature extraction on a small configurable sample from data/splits/train.csv,
validating audio decoding, candidate segmentation, feature dimensions, Parquet integrity,
schema compliance, NaN/infinity checks, UUID/label preservation, configuration fingerprinting,
and error recording.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coughvid.config import load_config
from coughvid.features.extractor import BatchFeatureExtractor, ConfigurationMismatchError, load_spectrogram
from coughvid.features.schema import (
    ACOUSTIC_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    RECORDING_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    compute_config_fingerprint,
    validate_features_dataframe,
)


def run_pilot(sample_size: int = 70, overwrite: bool = True) -> None:
    cfg = load_config()
    data_dir = Path(cfg.paths.data_dir)
    manifest_path = data_dir / "splits" / "train.csv"
    current_fp = compute_config_fingerprint(cfg)

    print("=" * 75)
    print("COUGHVID ML - Milestone 4: Acoustic Feature Extraction Pilot Validation")
    print("=" * 75)
    print(f"Manifest Path       : {manifest_path}")
    print(f"Pilot Size          : {sample_size} records")
    print(f"Overwrite Mode      : {overwrite}")
    print(f"Schema Version      : {SCHEMA_VERSION}")
    print(f"Config Fingerprint  : {current_fp}\n")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Train split manifest not found at {manifest_path}")

    # Initialize Extractor
    features_out_dir = data_dir / "features"
    reports_out_dir = Path(cfg.paths.reports_dir)
    extractor = BatchFeatureExtractor(
        config=cfg,
        output_dir=features_out_dir,
        reports_dir=reports_out_dir,
    )

    # 1. Execute Pilot Extraction
    print("Step 1: Running batch feature extraction on pilot sample...")
    summary = extractor.process_manifest(
        manifest_path=manifest_path,
        partition_name="pilot_train",
        max_records=sample_size,
        batch_size=25,
        overwrite=overwrite,
    )

    # 2. Inspect Parquet Features Table and Validate Schema
    print("\nStep 2: Inspecting output Parquet table and validating schema...")
    parquet_path = Path(summary.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Expected Parquet table not found: {parquet_path}")

    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    file_size_kb = parquet_path.stat().st_size / 1024.0

    print(f"  Parquet File Path : {parquet_path}")
    print(f"  Parquet File Size : {file_size_kb:.2f} KB")
    print(f"  Total Rows Cached : {len(df):,}")
    print(f"  Total Columns     : {len(df.columns)}")
    print(f"  Acoustic Features : {summary.num_tabular_features} (Authoritative: {len(ACOUSTIC_FEATURE_COLUMNS)})")
    print(f"  Metadata Columns  : {len(METADATA_COLUMNS)}")

    # Validate exact schema adherence (names and ordering)
    validate_features_dataframe(df)
    print(f"  [PASS] DataFrame matches authoritative Schema v{SCHEMA_VERSION} (exact 217 columns in canonical order).")

    # Verify Parquet metadata contains config fingerprint
    pq_meta = table.schema.metadata or {}
    fp_in_pq = pq_meta.get(b"config_fingerprint", b"").decode("utf-8")
    assert fp_in_pq == current_fp, f"Fingerprint mismatch in Parquet metadata! Expected {current_fp}, got {fp_in_pq}"
    print(f"  [PASS] Parquet metadata contains verified configuration fingerprint: {fp_in_pq}")

    # 3. Numeric Integrity: NaN and Infinity Checks
    print("\nStep 3: Validating numerical integrity (NaN and Infinity checks)...")
    meta_cols = set(METADATA_COLUMNS)
    acoustic_cols = [c for c in df.columns if c not in meta_cols]
    acoustic_nan_counts = int(df[acoustic_cols].isna().sum().sum())
    acoustic_inf_counts = int(np.isinf(df[acoustic_cols].to_numpy()).sum())
    meta_nan_counts = int(df[["age"]].isna().sum().sum()) if "age" in df.columns else 0

    print(f"  Acoustic Columns Scanned   : {len(acoustic_cols)}")
    print(f"  Acoustic NaN values        : {acoustic_nan_counts}")
    print(f"  Acoustic Inf values        : {acoustic_inf_counts}")
    print(f"  Optional Metadata NaNs     : {meta_nan_counts} (self-reported 'age' field)")

    if acoustic_nan_counts > 0 or acoustic_inf_counts > 0:
        raise ValueError("Pilot validation failed: non-finite values detected in acoustic features!")
    print(f"  [PASS] All {len(ACOUSTIC_FEATURE_COLUMNS)} acoustic features are 100% finite (0 NaN, 0 Inf).")

    # 4. UUID and Label Preservation Check
    print("\nStep 4: Verifying UUID and Label preservation against split manifest...")
    manifest_df = pd.read_csv(manifest_path, nrows=sample_size)
    manifest_by_uuid = {str(r["uuid"]): int(r["label"]) for _, r in manifest_df.iterrows()}

    for _, row in df.iterrows():
        rec_uuid = str(row["uuid"])
        rec_label = int(row["label"])
        assert rec_uuid in manifest_by_uuid, f"UUID '{rec_uuid}' not found in manifest!"
        assert rec_label == manifest_by_uuid[rec_uuid], (
            f"Label mismatch for UUID '{rec_uuid}': expected {manifest_by_uuid[rec_uuid]}, got {rec_label}"
        )
    print(f"  [PASS] Verified {len(df)} rows: perfect 1-to-1 UUID and label alignment with train manifest.")

    # 5. Candidate Segmentation & Exclusion Audit (Separate class accounting)
    print("\nStep 5: Auditing candidate segmentation and class-specific extraction...")
    print(f"  Input Records       : {summary.processed_records}")
    print(f"  Successful Caches   : {summary.success_count} ({summary.success_count / summary.processed_records * 100:.1f}%)")
    print(f"  Exclusions          : {summary.exclusion_count} ({summary.exclusion_count / summary.processed_records * 100:.1f}%)")
    print(f"  Failure Breakdown   : {summary.failure_reasons}")
    print(f"  Input Classes       : {summary.class_distribution_input}")
    print(f"  Success Classes     : {summary.class_distribution_success}")
    print(f"  Excluded Classes    : {summary.class_distribution_excluded}")
    print(f"  Success Rate/Class  : {summary.class_success_rates}")

    # Inspect error records
    error_json_p = Path(summary.error_manifest_path)
    if error_json_p.exists():
        with open(error_json_p, "r", encoding="utf-8") as f:
            err_records = json.load(f)
        print(f"  Error Manifest JSON : {error_json_p} ({len(err_records)} records)")
        # Verify that excluded records were not assigned fabricated features
        error_uuids = {e["uuid"] for e in err_records}
        assert not df["uuid"].isin(error_uuids).any(), "CRITICAL: Excluded UUIDs found in Parquet feature table!"
        print("  [PASS] Verified: Zero excluded recordings were assigned fabricated features.")

    # 6. Spectrogram Archive Alignment & Direct UUID Retrieval
    print("\nStep 6: Verifying Log-Mel spectrogram storage and direct UUID retrieval...")
    spec_path = Path(summary.spectrogram_path)
    if spec_path.exists():
        spec_size_kb = spec_path.stat().st_size / 1024.0
        with np.load(spec_path) as npz:
            spec_keys = set(npz.files)
        pq_uuids = set(df["uuid"])
        assert spec_keys == pq_uuids, "Spectrogram keys mismatch with Parquet UUIDs!"
        print(f"  Spectrogram NPZ     : {spec_path} ({spec_size_kb:.2f} KB)")
        print(f"  Tensors Cached      : {len(spec_keys)}")

        # Test load_spectrogram helper on a sample UUID
        sample_uuid = list(pq_uuids)[0]
        retrieved_spec = load_spectrogram(spec_path, sample_uuid)
        print(f"  Direct UUID Retrieve: {sample_uuid} -> shape {retrieved_spec.shape}, dtype {retrieved_spec.dtype}")
        assert retrieved_spec.shape == (cfg.features.n_mels, cfg.features.spectrogram_frames)
        print("  [PASS] Fixed-shape Log-Mel spectrograms safely indexed by UUID.")

    # 7. Resumability & Configuration Fingerprint Enforcement
    print("\nStep 7: Verifying resumability and configuration fingerprint guard...")
    # A. Resuming with identical config should skip existing rows
    resumed_summary = extractor.process_manifest(
        manifest_path=manifest_path,
        partition_name="pilot_train",
        max_records=sample_size,
        overwrite=False,
    )
    table_recheck = pq.read_table(parquet_path)
    df_recheck = table_recheck.to_pandas()
    assert len(df_recheck) == len(df), f"Duplicate rows detected! Expected {len(df)}, got {len(df_recheck)}"
    print(f"  [PASS] Resumed run safely skipped all {len(df)} existing rows with zero duplicate records.")

    # B. Resuming with altered config without overwrite should raise ConfigurationMismatchError
    mock_altered_cfg = load_config()
    mock_altered_cfg.audio.lowpass_hz = 5000  # Alter lowpass cutoff
    extractor_altered = BatchFeatureExtractor(
        config=mock_altered_cfg,
        output_dir=features_out_dir,
        reports_dir=reports_out_dir,
    )
    try:
        extractor_altered.process_manifest(
            manifest_path=manifest_path,
            partition_name="pilot_train",
            max_records=sample_size,
            overwrite=False,
        )
        raise AssertionError("Expected ConfigurationMismatchError was not raised!")
    except ConfigurationMismatchError as e:
        print(f"  [PASS] ConfigurationMismatchError successfully caught on config change: {e}")

    print("\n" + "=" * 75)
    print("Milestone 4 Focused Validation and Audit PASSED successfully!")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milestone 4 feature extraction pilot.")
    parser.add_argument("--sample-size", type=int, default=70, help="Number of records to process.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing files.")
    args = parser.parse_args()

    run_pilot(sample_size=args.sample_size, overwrite=not args.no_overwrite)
