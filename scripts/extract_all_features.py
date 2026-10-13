"""Full Feature Extraction Runner for Training and Test Partitions.

Executes batch feature extraction on approved manifests:
- data/splits/train.csv -> data/features/train_features.parquet, train_spectrograms.npz
- data/splits/test.csv -> data/features/test_features.parquet, test_spectrograms.npz

Enforces authoritative Schema v1.0.0, configuration fingerprinting, separate
spectrogram NPZ storage, and structured error manifests without leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coughvid.config import load_config
from coughvid.features.extractor import BatchFeatureExtractor, load_spectrogram
from coughvid.features.schema import (
    ACOUSTIC_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    RECORDING_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    compute_config_fingerprint,
    validate_features_dataframe,
)


def verify_partition_integrity(
    parquet_path: Path,
    spec_path: Path,
    manifest_path: Path,
    expected_fingerprint: str,
) -> dict:
    """Validate schema, fingerprints, NaN/Inf counts, and UUID alignment."""
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # 1. Schema v1.0.0 check
    validate_features_dataframe(df)

    # 2. Config fingerprint check
    pq_meta = table.schema.metadata or {}
    fp_in_pq = pq_meta.get(b"config_fingerprint", b"").decode("utf-8")
    if fp_in_pq != expected_fingerprint:
        raise ValueError(
            f"Fingerprint mismatch in {parquet_path.name}: expected {expected_fingerprint}, got {fp_in_pq}"
        )

    # 3. Duplicate UUID check
    if len(df["uuid"].unique()) != len(df):
        raise ValueError(f"Duplicate UUIDs detected in {parquet_path.name}!")

    # 4. Numeric integrity check (0 NaN, 0 Inf in acoustic columns)
    meta_cols = set(METADATA_COLUMNS)
    acoustic_cols = [c for c in df.columns if c not in meta_cols]
    nan_count = int(df[acoustic_cols].isna().sum().sum())
    inf_count = int(np.isinf(df[acoustic_cols].to_numpy()).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"Non-finite values detected in {parquet_path.name}: {nan_count} NaNs, {inf_count} Infs!")

    # 5. UUID alignment with NPZ
    with np.load(spec_path) as npz:
        spec_keys = set(npz.files)
    pq_uuids = set(df["uuid"])
    if spec_keys != pq_uuids:
        missing_in_npz = len(pq_uuids - spec_keys)
        missing_in_pq = len(spec_keys - pq_uuids)
        raise ValueError(
            f"Spectrogram/Parquet UUID mismatch! Missing in NPZ: {missing_in_npz}, Missing in Parquet: {missing_in_pq}"
        )

    # 6. Alignment with source manifest
    manifest_df = pd.read_csv(manifest_path)
    manifest_map = {str(r["uuid"]): int(r["label"]) for _, r in manifest_df.iterrows()}
    for _, row in df.iterrows():
        u = str(row["uuid"])
        lbl = int(row["label"])
        if u not in manifest_map:
            raise ValueError(f"UUID '{u}' not found in manifest {manifest_path.name}!")
        if lbl != manifest_map[u]:
            raise ValueError(f"Label mismatch for UUID '{u}': manifest={manifest_map[u]}, parquet={lbl}")

    return {
        "parquet_rows": len(df),
        "parquet_cols": len(df.columns),
        "acoustic_features": len(acoustic_cols),
        "spectrogram_tensors": len(spec_keys),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "class_distribution": df["label"].value_counts().to_dict(),
    }


def run_full_extraction(
    num_workers: int = 6,
    batch_size: int = 200,
    overwrite: bool = False,
) -> None:
    cfg = load_config()
    data_dir = Path(cfg.paths.data_dir)
    splits_dir = data_dir / "splits"
    features_dir = data_dir / "features"
    reports_dir = Path(cfg.paths.reports_dir)

    train_manifest = splits_dir / "train.csv"
    test_manifest = splits_dir / "test.csv"

    if not train_manifest.exists() or not test_manifest.exists():
        raise FileNotFoundError("Approved split manifests (train.csv / test.csv) not found in data/splits!")

    current_fp = compute_config_fingerprint(cfg)

    print("=" * 80)
    print("COUGHVID ML - Milestone 5: Full Acoustic Feature Extraction")
    print("=" * 80)
    print(f"Target Sample Rate  : {cfg.audio.target_sr} Hz")
    print(f"Lowpass Filter      : {cfg.audio.lowpass_hz} Hz (Order {cfg.audio.filter_order})")
    print(f"Schema Version      : {SCHEMA_VERSION}")
    print(f"Config Fingerprint  : {current_fp}")
    print(f"Worker Concurrency  : {num_workers} threads")
    print(f"Batch Size          : {batch_size}")
    print(f"Overwrite Mode      : {overwrite}\n")

    extractor = BatchFeatureExtractor(
        config=cfg,
        output_dir=features_dir,
        reports_dir=reports_dir,
    )

    t_start = time.time()

    # ----------------------------------------------------
    # PARTITION 1: TRAINING PARTITION
    # ----------------------------------------------------
    print(f"--- [1/2] Processing Training Partition ({train_manifest.name}) ---")
    t0 = time.time()
    train_summary = extractor.process_manifest(
        manifest_path=train_manifest,
        partition_name="train",
        batch_size=batch_size,
        num_workers=num_workers,
        overwrite=overwrite,
    )
    t1 = time.time()
    print(f"Training partition extraction finished in {t1 - t0:.1f} s ({t1 - t0:.1f}s).")
    print(f"  Total Processed : {train_summary.processed_records:,}")
    print(f"  Success Count   : {train_summary.success_count:,} ({train_summary.success_count / train_summary.processed_records * 100:.1f}%)")
    print(f"  Exclusions      : {train_summary.exclusion_count:,} ({train_summary.exclusion_count / train_summary.processed_records * 100:.1f}%)")
    print(f"  Class Breakdown : Input={train_summary.class_distribution_input}, Success={train_summary.class_distribution_success}, Excluded={train_summary.class_distribution_excluded}")
    print(f"  Success Rates   : {train_summary.class_success_rates}")

    # Verify Training Integrity
    train_pq = features_dir / "train_features.parquet"
    train_npz = features_dir / "train_spectrograms.npz"
    train_audit = verify_partition_integrity(train_pq, train_npz, train_manifest, current_fp)
    print(f"  [PASS] Training Parquet integrity verified: {train_audit['parquet_rows']:,} rows, {train_audit['parquet_cols']} cols, 0 NaN/Inf.")

    # ----------------------------------------------------
    # PARTITION 2: TEST PARTITION (LOCKED HOLDOUT)
    # ----------------------------------------------------
    print(f"\n--- [2/2] Processing Test Partition ({test_manifest.name}) ---")
    t2 = time.time()
    test_summary = extractor.process_manifest(
        manifest_path=test_manifest,
        partition_name="test",
        batch_size=batch_size,
        num_workers=num_workers,
        overwrite=overwrite,
    )
    t3 = time.time()
    print(f"Test partition extraction finished in {t3 - t2:.1f} s ({t3 - t2:.1f}s).")
    print(f"  Total Processed : {test_summary.processed_records:,}")
    print(f"  Success Count   : {test_summary.success_count:,} ({test_summary.success_count / test_summary.processed_records * 100:.1f}%)")
    print(f"  Exclusions      : {test_summary.exclusion_count:,} ({test_summary.exclusion_count / test_summary.processed_records * 100:.1f}%)")
    print(f"  Class Breakdown : Input={test_summary.class_distribution_input}, Success={test_summary.class_distribution_success}, Excluded={test_summary.class_distribution_excluded}")
    print(f"  Success Rates   : {test_summary.class_success_rates}")

    # Verify Test Integrity
    test_pq = features_dir / "test_features.parquet"
    test_npz = features_dir / "test_spectrograms.npz"
    test_audit = verify_partition_integrity(test_pq, test_npz, test_manifest, current_fp)
    print(f"  [PASS] Test Parquet integrity verified: {test_audit['parquet_rows']:,} rows, {test_audit['parquet_cols']} cols, 0 NaN/Inf.")

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"Phase A Full Feature Extraction Complete in {total_time / 60:.2f} minutes!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full feature extraction for train and test partitions.")
    parser.add_argument("--workers", type=int, default=6, help="Worker threads.")
    parser.add_argument("--batch-size", type=int, default=200, help="Batch size for checkpoints.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature stores.")
    args = parser.parse_args()

    run_full_extraction(num_workers=args.workers, batch_size=args.batch_size, overwrite=args.overwrite)
