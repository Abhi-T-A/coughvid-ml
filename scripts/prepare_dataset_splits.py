"""Prepare leak-free dataset splits, audit metadata, and generate machine-readable reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coughvid.config import load_config
from coughvid.data.loader import audit_metadata, load_raw_metadata
from coughvid.data.tasks import build_working_set
from coughvid.data.splitters import (
    audit_leakage_and_splits,
    create_cross_validation_folds,
    create_holdout_split,
    save_splits_manifests,
)


def run_pipeline() -> None:
    cfg = load_config()
    raw_dir = cfg.resolve_raw_data_dir()
    meta_path = cfg.resolve_metadata_path()

    print("=" * 70)
    print("COUGHVID ML - Milestone 3: Ingestion, Validation & Leak-Free Splitting")
    print("=" * 70)
    print(f"Raw Dataset Dir : {raw_dir}")
    print(f"Metadata Path   : {meta_path}\n")

    # 1. Ingest raw metadata (without decoding audio)
    print("Step 1: Ingesting master metadata and resolving audio file existence...")
    raw_rows = load_raw_metadata(meta_path, raw_data_dir=raw_dir)
    print(f"  Total records ingested: {len(raw_rows):,}")

    # 2. Audit raw metadata
    print("\nStep 2: Performing data quality audit on raw metadata...")
    raw_audit = audit_metadata(raw_rows)
    print(f"  Unique UUIDs: {raw_audit.unique_uuids:,} | Duplicates: {raw_audit.duplicate_uuids}")
    print(f"  Missing 'status': {raw_audit.missing_status_count:,} ({raw_audit.missing_status_pct:.2f}%)")
    print(f"  Missing audio on disk: {raw_audit.missing_audio_count}")
    print(f"  Status distribution (raw): {raw_audit.status_distribution}")
    print(f"  cough_detected >= 0.8: {int(raw_audit.cough_detected_stats['ge_0_8_count']):,}")

    # Save raw audit to reports
    reports_dir = Path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_audit_path = reports_dir / "metadata_raw_audit.json"
    with open(raw_audit_path, "w", encoding="utf-8") as f:
        json.dump(raw_audit.to_dict(), f, indent=2)
    print(f"  Saved raw audit report -> {raw_audit_path}")

    # 3. Build task working set (covid_binary)
    print(f"\nStep 3: Building task working set for '{cfg.dataset.target_task}'...")
    working_set = build_working_set(
        raw_rows,
        task=cfg.dataset.target_task,
        cough_threshold=cfg.dataset.cough_threshold,
        require_audio=True,
    )
    print(f"  Working set size: {working_set.num_samples:,}")
    print(f"  Class counts: {working_set.class_counts} (0=Healthy, 1=COVID-19)")
    print(f"  Class proportions: {working_set.class_proportions}")
    print("  Attrition log (records dropped at each stage):")
    for step, count in working_set.attrition_log.items():
        print(f"    - {step:30s}: {count:,}")

    # Save working set CSV
    data_dir = Path(cfg.paths.data_dir)
    ws_csv_path = data_dir / "metadata_working_set.csv"
    working_set.to_csv(ws_csv_path)
    print(f"  Saved working set CSV -> {ws_csv_path}")

    # 4. Strict Holdout Split First (85/15)
    print(f"\nStep 4: Creating hold-out test partition ({cfg.splits.test_size*100:.0f}%)...")
    train_records, test_records = create_holdout_split(
        working_set.records,
        test_size=cfg.splits.test_size,
        random_seed=cfg.splits.random_seed,
    )
    print(f"  Train set size: {len(train_records):,} | Test set size: {len(test_records):,}")

    # 5. K-Fold CV generated EXCLUSIVELY on training partition
    print(f"\nStep 5: Generating {cfg.splits.n_splits}-fold CV splits exclusively on train partition...")
    folds = create_cross_validation_folds(
        train_records,
        n_splits=cfg.splits.n_splits,
        random_seed=cfg.splits.random_seed,
    )
    for i, (ft, fv) in enumerate(folds):
        ft_pos = sum(1 for r in ft if r["label"] == 1)
        fv_pos = sum(1 for r in fv if r["label"] == 1)
        print(f"  Fold {i}: train={len(ft):,} (pos={ft_pos}), val={len(fv):,} (pos={fv_pos})")

    # 6. Leakage Audit
    print("\nStep 6: Running leakage audit and demographic cluster investigation...")
    split_audit = audit_leakage_and_splits(train_records, test_records, folds=folds)
    print(f"  [CHECK] UUID overlap between train and test: {split_audit.uuid_leakage_train_test}")
    for i, (v_test_leak, tr_val_leak) in enumerate(zip(split_audit.folds_leakage_val_test, split_audit.folds_leakage_train_val)):
        print(f"  [CHECK] Fold {i}: val/test overlap={v_test_leak}, train/val overlap={tr_val_leak}")

    print(f"  Shared demographic/spatial clusters between train and test: {split_audit.shared_demographic_clusters_count}")
    print(f"  Test records belonging to shared clusters: {split_audit.shared_demographic_records_in_test}")
    print(f"  Cluster disclaimer: {split_audit.cluster_disclaimer}")

    # 7. Save clean split manifests
    splits_dir = data_dir / "splits"
    saved_paths = save_splits_manifests(train_records, test_records, folds=folds, output_dir=splits_dir)
    print(f"\nStep 7: Saved clean split manifests to {splits_dir}:")
    for name, p in saved_paths.items():
        print(f"  - {name:15s}: {p.name}")

    print("\nMilestone 3 dataset preparation complete.")


if __name__ == "__main__":
    run_pipeline()
