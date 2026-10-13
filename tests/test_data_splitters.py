"""Unit tests for leak-free splitting protocols, cross-validation, and leakage audits."""

from pathlib import Path
import pytest

from coughvid.data.splitters import (
    SplitAuditReport,
    audit_leakage_and_splits,
    create_cross_validation_folds,
    create_holdout_split,
    save_splits_manifests,
)


@pytest.fixture
def synthetic_labeled_dataset() -> list[dict]:
    """Generate 100 synthetic records with an 80/20 class imbalance and demographic clusters."""
    records = []
    # 80 class 0 (healthy), 20 class 1 (COVID)
    for i in range(100):
        label = 1 if i < 20 else 0
        # Cluster 5 records per demographic/location bucket to simulate repeat participants
        cluster_id = i // 5
        records.append({
            "uuid": f"uuid-{i:03d}",
            "label": label,
            "latitude": f"{40.0 + (cluster_id * 0.1):.1f}",
            "longitude": f"{10.0 + (cluster_id * 0.1):.1f}",
            "age": f"{25 + (cluster_id % 10)}",
            "gender": "male" if i % 2 == 0 else "female",
        })
    return records


def test_holdout_split_stratification_and_no_leakage(synthetic_labeled_dataset):
    """Verify hold-out split strictly preserves class ratio and has zero UUID overlap."""
    train_records, test_records = create_holdout_split(
        synthetic_labeled_dataset,
        test_size=0.15,
        random_seed=42,
    )

    train_uuids = set(r["uuid"] for r in train_records)
    test_uuids = set(r["uuid"] for r in test_records)

    # 1. Zero UUID overlap
    assert len(train_uuids.intersection(test_uuids)) == 0
    assert len(train_records) + len(test_records) == 100

    # 2. Stratification: 15% of 20 positives is 3; 15% of 80 negatives is 12 -> 15 test samples
    test_pos = sum(1 for r in test_records if r["label"] == 1)
    test_neg = sum(1 for r in test_records if r["label"] == 0)
    assert test_pos == 3
    assert test_neg == 12


def test_holdout_split_reproducibility(synthetic_labeled_dataset):
    """Verify split is perfectly deterministic given the same random seed."""
    tr1, te1 = create_holdout_split(synthetic_labeled_dataset, test_size=0.15, random_seed=42)
    tr2, te2 = create_holdout_split(synthetic_labeled_dataset, test_size=0.15, random_seed=42)
    tr3, te3 = create_holdout_split(synthetic_labeled_dataset, test_size=0.15, random_seed=999)

    assert [r["uuid"] for r in tr1] == [r["uuid"] for r in tr2]
    assert [r["uuid"] for r in te1] == [r["uuid"] for r in te2]
    assert [r["uuid"] for r in tr1] != [r["uuid"] for r in tr3]


def test_cross_validation_folds_exclusivity(synthetic_labeled_dataset):
    """Verify CV folds are generated exclusively from train partition and are disjoint."""
    train_records, test_records = create_holdout_split(
        synthetic_labeled_dataset,
        test_size=0.15,
        random_seed=42,
    )

    folds = create_cross_validation_folds(train_records, n_splits=5, random_seed=42)
    assert len(folds) == 5

    test_uuids = set(r["uuid"] for r in test_records)
    all_val_uuids = []

    for fold_train, fold_val in folds:
        ft_uuids = set(r["uuid"] for r in fold_train)
        fv_uuids = set(r["uuid"] for r in fold_val)

        # Train and val within fold are disjoint
        assert len(ft_uuids.intersection(fv_uuids)) == 0
        # No fold validation sample exists in test set
        assert len(fv_uuids.intersection(test_uuids)) == 0
        # No fold training sample exists in test set
        assert len(ft_uuids.intersection(test_uuids)) == 0

        all_val_uuids.extend(r["uuid"] for r in fold_val)

    # Every training sample appeared in validation exactly once
    assert sorted(all_val_uuids) == sorted(r["uuid"] for r in train_records)


def test_audit_leakage_and_splits(synthetic_labeled_dataset):
    """Verify audit correctly reports zero UUID leakage and audits shared demographic clusters."""
    train_records, test_records = create_holdout_split(
        synthetic_labeled_dataset,
        test_size=0.15,
        random_seed=42,
    )
    folds = create_cross_validation_folds(train_records, n_splits=5, random_seed=42)

    audit = audit_leakage_and_splits(train_records, test_records, folds=folds)
    assert isinstance(audit, SplitAuditReport)
    assert audit.uuid_leakage_train_test == 0
    assert audit.num_folds == 5
    assert all(leak == 0 for leak in audit.folds_leakage_val_test)
    assert all(leak == 0 for leak in audit.folds_leakage_train_val)
    # Check that demographic clusters are reported
    assert audit.shared_demographic_clusters_count > 0
    assert "cluster_disclaimer" in audit.to_dict()
    assert "do NOT conclusively prove participant identity" in audit.cluster_disclaimer


def test_save_splits_manifests(synthetic_labeled_dataset, tmp_path):
    """Verify save_splits_manifests writes clean, readable split manifests and audit JSON."""
    train_records, test_records = create_holdout_split(
        synthetic_labeled_dataset,
        test_size=0.15,
        random_seed=42,
    )
    folds = create_cross_validation_folds(train_records, n_splits=5, random_seed=42)

    paths = save_splits_manifests(train_records, test_records, folds=folds, output_dir=tmp_path)
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()
    assert (tmp_path / "split_audit.json").exists()
    for i in range(5):
        assert (tmp_path / f"fold_{i}_train.csv").exists()
        assert (tmp_path / f"fold_{i}_val.csv").exists()
