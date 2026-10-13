"""Leak-free dataset partitioning, cross-validation protocol, and leakage auditing.

Strictly enforces:
1. Holdout partition created first before any modeling or fold creation.
2. Cross-validation folds generated exclusively from the training partition.
3. Verification of 0 UUID overlap across train, val, and test partitions.
4. Audit of shared demographic/location clusters across partitions with proper disclaimers.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SplitAuditReport:
    """Comprehensive, machine-readable split verification and leakage audit."""
    train_size: int
    test_size: int
    train_class_counts: Dict[int, int]
    test_class_counts: Dict[int, int]
    train_class_proportions: Dict[int, float]
    test_class_proportions: Dict[int, float]
    uuid_leakage_train_test: int
    num_folds: int
    folds_leakage_val_test: List[int]
    folds_leakage_train_val: List[int]
    shared_demographic_clusters_count: int
    shared_demographic_records_in_test: int
    cluster_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, output_path: str | Path) -> Path:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out


def _stratified_split_indices(
    labels: List[int],
    test_fraction: float,
    random_seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Deterministic stratified index splitting."""
    rng = random.Random(random_seed)
    class_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    train_indices: List[int] = []
    test_indices: List[int] = []

    for label, idxs in class_indices.items():
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])

    # Sort indices to preserve determinism
    train_indices.sort()
    test_indices.sort()
    return train_indices, test_indices


def create_holdout_split(
    records: List[Dict[str, Any]],
    test_size: float = 0.15,
    random_seed: int = 42,
    label_key: str = "label",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create a stratified hold-out test split before any cross-validation or modeling.

    Parameters
    ----------
    records : list[dict]
        Dataset records containing label_key.
    test_size : float, default=0.15
        Proportion of samples reserved for hold-out test set.
    random_seed : int, default=42
        Seed for reproducibility.
    label_key : str, default='label'
        Dictionary key indicating the class target.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (train_records, test_records)
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")
    if not records:
        return [], []

    labels = [int(r[label_key]) for r in records]
    train_idx, test_idx = _stratified_split_indices(labels, test_fraction=test_size, random_seed=random_seed)

    train_records = [records[i] for i in train_idx]
    test_records = [records[i] for i in test_idx]
    return train_records, test_records


def create_cross_validation_folds(
    train_records: List[Dict[str, Any]],
    n_splits: int = 5,
    random_seed: int = 42,
    label_key: str = "label",
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Generate stratified K-fold cross-validation folds strictly from training records.

    Parameters
    ----------
    train_records : list[dict]
        Training set records (must NOT contain test set samples).
    n_splits : int, default=5
        Number of cross-validation folds.
    random_seed : int, default=42
        Random seed for fold shuffling.
    label_key : str, default='label'
        Target label column name.

    Returns
    -------
    list[tuple[list[dict], list[dict]]]
        List of (fold_train_records, fold_val_records) for each fold.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}.")
    if not train_records:
        return []

    rng = random.Random(random_seed)
    class_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, r in enumerate(train_records):
        class_indices[int(r[label_key])].append(idx)

    # Distribute each class indices evenly into n_splits buckets
    fold_val_indices: List[List[int]] = [[] for _ in range(n_splits)]
    for label, idxs in class_indices.items():
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        for i, idx in enumerate(shuffled):
            fold_val_indices[i % n_splits].append(idx)

    folds: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    total_indices = set(range(len(train_records)))

    for k in range(n_splits):
        val_idx_set = set(fold_val_indices[k])
        train_idx_set = total_indices - val_idx_set

        val_records = [train_records[i] for i in sorted(val_idx_set)]
        fold_train_records = [train_records[i] for i in sorted(train_idx_set)]
        folds.append((fold_train_records, val_records))

    return folds


def audit_leakage_and_splits(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    folds: Optional[List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]] = None,
    label_key: str = "label",
) -> SplitAuditReport:
    """Verify data leakage and audit class distributions across splits."""
    train_uuids = set(r["uuid"] for r in train_records)
    test_uuids = set(r["uuid"] for r in test_records)
    leakage_train_test = len(train_uuids.intersection(test_uuids))

    # Fold leakage checks
    val_test_leakages: List[int] = []
    train_val_leakages: List[int] = []
    if folds:
        for fold_train, fold_val in folds:
            fold_train_uuids = set(r["uuid"] for r in fold_train)
            fold_val_uuids = set(r["uuid"] for r in fold_val)
            val_test_leakages.append(len(fold_val_uuids.intersection(test_uuids)))
            train_val_leakages.append(len(fold_train_uuids.intersection(fold_val_uuids)))

    # Demographic / spatial cluster investigation
    def get_cluster(r: Dict[str, Any]) -> Optional[Tuple[str, str, str, str]]:
        lat = str(r.get("latitude", "")).strip()
        lon = str(r.get("longitude", "")).strip()
        age = str(r.get("age", "")).strip()
        gen = str(r.get("gender", "")).strip()
        if lat and lon and age and gen and lat != "nan" and lon != "nan":
            return (lat, lon, age, gen)
        return None

    train_clusters = set(filter(None, (get_cluster(r) for r in train_records)))
    test_cluster_records = [r for r in test_records if get_cluster(r) in train_clusters]
    shared_clusters = set(filter(None, (get_cluster(r) for r in test_cluster_records)))

    # Class distributions
    train_counts = dict(Counter(int(r[label_key]) for r in train_records))
    test_counts = dict(Counter(int(r[label_key]) for r in test_records))

    train_props = {
        lbl: (count / len(train_records)) if train_records else 0.0
        for lbl, count in train_counts.items()
    }
    test_props = {
        lbl: (count / len(test_records)) if test_records else 0.0
        for lbl, count in test_counts.items()
    }

    disclaimer = (
        "Shared demographic/spatial coordinates (latitude, longitude, age, gender) "
        "indicate possible repeat submissions from the same individual or household, "
        "but do NOT conclusively prove participant identity without unique participant IDs."
    )

    return SplitAuditReport(
        train_size=len(train_records),
        test_size=len(test_records),
        train_class_counts=train_counts,
        test_class_counts=test_counts,
        train_class_proportions=train_props,
        test_class_proportions=test_props,
        uuid_leakage_train_test=leakage_train_test,
        num_folds=len(folds) if folds else 0,
        folds_leakage_val_test=val_test_leakages,
        folds_leakage_train_val=train_val_leakages,
        shared_demographic_clusters_count=len(shared_clusters),
        shared_demographic_records_in_test=len(test_cluster_records),
        cluster_disclaimer=disclaimer,
    )


def save_splits_manifests(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    folds: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Save clean, reproducible split CSVs and audit report to disk."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    def write_csv(p: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            p.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    train_p = out_dir / "train.csv"
    test_p = out_dir / "test.csv"
    write_csv(train_p, train_records)
    write_csv(test_p, test_records)
    paths["train"] = train_p
    paths["test"] = test_p

    for i, (fold_train, fold_val) in enumerate(folds):
        ft_p = out_dir / f"fold_{i}_train.csv"
        fv_p = out_dir / f"fold_{i}_val.csv"
        write_csv(ft_p, fold_train)
        write_csv(fv_p, fold_val)
        paths[f"fold_{i}_train"] = ft_p
        paths[f"fold_{i}_val"] = fv_p

    audit = audit_leakage_and_splits(train_records, test_records, folds=folds)
    audit_p = out_dir / "split_audit.json"
    audit.to_json(audit_p)
    paths["audit"] = audit_p

    return paths
