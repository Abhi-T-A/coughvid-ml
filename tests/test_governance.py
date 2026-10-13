"""Unit tests for model governance, serialization integrity, and inference safety."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from coughvid.config import load_config
from coughvid.features.schema import ACOUSTIC_FEATURE_COLUMNS
from coughvid.models.baseline import create_baseline_pipeline
from coughvid.models.evaluator import LOCKED_OPERATING_THRESHOLDS


def test_manifest_exclusivity():
    """Verify zero UUID overlap between train and test manifests."""
    cfg = load_config()
    splits_dir = Path(cfg.paths.data_dir) / "splits"

    train_df = pd.read_csv(splits_dir / "train.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    train_uuids = set(train_df["uuid"].astype(str))
    test_uuids = set(test_df["uuid"].astype(str))

    overlap = train_uuids & test_uuids
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping UUIDs between train and test!"


def test_pre_locked_thresholds_values():
    """Verify pre-locked operating thresholds match exact Milestone 5 training OOF values."""
    assert "xgboost" in LOCKED_OPERATING_THRESHOLDS
    assert "random_forest" in LOCKED_OPERATING_THRESHOLDS

    xgb_th = LOCKED_OPERATING_THRESHOLDS["xgboost"]
    assert xgb_th["default"] == 0.500
    assert xgb_th["youden_optimal"] == 0.330
    assert xgb_th["high_sensitivity_80"] == 0.230

    rf_th = LOCKED_OPERATING_THRESHOLDS["random_forest"]
    assert rf_th["default"] == 0.500
    assert rf_th["youden_optimal"] == 0.390
    assert rf_th["high_sensitivity_80"] == 0.340


def test_production_model_serialization_and_reload(tmp_path):
    """Verify production pipeline serialization and deserialization produces identical predictions."""
    rng = np.random.RandomState(42)
    n_samples = 50
    n_features = len(ACOUSTIC_FEATURE_COLUMNS)  # 208

    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, 2, size=n_samples)

    pipeline = create_baseline_pipeline(model_type="xgboost", random_seed=42)
    pipeline.fit(X, y)

    orig_probs = pipeline.predict_proba(X)[:, 1]

    save_path = tmp_path / "test_production_model.joblib"
    joblib.dump(pipeline, save_path)

    reloaded_pipeline = joblib.load(save_path)
    reloaded_probs = reloaded_pipeline.predict_proba(X)[:, 1]

    np.testing.assert_allclose(orig_probs, reloaded_probs, rtol=1e-5, atol=1e-6)


def test_single_instance_inference_pipeline_with_nan():
    """Verify pipeline handles a single 1D/2D inference request containing missing values."""
    rng = np.random.RandomState(42)
    n_features = len(ACOUSTIC_FEATURE_COLUMNS)

    X_train = rng.randn(40, n_features).astype(np.float32)
    y_train = rng.randint(0, 2, size=40)

    pipeline = create_baseline_pipeline(model_type="random_forest", random_seed=42)
    pipeline.fit(X_train, y_train)

    # Single instance vector with injected NaNs
    single_instance = rng.randn(1, n_features).astype(np.float32)
    single_instance[0, 5] = np.nan
    single_instance[0, 15] = np.nan

    prob = pipeline.predict_proba(single_instance)[0, 1]

    assert np.isfinite(prob)
    assert 0.0 <= prob <= 1.0


def test_production_model_feature_dimension_guard():
    """Verify that passing incorrect feature dimensions raises an error."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(20, len(ACOUSTIC_FEATURE_COLUMNS)).astype(np.float32)
    y_train = rng.randint(0, 2, size=20)

    pipeline = create_baseline_pipeline(model_type="xgboost", random_seed=42)
    pipeline.fit(X_train, y_train)

    # Wrong feature count (e.g. 50 instead of 208)
    X_bad = rng.randn(5, 50).astype(np.float32)
    with pytest.raises(ValueError):
        pipeline.predict_proba(X_bad)
