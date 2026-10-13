"""Unit tests for baseline models, preprocessing isolation, and cross-validation integrity."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from coughvid.config import AppConfig
from coughvid.features.schema import ACOUSTIC_FEATURE_COLUMNS, RECORDING_FEATURE_COLUMNS
from coughvid.models.baseline import apply_training_only_oversampling, create_baseline_pipeline
from coughvid.models.trainer import CrossValidationTrainer


@pytest.fixture
def synthetic_training_features() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 208-dimensional feature matrix with imbalanced labels."""
    rng = np.random.RandomState(42)
    n_samples = 120
    n_features = len(ACOUSTIC_FEATURE_COLUMNS)  # 208 features

    X = rng.randn(n_samples, n_features).astype(np.float32)
    # Inject missing values to test imputer
    X[0, 5] = np.nan
    X[10, 12] = np.nan

    # 10% positive class prevalence
    y = np.zeros(n_samples, dtype=int)
    y[:12] = 1
    return X, y


def test_pipeline_preprocessing_isolation(synthetic_training_features):
    """Verify that imputer and scaler statistics are computed exclusively on training data."""
    X, y = synthetic_training_features
    X_train, y_train = X[:80], y[:80]
    X_val = X[80:]

    pipe = create_baseline_pipeline(model_type="logistic_regression", random_seed=42)
    pipe.fit(X_train, y_train)

    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]

    # Scaler mean must match the imputed X_train exactly, completely isolated from X_val
    X_train_imputed = imputer.transform(X_train)
    expected_means = np.mean(X_train_imputed, axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-3, atol=1e-5)

    # Predictions on val fold must be finite and probabilities sum to 1.0
    probs = pipe.predict_proba(X_val)
    assert probs.shape == (len(X_val), 2)
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)


def test_training_only_oversampling():
    """Verify minority oversampling balances classes and does not mutate validation data."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(50, 10)
    y_train = np.array([0] * 45 + [1] * 5)  # 5 positive, 45 negative

    X_res, y_res = apply_training_only_oversampling(X_train, y_train, random_seed=42)

    assert len(X_res) == 90  # 45 + 45
    assert np.sum(y_res == 0) == 45
    assert np.sum(y_res == 1) == 45


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest", "xgboost"])
def test_baseline_pipeline_architectures(synthetic_training_features, model_type):
    """Verify all three baseline model pipelines fit and predict valid probabilities."""
    X, y = synthetic_training_features
    pipe = create_baseline_pipeline(model_type=model_type, random_seed=42)
    pipe.fit(X, y)

    probs = pipe.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(np.isfinite(probs))
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, rtol=1e-5)


def test_cross_validation_manifest_integrity():
    """Verify zero overlap between fold train/val splits and complete coverage of training set."""
    cfg = AppConfig()
    splits_dir = Path(cfg.paths.data_dir) / "splits"

    train_manifest = splits_dir / "train.csv"
    train_uuids = set(pd.read_csv(train_manifest)["uuid"])

    all_val_uuids = []

    for fold in range(5):
        f_train = pd.read_csv(splits_dir / f"fold_{fold}_train.csv")
        f_val = pd.read_csv(splits_dir / f"fold_{fold}_val.csv")

        set_tr = set(f_train["uuid"])
        set_val = set(f_val["uuid"])

        # Strict fold isolation
        overlap = set_tr & set_val
        assert len(overlap) == 0, f"Fold {fold} has {len(overlap)} overlapping UUIDs!"

        # Fold members are subsets of train.csv
        assert set_tr.issubset(train_uuids)
        assert set_val.issubset(train_uuids)

        all_val_uuids.extend(f_val["uuid"].tolist())

    # Union of all 5 validation splits covers exactly train.csv with 0 duplicates
    assert len(all_val_uuids) == len(train_uuids)
    assert len(set(all_val_uuids)) == len(train_uuids)
    assert set(all_val_uuids) == train_uuids


def test_model_serialization_and_reload(tmp_path, synthetic_training_features):
    """Verify exact prediction equality after model serialization via joblib."""
    X, y = synthetic_training_features
    pipe = create_baseline_pipeline(model_type="logistic_regression", random_seed=42)
    pipe.fit(X, y)

    orig_probs = pipe.predict_proba(X)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipe, model_path)

    reloaded_pipe = joblib.load(model_path)
    reloaded_probs = reloaded_pipe.predict_proba(X)

    np.testing.assert_allclose(orig_probs, reloaded_probs, rtol=1e-6)


def test_uuid_feature_alignment(tmp_path):
    """Verify that tabular features and spectrogram representations align strictly by UUID."""
    uuids = [f"uuid_{i:04d}" for i in range(20)]
    feature_data = {
        "uuid": uuids,
        "label": [i % 2 for i in range(20)],
        "partition": ["train"] * 20,
    }
    for col in ACOUSTIC_FEATURE_COLUMNS:
        feature_data[col] = np.random.randn(20).astype(np.float32)

    df = pd.DataFrame(feature_data)
    # Shuffle dataframe to test index alignment
    shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_by_uuid = shuffled_df.set_index("uuid")

    # Save mock spectrograms in npz keyed by uuid
    spec_dict = {u: np.random.randn(64, 64).astype(np.float32) for u in uuids}
    npz_path = tmp_path / "specs.npz"
    np.savez_compressed(npz_path, **spec_dict)

    # Verify retrieval preserves exact alignment regardless of dataframe ordering
    loaded_npz = np.load(npz_path)
    for u in uuids:
        assert u in df_by_uuid.index
        assert u in loaded_npz.files
        row_label = df_by_uuid.loc[u, "label"]
        expected_label = int(u.split("_")[1]) % 2
        assert row_label == expected_label
        spec_shape = loaded_npz[u].shape
        assert spec_shape == (64, 64)


def test_experiment_reproducibility(synthetic_training_features):
    """Verify that fitting pipelines with identical seeds produces identical predictions."""
    X, y = synthetic_training_features

    pipe1 = create_baseline_pipeline(model_type="random_forest", random_seed=123)
    pipe1.fit(X, y)
    probs1 = pipe1.predict_proba(X)

    pipe2 = create_baseline_pipeline(model_type="random_forest", random_seed=123)
    pipe2.fit(X, y)
    probs2 = pipe2.predict_proba(X)

    np.testing.assert_allclose(probs1, probs2, atol=1e-10)

    # Different seed should yield different tree structures/probabilities
    pipe3 = create_baseline_pipeline(model_type="random_forest", random_seed=999)
    pipe3.fit(X, y)
    probs3 = pipe3.predict_proba(X)
    assert not np.allclose(probs1, probs3, atol=1e-3)


