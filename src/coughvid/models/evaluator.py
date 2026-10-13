"""Production model evaluation, serialization, and holdout benchmark engine.

Implements leak-free production training on the full training partition,
production artifact serialization, and locked-threshold evaluation on the holdout test set.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from coughvid.config import AppConfig, load_config
from coughvid.evaluation.metrics import (
    ThresholdMetrics,
    compute_bootstrap_confidence_intervals,
    compute_threshold_metrics,
)
from coughvid.features.schema import (
    ACOUSTIC_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    compute_config_fingerprint,
)
from coughvid.models.baseline import create_baseline_pipeline

logger = logging.getLogger(__name__)

# PRE-LOCKED THRESHOLDS DERIVED STRICTLY FROM MILESTONE 5 TRAINING OOF PREDICTIONS
LOCKED_OPERATING_THRESHOLDS = {
    "xgboost": {
        "default": 0.500,
        "youden_optimal": 0.330,
        "high_sensitivity_80": 0.230,
    },
    "random_forest": {
        "default": 0.500,
        "youden_optimal": 0.390,
        "high_sensitivity_80": 0.340,
    },
}


@dataclass
class HoldoutModelResult:
    """Evaluation metrics for a production model on the holdout test set."""
    model_name: str
    num_test_samples: int
    num_test_positives: int
    num_test_negatives: int
    test_prevalence: float
    roc_auc: float
    average_precision: float
    brier_score: float
    log_loss: float
    threshold_evaluations: Dict[str, Dict[str, Any]]
    roc_auc_ci_95: Tuple[float, float]
    average_precision_ci_95: Tuple[float, float]
    brier_score_ci_95: Tuple[float, float]
    production_model_path: str
    schema_version: str
    config_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HoldoutEvaluator:
    """Manages full training, production serialization, and single-pass holdout evaluation."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        artifacts_dir: Optional[Path | str] = None,
        reports_dir: Optional[Path | str] = None,
    ) -> None:
        self.config = config or load_config()
        self.artifacts_dir = Path(artifacts_dir or (Path(self.config.paths.artifacts_dir) / "models"))
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.reports_dir = Path(reports_dir or self.config.paths.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = Path(self.config.paths.data_dir)
        self.splits_dir = self.data_dir / "splits"
        self.features_dir = self.data_dir / "features"

    def fit_and_serialize_production_model(
        self,
        model_type: str = "xgboost",
        random_seed: int = 42,
    ) -> Tuple[Pipeline, Path]:
        """Fit a production pipeline on the complete eligible training partition.

        Parameters
        ----------
        model_type : {"xgboost", "random_forest"}
            Architecture to fit.
        random_seed : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        Tuple[Pipeline, Path]
            Fitted scikit-learn Pipeline and the path to the serialized artifact.
        """
        train_pq_path = self.features_dir / "train_features.parquet"
        if not train_pq_path.exists():
            raise FileNotFoundError(f"Training features not found at {train_pq_path}")

        train_df = pd.read_parquet(train_pq_path)
        feature_cols = [c for c in ACOUSTIC_FEATURE_COLUMNS if c in train_df.columns]
        if len(feature_cols) != len(ACOUSTIC_FEATURE_COLUMNS):
            raise ValueError(
                f"Feature mismatch: expected {len(ACOUSTIC_FEATURE_COLUMNS)}, found {len(feature_cols)}"
            )

        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df["label"].to_numpy(dtype=int)

        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

        print(f"\n--- Fitting Production Model: '{model_type}' on Full Training Set ---")
        print(f"  Training Samples: {len(X_train):,} (Positives: {n_pos:,}, Negatives: {n_neg:,})")
        print(f"  Feature Dimensions: {X_train.shape[1]} acoustic features")

        pipeline = create_baseline_pipeline(
            model_type=model_type,
            random_seed=random_seed,
            class_weight="balanced" if model_type == "random_forest" else None,
            scale_pos_weight=scale_pos_weight if model_type == "xgboost" else None,
        )

        pipeline.fit(X_train, y_train)

        # Serialize model artifact
        model_path = self.artifacts_dir / f"production_{model_type}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"  Saved production model -> {model_path}")

        # Serialize metadata card
        current_fp = compute_config_fingerprint(self.config)
        metadata = {
            "model_name": model_type,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": current_fp,
            "training_samples": len(X_train),
            "training_positives": n_pos,
            "training_negatives": n_neg,
            "scale_pos_weight": scale_pos_weight if model_type == "xgboost" else None,
            "feature_columns": feature_cols,
            "pre_locked_thresholds": LOCKED_OPERATING_THRESHOLDS.get(model_type, {}),
            "random_seed": random_seed,
        }
        meta_path = self.artifacts_dir / f"production_{model_type}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved model metadata -> {meta_path}")

        return pipeline, model_path

    def evaluate_holdout_test_set(
        self,
        model_type: str = "xgboost",
        pipeline: Optional[Pipeline] = None,
        n_bootstraps: int = 1000,
        random_seed: int = 42,
    ) -> HoldoutModelResult:
        """Execute a single-pass evaluation on the holdout test partition.

        Parameters
        ----------
        model_type : {"xgboost", "random_forest"}
            Model to evaluate.
        pipeline : Pipeline, optional
            Fitted pipeline. If None, loaded from artifacts/models/production_{model_type}.joblib.
        n_bootstraps : int, default=1000
            Number of bootstrap resamples for 95% confidence intervals.
        random_seed : int, default=42
            Seed for bootstrap sampling.

        Returns
        -------
        HoldoutModelResult
            Complete holdout evaluation result.
        """
        # 1. Load pipeline if not passed
        if pipeline is None:
            model_path = self.artifacts_dir / f"production_{model_type}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Production model not found at {model_path}. Fit it first.")
            pipeline = joblib.load(model_path)
        else:
            model_path = self.artifacts_dir / f"production_{model_type}.joblib"

        # 2. Load test features
        test_pq_path = self.features_dir / "test_features.parquet"
        if not test_pq_path.exists():
            raise FileNotFoundError(f"Holdout test features not found at {test_pq_path}")

        test_df = pd.read_parquet(test_pq_path)
        feature_cols = [c for c in ACOUSTIC_FEATURE_COLUMNS if c in test_df.columns]
        if len(feature_cols) != len(ACOUSTIC_FEATURE_COLUMNS):
            raise ValueError(f"Test feature column mismatch! Expected {len(ACOUSTIC_FEATURE_COLUMNS)}")

        # Verify zero overlap with train set
        train_manifest = pd.read_csv(self.splits_dir / "train.csv")
        train_uuids = set(train_manifest["uuid"].astype(str))
        test_uuids = set(test_df["uuid"].astype(str))
        overlap = train_uuids & test_uuids
        if len(overlap) > 0:
            raise ValueError(f"CRITICAL: Leakage detected! {len(overlap)} UUIDs overlap between train and test!")

        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_df["label"].to_numpy(dtype=int)

        n_samples = len(X_test)
        n_pos = int(np.sum(y_test == 1))
        n_neg = int(np.sum(y_test == 0))
        prevalence = float(n_pos) / float(n_samples)

        print(f"\n--- Evaluating Holdout Test Set: '{model_type}' ---")
        print(f"  Test Samples: {n_samples:,} (Positives: {n_pos:,}, Negatives: {n_neg:,}, Prev: {prevalence*100:.2f}%)")

        # 3. Single-pass inference
        y_probs = pipeline.predict_proba(X_test)[:, 1]

        # 4. Threshold-independent metrics
        test_roc = float(roc_auc_score(y_test, y_probs))
        test_ap = float(average_precision_score(y_test, y_probs))
        test_brier = float(brier_score_loss(y_test, y_probs))
        clipped_p = np.clip(y_probs, 1e-15, 1.0 - 1e-15)
        test_logloss = float(log_loss(y_test, clipped_p))

        # 5. Fixed-threshold evaluations
        locked_thresholds = LOCKED_OPERATING_THRESHOLDS.get(model_type, {"default": 0.500})
        threshold_evals: Dict[str, Dict[str, Any]] = {}

        for th_name, th_val in locked_thresholds.items():
            tm = compute_threshold_metrics(y_test, y_probs, threshold=th_val)
            threshold_evals[th_name] = tm.to_dict()

        # 6. Bootstrap 95% Confidence Intervals
        cis = compute_bootstrap_confidence_intervals(
            y_true=y_test,
            y_proba=y_probs,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
        )

        current_fp = compute_config_fingerprint(self.config)

        res = HoldoutModelResult(
            model_name=model_type,
            num_test_samples=n_samples,
            num_test_positives=n_pos,
            num_test_negatives=n_neg,
            test_prevalence=round(prevalence, 4),
            roc_auc=round(test_roc, 4),
            average_precision=round(test_ap, 4),
            brier_score=round(test_brier, 4),
            log_loss=round(test_logloss, 4),
            threshold_evaluations=threshold_evals,
            roc_auc_ci_95=cis["roc_auc_ci"],
            average_precision_ci_95=cis["pr_auc_ci"],
            brier_score_ci_95=cis["brier_score_ci"],
            production_model_path=str(model_path),
            schema_version=SCHEMA_VERSION,
            config_fingerprint=current_fp,
        )

        return res
