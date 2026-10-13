"""Cross-validation training and benchmark evaluation runner.

Orchestrates leak-free 5-fold cross-validation across baseline models:
- Fits preprocessing (median imputation, standardization) strictly on training fold.
- Evaluates fold-level and aggregated out-of-fold (OOF) predictions.
- Computes threshold-independent metrics (PR-AUC, ROC-AUC, Brier score) and
  training-only operating threshold sweeps without ever looking at the final test set.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from coughvid.config import AppConfig, load_config
from coughvid.evaluation.metrics import (
    ModelEvaluationReport,
    evaluate_predictions,
    find_optimal_threshold,
)
from coughvid.features.schema import (
    ACOUSTIC_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    RECORDING_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    compute_config_fingerprint,
)
from coughvid.models.baseline import apply_training_only_oversampling, create_baseline_pipeline

logger = logging.getLogger(__name__)


@dataclass
class FoldMetrics:
    """Metrics achieved on a single cross-validation fold."""
    fold_index: int
    train_samples: int
    val_samples: int
    train_positives: int
    val_positives: int
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrossValidationExperimentResult:
    """Comprehensive results of a 5-fold cross-validation experiment."""
    model_name: str
    n_splits: int
    total_training_records: int
    oof_predictions_count: int
    oof_coverage_pct: float
    fold_metrics: List[FoldMetrics]
    mean_roc_auc: float
    std_roc_auc: float
    mean_pr_auc: float
    std_pr_auc: float
    mean_brier_score: float
    std_brier_score: float
    oof_evaluation: ModelEvaluationReport
    optimal_thresholds: Dict[str, Any]
    schema_version: str
    config_fingerprint: str
    artifacts_dir: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fold_metrics"] = [f.to_dict() for f in self.fold_metrics]
        d["oof_evaluation"] = self.oof_evaluation.to_dict()
        return d


class CrossValidationTrainer:
    """Manages leakage-free cross-validation training and out-of-fold evaluation."""

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

    def load_feature_dataset(self, partition: str = "train") -> pd.DataFrame:
        """Load feature store and validate schema."""
        pq_path = self.features_dir / f"{partition}_features.parquet"
        if not pq_path.exists():
            raise FileNotFoundError(f"Feature store not found at {pq_path}. Run feature extraction first.")

        table = pq.read_table(pq_path)
        df = table.to_pandas()
        return df

    def train_baseline_model(
        self,
        model_type: str = "logistic_regression",
        use_oversampling: bool = False,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> CrossValidationExperimentResult:
        """Train and evaluate a baseline model across all 5 cross-validation folds.

        Parameters
        ----------
        model_type : {"logistic_regression", "random_forest", "xgboost"}
            Classifier architecture to train.
        use_oversampling : bool, default=False
            If True, apply random minority oversampling strictly to the training split of each fold.
        hyperparameters : dict, optional
            Custom hyperparameters passed to pipeline creation.

        Returns
        -------
        CrossValidationExperimentResult
            Complete metrics, OOF predictions, and model evaluation report.
        """
        # 1. Load full training feature store
        train_df = self.load_feature_dataset(partition="train")
        train_df_by_uuid = train_df.set_index("uuid")

        feature_cols = [c for c in ACOUSTIC_FEATURE_COLUMNS if c in train_df.columns]
        if len(feature_cols) != len(ACOUSTIC_FEATURE_COLUMNS):
            raise ValueError(
                f"Feature column mismatch! Expected {len(ACOUSTIC_FEATURE_COLUMNS)}, found {len(feature_cols)} in table."
            )

        n_splits = self.config.splits.n_splits
        fold_metrics_list: List[FoldMetrics] = []
        oof_records: List[Dict[str, Any]] = []
        fitted_pipelines: Dict[int, Any] = {}

        print(f"\n--- Training Baseline Model: '{model_type}' ({n_splits}-Fold CV) ---")
        if use_oversampling:
            print("  [Mode] Minority oversampling applied strictly to fold training splits.")

        for fold_idx in range(n_splits):
            fold_train_manifest = self.splits_dir / f"fold_{fold_idx}_train.csv"
            fold_val_manifest = self.splits_dir / f"fold_{fold_idx}_val.csv"

            if not fold_train_manifest.exists() or not fold_val_manifest.exists():
                raise FileNotFoundError(f"Manifest for fold {fold_idx} missing in {self.splits_dir}!")

            train_uuids = pd.read_csv(fold_train_manifest)["uuid"].astype(str).tolist()
            val_uuids = pd.read_csv(fold_val_manifest)["uuid"].astype(str).tolist()

            # Verify zero leakage across manifests
            overlap = set(train_uuids) & set(val_uuids)
            assert len(overlap) == 0, f"CRITICAL: Leakage detected in fold {fold_idx}! Overlap: {overlap}"

            # Filter to successfully cached recordings in feature store
            valid_train_uuids = [u for u in train_uuids if u in train_df_by_uuid.index]
            valid_val_uuids = [u for u in val_uuids if u in train_df_by_uuid.index]

            fold_train_df = train_df_by_uuid.loc[valid_train_uuids]
            fold_val_df = train_df_by_uuid.loc[valid_val_uuids]

            X_train = fold_train_df[feature_cols].to_numpy(dtype=np.float32)
            y_train = fold_train_df["label"].to_numpy(dtype=int)

            X_val = fold_val_df[feature_cols].to_numpy(dtype=np.float32)
            y_val = fold_val_df["label"].to_numpy(dtype=int)

            # Optional train-only oversampling
            if use_oversampling:
                X_train_fit, y_train_fit = apply_training_only_oversampling(
                    X_train, y_train, random_seed=self.config.splits.random_seed + fold_idx
                )
                scale_pos_weight = 1.0
                class_weight = None
            else:
                X_train_fit, y_train_fit = X_train, y_train
                n_pos = int(np.sum(y_train == 1))
                n_neg = int(np.sum(y_train == 0))
                scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
                class_weight = "balanced"

            # Create and fit pipeline with fold-isolated imputer and scaler
            pipeline = create_baseline_pipeline(
                model_type=model_type,
                random_seed=self.config.training.random_seed + fold_idx,
                class_weight=class_weight,
                scale_pos_weight=scale_pos_weight,
                hyperparameters=hyperparameters,
            )

            pipeline.fit(X_train_fit, y_train_fit)
            fitted_pipelines[fold_idx] = pipeline

            # Predict on hold-out validation fold
            y_val_prob = pipeline.predict_proba(X_val)[:, 1]

            # Collect OOF predictions
            for u, true_lbl, prob in zip(valid_val_uuids, y_val, y_val_prob):
                oof_records.append({
                    "uuid": u,
                    "fold": fold_idx,
                    "y_true": int(true_lbl),
                    "y_prob": float(prob),
                })

            # Evaluate fold metrics
            f_eval = evaluate_predictions(y_val, y_val_prob, n_bootstraps=200)
            fm = FoldMetrics(
                fold_index=fold_idx,
                train_samples=len(X_train),
                val_samples=len(X_val),
                train_positives=int(np.sum(y_train == 1)),
                val_positives=int(np.sum(y_val == 1)),
                roc_auc=f_eval.roc_auc,
                pr_auc=f_eval.pr_auc,
                brier_score=f_eval.brier_score,
                log_loss=f_eval.log_loss,
            )
            fold_metrics_list.append(fm)
            print(
                f"  Fold {fold_idx}: ROC-AUC={fm.roc_auc:.4f} | PR-AUC={fm.pr_auc:.4f} | "
                f"Brier={fm.brier_score:.4f} (Train={fm.train_samples:,}, Val={fm.val_samples:,})"
            )

            # Serialize fold model artifact
            model_artifact_path = self.artifacts_dir / f"{model_type}_fold_{fold_idx}.joblib"
            joblib.dump(pipeline, model_artifact_path)

        # 2. Out-of-Fold Analysis across aggregated predictions
        oof_df = pd.DataFrame(oof_records).sort_values("uuid").reset_index(drop=True)
        oof_csv_path = self.artifacts_dir / f"{model_type}_oof_predictions.csv"
        oof_df.to_csv(oof_csv_path, index=False)

        total_cached = len(train_df)
        oof_count = len(oof_df)
        oof_coverage = (float(oof_count) / float(total_cached)) * 100.0

        # Overall OOF evaluation
        oof_eval = evaluate_predictions(
            y_true=oof_df["y_true"].to_numpy(),
            y_proba=oof_df["y_prob"].to_numpy(),
            n_bootstraps=1000,
            random_seed=self.config.training.random_seed,
        )

        # Candidate operating thresholds computed strictly on training OOF predictions
        optimal_thresholds = {
            "youden_optimal": find_optimal_threshold(
                oof_df["y_true"].to_numpy(), oof_df["y_prob"].to_numpy(), strategy="youden"
            ).to_dict(),
            "f1_optimal": find_optimal_threshold(
                oof_df["y_true"].to_numpy(), oof_df["y_prob"].to_numpy(), strategy="f1"
            ).to_dict(),
            "high_sensitivity_80": find_optimal_threshold(
                oof_df["y_true"].to_numpy(), oof_df["y_prob"].to_numpy(), strategy="target_sensitivity", min_sensitivity=0.80
            ).to_dict(),
        }

        # Compute summary statistics across folds
        roc_list = [f.roc_auc for f in fold_metrics_list]
        pr_list = [f.pr_auc for f in fold_metrics_list]
        brier_list = [f.brier_score for f in fold_metrics_list]

        current_fp = compute_config_fingerprint(self.config)

        experiment_result = CrossValidationExperimentResult(
            model_name=model_type,
            n_splits=n_splits,
            total_training_records=total_cached,
            oof_predictions_count=oof_count,
            oof_coverage_pct=round(oof_coverage, 2),
            fold_metrics=fold_metrics_list,
            mean_roc_auc=round(float(np.mean(roc_list)), 4),
            std_roc_auc=round(float(np.std(roc_list)), 4),
            mean_pr_auc=round(float(np.mean(pr_list)), 4),
            std_pr_auc=round(float(np.std(pr_list)), 4),
            mean_brier_score=round(float(np.mean(brier_list)), 4),
            std_brier_score=round(float(np.std(brier_list)), 4),
            oof_evaluation=oof_eval,
            optimal_thresholds=optimal_thresholds,
            schema_version=SCHEMA_VERSION,
            config_fingerprint=current_fp,
            artifacts_dir=str(self.artifacts_dir),
        )

        # Save experiment report to JSON
        report_path = self.reports_dir / f"{model_type}_cv_evaluation.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(experiment_result.to_dict(), f, indent=2)

        print(f"\n  [Summary: {model_type}]")
        print(f"    ROC-AUC : {experiment_result.mean_roc_auc:.4f} +/- {experiment_result.std_roc_auc:.4f} (OOF: {oof_eval.roc_auc:.4f}, 95% CI: {oof_eval.roc_auc_ci_95})")
        print(f"    PR-AUC  : {experiment_result.mean_pr_auc:.4f} +/- {experiment_result.std_pr_auc:.4f} (OOF: {oof_eval.pr_auc:.4f}, 95% CI: {oof_eval.pr_auc_ci_95})")
        print(f"    Brier   : {experiment_result.mean_brier_score:.4f} +/- {experiment_result.std_brier_score:.4f}")
        print(f"    OOF Cov : {experiment_result.oof_coverage_pct:.1f}% ({oof_count:,}/{total_cached:,})")
        print(f"    Saved report -> {report_path}")

        return experiment_result
