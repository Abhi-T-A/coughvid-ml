"""Baseline machine learning models for acoustic cough classification.

Implements scikit-learn compatible pipelines with strict preprocessing isolation:
- Median imputation fitted strictly on training data.
- Feature standardization (Z-score) fitted strictly on training data.
- Class imbalance handling (balanced class weighting vs. train-only oversampling).
- Standard baselines:
  1. Regularized Logistic Regression (L2)
  2. Balanced Random Forest
  3. Gradient Boosted Trees (XGBoost)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def create_baseline_pipeline(
    model_type: str = "logistic_regression",
    random_seed: int = 42,
    class_weight: Optional[str | Dict[int, float]] = "balanced",
    scale_pos_weight: Optional[float] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Create a leakage-free Scikit-Learn Pipeline with preprocessing and classifier.

    Parameters
    ----------
    model_type : {"logistic_regression", "random_forest", "xgboost"}, default="logistic_regression"
        Model architecture identifier.
    random_seed : int, default=42
        Random seed for reproducible training.
    class_weight : str or dict, default="balanced"
        Class weighting strategy for Logistic Regression and Random Forest.
    scale_pos_weight : float, optional
        Positive class scale factor for XGBoost (typically n_neg / n_pos).
    hyperparameters : dict, optional
        Custom model parameters overriding defaults.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn Pipeline encapsulating imputer, scaler, and classifier.
    """
    params = hyperparameters or {}

    # 1. Pipeline steps: Imputer + Scaler
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # 2. Classifier Selection
    if model_type == "logistic_regression":
        lr_kwargs = {
            "C": params.get("C", 0.1),
            "solver": params.get("solver", "lbfgs"),
            "max_iter": params.get("max_iter", 1000),
            "class_weight": class_weight,
            "random_state": random_seed,
        }
        classifier = LogisticRegression(**lr_kwargs)

    elif model_type == "random_forest":
        rf_kwargs = {
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", 8),
            "min_samples_split": params.get("min_samples_split", 5),
            "min_samples_leaf": params.get("min_samples_leaf", 2),
            "max_features": params.get("max_features", "sqrt"),
            "class_weight": class_weight,
            "random_state": random_seed,
            "n_jobs": params.get("n_jobs", 2),
        }
        classifier = RandomForestClassifier(**rf_kwargs)

    elif model_type == "xgboost":
        spw = scale_pos_weight if scale_pos_weight is not None else 1.0
        xgb_kwargs = {
            "n_estimators": params.get("n_estimators", 150),
            "max_depth": params.get("max_depth", 4),
            "learning_rate": params.get("learning_rate", 0.05),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "random_state": random_seed,
            "n_jobs": params.get("n_jobs", 2),
        }
        classifier = xgb.XGBClassifier(**xgb_kwargs)

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from 'logistic_regression', 'random_forest', 'xgboost'."
        )

    return Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("classifier", classifier),
    ])


def apply_training_only_oversampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform simple random minority oversampling strictly on training fold data.

    Guarantees validation splits are never contaminated by oversampled duplicates.
    """
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0 or n_pos >= n_neg:
        return X_train, y_train

    rng = np.random.RandomState(random_seed)
    oversampled_pos_idx = rng.choice(pos_idx, size=n_neg, replace=True)

    all_idx = np.concatenate([neg_idx, oversampled_pos_idx])
    rng.shuffle(all_idx)

    return X_train[all_idx], y_train[all_idx]
