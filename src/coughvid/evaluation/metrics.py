"""Comprehensive acoustic model evaluation and calibration metrics module.

Computes threshold-independent metrics (PR-AUC, ROC-AUC, Brier score, Log Loss),
threshold-dependent clinical classification metrics (Sensitivity, Specificity,
Precision, FPR, F1, Youden's J), bootstrap confidence intervals, and training-only
threshold sweep curves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


@dataclass
class ThresholdMetrics:
    """Metrics evaluated at a specific decision threshold."""
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    sensitivity: float  # Recall
    specificity: float
    precision: float
    false_positive_rate: float
    f1_score: float
    balanced_accuracy: float
    youdens_j: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelEvaluationReport:
    """Comprehensive evaluation metrics for binary classification."""
    num_samples: int
    num_positive: int
    num_negative: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float
    operating_threshold: float
    operating_metrics: ThresholdMetrics
    roc_auc_ci_95: Tuple[float, float]
    pr_auc_ci_95: Tuple[float, float]
    brier_score_ci_95: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["operating_metrics"] = self.operating_metrics.to_dict()
        return d


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> ThresholdMetrics:
    """Calculate clinical and classification metrics at an exact decision threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_proba : np.ndarray
        Predicted probabilities for the positive class (in [0.0, 1.0]).
    threshold : float, default=0.5
        Decision threshold.

    Returns
    -------
    ThresholdMetrics
        Detailed confusion matrix metrics at the specified threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    bal_acc = float((sensitivity + specificity) / 2.0)
    youdens_j = float(sensitivity + specificity - 1.0)

    return ThresholdMetrics(
        threshold=float(threshold),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        sensitivity=round(sensitivity, 5),
        specificity=round(specificity, 5),
        precision=round(precision, 5),
        false_positive_rate=round(fpr, 5),
        f1_score=round(f1, 5),
        balanced_accuracy=round(bal_acc, 5),
        youdens_j=round(youdens_j, 5),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
) -> List[ThresholdMetrics]:
    """Evaluate classification metrics across a dense grid of candidate thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    return [compute_threshold_metrics(y_true, y_proba, float(t)) for t in thresholds]


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "youden",
    min_sensitivity: float = 0.80,
) -> ThresholdMetrics:
    """Find the optimal decision threshold based on a specified training-only objective.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_proba : np.ndarray
        Predicted probabilities.
    strategy : {"youden", "f1", "target_sensitivity"}, default="youden"
        - "youden": Maximizes Youden's J statistic (Sensitivity + Specificity - 1).
        - "f1": Maximizes F1-Score.
        - "target_sensitivity": Targets Sensitivity >= min_sensitivity while maximizing Specificity.
    min_sensitivity : float, default=0.80
        Target minimum sensitivity for "target_sensitivity" strategy.

    Returns
    -------
    ThresholdMetrics
        Metrics at the selected optimal threshold.
    """
    candidates = sweep_thresholds(y_true, y_proba)

    if strategy == "f1":
        best = max(candidates, key=lambda m: m.f1_score)
    elif strategy == "target_sensitivity":
        # Filter to thresholds meeting target sensitivity
        valid = [m for m in candidates if m.sensitivity >= min_sensitivity]
        if valid:
            best = max(valid, key=lambda m: (m.specificity, m.precision))
        else:
            best = max(candidates, key=lambda m: m.sensitivity)
    else:  # "youden"
        best = max(candidates, key=lambda m: m.youdens_j)

    return best


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Compute non-parametric bootstrap confidence intervals (e.g. 95% CI) for core metrics."""
    rng = np.random.RandomState(random_seed)
    n = len(y_true)

    roc_scores: List[float] = []
    pr_scores: List[float] = []
    brier_scores: List[float] = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_sample = y_true[idx]
        yp_sample = y_proba[idx]

        # Require at least one positive and one negative sample in bootstrap
        if len(np.unique(yt_sample)) < 2:
            continue

        roc_scores.append(roc_auc_score(yt_sample, yp_sample))
        pr_scores.append(average_precision_score(yt_sample, yp_sample))
        brier_scores.append(brier_score_loss(yt_sample, yp_sample))

    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0

    return {
        "roc_auc_ci": (
            round(float(np.percentile(roc_scores, lower_p)), 4),
            round(float(np.percentile(roc_scores, upper_p)), 4),
        ),
        "pr_auc_ci": (
            round(float(np.percentile(pr_scores, lower_p)), 4),
            round(float(np.percentile(pr_scores, upper_p)), 4),
        ),
        "brier_score_ci": (
            round(float(np.percentile(brier_scores, lower_p)), 4),
            round(float(np.percentile(brier_scores, upper_p)), 4),
        ),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: Optional[float] = None,
    threshold_strategy: str = "youden",
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> ModelEvaluationReport:
    """Generate a complete evaluation report for probabilistic predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_proba : np.ndarray
        Predicted probabilities for positive class in [0.0, 1.0].
    threshold : float, optional
        Pre-determined threshold. If None, chosen via threshold_strategy.
    threshold_strategy : str, default="youden"
        Strategy to determine operating threshold if threshold is not provided.
    n_bootstraps : int, default=1000
        Number of bootstrap iterations for 95% confidence intervals.
    random_seed : int, default=42
        Seed for bootstrap sampling.

    Returns
    -------
    ModelEvaluationReport
        Complete metrics and confidence intervals.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    n_total = len(y_true)
    prevalence = float(n_pos) / float(n_total)

    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))

    # Clip probabilities slightly to avoid numerical log(0)
    eps = 1e-15
    y_clipped = np.clip(y_proba, eps, 1.0 - eps)
    ll = float(log_loss(y_true, y_clipped))

    if threshold is not None:
        op_metrics = compute_threshold_metrics(y_true, y_proba, threshold)
    else:
        op_metrics = find_optimal_threshold(y_true, y_proba, strategy=threshold_strategy)

    # Bootstrap 95% Confidence Intervals
    boot_cis = compute_bootstrap_confidence_intervals(
        y_true, y_proba, n_bootstraps=n_bootstraps, random_seed=random_seed
    )

    return ModelEvaluationReport(
        num_samples=n_total,
        num_positive=n_pos,
        num_negative=n_neg,
        prevalence=round(prevalence, 5),
        roc_auc=round(roc_auc, 5),
        pr_auc=round(pr_auc, 5),
        brier_score=round(brier, 5),
        log_loss=round(ll, 5),
        operating_threshold=op_metrics.threshold,
        operating_metrics=op_metrics,
        roc_auc_ci_95=boot_cis["roc_auc_ci"],
        pr_auc_ci_95=boot_cis["pr_auc_ci"],
        brier_score_ci_95=boot_cis["brier_score_ci"],
    )
