"""Unit tests for model evaluation, metrics, threshold sweep, and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from coughvid.evaluation.metrics import (
    compute_bootstrap_confidence_intervals,
    compute_threshold_metrics,
    evaluate_predictions,
    find_optimal_threshold,
    sweep_thresholds,
)


@pytest.fixture
def synthetic_binary_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Generate known ground truth and probabilistic predictions."""
    np.random.seed(42)
    # 100 samples, 20 positives (20% prevalence)
    y_true = np.zeros(100, dtype=int)
    y_true[:20] = 1

    # Well-separated probabilities with some overlap
    y_proba = np.empty(100, dtype=float)
    y_proba[:20] = np.random.beta(5, 2, size=20)  # High prob for positives
    y_proba[20:] = np.random.beta(2, 6, size=80)  # Low prob for negatives
    return y_true, y_proba


def test_compute_threshold_metrics_exact_math():
    """Verify exact confusion matrix and metric calculations on small deterministic input."""
    y_true = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    y_proba = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.9], dtype=float)

    # Threshold at 0.5:
    # y_pred = [0, 0, 0, 1, 1, 1]
    # TP=2, FP=1, TN=3, FN=0
    m5 = compute_threshold_metrics(y_true, y_proba, threshold=0.5)
    assert m5.true_positives == 2
    assert m5.false_positives == 1
    assert m5.true_negatives == 3
    assert m5.false_negatives == 0
    assert m5.sensitivity == 1.0
    assert m5.specificity == 0.75
    assert m5.false_positive_rate == 0.25
    assert round(m5.precision, 4) == round(2.0 / 3.0, 4)
    assert m5.youdens_j == 0.75

    # Threshold at 0.8:
    # y_pred = [0, 0, 0, 0, 0, 1]
    # TP=1, FP=0, TN=4, FN=1
    m8 = compute_threshold_metrics(y_true, y_proba, threshold=0.8)
    assert m8.true_positives == 1
    assert m8.false_positives == 0
    assert m8.true_negatives == 4
    assert m8.false_negatives == 1
    assert m8.sensitivity == 0.5
    assert m8.specificity == 1.0
    assert m8.precision == 1.0


def test_sweep_thresholds_monotonicity_and_bounds(synthetic_binary_predictions):
    """Verify that sensitivity monotonically decreases and specificity increases as threshold rises."""
    y_true, y_proba = synthetic_binary_predictions
    thresholds = np.linspace(0.1, 0.9, 9)
    metrics_list = sweep_thresholds(y_true, y_proba, thresholds=thresholds)

    assert len(metrics_list) == 9
    for i in range(len(metrics_list) - 1):
        # Sensitivity non-increasing
        assert metrics_list[i].sensitivity >= metrics_list[i + 1].sensitivity - 1e-6
        # Specificity non-decreasing
        assert metrics_list[i].specificity <= metrics_list[i + 1].specificity + 1e-6


def test_find_optimal_threshold_strategies(synthetic_binary_predictions):
    """Verify threshold selection by Youden's J, F1, and target sensitivity."""
    y_true, y_proba = synthetic_binary_predictions

    # Youden J
    best_youden = find_optimal_threshold(y_true, y_proba, strategy="youden")
    assert 0.0 < best_youden.threshold < 1.0
    assert best_youden.youdens_j > 0.4

    # F1
    best_f1 = find_optimal_threshold(y_true, y_proba, strategy="f1")
    assert 0.0 < best_f1.threshold < 1.0
    assert best_f1.f1_score > 0.5

    # High sensitivity constraint (>= 0.85)
    best_sens = find_optimal_threshold(y_true, y_proba, strategy="target_sensitivity", min_sensitivity=0.85)
    assert best_sens.sensitivity >= 0.85


def test_bootstrap_confidence_intervals_bounds(synthetic_binary_predictions):
    """Verify non-parametric bootstrap confidence intervals are bounded and valid."""
    y_true, y_proba = synthetic_binary_predictions
    cis = compute_bootstrap_confidence_intervals(y_true, y_proba, n_bootstraps=200, random_seed=42)

    assert "roc_auc_ci" in cis
    assert "pr_auc_ci" in cis
    assert "brier_score_ci" in cis

    roc_low, roc_high = cis["roc_auc_ci"]
    assert 0.0 <= roc_low <= roc_high <= 1.0

    pr_low, pr_high = cis["pr_auc_ci"]
    assert 0.0 <= pr_low <= pr_high <= 1.0

    brier_low, brier_high = cis["brier_score_ci"]
    assert 0.0 <= brier_low <= brier_high <= 1.0


def test_evaluate_predictions_complete_report(synthetic_binary_predictions):
    """Verify complete ModelEvaluationReport fields and types."""
    y_true, y_proba = synthetic_binary_predictions
    report = evaluate_predictions(y_true, y_proba, n_bootstraps=150)

    assert report.num_samples == 100
    assert report.num_positive == 20
    assert report.num_negative == 80
    assert report.prevalence == 0.2
    assert 0.5 < report.roc_auc <= 1.0
    assert 0.2 < report.pr_auc <= 1.0
    assert 0.0 <= report.brier_score <= 1.0
    assert report.operating_threshold > 0.0
    assert report.operating_metrics.sensitivity > 0.0
