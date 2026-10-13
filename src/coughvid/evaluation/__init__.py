"""Model evaluation, metrics, threshold sweep, and confidence interval estimation module."""

from coughvid.evaluation.metrics import (
    ModelEvaluationReport,
    ThresholdMetrics,
    compute_bootstrap_confidence_intervals,
    compute_threshold_metrics,
    evaluate_predictions,
    find_optimal_threshold,
    sweep_thresholds,
)

__all__ = [
    "ThresholdMetrics",
    "ModelEvaluationReport",
    "compute_threshold_metrics",
    "sweep_thresholds",
    "find_optimal_threshold",
    "compute_bootstrap_confidence_intervals",
    "evaluate_predictions",
]
