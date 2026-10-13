"""Baseline model training and cross-validation module."""

from coughvid.models.baseline import (
    apply_training_only_oversampling,
    create_baseline_pipeline,
)
from coughvid.models.trainer import (
    CrossValidationExperimentResult,
    CrossValidationTrainer,
    FoldMetrics,
)

__all__ = [
    "create_baseline_pipeline",
    "apply_training_only_oversampling",
    "CrossValidationTrainer",
    "FoldMetrics",
    "CrossValidationExperimentResult",
]
