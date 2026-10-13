"""Milestone 5 Baseline Model Training and Cross-Validation Benchmark Runner.

Executes 5-fold cross-validation for:
1. Logistic Regression (L2 regularized, balanced class weights)
2. Random Forest (balanced class weights, depth-regularized)
3. XGBoost (gradient boosted trees, scale_pos_weight on fold train)

Performs out-of-fold prediction aggregation, threshold selection on training data only,
and exports comprehensive benchmark metrics and serialized artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coughvid.config import load_config
from coughvid.features.schema import SCHEMA_VERSION, compute_config_fingerprint
from coughvid.models.trainer import CrossValidationTrainer


def run_benchmark(
    models: list[str] | None = None,
    compare_oversampling: bool = False,
) -> None:
    cfg = load_config()
    current_fp = compute_config_fingerprint(cfg)
    reports_dir = Path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ["logistic_regression", "random_forest", "xgboost"]

    print("=" * 80)
    print("COUGHVID ML - Milestone 5: Baseline Models Cross-Validation Benchmark")
    print("=" * 80)
    print(f"Features Schema Version : {SCHEMA_VERSION}")
    print(f"Config Fingerprint      : {current_fp}")
    print(f"Models to Evaluate      : {models}")
    print(f"Compare Oversampling    : {compare_oversampling}\n")

    trainer = CrossValidationTrainer(config=cfg)

    results = {}
    start_time = time.time()

    for model_name in models:
        t0 = time.time()
        res = trainer.train_baseline_model(
            model_type=model_name,
            use_oversampling=False,
        )
        t1 = time.time()
        results[model_name] = res

        if compare_oversampling and model_name == "logistic_regression":
            print(f"\n--- [Comparison Experiment] {model_name} with Minority Oversampling ---")
            res_os = trainer.train_baseline_model(
                model_type=model_name,
                use_oversampling=True,
            )
            results[f"{model_name}_oversampled"] = res_os

    elapsed = time.time() - start_time

    # ----------------------------------------------------
    # BENCHMARK COMPARISON TABLE
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("MILESTONE 5: BASELINE CROSS-VALIDATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':25s} | {'Mean ROC-AUC (CV)':18s} | {'Mean PR-AUC (CV)':17s} | {'Mean Brier':12s} | {'OOF PR-AUC [95% CI]':22s}")
    print("-" * 105)

    summary_records = []
    for name, r in results.items():
        roc_str = f"{r.mean_roc_auc:.4f} +/- {r.std_roc_auc:.4f}"
        pr_str = f"{r.mean_pr_auc:.4f} +/- {r.std_pr_auc:.4f}"
        brier_str = f"{r.mean_brier_score:.4f} +/- {r.std_brier_score:.4f}"
        oof_pr_ci = f"{r.oof_evaluation.pr_auc:.4f} [{r.oof_evaluation.pr_auc_ci_95[0]:.3f}, {r.oof_evaluation.pr_auc_ci_95[1]:.3f}]"

        print(f"{name:25s} | {roc_str:18s} | {pr_str:17s} | {brier_str:12s} | {oof_pr_ci:22s}")

        summary_records.append({
            "model": name,
            "mean_roc_auc": r.mean_roc_auc,
            "std_roc_auc": r.std_roc_auc,
            "mean_pr_auc": r.mean_pr_auc,
            "std_pr_auc": r.std_pr_auc,
            "mean_brier_score": r.mean_brier_score,
            "std_brier_score": r.std_brier_score,
            "oof_roc_auc": r.oof_evaluation.roc_auc,
            "oof_roc_auc_ci_95": r.oof_evaluation.roc_auc_ci_95,
            "oof_pr_auc": r.oof_evaluation.pr_auc,
            "oof_pr_auc_ci_95": r.oof_evaluation.pr_auc_ci_95,
            "oof_brier_score": r.oof_evaluation.brier_score,
            "optimal_thresholds": r.optimal_thresholds,
            "oof_coverage_pct": r.oof_coverage_pct,
            "total_records": r.total_training_records,
        })

    # Save benchmark summary JSON
    summary_path = reports_dir / "baseline_models_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": current_fp,
            "models": summary_records,
            "duration_sec": round(elapsed, 2),
        }, f, indent=2)

    print(f"\nSaved benchmark summary report -> {summary_path}")

    # Threshold analysis display
    print("\n--- Training-Only Operating Threshold Analysis (Out-of-Fold) ---")
    for name, r in results.items():
        print(f"\nModel: {name}")
        for th_name, th_info in r.optimal_thresholds.items():
            print(
                f"  [{th_name:22s}] Threshold: {th_info['threshold']:.3f} | "
                f"Sensitivity: {th_info['sensitivity']*100:.1f}% | "
                f"Specificity: {th_info['specificity']*100:.1f}% | "
                f"Precision: {th_info['precision']*100:.1f}% | "
                f"F1: {th_info['f1_score']:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milestone 5 baseline model training benchmark.")
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest", "xgboost"])
    parser.add_argument("--compare-oversampling", action="store_true", help="Compare with training-only oversampling.")
    args = parser.parse_args()

    run_benchmark(models=args.models, compare_oversampling=args.compare_oversampling)
