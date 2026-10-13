"""Milestone 7 Production Model Training and Single-Pass Holdout Evaluation Runner.

Usage:
  # Phase 1: Fit and serialize production models only (does NOT access test labels)
  python scripts/evaluate_holdout.py --fit-only

  # Phase 2: Execute single-pass holdout evaluation (after approval)
  python scripts/evaluate_holdout.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coughvid.config import load_config
from coughvid.models.evaluator import HoldoutEvaluator, LOCKED_OPERATING_THRESHOLDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 7 Holdout Evaluation and Governance.")
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Fit and serialize production models on training data only without accessing test labels.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for 95% confidence intervals.",
    )
    args = parser.parse_args()

    cfg = load_config()
    evaluator = HoldoutEvaluator(config=cfg)

    candidates = ["xgboost", "random_forest"]

    print("=" * 90)
    print("COUGHVID ML - Milestone 7: Production Model Training & Holdout Evaluation")
    print("=" * 90)
    print(f"Candidates to Process : {candidates}")
    print(f"Fit-Only Mode (Locked): {args.fit_only}")
    print(f"Pre-Locked Thresholds : {LOCKED_OPERATING_THRESHOLDS}")

    # 1. Fit production models on the complete training set
    fitted_pipelines = {}
    for model_name in candidates:
        pipe, model_path = evaluator.fit_and_serialize_production_model(model_type=model_name)
        fitted_pipelines[model_name] = pipe

    if args.fit_only:
        print("\n[PHASE 1 COMPLETE] Production models successfully fit and serialized.")
        print("Holdout test set remains STRICTLY LOCKED.")
        return

    # 2. Single-Pass Holdout Evaluation (Only executed when fit-only is False)
    print("\n" + "=" * 90)
    print("UNLOCKING HOLDOUT TEST LABELS FOR A SINGLE EVALUATION PASS")
    print("=" * 90)

    test_results = {}
    for model_name in candidates:
        res = evaluator.evaluate_holdout_test_set(
            model_type=model_name,
            pipeline=fitted_pipelines[model_name],
            n_bootstraps=args.n_bootstraps,
        )
        test_results[model_name] = res

    # 3. Print Holdout Comparison Summary
    print("\n" + "=" * 110)
    print("MILESTONE 7: FINAL HOLDOUT TEST EVALUATION RESULTS")
    print("=" * 110)
    print(
        f"{'Model Candidate':25s} | {'Test ROC-AUC [95% CI]':25s} | "
        f"{'Test Avg Prec [95% CI]':26s} | {'Test Brier [95% CI]':25s}"
    )
    print("-" * 110)

    report_records = []
    for name, r in test_results.items():
        roc_str = f"{r.roc_auc:.4f} [{r.roc_auc_ci_95[0]:.3f}, {r.roc_auc_ci_95[1]:.3f}]"
        ap_str = f"{r.average_precision:.4f} [{r.average_precision_ci_95[0]:.3f}, {r.average_precision_ci_95[1]:.3f}]"
        brier_str = f"{r.brier_score:.4f} [{r.brier_score_ci_95[0]:.3f}, {r.brier_score_ci_95[1]:.3f}]"

        print(f"{name:25s} | {roc_str:25s} | {ap_str:26s} | {brier_str:25s}")
        report_records.append(r.to_dict())

    # 4. Print Fixed-Threshold Confusion Matrix & Clinical Screening Metrics
    for name, r in test_results.items():
        print(f"\n--- Model: {name} (Pre-Locked Decision Thresholds) ---")
        for th_name, m in r.threshold_evaluations.items():
            print(
                f"  [{th_name:22s}] Threshold: {m['threshold']:.3f} | "
                f"TP={m['true_positives']}, FP={m['false_positives']}, "
                f"TN={m['true_negatives']}, FN={m['false_negatives']} | "
                f"Sens: {m['sensitivity']*100:.1f}%, Spec: {m['specificity']*100:.1f}%, "
                f"Prec: {m['precision']*100:.1f}%, F1: {m['f1_score']:.3f}"
            )

    # 5. Export Holdout Report JSON
    report_path = Path(cfg.paths.reports_dir) / "holdout_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_population": "1,595 approved holdout test recordings",
            "models": report_records,
        }, f, indent=2)

    print(f"\nSaved final holdout evaluation report -> {report_path}")


if __name__ == "__main__":
    main()
