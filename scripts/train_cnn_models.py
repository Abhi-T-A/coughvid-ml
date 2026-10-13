"""Milestone 6 CoughAudioCNN Benchmark and Cross-Validation Runner.

Executes 5-fold cross-validation for the 2D CNN spectrogram baseline, restores
best checkpoints based on validation Average Precision, compiles out-of-fold predictions,
and generates a comparative evaluation against Milestone 5 tabular baselines.
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
from coughvid.models.cnn_trainer import CNNCrossValidationTrainer


def run_cnn_benchmark(
    max_epochs: int = 10,
    patience: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> None:
    cfg = load_config()
    reports_dir = Path(cfg.paths.reports_dir)

    t0 = time.time()
    trainer = CNNCrossValidationTrainer(config=cfg)
    cnn_result = trainer.run_cross_validation(
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
    )
    elapsed_total = time.time() - t0

    # --------------------------------------------------------------------------
    # COMPARATIVE BENCHMARK WITH TABULAR BASELINES (MILESTONE 5)
    # --------------------------------------------------------------------------
    baseline_summary_path = reports_dir / "baseline_models_benchmark_summary.json"
    if not baseline_summary_path.exists():
        print(f"\nWarning: Milestone 5 baseline report not found at {baseline_summary_path}")
        return

    with open(baseline_summary_path, "r", encoding="utf-8") as f:
        baseline_summary = json.load(f)

    print("\n" + "=" * 115)
    print("MILESTONE 6: MULTI-MODAL MODEL BENCHMARK COMPARISON (TABULAR VS. SPECTROGRAM 2D CNN)")
    print("=" * 115)
    print(
        f"{'Model Architecture':30s} | {'Mean ROC-AUC (CV)':18s} | "
        f"{'Mean Avg Precision (CV)':22s} | {'Mean Brier':12s} | {'Pooled OOF AP [95% CI]':24s}"
    )
    print("-" * 115)

    all_models_summary = []

    # Display tabular baselines
    for m in baseline_summary.get("models", []):
        m_name = m["model"]
        roc_str = f"{m['mean_roc_auc']:.4f} +/- {m['std_roc_auc']:.4f}"
        pr_str = f"{m['mean_pr_auc']:.4f} +/- {m['std_pr_auc']:.4f}"
        brier_str = f"{m['mean_brier_score']:.4f} +/- {m['std_brier_score']:.4f}"
        oof_ap_ci = f"{m['oof_pr_auc']:.4f} [{m['oof_pr_auc_ci_95'][0]:.3f}, {m['oof_pr_auc_ci_95'][1]:.3f}]"

        print(f"{m_name:30s} | {roc_str:18s} | {pr_str:22s} | {brier_str:12s} | {oof_ap_ci:24s}")
        all_models_summary.append(m)

    # Display CNN baseline
    cnn_roc_str = f"{cnn_result.mean_roc_auc:.4f} +/- {cnn_result.std_roc_auc:.4f}"
    cnn_ap_str = f"{cnn_result.mean_average_precision:.4f} +/- {cnn_result.std_average_precision:.4f}"
    cnn_brier_str = f"{cnn_result.mean_brier_score:.4f} +/- {cnn_result.std_brier_score:.4f}"
    cnn_oof_ap_ci = (
        f"{cnn_result.oof_evaluation.pr_auc:.4f} "
        f"[{cnn_result.oof_evaluation.pr_auc_ci_95[0]:.3f}, {cnn_result.oof_evaluation.pr_auc_ci_95[1]:.3f}]"
    )

    print(
        f"{'cough_audio_cnn (2D Spectrogram)':30s} | {cnn_roc_str:18s} | "
        f"{cnn_ap_str:22s} | {cnn_brier_str:12s} | {cnn_oof_ap_ci:24s}"
    )

    all_models_summary.append({
        "model": "cough_audio_cnn",
        "mean_roc_auc": cnn_result.mean_roc_auc,
        "std_roc_auc": cnn_result.std_roc_auc,
        "mean_pr_auc": cnn_result.mean_average_precision,
        "std_pr_auc": cnn_result.std_average_precision,
        "mean_brier_score": cnn_result.mean_brier_score,
        "std_brier_score": cnn_result.std_brier_score,
        "oof_roc_auc": cnn_result.oof_evaluation.roc_auc,
        "oof_roc_auc_ci_95": cnn_result.oof_evaluation.roc_auc_ci_95,
        "oof_pr_auc": cnn_result.oof_evaluation.pr_auc,
        "oof_pr_auc_ci_95": cnn_result.oof_evaluation.pr_auc_ci_95,
        "oof_brier_score": cnn_result.oof_evaluation.brier_score,
        "optimal_thresholds": cnn_result.optimal_thresholds,
        "oof_coverage_pct": cnn_result.oof_coverage_pct,
        "total_records": cnn_result.total_training_records,
    })

    # Save combined comparison report (preserving baseline_models_benchmark_summary.json)
    combined_report_path = reports_dir / "combined_models_benchmark_summary.json"
    with open(combined_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_population": "8,964 approved training recordings (5-fold CV)",
            "models": all_models_summary,
            "cnn_total_duration_sec": round(elapsed_total, 2),
        }, f, indent=2)

    print(f"\nSaved combined multi-modal benchmark report -> {combined_report_path}")

    # Threshold analysis display for CNN
    print("\n--- Training-Only Operating Threshold Analysis for CNN (Out-of-Fold) ---")
    for th_name, th_info in cnn_result.optimal_thresholds.items():
        print(
            f"  [{th_name:22s}] Threshold: {th_info['threshold']:.3f} | "
            f"Sensitivity: {th_info['sensitivity']*100:.1f}% | "
            f"Specificity: {th_info['specificity']*100:.1f}% | "
            f"Precision: {th_info['precision']*100:.1f}% | "
            f"F1: {th_info['f1_score']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Milestone 6 CNN benchmark.")
    parser.add_argument("--max-epochs", type=int, default=10, help="Max training epochs per fold.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    args = parser.parse_args()

    run_cnn_benchmark(
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
