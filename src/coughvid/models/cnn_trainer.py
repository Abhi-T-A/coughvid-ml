"""PyTorch 2D CNN Cross-Validation Trainer and Out-of-Fold Evaluation Runner.

Implements leak-free 5-fold cross-validation for CoughAudioCNN:
- Reuses existing fold manifests without modification.
- Computes pos_weight strictly on each fold's training split.
- Monitors validation Average Precision (AP) for checkpoint selection and early stopping.
- Explicitly restores the best checkpoint prior to out-of-fold prediction generation.
- Guarantees 100% OOF prediction coverage (8,964 UUIDs) with zero missing and zero duplicates.
- Evaluates Brier score and probability calibration on the unweighted validation distribution.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader

from coughvid.config import AppConfig, load_config
from coughvid.evaluation.metrics import (
    ModelEvaluationReport,
    evaluate_predictions,
    find_optimal_threshold,
)
from coughvid.features.schema import SCHEMA_VERSION, compute_config_fingerprint
from coughvid.models.cnn import CoughAudioCNN
from coughvid.models.dataset import SpecAugment, SpectrogramDataset, preload_spectrogram_dict

logger = logging.getLogger(__name__)


@dataclass
class EpochLog:
    """Performance metrics logged for a single training epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    val_roc_auc: float
    val_average_precision: float
    val_brier_score: float
    duration_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CNNFoldResult:
    """Results achieved on a single cross-validation fold after best-checkpoint restoration."""
    fold_index: int
    train_samples: int
    val_samples: int
    train_positives: int
    val_positives: int
    pos_weight: float
    best_epoch: int
    best_val_ap: float
    roc_auc: float
    average_precision: float
    brier_score: float
    log_loss: float
    epoch_history: List[EpochLog]
    checkpoint_path: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["epoch_history"] = [e.to_dict() for e in self.epoch_history]
        return d


@dataclass
class CNNExperimentResult:
    """Comprehensive results of a 5-fold CNN cross-validation experiment."""
    model_name: str
    n_splits: int
    total_training_records: int
    oof_predictions_count: int
    oof_coverage_pct: float
    fold_results: List[CNNFoldResult]
    mean_roc_auc: float
    std_roc_auc: float
    mean_average_precision: float
    std_average_precision: float
    mean_brier_score: float
    std_brier_score: float
    oof_evaluation: ModelEvaluationReport
    optimal_thresholds: Dict[str, Any]
    schema_version: str
    config_fingerprint: str
    artifacts_dir: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fold_results"] = [f.to_dict() for f in self.fold_results]
        d["oof_evaluation"] = self.oof_evaluation.to_dict()
        return d


class CNNCrossValidationTrainer:
    """Orchestrates 5-fold cross-validation for CoughAudioCNN on CPU."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        artifacts_dir: Optional[Path | str] = None,
        reports_dir: Optional[Path | str] = None,
        spec_store: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.config = config or load_config()
        self.artifacts_dir = Path(artifacts_dir or (Path(self.config.paths.artifacts_dir) / "models"))
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.reports_dir = Path(reports_dir or self.config.paths.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = Path(self.config.paths.data_dir)
        self.splits_dir = self.data_dir / "splits"
        self.features_dir = self.data_dir / "features"

        self.npz_path = self.features_dir / "train_spectrograms.npz"
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Spectrogram store not found at {self.npz_path}. Run extraction first.")

        # In-memory spectrogram cache to eliminate disk I/O overhead during CPU epochs
        self.spec_store = spec_store or preload_spectrogram_dict(self.npz_path)

    def train_single_fold(
        self,
        fold_idx: int,
        max_epochs: int = 10,
        patience: int = 3,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        random_seed: int = 42,
    ) -> Tuple[CNNFoldResult, Dict[str, float]]:
        """Train CoughAudioCNN on a single cross-validation fold.

        Parameters
        ----------
        fold_idx : int
            Index of the fold (0 to 4).
        max_epochs : int, default=10
            Maximum training epochs.
        patience : int, default=3
            Early stopping patience monitoring validation Average Precision.
        batch_size : int, default=64
            Batch size for DataLoader.
        lr : float, default=1e-3
            AdamW initial learning rate.
        weight_decay : float, default=1e-4
            AdamW weight decay.
        random_seed : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        Tuple[CNNFoldResult, Dict[str, float]]
            Fold result object and mapping of {uuid: out_of_fold_predicted_probability}.
        """
        train_manifest = self.splits_dir / f"fold_{fold_idx}_train.csv"
        val_manifest = self.splits_dir / f"fold_{fold_idx}_val.csv"

        if not train_manifest.exists() or not val_manifest.exists():
            raise FileNotFoundError(f"Manifests for fold {fold_idx} missing in {self.splits_dir}!")

        df_train = pd.read_csv(train_manifest)
        df_val = pd.read_csv(val_manifest)

        train_labels = dict(zip(df_train["uuid"].astype(str), df_train["label"].astype(int)))
        val_labels = dict(zip(df_val["uuid"].astype(str), df_val["label"].astype(int)))

        # Leakage check: zero overlap
        overlap = set(train_labels.keys()) & set(val_labels.keys())
        assert len(overlap) == 0, f"Leakage detected in fold {fold_idx}! Overlap: {overlap}"

        # Match with available spectrograms in cache
        train_uuids = [u for u in df_train["uuid"].astype(str) if u in self.spec_store]
        val_uuids = [u for u in df_val["uuid"].astype(str) if u in self.spec_store]

        # Compute fold-isolated pos_weight strictly on training split
        n_pos = sum(train_labels[u] for u in train_uuids)
        n_neg = len(train_uuids) - n_pos
        pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

        val_pos = sum(val_labels[u] for u in val_uuids)

        # Datasets & DataLoaders
        augmenter = SpecAugment(
            max_freq_mask=8,
            max_time_mask=16,
            fill_value=-1.0,
            rng=np.random.RandomState(random_seed + fold_idx),
        )
        train_ds = SpectrogramDataset(
            uuids=train_uuids,
            labels=train_labels,
            spectrogram_source=self.spec_store,
            is_train=True,
            spec_augment=augmenter,
        )
        val_ds = SpectrogramDataset(
            uuids=val_uuids,
            labels=val_labels,
            spectrogram_source=self.spec_store,
            is_train=False,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Initialize model, loss, optimizer
        torch.manual_seed(random_seed + fold_idx)
        model = CoughAudioCNN()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        val_criterion = nn.BCEWithLogitsLoss()  # Unweighted validation loss for objective calibration
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        checkpoint_path = self.artifacts_dir / f"cnn_fold_{fold_idx}_best.pt"

        best_val_ap = -1.0
        best_epoch = 1
        patience_left = patience
        epoch_history: List[EpochLog] = []

        print(f"\n--- Training CNN Fold {fold_idx} ({len(train_uuids):,} train, {len(val_uuids):,} val) ---")
        print(f"  Class Weights: pos_weight={pos_weight:.2f} (pos={n_pos:,}, neg={n_neg:,})")

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            # Train pass
            model.train()
            train_loss_sum = 0.0
            for x_b, y_b, _ in train_loader:
                optimizer.zero_grad()
                logits = model(x_b).squeeze(-1)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()

            scheduler.step()
            train_loss = train_loss_sum / len(train_loader)

            # Validation pass
            model.eval()
            val_loss_sum = 0.0
            val_probs: List[float] = []
            val_targets: List[int] = []

            with torch.no_grad():
                for x_b, y_b, _ in val_loader:
                    logits = model(x_b).squeeze(-1)
                    v_loss = val_criterion(logits, y_b)
                    val_loss_sum += v_loss.item()
                    probs = torch.sigmoid(logits)
                    val_probs.extend(probs.cpu().numpy().tolist())
                    val_targets.extend(y_b.cpu().numpy().astype(int).tolist())

            val_loss = val_loss_sum / len(val_loader)
            dur = time.time() - t0

            # Metrics
            targets_arr = np.array(val_targets)
            probs_arr = np.array(val_probs)
            v_roc = float(roc_auc_score(targets_arr, probs_arr))
            v_ap = float(average_precision_score(targets_arr, probs_arr))
            v_brier = float(brier_score_loss(targets_arr, probs_arr))

            log_entry = EpochLog(
                epoch=epoch,
                train_loss=round(train_loss, 4),
                val_loss=round(val_loss, 4),
                val_roc_auc=round(v_roc, 4),
                val_average_precision=round(v_ap, 4),
                val_brier_score=round(v_brier, 4),
                duration_sec=round(dur, 2),
            )
            epoch_history.append(log_entry)

            is_best = v_ap > best_val_ap
            if is_best:
                best_val_ap = v_ap
                best_epoch = epoch
                patience_left = patience
                torch.save(model.state_dict(), checkpoint_path)
                star = " [*BEST]"
            else:
                patience_left -= 1
                star = ""

            print(
                f"  Epoch {epoch:2d}/{max_epochs:2d} | TrainLoss: {train_loss:.4f} | "
                f"ValLoss: {val_loss:.4f} | Val ROC: {v_roc:.4f} | Val AP: {v_ap:.4f} | "
                f"Brier: {v_brier:.4f} ({dur:.1f}s){star}"
            )

            if patience_left <= 0:
                print(f"  Early stopping triggered at epoch {epoch} (patience exhausted).")
                break

        # EXPLICIT BEST CHECKPOINT RESTORATION BEFORE PREDICTION GENERATION
        print(f"  Restoring best checkpoint from epoch {best_epoch} ({checkpoint_path.name})...")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        model.eval()

        final_probs: List[float] = []
        final_targets: List[int] = []
        final_uuids: List[str] = []

        with torch.no_grad():
            for x_b, y_b, u_b in val_loader:
                probs = model.predict_proba(x_b).squeeze(-1)
                final_probs.extend(probs.cpu().numpy().tolist())
                final_targets.extend(y_b.cpu().numpy().astype(int).tolist())
                final_uuids.extend(list(u_b))

        y_true_final = np.array(final_targets)
        y_prob_final = np.array(final_probs)
        final_roc = float(roc_auc_score(y_true_final, y_prob_final))
        final_ap = float(average_precision_score(y_true_final, y_prob_final))
        final_brier = float(brier_score_loss(y_true_final, y_prob_final))
        # Clip probabilities slightly to avoid log(0) in log_loss
        clipped_p = np.clip(y_prob_final, 1e-15, 1.0 - 1e-15)
        final_logloss = float(log_loss(y_true_final, clipped_p))

        fold_res = CNNFoldResult(
            fold_index=fold_idx,
            train_samples=len(train_uuids),
            val_samples=len(val_uuids),
            train_positives=n_pos,
            val_positives=val_pos,
            pos_weight=round(pos_weight, 2),
            best_epoch=best_epoch,
            best_val_ap=round(best_val_ap, 4),
            roc_auc=round(final_roc, 4),
            average_precision=round(final_ap, 4),
            brier_score=round(final_brier, 4),
            log_loss=round(final_logloss, 4),
            epoch_history=epoch_history,
            checkpoint_path=str(checkpoint_path),
        )

        oof_fold_dict = dict(zip(final_uuids, final_probs))
        return fold_res, oof_fold_dict

    def run_cross_validation(
        self,
        max_epochs: int = 10,
        patience: int = 3,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> CNNExperimentResult:
        """Run complete 5-fold cross-validation and compile out-of-fold predictions."""
        n_splits = self.config.splits.n_splits
        all_fold_results: List[CNNFoldResult] = []
        oof_records: List[Dict[str, Any]] = []

        # Load train manifest to track full coverage
        full_train_df = pd.read_csv(self.splits_dir / "train.csv")
        eligible_uuids = set(full_train_df["uuid"].astype(str)) & set(self.spec_store.keys())
        total_eligible = len(eligible_uuids)

        print("=" * 80)
        print("COUGHVID ML - Milestone 6: CoughAudioCNN 5-Fold Cross-Validation")
        print("=" * 80)
        print(f"Eligible Training Recordings: {total_eligible:,}")
        print(f"Spectrogram Store           : {self.npz_path}")
        print(f"Epoch Budget per Fold       : {max_epochs} (Patience = {patience})")
        print(f"Batch Size                  : {batch_size}")
        print(f"Initial LR / Weight Decay   : {lr} / {weight_decay}")

        for fold_idx in range(n_splits):
            f_res, fold_oof_dict = self.train_single_fold(
                fold_idx=fold_idx,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
            )
            all_fold_results.append(f_res)

            # Read val manifest to get ground truth labels
            val_df = pd.read_csv(self.splits_dir / f"fold_{fold_idx}_val.csv")
            val_labels = dict(zip(val_df["uuid"].astype(str), val_df["label"].astype(int)))

            for u, prob in fold_oof_dict.items():
                oof_records.append({
                    "uuid": u,
                    "fold": fold_idx,
                    "y_true": val_labels[u],
                    "y_prob": float(prob),
                })

        # Out-of-fold integrity assertions
        oof_df = pd.DataFrame(oof_records).sort_values("uuid").reset_index(drop=True)
        oof_csv_path = self.artifacts_dir / "cnn_oof_predictions.csv"
        oof_df.to_csv(oof_csv_path, index=False)

        oof_count = len(oof_df)
        oof_unique_count = oof_df["uuid"].nunique()

        # Guarantee exactly one prediction per eligible training recording
        assert oof_count == total_eligible, (
            f"OOF count mismatch! Expected {total_eligible}, got {oof_count}"
        )
        assert oof_unique_count == total_eligible, (
            f"Duplicate UUIDs detected in OOF predictions! Unique: {oof_unique_count}, Total: {oof_count}"
        )

        oof_coverage_pct = round((float(oof_count) / float(total_eligible)) * 100.0, 2)

        # Pooled OOF Evaluation (1,000 bootstrap iterations)
        print("\nEvaluating pooled out-of-fold metrics and bootstrap confidence intervals...")
        oof_eval = evaluate_predictions(
            y_true=oof_df["y_true"].to_numpy(),
            y_proba=oof_df["y_prob"].to_numpy(),
            n_bootstraps=1000,
            random_seed=self.config.training.random_seed,
        )

        # Operating thresholds strictly on training OOF predictions
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

        # Fold-level summary statistics
        roc_list = [f.roc_auc for f in all_fold_results]
        ap_list = [f.average_precision for f in all_fold_results]
        brier_list = [f.brier_score for f in all_fold_results]

        current_fp = compute_config_fingerprint(self.config)

        exp_result = CNNExperimentResult(
            model_name="cnn",
            n_splits=n_splits,
            total_training_records=total_eligible,
            oof_predictions_count=oof_count,
            oof_coverage_pct=oof_coverage_pct,
            fold_results=all_fold_results,
            mean_roc_auc=round(float(np.mean(roc_list)), 4),
            std_roc_auc=round(float(np.std(roc_list)), 4),
            mean_average_precision=round(float(np.mean(ap_list)), 4),
            std_average_precision=round(float(np.std(ap_list)), 4),
            mean_brier_score=round(float(np.mean(brier_list)), 4),
            std_brier_score=round(float(np.std(brier_list)), 4),
            oof_evaluation=oof_eval,
            optimal_thresholds=optimal_thresholds,
            schema_version=SCHEMA_VERSION,
            config_fingerprint=current_fp,
            artifacts_dir=str(self.artifacts_dir),
        )

        # Save experiment report to JSON
        report_path = self.reports_dir / "cnn_cv_evaluation.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(exp_result.to_dict(), f, indent=2)

        print("\n" + "=" * 80)
        print("COUGHAUDIOCNN CROSS-VALIDATION COMPLETE")
        print("=" * 80)
        print(f"  Mean ROC-AUC (CV)     : {exp_result.mean_roc_auc:.4f} +/- {exp_result.std_roc_auc:.4f}")
        print(f"  Pooled OOF ROC-AUC    : {oof_eval.roc_auc:.4f} [95% CI: {oof_eval.roc_auc_ci_95[0]:.4f}, {oof_eval.roc_auc_ci_95[1]:.4f}]")
        print(f"  Mean Avg Precision    : {exp_result.mean_average_precision:.4f} +/- {exp_result.std_average_precision:.4f}")
        print(f"  Pooled OOF Avg Prec   : {oof_eval.pr_auc:.4f} [95% CI: {oof_eval.pr_auc_ci_95[0]:.4f}, {oof_eval.pr_auc_ci_95[1]:.4f}]")
        print(f"  Mean Brier Score      : {exp_result.mean_brier_score:.4f} +/- {exp_result.std_brier_score:.4f}")
        print(f"  OOF Coverage          : {exp_result.oof_coverage_pct:.1f}% ({oof_count:,}/{total_eligible:,})")
        print(f"  Saved Checkpoints     : {self.artifacts_dir}/cnn_fold_*.pt")
        print(f"  Saved OOF Predictions : {oof_csv_path}")
        print(f"  Saved JSON Report     : {report_path}")

        return exp_result
