"""Benchmark 1 epoch of CoughAudioCNN training and evaluation on Fold 0 on CPU.

Records exact runtime, batches/sec, and memory footprint before starting 5-fold cross-validation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coughvid.config import load_config
from coughvid.evaluation.metrics import evaluate_predictions
from coughvid.models.cnn import CoughAudioCNN
from coughvid.models.dataset import SpecAugment, SpectrogramDataset, preload_spectrogram_dict


def run_fold0_benchmark(batch_size: int = 64, num_workers: int = 0) -> None:
    cfg = load_config()
    splits_dir = Path(cfg.paths.data_dir) / "splits"
    features_dir = Path(cfg.paths.data_dir) / "features"

    train_manifest = splits_dir / "fold_0_train.csv"
    val_manifest = splits_dir / "fold_0_val.csv"
    npz_path = features_dir / "train_spectrograms.npz"

    print("=" * 80)
    print("COUGHVID ML - Milestone 6: Phase C CPU Benchmark (Fold 0, 1 Epoch)")
    print("=" * 80)
    print(f"Batch Size     : {batch_size}")
    print(f"Num Workers    : {num_workers}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device         : CPU (Intra-op Threads: {torch.get_num_threads()})")

    # 1. Load manifests
    df_train = pd.read_csv(train_manifest)
    df_val = pd.read_csv(val_manifest)

    train_labels = dict(zip(df_train["uuid"].astype(str), df_train["label"].astype(int)))
    val_labels = dict(zip(df_val["uuid"].astype(str), df_val["label"].astype(int)))

    # 2. Preload spectrogram store into RAM
    t_load_start = time.time()
    spec_store = preload_spectrogram_dict(npz_path)
    t_load_elapsed = time.time() - t_load_start
    print(f"Loaded {len(spec_store):,} spectrograms into RAM in {t_load_elapsed:.2f} s")

    # 3. Create datasets and dataloaders
    train_uuids = [u for u in df_train["uuid"].astype(str) if u in spec_store]
    val_uuids = [u for u in df_val["uuid"].astype(str) if u in spec_store]

    print(f"Fold 0 Train Samples: {len(train_uuids):,} | Val Samples: {len(val_uuids):,}")

    augmenter = SpecAugment(max_freq_mask=8, max_time_mask=16, fill_value=-1.0)
    train_dataset = SpectrogramDataset(
        uuids=train_uuids,
        labels=train_labels,
        spectrogram_source=spec_store,
        is_train=True,
        spec_augment=augmenter,
    )
    val_dataset = SpectrogramDataset(
        uuids=val_uuids,
        labels=val_labels,
        spectrogram_source=spec_store,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # 4. Compute fold-isolated pos_weight
    n_pos = sum(train_labels[u] for u in train_uuids)
    n_neg = len(train_uuids) - n_pos
    pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    print(f"Fold 0 Positives: {n_pos:,} | Negatives: {n_neg:,} | Calculated pos_weight: {pos_weight:.2f}")

    # 5. Initialize model, optimizer, criterion
    torch.manual_seed(42)
    model = CoughAudioCNN()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"Model Parameter Count: {model.count_parameters():,}")
    print("\n--- Starting 1-Epoch Training on Fold 0 ---")

    # 6. Train 1 Epoch
    model.train()
    t_train_start = time.time()
    running_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (x, y, _) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(x).squeeze(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    t_train_elapsed = time.time() - t_train_start
    avg_train_loss = running_loss / num_batches
    throughput = len(train_uuids) / t_train_elapsed

    print(f"Epoch 1 Training Complete in {t_train_elapsed:.2f} s")
    print(f"  Avg Train Loss : {avg_train_loss:.4f}")
    print(f"  Throughput     : {throughput:.1f} samples/sec ({num_batches / t_train_elapsed:.1f} batches/sec)")

    # 7. Evaluate on Fold 0 Validation
    model.eval()
    t_val_start = time.time()
    val_probs: list[float] = []
    val_targets: list[int] = []

    with torch.no_grad():
        for x, y, _ in val_loader:
            probs = model.predict_proba(x).squeeze(-1)
            val_probs.extend(probs.cpu().numpy().tolist())
            val_targets.extend(y.cpu().numpy().astype(int).tolist())

    t_val_elapsed = time.time() - t_val_start
    report = evaluate_predictions(np.array(val_targets), np.array(val_probs), n_bootstraps=200)

    print(f"\nFold 0 Validation Complete in {t_val_elapsed:.2f} s")
    print(f"  Val ROC-AUC    : {report.roc_auc:.4f}")
    print(f"  Val Avg Prec   : {report.pr_auc:.4f} (Random Baseline = {report.prevalence:.4f})")
    print(f"  Val Brier Score: {report.brier_score:.4f}")

    total_time = t_train_elapsed + t_val_elapsed
    print(f"\nTotal Epoch Time (Train + Val): {total_time:.2f} s")
    print(f"Estimated 10 Epochs / Fold   : {total_time * 10 / 60:.1f} minutes")
    print(f"Estimated 5 Folds Total       : {total_time * 10 * 5 / 60:.1f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    run_fold0_benchmark(batch_size=64, num_workers=0)
