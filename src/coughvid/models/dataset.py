"""PyTorch dataset and audio spectrogram augmentation module for COUGHVID.

Provides UUID-keyed retrieval of 64x128 Log-Mel spectrograms, strict isolation
of training-time augmentations (SpecAugment), and deterministic validation/inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SpecAugment:
    """Time and Frequency Masking for 2D Audio Spectrograms.

    Applies random horizontal and vertical rectangular masks with a baseline fill value.
    Applied exclusively during training passes.
    """

    def __init__(
        self,
        max_freq_mask: int = 8,
        max_time_mask: int = 16,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        fill_value: float = -1.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> None:
        self.max_freq_mask = max_freq_mask
        self.max_time_mask = max_time_mask
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.fill_value = fill_value
        self.rng = rng or np.random.RandomState()

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply masking to a spectrogram tensor of shape (1, F, T) or (F, T)."""
        augmented = spec.clone()
        if augmented.ndim == 3:
            _, num_freqs, num_times = augmented.shape
            is_3d = True
        elif augmented.ndim == 2:
            num_freqs, num_times = augmented.shape
            is_3d = False
        else:
            raise ValueError(f"SpecAugment expects 2D or 3D tensor, got shape {spec.shape}")

        # Frequency masking (mask vertical frequency bands)
        for _ in range(self.num_freq_masks):
            f_width = self.rng.randint(0, self.max_freq_mask + 1)
            if f_width > 0:
                f0 = self.rng.randint(0, max(1, num_freqs - f_width + 1))
                if is_3d:
                    augmented[:, f0 : f0 + f_width, :] = self.fill_value
                else:
                    augmented[f0 : f0 + f_width, :] = self.fill_value

        # Time masking (mask horizontal time frames)
        for _ in range(self.num_time_masks):
            t_width = self.rng.randint(0, self.max_time_mask + 1)
            if t_width > 0:
                t0 = self.rng.randint(0, max(1, num_times - t_width + 1))
                if is_3d:
                    augmented[:, :, t0 : t0 + t_width] = self.fill_value
                else:
                    augmented[:, t0 : t0 + t_width] = self.fill_value

        return augmented


class SpectrogramDataset(Dataset):
    """UUID-keyed PyTorch dataset for COUGHVID Log-Mel spectrograms.

    Guarantees:
    - Spectrograms are mapped strictly by UUID string from an in-memory dictionary
      or an NPZ archive, completely preventing positional index drift.
    - Labels are aligned by UUID string lookup.
    - Validation and inference passes (is_train=False) are strictly deterministic
      with zero augmentation.
    """

    def __init__(
        self,
        uuids: Sequence[str],
        labels: Mapping[str, Union[int, float]],
        spectrogram_source: Union[Mapping[str, np.ndarray], str, Path],
        is_train: bool = False,
        spec_augment: Optional[SpecAugment] = None,
    ) -> None:
        self.uuids: List[str] = [str(u) for u in uuids]
        self.labels: Dict[str, float] = {str(k): float(v) for k, v in labels.items()}
        self.is_train = is_train
        self.spec_augment = spec_augment if is_train else None

        # Verify that all provided UUIDs exist in the label mapping
        missing_labels = [u for u in self.uuids if u not in self.labels]
        if missing_labels:
            raise ValueError(
                f"Missing labels for {len(missing_labels)} UUIDs (first 3: {missing_labels[:3]})"
            )

        # Initialize spectrogram store
        if isinstance(spectrogram_source, (str, Path)):
            npz_path = Path(spectrogram_source)
            if not npz_path.exists():
                raise FileNotFoundError(f"Spectrogram archive not found at {npz_path}")
            # Load into memory dictionary to maximize throughput and enable thread-safe reads
            loaded_npz = np.load(npz_path)
            self._specs: Mapping[str, np.ndarray] = {u: loaded_npz[u] for u in self.uuids if u in loaded_npz}
        elif isinstance(spectrogram_source, Mapping):
            self._specs = spectrogram_source
        else:
            raise TypeError(
                f"spectrogram_source must be a mapping or path, got {type(spectrogram_source)}"
            )

        # Verify that all provided UUIDs exist in the spectrogram store
        missing_specs = [u for u in self.uuids if u not in self._specs]
        if missing_specs:
            raise ValueError(
                f"Missing spectrograms for {len(missing_specs)} UUIDs (first 3: {missing_specs[:3]})"
            )

    def __len__(self) -> int:
        return len(self.uuids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        uuid = self.uuids[idx]
        spec_np = self._specs[uuid]
        label_val = self.labels[uuid]

        # Convert to float32 tensor of shape (1, F, T)
        spec_tensor = torch.from_numpy(spec_np).float()
        if spec_tensor.ndim == 2:
            spec_tensor = spec_tensor.unsqueeze(0)  # Shape (1, 64, 128)

        # Apply SpecAugment strictly in training mode
        if self.is_train and self.spec_augment is not None:
            spec_tensor = self.spec_augment(spec_tensor)

        label_tensor = torch.tensor(label_val, dtype=torch.float32)
        return spec_tensor, label_tensor, uuid


def preload_spectrogram_dict(npz_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Preload all spectrogram arrays from an NPZ file into a RAM dictionary.

    Loads the ~280 MB dataset once to eliminate disk seeking overhead during CPU epochs.
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ archive not found: {path}")

    logger.info("Pre-loading spectrograms into memory from %s...", path)
    data = np.load(path)
    store = {k: data[k] for k in data.files}
    logger.info("Successfully loaded %d spectrograms into memory.", len(store))
    return store
