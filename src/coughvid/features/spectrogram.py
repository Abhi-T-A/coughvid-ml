"""Log-Mel Spectrogram extraction module.

Generates fixed-shape, 2D Log-Mel spectrogram tensors for audio classification.
Provides deterministic, pure NumPy/SciPy computation without requiring heavy
deep-learning frameworks.

TENSOR DIMENSIONS:
- Output shape: (n_mels, target_frames), default (64, 128)
- Data type: np.float32

NORMALIZATION OPTIONS:
- "db": Decibel scale relative to peak power, clipped to [-top_db, 0.0] dB and
  scaled to [-1.0, 1.0].
- "standard": Spectrogram-level z-score standardization (zero mean, unit variance).
- "minmax": Scaled to [0.0, 1.0] across dynamic range.
- "none": Unscaled log-power energies.
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np

from coughvid.features.tabular import build_mel_filterbank, frame_signal


def compute_log_mel_spectrogram(
    y: np.ndarray,
    sr: int = 12000,
    n_mels: int = 64,
    n_fft: int = 512,
    hop_length: int = 128,
    target_frames: int = 128,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    normalize: str = "db",
    top_db: float = 80.0,
    crop_mode: str = "center",
) -> np.ndarray:
    """Extract a fixed-shape Log-Mel spectrogram tensor from an audio waveform.

    Parameters
    ----------
    y : np.ndarray
        1D float32 audio waveform.
    sr : int, default=12000
        Audio sample rate in Hz.
    n_mels : int, default=64
        Number of Mel frequency bins (first dimension).
    n_fft : int, default=512
        FFT window length.
    hop_length : int, default=128
        Hop length between successive FFT frames.
    target_frames : int, default=128
        Fixed number of time frames (second dimension).
    f_min : float, default=0.0
        Lowest frequency in Hz.
    f_max : float, optional
        Highest frequency in Hz. Defaults to Nyquist (sr / 2).
    normalize : {"db", "standard", "minmax", "none"}, default="db"
        Normalization strategy.
    top_db : float, default=80.0
        Dynamic range in decibels below peak power for "db" normalization.
    crop_mode : {"center", "head", "energy_peak"}, default="center"
        Strategy to resize longer sequences to target_frames.

    Returns
    -------
    np.ndarray
        Fixed-shape 2D array of shape (n_mels, target_frames) with dtype np.float32.

    Raises
    ------
    ValueError
        If waveform is empty, non-finite, or parameters are invalid.
    """
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError(f"Waveform must be a 1D numpy array, got {type(y)}.")
    if len(y) == 0:
        raise ValueError("Audio waveform cannot be empty (0 samples).")
    if not np.all(np.isfinite(y)):
        raise ValueError("Audio waveform contains non-finite values (NaN or Inf).")
    if sr <= 0:
        raise ValueError(f"Sample rate must be strictly positive, got {sr}.")
    if n_mels <= 0 or n_fft <= 0 or hop_length <= 0 or target_frames <= 0:
        raise ValueError("n_mels, n_fft, hop_length, and target_frames must all be positive integers.")

    if f_max is None:
        f_max = float(sr) / 2.0

    # 1. Slice into overlapping frames
    frames = frame_signal(y, frame_length=n_fft, hop_length=hop_length)
    num_frames = frames.shape[0]

    # 2. Window and FFT
    window = np.hanning(n_fft).astype(np.float32)
    windowed = frames * window[np.newaxis, :]
    stft_complex = np.fft.rfft(windowed, n=n_fft, axis=-1)
    mag_spectrum = np.abs(stft_complex).astype(np.float32)
    power_spectrum = (mag_spectrum ** 2) / float(n_fft)

    # 3. Apply Mel filterbank
    filterbank = build_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, f_min=f_min, f_max=f_max)
    # Shape: (num_frames, n_mels)
    mel_energies = np.dot(power_spectrum, filterbank.T)

    # 4. Transpose to (n_mels, num_frames)
    mel_spec = mel_energies.T

    # 5. Fix time dimension to target_frames
    if num_frames < target_frames:
        # Pad with zeros along time dimension (axis 1)
        pad_width = target_frames - num_frames
        mel_spec_fixed = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    elif num_frames > target_frames:
        if crop_mode == "head":
            mel_spec_fixed = mel_spec[:, :target_frames]
        elif crop_mode == "energy_peak":
            # Find window of length target_frames with highest total energy
            frame_energies = np.sum(mel_spec, axis=0)
            window_weights = np.ones(target_frames, dtype=np.float32)
            conv_energy = np.convolve(frame_energies, window_weights, mode="valid")
            best_start = int(np.argmax(conv_energy))
            mel_spec_fixed = mel_spec[:, best_start : best_start + target_frames]
        else:  # "center"
            start = (num_frames - target_frames) // 2
            mel_spec_fixed = mel_spec[:, start : start + target_frames]
    else:
        mel_spec_fixed = mel_spec

    # 6. Log / Decibel transform and Normalization
    eps = 1e-10
    peak_val = float(np.max(mel_spec_fixed))

    if normalize == "db":
        # Decibel power relative to peak
        if peak_val < eps:
            # Silent audio: return constant floor (-1.0)
            return np.full((n_mels, target_frames), -1.0, dtype=np.float32)
        log_power = 10.0 * np.log10(np.maximum(mel_spec_fixed, eps))
        ref_db = 10.0 * np.log10(peak_val)
        db_spec = log_power - ref_db  # peak is 0.0 dB, lower values are negative
        db_clipped = np.clip(db_spec, -top_db, 0.0)
        # Scale [-top_db, 0.0] to [-1.0, 1.0]
        norm_spec = 2.0 * ((db_clipped + top_db) / top_db) - 1.0
        return norm_spec.astype(np.float32)

    elif normalize == "standard":
        log_spec = np.log10(np.maximum(mel_spec_fixed, eps))
        mean = float(np.mean(log_spec))
        std = float(np.std(log_spec))
        if std < 1e-6:
            return np.zeros_like(log_spec, dtype=np.float32)
        norm_spec = (log_spec - mean) / std
        return norm_spec.astype(np.float32)

    elif normalize == "minmax":
        log_spec = np.log10(np.maximum(mel_spec_fixed, eps))
        s_min = float(np.min(log_spec))
        s_max = float(np.max(log_spec))
        if (s_max - s_min) < 1e-6:
            return np.zeros_like(log_spec, dtype=np.float32)
        norm_spec = (log_spec - s_min) / (s_max - s_min)
        return norm_spec.astype(np.float32)

    elif normalize == "none":
        return np.log10(np.maximum(mel_spec_fixed, eps)).astype(np.float32)

    else:
        raise ValueError(
            f"Unknown normalization '{normalize}'. Choose from 'db', 'standard', 'minmax', 'none'."
        )
