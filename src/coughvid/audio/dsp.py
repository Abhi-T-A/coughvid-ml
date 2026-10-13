"""Digital Signal Processing (DSP) utilities for acoustic audio.

Includes waveform validation, Nyquist-safe Butterworth filtering,
polyphase rational resampling, and robust normalization.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import signal


def validate_waveform(y: np.ndarray, sr: int) -> None:
    """Validate that the input is a valid, 1D, finite audio array with a positive sample rate.

    Raises
    ------
    TypeError
        If y is not a numpy array.
    ValueError
        If y is not 1D, is empty, contains non-finite numbers, or sr <= 0.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(f"Audio waveform must be a numpy.ndarray, got {type(y).__name__}.")
    if y.ndim != 1:
        raise ValueError(f"Audio waveform must be 1D (mono), got ndim={y.ndim} with shape {y.shape}.")
    if len(y) == 0:
        raise ValueError("Audio waveform cannot be empty (0 samples).")
    if not np.all(np.isfinite(y)):
        raise ValueError("Audio waveform contains non-finite values (NaN or Inf).")
    if not isinstance(sr, (int, np.integer)) or sr <= 0:
        raise ValueError(f"Sample rate must be a strictly positive integer, got {sr}.")


def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    """Resample audio waveform from orig_sr to target_sr using polyphase filtering.

    Parameters
    ----------
    y : np.ndarray
        1D float array of audio samples.
    orig_sr : int
        Original sample rate (Hz).
    target_sr : int
        Desired sample rate (Hz).

    Returns
    -------
    tuple[np.ndarray, int]
        (resampled_waveform, target_sr)
    """
    validate_waveform(y, orig_sr)
    if not isinstance(target_sr, (int, np.integer)) or target_sr <= 0:
        raise ValueError(f"Target sample rate must be a positive integer, got {target_sr}.")

    if orig_sr == target_sr:
        return y.astype(np.float32, copy=True), orig_sr

    # Find greatest common divisor to keep up/down integer factors small and fast
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    resampled = signal.resample_poly(y, up=up, down=down).astype(np.float32)
    return resampled, target_sr


def butterworth_lowpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase forward-backward Butterworth low-pass filter (filtfilt).

    Parameters
    ----------
    y : np.ndarray
        1D audio samples.
    sr : int
        Sample rate in Hz.
    cutoff_hz : float
        Filter cutoff frequency in Hz. Must satisfy 0 < cutoff_hz < 0.5 * sr.
    order : int, default=5
        Filter order (must be positive).

    Returns
    -------
    np.ndarray
        Filtered audio waveform (float32).

    Raises
    ------
    ValueError
        If cutoff_hz is non-positive or exceeds the Nyquist frequency.
    """
    validate_waveform(y, sr)

    if order <= 0:
        raise ValueError(f"Filter order must be a positive integer, got {order}.")

    nyquist = 0.5 * sr
    if cutoff_hz <= 0.0:
        raise ValueError(f"Cutoff frequency must be strictly positive, got {cutoff_hz} Hz.")
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"Cutoff frequency ({cutoff_hz} Hz) must be strictly less than the "
            f"Nyquist frequency ({nyquist} Hz) for sample rate {sr} Hz."
        )

    normalized_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype="low")
    filtered = signal.filtfilt(b, a, y)
    return filtered.astype(np.float32)


def normalize_audio(
    y: np.ndarray,
    method: str = "peak",
    target_peak: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize audio amplitude with safe handling of silence.

    Parameters
    ----------
    y : np.ndarray
        1D audio waveform.
    method : {"peak", "percentile"}, default="peak"
        - "peak": Scale peak absolute value to target_peak.
        - "percentile": Scale based on 99.5th percentile to resist isolated mic clicks,
          then clip strictly to [-target_peak, target_peak].
    target_peak : float, default=1.0
        Target maximum amplitude (typically 1.0).
    eps : float, default=1e-8
        Threshold below which the signal is considered silent.

    Returns
    -------
    np.ndarray
        Normalized waveform (float32).
    """
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("Audio waveform must be a 1D numpy array.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Audio waveform contains non-finite values.")

    abs_max = float(np.max(np.abs(y))) if len(y) > 0 else 0.0

    # Handle silence and near-silence safely: do not amplify numerical noise or divide by zero
    if abs_max < eps:
        return np.zeros_like(y, dtype=np.float32)

    if method == "peak":
        norm_factor = target_peak / (abs_max + eps)
        normalized = y * norm_factor
    elif method == "percentile":
        # 99.5th percentile resists single-sample pop clicks
        p99 = float(np.percentile(np.abs(y), 99.5))
        if p99 < eps:
            norm_factor = target_peak / (abs_max + eps)
        else:
            norm_factor = target_peak / (p99 + eps)
        normalized = np.clip(y * norm_factor, -target_peak, target_peak)
    else:
        raise ValueError(f"Unknown normalization method '{method}'. Choose 'peak' or 'percentile'.")

    return normalized.astype(np.float32)


def pad_or_trim(y: np.ndarray, target_length: int) -> np.ndarray:
    """Pad waveform with zeros or trim to exact target sample count.

    Parameters
    ----------
    y : np.ndarray
        1D input audio waveform.
    target_length : int
        Target number of samples.

    Returns
    -------
    np.ndarray
        Adjusted waveform with shape (target_length,).
    """
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}.")
    if len(y) > target_length:
        return y[:target_length].astype(np.float32)
    if len(y) < target_length:
        return np.pad(y, (0, target_length - len(y)), mode="constant").astype(np.float32)
    return y.astype(np.float32, copy=True)
