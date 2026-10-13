"""Acoustic feature extraction for segmented cough audio.

Extracts deterministic, documented acoustic representations:
- Mel-Frequency Cepstral Coefficients (MFCCs) and summary statistics.
- Delta and Delta-Delta MFCC statistics.
- Spectral centroid, bandwidth, rolloff (85%), and spectral slope.
- Time-domain Zero-Crossing Rate (ZCR) and Root Mean Square (RMS) energy.
- Power Spectral Density (PSD) frequency-band energy integrals.

All calculations use pure NumPy and SciPy implementations with strict
guards against non-finite values, zero-division, and empty/short audio segments.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import fft, integrate, signal
from scipy.ndimage import convolve1d

from coughvid.config import FeaturesConfig


def hz_to_mel(f_hz: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency in Hertz to Mel scale using the standard formula."""
    return 2595.0 * np.log10(1.0 + f_hz / 700.0)


def mel_to_hz(m_mel: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency in Mel scale to Hertz."""
    return 700.0 * (10.0 ** (m_mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int = 20,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> np.ndarray:
    """Construct a triangular Mel-scale filterbank matrix.

    Parameters
    ----------
    sr : int
        Audio sample rate in Hz.
    n_fft : int
        FFT window length.
    n_mels : int, default=20
        Number of Mel filters.
    f_min : float, default=0.0
        Lowest frequency in Hz.
    f_max : float, optional
        Highest frequency in Hz. Defaults to Nyquist (sr / 2).

    Returns
    -------
    np.ndarray
        Filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    if f_max is None:
        f_max = float(sr) / 2.0

    if f_min < 0.0:
        raise ValueError(f"f_min must be non-negative, got {f_min}")
    if f_max > sr / 2.0:
        raise ValueError(f"f_max ({f_max}) cannot exceed Nyquist ({sr / 2.0})")
    if f_min >= f_max:
        raise ValueError(f"f_min ({f_min}) must be less than f_max ({f_max})")

    num_bins = n_fft // 2 + 1
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Linearly spaced points in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Map Hz frequencies to FFT bin indices
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, num_bins - 1)

    weights = np.zeros((n_mels, num_bins), dtype=np.float32)

    for m in range(n_mels):
        left = bin_points[m]
        center = bin_points[m + 1]
        right = bin_points[m + 2]

        if center > left:
            weights[m, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            weights[m, center:right] = (right - np.arange(center, right)) / (right - center)

    # Normalize area of each triangular filter to preserve energy across bandwidths
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights


def frame_signal(
    y: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """Slice a 1D audio waveform into overlapping frames.

    If the signal is shorter than frame_length, it is zero-padded to at least frame_length.

    Parameters
    ----------
    y : np.ndarray
        1D input signal.
    frame_length : int
        Length of each frame in samples.
    hop_length : int
        Step size between frames in samples.

    Returns
    -------
    np.ndarray
        2D array of shape (num_frames, frame_length).
    """
    if len(y) == 0:
        return np.zeros((1, frame_length), dtype=np.float32)

    if len(y) < frame_length:
        padded = np.pad(y, (0, frame_length - len(y)), mode="constant")
        return padded[np.newaxis, :].astype(np.float32)

    num_frames = 1 + int(math.floor((len(y) - frame_length) / hop_length))
    # Efficient strided or slicing array creation
    frames = np.empty((num_frames, frame_length), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = y[start : start + frame_length]

    return frames


def compute_deltas(data: np.ndarray, width: int = 5) -> np.ndarray:
    """Compute first-order regression delta coefficients along time axis (axis 0).

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (num_frames, n_features).
    width : int, default=5
        Number of context frames for regression filter (must be odd and >= 3).

    Returns
    -------
    np.ndarray
        Delta array of shape (num_frames, n_features).
    """
    num_frames = data.shape[0]
    if num_frames < 2:
        return np.zeros_like(data)

    if num_frames < width:
        # For very short sequences, use finite differences
        deltas = np.zeros_like(data)
        deltas[1:] = data[1:] - data[:-1]
        deltas[0] = deltas[1] if num_frames > 1 else np.zeros(data.shape[1])
        return deltas

    half_width = width // 2
    n_range = np.arange(-half_width, half_width + 1, dtype=np.float32)
    denom = 2.0 * float(np.sum(np.arange(1, half_width + 1) ** 2))
    kernel = n_range / denom

    return convolve1d(data, kernel, axis=0, mode="nearest").astype(np.float32)


def compute_stft_and_spectrum(
    y: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform magnitude, power, and frequencies.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (magnitude_spectrum [num_frames, num_bins],
         power_spectrum [num_frames, num_bins],
         freq_bins [num_bins])
    """
    frames = frame_signal(y, frame_length=n_fft, hop_length=hop_length)
    window = np.hanning(n_fft).astype(np.float32)
    windowed = frames * window[np.newaxis, :]

    stft_complex = np.fft.rfft(windowed, n=n_fft, axis=-1)
    mag_spectrum = np.abs(stft_complex).astype(np.float32)
    power_spectrum = (mag_spectrum ** 2) / float(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    return mag_spectrum, power_spectrum, freqs


def compute_mfcc_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 40,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> Dict[str, float]:
    """Extract summary statistics for MFCCs, Delta MFCCs, and Delta-Delta MFCCs.

    Returns deterministic metrics across frames:
    - mfcc_{k}_mean, mfcc_{k}_std, mfcc_{k}_min, mfcc_{k}_max
    - delta_mfcc_{k}_mean, delta_mfcc_{k}_std
    - delta2_mfcc_{k}_mean, delta2_mfcc_{k}_std
    """
    if f_max is None:
        f_max = float(sr) / 2.0

    _, power_spectrum, _ = compute_stft_and_spectrum(y, sr, n_fft=n_fft, hop_length=hop_length)
    filterbank = build_mel_filterbank(sr, n_fft=n_fft, n_mels=n_mels, f_min=f_min, f_max=f_max)

    # Mel spectrum: (num_frames, n_mels)
    mel_energies = np.dot(power_spectrum, filterbank.T)
    # Log-mel with floor epsilon for numerical stability
    log_mel = np.log10(np.maximum(mel_energies, 1e-10))

    # Discrete Cosine Transform (ortho-normalized)
    mfccs = fft.dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc].astype(np.float32)

    deltas = compute_deltas(mfccs)
    deltas2 = compute_deltas(deltas)

    features: Dict[str, float] = {}

    for k in range(n_mfcc):
        idx = k + 1
        coeff = mfccs[:, k]
        features[f"mfcc_{idx}_mean"] = float(np.mean(coeff))
        features[f"mfcc_{idx}_std"] = float(np.std(coeff))
        features[f"mfcc_{idx}_min"] = float(np.min(coeff))
        features[f"mfcc_{idx}_max"] = float(np.max(coeff))

        d_coeff = deltas[:, k]
        features[f"delta_mfcc_{idx}_mean"] = float(np.mean(d_coeff))
        features[f"delta_mfcc_{idx}_std"] = float(np.std(d_coeff))

        d2_coeff = deltas2[:, k]
        features[f"delta2_mfcc_{idx}_mean"] = float(np.mean(d2_coeff))
        features[f"delta2_mfcc_{idx}_std"] = float(np.std(d2_coeff))

    return features


def compute_spectral_features(
    y: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
    rolloff_percent: float = 0.85,
) -> Dict[str, float]:
    """Compute Spectral Centroid, Bandwidth, Rolloff (85%), and Spectral Slope.

    Returns summary statistics across time frames:
    - spectral_centroid (mean, std, min, max)
    - spectral_bandwidth (mean, std, min, max)
    - spectral_rolloff (mean, std, min, max)
    - spectral_slope (mean, std)
    """
    mag_spec, _, freqs = compute_stft_and_spectrum(y, sr, n_fft=n_fft, hop_length=hop_length)
    num_frames, num_bins = mag_spec.shape
    eps = 1e-9

    # Frame-wise sums
    spec_sum = np.sum(mag_spec, axis=1)  # (num_frames,)

    # 1. Spectral Centroid: C = sum(f * S) / sum(S)
    weighted_freqs = np.dot(mag_spec, freqs)  # (num_frames,)
    centroid = np.where(spec_sum > eps, weighted_freqs / (spec_sum + eps), 0.0)

    # 2. Spectral Bandwidth: sqrt( sum((f - C)^2 * S) / sum(S) )
    # (num_frames, num_bins)
    freq_diff_sq = (freqs[np.newaxis, :] - centroid[:, np.newaxis]) ** 2
    bandwidth = np.sqrt(
        np.where(spec_sum > eps, np.sum(freq_diff_sq * mag_spec, axis=1) / (spec_sum + eps), 0.0)
    )

    # 3. Spectral Rolloff: frequency below which rolloff_percent of energy lies
    rolloff = np.zeros(num_frames, dtype=np.float32)
    cumsum_spec = np.cumsum(mag_spec, axis=1)
    thresholds = rolloff_percent * spec_sum

    for i in range(num_frames):
        if spec_sum[i] > eps:
            idx = np.searchsorted(cumsum_spec[i], thresholds[i])
            idx = min(idx, num_bins - 1)
            rolloff[i] = freqs[idx]
        else:
            rolloff[i] = 0.0

    # 4. Spectral Slope: linear regression slope of magnitude spectrum vs frequencies
    # slope = sum((f - f_mean) * (S - S_mean)) / sum((f - f_mean)^2)
    f_mean = np.mean(freqs)
    f_diff = freqs - f_mean
    f_var = np.sum(f_diff ** 2)

    slopes = np.zeros(num_frames, dtype=np.float32)
    if f_var > eps:
        s_mean = np.mean(mag_spec, axis=1, keepdims=True)
        s_diff = mag_spec - s_mean
        slopes = (np.dot(s_diff, f_diff) / f_var).astype(np.float32)

    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_centroid_min": float(np.min(centroid)),
        "spectral_centroid_max": float(np.max(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_bandwidth_min": float(np.min(bandwidth)),
        "spectral_bandwidth_max": float(np.max(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "spectral_rolloff_min": float(np.min(rolloff)),
        "spectral_rolloff_max": float(np.max(rolloff)),
        "spectral_slope_mean": float(np.mean(slopes)),
        "spectral_slope_std": float(np.std(slopes)),
    }


def compute_time_domain_features(
    y: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 128,
) -> Dict[str, float]:
    """Compute Zero-Crossing Rate (ZCR) and Root Mean Square (RMS) energy.

    Returns:
    - zcr_mean, zcr_std, zcr_min, zcr_max
    - rms_mean, rms_std, rms_min, rms_max, rms_peak
    """
    frames = frame_signal(y, frame_length=frame_length, hop_length=hop_length)
    num_frames = frames.shape[0]

    # RMS energy per frame
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Zero-crossing rate per frame
    # zcr = sum(|sign(x_n) - sign(x_{n-1})|) / (2 * (N - 1))
    signs = np.sign(frames)
    # Treat exact zero as 0 to avoid false sign flips
    diffs = np.abs(signs[:, 1:] - signs[:, :-1])
    # Diffs can be 2 (sign change) or 1 (transition to/from 0). A full sign cross is diff == 2.
    # Standard engineering ZCR: count non-zero transitions
    zcr = np.sum(diffs > 0, axis=1) / (2.0 * float(frame_length - 1))

    return {
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "zcr_min": float(np.min(zcr)),
        "zcr_max": float(np.max(zcr)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_min": float(np.min(rms)),
        "rms_max": float(np.max(rms)),
        "rms_peak": float(np.max(np.abs(y))) if len(y) > 0 else 0.0,
    }


def compute_psd_band_features(
    y: np.ndarray,
    sr: int,
    bands: Sequence[Tuple[int, int]],
    n_fft: int = 512,
) -> Dict[str, float]:
    """Compute frequency-band energy and relative energy via Welch's Power Spectral Density.

    Parameters
    ----------
    y : np.ndarray
        1D audio waveform.
    sr : int
        Sample rate in Hz.
    bands : Sequence[Tuple[int, int]]
        List of (f_low, f_high) frequency tuples in Hz.
    n_fft : int, default=512
        Segment length for Welch's PSD calculation.

    Returns
    -------
    Dict[str, float]
        Dictionary with absolute energy and relative energy for each band.
    """
    features: Dict[str, float] = {}
    if len(y) == 0:
        for f_low, f_high in bands:
            features[f"psd_band_{f_low}_{f_high}_energy"] = 0.0
            features[f"psd_band_{f_low}_{f_high}_rel"] = 0.0
        return features

    nperseg = min(len(y), n_fft)
    freqs, psd = signal.welch(y, fs=sr, nperseg=nperseg, scaling="density")

    total_energy = float(integrate.trapezoid(psd, freqs)) if len(freqs) > 1 else float(np.sum(psd))
    eps = 1e-12

    for f_low, f_high in bands:
        mask = (freqs >= f_low) & (freqs <= f_high)
        if np.any(mask) and np.sum(mask) > 1:
            band_energy = float(integrate.trapezoid(psd[mask], freqs[mask]))
        elif np.any(mask):
            band_energy = float(np.sum(psd[mask]))
        else:
            band_energy = 0.0

        rel_energy = band_energy / (total_energy + eps)
        features[f"psd_band_{f_low}_{f_high}_energy"] = band_energy
        features[f"psd_band_{f_low}_{f_high}_rel"] = rel_energy

    return features


def extract_tabular_features(
    y: np.ndarray,
    sr: int,
    config: Optional[FeaturesConfig] = None,
) -> Dict[str, float]:
    """Extract full documented suite of acoustic tabular features from an audio waveform.

    Parameters
    ----------
    y : np.ndarray
        1D float32 audio waveform.
    sr : int
        Audio sample rate in Hz.
    config : FeaturesConfig, optional
        Feature configuration settings.

    Returns
    -------
    Dict[str, float]
        Deterministic dictionary of named acoustic features. All values are validated
        finite floating-point numbers.

    Raises
    ------
    ValueError
        If waveform contains non-finite numbers or sample rate is non-positive.
    """
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError(f"Waveform must be a 1D numpy array, got {type(y)}.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Waveform contains non-finite values (NaN or Inf).")
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}.")

    n_mfcc = 20 if config is None else config.n_mfcc
    bands = (
        [
            (0, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 3000),
            (3000, 4000),
            (4000, 5000),
            (5000, 6000),
        ]
        if config is None
        else config.psd_bands
    )
    n_fft = getattr(config, "n_fft", 512)
    hop_length = getattr(config, "hop_length", 128)
    n_mels = getattr(config, "n_mels", 40)

    # 1. MFCCs and Deltas
    mfcc_feats = compute_mfcc_features(
        y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # 2. Spectral features
    spectral_feats = compute_spectral_features(
        y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )

    # 3. Time domain features
    time_feats = compute_time_domain_features(
        y, frame_length=n_fft, hop_length=hop_length
    )

    # 4. PSD bands
    psd_feats = compute_psd_band_features(
        y, sr=sr, bands=bands, n_fft=n_fft
    )

    # Combine all feature dictionaries
    combined: Dict[str, float] = {}
    combined.update(mfcc_feats)
    combined.update(spectral_feats)
    combined.update(time_feats)
    combined.update(psd_feats)

    # Strict numerical validation: ensure no NaN or Inf is silently emitted
    for name, val in combined.items():
        if not np.isfinite(val):
            raise ValueError(f"Extracted feature '{name}' is non-finite: {val}")

    return combined
