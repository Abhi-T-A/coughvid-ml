"""Unit tests for digital signal processing (DSP) utilities."""

import numpy as np
import pytest

from coughvid.audio.dsp import (
    butterworth_lowpass_filter,
    normalize_audio,
    pad_or_trim,
    resample_audio,
    validate_waveform,
)


def test_validate_waveform_valid():
    """Verify that a valid 1D float waveform passes validation without error."""
    y = np.sin(np.linspace(0, 100, 1000)).astype(np.float32)
    validate_waveform(y, 16000)  # Should not raise


def test_validate_waveform_invalid_types_and_dimensions():
    """Verify validate_waveform rejects lists, 2D arrays, and empty arrays."""
    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        validate_waveform([1.0, 2.0], 16000)  # type: ignore

    with pytest.raises(ValueError, match="must be 1D"):
        validate_waveform(np.zeros((2, 100), dtype=np.float32), 16000)

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_waveform(np.array([], dtype=np.float32), 16000)


def test_validate_waveform_non_finite():
    """Verify validate_waveform rejects NaNs and infinities."""
    y_nan = np.array([0.1, np.nan, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite values"):
        validate_waveform(y_nan, 16000)

    y_inf = np.array([0.1, np.inf, 0.3], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite values"):
        validate_waveform(y_inf, 16000)


def test_validate_waveform_invalid_sample_rate():
    """Verify validate_waveform rejects non-positive or non-integer sample rates."""
    y = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError, match="Sample rate must be a strictly positive integer"):
        validate_waveform(y, 0)

    with pytest.raises(ValueError, match="Sample rate must be a strictly positive integer"):
        validate_waveform(y, -16000)


def test_resample_audio_identity():
    """Verify resampling with orig_sr == target_sr returns an exact copy."""
    y = np.sin(np.linspace(0, 20, 500)).astype(np.float32)
    y_resampled, sr = resample_audio(y, 16000, 16000)
    assert sr == 16000
    assert np.allclose(y, y_resampled)


def test_resample_audio_downsample_and_upsample():
    """Verify rational downsampling and upsampling adjust sample lengths correctly."""
    sr_orig = 48000
    sr_target = 12000
    duration_sec = 0.5
    num_samples = int(sr_orig * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Downsample: 48 kHz -> 12 kHz (factor of 4 reduction)
    y_down, sr_down = resample_audio(y, sr_orig, sr_target)
    assert sr_down == 12000
    expected_samples = int(sr_target * duration_sec)
    assert abs(len(y_down) - expected_samples) <= 2
    assert np.all(np.isfinite(y_down))

    # Upsample: 12 kHz -> 24 kHz (factor of 2 increase)
    y_up, sr_up = resample_audio(y_down, 12000, 24000)
    assert sr_up == 24000
    assert abs(len(y_up) - 2 * len(y_down)) <= 2
    assert np.all(np.isfinite(y_up))


def test_butterworth_filter_nyquist_bounds():
    """Verify butterworth_lowpass_filter strictly rejects cutoff >= Nyquist or <= 0."""
    y = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    sr = 12000
    nyquist = 6000.0

    # Cutoff at Nyquist should fail
    with pytest.raises(ValueError, match="strictly less than the Nyquist"):
        butterworth_lowpass_filter(y, sr, cutoff_hz=6000.0)

    # Cutoff above Nyquist should fail
    with pytest.raises(ValueError, match="strictly less than the Nyquist"):
        butterworth_lowpass_filter(y, sr, cutoff_hz=7000.0)

    # Non-positive cutoff should fail
    with pytest.raises(ValueError, match="strictly positive"):
        butterworth_lowpass_filter(y, sr, cutoff_hz=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        butterworth_lowpass_filter(y, sr, cutoff_hz=-100.0)


def test_butterworth_filter_frequency_attenuation():
    """Verify Butterworth lowpass filter passes low frequencies and attenuates high frequencies."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Sum of low frequency (200 Hz) and high frequency (4000 Hz)
    f_low = 200.0
    f_high = 4000.0
    y = (np.sin(2 * np.pi * f_low * t) + np.sin(2 * np.pi * f_high * t)).astype(np.float32)

    cutoff = 1000.0  # Cutoff between 200 Hz and 4000 Hz
    y_filtered = butterworth_lowpass_filter(y, sr, cutoff_hz=cutoff, order=5)

    assert len(y_filtered) == len(y)
    assert np.all(np.isfinite(y_filtered))

    # Energy of the high-frequency component should be substantially eliminated
    # In time domain, variance of filtered signal should be close to 0.5 (variance of single sine wave)
    var_orig = np.var(y)          # ~ 1.0 (0.5 + 0.5)
    var_filt = np.var(y_filtered)  # ~ 0.5 (only 200 Hz remains)
    assert var_filt < 0.65 * var_orig


def test_normalize_audio_peak():
    """Verify peak normalization scales max amplitude to target_peak."""
    y = np.array([-0.5, 0.2, 0.4, -0.1], dtype=np.float32)
    y_norm = normalize_audio(y, method="peak", target_peak=1.0)
    assert np.isclose(np.max(np.abs(y_norm)), 1.0)


def test_normalize_audio_silent():
    """Verify normalize_audio on silence returns zeros without dividing by zero."""
    y_silent = np.zeros(500, dtype=np.float32)
    y_norm = normalize_audio(y_silent, method="peak")
    assert np.all(y_norm == 0.0)

    y_near_silent = np.full(500, 1e-12, dtype=np.float32)
    y_near_norm = normalize_audio(y_near_silent, method="peak")
    assert np.all(y_near_norm == 0.0)


def test_normalize_audio_percentile():
    """Verify percentile normalization clips outliers strictly within bounds."""
    # Signal with a single impulsive click (+10.0)
    y = np.sin(np.linspace(0, 10, 1000)).astype(np.float32) * 0.2
    y[500] = 10.0  # Outlier pop click
    y_norm = normalize_audio(y, method="percentile", target_peak=1.0)
    assert np.max(np.abs(y_norm)) <= 1.0 + 1e-6


def test_pad_or_trim():
    """Verify pad_or_trim correctly pads shorter arrays and trims longer arrays."""
    y_short = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_padded = pad_or_trim(y_short, target_length=5)
    assert len(y_padded) == 5
    assert np.allclose(y_padded, [1.0, 2.0, 3.0, 0.0, 0.0])

    y_long = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y_trimmed = pad_or_trim(y_long, target_length=3)
    assert len(y_trimmed) == 3
    assert np.allclose(y_trimmed, [1.0, 2.0, 3.0])
