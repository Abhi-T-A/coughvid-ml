"""Unit tests for energy-envelope hysteresis cough burst segmentation."""

import numpy as np
import pytest

from coughvid.audio.segmenter import (
    CoughSegment,
    SegmentationResult,
    compute_energy_envelope,
    segment_cough_bursts,
)
from coughvid.config import SegmentationConfig


def _create_synthetic_burst_signal(
    sr: int = 12000,
    total_sec: float = 3.0,
    burst_windows: list[tuple[float, float]] | None = None,
    burst_freq: float = 300.0,
    burst_amp: float = 0.8,
    noise_amp: float = 0.01,
) -> np.ndarray:
    """Generate deterministic synthetic signal with baseline noise and localized bursts."""
    num_samples = int(sr * total_sec)
    t = np.linspace(0, total_sec, num_samples, endpoint=False)
    # Baseline low-level noise
    np.random.seed(42)
    y = (np.random.randn(num_samples) * noise_amp).astype(np.float32)

    if burst_windows:
        for start_t, end_t in burst_windows:
            start_idx = int(start_t * sr)
            end_idx = int(end_t * sr)
            burst_t = t[start_idx:end_idx]
            # Modulate with a smooth half-sine envelope
            window = np.sin(np.linspace(0, np.pi, len(burst_t)))
            burst_waveform = (burst_amp * np.sin(2 * np.pi * burst_freq * burst_t) * window).astype(np.float32)
            y[start_idx:end_idx] += burst_waveform

    return y


def test_compute_energy_envelope_empty():
    """Verify compute_energy_envelope on empty input returns empty arrays."""
    env, times = compute_energy_envelope(np.array([], dtype=np.float32), 12000)
    assert len(env) == 0
    assert len(times) == 0


def test_compute_energy_envelope_tracks_burst():
    """Verify that the envelope value during an active burst is substantially higher than baseline noise."""
    sr = 12000
    y = _create_synthetic_burst_signal(
        sr=sr,
        total_sec=2.0,
        burst_windows=[(0.8, 1.2)],
        burst_amp=0.9,
        noise_amp=0.01,
    )
    env, times = compute_energy_envelope(y, sr, frame_length_ms=30.0, hop_length_ms=10.0)

    # Find peak envelope time
    peak_idx = int(np.argmax(env))
    peak_time = times[peak_idx]
    assert 0.8 <= peak_time <= 1.2, f"Peak envelope time {peak_time:.2f}s is not within burst window [0.8, 1.2]"

    # Energy during silence (e.g. at 0.3s) should be much lower than peak
    baseline_idx = int(np.argmin(np.abs(times - 0.3)))
    assert env[peak_idx] > 10.0 * env[baseline_idx]


def test_segment_single_synthetic_burst():
    """Verify detection of a single isolated burst with known onset and offset."""
    sr = 12000
    burst_start = 0.8
    burst_end = 1.2
    y = _create_synthetic_burst_signal(
        sr=sr,
        total_sec=2.5,
        burst_windows=[(burst_start, burst_end)],
        burst_amp=0.85,
        noise_amp=0.01,
    )

    res = segment_cough_bursts(
        y,
        sr,
        min_burst_duration=0.15,
        max_burst_duration=1.5,
        pad_context_duration=0.05,
    )

    assert isinstance(res, SegmentationResult)
    assert res.has_candidates is True
    assert res.num_segments == 1

    seg = res.segments[0]
    assert isinstance(seg, CoughSegment)
    # Check boundaries match known burst with context padding tolerance (~0.1s)
    assert abs(seg.start_sec - (burst_start - 0.05)) < 0.12
    assert abs(seg.end_sec - (burst_end + 0.05)) < 0.12
    assert len(seg.waveform) == len(res.concatenated_waveform)


def test_segment_multiple_bursts():
    """Verify detection of multiple independent bursts separated by silence."""
    sr = 12000
    y = _create_synthetic_burst_signal(
        sr=sr,
        total_sec=3.5,
        burst_windows=[(0.5, 0.8), (1.8, 2.2)],
        burst_amp=0.85,
        noise_amp=0.01,
    )

    res = segment_cough_bursts(y, sr, min_burst_duration=0.15, max_burst_duration=1.5)
    assert res.num_segments == 2
    assert res.segments[0].end_sec < res.segments[1].start_sec


def test_merge_close_bursts():
    """Verify that bursts separated by a gap <= merge_gap_duration are merged."""
    sr = 12000
    # Two bursts separated by only 40 ms gap (less than merge_gap_duration=0.10s)
    y = _create_synthetic_burst_signal(
        sr=sr,
        total_sec=2.5,
        burst_windows=[(0.6, 0.85), (0.89, 1.15)],
        burst_amp=0.85,
        noise_amp=0.01,
    )

    res = segment_cough_bursts(
        y,
        sr,
        merge_gap_duration=0.10,
        min_burst_duration=0.15,
        max_burst_duration=1.5,
    )
    # Should merge into 1 composite segment
    assert res.num_segments == 1
    seg = res.segments[0]
    assert seg.duration_sec >= 0.50


def test_duration_filter_rejects_brief_clicks():
    """Verify that very short transient clicks (< min_burst_duration) are discarded."""
    sr = 12000
    # Single 40 ms spike (shorter than min_burst_duration=0.15s)
    y = _create_synthetic_burst_signal(
        sr=sr,
        total_sec=2.0,
        burst_windows=[(0.5, 0.54)],
        burst_amp=0.85,
        noise_amp=0.01,
    )

    res = segment_cough_bursts(y, sr, min_burst_duration=0.15)
    assert res.num_segments == 0
    assert res.has_candidates is False
    assert len(res.concatenated_waveform) == 0


def test_silent_recording_returns_no_candidates():
    """Verify pure silence returns zero candidate segments cleanly."""
    sr = 12000
    y_silence = np.zeros(24000, dtype=np.float32)
    res = segment_cough_bursts(y_silence, sr)

    assert res.has_candidates is False
    assert res.num_segments == 0
    assert len(res.segments) == 0
    assert len(res.concatenated_waveform) == 0


def test_boundary_conditions_padding():
    """Verify context padding at sample 0 and end of file does not exceed array bounds."""
    sr = 12000
    # Burst starting immediately at t=0.0s
    y = _create_synthetic_burst_signal(
        sr=sr,
        total_sec=1.5,
        burst_windows=[(0.0, 0.4)],
        burst_amp=0.85,
    )

    res = segment_cough_bursts(y, sr, pad_context_duration=0.10, min_burst_duration=0.15)
    assert res.has_candidates is True
    seg = res.segments[0]
    assert seg.start_sample == 0
    assert seg.start_sec == 0.0
    assert seg.end_sample <= len(y)


def test_segmentation_with_config_object():
    """Verify segment_cough_bursts correctly honors a SegmentationConfig instance."""
    sr = 12000
    cfg = SegmentationConfig(
        enabled=True,
        min_burst_duration=0.20,
        max_burst_duration=1.0,
        pad_context_duration=0.05,
    )
    y = _create_synthetic_burst_signal(
        sr=sr,
        total_sec=2.0,
        burst_windows=[(0.5, 0.85)],
    )

    res = segment_cough_bursts(y, sr, config=cfg)
    assert res.has_candidates is True
    assert res.num_segments == 1
