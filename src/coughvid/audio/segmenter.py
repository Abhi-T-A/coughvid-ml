"""Energy-envelope hysteresis cough burst segmentation module.

Detects candidate cough expiratory bursts by tracking short-time energy (RMS)
with dual-threshold hysteresis (separate onset and offset thresholds),
contextual padding, and minimum/maximum duration constraints.

LIMITATIONS:
Energy-based segmentation identifies transient acoustic energy concentrations
characteristic of expiratory cough bursts, but cannot perfectly discriminate
cough sounds from speech plosives, loud background thumps, or throat-clearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from coughvid.config import SegmentationConfig


@dataclass(frozen=True)
class CoughSegment:
    """Represents a single detected candidate cough burst."""
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float
    peak_energy: float
    mean_energy: float
    waveform: np.ndarray


@dataclass
class SegmentationResult:
    """Encapsulates the complete result of acoustic burst segmentation."""
    segments: List[CoughSegment]
    has_candidates: bool
    envelope: np.ndarray
    envelope_times: np.ndarray
    onset_threshold: float
    offset_threshold: float
    noise_floor: float
    concatenated_waveform: np.ndarray = field(repr=False)

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def total_duration_sec(self) -> float:
        return sum(s.duration_sec for s in self.segments)


def compute_energy_envelope(
    y: np.ndarray,
    sr: int,
    frame_length_ms: float = 30.0,
    hop_length_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the short-time RMS energy envelope of an audio waveform.

    Parameters
    ----------
    y : np.ndarray
        1D audio samples.
    sr : int
        Sample rate (Hz).
    frame_length_ms : float, default=30.0
        Duration of the analysis window in milliseconds.
    hop_length_ms : float, default=10.0
        Step size between frames in milliseconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (envelope, envelope_times)
    """
    if len(y) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    frame_length = max(1, int(sr * (frame_length_ms / 1000.0)))
    hop_length = max(1, int(sr * (hop_length_ms / 1000.0)))

    # Compute moving-window RMS energy
    squared = y.astype(np.float64) ** 2
    window = np.ones(frame_length, dtype=np.float64) / frame_length

    # Convolve to compute short-time mean square energy
    mean_square = np.convolve(squared, window, mode="same")
    envelope = np.sqrt(np.maximum(mean_square, 0.0)).astype(np.float32)

    # Subsample according to hop length for efficient thresholding
    indices = np.arange(0, len(y), hop_length)
    env_subsampled = envelope[indices]
    times_subsampled = (indices / sr).astype(np.float32)

    return env_subsampled, times_subsampled


def segment_cough_bursts(
    y: np.ndarray,
    sr: int,
    config: SegmentationConfig | None = None,
    onset_threshold_factor: float = 0.20,
    offset_threshold_factor: float = 0.08,
    frame_length_ms: float = 30.0,
    hop_length_ms: float = 10.0,
    min_burst_duration: float = 0.15,
    max_burst_duration: float = 2.0,
    merge_gap_duration: float = 0.10,
    pad_context_duration: float = 0.05,
) -> SegmentationResult:
    """Identify candidate cough bursts in an audio waveform using hysteresis energy tracking.

    Parameters
    ----------
    y : np.ndarray
        1D mono audio waveform.
    sr : int
        Sample rate (Hz).
    config : SegmentationConfig, optional
        Pre-configured settings. If provided, values override the default arguments.
    onset_threshold_factor : float, default=0.20
        Fraction of dynamic range above noise floor to trigger burst onset.
    offset_threshold_factor : float, default=0.08
        Fraction of dynamic range above noise floor to trigger burst offset (hysteresis).
    frame_length_ms : float, default=30.0
        Frame analysis window duration in ms.
    hop_length_ms : float, default=10.0
        Hop size between frames in ms.
    min_burst_duration : float, default=0.15
        Minimum valid burst duration in seconds (shorter bursts discarded).
    max_burst_duration : float, default=2.0
        Maximum valid burst duration in seconds (longer bursts discarded).
    merge_gap_duration : float, default=0.10
        Maximum silence gap in seconds between bursts to merge into a single event.
    pad_context_duration : float, default=0.05
        Context padding added to start and end of each burst in seconds.

    Returns
    -------
    SegmentationResult
        Object containing detected segments, boundaries, envelope, and concatenated audio.
    """
    if config is not None:
        min_burst_duration = config.min_burst_duration
        max_burst_duration = config.max_burst_duration
        pad_context_duration = config.pad_context_duration
        onset_threshold_factor = getattr(config, "power_threshold_factor", onset_threshold_factor)

    if len(y) == 0:
        return SegmentationResult(
            segments=[],
            has_candidates=False,
            envelope=np.array([], dtype=np.float32),
            envelope_times=np.array([], dtype=np.float32),
            onset_threshold=0.0,
            offset_threshold=0.0,
            noise_floor=0.0,
            concatenated_waveform=np.array([], dtype=np.float32),
        )

    # 1. Compute RMS energy envelope
    env, env_times = compute_energy_envelope(
        y, sr, frame_length_ms=frame_length_ms, hop_length_ms=hop_length_ms
    )

    if len(env) == 0:
        return SegmentationResult(
            segments=[],
            has_candidates=False,
            envelope=env,
            envelope_times=env_times,
            onset_threshold=0.0,
            offset_threshold=0.0,
            noise_floor=0.0,
            concatenated_waveform=np.array([], dtype=np.float32),
        )

    peak_env = float(np.max(env))
    noise_floor = float(np.percentile(env, 15.0))  # 15th percentile as baseline ambient noise
    dynamic_range = peak_env - noise_floor

    # Handle silent or flat recordings
    if dynamic_range < 1e-5 or peak_env < 1e-4:
        return SegmentationResult(
            segments=[],
            has_candidates=False,
            envelope=env,
            envelope_times=env_times,
            onset_threshold=0.0,
            offset_threshold=0.0,
            noise_floor=noise_floor,
            concatenated_waveform=np.array([], dtype=np.float32),
        )

    onset_thresh = noise_floor + (onset_threshold_factor * dynamic_range)
    offset_thresh = noise_floor + (offset_threshold_factor * dynamic_range)

    # Ensure valid hysteresis relation
    if offset_thresh >= onset_thresh:
        offset_thresh = onset_thresh * 0.5

    # 2. Hysteresis onset-offset state machine
    candidates: List[Tuple[int, int]] = []
    in_burst = False
    start_idx = 0

    for i, val in enumerate(env):
        if not in_burst:
            if val >= onset_thresh:
                in_burst = True
                start_idx = i
        else:
            if val < offset_thresh:
                in_burst = False
                candidates.append((start_idx, i))

    if in_burst:
        candidates.append((start_idx, len(env) - 1))

    # Convert candidate frame indices to time boundaries
    time_candidates: List[Tuple[float, float]] = [
        (float(env_times[start]), float(env_times[end]))
        for start, end in candidates
    ]

    # 3. Merge close candidate regions
    merged: List[Tuple[float, float]] = []
    for start, end in time_candidates:
        if not merged:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            if (start - prev_end) <= merge_gap_duration:
                # Merge into previous candidate
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

    # 4. Enforce duration bounds, apply context padding, and construct segments
    segments: List[CoughSegment] = []
    for start_t, end_t in merged:
        raw_duration = end_t - start_t
        if raw_duration < min_burst_duration or raw_duration > max_burst_duration:
            continue

        # Add context padding
        padded_start_t = max(0.0, start_t - pad_context_duration)
        padded_end_t = min(len(y) / sr, end_t + pad_context_duration)

        start_samp = int(padded_start_t * sr)
        end_samp = min(len(y), int(padded_end_t * sr))

        segment_audio = y[start_samp:end_samp]
        if len(segment_audio) == 0:
            continue

        seg_energy = segment_audio ** 2
        segments.append(
            CoughSegment(
                start_sample=start_samp,
                end_sample=end_samp,
                start_sec=padded_start_t,
                end_sec=padded_end_t,
                duration_sec=padded_end_t - padded_start_t,
                peak_energy=float(np.max(seg_energy)),
                mean_energy=float(np.mean(seg_energy)),
                waveform=segment_audio.copy(),
            )
        )

    # Concatenate all segmented bursts
    if segments:
        concatenated = np.concatenate([s.waveform for s in segments]).astype(np.float32)
    else:
        concatenated = np.array([], dtype=np.float32)

    return SegmentationResult(
        segments=segments,
        has_candidates=len(segments) > 0,
        envelope=env,
        envelope_times=env_times,
        onset_threshold=float(onset_thresh),
        offset_threshold=float(offset_thresh),
        noise_floor=float(noise_floor),
        concatenated_waveform=concatenated,
    )
