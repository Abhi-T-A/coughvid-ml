"""Audio loading, decoding, digital signal processing, and segmentation module."""

from coughvid.audio.dsp import (
    butterworth_lowpass_filter,
    normalize_audio,
    pad_or_trim,
    resample_audio,
    validate_waveform,
)
from coughvid.audio.io import (
    AudioCorruptError,
    AudioData,
    AudioDecodeError,
    AudioError,
    DecoderUnavailableError,
    UnsupportedAudioFormatError,
    find_ffmpeg,
    load_audio,
)
from coughvid.audio.segmenter import (
    CoughSegment,
    SegmentationResult,
    compute_energy_envelope,
    segment_cough_bursts,
)

__all__ = [
    "AudioData",
    "AudioError",
    "AudioDecodeError",
    "UnsupportedAudioFormatError",
    "DecoderUnavailableError",
    "AudioCorruptError",
    "find_ffmpeg",
    "load_audio",
    "validate_waveform",
    "resample_audio",
    "butterworth_lowpass_filter",
    "normalize_audio",
    "pad_or_trim",
    "CoughSegment",
    "SegmentationResult",
    "compute_energy_envelope",
    "segment_cough_bursts",
]
