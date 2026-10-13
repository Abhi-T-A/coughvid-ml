"""Unit tests for audio I/O, format support, and decoder handling."""

from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile

from coughvid.audio.io import (
    AudioCorruptError,
    AudioData,
    AudioDecodeError,
    DecoderUnavailableError,
    UnsupportedAudioFormatError,
    find_ffmpeg,
    load_audio,
)


def test_load_audio_nonexistent():
    """Verify load_audio raises FileNotFoundError for missing paths."""
    with pytest.raises(FileNotFoundError):
        load_audio("nonexistent_recording_file.wav")


def test_load_audio_unsupported_extension(tmp_path):
    """Verify load_audio raises UnsupportedAudioFormatError for unrecognized formats."""
    bad_file = tmp_path / "recording.mp3"
    bad_file.write_bytes(b"dummy")
    with pytest.raises(UnsupportedAudioFormatError, match="Unsupported audio format '.mp3'"):
        load_audio(bad_file)


def test_decode_synthetic_pcm_wav_mono(tmp_path):
    """Verify decoding a synthetic 16-bit PCM mono WAV file."""
    sr = 16000
    duration = 0.5
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    # Generate 440 Hz sine wave with amplitude 0.8 in int16
    int16_max = 32767
    signal = (0.8 * int16_max * np.sin(2 * np.pi * 440 * t)).astype(np.int16)

    wav_path = tmp_path / "test_synth.wav"
    wavfile.write(str(wav_path), sr, signal)

    audio = load_audio(wav_path)
    assert isinstance(audio, AudioData)
    assert audio.sample_rate == sr
    assert audio.num_samples == num_samples
    assert audio.waveform.ndim == 1
    assert audio.waveform.dtype == np.float32
    assert np.isclose(audio.duration_sec, duration, atol=1e-3)
    assert audio.format == "wav"
    assert np.max(np.abs(audio.waveform)) <= 1.0


def test_decode_synthetic_pcm_wav_stereo(tmp_path):
    """Verify stereo audio is automatically downmixed to 1D mono."""
    sr = 22050
    num_samples = 1000
    # 2D stereo: left channel 0.4, right channel 0.6
    stereo = np.zeros((num_samples, 2), dtype=np.int16)
    stereo[:, 0] = 10000
    stereo[:, 1] = 20000

    wav_path = tmp_path / "test_stereo.wav"
    wavfile.write(str(wav_path), sr, stereo)

    audio = load_audio(wav_path)
    assert audio.waveform.ndim == 1
    assert audio.num_samples == num_samples
    # Average of 10000 and 20000 normalized to int16 is ~15000 / 32768 ~ 0.457
    expected = 15000.0 / 32768.0
    assert np.allclose(audio.waveform, expected, atol=1e-3)


def test_decode_empty_wav(tmp_path):
    """Verify empty WAV file raises AudioCorruptError or AudioDecodeError."""
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    with pytest.raises((AudioDecodeError, AudioCorruptError)):
        load_audio(empty_file)


def test_decode_webm_without_ffmpeg_raises_guidance(tmp_path, monkeypatch):
    """Verify attempting to decode .webm without FFmpeg raises DecoderUnavailableError with instructions."""
    webm_file = tmp_path / "sample.webm"
    webm_file.write_bytes(b"dummy webm data")

    # Force find_ffmpeg to return None
    monkeypatch.setattr("coughvid.audio.io.find_ffmpeg", lambda: None)

    with pytest.raises(DecoderUnavailableError, match="requires FFmpeg"):
        load_audio(webm_file)


def test_decode_ogg_without_ffmpeg_raises_guidance(tmp_path, monkeypatch):
    """Verify attempting to decode .ogg without FFmpeg raises DecoderUnavailableError."""
    ogg_file = tmp_path / "sample.ogg"
    ogg_file.write_bytes(b"dummy ogg data")

    monkeypatch.setattr("coughvid.audio.io.find_ffmpeg", lambda: None)

    with pytest.raises(DecoderUnavailableError, match="requires FFmpeg"):
        load_audio(ogg_file)


def test_find_ffmpeg_env_var_override(monkeypatch, tmp_path):
    """Verify FFMPEG_PATH environment variable is respected when executable exists."""
    fake_ffmpeg = tmp_path / "fake_ffmpeg.exe"
    fake_ffmpeg.write_bytes(b"binary")
    monkeypatch.setenv("FFMPEG_PATH", str(fake_ffmpeg))

    # On Windows, os.access(..., os.X_OK) passes for existing files
    detected = find_ffmpeg()
    assert detected == str(fake_ffmpeg)
