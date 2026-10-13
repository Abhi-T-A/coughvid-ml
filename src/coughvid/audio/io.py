"""Robust, multi-platform audio input/output and decoding module.

Supports COUGHVID dataset formats (.webm, .ogg, .wav). Uses native decoders
for standard PCM WAV files and FFmpeg subprocess streaming for Opus/AAC containers.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import wavfile


class AudioError(Exception):
    """Base exception for audio processing errors."""


class AudioDecodeError(AudioError):
    """Raised when an audio file cannot be decoded or is corrupt."""


class UnsupportedAudioFormatError(AudioError):
    """Raised when an unsupported audio format is encountered."""


class DecoderUnavailableError(AudioError):
    """Raised when a required external decoder (e.g., FFmpeg) is missing."""


class AudioCorruptError(AudioError):
    """Raised when decoded audio contains non-finite values or zero length."""


@dataclass(frozen=True)
class AudioData:
    """Immutable container for decoded mono audio waveform and metadata."""
    waveform: np.ndarray
    sample_rate: int
    num_samples: int
    duration_sec: float
    source_path: Path
    format: str

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, np.ndarray):
            raise TypeError("waveform must be a numpy.ndarray.")
        if self.waveform.ndim != 1:
            raise ValueError(f"waveform must be 1D (mono), got ndim={self.waveform.ndim}.")
        if self.num_samples == 0:
            raise AudioCorruptError("waveform cannot be empty (0 samples).")
        if not np.all(np.isfinite(self.waveform)):
            raise AudioCorruptError("waveform contains non-finite values (NaN or Inf).")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}.")


def find_ffmpeg() -> str | None:
    """Find the path to the FFmpeg executable on the system.

    Checks:
    1. Environment variable: FFMPEG_PATH
    2. System PATH via shutil.which('ffmpeg')
    3. Windows WinGet package directory: %LOCALAPPDATA%/Microsoft/WinGet/Packages/Gyan.FFmpeg*
    4. Common Unix paths (/opt/homebrew/bin/ffmpeg, /usr/local/bin/ffmpeg, /usr/bin/ffmpeg)
    """
    env_ffmpeg = os.environ.get("FFMPEG_PATH")
    if env_ffmpeg and os.path.isfile(env_ffmpeg):
        return env_ffmpeg

    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path

    # Check Windows WinGet install locations
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        winget_pkg_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkg_dir.exists():
            candidates = list(winget_pkg_dir.glob("**/ffmpeg.exe"))
            if candidates and candidates[0].is_file():
                return str(candidates[0].resolve())

    # Check common standard locations on macOS / Linux
    for fallback in (
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ):
        if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
            return fallback

    return None


def _get_ffmpeg_install_guidance() -> str:
    """Return platform-specific installation instructions for FFmpeg."""
    return (
        "FFmpeg is required to decode compressed audio formats (.webm, .ogg, non-RIFF .wav).\n"
        "Installation instructions by platform:\n"
        "  - Windows: winget install Gyan.FFmpeg   OR   choco install ffmpeg\n"
        "  - Ubuntu/Debian: sudo apt update && sudo apt install -y ffmpeg\n"
        "  - macOS: brew install ffmpeg\n"
        "After installation, ensure 'ffmpeg' is on your system PATH or set FFMPEG_PATH."
    )


def _decode_pcm_wav(file_path: Path) -> tuple[np.ndarray, int]:
    """Decode a standard RIFF/RF64 WAV file into mono float32 array in [-1.0, 1.0]."""
    if file_path.stat().st_size == 0:
        raise AudioCorruptError(f"WAV file {file_path.name} is empty (0 bytes).")

    try:
        sr, data = wavfile.read(str(file_path))
    except Exception as exc:
        raise AudioDecodeError(f"Failed to read WAV file via scipy.io.wavfile: {exc}") from exc

    if data.size == 0:
        raise AudioCorruptError(f"WAV file {file_path.name} contains zero audio samples.")

    # Convert to float32 normalized to [-1.0, 1.0] before downmixing
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if data.dtype == np.uint8:
            float_data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            float_data = data.astype(np.float32) / max(abs(float(info.min)), float(info.max))
    elif np.issubdtype(data.dtype, np.floating):
        float_data = data.astype(np.float32)
    else:
        raise AudioDecodeError(f"Unsupported WAV sample dtype: {data.dtype}")

    # Convert to mono if multi-channel
    if float_data.ndim > 1:
        float_data = np.mean(float_data, axis=-1)

    return float_data, int(sr)


def _decode_ffmpeg_stream(file_path: Path, ffmpeg_bin: str) -> tuple[np.ndarray, int]:
    """Decode an audio file via FFmpeg subprocess into mono float32 array in [-1.0, 1.0]."""
    cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-i", str(file_path),
        "-vn",
        "-ac", "1",
        "-f", "wav",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(
            f"FFmpeg failed to decode {file_path.name}: {err_msg or 'unknown error'}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(f"FFmpeg decoding timed out on {file_path.name}.") from exc

    if not proc.stdout:
        raise AudioCorruptError(f"FFmpeg returned empty stream for {file_path.name}.")

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
            sr, data = wavfile.read(io.BytesIO(proc.stdout))
    except Exception as exc:
        raise AudioDecodeError(
            f"Failed to parse FFmpeg WAV output for {file_path.name}: {exc}"
        ) from exc

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return data.astype(np.float32), int(sr)


def load_audio(path: str | Path) -> AudioData:
    """Load and decode an audio file into a validated AudioData container.

    Parameters
    ----------
    path : str or Path
        Path to the audio file (.webm, .ogg, or .wav).

    Returns
    -------
    AudioData
        Validated container holding waveform, sample rate, and metadata.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    UnsupportedAudioFormatError
        If file extension is not recognized.
    DecoderUnavailableError
        If FFmpeg is required to decode this file but is not installed.
    AudioDecodeError
        If decoding fails due to corruption or format error.
    AudioCorruptError
        If decoded waveform is empty or contains non-finite values.
    """
    file_path = Path(path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise AudioCorruptError(f"Audio file '{file_path.name}' is empty (0 bytes).")

    ext = file_path.suffix.lower().lstrip(".")
    if ext not in ("webm", "ogg", "wav"):
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '.{ext}'. Supported formats are: webm, ogg, wav."
        )

    ffmpeg_bin = find_ffmpeg()

    # Attempt native PCM decoding for .wav files
    if ext == "wav":
        try:
            waveform, sr = _decode_pcm_wav(file_path)
        except Exception as wav_exc:
            # If native reader fails (e.g. MP4/AAC container in a .wav file), try FFmpeg fallback
            if ffmpeg_bin:
                try:
                    waveform, sr = _decode_ffmpeg_stream(file_path, ffmpeg_bin)
                except Exception as ffmpeg_exc:
                    raise AudioDecodeError(
                        f"Failed to decode '{file_path.name}' with native WAV reader ({wav_exc}) "
                        f"and FFmpeg ({ffmpeg_exc})."
                    ) from ffmpeg_exc
            else:
                raise DecoderUnavailableError(
                    f"'{file_path.name}' is not a standard RIFF WAV file ({wav_exc}) and requires FFmpeg to decode.\n"
                    + _get_ffmpeg_install_guidance()
                ) from wav_exc
    else:
        # .webm and .ogg require container-level Opus decoding via FFmpeg
        if not ffmpeg_bin:
            raise DecoderUnavailableError(
                f"File '{file_path.name}' is an Opus-encoded .{ext} file which requires FFmpeg.\n"
                + _get_ffmpeg_install_guidance()
            )
        waveform, sr = _decode_ffmpeg_stream(file_path, ffmpeg_bin)

    return AudioData(
        waveform=waveform,
        sample_rate=sr,
        num_samples=len(waveform),
        duration_sec=float(len(waveform)) / float(sr),
        source_path=file_path,
        format=ext,
    )
