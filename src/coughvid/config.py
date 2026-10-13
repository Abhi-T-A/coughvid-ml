"""Configuration loader and schema validation for the COUGHVID ML pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field, field_validator


def get_project_root() -> Path:
    """Return the absolute path to the coughvid-ml project root directory."""
    # Assuming config.py is at: <root>/src/coughvid/config.py
    return Path(__file__).resolve().parent.parent.parent


class DatasetConfig(BaseModel):
    """Dataset paths and selection thresholds."""
    raw_data_dir: str = Field(
        default="../AT/ED/coughvid_20211012",
        description="Path to directory containing raw audio and metadata files.",
    )
    metadata_filename: str = Field(
        default="metadata_compiled.csv",
        description="Name of the compiled metadata CSV file.",
    )
    cough_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="EPFL recommended cough detection confidence threshold.",
    )
    target_task: str = Field(
        default="covid_binary",
        description="Classification task ('covid_binary', 'expert_covid', 'wet_dry').",
    )


class AudioConfig(BaseModel):
    """Acoustic preprocessing parameters."""
    target_sr: int = Field(default=12000, gt=0, description="Target sample rate (Hz).")
    lowpass_hz: int = Field(default=6000, gt=0, description="Low-pass filter cutoff (Hz).")
    filter_order: int = Field(default=5, gt=0, description="Butterworth filter order.")
    max_duration: float = Field(default=10.0, gt=0.0, description="Maximum duration (s).")
    normalize_peak: bool = Field(default=True, description="Whether to peak-normalize audio.")

    @field_validator("lowpass_hz")
    @classmethod
    def validate_lowpass(cls, v: int, info: Any) -> int:
        target_sr = info.data.get("target_sr", 12000)
        nyquist = target_sr / 2.0
        if v > nyquist:
            raise ValueError(
                f"lowpass_hz ({v}) cannot exceed the target Nyquist frequency ({nyquist}) "
                f"for target_sr={target_sr} to prevent aliasing."
            )
        return v


class SegmentationConfig(BaseModel):
    """Cough burst energy-envelope hysteresis segmentation parameters."""
    enabled: bool = Field(default=True, description="Enable cough burst segmentation.")
    min_burst_duration: float = Field(default=0.2, gt=0.0, description="Minimum burst length (s).")
    max_burst_duration: float = Field(default=1.5, gt=0.0, description="Maximum burst length (s).")
    power_threshold_factor: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Relative power threshold for burst onset/offset."
    )
    pad_context_duration: float = Field(
        default=0.05, ge=0.0, description="Context padding added around detected burst (s)."
    )


class FeaturesConfig(BaseModel):
    """Feature extraction parameters."""
    n_mfcc: int = Field(default=20, gt=0, description="Number of MFCC coefficients.")
    n_fft: int = Field(default=512, gt=0, description="FFT window size.")
    hop_length: int = Field(default=128, gt=0, description="Hop length between frames.")
    n_mels: int = Field(default=64, gt=0, description="Number of Mel frequency bins.")
    spectrogram_frames: int = Field(
        default=128, gt=0, description="Fixed number of frames for spectrogram tensors."
    )
    f_min: float = Field(default=0.0, ge=0.0, description="Minimum frequency for Mel scale (Hz).")
    f_max: Optional[float] = Field(default=None, description="Maximum frequency for Mel scale (Hz).")
    batch_size: int = Field(default=100, gt=0, description="Extraction batch size for persistence.")
    num_workers: int = Field(default=4, ge=1, description="Worker threads for batch extraction.")
    psd_bands: List[Tuple[int, int]] = Field(
        default=[
            (0, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 3000),
            (3000, 4000),
            (4000, 5000),
            (5000, 6000),
        ],
        description="Frequency bands for PSD energy integration.",
    )


class SplitsConfig(BaseModel):
    """Dataset splitting and cross-validation parameters."""
    test_size: float = Field(default=0.15, gt=0.0, lt=1.0, description="Holdout test proportion.")
    n_splits: int = Field(default=5, ge=2, description="Number of cross-validation folds.")
    random_seed: int = Field(default=42, description="Global random seed.")
    group_by_cluster: bool = Field(
        default=False,
        description="Whether to group CV splits by demographic/spatial cluster.",
    )


class TrainingConfig(BaseModel):
    """Model training and probability calibration parameters."""
    random_seed: int = Field(default=42, description="Random seed for estimators.")
    calibrate_probabilities: bool = Field(
        default=True,
        description="Whether to apply probability calibration (e.g. CalibratedClassifierCV).",
    )


class PathsConfig(BaseModel):
    """Pipeline working and artifact paths."""
    data_dir: str = Field(default="data", description="Working data output directory.")
    artifacts_dir: str = Field(default="artifacts", description="Model and checkpoint directory.")
    reports_dir: str = Field(default="reports", description="Evaluation reports directory.")


class AppConfig(BaseModel):
    """Master application configuration."""
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def resolve_raw_data_dir(self, project_root: Path | None = None) -> Path:
        """Resolve and validate the raw dataset directory path.

        Priority order:
        1. Environment variable: COUGHVID_DATA_DIR
        2. Configured dataset.raw_data_dir (relative to project_root or absolute)
        """
        root = project_root or get_project_root()
        env_path = os.environ.get("COUGHVID_DATA_DIR")
        if env_path:
            candidate = Path(env_path).resolve()
        else:
            raw_path = Path(self.dataset.raw_data_dir)
            if raw_path.is_absolute():
                candidate = raw_path
            else:
                candidate = (root / raw_path).resolve()

        if not candidate.exists():
            raise FileNotFoundError(
                f"Raw dataset directory not found at: {candidate}\n"
                f"Please ensure the dataset exists or set the environment variable:\n"
                f"  export COUGHVID_DATA_DIR=/path/to/coughvid_20211012   (Linux/macOS)\n"
                f"  $env:COUGHVID_DATA_DIR='C:\\path\\to\\coughvid_20211012' (Windows PowerShell)"
            )
        return candidate

    def resolve_metadata_path(self, project_root: Path | None = None) -> Path:
        """Resolve and validate the path to metadata_compiled.csv."""
        raw_dir = self.resolve_raw_data_dir(project_root)
        meta_file = raw_dir / self.dataset.metadata_filename
        if not meta_file.exists():
            raise FileNotFoundError(
                f"Metadata file '{self.dataset.metadata_filename}' not found in {raw_dir}"
            )
        return meta_file

    def resolve_path(self, rel_or_abs: str | Path, project_root: Path | None = None) -> Path:
        """Resolve an internal working path against project_root."""
        root = project_root or get_project_root()
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (root / p).resolve()


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load configuration from a YAML file or use defaults."""
    root = get_project_root()
    if config_path is None:
        default_yaml = root / "configs" / "base_config.yaml"
        if default_yaml.exists():
            config_path = default_yaml

    if config_path is not None:
        p = Path(config_path)
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)

    return AppConfig()
