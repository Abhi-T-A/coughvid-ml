"""Unit tests for configuration loading, path resolution, and schema validation."""

import os
from pathlib import Path
import pytest
from pydantic import ValidationError

from coughvid.config import (
    AppConfig,
    AudioConfig,
    DatasetConfig,
    get_project_root,
    load_config,
)


def test_project_root():
    """Verify that get_project_root correctly identifies the project root."""
    root = get_project_root()
    assert root.exists()
    assert (root / "configs" / "base_config.yaml").exists()


def test_default_config_initialization():
    """Verify default AppConfig creates valid nested configurations."""
    cfg = AppConfig()
    assert cfg.dataset.cough_threshold == 0.8
    assert cfg.audio.target_sr == 12000
    assert cfg.audio.lowpass_hz == 6000
    assert cfg.splits.test_size == 0.15
    assert cfg.splits.n_splits == 5


def test_load_config_from_yaml():
    """Verify loading from the default YAML file."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.dataset.target_task == "covid_binary"
    assert cfg.segmentation.enabled is True
    assert len(cfg.features.psd_bands) == 7


def test_raw_data_dir_resolution_default():
    """Verify that raw_data_dir resolves to the existing dataset directory."""
    cfg = load_config()
    raw_dir = cfg.resolve_raw_data_dir()
    assert raw_dir.exists(), f"Raw directory does not exist: {raw_dir}"
    assert raw_dir.is_dir()


def test_metadata_path_resolution():
    """Verify that metadata_compiled.csv is located within the resolved dataset dir."""
    cfg = load_config()
    meta_path = cfg.resolve_metadata_path()
    assert meta_path.exists(), f"Metadata CSV does not exist: {meta_path}"
    assert meta_path.name == "metadata_compiled.csv"


def test_env_var_override(monkeypatch, tmp_path):
    """Verify that COUGHVID_DATA_DIR environment variable overrides configured path."""
    fake_data_dir = tmp_path / "mock_coughvid"
    fake_data_dir.mkdir()

    monkeypatch.setenv("COUGHVID_DATA_DIR", str(fake_data_dir))
    cfg = AppConfig()
    resolved = cfg.resolve_raw_data_dir()
    assert resolved == fake_data_dir.resolve()


def test_missing_data_dir_raises_filenotfound(monkeypatch, tmp_path):
    """Verify that pointing to a nonexistent directory raises FileNotFoundError with guidance."""
    nonexistent = tmp_path / "does_not_exist"
    monkeypatch.setenv("COUGHVID_DATA_DIR", str(nonexistent))

    cfg = AppConfig()
    with pytest.raises(FileNotFoundError, match="Raw dataset directory not found"):
        cfg.resolve_raw_data_dir()


def test_invalid_lowpass_hz_above_nyquist():
    """Verify AudioConfig rejects lowpass_hz > target_sr / 2 (Nyquist limit)."""
    with pytest.raises(ValidationError, match="lowpass_hz .* cannot exceed the target Nyquist"):
        AudioConfig(target_sr=12000, lowpass_hz=7000)  # 7000 > 12000 / 2 (6000) -> should fail


def test_invalid_cough_threshold_bounds():
    """Verify DatasetConfig rejects threshold outside [0, 1]."""
    with pytest.raises(ValidationError):
        DatasetConfig(cough_threshold=1.5)

    with pytest.raises(ValidationError):
        DatasetConfig(cough_threshold=-0.1)


def test_missing_config_yaml_file(tmp_path):
    """Verify load_config raises FileNotFoundError when given a nonexistent YAML path."""
    fake_yaml = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(config_path=fake_yaml)
