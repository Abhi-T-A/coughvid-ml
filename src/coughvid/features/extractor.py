"""Batch acoustic feature extraction and Parquet caching module.

Extracts recording-level and segment-level acoustic features from approved split
manifests (train.csv / test.csv). Preprocesses waveforms using validated DSP,
detects candidate cough bursts via energy-envelope hysteresis, computes
deterministic tabular features and fixed-shape Log-Mel spectrograms, and saves
persistent outputs using PyArrow Parquet and compressed NumPy archives.

KEY ARCHITECTURAL GUARANTEES:
1. Leak-Free Association: Segments from the same recording are never split across
   partitions; features are aggregated to a unique recording-level UUID representation.
2. Safe Failure Handling: Failed decodings and unsegmented recordings are NEVER
   silently imputed or labeled healthy. They are recorded in structured error manifests.
3. Resumability & Configuration Fingerprinting: Extraction tracks existing UUIDs and
   validates that the active DSP/feature configuration matches the cached Parquet
   fingerprint, preventing silent retention of stale features when settings change.
4. Separate Tensor Storage: Spectrogram tensors are stored in compressed NPZ archives
   keyed by recording UUID rather than relying on positional correspondence.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from coughvid.audio.dsp import butterworth_lowpass_filter, normalize_audio, resample_audio
from coughvid.audio.io import (
    AudioCorruptError,
    AudioDecodeError,
    AudioError,
    DecoderUnavailableError,
    UnsupportedAudioFormatError,
    load_audio,
)
from coughvid.audio.segmenter import CoughSegment, SegmentationResult, segment_cough_bursts
from coughvid.config import AppConfig, FeaturesConfig, SegmentationConfig, load_config
from coughvid.features.schema import (
    ACOUSTIC_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    RECORDING_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    SEGMENTATION_FEATURE_COLUMNS,
    SIGNAL_FEATURE_COLUMNS,
    compute_config_fingerprint,
    validate_features_dataframe,
)
from coughvid.features.spectrogram import compute_log_mel_spectrogram
from coughvid.features.tabular import extract_tabular_features

logger = logging.getLogger(__name__)


class ConfigurationMismatchError(ValueError):
    """Raised when attempting to resume feature extraction with an incompatible configuration."""


@dataclass
class ExtractionErrorRecord:
    """Structured record of a recording that failed ingestion, segmentation, or extraction."""
    uuid: str
    audio_path: str
    label: Optional[int]
    status: Optional[str]
    error_stage: str  # "io", "dsp", "segmentation", "features"
    error_type: str
    error_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionSummary:
    """Summary statistics for a feature extraction run."""
    partition: str
    manifest_path: str
    total_manifest_records: int
    processed_records: int
    success_count: int
    exclusion_count: int
    failure_reasons: Dict[str, int]
    class_distribution_input: Dict[str, int]
    class_distribution_success: Dict[str, int]
    class_distribution_excluded: Dict[str, int]
    class_success_rates: Dict[str, float]
    num_tabular_features: int  # Authoritative acoustic feature count (208)
    total_columns: int  # Full table width including metadata (217)
    spectrogram_shape: List[int]
    parquet_path: str
    spectrogram_path: str
    error_manifest_path: str
    nan_or_inf_detected: bool
    schema_version: str = SCHEMA_VERSION
    config_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_spectrogram(spectrogram_path: Path | str, uuid: str) -> np.ndarray:
    """Retrieve a fixed-shape Log-Mel spectrogram by recording UUID from an NPZ archive.

    Parameters
    ----------
    spectrogram_path : Path or str
        Path to the .npz archive (e.g. data/features/train_spectrograms.npz).
    uuid : str
        Unique recording identifier.

    Returns
    -------
    np.ndarray
        2D float32 array of shape (n_mels, target_frames).

    Raises
    ------
    KeyError
        If the specified UUID is not present in the archive.
    FileNotFoundError
        If the archive does not exist on disk.
    """
    path = Path(spectrogram_path)
    if not path.exists():
        raise FileNotFoundError(f"Spectrogram archive not found: {path}")

    with np.load(path) as data:
        if uuid not in data:
            raise KeyError(f"Recording UUID '{uuid}' not found in spectrogram archive {path.name}.")
        return data[uuid].copy()


def resolve_recording_audio_path(
    audio_path_raw: Optional[str],
    uuid: str,
    raw_data_dir: Path,
) -> Optional[Path]:
    """Resolve the physical audio file path on disk, supporting portable execution."""
    if audio_path_raw:
        direct = Path(audio_path_raw)
        if direct.exists() and direct.is_file():
            return direct

    # Fallback to checking raw_data_dir with supported extensions
    for ext in ("webm", "ogg", "wav"):
        candidate = raw_data_dir / f"{uuid}.{ext}"
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def extract_recording_features(
    record: Dict[str, Any],
    raw_data_dir: Path,
    config: AppConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[np.ndarray], Optional[ExtractionErrorRecord]]:
    """Process a single audio recording end-to-end.

    Returns
    -------
    tuple:
        (recording_level_row, segment_level_rows, spectrogram_tensor, error_record)
        If extraction succeeds, error_record is None.
        If extraction fails or no segments are found, first three items are None and
        error_record contains structured diagnosis.
    """
    uuid = str(record.get("uuid", "")).strip()
    raw_audio_path = record.get("audio_path")
    label_val = record.get("label")
    try:
        label = int(label_val) if label_val is not None and label_val != "" else None
    except (ValueError, TypeError):
        label = None
    status = record.get("status")

    # 1. Resolve audio file path
    resolved_path = resolve_recording_audio_path(raw_audio_path, uuid, raw_data_dir)
    if resolved_path is None:
        err = ExtractionErrorRecord(
            uuid=uuid,
            audio_path=str(raw_audio_path),
            label=label,
            status=status,
            error_stage="io",
            error_type="FileNotFoundError",
            error_message=f"Audio file could not be found on disk for UUID '{uuid}'.",
        )
        return None, None, None, err

    # 2. Audio Decoding
    try:
        audio = load_audio(resolved_path)
    except (AudioDecodeError, DecoderUnavailableError, AudioCorruptError, UnsupportedAudioFormatError, FileNotFoundError) as e:
        err = ExtractionErrorRecord(
            uuid=uuid,
            audio_path=str(resolved_path),
            label=label,
            status=status,
            error_stage="io",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return None, None, None, err
    except Exception as e:
        err = ExtractionErrorRecord(
            uuid=uuid,
            audio_path=str(resolved_path),
            label=label,
            status=status,
            error_stage="io",
            error_type="UnexpectedAudioError",
            error_message=str(e),
        )
        return None, None, None, err

    # 3. DSP Preprocessing
    target_sr = config.audio.target_sr
    try:
        # Resample
        y, sr = resample_audio(audio.waveform, audio.sample_rate, target_sr)

        # Nyquist-safe Lowpass filter
        nyquist = sr / 2.0
        cutoff = float(config.audio.lowpass_hz)
        if cutoff >= nyquist:
            cutoff = nyquist * 0.95
        if cutoff > 0:
            y = butterworth_lowpass_filter(y, sr, cutoff, order=config.audio.filter_order)

        # Normalize
        if config.audio.normalize_peak:
            y = normalize_audio(y, method="peak")
    except Exception as e:
        err = ExtractionErrorRecord(
            uuid=uuid,
            audio_path=str(resolved_path),
            label=label,
            status=status,
            error_stage="dsp",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return None, None, None, err

    # 4. Energy-Envelope Cough Segmentation
    try:
        seg_result = segment_cough_bursts(y, sr, config=config.segmentation)
    except Exception as e:
        err = ExtractionErrorRecord(
            uuid=uuid,
            audio_path=str(resolved_path),
            label=label,
            status=status,
            error_stage="segmentation",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return None, None, None, err

    # Check candidate segments
    if not seg_result.has_candidates or len(seg_result.segments) == 0:
        err = ExtractionErrorRecord(
            uuid=uuid,
            audio_path=str(resolved_path),
            label=label,
            status=status,
            error_stage="segmentation",
            error_type="NoSegmentsDetectedError",
            error_message="Energy-envelope hysteresis detected zero valid cough bursts.",
        )
        return None, None, None, err

    # 5. Acoustic Feature Extraction
    try:
        # Extract on concatenated burst waveform
        acoustic_feats = extract_tabular_features(seg_result.concatenated_waveform, sr, config=config.features)

        # Extract fixed-shape Log-Mel spectrogram on concatenated burst waveform
        spectrogram = compute_log_mel_spectrogram(
            seg_result.concatenated_waveform,
            sr=sr,
            n_mels=config.features.n_mels,
            n_fft=config.features.n_fft,
            hop_length=config.features.hop_length,
            target_frames=config.features.spectrogram_frames,
            f_min=config.features.f_min,
            f_max=config.features.f_max,
        )
    except Exception as e:
        err = ExtractionErrorRecord(
            uuid=uuid,
            audio_path=str(resolved_path),
            label=label,
            status=status,
            error_stage="features",
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return None, None, None, err

    # 6. Aggregate burst statistics (11 features)
    durations = [s.duration_sec for s in seg_result.segments]
    energies = [s.peak_energy for s in seg_result.segments]
    mean_energies = [s.mean_energy for s in seg_result.segments]

    burst_summary: Dict[str, Any] = {
        "num_cough_segments": len(seg_result.segments),
        "total_cough_duration": float(sum(durations)),
        "mean_cough_duration": float(np.mean(durations)),
        "std_cough_duration": float(np.std(durations)) if len(durations) > 1 else 0.0,
        "min_cough_duration": float(np.min(durations)),
        "max_cough_duration": float(np.max(durations)),
        "peak_burst_energy": float(np.max(energies)),
        "mean_burst_energy": float(np.mean(mean_energies)),
        "segmenter_noise_floor": float(seg_result.noise_floor),
        "segmenter_onset_threshold": float(seg_result.onset_threshold),
        "segmenter_offset_threshold": float(seg_result.offset_threshold),
    }

    # 7. Construct recording-level row strictly adhering to schema order
    raw_row: Dict[str, Any] = {
        "uuid": uuid,
        "label": label,
        "status": status,
        "cough_detected": float(record["cough_detected"]) if record.get("cough_detected") not in (None, "") else None,
        "age": float(record["age"]) if record.get("age") not in (None, "") else None,
        "gender": record.get("gender"),
        "respiratory_condition": record.get("respiratory_condition") in (True, "True", "true", 1, "1"),
        "fever_muscle_pain": record.get("fever_muscle_pain") in (True, "True", "true", 1, "1"),
        "audio_path": str(resolved_path),
    }
    raw_row.update(burst_summary)
    raw_row.update(acoustic_feats)

    # Order keys strictly according to authoritative schema
    rec_row: Dict[str, Any] = {col: raw_row[col] for col in RECORDING_FEATURE_COLUMNS if col in raw_row}

    # 8. Construct individual segment rows
    segment_rows: List[Dict[str, Any]] = []
    for idx, seg in enumerate(seg_result.segments):
        try:
            seg_feats = extract_tabular_features(seg.waveform, sr, config=config.features)
        except Exception:
            seg_feats = {}
        s_row: Dict[str, Any] = {
            "uuid": uuid,
            "segment_index": idx,
            "label": label,
            "start_sample": seg.start_sample,
            "end_sample": seg.end_sample,
            "start_sec": seg.start_sec,
            "end_sec": seg.end_sec,
            "duration_sec": seg.duration_sec,
            "peak_energy": seg.peak_energy,
            "mean_energy": seg.mean_energy,
        }
        s_row.update(seg_feats)
        segment_rows.append(s_row)

    return rec_row, segment_rows, spectrogram, None


class BatchFeatureExtractor:
    """Manages resilient batch extraction, Parquet writing, and resumable caching."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        output_dir: Optional[Path | str] = None,
        reports_dir: Optional[Path | str] = None,
    ) -> None:
        self.config = config or load_config()
        self.raw_data_dir = self.config.resolve_raw_data_dir()

        self.data_dir = Path(output_dir or (Path(self.config.paths.data_dir) / "features"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.reports_dir = Path(reports_dir or self.config.paths.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _read_manifest(self, manifest_path: Path) -> List[Dict[str, Any]]:
        """Read records from split manifest CSV."""
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        records: List[Dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records

    def _check_existing_cache_compatibility(self, parquet_path: Path) -> Set[str]:
        """Verify configuration fingerprint and return set of already cached UUIDs."""
        if not parquet_path.exists():
            return set()

        try:
            parquet_file = pq.ParquetFile(parquet_path)
            meta = parquet_file.schema_arrow.metadata or {}
            existing_fp = meta.get(b"config_fingerprint")
            current_fp = compute_config_fingerprint(self.config)

            # Check sidecar meta JSON as fallback if arrow metadata is missing
            meta_json_path = parquet_path.with_suffix(".meta.json")
            if existing_fp is None and meta_json_path.exists():
                try:
                    with open(meta_json_path, "r", encoding="utf-8") as f:
                        sidecar_meta = json.load(f)
                    if "config_fingerprint" in sidecar_meta:
                        existing_fp = sidecar_meta["config_fingerprint"].encode("utf-8")
                except Exception:
                    pass

            if existing_fp is not None:
                existing_fp_str = existing_fp.decode("utf-8")
                if existing_fp_str != current_fp:
                    raise ConfigurationMismatchError(
                        f"Cannot resume feature extraction at '{parquet_path.name}': "
                        f"Existing cache was generated with configuration fingerprint '{existing_fp_str}', "
                        f"but current pipeline configuration has fingerprint '{current_fp}'. "
                        f"Specify overwrite=True to re-extract features with updated settings."
                    )

            table = pq.read_table(parquet_path, columns=["uuid"])
            return set(table["uuid"].to_pylist())
        except ConfigurationMismatchError:
            raise
        except Exception as e:
            logger.warning(f"Could not read existing parquet table at {parquet_path}: {e}")
            return set()

    def process_manifest(
        self,
        manifest_path: Path | str,
        partition_name: str,
        max_records: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        overwrite: bool = False,
    ) -> ExtractionSummary:
        """Execute feature extraction on a validated manifest.

        Parameters
        ----------
        manifest_path : Path or str
            Path to approved split manifest (e.g. data/splits/train.csv).
        partition_name : str
            Identifier for output files (e.g. "train", "test", "pilot_train").
        max_records : int, optional
            Limit number of input records (e.g. for small pilot verification).
        batch_size : int, optional
            Number of records per processing and persistence chunk.
        num_workers : int, optional
            Worker concurrency count. Defaults to config.features.num_workers (4).
        overwrite : bool, default=False
            If True, ignore existing caches and restart from scratch.

        Returns
        -------
        ExtractionSummary
            Complete metrics of processed, successful, and excluded records.
        """
        manifest_p = Path(manifest_path).resolve()
        batch_sz = batch_size or self.config.features.batch_size
        workers = num_workers if num_workers is not None else getattr(self.config.features, "num_workers", 4)

        parquet_path = self.data_dir / f"{partition_name}_features.parquet"
        seg_parquet_path = self.data_dir / f"{partition_name}_segments.parquet"
        spec_path = self.data_dir / f"{partition_name}_spectrograms.npz"
        error_json_path = self.reports_dir / f"{partition_name}_extraction_errors.json"
        error_csv_path = self.reports_dir / f"{partition_name}_extraction_errors.csv"
        summary_json_path = self.reports_dir / f"{partition_name}_extraction_summary.json"

        # Read input records
        records = self._read_manifest(manifest_p)
        total_input_count = len(records)
        if max_records is not None and max_records > 0:
            records = records[:max_records]

        # Check existing cached UUIDs and verify configuration compatibility
        existing_uuids = set() if overwrite else self._check_existing_cache_compatibility(parquet_path)
        if existing_uuids:
            logger.info(f"Resuming '{partition_name}': found {len(existing_uuids)} already-extracted UUIDs.")

        to_process = [r for r in records if r.get("uuid") not in existing_uuids]

        # Load existing spectrograms if resuming
        spectrogram_store: Dict[str, np.ndarray] = {}
        if not overwrite and spec_path.exists():
            try:
                with np.load(spec_path) as loaded:
                    for k in loaded.files:
                        spectrogram_store[k] = loaded[k]
            except Exception as e:
                logger.warning(f"Could not load existing spectrograms from {spec_path}: {e}")

        # Load existing errors if resuming
        existing_errors: List[Dict[str, Any]] = []
        if not overwrite and error_json_path.exists():
            try:
                with open(error_json_path, "r", encoding="utf-8") as f:
                    existing_errors = json.load(f)
            except Exception:
                existing_errors = []

        all_errors: List[ExtractionErrorRecord] = [
            ExtractionErrorRecord(**e) for e in existing_errors
        ]
        error_uuids = {e.uuid for e in all_errors}

        # Filter out records that were already excluded in previous runs
        if not overwrite:
            to_process = [r for r in to_process if r.get("uuid") not in error_uuids]

        # Process in chunks of batch_sz
        total_to_process = len(to_process)
        for chunk_start in range(0, total_to_process, batch_sz):
            chunk = to_process[chunk_start : chunk_start + batch_sz]

            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results = list(
                        executor.map(
                            lambda r: extract_recording_features(r, self.raw_data_dir, self.config),
                            chunk,
                        )
                    )
            else:
                results = [
                    extract_recording_features(r, self.raw_data_dir, self.config)
                    for r in chunk
                ]

            new_recording_rows: List[Dict[str, Any]] = []
            new_segment_rows: List[Dict[str, Any]] = []

            for rec_row, seg_rows, spec_tensor, err in results:
                if err is not None:
                    all_errors.append(err)
                else:
                    if rec_row is not None:
                        new_recording_rows.append(rec_row)
                    if seg_rows:
                        new_segment_rows.extend(seg_rows)
                    if spec_tensor is not None:
                        spectrogram_store[rec_row["uuid"]] = spec_tensor

            # Persist batch checkpoint
            is_initial = overwrite and (chunk_start == 0)
            if new_recording_rows:
                self._persist_parquet_batch(new_recording_rows, parquet_path, overwrite=is_initial)
            if new_segment_rows:
                self._persist_parquet_batch(new_segment_rows, seg_parquet_path, overwrite=is_initial)
            if spectrogram_store:
                np.savez_compressed(spec_path, **spectrogram_store)

            processed_so_far = chunk_start + len(chunk)
            print(
                f"[{partition_name}] {processed_so_far:,}/{total_to_process:,} records "
                f"({processed_so_far / total_to_process * 100:.1f}%) | "
                f"Successes: {len(spectrogram_store):,} | Exclusions: {len(all_errors):,}",
                flush=True,
            )

        # Write final errors
        error_dicts = [e.to_dict() for e in all_errors]
        with open(error_json_path, "w", encoding="utf-8") as f:
            json.dump(error_dicts, f, indent=2)

        if error_dicts:
            err_df = pd.DataFrame(error_dicts)
            err_df.to_csv(error_csv_path, index=False)
        else:
            # Create empty csv with headers
            pd.DataFrame(columns=["uuid", "audio_path", "label", "status", "error_stage", "error_type", "error_message"]).to_csv(
                error_csv_path, index=False
            )

        # Build comprehensive summary
        summary = self._compute_summary(
            partition_name=partition_name,
            manifest_path=str(manifest_p),
            total_manifest_records=total_input_count,
            processed_records=len(records),
            parquet_path=parquet_path,
            spectrogram_path=spec_path,
            error_manifest_path=str(error_json_path),
            errors=all_errors,
            input_records=records,
        )

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        return summary

    def _persist_parquet_batch(
        self,
        batch_rows: List[Dict[str, Any]],
        parquet_path: Path,
        overwrite: bool,
    ) -> None:
        """Write a batch of rows to Parquet using PyArrow, enforcing authoritative schema."""
        new_df = pd.DataFrame(batch_rows)
        # Ensure column ordering strictly matches RECORDING_FEATURE_COLUMNS if recording table
        if set(RECORDING_FEATURE_COLUMNS).issubset(set(new_df.columns)):
            new_df = new_df[RECORDING_FEATURE_COLUMNS]

        current_fp = compute_config_fingerprint(self.config)
        custom_meta = {
            "config_fingerprint": current_fp,
            "schema_version": SCHEMA_VERSION,
        }

        if not overwrite and parquet_path.exists():
            try:
                old_table = pq.read_table(parquet_path)
                old_df = old_table.to_pandas()
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                dedup_cols = ["uuid", "segment_index"] if "segment_index" in combined_df.columns else ["uuid"]
                dedup_df = combined_df.drop_duplicates(subset=dedup_cols, keep="last")
                if set(RECORDING_FEATURE_COLUMNS).issubset(set(dedup_df.columns)):
                    dedup_df = dedup_df[RECORDING_FEATURE_COLUMNS]
                final_df = dedup_df
            except Exception as e:
                logger.warning(f"Failed to merge existing table ({e}); overwriting with current batch.")
                final_df = new_df
        else:
            final_df = new_df

        final_table = pa.Table.from_pandas(final_df, preserve_index=False)
        # Attach schema metadata
        existing_meta = final_table.schema.metadata or {}
        new_meta = {**existing_meta, **{k.encode("utf-8"): v.encode("utf-8") for k, v in custom_meta.items()}}
        final_table = final_table.replace_schema_metadata(new_meta)

        # Atomic write via temporary file
        tmp_path = parquet_path.with_suffix(".tmp")
        pq.write_table(final_table, tmp_path, compression="snappy")
        tmp_path.replace(parquet_path)

        # Also write sidecar JSON metadata
        sidecar_path = parquet_path.with_suffix(".meta.json")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump({
                "config_fingerprint": current_fp,
                "schema_version": SCHEMA_VERSION,
                "num_rows": len(final_df),
                "num_columns": len(final_df.columns),
            }, f, indent=2)

    def _compute_summary(
        self,
        partition_name: str,
        manifest_path: str,
        total_manifest_records: int,
        processed_records: int,
        parquet_path: Path,
        spectrogram_path: Path,
        error_manifest_path: str,
        errors: List[ExtractionErrorRecord],
        input_records: List[Dict[str, Any]],
    ) -> ExtractionSummary:
        """Compute final verification metrics and class balance distributions."""
        current_fp = compute_config_fingerprint(self.config)

        # Class distribution of input (explicitly initialized for both classes)
        input_classes: Dict[str, int] = {"0": 0, "1": 0}
        for r in input_records:
            lbl = str(r.get("label", "unknown"))
            input_classes[lbl] = input_classes.get(lbl, 0) + 1

        # Read successful parquet table
        has_nan = False
        num_acoustic_features = len(ACOUSTIC_FEATURE_COLUMNS)  # 208
        total_cols = len(RECORDING_FEATURE_COLUMNS)  # 217
        success_classes: Dict[str, int] = {"0": 0, "1": 0}
        success_count = 0

        if parquet_path.exists():
            try:
                tab = pq.read_table(parquet_path)
                df = tab.to_pandas()
                success_count = len(df)
                total_cols = len(df.columns)

                # Check for any NaNs or Infs in acoustic features (excluding optional demographic metadata like age)
                meta_cols = set(METADATA_COLUMNS)
                acoustic_cols = [c for c in df.columns if c not in meta_cols]
                num_acoustic_features = len(acoustic_cols)
                has_nan = bool(
                    df[acoustic_cols].isna().any().any()
                    or np.isinf(df[acoustic_cols].to_numpy()).any()
                )

                if "label" in df.columns:
                    for lbl, count in df["label"].value_counts().items():
                        success_classes[str(lbl)] = int(count)
            except Exception as e:
                logger.error(f"Error inspecting output parquet table: {e}")

        # Failure breakdown
        failure_reasons: Dict[str, int] = {}
        excluded_classes: Dict[str, int] = {"0": 0, "1": 0}
        for err in errors:
            failure_reasons[err.error_type] = failure_reasons.get(err.error_type, 0) + 1
            lbl_key = str(err.label if err.label is not None else "unknown")
            excluded_classes[lbl_key] = excluded_classes.get(lbl_key, 0) + 1

        # Success rates by class
        success_rates: Dict[str, float] = {}
        for cls_name, in_count in input_classes.items():
            succ = success_classes.get(cls_name, 0)
            success_rates[cls_name] = round(float(succ) / float(in_count), 4) if in_count > 0 else 0.0

        spec_shape = [self.config.features.n_mels, self.config.features.spectrogram_frames]

        return ExtractionSummary(
            partition=partition_name,
            manifest_path=manifest_path,
            total_manifest_records=total_manifest_records,
            processed_records=processed_records,
            success_count=success_count,
            exclusion_count=len(errors),
            failure_reasons=failure_reasons,
            class_distribution_input=input_classes,
            class_distribution_success=success_classes,
            class_distribution_excluded=excluded_classes,
            class_success_rates=success_rates,
            num_tabular_features=num_acoustic_features,
            total_columns=total_cols,
            spectrogram_shape=spec_shape,
            parquet_path=str(parquet_path),
            spectrogram_path=str(spectrogram_path),
            error_manifest_path=error_manifest_path,
            nan_or_inf_detected=has_nan,
            schema_version=SCHEMA_VERSION,
            config_fingerprint=current_fp,
        )
