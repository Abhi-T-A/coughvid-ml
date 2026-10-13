"""Validation of audio loading, DSP, and hysteresis burst segmentation on real dataset recordings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path when script is run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coughvid.config import load_config
from coughvid.audio.io import load_audio, AudioError, DecoderUnavailableError
from coughvid.audio.dsp import (
    butterworth_lowpass_filter,
    normalize_audio,
    resample_audio,
)
from coughvid.audio.segmenter import segment_cough_bursts, SegmentationResult


def render_diagnostic_figure(
    original_audio,
    processed_y: np.ndarray,
    target_sr: int,
    seg_result: SegmentationResult,
    output_path: Path,
) -> None:
    """Render a clean 3-panel visual diagnostic image using Pillow."""
    width = 1200
    height = 800
    bg_color = (255, 255, 255)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(0, 0), (width, 55)], fill=(24, 32, 44))
    draw.text(
        (25, 18),
        f"COUGHVID Acoustic Burst Segmentation Diagnostics - {original_audio.source_path.name}",
        fill=(255, 255, 255),
    )

    # Panel 1: Original Waveform
    # Rect: (80, 80) to (1120, 260)
    p1_box = (80, 80, 1120, 260)
    draw.rectangle(p1_box, outline=(200, 205, 215), fill=(248, 250, 252))
    draw.text((85, 62), "1. Normalized Audio Waveform (Time Domain)", fill=(30, 41, 59))

    # Center horizontal line
    mid_y1 = (p1_box[1] + p1_box[3]) // 2
    draw.line([(p1_box[0], mid_y1), (p1_box[2], mid_y1)], fill=(210, 215, 225))

    # Plot waveform points
    n_points = p1_box[2] - p1_box[0]
    step = len(processed_y) / n_points
    half_h1 = (p1_box[3] - p1_box[1]) / 2.0 - 5

    points1 = []
    for i in range(n_points):
        x = p1_box[0] + i
        sample_idx = int(i * step)
        if sample_idx < len(processed_y):
            val = float(processed_y[sample_idx])
            y_pixel = mid_y1 - int(val * half_h1)
            points1.append((x, y_pixel))

    for i in range(len(points1) - 1):
        draw.line([points1[i], points1[i + 1]], fill=(37, 99, 235), width=1)

    # Highlight detected burst regions on waveform
    total_time = len(processed_y) / target_sr
    for seg in seg_result.segments:
        x_start = p1_box[0] + int((seg.start_sec / total_time) * n_points)
        x_end = p1_box[0] + int((seg.end_sec / total_time) * n_points)
        x_start = max(p1_box[0], min(p1_box[2], x_start))
        x_end = max(p1_box[0], min(p1_box[2], x_end))
        # Draw translucent or framed box for burst
        draw.rectangle([x_start, p1_box[1], x_end, p1_box[3]], outline=(34, 197, 94), width=2)

    # Panel 2: Energy Envelope & Hysteresis Thresholds
    # Rect: (80, 310) to (1120, 490)
    p2_box = (80, 310, 1120, 490)
    draw.rectangle(p2_box, outline=(200, 205, 215), fill=(248, 250, 252))
    draw.text((85, 292), "2. Short-Time RMS Energy Envelope & Hysteresis Thresholds", fill=(30, 41, 59))

    env = seg_result.envelope
    max_env = max(float(np.max(env)), 1e-4)
    h2 = p2_box[3] - p2_box[1] - 10

    # Draw onset & offset threshold lines
    onset_y = p2_box[3] - 5 - int((seg_result.onset_threshold / max_env) * h2)
    offset_y = p2_box[3] - 5 - int((seg_result.offset_threshold / max_env) * h2)

    draw.line([(p2_box[0], onset_y), (p2_box[2], onset_y)], fill=(22, 163, 74), width=2)
    draw.text((p2_box[0] + 10, max(p2_box[1], onset_y - 14)), f"Onset Threshold ({seg_result.onset_threshold:.3f})", fill=(22, 163, 74))

    draw.line([(p2_box[0], offset_y), (p2_box[2], offset_y)], fill=(220, 38, 38), width=1)
    draw.text((p2_box[0] + 10, max(p2_box[1], offset_y + 2)), f"Offset Threshold ({seg_result.offset_threshold:.3f})", fill=(220, 38, 38))

    # Plot envelope
    if len(env) > 1:
        n_env = len(env)
        points2 = []
        for i in range(n_points):
            x = p2_box[0] + i
            env_idx = int((i / n_points) * (n_env - 1))
            val = float(env[env_idx])
            y_pixel = p2_box[3] - 5 - int((val / max_env) * h2)
            points2.append((x, y_pixel))

        for i in range(len(points2) - 1):
            draw.line([points2[i], points2[i + 1]], fill=(234, 88, 12), width=2)

    # Highlight segments on envelope panel
    for seg in seg_result.segments:
        x_start = p2_box[0] + int((seg.start_sec / total_time) * n_points)
        x_end = p2_box[0] + int((seg.end_sec / total_time) * n_points)
        x_start = max(p2_box[0], min(p2_box[2], x_start))
        x_end = max(p2_box[0], min(p2_box[2], x_end))
        draw.rectangle([x_start, p2_box[1], x_end, p2_box[3]], outline=(34, 197, 94), width=2)

    # Panel 3: Extracted Concatenated Burst Waveform
    # Rect: (80, 540) to (1120, 680)
    p3_box = (80, 540, 1120, 680)
    draw.rectangle(p3_box, outline=(200, 205, 215), fill=(248, 250, 252))
    concat_audio = seg_result.concatenated_waveform
    draw.text(
        (85, 522),
        f"3. Extracted Candidate Bursts Only (Concatenated, Silence Stripped: {len(concat_audio)/target_sr:.2f}s total)",
        fill=(30, 41, 59),
    )

    mid_y3 = (p3_box[1] + p3_box[3]) // 2
    draw.line([(p3_box[0], mid_y3), (p3_box[2], mid_y3)], fill=(210, 215, 225))

    if len(concat_audio) > 0:
        half_h3 = (p3_box[3] - p3_box[1]) / 2.0 - 5
        points3 = []
        step3 = len(concat_audio) / n_points
        for i in range(n_points):
            x = p3_box[0] + i
            sample_idx = int(i * step3)
            if sample_idx < len(concat_audio):
                val = float(concat_audio[sample_idx])
                y_pixel = mid_y3 - int(val * half_h3)
                points3.append((x, y_pixel))

        for i in range(len(points3) - 1):
            draw.line([points3[i], points3[i + 1]], fill=(16, 185, 129), width=1)
    else:
        draw.text((p3_box[0] + 20, mid_y3 - 8), "No bursts detected - signal contains silence or continuous background noise only.", fill=(156, 163, 175))

    # Bottom Metadata Box
    meta_box = (80, 700, 1120, 775)
    draw.rectangle(meta_box, fill=(241, 245, 249), outline=(203, 213, 225))
    seg_summary = ", ".join(f"[{s.start_sec:.2f}s - {s.end_sec:.2f}s]" for s in seg_result.segments) or "None"
    meta_text = (
        f"Format: {original_audio.format.upper()} | Original SR: {original_audio.sample_rate} Hz | "
        f"Target SR: {target_sr} Hz | Lowpass Cutoff: 5,500 Hz | Original Length: {original_audio.duration_sec:.2f}s\n"
        f"Candidate Bursts Found: {seg_result.num_segments} | Total Burst Audio: {seg_result.total_duration_sec:.2f}s "
        f"({seg_result.total_duration_sec/max(original_audio.duration_sec, 0.01)*100:.1f}% of total)\n"
        f"Detected Burst Windows: {seg_summary}"
    )
    draw.text((meta_box[0] + 15, meta_box[1] + 12), meta_text, fill=(51, 65, 85))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="PNG")
    print(f"  [SAVED] Diagnostic figure saved to: {output_path}")


def run_real_recording_validation() -> dict:
    """Run real recording validation on diverse files from the dataset."""
    cfg = load_config()
    raw_dir = cfg.resolve_raw_data_dir()
    print("=" * 70)
    print("COUGHVID ML - Milestone 2: Real Recording Audio & Segmentation Audit")
    print("=" * 70)
    print(f"Raw Dataset Path: {raw_dir}\n")

    # Representative sample files across available extensions
    test_candidates = [
        # Real PCM WAV files
        ("0038583a-93d8-4520-bf7e-0701c840ef7b.wav", "PCM RIFF WAV (16 kHz mono)"),
        ("00422dad-066b-402f-9cac-e9efe51442af.wav", "PCM RIFF WAV (48 kHz mono)"),
        ("0050ddaf-2cf9-414a-8c51-9863dd0550f7.wav", "PCM RIFF WAV (48 kHz mono)"),
        # Compressed WebM Opus file
        ("00014dcc-0f06-4c27-8c7b-737b18a2cf4c.webm", "Opus in WebM container (48 kHz)"),
        # Compressed Ogg Opus file
        ("005887c9-4bb1-4f13-86b2-1c7b3cee0881.ogg", "Opus in OGG container (48 kHz)"),
        # Misnamed non-RIFF MP4/AAC file with .wav extension
        ("00098cdb-4da1-4aa7-825a-4f1b9abc214b.wav", "AAC/MP4 container misnamed as .wav"),
    ]

    results = {}
    reports_fig_dir = Path(cfg.paths.reports_dir) / "figures"
    diagnostic_saved = False

    for filename, description in test_candidates:
        file_path = raw_dir / filename
        print(f"Testing [{filename}] ({description}):")
        if not file_path.exists():
            print(f"  [SKIP] File does not exist on disk: {file_path}")
            continue

        try:
            audio = load_audio(file_path)
            print(f"  [DECODED] Format: {audio.format}, SR: {audio.sample_rate} Hz, "
                  f"Samples: {audio.num_samples:,}, Duration: {audio.duration_sec:.2f}s")

            # Apply DSP pipeline
            target_sr = cfg.audio.target_sr
            # 1. Resample to 12 kHz
            y_resampled, sr_resampled = resample_audio(audio.waveform, audio.sample_rate, target_sr)
            # 2. Normalize
            y_norm = normalize_audio(y_resampled, method="peak")
            # 3. Butterworth filter (use 5500 Hz cutoff strictly below 6000 Hz Nyquist)
            cutoff = min(cfg.audio.lowpass_hz, target_sr * 0.45)
            y_filtered = butterworth_lowpass_filter(y_norm, sr_resampled, cutoff_hz=cutoff)

            # 4. Energy-envelope hysteresis burst segmentation
            seg_result = segment_cough_bursts(y_filtered, sr_resampled, config=cfg.segmentation)
            print(f"  [SEGMENTED] Candidates: {seg_result.has_candidates}, "
                  f"Bursts Found: {seg_result.num_segments}, "
                  f"Total Burst Duration: {seg_result.total_duration_sec:.2f}s")
            for i, seg in enumerate(seg_result.segments, 1):
                print(f"    Burst #{i}: [{seg.start_sec:.2f}s - {seg.end_sec:.2f}s] "
                      f"({seg.duration_sec:.2f}s) | Peak Energy: {seg.peak_energy:.4f}")

            # Generate visual diagnostic plot for the first successfully decoded file
            if not diagnostic_saved and seg_result.has_candidates:
                fig_path = reports_fig_dir / "segmentation_diagnostics.png"
                render_diagnostic_figure(audio, y_filtered, sr_resampled, seg_result, fig_path)
                diagnostic_saved = True

            results[filename] = {
                "status": "SUCCESS",
                "format": audio.format,
                "sr": audio.sample_rate,
                "duration": audio.duration_sec,
                "segments": seg_result.num_segments,
                "burst_duration": seg_result.total_duration_sec,
            }

        except DecoderUnavailableError as exc:
            first_line = str(exc).split("\n")[0]
            print(f"  [DECODER UNAVAILABLE] {first_line}")
            results[filename] = {"status": "DECODER_UNAVAILABLE", "error": first_line}

        except AudioError as exc:
            print(f"  [AUDIO ERROR] {exc}")
            results[filename] = {"status": "AUDIO_ERROR", "error": str(exc)}

        except Exception as exc:
            print(f"  [UNEXPECTED ERROR] {exc}")
            results[filename] = {"status": "ERROR", "error": str(exc)}

        print()

    return results


if __name__ == "__main__":
    run_real_recording_validation()
