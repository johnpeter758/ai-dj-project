from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import math

import librosa
import numpy as np
import soundfile as sf

from .manifest import ResolvedRenderPlan
from .transitions import equal_power_fade_in, equal_power_fade_out


@dataclass(slots=True)
class RenderResult:
    manifest_path: str
    raw_wav_path: str
    master_wav_path: str
    master_mp3_path: str | None


def _load_stereo(path: str, sr: int) -> tuple[np.ndarray, int]:
    audio, out_sr = librosa.load(path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.vstack([audio, audio])
    return audio.astype(np.float32), int(out_sr)


def _require_finite_nonnegative(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def _validate_manifest(manifest: ResolvedRenderPlan) -> None:
    if manifest.sample_rate <= 0:
        raise ValueError("manifest sample_rate must be positive")
    if len(manifest.sections) != len(manifest.work_orders):
        raise ValueError("manifest sections/work_orders length mismatch")

    previous_start_sec = -1.0
    for idx, section in enumerate(manifest.sections):
        start_sec = _require_finite_nonnegative(section.target.start_sec, f"section[{idx}].target.start_sec")
        end_sec = _require_finite_nonnegative(section.target.end_sec, f"section[{idx}].target.end_sec")
        duration_sec = _require_finite_nonnegative(section.target.duration_sec, f"section[{idx}].target.duration_sec")
        if end_sec < start_sec:
            raise ValueError(f"section[{idx}] target end precedes start")
        if not math.isclose(end_sec - start_sec, duration_sec, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"section[{idx}] target duration does not match start/end timing")
        if start_sec < previous_start_sec:
            raise ValueError("manifest sections must be sorted by target start time")
        previous_start_sec = start_sec

    seen_order_ids: set[str] = set()
    for idx, work in enumerate(manifest.work_orders):
        if work.order_id in seen_order_ids:
            raise ValueError(f"duplicate work order id: {work.order_id}")
        seen_order_ids.add(work.order_id)
        _require_finite_nonnegative(work.source_start_sec, f"work_orders[{idx}].source_start_sec")
        _require_finite_nonnegative(work.source_end_sec, f"work_orders[{idx}].source_end_sec")
        _require_finite_nonnegative(work.target_start_sec, f"work_orders[{idx}].target_start_sec")
        _require_finite_nonnegative(work.target_duration_sec, f"work_orders[{idx}].target_duration_sec")
        _require_finite_nonnegative(work.fade_in_sec, f"work_orders[{idx}].fade_in_sec")
        _require_finite_nonnegative(work.fade_out_sec, f"work_orders[{idx}].fade_out_sec")
        if work.source_end_sec <= work.source_start_sec:
            raise ValueError(f"work_orders[{idx}] source window must have positive duration")
        if work.stretch_ratio <= 0.0 or not math.isfinite(work.stretch_ratio):
            raise ValueError(f"work_orders[{idx}] stretch_ratio must be finite and positive")


def _extract(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start = max(0, int(round(start_sec * sr)))
    end = max(start + 1, int(round(end_sec * sr)))
    end = min(end, audio.shape[1])
    return audio[:, start:end]


def _fit_to_duration(segment: np.ndarray, sr: int, target_seconds: float, stretch_ratio: float) -> np.ndarray:
    target_samples = max(1, int(round(target_seconds * sr)))
    if segment.shape[1] == 0:
        return np.zeros((2, target_samples), dtype=np.float32)
    if segment.shape[1] == target_samples:
        return segment.astype(np.float32)

    rate = max(float(stretch_ratio), 1e-6)
    stretched_channels = []
    for ch in range(segment.shape[0]):
        y = librosa.effects.time_stretch(segment[ch], rate=rate)
        stretched_channels.append(y)
    out = np.vstack(stretched_channels)
    if out.shape[1] > target_samples:
        out = out[:, :target_samples]
    elif out.shape[1] < target_samples:
        out = np.pad(out, ((0, 0), (0, target_samples - out.shape[1])))
    return out.astype(np.float32)


def _apply_edge_fades(segment: np.ndarray, sr: int, fade_in_sec: float, fade_out_sec: float) -> np.ndarray:
    out = segment.copy()
    n = out.shape[1]
    fi = min(n, max(0, int(round(fade_in_sec * sr))))
    fo = min(n, max(0, int(round(fade_out_sec * sr))))
    if fi > 0:
        env = equal_power_fade_in(fi)
        out[:, :fi] *= env
    if fo > 0:
        env = equal_power_fade_out(fo)
        out[:, -fo:] *= env
    return out


def _apply_gain_db(segment: np.ndarray, gain_db: float) -> np.ndarray:
    if not np.isfinite(gain_db) or gain_db == 0.0:
        return segment.astype(np.float32)
    return (segment * np.float32(10 ** (gain_db / 20.0))).astype(np.float32)


def _peak_normalize(audio: np.ndarray, target_peak_dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        return audio.astype(np.float32)
    target_linear = 10 ** (target_peak_dbfs / 20.0)
    return (audio * (target_linear / peak)).astype(np.float32)


def _write_manifest(manifest: ResolvedRenderPlan, path: str | Path, raw_wav: str, master_wav: str, master_mp3: str | None) -> str:
    payload = manifest.to_dict()
    payload["outputs"] = {
        "raw_wav": raw_wav,
        "master_wav": master_wav,
        "master_mp3": master_mp3,
    }
    out = Path(path)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(out)


def render_resolved_plan(manifest: ResolvedRenderPlan, output_dir: str | Path) -> RenderResult:
    _validate_manifest(manifest)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sr = manifest.sample_rate
    total_duration = max((section.target.end_sec for section in manifest.sections), default=0.0)
    total_samples = max(1, int(round(total_duration * sr)))
    master = np.zeros((2, total_samples), dtype=np.float32)

    for work in sorted(manifest.work_orders, key=lambda item: (item.target_start_sec, item.order_id)):
        audio, _ = _load_stereo(work.source_path, sr)
        segment = _extract(audio, sr, work.source_start_sec, work.source_end_sec)
        segment = _fit_to_duration(segment, sr, work.target_duration_sec, work.stretch_ratio)
        segment = _apply_gain_db(segment, work.gain_db)
        segment = _apply_edge_fades(segment, sr, work.fade_in_sec, work.fade_out_sec)
        start_sample = int(round(work.target_start_sec * sr))
        end_sample = min(total_samples, start_sample + segment.shape[1])
        seg = segment[:, : max(0, end_sample - start_sample)]
        if seg.shape[1] > 0:
            master[:, start_sample:end_sample] += seg

    raw_wav = str((outdir / "child_raw.wav").resolve())
    master_wav = str((outdir / "child_master.wav").resolve())
    master_mp3 = str((outdir / "child_master.mp3").resolve())

    sf.write(raw_wav, master.T, sr, subtype="FLOAT")
    final_audio = _peak_normalize(master, -1.0)
    sf.write(master_wav, final_audio.T, sr, subtype="PCM_24")

    mp3_ok = False
    try:
        proc = subprocess.run([
            "ffmpeg", "-hide_banner", "-y", "-i", master_wav,
            "-codec:a", "libmp3lame", "-b:a", "320k", master_mp3,
        ], check=False, capture_output=True, text=True)
        mp3_ok = proc.returncode == 0
    except FileNotFoundError:
        mp3_ok = False

    manifest_path = _write_manifest(manifest, outdir / "render_manifest.json", raw_wav, master_wav, master_mp3 if mp3_ok else None)
    return RenderResult(
        manifest_path=manifest_path,
        raw_wav_path=raw_wav,
        master_wav_path=master_wav,
        master_mp3_path=master_mp3 if mp3_ok else None,
    )
