from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import math

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from .manifest import ResolvedRenderPlan
from .mastering import bpm_synced_glue_compress, lookahead_envelope_limit, lufs_normalize
from .spectral import apply_spectral_carve, compute_vocal_presence_mask
from .transitions import equal_power_fade_in, equal_power_fade_out


def _normalize_transition_mode_token(transition_mode: str | None) -> str | None:
    token = str(transition_mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    return token or None


def _one_pole_lowpass(audio: np.ndarray, cutoff_hz: np.ndarray, sr: int) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    out = np.zeros_like(audio, dtype=np.float32)
    dt = 1.0 / float(sr)
    rc = 1.0 / (2.0 * np.pi * np.maximum(cutoff_hz.astype(np.float32), 20.0))
    alpha = (dt / (rc + dt)).astype(np.float32)
    out[:, 0] = audio[:, 0]
    for idx in range(1, audio.shape[1]):
        a = alpha[idx]
        out[:, idx] = out[:, idx - 1] + a * (audio[:, idx] - out[:, idx - 1])
    return out


def _one_pole_highpass(audio: np.ndarray, cutoff_hz: np.ndarray, sr: int) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    out = np.zeros_like(audio, dtype=np.float32)
    dt = 1.0 / float(sr)
    rc = 1.0 / (2.0 * np.pi * np.maximum(cutoff_hz.astype(np.float32), 20.0))
    alpha = (rc / (rc + dt)).astype(np.float32)
    out[:, 0] = audio[:, 0]
    for idx in range(1, audio.shape[1]):
        a = alpha[idx]
        out[:, idx] = a * (out[:, idx - 1] + audio[:, idx] - audio[:, idx - 1])
    return out


def _transition_filter_profile(
    transition_type: str | None,
    transition_mode: str | None,
) -> tuple[float, float, float, float, float, float]:
    transition_mode = _normalize_transition_mode_token(transition_mode)
    incoming_start = 0.0
    incoming_end = 0.0
    outgoing_lowpass_start = 20000.0
    outgoing_lowpass_end = 20000.0
    outgoing_highpass_start = 0.0
    outgoing_highpass_end = 0.0

    if transition_type in {"blend", "lift"}:
        incoming_start, incoming_end = 1800.0, 40.0
        outgoing_lowpass_start, outgoing_lowpass_end = 18000.0, 7000.0
        outgoing_highpass_start, outgoing_highpass_end = 35.0, 120.0
    elif transition_type == "swap":
        incoming_start, incoming_end = 2600.0, 60.0
        outgoing_lowpass_start, outgoing_lowpass_end = 16000.0, 5000.0
        outgoing_highpass_start, outgoing_highpass_end = 40.0, 160.0
    elif transition_type == "drop":
        incoming_start, incoming_end = 3200.0, 80.0
        outgoing_lowpass_start, outgoing_lowpass_end = 14000.0, 4200.0
        outgoing_highpass_start, outgoing_highpass_end = 55.0, 220.0

    if transition_mode == "same_parent_flow":
        incoming_start *= 0.18
        incoming_end = min(incoming_end, 20.0)
        outgoing_lowpass_start = 20000.0
        outgoing_lowpass_end = 20000.0
        outgoing_highpass_start *= 0.35
        outgoing_highpass_end = min(max(outgoing_highpass_end * 0.35, 45.0), 85.0)
    elif transition_mode == "backbone_flow":
        incoming_start *= 0.12
        incoming_end = min(incoming_end, 16.0)
        outgoing_lowpass_start = 20000.0
        outgoing_lowpass_end = 20000.0
        outgoing_highpass_start *= 0.45
        outgoing_highpass_end = min(max(outgoing_highpass_end * 0.45, 60.0), 95.0)
    elif transition_mode in {"arrival_handoff", "single_owner_handoff"}:
        incoming_start *= 1.15
        outgoing_lowpass_end *= 0.6
        outgoing_highpass_start *= 1.15
        outgoing_highpass_end *= 1.35

    return (
        incoming_start,
        incoming_end,
        outgoing_lowpass_start,
        outgoing_lowpass_end,
        outgoing_highpass_start,
        outgoing_highpass_end,
    )


def _apply_transition_sonics(
    segment: np.ndarray,
    sr: int,
    fade_in_sec: float,
    fade_out_sec: float,
    transition_type: str | None,
    transition_mode: str | None,
) -> np.ndarray:
    out = segment.astype(np.float32, copy=True)
    if out.size == 0:
        return out

    (
        incoming_start,
        incoming_end,
        outgoing_lowpass_start,
        outgoing_lowpass_end,
        outgoing_highpass_start,
        outgoing_highpass_end,
    ) = _transition_filter_profile(transition_type, transition_mode)

    fi = min(out.shape[1], max(0, int(round(fade_in_sec * sr))))
    if fi > 8 and incoming_start > incoming_end > 0.0:
        cutoff = np.linspace(incoming_start, incoming_end, fi, endpoint=True, dtype=np.float32)
        out[:, :fi] = _one_pole_highpass(out[:, :fi], cutoff, sr)

    fo = min(out.shape[1], max(0, int(round(fade_out_sec * sr))))
    if fo > 8 and outgoing_highpass_end > outgoing_highpass_start > 0.0:
        cutoff = np.linspace(outgoing_highpass_start, outgoing_highpass_end, fo, endpoint=True, dtype=np.float32)
        out[:, -fo:] = _one_pole_highpass(out[:, -fo:], cutoff, sr)
    if fo > 8 and outgoing_lowpass_start > outgoing_lowpass_end > 0.0 and outgoing_lowpass_end < 19999.0:
        cutoff = np.linspace(outgoing_lowpass_start, outgoing_lowpass_end, fo, endpoint=True, dtype=np.float32)
        out[:, -fo:] = _one_pole_lowpass(out[:, -fo:], cutoff, sr)

    return out


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
    if not manifest.sections:
        raise ValueError("manifest must contain at least one section")
    if not manifest.work_orders:
        raise ValueError("manifest must contain at least one work order")

    previous_start_sec = -1.0
    section_indices: set[int] = set()
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
        section_indices.add(int(section.index))

    seen_order_ids: set[str] = set()
    section_order_counts: dict[int, int] = {index: 0 for index in section_indices}
    section_base_counts: dict[int, int] = {index: 0 for index in section_indices}
    for idx, work in enumerate(manifest.work_orders):
        if work.order_id in seen_order_ids:
            raise ValueError(f"duplicate work order id: {work.order_id}")
        seen_order_ids.add(work.order_id)
        if int(work.section_index) not in section_indices:
            raise ValueError(f"work_orders[{idx}] references unknown section index {work.section_index}")
        section_order_counts[int(work.section_index)] += 1
        if work.order_type == 'section_base':
            section_base_counts[int(work.section_index)] += 1
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

    missing_orders = sorted(index for index, count in section_order_counts.items() if count <= 0)
    if missing_orders:
        raise ValueError(f"manifest sections without work orders: {missing_orders}")
    missing_bases = sorted(index for index, count in section_base_counts.items() if count <= 0)
    if missing_bases:
        raise ValueError(f"manifest sections without section_base work orders: {missing_bases}")


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


def _find_cue_safe_head_offset_samples(segment: np.ndarray, sr: int, fade_in_sec: float) -> int:
    if segment.size == 0 or fade_in_sec <= 0.0:
        return 0

    max_shift_samples = min(
        segment.shape[1] // 8,
        max(0, int(round(min(fade_in_sec * 0.75, 0.5) * sr))),
    )
    if max_shift_samples < max(32, sr // 100):
        return 0

    mono = np.mean(segment, axis=0).astype(np.float32)
    guard_samples = min(max_shift_samples, max(0, int(round(0.02 * sr))))
    if guard_samples >= mono.size - 1:
        return 0

    search = mono[: max_shift_samples]
    if search.size < 16:
        return 0

    hop = max(64, min(512, sr // 100))
    frame = min(max(256, hop * 4), search.size)
    rms = librosa.feature.rms(y=search, frame_length=frame, hop_length=hop, center=False)[0]
    onset_env = librosa.onset.onset_strength(y=search, sr=sr, hop_length=hop, center=False)
    if rms.size == 0 or onset_env.size == 0:
        return 0

    usable = min(rms.size, onset_env.size)
    rms = rms[:usable]
    onset_env = onset_env[:usable]
    if usable < 3:
        return 0

    head_rms = float(np.mean(rms[: max(1, usable // 3)]))
    peak_rms = float(np.max(rms))
    peak_onset = float(np.max(onset_env))
    if peak_rms <= 1e-5 or peak_onset <= 1e-5:
        return 0

    energy_gate = max(head_rms * 1.6, peak_rms * 0.22)
    onset_gate = peak_onset * 0.35
    candidate = None
    for idx in range(1, usable):
        sample = idx * hop
        if sample <= guard_samples:
            continue
        if rms[idx] >= energy_gate and onset_env[idx] >= onset_gate:
            candidate = sample
            break

    if candidate is None:
        return 0
    return int(min(max_shift_samples, candidate))


def _cue_safe_transition_anchor(segment: np.ndarray, sr: int, fade_in_sec: float) -> np.ndarray:
    offset = _find_cue_safe_head_offset_samples(segment, sr, fade_in_sec)
    if offset <= 0:
        return segment.astype(np.float32)
    trimmed = segment[:, offset:]
    if trimmed.shape[1] == 0:
        return segment.astype(np.float32)
    return np.pad(trimmed, ((0, 0), (0, offset))).astype(np.float32)


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


def _safe_sosfiltfilt(sos: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if audio.shape[1] < 32:
        return audio.astype(np.float32)
    try:
        return signal.sosfiltfilt(sos, audio, axis=1).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, audio, axis=1).astype(np.float32)


def _highpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    cutoff = max(20.0, min(float(cutoff_hz), sr * 0.45))
    sos = signal.butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return _safe_sosfiltfilt(sos, audio)


def _bandstop(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    low = max(20.0, float(low_hz))
    high = min(float(high_hz), sr * 0.45)
    if high <= low + 10.0:
        return audio.astype(np.float32)
    sos = signal.butter(2, [low, high], btype="bandstop", fs=sr, output="sos")
    return _safe_sosfiltfilt(sos, audio)



def _lowpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    cutoff = min(float(cutoff_hz), sr * 0.45)
    if cutoff <= 40.0:
        return audio.astype(np.float32)
    sos = signal.butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
    return _safe_sosfiltfilt(sos, audio)



def _attenuate_mid(audio: np.ndarray, mid_gain_db: float) -> np.ndarray:
    if audio.shape[0] != 2:
        return _apply_gain_db(audio, mid_gain_db)
    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5
    mid = _apply_gain_db(mid[np.newaxis, :], mid_gain_db)[0]
    left = mid + side
    right = mid - side
    return np.vstack([left, right]).astype(np.float32)



def _prepare_role_layer(segment: np.ndarray, sr: int, work, section) -> np.ndarray:
    out = segment.astype(np.float32)
    role = str(getattr(work, 'role', '') or '')
    vocal_state = str(getattr(work, 'vocal_state', '') or '')
    if role in {'filtered_counterlayer', 'filtered_support'}:
        out = _highpass(out, sr, 150.0)
        out = _lowpass(out, sr, 7000.0)
        out = _attenuate_mid(out, -7.0 if vocal_state == 'none' else -5.5)
        out = _bandstop(out, sr, 280.0, 2200.0)
    elif role == 'foreground_counterlayer':
        out = _highpass(out, sr, 135.0)
        out = _lowpass(out, sr, 8200.0)
        out = _attenuate_mid(out, -3.5)
    if vocal_state == 'support':
        out = _attenuate_mid(out, -4.0)
        out = _highpass(out, sr, 90.0)
    return out.astype(np.float32)


def _apply_support_entry_shape(segment: np.ndarray, sr: int, work, section) -> np.ndarray:
    out = segment.astype(np.float32)
    if str(getattr(work, 'order_type', '') or '') != 'section_support':
        return out

    role = str(getattr(work, 'role', '') or '')
    if role not in {'filtered_support', 'filtered_counterlayer', 'foreground_counterlayer'}:
        return out

    section_label = str(getattr(section, 'label', '') or '').strip().lower()
    transition_mode = _normalize_transition_mode_token(getattr(section, 'transition_mode', None))

    entry_sec = float(min(max(getattr(work, 'fade_in_sec', 0.0), 0.0) * 1.35, 1.2, max(getattr(work, 'target_duration_sec', 0.0) * 0.45, 0.0)))
    entry_samples = min(out.shape[1], max(0, int(round(entry_sec * sr))))
    if entry_samples > 32:
        role_floor_db = -3.0 if role == 'foreground_counterlayer' else -4.5
        if section_label in {'build', 'payoff'}:
            role_floor_db -= 1.0

        floor = np.float32(10 ** (role_floor_db / 20.0))
        ramp = np.linspace(0.0, 1.0, entry_samples, endpoint=True, dtype=np.float32)
        ramp = np.power(ramp, np.float32(1.8))
        env = floor + (1.0 - floor) * ramp
        out[:, :entry_samples] *= env[np.newaxis, :]

        if role != 'foreground_counterlayer':
            hp_start = 240.0 if section_label in {'build', 'payoff'} else 200.0
            hp_end = 110.0
            cutoff = np.linspace(hp_start, hp_end, entry_samples, endpoint=True, dtype=np.float32)
            intro = _one_pole_highpass(out[:, :entry_samples], cutoff, sr)
            if section_label in {'build', 'payoff'} and role in {'filtered_support', 'filtered_counterlayer'}:
                # Early donor-support entry is where lead-vocal masking spikes.
                # Make the vocal-presence notch transition-aware so hard handoffs stay cleaner,
                # while same-parent/backbone transitions retain more support identity.
                notch_low_hz, notch_high_hz = 380.0, 2600.0
                notch_strength = 0.28
                if transition_mode in {'arrival_handoff', 'single_owner_handoff'}:
                    notch_low_hz, notch_high_hz = 340.0, 2900.0
                    notch_strength = 0.36
                elif transition_mode in {'same_parent_flow', 'backbone_flow'}:
                    notch_low_hz, notch_high_hz = 420.0, 2350.0
                    notch_strength = 0.22
                notched = _bandstop(intro, sr, notch_low_hz, notch_high_hz)
                notch_mix = np.linspace(1.0, 0.0, entry_samples, endpoint=True, dtype=np.float32)
                notch_mix = np.power(notch_mix, np.float32(1.4))[np.newaxis, :]
                intro = intro * (1.0 - notch_strength * notch_mix) + notched * (notch_strength * notch_mix)
            out[:, :entry_samples] = intro

    release_scale = 1.35 if section_label in {'build', 'payoff'} else 1.0
    release_sec = float(
        min(
            max(getattr(work, 'fade_out_sec', 0.0), 0.0) * release_scale,
            1.35,
            max(getattr(work, 'target_duration_sec', 0.0) * 0.5, 0.0),
        )
    )
    release_samples = min(out.shape[1], max(0, int(round(release_sec * sr))))
    if release_samples > 32:
        if role == 'foreground_counterlayer':
            tail_floor_db = -2.0 if section_label in {'build', 'payoff'} else -1.0
            lp_start_hz, lp_end_hz = 9000.0, 5600.0
        else:
            tail_floor_db = -4.5 if section_label in {'build', 'payoff'} else -3.0
            lp_start_hz, lp_end_hz = (7600.0, 3800.0) if section_label in {'build', 'payoff'} else (9000.0, 5200.0)

        release_ramp = np.linspace(0.0, 1.0, release_samples, endpoint=True, dtype=np.float32)
        release_ramp = np.power(release_ramp, np.float32(1.25))
        tail_floor = np.float32(10 ** (tail_floor_db / 20.0))
        tail_env = 1.0 - (1.0 - tail_floor) * release_ramp
        out[:, -release_samples:] *= tail_env[np.newaxis, :]

        if role != 'foreground_counterlayer':
            tail = out[:, -release_samples:]
            cutoff = np.linspace(lp_start_hz, lp_end_hz, release_samples, endpoint=True, dtype=np.float32)
            tail = _one_pole_lowpass(tail, cutoff, sr)
            if section_label in {'build', 'payoff'} and role in {'filtered_support', 'filtered_counterlayer'}:
                # Build/payoff overlays are the highest collision risk with incoming lead vox.
                # Keep release notch transition-aware: stronger for handoffs, lighter for flow sections.
                notch_low_hz, notch_high_hz = 320.0, 2100.0
                notch_strength = 1.0
                if transition_mode in {'arrival_handoff', 'single_owner_handoff'}:
                    notch_low_hz, notch_high_hz = 300.0, 2350.0
                    notch_strength = 1.0
                elif transition_mode in {'same_parent_flow', 'backbone_flow'}:
                    notch_low_hz, notch_high_hz = 360.0, 1950.0
                    notch_strength = 0.65

                notched_tail = _bandstop(tail, sr, notch_low_hz, notch_high_hz)
                tail = tail * (1.0 - notch_strength) + notched_tail * notch_strength
            out[:, -release_samples:] = tail

    return out.astype(np.float32)


def _compress_bus(audio: np.ndarray, threshold_db: float = -18.0, ratio: float = 2.0, makeup_db: float = 1.0) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    threshold_lin = 10 ** (threshold_db / 20.0)
    envelope = np.max(np.abs(audio), axis=0)
    over = np.maximum(envelope, threshold_lin) / threshold_lin
    gain = np.ones_like(envelope, dtype=np.float32)
    mask = envelope > threshold_lin
    if np.any(mask):
        compressed = np.power(over[mask], -(1.0 - 1.0 / max(ratio, 1.0)))
        gain[mask] = compressed.astype(np.float32)
    smoothed = np.convolve(gain, np.ones(2048, dtype=np.float32) / 2048.0, mode="same")
    out = audio * smoothed[np.newaxis, :]
    return _apply_gain_db(out, makeup_db)


def _soft_limit(audio: np.ndarray, drive: float = 1.1) -> np.ndarray:
    return np.tanh(audio * np.float32(drive)).astype(np.float32) / np.float32(np.tanh(drive))


def _should_apply_overlap_carve(work, section) -> bool:
    vocal_state = str(getattr(work, 'vocal_state', '') or '')
    role = str(getattr(work, 'role', '') or '')
    if not bool(getattr(section, 'allowed_overlap', False)):
        return False
    if vocal_state in {'lead', 'lead_only', 'support'}:
        return True
    return role in {'foreground_counterlayer', 'filtered_counterlayer', 'filtered_support'}


def _overlap_carve_settings(work, section) -> tuple[float, float, float]:
    vocal_state = str(getattr(work, 'vocal_state', '') or '')
    role = str(getattr(work, 'role', '') or '')
    transition_mode = _normalize_transition_mode_token(getattr(section, 'transition_mode', None))

    carve_db = 3.5
    carve_lo_hz = 160.0
    carve_hi_hz = 5200.0

    if vocal_state in {'lead', 'lead_only'}:
        carve_db = 5.25
        carve_lo_hz = 180.0
        carve_hi_hz = 6200.0
    elif vocal_state == 'support':
        carve_db = 4.25
        carve_lo_hz = 170.0
        carve_hi_hz = 5600.0
    elif role == 'foreground_counterlayer':
        carve_db = 4.0
        carve_lo_hz = 180.0
        carve_hi_hz = 5200.0
    elif role in {'filtered_counterlayer', 'filtered_support'}:
        carve_db = 2.5
        carve_lo_hz = 220.0
        carve_hi_hz = 4200.0

    if transition_mode in {'arrival_handoff', 'single_owner_handoff'}:
        carve_db += 0.75
        carve_hi_hz = max(carve_hi_hz, 5800.0)
    elif transition_mode == 'same_parent_flow':
        carve_db *= 0.7
        carve_hi_hz = min(carve_hi_hz, 4200.0)
    elif transition_mode == 'backbone_flow':
        carve_db *= 0.6
        carve_hi_hz = min(carve_hi_hz, 3600.0)

    return float(carve_db), float(carve_lo_hz), float(carve_hi_hz)


def _adaptive_overlap_carve_db(work, section, vocal_mask: np.ndarray, base_carve_db: float) -> float:
    if vocal_mask.size == 0:
        return float(base_carve_db)

    transition_mode = _normalize_transition_mode_token(getattr(section, 'transition_mode', None))
    vocal_state = str(getattr(work, 'vocal_state', '') or '')

    mask_mean = float(np.mean(vocal_mask))
    dense_presence = float(np.mean(vocal_mask >= 0.55))

    intensity = 0.55 * mask_mean + 0.45 * dense_presence
    intensity = float(np.clip(intensity, 0.0, 1.0))

    scaled_carve = float(base_carve_db) * (0.55 + 0.9 * intensity)

    if transition_mode in {'arrival_handoff', 'single_owner_handoff'} and vocal_state in {'lead', 'lead_only'}:
        scaled_carve += 0.25 + 0.65 * intensity
    elif transition_mode == 'same_parent_flow':
        scaled_carve *= 0.92
    elif transition_mode == 'backbone_flow':
        scaled_carve *= 0.85

    if dense_presence < 0.1 and mask_mean < 0.2:
        scaled_carve *= 0.8

    return float(np.clip(scaled_carve, 1.5, 7.0))


def _section_mix_cleanup(segment: np.ndarray, sr: int, work, section) -> np.ndarray:
    out = segment.astype(np.float32)
    transition_mode = _normalize_transition_mode_token(getattr(section, 'transition_mode', None))
    overlap_sec = min(float(work.fade_in_sec), max(0.0, float(work.target_duration_sec)))
    overlap_samples = min(out.shape[1], max(0, int(round(overlap_sec * sr))))
    if overlap_samples <= 0:
        return out

    intro = out[:, :overlap_samples]
    cleanup_gain_db = 0.0
    highpass_hz = 0.0
    bandstop_low_hz = 0.0
    bandstop_high_hz = 0.0
    recover_curve_exp = 1.0

    if section.owner_mode == 'backbone_plus_donor_support' and section.background_owner is not None and section.background_owner != section.foreground_owner:
        cleanup_gain_db = -2.75
        highpass_hz = 175.0
        bandstop_low_hz, bandstop_high_hz = 280.0, 5200.0
        recover_curve_exp = 2.1
    elif transition_mode in {"arrival_handoff", "single_owner_handoff"}:
        cleanup_gain_db = -1.5
        highpass_hz = 165.0
        bandstop_low_hz, bandstop_high_hz = 300.0, 5000.0
        recover_curve_exp = 1.6
    elif transition_mode == "same_parent_flow":
        cleanup_gain_db = -0.2
        highpass_hz = 70.0
        recover_curve_exp = 1.05
    elif transition_mode == "backbone_flow":
        cleanup_gain_db = -0.5
        highpass_hz = 95.0
        recover_curve_exp = 1.1
    elif section.allowed_overlap:
        cleanup_gain_db = -0.75
        highpass_hz = 125.0
        bandstop_low_hz, bandstop_high_hz = 260.0, 3200.0
        recover_curve_exp = 1.2

    if cleanup_gain_db != 0.0 or highpass_hz > 0.0 or bandstop_high_hz > bandstop_low_hz:
        cleaned = intro
        if cleanup_gain_db != 0.0:
            cleaned = _apply_gain_db(cleaned, cleanup_gain_db)
        if highpass_hz > 0.0:
            cleaned = _highpass(cleaned, sr, highpass_hz)
        if bandstop_high_hz > bandstop_low_hz:
            cleaned = _bandstop(cleaned, sr, bandstop_low_hz, bandstop_high_hz)

        recover = np.linspace(0.0, 1.0, overlap_samples, endpoint=True, dtype=np.float32)
        recover = np.power(recover, np.float32(recover_curve_exp))[np.newaxis, :]
        intro = cleaned * (1.0 - recover) + intro * recover

    out[:, :overlap_samples] = intro.astype(np.float32)
    return out


def _finalize_master(audio: np.ndarray, sr: int, bpm: float = 120.0) -> np.ndarray:
    finished = _highpass(audio.astype(np.float32), sr, 28.0)
    finished = bpm_synced_glue_compress(finished, sr=sr, bpm=bpm, threshold_db=-16.0, ratio=1.8, makeup_db=0.75)
    finished = _compress_bus(finished, threshold_db=-18.0, ratio=2.0, makeup_db=0.5)
    finished = lookahead_envelope_limit(finished, sr=sr, ceiling_db=-1.2, attack_ms=2.0, release_ms=60.0, lookahead_ms=1.5)
    finished = _soft_limit(finished, drive=1.05)
    finished = lufs_normalize(finished, target_lufs=-12.0, sr=sr)
    return _peak_normalize(finished, -1.0)


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
    section_map = {section.index: section for section in manifest.sections}

    for work in sorted(manifest.work_orders, key=lambda item: (item.target_start_sec, item.order_id)):
        audio, _ = _load_stereo(work.source_path, sr)
        segment = _extract(audio, sr, work.source_start_sec, work.source_end_sec)
        segment = _fit_to_duration(segment, sr, work.target_duration_sec, work.stretch_ratio)
        segment = _prepare_role_layer(segment, sr, work, section_map[work.section_index])
        segment = _apply_support_entry_shape(segment, sr, work, section_map[work.section_index])
        segment = _section_mix_cleanup(segment, sr, work, section_map[work.section_index])
        segment = _cue_safe_transition_anchor(segment, sr, work.fade_in_sec)
        segment = _apply_gain_db(segment, work.gain_db)
        segment = _apply_transition_sonics(
            segment,
            sr,
            work.fade_in_sec,
            work.fade_out_sec,
            work.transition_type,
            work.transition_mode,
        )
        segment = _apply_edge_fades(segment, sr, work.fade_in_sec, work.fade_out_sec)
        start_sample = int(round(work.target_start_sec * sr))
        end_sample = min(total_samples, start_sample + segment.shape[1])
        seg = segment[:, : max(0, end_sample - start_sample)]
        if seg.shape[1] > 0:
            existing = master[:, start_sample:end_sample]
            if existing.shape[1] == seg.shape[1] and _should_apply_overlap_carve(work, section_map[work.section_index]):
                existing_mono = np.mean(existing, axis=0)
                seg_mono = np.mean(seg, axis=0)
                mask = compute_vocal_presence_mask(existing_mono, seg_mono, sr)
                carve_db, carve_lo_hz, carve_hi_hz = _overlap_carve_settings(work, section_map[work.section_index])
                carve_db = _adaptive_overlap_carve_db(work, section_map[work.section_index], mask, carve_db)
                existing = apply_spectral_carve(
                    existing,
                    mask,
                    sr,
                    carve_db=carve_db,
                    carve_lo_hz=carve_lo_hz,
                    carve_hi_hz=carve_hi_hz,
                )
                master[:, start_sample:end_sample] = existing
            master[:, start_sample:end_sample] += seg

    raw_wav = str((outdir / "child_raw.wav").resolve())
    master_wav = str((outdir / "child_master.wav").resolve())
    master_mp3 = str((outdir / "child_master.mp3").resolve())

    sf.write(raw_wav, master.T, sr, subtype="FLOAT")
    final_audio = _finalize_master(master, sr, bpm=manifest.target_bpm)
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
