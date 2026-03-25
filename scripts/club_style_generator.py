#!/usr/bin/env python3
"""Reference-informed club track generator.

- Scans a local reference folder (your legally owned tracks)
- Builds a lightweight style profile (tempo/key tendencies)
- Generates an original full song draft from scratch
- Exports WAV + MP3 and a profile/report JSON

This is intentionally style-informed, not melody-copying.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.analysis import analyze_audio_file  # noqa: E402

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac"}


@dataclass
class StyleProfile:
    tempo_bpm: float
    key_label: str
    track_count: int
    energy_hint: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tempo_bpm": round(float(self.tempo_bpm), 2),
            "key_label": self.key_label,
            "track_count": int(self.track_count),
            "energy_hint": round(float(self.energy_hint), 3),
        }


def _discover_audio_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS])


def _safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return float(value[0])
        return float(value)
    except Exception:
        return default


def build_style_profile(reference_dir: Path) -> tuple[StyleProfile, dict[str, Any]]:
    files = _discover_audio_files(reference_dir)
    tempos: list[float] = []
    key_counts: dict[str, int] = {}
    energy_scores: list[float] = []
    analyzed: list[dict[str, Any]] = []

    for file_path in files:
        try:
            song = analyze_audio_file(str(file_path))
            tempo = _safe_float(getattr(song, "tempo_bpm", None), 126.0)
            tempos.append(tempo)

            key = getattr(song, "key", None) or {}
            tonic = str((key or {}).get("tonic") or "A")
            mode = str((key or {}).get("mode") or "minor")
            key_label = f"{tonic} {mode}"
            key_counts[key_label] = key_counts.get(key_label, 0) + 1

            energy = getattr(song, "energy", None) or {}
            bar_rms = list((energy or {}).get("bar_rms") or [])
            if bar_rms:
                energy_scores.append(float(sum(bar_rms) / len(bar_rms)))

            analyzed.append(
                {
                    "path": str(file_path),
                    "tempo_bpm": round(float(tempo), 3),
                    "key": key_label,
                }
            )
        except Exception as exc:
            analyzed.append({"path": str(file_path), "error": str(exc)})

    if tempos:
        tempo_bpm = float(statistics.median(tempos))
    else:
        tempo_bpm = 126.0

    if key_counts:
        key_label = max(key_counts.items(), key=lambda kv: kv[1])[0]
    else:
        key_label = "A minor"

    energy_hint = float(statistics.median(energy_scores)) if energy_scores else 0.24

    profile = StyleProfile(
        tempo_bpm=max(120.0, min(130.0, tempo_bpm)),
        key_label=key_label,
        track_count=len(files),
        energy_hint=max(0.08, min(0.6, energy_hint)),
    )

    debug = {
        "reference_dir": str(reference_dir),
        "files_found": len(files),
        "analyzed": analyzed,
        "tempo_samples": [round(float(x), 3) for x in tempos],
        "key_counts": key_counts,
        "energy_samples": [round(float(x), 4) for x in energy_scores],
    }
    return profile, debug


def _midi_to_hz(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def synthesize_original_track(profile: StyleProfile, out_wav: Path, seed: int = 42, style_brief: str = "") -> dict[str, Any]:
    random.seed(seed)
    brief = (style_brief or "").lower()
    chant_focus = any(token in brief for token in ("chant", "vocal", "hook"))
    shuffle_focus = any(token in brief for token in ("shuffle", "shuffling", "swing", "thumper", "tech house"))

    sr = 44100
    bpm = float(profile.tempo_bpm)
    spb = 60.0 / bpm
    bars = 96  # ~3 minutes at 126 BPM
    beats_total = bars * 4
    duration_sec = beats_total * spb
    n = int(duration_sec * sr)

    left = array("f", [0.0]) * n
    right = array("f", [0.0]) * n

    def add(start_s: float, length_s: float, fn, pan: float = 0.0):
        start = int(start_s * sr)
        length = int(length_s * sr)
        end = min(n, start + length)
        pan = max(-1.0, min(1.0, pan))
        l_gain = 0.5 * (1.0 - pan)
        r_gain = 0.5 * (1.0 + pan)
        for i in range(start, end):
            t = (i - start) / sr
            val = fn(t)
            left[i] += val * l_gain
            right[i] += val * r_gain

    transition_boundaries = {16, 40, 48, 72, 88}

    def section_gain(bar_idx: int) -> dict[str, float]:
        # 0-15 intro, 16-39 groove A, 40-47 break, 48-71 drop, 72-87 groove B, 88-95 outro
        if bar_idx < 16:
            return {"kick": 1.0, "clap": 0.30, "hat": 0.58, "bass": 0.0, "stab": 0.08, "fx": 0.28, "vocal": 0.12}
        if bar_idx < 40:
            return {"kick": 1.0, "clap": 0.74, "hat": 0.88, "bass": 0.96, "stab": 0.28, "fx": 0.30, "vocal": 0.36}
        if bar_idx < 48:
            return {"kick": 0.46, "clap": 0.30, "hat": 0.46, "bass": 0.34, "stab": 0.18, "fx": 0.58, "vocal": 0.46}
        if bar_idx < 72:
            return {"kick": 1.0, "clap": 0.90, "hat": 1.0, "bass": 1.0, "stab": 0.40, "fx": 0.36, "vocal": 0.54}
        if bar_idx < 88:
            return {"kick": 1.0, "clap": 0.80, "hat": 0.90, "bass": 0.94, "stab": 0.32, "fx": 0.32, "vocal": 0.40}
        return {"kick": 0.78, "clap": 0.40, "hat": 0.48, "bass": 0.30, "stab": 0.14, "fx": 0.22, "vocal": 0.10}

    def transition_shape(bar_idx: int, beat_in_bar: float) -> float:
        # Soften edges around major section boundaries to improve seam quality.
        factor = 1.0
        if (bar_idx + 1) in transition_boundaries and beat_in_bar >= 2.5:
            factor *= 0.56
        if bar_idx in transition_boundaries and beat_in_bar <= 1.0:
            factor *= 0.56
        return factor

    def kick_fn(amp: float = 1.0):
        def fn(t: float) -> float:
            if t > 0.24:
                return 0.0
            env = math.exp(-t * 24.0)
            f = 120.0 * math.exp(-t * 28.0) + 42.0
            body = math.sin(2.0 * math.pi * f * t)
            click = math.sin(2.0 * math.pi * 2100.0 * t) * math.exp(-t * 110.0)
            return amp * (0.92 * body * env + 0.08 * click)

        return fn

    def clap_fn(amp: float = 1.0):
        phase = random.random() * 6.28

        def fn(t: float) -> float:
            if t > 0.14:
                return 0.0
            env = math.exp(-t * 30.0)
            noise = random.random() * 2.0 - 1.0
            bright = math.sin(2 * math.pi * (1700.0) * t + phase) * 0.2
            return amp * (noise * env * 0.32 + bright * env)

        return fn

    def hat_fn(amp: float = 1.0):
        phase = random.random() * 6.28

        def fn(t: float) -> float:
            if t > 0.06:
                return 0.0
            env = math.exp(-t * 76.0)
            n = random.random() * 2.0 - 1.0
            metal = math.sin(2 * math.pi * 8500.0 * t + phase) + 0.5 * math.sin(2 * math.pi * 11100.0 * t + phase * 0.6)
            return amp * env * (0.13 * n + 0.08 * metal)

        return fn

    def bass_note(freq: float, amp: float = 1.0):
        def fn(t: float) -> float:
            if t > 0.43:
                return 0.0
            env = math.exp(-t * 6.1)
            sub = math.sin(2 * math.pi * freq * t)
            sat = math.sin(2 * math.pi * freq * t + 0.9 * math.sin(2 * math.pi * (freq * 0.5) * t))
            top = math.sin(2 * math.pi * freq * 2.0 * t) * 0.18
            return amp * env * (0.75 * sub + 0.40 * sat + top)

        return fn

    def stab_note(freq: float, amp: float = 1.0):
        def fn(t: float) -> float:
            if t > 0.34:
                return 0.0
            env = math.exp(-t * 10.8)
            a = math.sin(2 * math.pi * freq * t)
            b = math.sin(2 * math.pi * freq * 2.01 * t) * 0.55
            c = math.sin(2 * math.pi * freq * 3.01 * t) * 0.22
            wob = 0.64 + 0.36 * math.sin(2 * math.pi * 2.4 * t)
            return amp * env * wob * (a + b + c) * 0.6

        return fn

    def riser_fn(amp: float = 1.0):
        phase = random.random() * 6.28

        def fn(t: float) -> float:
            if t > 2.2:
                return 0.0
            env = min(1.0, t / 1.9) * (1.0 - max(0.0, t - 1.9) / 0.3)
            f = 200.0 + 2800.0 * (t / 2.2)
            tone = math.sin(2 * math.pi * f * t + phase)
            noise = (random.random() * 2 - 1) * 0.45
            return amp * env * (0.23 * tone + 0.11 * noise)

        return fn

    def chant_hit_fn(amp: float = 1.0, vowel_bias: float = 0.0):
        # synthetic chant-like stab (vocal-inspired, no copied melody/lyrics)
        base_f = 180.0 + 18.0 * vowel_bias
        form1 = 700.0 + 80.0 * vowel_bias
        form2 = 1400.0 + 120.0 * vowel_bias

        def fn(t: float) -> float:
            if t > 0.24:
                return 0.0
            env = math.exp(-t * 14.0)
            pitch = math.sin(2 * math.pi * base_f * t)
            f1 = math.sin(2 * math.pi * form1 * t)
            f2 = math.sin(2 * math.pi * form2 * t)
            grit = (random.random() * 2.0 - 1.0) * 0.05
            return amp * env * (0.56 * pitch + 0.30 * f1 + 0.22 * f2 + grit)

        return fn

    # Drums
    for beat in range(beats_total):
        bar = beat // 4
        g = section_gain(bar)
        t0 = beat * spb
        add(t0, 0.24, kick_fn(0.98 * g["kick"]), pan=0.0)
        # thumper ghost kick for shuffle drive
        if g["kick"] > 0.7 and (shuffle_focus or bar >= 16) and beat % 2 == 0:
            add(t0 + 0.74 * spb, 0.14, kick_fn(0.18 * g["kick"]), pan=0.0)
        if beat % 4 in (1, 3):
            add(t0 + 0.002, 0.15, clap_fn(0.65 * g["clap"]), pan=0.0)

    swing = 0.07 if shuffle_focus else 0.035
    for step in range(beats_total * 2):
        beat_pos = step * 0.5
        bar = int(beat_pos // 4)
        g = section_gain(bar)
        t0 = beat_pos * spb
        if step % 2 == 1:
            t_hat = t0 + swing * spb
            add(t_hat, 0.07, hat_fn(0.62 * g["hat"]), pan=0.22)
        if g["hat"] > 0.72 and random.random() < 0.22:
            add(t0 + (0.23 + swing * 0.35) * spb, 0.05, hat_fn(0.38 * g["hat"]), pan=-0.2)

    # Bass pattern (heavier, syncopated, hooky)
    pattern = [33, 33, 36, 33, 40, 38, 36, 31]
    syncopation = [0.0, 0.03, 0.0, 0.05, 0.0, 0.02, 0.0, 0.06]
    for bar in range(bars):
        g = section_gain(bar)
        if g["bass"] <= 0.01:
            continue
        bar_start = bar * 4 * spb
        for i, midi in enumerate(pattern):
            beat_in_bar = (i * 0.5)
            t0 = bar_start + (beat_in_bar + syncopation[i]) * spb
            edge = transition_shape(bar, beat_in_bar)
            amp = (0.31 + 0.08 * profile.energy_hint) * g["bass"] * edge
            if i in (4, 5, 7):
                amp *= 1.08
            add(t0, 0.36, bass_note(_midi_to_hz(midi), amp), pan=0.0)

    # Stabs
    chord_roots = [45, 48, 43, 40]
    for bar in range(bars):
        g = section_gain(bar)
        if g["stab"] <= 0.01:
            continue
        bar_start = bar * 4 * spb
        root = chord_roots[bar % len(chord_roots)]
        for hit in (0.0, 1.5, 3.0):
            t0 = bar_start + hit * spb
            base = _midi_to_hz(root)
            edge = transition_shape(bar, hit)
            amp = 0.18 * g["stab"] * edge
            pan = -0.25 if hit < 1.0 else (0.25 if hit > 2.0 else 0.0)
            add(t0, 0.32, stab_note(base, amp), pan=pan)
            add(t0, 0.32, stab_note(base * 1.26, amp * 0.78), pan=pan * 0.9)
            add(t0, 0.32, stab_note(base * 1.50, amp * 0.66), pan=pan * 0.8)

    # Chanted vocal hook (synthetic chant stabs)
    for bar in range(bars):
        g = section_gain(bar)
        if g["vocal"] <= 0.01:
            continue
        bar_start = bar * 4 * spb
        if 18 <= bar <= 87:
            chant_pattern = [0.5, 1.25, 2.0, 2.75]
            for idx, hit in enumerate(chant_pattern):
                if (idx == 0 and random.random() < 0.06):
                    continue
                edge = transition_shape(bar, hit)
                amp = (0.13 + (0.04 if chant_focus else 0.0)) * g["vocal"] * edge
                pan = -0.12 if idx % 2 == 0 else 0.12
                add(bar_start + hit * spb, 0.20, chant_hit_fn(amp, vowel_bias=(idx - 1.5) * 0.4), pan=pan)

    # FX risers
    for bar in [15, 39, 47, 71, 87]:
        g = section_gain(bar)
        t0 = (bar * 4 + 2.0) * spb
        add(t0, 2.2, riser_fn(0.52 * g["fx"]), pan=0.0)

    # Sidechain pump
    for i in range(n):
        t = i / sr
        beat_phase = (t / spb) % 1.0
        pump = 0.85 + 0.15 * min(1.0, beat_phase * 3.6)
        left[i] *= pump
        right[i] *= pump

    peak = max(1e-9, max(max(abs(x) for x in left), max(abs(x) for x in right)))
    target = 0.86
    scale = target / peak

    pcm = array("h")
    for l, r in zip(left, right):
        yl = math.tanh(1.42 * max(-1.0, min(1.0, l * scale)))
        yr = math.tanh(1.42 * max(-1.0, min(1.0, r * scale)))
        pcm.append(int(yl * 32767))
        pcm.append(int(yr * 32767))

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    return {
        "sample_rate": sr,
        "bars": bars,
        "duration_seconds": round(float(duration_sec), 3),
        "bpm": round(float(bpm), 3),
        "channels": 2,
        "shuffle_focus": bool(shuffle_focus),
        "chant_focus": bool(chant_focus),
    }


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    # Gentle cleanup chain to improve mix sanity without crushing transients.
    afilter = (
        "highpass=f=28,"
        "lowpass=f=16500,"
        "acompressor=threshold=-16dB:ratio=2.2:attack=10:release=120:makeup=1.8,"
        "alimiter=limit=0.93"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-af",
        afilter,
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "320k",
        str(mp3_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an original club draft from local references")
    parser.add_argument("--references-dir", required=True, help="Folder containing reference songs")
    parser.add_argument("--output-dir", default="/Users/johnpeter/Music/AI_DJ_Output", help="Output folder")
    parser.add_argument("--name", default="club_style_draft_v1", help="Output base filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--style-brief", default="", help="Optional text brief to bias groove/arrangement choices")
    args = parser.parse_args()

    references_dir = Path(args.references_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile, debug = build_style_profile(references_dir)

    wav_path = output_dir / f"{args.name}.wav"
    mp3_path = output_dir / f"{args.name}.mp3"
    profile_path = output_dir / f"{args.name}_style_profile.json"
    report_path = output_dir / f"{args.name}_generation_report.json"

    synth_info = synthesize_original_track(profile, wav_path, seed=int(args.seed), style_brief=str(args.style_brief or ""))
    wav_to_mp3(wav_path, mp3_path)

    profile_payload = profile.to_dict()
    profile_path.write_text(json.dumps(profile_payload, indent=2, sort_keys=True), encoding="utf-8")

    report_payload = {
        "style_brief": str(args.style_brief or ""),
        "profile": profile_payload,
        "synthesis": synth_info,
        "references_debug": debug,
        "outputs": {
            "wav": str(wav_path),
            "mp3": str(mp3_path),
            "profile": str(profile_path),
        },
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Generated MP3: {mp3_path}")
    print(f"Style profile: {profile_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
