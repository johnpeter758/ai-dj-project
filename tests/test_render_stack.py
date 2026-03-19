import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.core.analysis.models import SongDNA
from src.core.planner import build_stub_arrangement_plan
from src.core.planner.models import ChildArrangementPlan, CompatibilityFactors, ParentReference, PlannedSection
from src.core.render import resolve_render_plan, render_resolved_plan
from src.core.render.manifest import ResolvedRenderPlan
from src.core.render.renderer import _apply_transition_sonics
from src.core.render.transitions import incoming_gain_db, transition_overlap_beats, transition_overlap_seconds


def make_song(path: str, tempo: float, tonic: str, mode: str, camelot: str, sections: int, mean_rms: float) -> SongDNA:
    return SongDNA(
        source_path=path,
        sample_rate=44100,
        duration_seconds=12.0,
        tempo_bpm=tempo,
        key={"tonic": tonic, "mode": mode, "camelot": camelot, "confidence": 0.9},
        structure={
            "sections": [{"label": f"section_{i}", "start": 0.0, "end": 12.0} for i in range(sections)],
            "phrase_boundaries_seconds": [0.0, 4.0, 8.0],
        },
        energy={"summary": {"mean_rms": mean_rms}},
        metadata={"schema_version": "0.1.0", "tempo": {"beat_times": [i * 0.5 for i in range(25)]}},
        stems={"enabled": False, "files": {}},
    )


def write_sine(path: Path, frequency_hz: float, sr: int = 44100, seconds: float = 12.0, amplitude: float = 0.1) -> Path:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(path, (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32), sr)
    return path


def clone_manifest(manifest: ResolvedRenderPlan, **updates) -> ResolvedRenderPlan:
    payload = {
        "schema_version": manifest.schema_version,
        "sample_rate": manifest.sample_rate,
        "target_bpm": manifest.target_bpm,
        "sections": list(manifest.sections),
        "work_orders": list(manifest.work_orders),
        "warnings": list(manifest.warnings),
        "fallbacks": list(manifest.fallbacks),
    }
    payload.update(updates)
    return ResolvedRenderPlan(**payload)


def test_resolve_render_plan_emits_sections_and_work_orders(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)
    assert len(manifest.sections) == 3
    assert len(manifest.work_orders) >= len(manifest.sections)
    assert sum(1 for work in manifest.work_orders if work.order_type == 'section_base') == len(manifest.sections)


def test_resolve_render_plan_keeps_non_overlap_sections_single_owner(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    plan = _single_section_plan(source_parent='A', bar_count=4, source_section_label='phrase_0_2')

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[0].allowed_overlap is False
    assert manifest.sections[0].background_owner is None
    assert manifest.sections[0].foreground_owner == 'A'
    assert manifest.sections[0].low_end_owner == 'A'


def test_render_resolved_plan_writes_outputs(tmp_path: Path):
    sr = 44100
    t = np.linspace(0, 12.0, sr * 12, endpoint=False)
    a_audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    b_audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, a_audio, sr)
    sf.write(p2, b_audio, sr)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)
    result = render_resolved_plan(manifest, tmp_path / 'render')
    assert Path(result.raw_wav_path).exists()
    assert Path(result.master_wav_path).exists()
    assert Path(result.manifest_path).exists()


def test_resolve_render_plan_target_timing_contract(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)

    starts = [section.target.start_sec for section in manifest.sections]
    durations = [section.target.duration_sec for section in manifest.sections]
    assert starts == sorted(starts)
    assert all(duration > 0 for duration in durations)
    assert manifest.sections[-1].target.end_sec == max(section.target.end_sec for section in manifest.sections)
    assert all(section.target.anchor_bpm == pytest.approx(manifest.target_bpm) for section in manifest.sections)



def test_resolve_render_plan_uses_common_target_bpm_across_mixed_parent_sections(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 90.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 90.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='intro', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='verse', start_bar=4, bar_count=4, source_parent='B', source_section_label='phrase_0_2'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.target_bpm == pytest.approx(120.0)
    assert manifest.sections[0].target.duration_sec == pytest.approx(8.0)
    assert manifest.sections[1].target.start_sec == pytest.approx(8.0)
    assert manifest.sections[1].target.duration_sec == pytest.approx(8.0)
    assert manifest.sections[1].target.anchor_bpm == pytest.approx(120.0)
    assert manifest.work_orders[1].target_start_sec == pytest.approx(8.0)
    assert manifest.work_orders[1].target_duration_sec == pytest.approx(8.0)


def test_build_stub_arrangement_plan_prefers_phrase_labels_over_missing_section_fallback(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)
    joined = '\n'.join(manifest.warnings + manifest.fallbacks)
    assert all(section.source.source_section_label.startswith('phrase_') for section in manifest.sections)
    assert 'missing or unresolved' not in joined


def test_resolve_render_plan_avoids_full_song_window_for_coarse_section(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    a.structure['sections'] = [{'label': 'section_0', 'start': 0.0, 'end': 12.0}]
    plan = _single_section_plan(source_parent='A', bar_count=4, source_section_label='section_0')

    manifest = resolve_render_plan(plan, a, b)

    source = manifest.sections[0].source
    assert source.snapped_start_sec == 0.0
    assert source.snapped_end_sec == 8.0
    assert (source.snapped_end_sec - source.snapped_start_sec) < a.duration_seconds
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[0].warnings)
    assert 'too coarse for direct use' in joined


def test_resolve_render_plan_uses_phrase_safe_subwindow_inside_strong_section(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    a.structure['sections'] = [{'label': 'verse', 'start': 2.0, 'end': 10.0}]
    plan = _single_section_plan(source_parent='A', bar_count=2, source_section_label='verse')

    manifest = resolve_render_plan(plan, a, b)

    source = manifest.sections[0].source
    assert source.snapped_start_sec == 4.0
    assert source.snapped_end_sec == 8.0
    assert manifest.sections[0].stretch_ratio == 1.0


def test_resolve_render_plan_directly_uses_phrase_window_label(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    a.structure['phrase_boundaries_seconds'] = [0.0, 4.0, 8.0, 12.0]
    plan = _single_section_plan(source_parent='A', bar_count=4, source_section_label='phrase_1_3')

    manifest = resolve_render_plan(plan, a, b)

    source = manifest.sections[0].source
    assert source.raw_start_sec == 4.0
    assert source.raw_end_sec == 12.0
    assert source.snapped_start_sec == 4.0
    assert source.snapped_end_sec == 12.0
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[0].warnings)
    assert 'missing or unresolved' not in joined


def test_resolve_render_plan_directly_uses_trimmed_phrase_window_label(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 16, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 16, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    a.duration_seconds = 16.0
    a.structure['phrase_boundaries_seconds'] = [0.0, 4.0, 8.0, 12.0, 16.0]
    a.energy['beat_times'] = [float(i) for i in range(17)]
    plan = _single_section_plan(source_parent='A', bar_count=2, source_section_label='phrase_1_3_trim_head')

    manifest = resolve_render_plan(plan, a, b)

    source = manifest.sections[0].source
    assert source.raw_start_sec == 8.0
    assert source.raw_end_sec == 12.0
    assert source.snapped_start_sec == 8.0
    assert source.snapped_end_sec == 12.0
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[0].warnings)
    assert 'missing or unresolved' not in joined
    assert 'too coarse for direct use' not in joined


def test_render_resolved_plan_is_deterministic_for_same_inputs(tmp_path: Path):
    sr = 44100
    t = np.linspace(0, 12.0, sr * 12, endpoint=False)
    a_audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    b_audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, a_audio, sr)
    sf.write(p2, b_audio, sr)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)

    r1 = render_resolved_plan(manifest, tmp_path / 'render1')
    r2 = render_resolved_plan(manifest, tmp_path / 'render2')
    y1, _ = sf.read(r1.master_wav_path, always_2d=True)
    y2, _ = sf.read(r2.master_wav_path, always_2d=True)
    assert y1.shape == y2.shape
    assert np.allclose(y1, y2)


def test_render_manifest_contains_outputs_block(tmp_path: Path):
    sr = 44100
    t = np.linspace(0, 12.0, sr * 12, endpoint=False)
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    sf.write(p2, (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)
    result = render_resolved_plan(manifest, tmp_path / 'render')
    payload = json.loads(Path(result.manifest_path).read_text())
    assert 'outputs' in payload
    assert payload['outputs']['raw_wav']
    assert payload['outputs']['master_wav']


def _single_section_plan(*, source_parent: str = 'A', start_bar: int = 0, bar_count: int = 8, source_section_label: str | None = 'section_0') -> ChildArrangementPlan:
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    parents = [
        ParentReference('a.wav', 120.0, 'A', 'minor', 12.0),
        ParentReference('b.wav', 120.0, 'C', 'major', 12.0),
    ]
    sections = [
        PlannedSection(
            label='test',
            start_bar=start_bar,
            bar_count=bar_count,
            source_parent=source_parent,
            source_section_label=source_section_label,
        )
    ]
    return ChildArrangementPlan(parents=parents, compatibility=compatibility, sections=sections)


def test_resolve_render_plan_emits_support_order_for_integrated_two_parent_section(tmp_path: Path):
    p1 = write_sine(tmp_path / 'a.wav', 220.0)
    p2 = write_sine(tmp_path / 'b.wav', 440.0)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0),
        sections=[
            PlannedSection(
                label='payoff',
                start_bar=0,
                bar_count=4,
                source_parent='A',
                source_section_label='phrase_0_2',
                support_parent='B',
                support_section_label='phrase_0_2',
                support_gain_db=-8.0,
                support_mode='foreground_counterlayer',
            )
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert len(manifest.work_orders) == 2
    support = next(work for work in manifest.work_orders if work.order_type == 'section_support')
    assert support.parent_id == 'B'
    assert support.role == 'foreground_counterlayer'
    assert support.low_end_state == 'support'
    assert manifest.sections[0].owner_mode == 'integrated_two_parent_section'
    assert manifest.sections[0].foreground_owner == 'B'



def test_render_resolved_plan_support_layer_changes_audio_output(tmp_path: Path):
    p1 = write_sine(tmp_path / 'a.wav', 220.0)
    p2 = write_sine(tmp_path / 'b.wav', 880.0)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    base_plan = _single_section_plan(source_parent='A', bar_count=4, source_section_label='phrase_0_2')
    integrated_plan = ChildArrangementPlan(
        parents=base_plan.parents,
        compatibility=base_plan.compatibility,
        sections=[replace(base_plan.sections[0], label='payoff', support_parent='B', support_section_label='phrase_0_2', support_gain_db=-8.0, support_mode='filtered_counterlayer')],
    )
    base_manifest = resolve_render_plan(base_plan, a, b)
    integrated_manifest = resolve_render_plan(integrated_plan, a, b)
    base_result = render_resolved_plan(base_manifest, tmp_path / 'base_render')
    integrated_result = render_resolved_plan(integrated_manifest, tmp_path / 'integrated_render')
    base_audio, _ = sf.read(base_result.master_wav_path, always_2d=True)
    integrated_audio, _ = sf.read(integrated_result.master_wav_path, always_2d=True)
    assert integrated_manifest.sections[0].owner_mode == 'integrated_two_parent_section'
    assert not np.allclose(base_audio, integrated_audio)



def test_resolve_render_plan_rejects_overlapping_sections(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='first', start_bar=0, bar_count=8, source_parent='A', source_section_label='section_0'),
            PlannedSection(label='second', start_bar=4, bar_count=8, source_parent='B', source_section_label='section_0'),
        ],
    )

    with pytest.raises(ValueError, match='overlaps previous section'):
        resolve_render_plan(plan, a, b)



def test_resolve_render_plan_rejects_invalid_source_parent(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = _single_section_plan(source_parent='C')

    with pytest.raises(ValueError, match='unsupported source_parent'):
        resolve_render_plan(plan, a, b)



def test_resolve_render_plan_surfaces_overstretched_choice_as_warning(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 128.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 128.0, 'C', 'major', '8B', 1, 0.1)
    a.structure['phrase_boundaries_seconds'] = [0.0, 9.6, 12.0]
    plan = _single_section_plan(source_parent='A', bar_count=2, source_section_label='phrase_0_1')

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[0].stretch_ratio > 1.25
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[0].warnings)
    assert 'outside conservative bounds' in joined



def test_resolve_render_plan_clamps_extreme_stretch_ratio(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    a.structure['sections'] = [{'label': 'section_0', 'start': 0.0, 'end': 12.0}]
    plan = _single_section_plan(source_parent='A', bar_count=1, source_section_label='section_0')

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[0].stretch_ratio == 2.0
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[0].warnings)
    assert 'outside conservative bounds' in joined
    assert 'clamped to 2.00' in joined


def test_transition_overlap_seconds_is_safe_for_nonpositive_bpm():
    assert transition_overlap_seconds('blend', 0.0) > 0.0
    assert transition_overlap_seconds('cut', 0.0) == 0.0


def test_resolve_render_plan_applies_transition_aware_incoming_gain(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    plan = _single_section_plan(source_parent='A', bar_count=4, source_section_label='phrase_0_2')
    plan.sections[0].transition_in = 'blend'

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[0].allowed_overlap is True
    assert manifest.sections[0].overlap_beats_max == transition_overlap_beats('blend')
    assert manifest.work_orders[0].gain_db == pytest.approx(incoming_gain_db('blend') - 0.5)
    assert manifest.work_orders[0].gain_db < 0.0
    assert manifest.sections[0].background_owner is None



def test_resolve_render_plan_defaults_cross_parent_handoffs_to_backbone_only(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='intro', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='verse', start_bar=4, bar_count=4, source_parent='A', source_section_label='phrase_0_2', transition_in='blend'),
            PlannedSection(label='build', start_bar=8, bar_count=4, source_parent='B', source_section_label='phrase_0_2', transition_in='blend'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[1].allowed_overlap is True
    assert manifest.sections[1].background_owner is None
    assert manifest.sections[1].owner_mode == 'backbone_only'
    assert manifest.sections[1].arrival_focus == 'backbone_led'
    assert manifest.sections[2].allowed_overlap is True
    assert manifest.sections[2].background_owner is None
    assert manifest.sections[2].owner_mode == 'backbone_only'
    assert manifest.sections[2].arrival_focus == 'backbone_led'
    assert manifest.sections[2].transition_mode is None
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[2].warnings)
    assert 'cross-parent overlap defaulted to backbone-only ownership' in joined


def test_resolve_render_plan_caps_overlap_for_overstretched_transition(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 128.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 128.0, 'C', 'major', '8B', 1, 0.1)
    a.structure['phrase_boundaries_seconds'] = [0.0, 9.6, 12.0]
    plan = _single_section_plan(source_parent='A', bar_count=2, source_section_label='phrase_0_1')
    plan.sections[0].transition_in = 'blend'

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[0].stretch_ratio > 1.25
    assert manifest.sections[0].allowed_overlap is True
    assert manifest.sections[0].overlap_beats_max == 2.0
    assert manifest.work_orders[0].fade_in_sec == transition_overlap_seconds('blend', manifest.target_bpm, stretch_ratio=manifest.sections[0].stretch_ratio)
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[0].warnings)
    assert 'transition overlap capped to 2.0 beats' in joined


def test_resolve_render_plan_allows_explicit_crossfade_support_and_marks_donor_led_arrival(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='verse', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='build', start_bar=4, bar_count=4, source_parent='B', source_section_label='phrase_0_2', transition_in='blend', transition_mode='crossfade_support'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[1].background_owner == 'A'
    assert manifest.sections[1].owner_mode == 'backbone_plus_donor_support'
    assert manifest.sections[1].arrival_focus == 'donor_led'
    assert manifest.sections[1].low_end_owner == 'A'
    assert manifest.sections[1].backbone_owner == 'B'
    assert manifest.sections[1].donor_owner == 'A'
    assert manifest.sections[1].overlap_beats_max == 2.0
    assert manifest.work_orders[1].gain_db == pytest.approx(incoming_gain_db('blend', 'crossfade_support') - 0.75)
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[1].warnings)
    assert 'transition_mode=crossfade_support capped overlap from 8.0 to 2.0 beats' in joined



def test_resolve_render_plan_honors_single_owner_handoff_mode(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='build', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='payoff', start_bar=4, bar_count=4, source_parent='B', source_section_label='phrase_0_2', transition_in='blend', transition_mode='single_owner_handoff'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[1].allowed_overlap is True
    assert manifest.sections[1].overlap_beats_max == 0.5
    assert manifest.sections[1].background_owner is None
    assert manifest.sections[1].transition_mode == 'single_owner_handoff'
    assert manifest.work_orders[1].gain_db == incoming_gain_db('blend', 'single_owner_handoff')
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[1].warnings)
    assert 'transition_mode=single_owner_handoff capped overlap from 8.0 to 0.5 beats' in joined



def test_resolve_render_plan_caps_arrival_handoff_payoff_to_half_beat(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='verse', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='payoff', start_bar=4, bar_count=4, source_parent='B', source_section_label='phrase_0_2', transition_in='blend', transition_mode='arrival_handoff'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[1].allowed_overlap is True
    assert manifest.sections[1].overlap_beats_max == 0.5
    assert manifest.sections[1].background_owner is None
    assert manifest.sections[1].transition_mode == 'arrival_handoff'
    assert manifest.work_orders[1].fade_in_sec == pytest.approx(0.25)
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[1].warnings)
    assert 'transition_mode=arrival_handoff capped overlap from 8.0 to 0.5 beats' in joined



def test_resolve_render_plan_caps_single_owner_bridge_handoff_to_one_beat(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='verse', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='bridge', start_bar=4, bar_count=4, source_parent='B', source_section_label='phrase_0_2', transition_in='blend', transition_mode='single_owner_handoff'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[1].allowed_overlap is True
    assert manifest.sections[1].overlap_beats_max == 1.0
    assert manifest.sections[1].background_owner is None
    assert manifest.sections[1].transition_mode == 'single_owner_handoff'
    assert manifest.work_orders[1].fade_in_sec == pytest.approx(0.5)
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[1].warnings)
    assert 'transition_mode=single_owner_handoff capped overlap from 8.0 to 1.0 beat' in joined



def test_resolve_render_plan_trims_long_same_parent_flow_arrival_gain(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='verse', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='build', start_bar=4, bar_count=4, source_parent='A', source_section_label='phrase_0_2', transition_in='blend', transition_mode='same_parent_flow'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert transition_overlap_beats('blend') == 8.0
    assert manifest.sections[1].overlap_beats_max == 1.0
    assert manifest.work_orders[1].gain_db == pytest.approx(incoming_gain_db('blend'))
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[1].warnings)
    assert 'transition_mode=same_parent_flow capped overlap from 8.0 to 1.0 beat' in joined



def test_resolve_render_plan_caps_same_parent_flow_blend_to_verse_at_two_beats(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='intro', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='verse', start_bar=4, bar_count=4, source_parent='A', source_section_label='phrase_0_2', transition_in='blend', transition_mode='same_parent_flow'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[1].overlap_beats_max == 2.0
    assert manifest.work_orders[1].gain_db == pytest.approx(incoming_gain_db('blend'))
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[1].warnings)
    assert 'transition_mode=same_parent_flow capped overlap from 8.0 to 2.0 beats' in joined



def test_resolve_render_plan_caps_late_payoff_handoff_blend_overlap(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='payoff', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='outro', start_bar=4, bar_count=4, source_parent='B', source_section_label='phrase_0_2', transition_in='blend', transition_mode='arrival_handoff'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[1].allowed_overlap is True
    assert manifest.sections[1].overlap_beats_max == 0.5
    assert manifest.sections[1].background_owner is None
    assert manifest.sections[1].transition_mode == 'arrival_handoff'
    assert manifest.work_orders[1].fade_in_sec == pytest.approx(0.25)
    assert manifest.work_orders[1].gain_db == pytest.approx(incoming_gain_db('blend', 'arrival_handoff') - 0.75)
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[1].warnings)
    assert 'late payoff handoff overlap capped from 8.0 to 0.5 beats' in joined



def test_resolve_render_plan_tightens_late_payoff_bridge_handoff_for_explicit_single_owner_arrival(tmp_path: Path):
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, np.zeros(44100 * 12, dtype=np.float32), 44100)
    sf.write(p2, np.zeros(44100 * 12, dtype=np.float32), 44100)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    compatibility = CompatibilityFactors(tempo=1.0, harmony=1.0, structure=1.0, energy=1.0, stem_conflict=1.0)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(str(p1), 120.0, 'A', 'minor', 12.0),
            ParentReference(str(p2), 120.0, 'C', 'major', 12.0),
        ],
        compatibility=compatibility,
        sections=[
            PlannedSection(label='payoff', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2'),
            PlannedSection(label='bridge', start_bar=4, bar_count=4, source_parent='B', source_section_label='phrase_0_2', transition_in='blend', transition_mode='single_owner_handoff'),
        ],
    )

    manifest = resolve_render_plan(plan, a, b)

    assert manifest.sections[1].allowed_overlap is True
    assert manifest.sections[1].overlap_beats_max == 0.5
    assert manifest.sections[1].background_owner is None
    assert manifest.sections[1].transition_mode == 'single_owner_handoff'
    assert manifest.work_orders[1].fade_in_sec == pytest.approx(0.25)
    assert manifest.work_orders[1].gain_db == pytest.approx(incoming_gain_db('blend', 'single_owner_handoff') - 0.75)
    joined = '\n'.join(manifest.warnings + manifest.fallbacks + manifest.sections[1].warnings)
    assert 'late payoff handoff overlap capped from 8.0 to 0.5 beats' in joined



def test_render_resolved_plan_applies_work_order_gain_deterministically(tmp_path: Path):
    p1 = write_sine(tmp_path / 'a.wav', 220.0)
    p2 = write_sine(tmp_path / 'b.wav', 440.0)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)

    quieter_orders = [
        replace(order, gain_db=-6.0 if idx == 0 else order.gain_db)
        for idx, order in enumerate(manifest.work_orders)
    ]
    quieter_manifest = clone_manifest(manifest, work_orders=quieter_orders)

    base = render_resolved_plan(manifest, tmp_path / 'render_base')
    quiet = render_resolved_plan(quieter_manifest, tmp_path / 'render_quiet')
    y_base, _ = sf.read(base.raw_wav_path, always_2d=True)
    y_quiet, _ = sf.read(quiet.raw_wav_path, always_2d=True)

    assert y_base.shape == y_quiet.shape
    assert not np.allclose(y_base, y_quiet)
    assert np.max(np.abs(y_quiet)) < np.max(np.abs(y_base))



def test_apply_transition_sonics_highpasses_incoming_handoff_window():
    sr = 44100
    seconds = 4.0
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    low = np.sin(2 * np.pi * 80.0 * t)
    high = 0.5 * np.sin(2 * np.pi * 2400.0 * t)
    segment = np.vstack([low + high, low + high]).astype(np.float32)

    shaped = _apply_transition_sonics(segment, sr, fade_in_sec=1.0, fade_out_sec=0.0, transition_type='blend', transition_mode='arrival_handoff')

    early = shaped[0, : int(0.2 * sr)]
    early_low_projection = np.abs(np.mean(early * np.sin(2 * np.pi * 80.0 * t[: early.size])))
    early_high_projection = np.abs(np.mean(early * np.sin(2 * np.pi * 2400.0 * t[: early.size])))
    assert early_low_projection < early_high_projection * 0.6



def test_apply_transition_sonics_lowpasses_outgoing_tail_window():
    sr = 44100
    seconds = 4.0
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    low = np.sin(2 * np.pi * 120.0 * t)
    high = 0.6 * np.sin(2 * np.pi * 6000.0 * t)
    segment = np.vstack([low + high, low + high]).astype(np.float32)

    shaped = _apply_transition_sonics(segment, sr, fade_in_sec=0.0, fade_out_sec=1.0, transition_type='blend', transition_mode='single_owner_handoff')

    tail = shaped[0, -int(0.2 * sr):]
    tail_t = t[: tail.size]
    tail_low_projection = np.abs(np.mean(tail * np.sin(2 * np.pi * 120.0 * tail_t)))
    tail_high_projection = np.abs(np.mean(tail * np.sin(2 * np.pi * 6000.0 * tail_t)))
    assert tail_high_projection < tail_low_projection * 0.75



def test_apply_transition_sonics_highpasses_outgoing_tail_low_end_on_same_parent_flow():
    sr = 44100
    seconds = 4.0
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    low = np.sin(2 * np.pi * 60.0 * t)
    mids = 0.5 * np.sin(2 * np.pi * 1200.0 * t)
    segment = np.vstack([low + mids, low + mids]).astype(np.float32)

    shaped = _apply_transition_sonics(segment, sr, fade_in_sec=0.0, fade_out_sec=1.0, transition_type='blend', transition_mode='same_parent_flow')

    tail = shaped[0, -int(0.2 * sr):]
    raw_tail = segment[0, -int(0.2 * sr):]
    tail_t = t[: tail.size]
    carrier = np.sin(2 * np.pi * 60.0 * tail_t)
    shaped_low_projection = np.abs(np.mean(tail * carrier))
    raw_low_projection = np.abs(np.mean(raw_tail * carrier))
    assert shaped_low_projection < raw_low_projection * 0.85



def test_render_resolved_plan_rejects_invalid_target_timing_contract(tmp_path: Path):
    p1 = write_sine(tmp_path / 'a.wav', 220.0)
    p2 = write_sine(tmp_path / 'b.wav', 440.0)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)

    broken_section = manifest.sections[0]
    broken_section = replace(
        broken_section,
        target=replace(broken_section.target, duration_sec=broken_section.target.duration_sec + 0.25),
    )
    broken_manifest = clone_manifest(manifest, sections=[broken_section, *manifest.sections[1:]])

    with pytest.raises(ValueError, match='target duration does not match'):
        render_resolved_plan(broken_manifest, tmp_path / 'render_invalid')



def test_render_resolved_plan_mp3_fallback_records_missing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p1 = write_sine(tmp_path / 'a.wav', 220.0)
    p2 = write_sine(tmp_path / 'b.wav', 440.0)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)

    monkeypatch.setattr('src.core.render.renderer.subprocess.run', lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    result = render_resolved_plan(manifest, tmp_path / 'render_no_ffmpeg')
    payload = json.loads(Path(result.manifest_path).read_text())

    assert result.master_mp3_path is None
    assert payload['outputs']['master_mp3'] is None



def test_render_resolved_plan_cleans_up_low_end_on_cross_parent_overlap(tmp_path: Path):
    p1 = write_sine(tmp_path / 'a.wav', 220.0, amplitude=0.08)
    p2 = write_sine(tmp_path / 'b.wav', 60.0, amplitude=0.20)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(source_path=str(p1), tempo_bpm=120.0, key_tonic='A', key_mode='minor', duration_seconds=12.0),
            ParentReference(source_path=str(p2), tempo_bpm=120.0, key_tonic='C', key_mode='major', duration_seconds=12.0),
        ],
        compatibility=CompatibilityFactors(tempo=0.8, harmony=0.8, structure=0.8, energy=0.8, stem_conflict=0.2),
        sections=[
            PlannedSection(label='intro', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2', transition_in='cut'),
            PlannedSection(label='payoff', start_bar=4, bar_count=4, source_parent='B', source_section_label='phrase_0_2', transition_in='blend', transition_mode='crossfade_support'),
        ],
    )
    manifest = resolve_render_plan(plan, a, b)
    dirty_manifest = clone_manifest(manifest, sections=[manifest.sections[0], replace(manifest.sections[1], background_owner=None)])

    cleaned = render_resolved_plan(manifest, tmp_path / 'render_cleaned')
    dirty = render_resolved_plan(dirty_manifest, tmp_path / 'render_dirty')
    y_clean, sr = sf.read(cleaned.raw_wav_path, always_2d=True)
    y_dirty, _ = sf.read(dirty.raw_wav_path, always_2d=True)

    overlap_samples = int(round(manifest.work_orders[1].fade_in_sec * sr))
    clean_intro = y_clean[int(manifest.work_orders[1].target_start_sec * sr): int(manifest.work_orders[1].target_start_sec * sr) + overlap_samples]
    dirty_intro = y_dirty[int(manifest.work_orders[1].target_start_sec * sr): int(manifest.work_orders[1].target_start_sec * sr) + overlap_samples]

    assert np.sqrt(np.mean(clean_intro ** 2)) < np.sqrt(np.mean(dirty_intro ** 2))



def test_render_resolved_plan_cleans_up_mids_on_same_parent_overlap(tmp_path: Path):
    p1 = write_sine(tmp_path / 'a.wav', 1200.0, amplitude=0.15)
    p2 = write_sine(tmp_path / 'b.wav', 1200.0, amplitude=0.15)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(source_path=str(p1), tempo_bpm=120.0, key_tonic='A', key_mode='minor', duration_seconds=12.0),
            ParentReference(source_path=str(p2), tempo_bpm=120.0, key_tonic='C', key_mode='major', duration_seconds=12.0),
        ],
        compatibility=CompatibilityFactors(tempo=0.8, harmony=0.8, structure=0.8, energy=0.8, stem_conflict=0.2),
        sections=[
            PlannedSection(label='intro', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2', transition_in='cut'),
            PlannedSection(label='verse', start_bar=4, bar_count=4, source_parent='A', source_section_label='phrase_0_2', transition_in='blend', transition_mode='same_parent_flow'),
        ],
    )
    manifest = resolve_render_plan(plan, a, b)
    dirty_manifest = clone_manifest(
        manifest,
        sections=[manifest.sections[0], replace(manifest.sections[1], allowed_overlap=False)],
        work_orders=[manifest.work_orders[0], replace(manifest.work_orders[1], fade_in_sec=0.0)],
    )

    cleaned = render_resolved_plan(manifest, tmp_path / 'render_same_parent_cleaned')
    dirty = render_resolved_plan(dirty_manifest, tmp_path / 'render_same_parent_dirty')
    y_clean, sr = sf.read(cleaned.raw_wav_path, always_2d=True)
    y_dirty, _ = sf.read(dirty.raw_wav_path, always_2d=True)

    overlap_samples = int(round(manifest.work_orders[1].fade_in_sec * sr))
    start = int(round(manifest.work_orders[1].target_start_sec * sr))
    clean_intro = y_clean[start: start + overlap_samples, 0]
    dirty_intro = y_dirty[start: start + overlap_samples, 0]
    t = np.arange(clean_intro.size, dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * 1200.0 * t)

    clean_projection = np.abs(np.mean(clean_intro * carrier))
    dirty_projection = np.abs(np.mean(dirty_intro * carrier))
    assert clean_projection < dirty_projection * 0.9



def test_render_resolved_plan_keeps_handoff_presence_band_more_suppressed_early(tmp_path: Path):
    sr = 44100
    seconds = 12.0
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    source_a = (0.10 * np.sin(2 * np.pi * 180.0 * t) + 0.16 * np.sin(2 * np.pi * 2800.0 * t)).astype(np.float32)
    source_b = (0.10 * np.sin(2 * np.pi * 220.0 * t) + 0.16 * np.sin(2 * np.pi * 2800.0 * t)).astype(np.float32)
    p1 = tmp_path / 'a.wav'
    p2 = tmp_path / 'b.wav'
    sf.write(p1, source_a, sr)
    sf.write(p2, source_b, sr)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 1, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 1, 0.1)
    plan = ChildArrangementPlan(
        parents=[
            ParentReference(source_path=str(p1), tempo_bpm=120.0, key_tonic='A', key_mode='minor', duration_seconds=12.0),
            ParentReference(source_path=str(p2), tempo_bpm=120.0, key_tonic='C', key_mode='major', duration_seconds=12.0),
        ],
        compatibility=CompatibilityFactors(tempo=0.8, harmony=0.8, structure=0.8, energy=0.8, stem_conflict=0.2),
        sections=[
            PlannedSection(label='intro', start_bar=0, bar_count=4, source_parent='A', source_section_label='phrase_0_2', transition_in='cut'),
            PlannedSection(label='bridge', start_bar=4, bar_count=4, source_parent='B', source_section_label='phrase_0_2', transition_in='blend', transition_mode='arrival_handoff'),
        ],
    )
    manifest = resolve_render_plan(plan, a, b)
    raw_manifest = clone_manifest(
        manifest,
        sections=[manifest.sections[0], replace(manifest.sections[1], allowed_overlap=False)],
        work_orders=[manifest.work_orders[0], replace(manifest.work_orders[1], fade_in_sec=0.0)],
    )

    cleaned = render_resolved_plan(manifest, tmp_path / 'render_handoff_cleaned')
    raw = render_resolved_plan(raw_manifest, tmp_path / 'render_handoff_raw')
    y_clean, _ = sf.read(cleaned.raw_wav_path, always_2d=True)
    y_raw, _ = sf.read(raw.raw_wav_path, always_2d=True)

    overlap_samples = int(round(manifest.work_orders[1].fade_in_sec * sr))
    start = int(round(manifest.work_orders[1].target_start_sec * sr))
    early_samples = max(1, int(round(overlap_samples * 0.35)))
    clean_intro = y_clean[start: start + early_samples, 0]
    raw_intro = y_raw[start: start + early_samples, 0]
    tt = np.arange(clean_intro.size, dtype=np.float32) / sr
    carrier = np.sin(2 * np.pi * 2800.0 * tt)

    clean_projection = np.abs(np.mean(clean_intro * carrier))
    raw_projection = np.abs(np.mean(raw_intro * carrier))
    assert clean_projection < raw_projection * 0.75



def test_render_resolved_plan_applies_master_finish_not_just_peak_normalize(tmp_path: Path):
    p1 = write_sine(tmp_path / 'a.wav', 220.0, amplitude=0.35)
    p2 = write_sine(tmp_path / 'b.wav', 330.0, amplitude=0.35)
    a = make_song(str(p1), 120.0, 'A', 'minor', '8A', 2, 0.1)
    b = make_song(str(p2), 120.0, 'C', 'major', '8B', 2, 0.1)
    plan = build_stub_arrangement_plan(a, b)
    manifest = resolve_render_plan(plan, a, b)

    result = render_resolved_plan(manifest, tmp_path / 'render_master_finish')
    raw, _ = sf.read(result.raw_wav_path, always_2d=True)
    mastered, _ = sf.read(result.master_wav_path, always_2d=True)

    assert raw.shape == mastered.shape
    assert not np.allclose(raw, mastered)
    assert np.max(np.abs(mastered)) <= 10 ** (-1.0 / 20.0) + 1e-3
