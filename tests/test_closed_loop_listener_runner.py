from __future__ import annotations

import json
from pathlib import Path

from scripts import closed_loop_listener_runner as loop


BASE_REPORT = {
    'source_path': 'dummy.wav',
    'duration_seconds': 120.0,
    'structure': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'groove': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'energy_arc': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'transition': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'coherence': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'mix_sanity': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'song_likeness': {'score': 70.0, 'summary': 'ok', 'evidence': [], 'fixes': [], 'details': {}},
    'verdict': 'mixed',
    'top_reasons': [],
    'top_fixes': [],
    'gating': {'status': 'pass', 'raw_overall_score': 0.0},
    'analysis_version': '0.5.0',
}


def test_closed_loop_stops_on_quality_gate(monkeypatch, tmp_path: Path):
    scores = iter([72.0, 86.0])

    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        score = next(scores)
        payload = dict(BASE_REPORT)
        payload['overall_score'] = score
        payload['verdict'] = 'promising' if score >= 85.0 else 'mixed'
        payload['gating'] = {'status': 'pass', 'raw_overall_score': score}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'goal': {'target_listener_score': target_score, 'current_overall_score': 72.0, 'gap_to_target': 27.0},
            'ranked_interventions': [{'component': 'song_likeness', 'code_targets': ['src/core/planner/arrangement.py']}],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)
    monkeypatch.setattr(loop, '_run_shell_template', lambda *args, **kwargs: {'returncode': 0, 'stdout': '', 'stderr': '', 'command': 'noop'})

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop'),
        max_iterations=3,
        quality_gate=85.0,
        change_command='echo patch {iteration}',
    )

    assert report['stop_reason'] == 'quality_gate_reached:85.0'
    assert report['best_iteration']['iteration'] == 2
    assert Path(tmp_path / 'loop' / 'closed_loop_report.json').exists()



def test_closed_loop_stops_on_plateau_without_change_command(monkeypatch, tmp_path: Path):
    def fake_render(song_a: str, song_b: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'output_dir': str(output_dir), 'stdout': '', 'stderr': '', 'command': ['fusion']}

    def fake_candidate_report(_candidate_input: str):
        payload = dict(BASE_REPORT)
        payload['overall_score'] = 58.0
        payload['verdict'] = 'weak'
        payload['gating'] = {'status': 'reject', 'raw_overall_score': 58.0}
        return payload

    def fake_feedback(_candidate: str, _refs: list[str], target_score: float = 99.0):
        return {
            'goal': {'target_listener_score': target_score, 'current_overall_score': 58.0, 'gap_to_target': 41.0},
            'ranked_interventions': [{'component': 'song_likeness', 'code_targets': ['src/core/evaluation/listen.py']}],
        }

    monkeypatch.setattr(loop, 'render_iteration', fake_render)
    monkeypatch.setattr(loop, '_candidate_report', fake_candidate_report)
    monkeypatch.setattr(loop, 'build_feedback_brief', fake_feedback)

    report = loop.run_closed_loop(
        song_a='a.wav',
        song_b='b.wav',
        references=['ref.wav'],
        output_root=str(tmp_path / 'loop'),
        max_iterations=4,
        quality_gate=90.0,
        plateau_limit=2,
        change_command=None,
    )

    assert report['stop_reason'] == 'no_change_command_configured'
    assert report['best_iteration']['candidate_overall_score'] == 58.0
    saved = json.loads((tmp_path / 'loop' / 'closed_loop_report.json').read_text(encoding='utf-8'))
    assert saved['stop_reason'] == 'no_change_command_configured'
