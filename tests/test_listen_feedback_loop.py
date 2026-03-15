from __future__ import annotations

import json
from pathlib import Path

from scripts.listen_feedback_loop import build_feedback_brief


def _report(overall: float, *, verdict: str = 'promising', song_likeness: float = 80.0, groove: float = 80.0, energy_arc: float = 80.0, transition: float = 80.0, structure: float = 80.0, coherence: float = 80.0, mix_sanity: float = 80.0) -> dict:
    return {
        'source_path': 'dummy.wav',
        'duration_seconds': 120.0,
        'overall_score': overall,
        'structure': {'score': structure, 'summary': 'structure', 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': groove, 'summary': 'groove', 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': energy_arc, 'summary': 'energy_arc', 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': transition, 'summary': 'transition', 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': coherence, 'summary': 'coherence', 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': mix_sanity, 'summary': 'mix', 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {'score': song_likeness, 'summary': 'song_like', 'evidence': [], 'fixes': [], 'details': {}},
        'verdict': verdict,
        'top_reasons': ['good thing'],
        'top_fixes': ['fix thing'],
        'gating': {'status': 'pass', 'raw_overall_score': overall},
        'analysis_version': '0.5.0',
    }


def test_build_feedback_brief_ranks_biggest_gaps_and_code_targets(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref_a = tmp_path / 'ref_a.json'
    ref_b = tmp_path / 'ref_b.json'

    candidate.write_text(json.dumps(_report(58.0, verdict='weak', song_likeness=35.0, groove=42.0, energy_arc=40.0, transition=44.0)), encoding='utf-8')
    ref_a.write_text(json.dumps(_report(92.0, song_likeness=90.0, groove=89.0, energy_arc=91.0, transition=86.0)), encoding='utf-8')
    ref_b.write_text(json.dumps(_report(88.0, song_likeness=86.0, groove=84.0, energy_arc=87.0, transition=82.0)), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref_a), str(ref_b)], target_score=99.0)

    assert brief['goal']['current_overall_score'] == 58.0
    assert brief['goal']['gap_to_target'] == 41.0
    assert brief['gap_summary']['overall_vs_references'] < 0.0
    assert brief['ranked_interventions']
    top = brief['ranked_interventions'][0]
    assert top['component'] in {'song_likeness', 'energy_arc', 'groove'}
    assert brief['next_code_targets']
    assert any(path.endswith('src/core/planner/arrangement.py') for path in brief['next_code_targets'])
    assert brief['automation_loop'][0].startswith('Compare candidate against references')


def test_build_feedback_brief_collects_reference_strengths(tmp_path: Path):
    candidate = tmp_path / 'candidate.json'
    ref = tmp_path / 'ref.json'
    candidate.write_text(json.dumps(_report(70.0, song_likeness=68.0, groove=67.0, energy_arc=66.0)), encoding='utf-8')
    ref.write_text(json.dumps(_report(95.0, song_likeness=97.0, groove=96.0, energy_arc=94.0, transition=93.0, structure=92.0)), encoding='utf-8')

    brief = build_feedback_brief(str(candidate), [str(ref)])

    assert brief['references'][0]['overall_score'] == 95.0
    strengths = {item['component']: item['avg_reference_score'] for item in brief['reference_strengths']}
    assert strengths
    assert max(strengths.values()) >= 93.0
