from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'listen_gate_benchmark.py'


def _listen_report(
    case_name: str,
    overall: float,
    verdict: str,
    gating: str,
    *,
    energy_arc: float,
    groove: float,
    song_likeness: float,
    aggregate_metrics: dict | None = None,
) -> dict:
    return {
        'source_path': f'{case_name}.wav',
        'duration_seconds': 60.0,
        'overall_score': overall,
        'structure': {'score': overall + 1.0, 'summary': case_name, 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': groove, 'summary': case_name, 'evidence': [], 'fixes': [], 'details': {}},
        'energy_arc': {'score': energy_arc, 'summary': case_name, 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': overall - 1.0, 'summary': case_name, 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': overall, 'summary': case_name, 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': overall, 'summary': case_name, 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {
            'score': song_likeness,
            'summary': case_name,
            'evidence': [],
            'fixes': [],
            'details': {'aggregate_metrics': dict(aggregate_metrics or {})},
        },
        'verdict': verdict,
        'top_reasons': [],
        'top_fixes': [],
        'gating': {'status': gating, 'raw_overall_score': overall},
        'analysis_version': '0.5.0',
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_listen_gate_benchmark_passes_and_exposes_pairwise_deltas(tmp_path: Path):
    strong = _write_json(
        tmp_path / 'strong.json',
        _listen_report('strong', 86.0, 'promising', 'pass', energy_arc=92.0, groove=84.0, song_likeness=88.0),
    )
    weak = _write_json(
        tmp_path / 'weak.json',
        _listen_report('weak', 68.0, 'weak', 'reject', energy_arc=61.0, groove=66.0, song_likeness=52.0),
    )
    spec = {
        'expected_order': ['strong_case', 'weak_case'],
        'cases': [
            {
                'label': 'strong_case',
                'input': str(strong),
                'expect': {
                    'gating_status': 'pass',
                    'overall_score_at_least': 80.0,
                    'component_score_at_least': {'energy_arc': 90.0, 'song_likeness': 85.0},
                    'better_than': [
                        {
                            'other': 'weak_case',
                            'overall_score_delta_at_least': 15.0,
                            'component_score_delta_at_least': {'energy_arc': 25.0, 'song_likeness': 30.0},
                        }
                    ],
                },
            },
            {
                'label': 'weak_case',
                'input': str(weak),
                'expect': {
                    'gating_status': 'reject',
                    'overall_score_at_most': 70.0,
                },
            },
        ],
    }
    spec_path = _write_json(tmp_path / 'spec.json', spec)
    output = tmp_path / 'harness.json'

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(spec_path), '--output', str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['passed'] is True
    assert payload['benchmark']['winner'] == 'strong.json'
    assert [case['label'] for case in sorted(payload['cases'], key=lambda row: row['benchmark_rank'])] == ['strong_case', 'weak_case']
    pair = payload['pairwise'][0]
    assert pair['left'] == 'strong_case'
    assert pair['right'] == 'weak_case'
    assert pair['overall_score_delta'] == 18.0
    assert pair['component_score_deltas']['energy_arc'] == 31.0
    assert pair['component_score_deltas']['song_likeness'] == 36.0
    assert 'Harness: PASS' in result.stdout


def test_listen_gate_benchmark_fails_when_pairwise_delta_exceeds_maximum(tmp_path: Path):
    left = _write_json(
        tmp_path / 'left.json',
        _listen_report('left', 80.0, 'promising', 'pass', energy_arc=82.0, groove=75.0, song_likeness=81.0),
    )
    right = _write_json(
        tmp_path / 'right.json',
        _listen_report('right', 73.0, 'mixed', 'review', energy_arc=76.0, groove=70.0, song_likeness=74.0),
    )
    spec = {
        'expected_order': ['left_case', 'right_case'],
        'cases': [
            {
                'label': 'left_case',
                'input': str(left),
                'expect': {
                    'better_than': [
                        {
                            'other': 'right_case',
                            'overall_score_delta_at_least': 5.0,
                            'overall_score_delta_at_most': 6.0,
                            'component_score_delta_at_least': {'energy_arc': 5.0},
                            'component_score_delta_at_most': {'groove': 4.0},
                        }
                    ],
                },
            },
            {
                'label': 'right_case',
                'input': str(right),
            },
        ],
    }
    spec_path = _write_json(tmp_path / 'delta_band_spec.json', spec)
    output = tmp_path / 'delta_band_harness.json'

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(spec_path), '--output', str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['passed'] is False
    reasons = [row['reason'] for row in payload['failures']['pairwise_failures']]
    assert any('overall score delta <=' in reason for reason in reasons)
    assert any('groove delta <=' in reason for reason in reasons)
    assert 'FAIL left_case vs right_case' in result.stdout


def test_listen_gate_benchmark_fails_when_expected_human_order_regresses(tmp_path: Path):
    supposed_good = _write_json(
        tmp_path / 'supposed_good.json',
        _listen_report('supposed_good', 74.0, 'mixed', 'review', energy_arc=70.0, groove=73.0, song_likeness=72.0),
    )
    supposed_bad = _write_json(
        tmp_path / 'supposed_bad.json',
        _listen_report('supposed_bad', 79.0, 'promising', 'pass', energy_arc=82.0, groove=78.0, song_likeness=80.0),
    )
    spec = {
        'expected_order': ['supposed_good_case', 'supposed_bad_case'],
        'cases': [
            {
                'label': 'supposed_good_case',
                'input': str(supposed_good),
                'expect': {
                    'gating_status': 'pass',
                    'better_than': [
                        {'other': 'supposed_bad_case', 'overall_score_delta_at_least': 1.0}
                    ],
                },
            },
            {
                'label': 'supposed_bad_case',
                'input': str(supposed_bad),
                'expect': {'overall_score_at_most': 75.0},
            },
        ],
    }
    spec_path = _write_json(tmp_path / 'failing_spec.json', spec)
    output = tmp_path / 'failing_harness.json'

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(spec_path), '--output', str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['passed'] is False
    assert payload['failures']['pairwise_failures']
    assert payload['failures']['order_failures']
    assert payload['failures']['case_failures']['supposed_good_case'][0].startswith('gating_status')
    assert payload['failures']['case_failures']['supposed_bad_case'][0].startswith('overall_score')
    assert 'FAIL supposed_good_case vs supposed_bad_case' in result.stdout


def test_listen_gate_benchmark_checks_nested_payoff_conviction_metrics(tmp_path: Path):
    strong = _write_json(
        tmp_path / 'strong.json',
        _listen_report(
            'strong',
            84.0,
            'promising',
            'pass',
            energy_arc=86.0,
            groove=79.0,
            song_likeness=82.0,
            aggregate_metrics={
                'planner_audio_climax_conviction': 0.78,
                'climax_conviction': 0.81,
            },
        ),
    )
    weak = _write_json(
        tmp_path / 'weak.json',
        _listen_report(
            'weak',
            72.0,
            'mixed',
            'review',
            energy_arc=62.0,
            groove=70.0,
            song_likeness=69.0,
            aggregate_metrics={
                'planner_audio_climax_conviction': 0.41,
                'climax_conviction': 0.39,
            },
        ),
    )
    spec = {
        'expected_order': ['strong_case', 'weak_case'],
        'cases': [
            {
                'label': 'strong_case',
                'input': str(strong),
                'expect': {
                    'metric_at_least': {
                        'song_likeness.details.aggregate_metrics.planner_audio_climax_conviction': 0.75,
                        'song_likeness.details.aggregate_metrics.climax_conviction': 0.80,
                    },
                },
            },
            {
                'label': 'weak_case',
                'input': str(weak),
                'expect': {
                    'metric_at_most': {
                        'song_likeness.details.aggregate_metrics.planner_audio_climax_conviction': 0.45,
                        'song_likeness.details.aggregate_metrics.climax_conviction': 0.45,
                    },
                },
            },
        ],
    }
    spec_path = _write_json(tmp_path / 'metric_spec.json', spec)
    output = tmp_path / 'metric_harness.json'

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(spec_path), '--output', str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['passed'] is True
    assert payload['failures']['case_failures'] == {}
