from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.critic_benchmark_harness import build_fixture_spec, render_markdown, run_sprint_harness


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'critic_benchmark_harness.py'


def _listen_report(case_name: str, overall: float, verdict: str, gating: str, *, energy_arc: float, groove: float, song_likeness: float) -> dict:
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
        'song_likeness': {'score': song_likeness, 'summary': case_name, 'evidence': [], 'fixes': [], 'details': {}},
        'verdict': verdict,
        'top_reasons': [],
        'top_fixes': [],
        'gating': {'status': gating, 'raw_overall_score': overall},
        'analysis_version': '0.5.0',
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_build_fixture_spec_adds_default_monotonic_rules(tmp_path: Path):
    fixture = {
        'label': 'pair_one',
        'cases': {
            'baseline': str(tmp_path / 'baseline.json'),
            'adaptive': str(tmp_path / 'adaptive.json'),
            'critic': str(tmp_path / 'critic.json'),
        },
        'expectations': {
            'critic': {'gating_status': 'pass'},
        },
    }

    spec = build_fixture_spec(fixture)

    assert spec['expected_order'] == ['critic', 'adaptive', 'baseline']
    critic_case = next(case for case in spec['cases'] if case['label'] == 'critic')
    adaptive_case = next(case for case in spec['cases'] if case['label'] == 'adaptive')
    assert critic_case['expect']['better_than'] == [{'other': 'adaptive'}]
    assert adaptive_case['expect']['better_than'] == [{'other': 'baseline'}]



def test_run_sprint_harness_aggregates_fixture_rankings_and_lane_scoreboard(tmp_path: Path):
    fixture_one_dir = tmp_path / 'fixture_one'
    fixture_two_dir = tmp_path / 'fixture_two'
    fixture_one_dir.mkdir()
    fixture_two_dir.mkdir()

    config = {
        'fixtures': [
            {
                'label': 'pair_one',
                'pair': {'song_a': 'a1.wav', 'song_b': 'b1.wav'},
                'cases': {
                    'baseline': str(_write_json(fixture_one_dir / 'baseline.json', _listen_report('baseline1', 70.0, 'mixed', 'review', energy_arc=71.0, groove=68.0, song_likeness=66.0))),
                    'adaptive': str(_write_json(fixture_one_dir / 'adaptive.json', _listen_report('adaptive1', 79.0, 'promising', 'pass', energy_arc=80.0, groove=77.0, song_likeness=76.0))),
                    'critic': str(_write_json(fixture_one_dir / 'critic.json', _listen_report('critic1', 85.0, 'promising', 'pass', energy_arc=87.0, groove=82.0, song_likeness=84.0))),
                },
                'expectations': {
                    'critic': {'gating_status': 'pass', 'overall_score_at_least': 80.0},
                    'adaptive': {'gating_status': 'pass'},
                    'baseline': {'overall_score_at_most': 75.0},
                },
            },
            {
                'label': 'pair_two',
                'pair': {'song_a': 'a2.wav', 'song_b': 'b2.wav'},
                'cases': {
                    'baseline': str(_write_json(fixture_two_dir / 'baseline.json', _listen_report('baseline2', 68.0, 'mixed', 'review', energy_arc=69.0, groove=67.0, song_likeness=64.0))),
                    'adaptive': str(_write_json(fixture_two_dir / 'adaptive.json', _listen_report('adaptive2', 77.0, 'promising', 'pass', energy_arc=78.0, groove=75.0, song_likeness=74.0))),
                    'critic': str(_write_json(fixture_two_dir / 'critic.json', _listen_report('critic2', 83.0, 'promising', 'pass', energy_arc=84.0, groove=80.0, song_likeness=82.0))),
                },
                'expectations': {
                    'critic': {'gating_status': 'pass', 'overall_score_at_least': 80.0},
                    'adaptive': {'gating_status': 'pass'},
                    'baseline': {'overall_score_at_most': 72.0},
                },
            },
        ]
    }
    config_path = _write_json(tmp_path / 'critic_harness_config.json', config)

    report = run_sprint_harness(str(config_path), spec_output_dir=str(tmp_path / 'generated_specs'))

    assert report['passed'] is True
    assert report['aggregate']['fixture_count'] == 2
    assert report['aggregate']['passed_fixtures'] == 2
    assert [fixture['ranking'] for fixture in report['fixtures']] == [
        ['critic', 'adaptive', 'baseline'],
        ['critic', 'adaptive', 'baseline'],
    ]
    critic_row = report['aggregate']['lane_scoreboard']['critic']
    adaptive_row = report['aggregate']['lane_scoreboard']['adaptive']
    baseline_row = report['aggregate']['lane_scoreboard']['baseline']
    assert critic_row['wins_first'] == 2
    assert critic_row['average_rank'] == 1.0
    assert critic_row['passes'] == 2
    assert adaptive_row['average_rank'] == 2.0
    assert baseline_row['average_rank'] == 3.0
    assert baseline_row['gating_status_counts']['review'] == 2
    assert (tmp_path / 'generated_specs' / 'pair_one_benchmark_spec.json').exists()
    markdown = render_markdown(report)
    assert 'Critic Sprint Benchmark Harness' in markdown
    assert '| critic | 1.0 | 2 | 2 | 0 | 84.0 | pass:2 | promising:2 |' in markdown



def test_critic_benchmark_harness_cli_returns_failure_when_fixture_order_regresses(tmp_path: Path):
    fixture_dir = tmp_path / 'fixture_fail'
    fixture_dir.mkdir()
    config = {
        'fixtures': [
            {
                'label': 'regressed_pair',
                'cases': {
                    'baseline': str(_write_json(fixture_dir / 'baseline.json', _listen_report('baseline', 69.0, 'mixed', 'review', energy_arc=70.0, groove=68.0, song_likeness=66.0))),
                    'adaptive': str(_write_json(fixture_dir / 'adaptive.json', _listen_report('adaptive', 83.0, 'promising', 'pass', energy_arc=84.0, groove=81.0, song_likeness=82.0))),
                    'critic': str(_write_json(fixture_dir / 'critic.json', _listen_report('critic', 78.0, 'mixed', 'review', energy_arc=79.0, groove=76.0, song_likeness=75.0))),
                },
                'expectations': {
                    'adaptive': {'gating_status': 'pass'},
                    'critic': {'gating_status': 'pass'},
                },
            }
        ]
    }
    config_path = _write_json(tmp_path / 'regressed_config.json', config)
    output = tmp_path / 'critic_report.json'

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(config_path), '--output', str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['passed'] is False
    assert payload['fixtures'][0]['ranking'] == ['adaptive', 'critic', 'baseline']
    assert 'Critic benchmark harness: FAIL' in result.stdout
