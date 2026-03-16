from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'build_listen_gate_spec.py'


def test_build_listen_gate_spec_writes_expected_payload(tmp_path: Path):
    strong = tmp_path / 'strong.json'
    weak = tmp_path / 'weak.json'
    strong.write_text('{}', encoding='utf-8')
    weak.write_text('{}', encoding='utf-8')
    output = tmp_path / 'spec.json'

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--case', f'strong_case={strong}',
            '--case', f'weak_case={weak}',
            '--expected-order', 'strong_case,weak_case',
            '--expect-gating', 'strong_case:gating_status=pass',
            '--expect-gating', 'weak_case:gating_status=reject',
            '--overall-at-least', 'strong_case:overall_score_at_least=80',
            '--overall-at-most', 'weak_case:overall_score_at_most=70',
            '--component-at-least', 'strong_case:energy_arc=90',
            '--component-at-least', 'strong_case:song_likeness=85',
            '--metric-at-least', 'strong_case:song_likeness.details.aggregate_metrics.readable_section_ratio=0.72',
            '--metric-at-least', 'strong_case:song_likeness.details.aggregate_metrics.climax_conviction=0.70',
            '--metric-at-most', 'weak_case:song_likeness.details.aggregate_metrics.readable_section_ratio=0.45',
            '--metric-at-most', 'weak_case:song_likeness.details.aggregate_metrics.climax_conviction=0.45',
            '--better-than', 'strong_case>weak_case:overall=10:overall-max=25:component=energy_arc=20,song_likeness=25:component-max=groove=30',
            '--output', str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['expected_order'] == ['strong_case', 'weak_case']
    assert [case['label'] for case in payload['cases']] == ['strong_case', 'weak_case']
    strong_case = payload['cases'][0]
    weak_case = payload['cases'][1]
    assert strong_case['expect']['gating_status'] == 'pass'
    assert strong_case['expect']['overall_score_at_least'] == 80.0
    assert strong_case['expect']['component_score_at_least']['energy_arc'] == 90.0
    assert strong_case['expect']['component_score_at_least']['song_likeness'] == 85.0
    assert strong_case['expect']['metric_at_least']['song_likeness.details.aggregate_metrics.readable_section_ratio'] == 0.72
    assert strong_case['expect']['metric_at_least']['song_likeness.details.aggregate_metrics.climax_conviction'] == 0.70
    assert weak_case['expect']['metric_at_most']['song_likeness.details.aggregate_metrics.readable_section_ratio'] == 0.45
    assert weak_case['expect']['metric_at_most']['song_likeness.details.aggregate_metrics.climax_conviction'] == 0.45
    assert strong_case['expect']['better_than'][0]['other'] == 'weak_case'
    assert strong_case['expect']['better_than'][0]['overall_score_delta_at_least'] == 10.0
    assert strong_case['expect']['better_than'][0]['overall_score_delta_at_most'] == 25.0
    assert strong_case['expect']['better_than'][0]['component_score_delta_at_least']['energy_arc'] == 20.0
    assert strong_case['expect']['better_than'][0]['component_score_delta_at_least']['song_likeness'] == 25.0
    assert strong_case['expect']['better_than'][0]['component_score_delta_at_most']['groove'] == 30.0
    assert weak_case['expect']['gating_status'] == 'reject'
    assert weak_case['expect']['overall_score_at_most'] == 70.0
    assert 'Wrote benchmark spec:' in result.stdout


def test_build_listen_gate_spec_rejects_invalid_expected_order(tmp_path: Path):
    left = tmp_path / 'left.json'
    right = tmp_path / 'right.json'
    left.write_text('{}', encoding='utf-8')
    right.write_text('{}', encoding='utf-8')

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--case', f'left_case={left}',
            '--case', f'right_case={right}',
            '--expected-order', 'left_case',
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert 'must include each case label exactly once' in result.stderr


def test_build_listen_gate_spec_supports_curated_good_and_bad_cases(tmp_path: Path):
    good = tmp_path / 'good.json'
    neutral = tmp_path / 'neutral.json'
    bad = tmp_path / 'bad.json'
    good.write_text('{}', encoding='utf-8')
    neutral.write_text('{}', encoding='utf-8')
    bad.write_text('{}', encoding='utf-8')
    output = tmp_path / 'tiered_spec.json'

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--good-case', f'good_case={good}',
            '--case', f'neutral_case={neutral}',
            '--bad-case', f'bad_case={bad}',
            '--output', str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['expected_order'] == ['good_case', 'neutral_case', 'bad_case']
    assert [case['label'] for case in payload['cases']] == ['good_case', 'neutral_case', 'bad_case']
    assert payload['cases'][0]['curation_tier'] == 'good'
    assert payload['cases'][2]['curation_tier'] == 'bad'
    assert payload['cases'][0]['expect']['gating_status'] == 'pass'
    assert 'expect' not in payload['cases'][1]
    assert payload['cases'][2]['expect']['gating_status'] == 'reject'



def test_build_listen_gate_spec_supports_curated_review_cases(tmp_path: Path):
    good = tmp_path / 'good.json'
    neutral = tmp_path / 'neutral.json'
    review = tmp_path / 'review.json'
    bad = tmp_path / 'bad.json'
    good.write_text('{}', encoding='utf-8')
    neutral.write_text('{}', encoding='utf-8')
    review.write_text('{}', encoding='utf-8')
    bad.write_text('{}', encoding='utf-8')
    output = tmp_path / 'review_tiered_spec.json'

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--good-case', f'good_case={good}',
            '--case', f'neutral_case={neutral}',
            '--review-case', f'review_case={review}',
            '--bad-case', f'bad_case={bad}',
            '--output', str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['expected_order'] == ['good_case', 'neutral_case', 'review_case', 'bad_case']
    assert [case['label'] for case in payload['cases']] == ['good_case', 'neutral_case', 'review_case', 'bad_case']
    assert payload['cases'][2]['curation_tier'] == 'review'
    assert payload['cases'][2]['expect']['gating_status'] == 'review'
    assert payload['cases'][3]['expect']['gating_status'] == 'reject'



def test_build_listen_gate_spec_allows_explicit_gating_override_for_curated_case(tmp_path: Path):
    good = tmp_path / 'good.json'
    bad = tmp_path / 'bad.json'
    good.write_text('{}', encoding='utf-8')
    bad.write_text('{}', encoding='utf-8')
    output = tmp_path / 'override_spec.json'

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--good-case', f'good_case={good}',
            '--bad-case', f'bad_case={bad}',
            '--expect-gating', 'good_case:gating_status=review',
            '--output', str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['cases'][0]['expect']['gating_status'] == 'review'
    assert payload['cases'][1]['expect']['gating_status'] == 'reject'
