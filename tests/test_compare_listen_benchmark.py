from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ai_dj


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    overall_score: float
    component_scores: dict[str, float]
    verdict: str


def _listen_report(case: BenchmarkCase) -> dict:
    groove_details = {
        'beat_stability': round(case.component_scores['groove'] / 100.0, 3),
        'pocket_stability': round(max(case.component_scores['groove'] - 6.0, 0.0) / 100.0, 3),
        'collapse_severity': round(max(100.0 - case.component_scores['groove'], 0.0) / 100.0, 3),
    }
    return {
        'source_path': f'{case.name}.wav',
        'duration_seconds': 60.0,
        'overall_score': case.overall_score,
        'structure': {'score': case.component_scores['structure'], 'summary': case.name, 'evidence': [], 'fixes': [], 'details': {}},
        'groove': {'score': case.component_scores['groove'], 'summary': case.name, 'evidence': [], 'fixes': [], 'details': groove_details},
        'energy_arc': {'score': case.component_scores['energy_arc'], 'summary': case.name, 'evidence': [], 'fixes': [], 'details': {}},
        'transition': {'score': case.component_scores['transition'], 'summary': case.name, 'evidence': [], 'fixes': [], 'details': {}},
        'coherence': {'score': case.component_scores['coherence'], 'summary': case.name, 'evidence': [], 'fixes': [], 'details': {}},
        'mix_sanity': {'score': case.component_scores['mix_sanity'], 'summary': case.name, 'evidence': [], 'fixes': [], 'details': {}},
        'song_likeness': {'score': case.component_scores.get('song_likeness', case.overall_score), 'summary': case.name, 'evidence': [], 'fixes': [], 'details': {}},
        'verdict': case.verdict,
        'top_reasons': [],
        'top_fixes': [],
        'gating': {'status': 'pass', 'raw_overall_score': case.overall_score},
        'analysis_version': '0.5.0',
    }


def _write_case(tmp_path: Path, case: BenchmarkCase) -> Path:
    path = tmp_path / f'{case.name}.json'
    path.write_text(json.dumps(_listen_report(case)), encoding='utf-8')
    return path


def _round_robin_rank(case_paths: dict[str, Path]) -> list[dict[str, float | int | str]]:
    scoreboard = {
        name: {
            'name': name,
            'wins': 0,
            'ties': 0,
            'losses': 0,
            'net_score_delta': 0.0,
        }
        for name in case_paths
    }

    names = list(case_paths)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            comparison = ai_dj._build_listen_comparison(str(case_paths[left_name]), str(case_paths[right_name]))
            winner = comparison['winner']['overall']
            delta = float(comparison['deltas']['overall_score_delta'])

            scoreboard[left_name]['net_score_delta'] += delta
            scoreboard[right_name]['net_score_delta'] -= delta

            if winner == 'left':
                scoreboard[left_name]['wins'] += 1
                scoreboard[right_name]['losses'] += 1
            elif winner == 'right':
                scoreboard[right_name]['wins'] += 1
                scoreboard[left_name]['losses'] += 1
            else:
                scoreboard[left_name]['ties'] += 1
                scoreboard[right_name]['ties'] += 1

    return sorted(
        scoreboard.values(),
        key=lambda row: (-int(row['wins']), -float(row['net_score_delta']), row['name']),
    )


def test_compare_listen_round_robin_benchmark_ranks_stronger_reports_higher(tmp_path: Path):
    benchmark_cases = [
        BenchmarkCase(
            name='producer_grade',
            overall_score=88.0,
            component_scores={
                'structure': 89.0,
                'groove': 87.0,
                'energy_arc': 90.0,
                'transition': 86.0,
                'coherence': 88.0,
                'mix_sanity': 89.0,
            },
            verdict='promising',
        ),
        BenchmarkCase(
            name='middling',
            overall_score=78.0,
            component_scores={
                'structure': 79.0,
                'groove': 78.0,
                'energy_arc': 76.0,
                'transition': 77.0,
                'coherence': 79.0,
                'mix_sanity': 80.0,
            },
            verdict='mixed',
        ),
        BenchmarkCase(
            name='flashy_but_messy',
            overall_score=73.0,
            component_scores={
                'structure': 72.0,
                'groove': 81.0,
                'energy_arc': 84.0,
                'transition': 61.0,
                'coherence': 70.0,
                'mix_sanity': 68.0,
            },
            verdict='mixed',
        ),
        BenchmarkCase(
            name='weak',
            overall_score=66.0,
            component_scores={
                'structure': 68.0,
                'groove': 65.0,
                'energy_arc': 60.0,
                'transition': 64.0,
                'coherence': 67.0,
                'mix_sanity': 69.0,
            },
            verdict='weak',
        ),
    ]

    case_paths = {case.name: _write_case(tmp_path, case) for case in benchmark_cases}

    ranking = _round_robin_rank(case_paths)

    assert [row['name'] for row in ranking] == ['producer_grade', 'middling', 'flashy_but_messy', 'weak']
    assert ranking[0]['wins'] == 3
    assert ranking[1]['wins'] == 2
    assert ranking[2]['wins'] == 1
    assert ranking[3]['wins'] == 0
    assert ranking[0]['net_score_delta'] == 47.0
    assert ranking[1]['net_score_delta'] == 7.0
    assert ranking[2]['net_score_delta'] == -13.0
    assert ranking[3]['net_score_delta'] == -41.0


def test_compare_listen_profile_match_exposes_groove_similarity_diagnostics(tmp_path: Path):
    left = BenchmarkCase(
        name='left_groove',
        overall_score=82.0,
        component_scores={
            'structure': 80.0,
            'groove': 84.0,
            'energy_arc': 81.0,
            'transition': 79.0,
            'coherence': 80.0,
            'mix_sanity': 78.0,
        },
        verdict='promising',
    )
    right = BenchmarkCase(
        name='right_groove',
        overall_score=79.0,
        component_scores={
            'structure': 79.0,
            'groove': 72.0,
            'energy_arc': 78.0,
            'transition': 77.0,
            'coherence': 79.0,
            'mix_sanity': 77.0,
        },
        verdict='mixed',
    )

    comparison = ai_dj._build_listen_comparison(str(_write_case(tmp_path, left)), str(_write_case(tmp_path, right)))

    groove_match = comparison['diagnostics']['groove_profile_match']
    assert groove_match['shared_metric_count'] == 3
    assert groove_match['similarity'] is not None
    assert groove_match['largest_gaps'][0]['metric'] in {'beat_stability', 'pocket_stability', 'collapse_severity'}
    assert groove_match['largest_gaps'][0]['normalized_gap'] > 0.0


def test_compare_listen_profile_match_exposes_energy_similarity_diagnostics_for_nested_metrics(tmp_path: Path):
    left_payload = _listen_report(BenchmarkCase(
        name='left_energy',
        overall_score=83.0,
        component_scores={
            'structure': 80.0,
            'groove': 82.0,
            'energy_arc': 86.0,
            'transition': 80.0,
            'coherence': 81.0,
            'mix_sanity': 79.0,
        },
        verdict='promising',
    ))
    right_payload = _listen_report(BenchmarkCase(
        name='right_energy',
        overall_score=77.0,
        component_scores={
            'structure': 78.0,
            'groove': 79.0,
            'energy_arc': 68.0,
            'transition': 76.0,
            'coherence': 78.0,
            'mix_sanity': 77.0,
        },
        verdict='mixed',
    ))
    left_payload['energy_arc']['details'] = {
        'macro_profile': {'late_peak_ratio': 0.84, 'payoff_contrast': 0.77},
        'dynamic_range_rms': 0.42,
    }
    right_payload['energy_arc']['details'] = {
        'macro_profile': {'late_peak_ratio': 0.41, 'payoff_contrast': 0.39},
        'dynamic_range_rms': 0.18,
    }

    left_path = tmp_path / 'left_energy.json'
    right_path = tmp_path / 'right_energy.json'
    left_path.write_text(json.dumps(left_payload), encoding='utf-8')
    right_path.write_text(json.dumps(right_payload), encoding='utf-8')

    comparison = ai_dj._build_listen_comparison(str(left_path), str(right_path))

    energy_match = comparison['diagnostics']['energy_profile_match']
    assert energy_match['shared_metric_count'] == 3
    assert energy_match['similarity'] is not None
    assert energy_match['largest_gaps'][0]['metric'] in {'macro_profile.late_peak_ratio', 'macro_profile.payoff_contrast', 'dynamic_range_rms'}
    assert energy_match['largest_gaps'][0]['normalized_gap'] > 0.0


def test_compare_listen_benchmark_exposes_component_reason_for_a_win(tmp_path: Path):
    strong_arc = BenchmarkCase(
        name='strong_arc',
        overall_score=81.0,
        component_scores={
            'structure': 80.0,
            'groove': 79.0,
            'energy_arc': 90.0,
            'transition': 80.0,
            'coherence': 79.0,
            'mix_sanity': 78.0,
        },
        verdict='promising',
    )
    flat_arc = BenchmarkCase(
        name='flat_arc',
        overall_score=74.0,
        component_scores={
            'structure': 79.0,
            'groove': 79.0,
            'energy_arc': 65.0,
            'transition': 80.0,
            'coherence': 79.0,
            'mix_sanity': 78.0,
        },
        verdict='mixed',
    )

    comparison = ai_dj._build_listen_comparison(
        str(_write_case(tmp_path, strong_arc)),
        str(_write_case(tmp_path, flat_arc)),
    )

    assert comparison['winner']['overall'] == 'left'
    assert comparison['winner']['components']['energy_arc'] == 'left'
    assert comparison['deltas']['component_score_deltas']['energy_arc'] == 25.0
    assert any('energy arc' in line.lower() for line in comparison['summary'])


def test_build_listen_benchmark_handles_duplicate_basenames_without_label_collisions(tmp_path: Path):
    stronger = BenchmarkCase(
        name='fusion_rerun',
        overall_score=84.0,
        component_scores={
            'structure': 85.0,
            'groove': 84.0,
            'energy_arc': 86.0,
            'transition': 82.0,
            'coherence': 83.0,
            'mix_sanity': 82.0,
        },
        verdict='promising',
    )
    weaker = BenchmarkCase(
        name='fusion_rerun',
        overall_score=71.0,
        component_scores={
            'structure': 72.0,
            'groove': 71.0,
            'energy_arc': 68.0,
            'transition': 70.0,
            'coherence': 72.0,
            'mix_sanity': 73.0,
        },
        verdict='mixed',
    )

    pass_a = tmp_path / 'pass_a'
    pass_b = tmp_path / 'pass_b'
    pass_a.mkdir()
    pass_b.mkdir()
    left_path = pass_a / 'fusion_rerun.json'
    right_path = pass_b / 'fusion_rerun.json'
    left_path.write_text(json.dumps(_listen_report(stronger)), encoding='utf-8')
    right_path.write_text(json.dumps(_listen_report(weaker)), encoding='utf-8')

    benchmark = ai_dj._build_listen_benchmark([str(left_path), str(right_path)])

    assert benchmark['winner'] == 'pass_a/fusion_rerun.json'
    assert [row['label'] for row in benchmark['ranking']] == ['pass_a/fusion_rerun.json', 'pass_b/fusion_rerun.json']
    assert benchmark['comparisons'][0]['left'] == 'pass_a/fusion_rerun.json'
    assert benchmark['comparisons'][0]['right'] == 'pass_b/fusion_rerun.json'
    assert benchmark['case_index'][0]['input_label'] == 'fusion_rerun.json'


def test_build_listen_benchmark_returns_ranked_scoreboard(tmp_path: Path):
    producer = BenchmarkCase(
        name='producer_grade',
        overall_score=88.0,
        component_scores={
            'structure': 89.0,
            'groove': 87.0,
            'energy_arc': 90.0,
            'transition': 86.0,
            'coherence': 88.0,
            'mix_sanity': 89.0,
        },
        verdict='promising',
    )
    weak = BenchmarkCase(
        name='weak',
        overall_score=66.0,
        component_scores={
            'structure': 68.0,
            'groove': 65.0,
            'energy_arc': 60.0,
            'transition': 64.0,
            'coherence': 67.0,
            'mix_sanity': 69.0,
        },
        verdict='weak',
    )
    middling = BenchmarkCase(
        name='middling',
        overall_score=78.0,
        component_scores={
            'structure': 79.0,
            'groove': 78.0,
            'energy_arc': 76.0,
            'transition': 77.0,
            'coherence': 79.0,
            'mix_sanity': 80.0,
        },
        verdict='mixed',
    )

    benchmark = ai_dj._build_listen_benchmark(
        [
            str(_write_case(tmp_path, weak)),
            str(_write_case(tmp_path, producer)),
            str(_write_case(tmp_path, middling)),
        ]
    )

    assert benchmark['winner'] == 'producer_grade.json'
    assert [row['label'] for row in benchmark['ranking']] == ['producer_grade.json', 'middling.json', 'weak.json']
    assert len(benchmark['case_index']) == 3
    top = benchmark['ranking'][0]
    assert top['case_id']
    assert top['wins'] == 2
    assert top['losses'] == 0
    assert top['net_score_delta'] == 32.0
    assert top['strengths'][0]['component'] == 'energy_arc'
    assert top['pairwise']['middling.json']['winner'] == 'left'
    assert top['pairwise']['weak.json']['overall_score_delta'] == 22.0
    assert len(benchmark['comparisons']) == 3
    assert benchmark['comparisons'][0]['comparison_id']
    assert benchmark['comparisons'][0]['diagnostics']['biggest_swing']['component']


def test_benchmark_listen_writes_json_artifact(tmp_path: Path):
    left = BenchmarkCase(
        name='left',
        overall_score=82.0,
        component_scores={
            'structure': 83.0,
            'groove': 82.0,
            'energy_arc': 84.0,
            'transition': 81.0,
            'coherence': 82.0,
            'mix_sanity': 80.0,
        },
        verdict='promising',
    )
    right = BenchmarkCase(
        name='right',
        overall_score=74.0,
        component_scores={
            'structure': 75.0,
            'groove': 74.0,
            'energy_arc': 73.0,
            'transition': 74.0,
            'coherence': 75.0,
            'mix_sanity': 73.0,
        },
        verdict='mixed',
    )

    output = tmp_path / 'benchmark.json'
    rc = ai_dj.benchmark_listen(
        [str(_write_case(tmp_path, left)), str(_write_case(tmp_path, right))],
        str(output),
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['winner'] == 'left.json'
    assert payload['ranking'][0]['label'] == 'left.json'
    assert payload['ranking'][1]['label'] == 'right.json'
    assert payload['comparisons'][0]['diagnostics']['ranked_component_swings'][0]['winner'] == 'left'
    assert payload['ranking'][1]['weaknesses'][0]['component']
