from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.human_feedback import HumanFeedbackStore


def build_feedback_learning_summary(feedback_root: Path, *, limit: int = 5000) -> dict[str, Any]:
    store = HumanFeedbackStore(feedback_root)
    events = store.list_events(limit=limit)
    render_events = [event for event in events if event.get('type') == 'render']
    pairwise_events = [event for event in events if event.get('type') == 'pairwise']

    tag_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    timestamped_moments: list[dict[str, Any]] = []
    for event in render_events:
        decision = str(event.get('overall_label') or 'unknown')
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        for tag in event.get('tags') or []:
            key = str(tag).strip()
            if not key:
                continue
            tag_counts[key] = tag_counts.get(key, 0) + 1
        if event.get('timestamp_sec') is not None:
            timestamped_moments.append(
                {
                    'run_dir': event.get('run_dir'),
                    'timestamp_sec': event.get('timestamp_sec'),
                    'decision': decision,
                    'tags': list(event.get('tags') or []),
                    'note': event.get('note') or '',
                }
            )

    pairwise_winner_counts = {'left': 0, 'right': 0, 'tie': 0}
    for event in pairwise_events:
        winner = str(event.get('winner') or 'tie')
        if winner not in pairwise_winner_counts:
            pairwise_winner_counts[winner] = 0
        pairwise_winner_counts[winner] += 1
        for tag in event.get('tags') or []:
            key = str(tag).strip()
            if not key:
                continue
            tag_counts[key] = tag_counts.get(key, 0) + 1

    top_negative_tags = sorted(
        ((tag, count) for tag, count in tag_counts.items() if tag not in {'good backbone', 'promising', 'favorite'}),
        key=lambda item: (-item[1], item[0]),
    )[:12]
    top_positive_tags = sorted(
        ((tag, count) for tag, count in tag_counts.items() if tag in {'good backbone', 'promising', 'favorite', 'best so far'}),
        key=lambda item: (-item[1], item[0]),
    )[:12]

    derived_priors = {
        'medley_rejection_pressure': round(min(1.0, tag_counts.get('not one song', 0) / max(len(render_events), 1)), 3),
        'groove_rejection_pressure': round(min(1.0, tag_counts.get('bad groove', 0) / max(len(render_events), 1)), 3),
        'transition_rejection_pressure': round(min(1.0, tag_counts.get('track-switch seam', 0) / max(len(render_events), 1)), 3),
        'payoff_upgrade_pressure': round(min(1.0, tag_counts.get('weak payoff', 0) / max(len(render_events), 1)), 3),
        'backbone_reward_pressure': round(min(1.0, tag_counts.get('good backbone', 0) / max(len(render_events), 1)), 3),
    }
    return {
        'schema_version': '0.1.0',
        'summary': {
            'render_event_count': len(render_events),
            'pairwise_event_count': len(pairwise_events),
            'decision_counts': decision_counts,
            'pairwise_winner_counts': pairwise_winner_counts,
            'top_negative_tags': [{'tag': tag, 'count': count} for tag, count in top_negative_tags],
            'top_positive_tags': [{'tag': tag, 'count': count} for tag, count in top_positive_tags],
        },
        'derived_priors': derived_priors,
        'timestamped_moments': timestamped_moments[:128],
    }


def write_feedback_learning_summary(feedback_root: Path, output_path: Path, *, limit: int = 5000) -> dict[str, Any]:
    payload = build_feedback_learning_summary(feedback_root, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return payload
