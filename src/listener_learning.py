from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


RENDER_POSITIVE_LABELS = {
    "promising",
    "favorite",
    "best so far",
    "keep",
    "survivor",
    "strong",
}
RENDER_NEGATIVE_LABELS = {
    "reject",
    "weak",
    "poor",
    "bad",
    "borderline_reject",
}

LISTENER_FEATURES: tuple[str, ...] = (
    "overall_score",
    "structure",
    "groove",
    "energy_arc",
    "transition",
    "coherence",
    "mix_sanity",
    "song_likeness",
    "gate_pass",
    "gate_review",
    "gate_reject",
    "backbone_continuity",
    "recognizable_section_ratio",
    "boundary_recovery",
    "role_plausibility",
    "composite_song_risk",
    "background_only_identity_gap",
    "owner_switch_ratio",
    "avg_edge_cliff_risk",
    "avg_vocal_competition_risk",
    "manifest_switch_detector_risk",
    "crowding_burst_risk",
    "low_end_owner_stability_risk",
)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def listener_feature_map(report: dict[str, Any]) -> dict[str, float]:
    gating = report.get("gating") or {}
    gate_status = str(gating.get("status") or "").strip().lower()
    song_metrics = (((report.get("song_likeness") or {}).get("details") or {}).get("aggregate_metrics") or {})
    transition_metrics = (((report.get("transition") or {}).get("details") or {}).get("aggregate_metrics") or {})
    mix_aggregate = ((((report.get("mix_sanity") or {}).get("details") or {}).get("manifest_metrics") or {}).get("aggregate_metrics") or {})
    ownership_metrics = (((report.get("mix_sanity") or {}).get("details") or {}).get("ownership_clutter_metrics") or {})

    return {
        "overall_score": _safe_float(report.get("overall_score")),
        "structure": _safe_float(((report.get("structure") or {}).get("score"))),
        "groove": _safe_float(((report.get("groove") or {}).get("score"))),
        "energy_arc": _safe_float(((report.get("energy_arc") or {}).get("score"))),
        "transition": _safe_float(((report.get("transition") or {}).get("score"))),
        "coherence": _safe_float(((report.get("coherence") or {}).get("score"))),
        "mix_sanity": _safe_float(((report.get("mix_sanity") or {}).get("score"))),
        "song_likeness": _safe_float(((report.get("song_likeness") or {}).get("score"))),
        "gate_pass": 1.0 if gate_status == "pass" else 0.0,
        "gate_review": 1.0 if gate_status == "review" else 0.0,
        "gate_reject": 1.0 if gate_status == "reject" else 0.0,
        "backbone_continuity": _safe_float(song_metrics.get("backbone_continuity")),
        "recognizable_section_ratio": _safe_float(song_metrics.get("recognizable_section_ratio")),
        "boundary_recovery": _safe_float(song_metrics.get("boundary_recovery")),
        "role_plausibility": _safe_float(song_metrics.get("role_plausibility")),
        "composite_song_risk": _safe_float(song_metrics.get("composite_song_risk")),
        "background_only_identity_gap": _safe_float(song_metrics.get("background_only_identity_gap")),
        "owner_switch_ratio": _safe_float(song_metrics.get("owner_switch_ratio")),
        "avg_edge_cliff_risk": _safe_float(transition_metrics.get("avg_edge_cliff_risk")),
        "avg_vocal_competition_risk": _safe_float(transition_metrics.get("avg_vocal_competition_risk")),
        "manifest_switch_detector_risk": _safe_float(transition_metrics.get("manifest_switch_detector_risk")),
        "crowding_burst_risk": _safe_float(ownership_metrics.get("crowding_burst_risk")),
        "low_end_owner_stability_risk": _safe_float(mix_aggregate.get("low_end_owner_stability_risk")),
    }


def listener_feature_vector(report: dict[str, Any]) -> list[float]:
    features = listener_feature_map(report)
    return [features[name] for name in LISTENER_FEATURES]


def _listen_report_path_from_run_dir(run_dir: Any) -> Path | None:
    if not run_dir:
        return None
    root = Path(str(run_dir)).expanduser().resolve()
    candidates = [
        root / "listen_report.json",
        root / "render" / "listen_report.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_report_for_event(event: dict[str, Any], *, key: str) -> dict[str, Any] | None:
    report_path = None
    explicit = event.get(f"{key}_listen_report_path")
    if explicit:
        candidate = Path(str(explicit)).expanduser().resolve()
        if candidate.exists():
            report_path = candidate
    if report_path is None:
        report_path = _listen_report_path_from_run_dir(event.get(f"{key}_run_dir") if key in {"left", "right"} else event.get("run_dir"))
    if report_path is None:
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _render_event_label(event: dict[str, Any]) -> int | None:
    label = str(event.get("overall_label") or "").strip().lower()
    if label in RENDER_POSITIVE_LABELS:
        return 1
    if label in RENDER_NEGATIVE_LABELS:
        return 0
    return None


def _fit_logistic_classifier(rows: list[list[float]], labels: list[int]) -> dict[str, Any] | None:
    if len(rows) < 4 or len(set(labels)) < 2:
        return None
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    matrix = np.asarray(rows, dtype=float)
    targets = np.asarray(labels, dtype=int)
    model.fit(matrix, targets)
    probs = model.predict_proba(matrix)[:, 1]
    predictions = (probs >= 0.5).astype(int)
    accuracy = float((predictions == targets).mean())
    coefficients = {name: round(float(weight), 6) for name, weight in zip(LISTENER_FEATURES, model.coef_[0])}
    top_positive = sorted(coefficients.items(), key=lambda item: (-item[1], item[0]))[:6]
    top_negative = sorted(coefficients.items(), key=lambda item: (item[1], item[0]))[:6]
    return {
        "model": model,
        "training_examples": int(len(rows)),
        "positive_examples": int(sum(labels)),
        "negative_examples": int(len(labels) - sum(labels)),
        "train_accuracy": round(accuracy, 3),
        "intercept": round(float(model.intercept_[0]), 6),
        "coefficients": coefficients,
        "top_positive_features": [{"feature": name, "weight": round(weight, 3)} for name, weight in top_positive],
        "top_negative_features": [{"feature": name, "weight": round(weight, 3)} for name, weight in top_negative],
    }


def build_listener_learning_models(events: list[dict[str, Any]]) -> dict[str, Any]:
    render_rows: list[list[float]] = []
    render_labels: list[int] = []
    pairwise_rows: list[list[float]] = []
    pairwise_labels: list[int] = []

    for event in events:
        event_type = str(event.get("type") or "").strip().lower()
        if event_type == "render":
            label = _render_event_label(event)
            if label is None:
                continue
            report = _load_report_for_event(event, key="render")
            if not report:
                continue
            render_rows.append(listener_feature_vector(report))
            render_labels.append(label)
            continue

        if event_type != "pairwise":
            continue
        winner = str(event.get("winner") or "").strip().lower()
        if winner not in {"left", "right"}:
            continue
        left_report = _load_report_for_event(event, key="left")
        right_report = _load_report_for_event(event, key="right")
        if not left_report or not right_report:
            continue
        left_vec = np.asarray(listener_feature_vector(left_report), dtype=float)
        right_vec = np.asarray(listener_feature_vector(right_report), dtype=float)
        delta = (left_vec - right_vec).tolist()
        pairwise_rows.append(delta)
        pairwise_labels.append(1 if winner == "left" else 0)
        pairwise_rows.append((-1.0 * (left_vec - right_vec)).tolist())
        pairwise_labels.append(0 if winner == "left" else 1)

    render_model = _fit_logistic_classifier(render_rows, render_labels)
    pairwise_model = _fit_logistic_classifier(pairwise_rows, pairwise_labels)
    return {
        "render_acceptor": render_model,
        "pairwise_preference": pairwise_model,
    }


def summarize_listener_learning_models(models: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("render_acceptor", "pairwise_preference"):
        payload = models.get(key)
        if not payload:
            summary[key] = {"available": False}
            continue
        summary[key] = {
            "available": True,
            "training_examples": payload["training_examples"],
            "positive_examples": payload["positive_examples"],
            "negative_examples": payload["negative_examples"],
            "train_accuracy": payload["train_accuracy"],
            "intercept": payload["intercept"],
            "top_positive_features": payload["top_positive_features"],
            "top_negative_features": payload["top_negative_features"],
        }
    return summary


def score_report_with_listener_learning(report: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    features = listener_feature_vector(report)
    feature_map = listener_feature_map(report)
    render_payload = models.get("render_acceptor")
    pairwise_payload = models.get("pairwise_preference")

    result: dict[str, Any] = {
        "available": False,
        "features": {name: round(feature_map[name], 4) for name in LISTENER_FEATURES},
    }
    if render_payload:
        model = render_payload["model"]
        probability = float(model.predict_proba([features])[0][1])
        margin = float(model.decision_function([features])[0])
        result["available"] = True
        result["acceptance_probability"] = round(probability, 4)
        result["acceptance_margin"] = round(margin, 4)
    if pairwise_payload:
        model = pairwise_payload["model"]
        utility = float(model.decision_function([features])[0])
        preference_probability = float(model.predict_proba([features])[0][1])
        result["available"] = True
        result["preference_score"] = round(utility, 4)
        result["preference_probability"] = round(preference_probability, 4)
    return result
