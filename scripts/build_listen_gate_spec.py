#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class SpecBuildError(RuntimeError):
    pass


def _parse_case(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SpecBuildError(f"Invalid --case '{raw}'. Expected label=path")
    label, raw_path = raw.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise SpecBuildError(f"Invalid --case '{raw}'. Expected non-empty label and path")
    return label, str(Path(raw_path).expanduser().resolve())


def _ensure_case(cases: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    if label not in cases:
        raise SpecBuildError(f"Unknown case label '{label}' in expectation")
    return cases[label]


def _ensure_expect(case: dict[str, Any]) -> dict[str, Any]:
    expect = case.get("expect")
    if not isinstance(expect, dict):
        expect = {}
        case["expect"] = expect
    return expect


def _parse_scalar_expect(raw: str) -> tuple[str, str, str]:
    if ":" not in raw or "=" not in raw:
        raise SpecBuildError(f"Invalid expectation '{raw}'. Expected label:key=value")
    label, remainder = raw.split(":", 1)
    key, value = remainder.split("=", 1)
    label = label.strip()
    key = key.strip()
    value = value.strip()
    if not label or not key or not value:
        raise SpecBuildError(f"Invalid expectation '{raw}'. Expected label:key=value")
    return label, key, value


def _parse_component_expect(raw: str) -> tuple[str, str, float]:
    label, component, value = _parse_scalar_expect(raw)
    try:
        numeric = float(value)
    except ValueError as exc:
        raise SpecBuildError(f"Invalid numeric component expectation '{raw}'") from exc
    return label, component, numeric


def _parse_better_than(raw: str) -> tuple[str, dict[str, Any]]:
    parts = [part.strip() for part in raw.split(":") if part.strip()]
    if not parts or ">" not in parts[0]:
        raise SpecBuildError(
            f"Invalid --better-than '{raw}'. Expected left>right[:overall=NUM][:overall-max=NUM][:component=name=NUM,...][:component-max=name=NUM,...]"
        )
    left, right = [item.strip() for item in parts[0].split(">", 1)]
    if not left or not right:
        raise SpecBuildError(
            f"Invalid --better-than '{raw}'. Expected left>right[:overall=NUM][:overall-max=NUM][:component=name=NUM,...][:component-max=name=NUM,...]"
        )

    rule: dict[str, Any] = {"other": right}
    component_deltas: dict[str, float] = {}
    component_max_deltas: dict[str, float] = {}
    for part in parts[1:]:
        if part.startswith("overall="):
            try:
                rule["overall_score_delta_at_least"] = float(part.split("=", 1)[1])
            except ValueError as exc:
                raise SpecBuildError(f"Invalid overall delta in --better-than '{raw}'") from exc
            continue
        if part.startswith("overall-max="):
            try:
                rule["overall_score_delta_at_most"] = float(part.split("=", 1)[1])
            except ValueError as exc:
                raise SpecBuildError(f"Invalid overall-max delta in --better-than '{raw}'") from exc
            continue
        if part.startswith("component="):
            component_spec = part.split("=", 1)[1]
            target = component_deltas
        elif part.startswith("component-max="):
            component_spec = part.split("=", 1)[1]
            target = component_max_deltas
        else:
            raise SpecBuildError(f"Unknown --better-than clause '{part}' in '{raw}'")

        for item in component_spec.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SpecBuildError(f"Invalid component delta '{item}' in --better-than '{raw}'")
            component, value = [token.strip() for token in item.split("=", 1)]
            if not component:
                raise SpecBuildError(f"Invalid component delta '{item}' in --better-than '{raw}'")
            try:
                target[component] = float(value)
            except ValueError as exc:
                raise SpecBuildError(f"Invalid component delta '{item}' in --better-than '{raw}'") from exc

    if component_deltas:
        rule["component_score_delta_at_least"] = component_deltas
    if component_max_deltas:
        rule["component_score_delta_at_most"] = component_max_deltas
    return left, rule


def build_spec(
    *,
    cases_raw: list[str] | None = None,
    reference_cases_raw: list[str] | None = None,
    good_cases_raw: list[str] | None = None,
    review_cases_raw: list[str] | None = None,
    bad_cases_raw: list[str] | None = None,
    expected_order_raw: str | None = None,
    gating_expectations: list[str] | None = None,
    verdict_expectations: list[str] | None = None,
    overall_at_least: list[str] | None = None,
    overall_at_most: list[str] | None = None,
    component_at_least: list[str] | None = None,
    component_at_most: list[str] | None = None,
    metric_at_least: list[str] | None = None,
    metric_at_most: list[str] | None = None,
    better_than_raw: list[str] | None = None,
) -> dict[str, Any]:
    neutral_cases = list(cases_raw or [])
    reference_cases = list(reference_cases_raw or [])
    good_cases = list(good_cases_raw or [])
    review_cases = list(review_cases_raw or [])
    bad_cases = list(bad_cases_raw or [])
    combined_count = len(neutral_cases) + len(reference_cases) + len(good_cases) + len(review_cases) + len(bad_cases)
    if combined_count < 2:
        raise SpecBuildError("Provide at least two cases across --case/--reference-case/--good-case/--review-case/--bad-case")

    cases: dict[str, dict[str, Any]] = {}
    ordered_labels: list[str] = []

    def add_cases(raw_values: list[str], *, tier: str | None = None) -> None:
        for raw in raw_values:
            label, path = _parse_case(raw)
            if label in cases:
                raise SpecBuildError(f"Duplicate case label '{label}'")
            case = {"label": label, "input": path}
            if tier:
                case["curation_tier"] = tier
            cases[label] = case
            ordered_labels.append(label)

    add_cases(reference_cases, tier="reference")
    add_cases(good_cases, tier="good")
    add_cases(neutral_cases, tier=None)
    add_cases(review_cases, tier="review")
    add_cases(bad_cases, tier="bad")

    expected_order = ordered_labels
    if expected_order_raw:
        expected_order = [item.strip() for item in expected_order_raw.split(",") if item.strip()]
        if set(expected_order) != set(ordered_labels) or len(expected_order) != len(ordered_labels):
            raise SpecBuildError(
                "--expected-order must include each case label exactly once"
            )

    for label in ordered_labels:
        tier = cases[label].get("curation_tier")
        if tier in {"reference", "good"}:
            _ensure_expect(cases[label]).setdefault("gating_status", "pass")
        elif tier == "review":
            _ensure_expect(cases[label]).setdefault("gating_status", "review")
        elif tier == "bad":
            _ensure_expect(cases[label]).setdefault("gating_status", "reject")

    scalar_fields = [
        (gating_expectations or [], "gating_status", str),
        (verdict_expectations or [], "verdict", str),
        (overall_at_least or [], "overall_score_at_least", float),
        (overall_at_most or [], "overall_score_at_most", float),
    ]
    for raw_values, output_key, caster in scalar_fields:
        for raw in raw_values:
            label, _, value = _parse_scalar_expect(raw)
            case = _ensure_case(cases, label)
            expect = _ensure_expect(case)
            try:
                expect[output_key] = caster(value)
            except ValueError as exc:
                raise SpecBuildError(f"Invalid value for {output_key}: '{raw}'") from exc

    for raw in component_at_least or []:
        label, component, value = _parse_component_expect(raw)
        case = _ensure_case(cases, label)
        expect = _ensure_expect(case)
        bucket = expect.setdefault("component_score_at_least", {})
        bucket[component] = value

    for raw in component_at_most or []:
        label, component, value = _parse_component_expect(raw)
        case = _ensure_case(cases, label)
        expect = _ensure_expect(case)
        bucket = expect.setdefault("component_score_at_most", {})
        bucket[component] = value

    for raw in metric_at_least or []:
        label, metric_path, value = _parse_component_expect(raw)
        case = _ensure_case(cases, label)
        expect = _ensure_expect(case)
        bucket = expect.setdefault("metric_at_least", {})
        bucket[metric_path] = value

    for raw in metric_at_most or []:
        label, metric_path, value = _parse_component_expect(raw)
        case = _ensure_case(cases, label)
        expect = _ensure_expect(case)
        bucket = expect.setdefault("metric_at_most", {})
        bucket[metric_path] = value

    for raw in better_than_raw or []:
        left, rule = _parse_better_than(raw)
        _ensure_case(cases, left)
        _ensure_case(cases, str(rule["other"]))
        expect = _ensure_expect(cases[left])
        rules = expect.setdefault("better_than", [])
        rules.append(rule)

    payload = {
        "expected_order": expected_order,
        "cases": [cases[label] for label in ordered_labels],
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a listen_gate_benchmark JSON spec from CLI flags.")
    parser.add_argument("--case", action="append", default=[], help="label=path to a listen report or run dir; repeatable")
    parser.add_argument("--reference-case", action="append", default=[], help="label=path for a known-good reference anchor; defaults gating_status=pass and sorts ahead of all other cases")
    parser.add_argument("--good-case", action="append", default=[], help="label=path for a curated good fixture; defaults gating_status=pass and sorts after reference anchors but ahead of neutral/review/bad cases")
    parser.add_argument("--review-case", action="append", default=[], help="label=path for a curated borderline fixture; defaults gating_status=review and sorts after good/neutral but ahead of bad cases")
    parser.add_argument("--bad-case", action="append", default=[], help="label=path for a curated bad fixture; defaults gating_status=reject and sorts after good/neutral/review cases")
    parser.add_argument("--expected-order", help="Comma-separated case labels in strongest->weakest expected order")
    parser.add_argument("--expect-gating", action="append", default=[], help="label:gating_status=value")
    parser.add_argument("--expect-verdict", action="append", default=[], help="label:verdict=value")
    parser.add_argument("--overall-at-least", action="append", default=[], help="label:overall_score_at_least=value")
    parser.add_argument("--overall-at-most", action="append", default=[], help="label:overall_score_at_most=value")
    parser.add_argument("--component-at-least", action="append", default=[], help="label:component=value")
    parser.add_argument("--component-at-most", action="append", default=[], help="label:component=value")
    parser.add_argument("--metric-at-least", action="append", default=[], help="label:dot.path.to.metric=value")
    parser.add_argument("--metric-at-most", action="append", default=[], help="label:dot.path.to.metric=value")
    parser.add_argument(
        "--better-than",
        action="append",
        default=[],
        help="left>right[:overall=NUM][:overall-max=NUM][:component=name=NUM,...][:component-max=name=NUM,...]",
    )
    parser.add_argument("--output", "-o", help="Path to write spec JSON; defaults to stdout")
    args = parser.parse_args(argv)

    try:
        payload = build_spec(
            cases_raw=args.case,
            reference_cases_raw=args.reference_case,
            good_cases_raw=args.good_case,
            review_cases_raw=args.review_case,
            bad_cases_raw=args.bad_case,
            expected_order_raw=args.expected_order,
            gating_expectations=args.expect_gating,
            verdict_expectations=args.expect_verdict,
            overall_at_least=args.overall_at_least,
            overall_at_most=args.overall_at_most,
            component_at_least=args.component_at_least,
            component_at_most=args.component_at_most,
            metric_at_least=args.metric_at_least,
            metric_at_most=args.metric_at_most,
            better_than_raw=args.better_than,
        )
    except SpecBuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote benchmark spec: {output_path}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
