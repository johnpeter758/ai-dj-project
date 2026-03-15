#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ai_dj


class HarnessError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_spec_case(case: dict[str, Any]) -> dict[str, Any]:
    if "label" not in case or "input" not in case:
        raise HarnessError(f"Each case needs 'label' and 'input': {case}")
    return {
        "label": str(case["label"]),
        "input": str(Path(case["input"]).expanduser().resolve()),
        "expect": dict(case.get("expect") or {}),
    }


def _numeric_at_least(label: str, actual: Any, minimum: float, failures: list[str]) -> None:
    if actual is None or float(actual) < float(minimum):
        failures.append(f"{label}: expected >= {minimum}, got {actual}")


def _numeric_at_most(label: str, actual: Any, maximum: float, failures: list[str]) -> None:
    if actual is None or float(actual) > float(maximum):
        failures.append(f"{label}: expected <= {maximum}, got {actual}")


def _evaluate_case_expectations(case_result: dict[str, Any]) -> list[str]:
    expect = case_result.get("expect") or {}
    report = case_result.get("report") or {}
    failures: list[str] = []

    if "gating_status" in expect:
        actual = ((report.get("gating") or {}).get("status"))
        if actual != expect["gating_status"]:
            failures.append(f"gating_status: expected {expect['gating_status']}, got {actual}")
    if "verdict" in expect:
        actual = report.get("verdict")
        if actual != expect["verdict"]:
            failures.append(f"verdict: expected {expect['verdict']}, got {actual}")
    if "overall_score_at_least" in expect:
        _numeric_at_least("overall_score", report.get("overall_score"), float(expect["overall_score_at_least"]), failures)
    if "overall_score_at_most" in expect:
        _numeric_at_most("overall_score", report.get("overall_score"), float(expect["overall_score_at_most"]), failures)

    component_scores = {
        key: float((report.get(key) or {}).get("score"))
        for key in ai_dj.LISTEN_COMPONENT_KEYS
        if (report.get(key) or {}).get("score") is not None
    }
    for key, minimum in (expect.get("component_score_at_least") or {}).items():
        _numeric_at_least(f"component[{key}]", component_scores.get(key), float(minimum), failures)
    for key, maximum in (expect.get("component_score_at_most") or {}).items():
        _numeric_at_most(f"component[{key}]", component_scores.get(key), float(maximum), failures)

    return failures


def _evaluate_pairwise_expectations(
    cases: list[dict[str, Any]],
    comparison_map: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    by_label = {case["label"]: case for case in cases}

    for case in cases:
        label = case["label"]
        comparisons = case.get("expect", {}).get("better_than") or []
        for rule in comparisons:
            other = str(rule["other"])
            pair = comparison_map.get((label, other))
            if pair is None:
                failures.append({
                    "case": label,
                    "other": other,
                    "reason": "missing comparison",
                })
                continue
            if pair["winner"]["overall"] != "left":
                failures.append({
                    "case": label,
                    "other": other,
                    "reason": f"expected overall win over {other}, got {pair['winner']['overall']}",
                })
                continue
            min_delta = rule.get("overall_score_delta_at_least")
            if min_delta is not None and float(pair["deltas"]["overall_score_delta"]) < float(min_delta):
                failures.append({
                    "case": label,
                    "other": other,
                    "reason": (
                        f"expected overall score delta >= {min_delta}, got {pair['deltas']['overall_score_delta']}"
                    ),
                })
            for component, minimum in (rule.get("component_score_delta_at_least") or {}).items():
                actual = (pair["deltas"].get("component_score_deltas") or {}).get(component)
                if actual is None or float(actual) < float(minimum):
                    failures.append({
                        "case": label,
                        "other": other,
                        "reason": f"expected {component} delta >= {minimum}, got {actual}",
                    })

    expected_order = [str(item) for item in (cases[0].get("_spec_order") or [])] if cases else []
    if expected_order:
        ranking = [case["label"] for case in sorted(by_label.values(), key=lambda row: int(row["benchmark_rank"]))]
        for left, right in zip(expected_order, expected_order[1:]):
            if ranking.index(left) > ranking.index(right):
                failures.append({
                    "case": left,
                    "other": right,
                    "reason": f"expected ranking order {left} ahead of {right}, got {ranking}",
                })

    return failures


def run_harness(spec_path: str) -> dict[str, Any]:
    spec_file = Path(spec_path).expanduser().resolve()
    spec = _load_json(spec_file)
    raw_cases = spec.get("cases") or []
    if len(raw_cases) < 2:
        raise HarnessError("Spec must define at least two benchmark cases")

    cases = [_resolve_spec_case(case) for case in raw_cases]
    expected_order = [str(item) for item in (spec.get("expected_order") or [])]
    for case in cases:
        case["_spec_order"] = expected_order

    benchmark = ai_dj._build_listen_benchmark([case["input"] for case in cases])
    ranking = benchmark.get("ranking") or []
    rank_lookup = {row["label"]: index + 1 for index, row in enumerate(ranking)}

    comparison_map: dict[tuple[str, str], dict[str, Any]] = {}
    pairwise: list[dict[str, Any]] = []
    for left in cases:
        for right in cases:
            if left["label"] == right["label"]:
                continue
            comparison = ai_dj._build_listen_comparison(left["input"], right["input"])
            comparison_map[(left["label"], right["label"])] = comparison
            if left["label"] < right["label"]:
                pairwise.append(
                    {
                        "left": left["label"],
                        "right": right["label"],
                        "overall_score_delta": comparison["deltas"]["overall_score_delta"],
                        "component_score_deltas": comparison["deltas"]["component_score_deltas"],
                        "winner": comparison["winner"],
                        "decision": comparison.get("decision", {}),
                    }
                )

    case_results: list[dict[str, Any]] = []
    benchmark_by_input = {str(Path(item["input_path"]).resolve()): item for item in ranking}
    for case in cases:
        resolved = ai_dj._resolve_compare_input(case["input"])
        report = resolved["report"]
        bench_row = benchmark_by_input[str(Path(resolved["input_path"]).resolve())]
        result = {
            "label": case["label"],
            "input": resolved["input_path"],
            "input_label": resolved["input_label"],
            "report_origin": resolved["report_origin"],
            "resolved_audio_path": resolved.get("resolved_audio_path"),
            "render_manifest_path": resolved.get("render_manifest_path"),
            "report": report,
            "expect": case.get("expect") or {},
            "benchmark_rank": rank_lookup[bench_row["label"]],
            "benchmark_row": bench_row,
        }
        result["case_failures"] = _evaluate_case_expectations(result)
        case_results.append(result)

    label_results = {case["label"]: case for case in case_results}
    pair_failures = _evaluate_pairwise_expectations(case_results, comparison_map)
    order_failures: list[dict[str, Any]] = []
    if expected_order:
        actual_order = [case["label"] for case in sorted(case_results, key=lambda row: int(row["benchmark_rank"]))]
        if actual_order != expected_order:
            order_failures.append(
                {
                    "case": expected_order[0],
                    "other": expected_order[-1],
                    "reason": f"expected exact benchmark order {expected_order}, got {actual_order}",
                }
            )

    passed = all(not case["case_failures"] for case in case_results) and not pair_failures and not order_failures
    summary_lines = []
    for case in sorted(case_results, key=lambda row: int(row["benchmark_rank"])):
        summary_lines.append(
            f"#{case['benchmark_rank']} {case['label']}: overall={case['report']['overall_score']} "
            f"verdict={case['report'].get('verdict')} gating={(case['report'].get('gating') or {}).get('status')} "
            f"wins={case['benchmark_row']['wins']} net_delta={case['benchmark_row']['net_score_delta']:+.1f}"
        )
    for failure in pair_failures + order_failures:
        summary_lines.append(f"FAIL {failure['case']} vs {failure['other']}: {failure['reason']}")
    for case in case_results:
        for failure in case["case_failures"]:
            summary_lines.append(f"FAIL {case['label']}: {failure}")

    return {
        "schema_version": "0.1.0",
        "spec_path": str(spec_file),
        "passed": passed,
        "expected_order": expected_order,
        "benchmark": benchmark,
        "cases": case_results,
        "pairwise": pairwise,
        "failures": {
            "case_failures": {case['label']: case['case_failures'] for case in case_results if case['case_failures']},
            "pairwise_failures": pair_failures,
            "order_failures": order_failures,
        },
        "summary": summary_lines,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a fixed listen benchmark + gate harness from a JSON spec.")
    parser.add_argument("spec", help="Path to benchmark spec JSON")
    parser.add_argument("--output", "-o", help="Path to write harness JSON output")
    args = parser.parse_args(argv)

    try:
        payload = run_harness(args.spec)
    except (HarnessError, ai_dj.CliError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote listen harness: {output_path}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))

    print(f"Harness: {'PASS' if payload['passed'] else 'FAIL'}")
    for line in payload["summary"]:
        print(f"- {line}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
