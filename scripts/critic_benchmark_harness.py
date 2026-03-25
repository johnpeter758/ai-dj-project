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

from scripts.listen_gate_benchmark import HarnessError, run_harness


class SprintHarnessError(RuntimeError):
    pass


DEFAULT_LANES = ["baseline", "adaptive", "critic"]
DEFAULT_EXPECTED_LANE_ORDER = ["critic", "adaptive", "baseline"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_lane_inputs(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise SprintHarnessError("Fixture 'cases' must be a mapping of lane->path")
    normalized: dict[str, str] = {}
    for lane, value in raw.items():
        lane_name = str(lane).strip()
        if not lane_name:
            raise SprintHarnessError("Fixture case lane names must be non-empty")
        if not value:
            raise SprintHarnessError(f"Fixture case '{lane_name}' must provide a path")
        normalized[lane_name] = str(Path(str(value)).expanduser().resolve())
    return normalized


def _normalize_fixture(raw: dict[str, Any], expected_lanes: list[str]) -> dict[str, Any]:
    label = str(raw.get("label") or "").strip()
    if not label:
        raise SprintHarnessError("Each fixture needs a non-empty 'label'")

    cases = _normalize_lane_inputs(raw.get("cases"))
    missing = [lane for lane in expected_lanes if lane not in cases]
    if missing:
        raise SprintHarnessError(f"Fixture '{label}' is missing required lanes: {', '.join(missing)}")

    payload = {
        "label": label,
        "cases": {lane: cases[lane] for lane in expected_lanes},
        "expectations": dict(raw.get("expectations") or {}),
        "notes": list(raw.get("notes") or []),
    }
    if raw.get("pair") is not None:
        payload["pair"] = dict(raw.get("pair") or {})
    return payload


def build_fixture_spec(
    fixture: dict[str, Any],
    *,
    lanes: list[str] | None = None,
    expected_lane_order: list[str] | None = None,
    require_monotonic_improvement: bool = True,
) -> dict[str, Any]:
    active_lanes = list(lanes or DEFAULT_LANES)
    lane_order = list(expected_lane_order or DEFAULT_EXPECTED_LANE_ORDER)
    if set(active_lanes) != set(lane_order):
        raise SprintHarnessError("lanes and expected_lane_order must contain the same lane labels")

    expectations = dict(fixture.get("expectations") or {})
    cases: list[dict[str, Any]] = []
    for lane in active_lanes:
        case = {
            "label": lane,
            "input": fixture["cases"][lane],
        }
        lane_expect = expectations.get(lane)
        if lane_expect:
            case["expect"] = dict(lane_expect)
        cases.append(case)

    if require_monotonic_improvement:
        for stronger, weaker in zip(lane_order, lane_order[1:]):
            lane_case = next(case for case in cases if case["label"] == stronger)
            expect = lane_case.setdefault("expect", {})
            better_than = expect.setdefault("better_than", [])
            if not any(str(rule.get("other")) == weaker for rule in better_than):
                better_than.append({"other": weaker})

    return {
        "fixture_label": fixture["label"],
        "expected_order": lane_order,
        "cases": cases,
        "notes": list(fixture.get("notes") or []),
        "pair": dict(fixture.get("pair") or {}),
    }


def _lane_outcome(case: dict[str, Any]) -> dict[str, Any]:
    report = case.get("report") or {}
    benchmark_row = case.get("benchmark_row") or {}
    return {
        "label": case["label"],
        "benchmark_rank": case["benchmark_rank"],
        "overall_score": report.get("overall_score"),
        "verdict": report.get("verdict"),
        "gating_status": (report.get("gating") or {}).get("status"),
        "wins": benchmark_row.get("wins"),
        "losses": benchmark_row.get("losses"),
        "ties": benchmark_row.get("ties"),
        "net_score_delta": benchmark_row.get("net_score_delta"),
        "case_failures": list(case.get("case_failures") or []),
    }


def _scoreboard_template(lanes: list[str]) -> dict[str, dict[str, Any]]:
    return {
        lane: {
            "fixtures": 0,
            "rank_sum": 0,
            "average_rank": None,
            "wins_first": 0,
            "passes": 0,
            "fails": 0,
            "gating_status_counts": {},
            "verdict_counts": {},
            "mean_overall_score": None,
            "overall_score_sum": 0.0,
        }
        for lane in lanes
    }


def _update_scoreboard(scoreboard: dict[str, dict[str, Any]], fixture_result: dict[str, Any]) -> None:
    for lane_result in fixture_result["lanes"]:
        bucket = scoreboard[lane_result["label"]]
        bucket["fixtures"] += 1
        bucket["rank_sum"] += int(lane_result["benchmark_rank"])
        bucket["overall_score_sum"] += float(lane_result["overall_score"])
        if int(lane_result["benchmark_rank"]) == 1:
            bucket["wins_first"] += 1
        if lane_result["case_failures"]:
            bucket["fails"] += 1
        else:
            bucket["passes"] += 1
        gating = str(lane_result.get("gating_status"))
        verdict = str(lane_result.get("verdict"))
        bucket["gating_status_counts"][gating] = bucket["gating_status_counts"].get(gating, 0) + 1
        bucket["verdict_counts"][verdict] = bucket["verdict_counts"].get(verdict, 0) + 1


def _finalize_scoreboard(scoreboard: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for bucket in scoreboard.values():
        fixtures = bucket["fixtures"]
        if fixtures:
            bucket["average_rank"] = round(bucket["rank_sum"] / fixtures, 3)
            bucket["mean_overall_score"] = round(bucket["overall_score_sum"] / fixtures, 3)
        del bucket["rank_sum"]
        del bucket["overall_score_sum"]
    return scoreboard


def run_sprint_harness(
    config_path: str,
    *,
    spec_output_dir: str | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    config = _load_json(config_file)
    lanes = [str(item) for item in (config.get("lanes") or DEFAULT_LANES)]
    expected_lane_order = [str(item) for item in (config.get("expected_lane_order") or DEFAULT_EXPECTED_LANE_ORDER)]
    require_monotonic_improvement = bool(config.get("require_monotonic_improvement", True))

    raw_fixtures = config.get("fixtures") or []
    if not raw_fixtures:
        raise SprintHarnessError("Config must define at least one fixture")

    fixtures = [_normalize_fixture(dict(raw), lanes) for raw in raw_fixtures]
    spec_dir = Path(spec_output_dir).expanduser().resolve() if spec_output_dir else None
    if spec_dir:
        spec_dir.mkdir(parents=True, exist_ok=True)

    fixture_results: list[dict[str, Any]] = []
    scoreboard = _scoreboard_template(lanes)

    for fixture in fixtures:
        spec = build_fixture_spec(
            fixture,
            lanes=lanes,
            expected_lane_order=expected_lane_order,
            require_monotonic_improvement=require_monotonic_improvement,
        )
        if spec_dir:
            spec_path = spec_dir / f"{fixture['label']}_benchmark_spec.json"
            spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
        harness_report = run_harness_payload(spec)
        lane_results = [_lane_outcome(case) for case in sorted(harness_report["cases"], key=lambda row: int(row["benchmark_rank"]))]
        fixture_result = {
            "fixture_label": fixture["label"],
            "pair": dict(fixture.get("pair") or {}),
            "notes": list(fixture.get("notes") or []),
            "passed": harness_report["passed"],
            "expected_order": harness_report["expected_order"],
            "ranking": [row["label"] for row in lane_results],
            "lanes": lane_results,
            "pairwise": harness_report["pairwise"],
            "failures": harness_report["failures"],
            "summary": harness_report["summary"],
        }
        _update_scoreboard(scoreboard, fixture_result)
        fixture_results.append(fixture_result)

    passed = all(item["passed"] for item in fixture_results)
    return {
        "schema_version": "0.1.0",
        "config_path": str(config_file),
        "passed": passed,
        "lanes": lanes,
        "expected_lane_order": expected_lane_order,
        "require_monotonic_improvement": require_monotonic_improvement,
        "fixtures": fixture_results,
        "aggregate": {
            "fixture_count": len(fixture_results),
            "passed_fixtures": sum(1 for item in fixture_results if item["passed"]),
            "failed_fixtures": sum(1 for item in fixture_results if not item["passed"]),
            "lane_scoreboard": _finalize_scoreboard(scoreboard),
        },
    }


def run_harness_payload(spec: dict[str, Any]) -> dict[str, Any]:
    temp_spec = ROOT / "runs" / "benchmark_specs" / "_temp_critic_benchmark_spec.json"
    temp_spec.parent.mkdir(parents=True, exist_ok=True)
    temp_spec.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    try:
        return run_harness(str(temp_spec))
    finally:
        if temp_spec.exists():
            temp_spec.unlink()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Critic Sprint Benchmark Harness",
        "",
        f"Config: `{report['config_path']}`",
        f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"Fixtures: {report['aggregate']['passed_fixtures']}/{report['aggregate']['fixture_count']} passed",
        "",
        "## Aggregate lane scoreboard",
        "",
        "| lane | avg rank | #1 finishes | case passes | case fails | mean overall | gating | verdicts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for lane in report["lanes"]:
        row = report["aggregate"]["lane_scoreboard"][lane]
        gating = ", ".join(f"{key}:{value}" for key, value in sorted(row["gating_status_counts"].items()))
        verdicts = ", ".join(f"{key}:{value}" for key, value in sorted(row["verdict_counts"].items()))
        lines.append(
            f"| {lane} | {row['average_rank']} | {row['wins_first']} | {row['passes']} | {row['fails']} | {row['mean_overall_score']} | {gating} | {verdicts} |"
        )

    for fixture in report["fixtures"]:
        pair = fixture.get("pair") or {}
        pair_text = ""
        if pair:
            pair_text = f" — `{pair.get('song_a', '?')}` × `{pair.get('song_b', '?')}`"
        lines += [
            "",
            f"## {fixture['fixture_label']}{pair_text}",
            f"- status: **{'PASS' if fixture['passed'] else 'FAIL'}**",
            f"- expected order: `{fixture['expected_order']}`",
            f"- actual ranking: `{fixture['ranking']}`",
        ]
        for lane in fixture["lanes"]:
            lines.append(
                f"- {lane['label']}: rank={lane['benchmark_rank']} overall={lane['overall_score']} verdict={lane['verdict']} gating={lane['gating_status']} wins={lane['wins']} net_delta={lane['net_score_delta']}"
            )
        for failure_group, value in fixture["failures"].items():
            if value:
                lines.append(f"- {failure_group}: `{json.dumps(value, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run baseline vs adaptive vs critic benchmark harness across fixed fixture pairs.")
    parser.add_argument("config", help="Path to critic benchmark harness config JSON")
    parser.add_argument("--output", "-o", help="Path to write aggregate JSON report")
    parser.add_argument("--markdown", help="Path to write aggregate markdown summary")
    parser.add_argument("--spec-output-dir", help="Optional directory to persist generated per-fixture benchmark specs")
    args = parser.parse_args(argv)

    try:
        report = run_sprint_harness(args.config, spec_output_dir=args.spec_output_dir)
    except (SprintHarnessError, HarnessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote critic benchmark report: {output_path}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.markdown:
        markdown_path = Path(args.markdown).expanduser().resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"Wrote critic benchmark markdown: {markdown_path}")

    print(f"Critic benchmark harness: {'PASS' if report['passed'] else 'FAIL'}")
    for fixture in report["fixtures"]:
        print(f"- {fixture['fixture_label']}: {'PASS' if fixture['passed'] else 'FAIL'} ({fixture['ranking']})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
