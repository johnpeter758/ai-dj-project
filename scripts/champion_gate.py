from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARKS_DIR = Path("runs/phase-12")
DEFAULT_CHAMPION_PTR = Path("runs/champion/current.json")
DEFAULT_CHAMPION_HISTORY = Path("runs/champion/history.json")


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "passed"}
    return False


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(summary.get("run_id") or summary.get("id") or summary.get("timestamp") or "unknown"),
        "source": str(summary.get("source") or summary.get("path") or "unknown"),
        "overall_pass": _to_bool(summary.get("overall_pass", summary.get("pass"))),
        "gating_pass": _to_bool(summary.get("gating_pass", summary.get("guardrails_pass"))),
        "song_likeness": _to_float(summary.get("song_likeness", summary.get("song_likeness_score"))),
        "score": _to_float(summary.get("score", summary.get("overall_score"))),
    }


def find_latest_benchmark(benchmarks_dir: Path) -> Path:
    candidates = sorted(
        [
            p
            for p in benchmarks_dir.rglob("*.json")
            if "benchmark" in p.name.lower() or "summary" in p.name.lower()
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No benchmark summary JSON found under {benchmarks_dir}")
    return candidates[-1]


def evaluate(
    challenger: dict[str, Any],
    champion: dict[str, Any] | None,
    *,
    min_song_likeness: float,
    min_score: float,
    min_delta: float,
) -> dict[str, Any]:
    guardrails = {
        "overall_pass": bool(challenger["overall_pass"]),
        "gating_pass": bool(challenger["gating_pass"]),
        "song_likeness_floor": challenger["song_likeness"] >= min_song_likeness,
        "score_floor": challenger["score"] >= min_score,
    }
    guardrails_ok = all(guardrails.values())

    champion_score = _to_float((champion or {}).get("score"), default=float("-inf"))
    score_delta = round(challenger["score"] - champion_score, 6)
    measurable_win = champion is None or score_delta >= min_delta
    promote = bool(guardrails_ok and measurable_win)

    reason = "promoted"
    if not guardrails_ok:
        reason = "guardrails_failed"
    elif not measurable_win:
        reason = "delta_below_threshold"

    return {
        "promote": promote,
        "reason": reason,
        "guardrails": guardrails,
        "guardrails_ok": guardrails_ok,
        "score_delta": score_delta if champion is not None else None,
        "min_delta": min_delta,
    }


def gate(
    *,
    benchmark_path: Path | None,
    benchmarks_dir: Path,
    champion_pointer_path: Path,
    champion_history_path: Path,
    min_song_likeness: float,
    min_score: float,
    min_delta: float,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_benchmark = benchmark_path or find_latest_benchmark(benchmarks_dir)
    challenger_raw = _safe_read_json(selected_benchmark)
    challenger = extract_metrics({**challenger_raw, "source": str(selected_benchmark)})

    champion_raw = _safe_read_json(champion_pointer_path)
    champion = extract_metrics(champion_raw) if champion_raw else None

    eval_result = evaluate(
        challenger,
        champion,
        min_song_likeness=min_song_likeness,
        min_score=min_score,
        min_delta=min_delta,
    )

    history = _safe_read_json(champion_history_path)
    if not isinstance(history, list):
        history = []

    outcome = {
        "timestamp": utc_now(),
        "challenger": challenger,
        "champion_before": champion,
        "decision": eval_result,
    }

    promoted = False
    if eval_result["promote"] and not dry_run:
        _safe_write_json(champion_pointer_path, challenger)
        promoted = True

    outcome["promoted"] = promoted
    outcome["champion_after"] = challenger if promoted else champion

    history.append(outcome)
    if not dry_run:
        _safe_write_json(champion_history_path, history)

    result = {
        "ok": True,
        "benchmark_path": str(selected_benchmark),
        "promoted": promoted,
        "reason": eval_result["reason"],
        "guardrails_ok": eval_result["guardrails_ok"],
        "score_delta": eval_result["score_delta"],
        "challenger_id": challenger["id"],
        "champion_id_before": champion["id"] if champion else None,
        "champion_id_after": outcome["champion_after"]["id"] if outcome["champion_after"] else None,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Champion/challenger promotion gate")
    parser.add_argument("--benchmark-path", type=Path, default=None)
    parser.add_argument("--benchmarks-dir", type=Path, default=DEFAULT_BENCHMARKS_DIR)
    parser.add_argument("--champion-pointer", type=Path, default=DEFAULT_CHAMPION_PTR)
    parser.add_argument("--champion-history", type=Path, default=DEFAULT_CHAMPION_HISTORY)
    parser.add_argument("--min-song-likeness", type=float, default=0.72)
    parser.add_argument("--min-score", type=float, default=0.80)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = gate(
        benchmark_path=args.benchmark_path,
        benchmarks_dir=args.benchmarks_dir,
        champion_pointer_path=args.champion_pointer,
        champion_history_path=args.champion_history,
        min_song_likeness=args.min_song_likeness,
        min_score=args.min_score,
        min_delta=args.min_delta,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
