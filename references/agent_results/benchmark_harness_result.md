# Benchmark harness implementation result

Implemented a critic-sprint benchmark harness that compares **baseline vs adaptive vs critic** outputs across a fixed set of fixture pairs without touching planner/render core files.

## What changed
- Added `scripts/critic_benchmark_harness.py`
  - reads a small JSON config describing fixed fixture pairs and lane outputs
  - auto-builds per-fixture benchmark specs for `baseline`, `adaptive`, and `critic`
  - reuses the existing `scripts/listen_gate_benchmark.py` harness for each fixture
  - enforces expected lane ordering (default `critic > adaptive > baseline`)
  - aggregates per-fixture results into one sprint report + scoreboard
  - can optionally persist generated specs for debugging/regression triage
  - can emit both JSON and Markdown summaries
- Added `scripts/critic_benchmark_harness.example.json`
  - shows the expected config shape for curated fixture-pair runs
- Added `tests/test_critic_benchmark_harness.py`
  - covers default monotonic ordering rule generation
  - covers aggregate scoreboard/ranking behavior across multiple fixtures
  - covers CLI failure behavior when critic/adaptive ordering regresses
- Updated `README.md`
  - documented the new critic sprint benchmark harness workflow and example command

## Validation
Ran:

```bash
./.venv/bin/python -m pytest -q tests/test_critic_benchmark_harness.py tests/test_listen_gate_benchmark.py tests/test_build_listen_gate_spec.py
```

Result:
- `13 passed`

## Notes for main agent
- Scope stayed inside `scripts/`, `tests/`, lightweight config/example JSON, and README docs only.
- No planner or render core files were modified.
- The new harness is intentionally thin: it layers on top of the existing listen benchmark/spec machinery so future sprint comparisons use the same comparison semantics already present in the repo.
