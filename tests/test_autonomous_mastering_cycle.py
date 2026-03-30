from pathlib import Path

from scripts.autonomous_mastering_cycle import find_latest_song_birth_summary


def _write_summary(path: Path, stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"timestamp": "%s", "passed": true}' % stamp, encoding="utf-8")


def test_find_latest_song_birth_summary_prefers_newest_mtime(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    first = runs_dir / "song_birth_phase12_20260330_000001" / "song_birth_benchmark_summary.json"
    second = runs_dir / "song_birth_phase12_20260330_000002" / "song_birth_benchmark_summary.json"

    _write_summary(first, "20260330_000001")
    _write_summary(second, "20260330_000002")

    # Ensure deterministic ordering by mtime for the assertion.
    first.touch()
    second.touch()

    latest = find_latest_song_birth_summary(runs_dir)
    assert latest == second
