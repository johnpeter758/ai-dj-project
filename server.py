#!/usr/bin/env python3
"""VocalFusion prototype debug server."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, render_template, send_file, send_from_directory

from scripts.build_listen_gate_spec import SpecBuildError, build_spec
from scripts.reference_input_normalizer import ReferenceInputError, normalize_reference_inputs
from src.core.analysis import analyze_audio_file
from src.core.evaluation.listen import evaluate_song
from src.core.planner import build_compatibility_report, build_stub_arrangement_plan
from src.feedback_learning import build_feedback_learning_summary, write_feedback_learning_summary
from src.human_feedback import HumanFeedbackStore

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
MUSIC_DIR = BASE_DIR / "music"
RUNS_DIR = BASE_DIR / "runs"
UPLOADS_DIR = RUNS_DIR / "ui_uploads"
PROJECT_PYTHON = Path("/Users/johnpeter/venvs/vocalfusion-env/bin/python")
VAULT_DIR = Path("/Users/johnpeter/VocalFusionVault")
TOOLS_LOG_PATH = VAULT_DIR / "memory" / "TOOLS_RUNNING_LOG.md"
MEMORY_DIR = VAULT_DIR / "memory"
HUMAN_FEEDBACK_DIR = BASE_DIR / "data" / "human_feedback"
SIMPLE_FUSE_ADVANCED_TIMEOUT_SECONDS = 180
SIMPLE_FUSE_DIRECT_TIMEOUT_SECONDS = 3600
SIMPLE_FUSE_JOB_TTL_SECONDS = 24 * 60 * 60
_SIMPLE_FUSE_JOBS: dict[str, dict] = {}
_SIMPLE_FUSE_LOCK = threading.Lock()


def _split_labeled_path(raw: str) -> tuple[str | None, str]:
    text = str(raw).strip()
    for delimiter in ("=", "::"):
        if delimiter not in text:
            continue
        label, candidate = text.split(delimiter, 1)
        label = label.strip()
        candidate = candidate.strip()
        if label and candidate:
            return label, candidate
    return None, text


def load_songs() -> list[dict]:
    songs: list[dict] = []
    if not MUSIC_DIR.exists():
        return songs

    for root, _, files in os.walk(MUSIC_DIR):
        for filename in files:
            if filename.lower().endswith((".mp3", ".wav", ".flac", ".m4a", ".aac")):
                full_path = Path(root) / filename
                rel_path = full_path.relative_to(MUSIC_DIR)
                song_id = str(rel_path).replace(os.sep, "__")
                songs.append(
                    {
                        "id": song_id,
                        "title": full_path.stem,
                        "artist": rel_path.parts[0] if len(rel_path.parts) > 1 else "Unknown",
                        "file": str(rel_path),
                        "absolute_path": str(full_path),
                    }
                )
    songs.sort(key=lambda s: (s["artist"].lower(), s["title"].lower()))
    return songs


def resolve_song_path(song_id: str) -> Path | None:
    for song in load_songs():
        if song["id"] == song_id:
            return Path(song["absolute_path"])
    return None


def _component_score(report: dict, key: str) -> float:
    try:
        return float((report.get(key) or {}).get("score") or 0.0)
    except Exception:
        return 0.0


def _chart_calibrated_score(report: dict) -> float:
    """Heavier commercial calibration: reward chart-like traits, punish rough mixes hard."""
    overall = float(report.get("overall_score") or 0.0)
    song_likeness = _component_score(report, "song_likeness")
    mix_sanity = _component_score(report, "mix_sanity")
    structure = _component_score(report, "structure")
    coherence = _component_score(report, "coherence")
    groove = _component_score(report, "groove")
    energy_arc = _component_score(report, "energy_arc")
    transition = _component_score(report, "transition")

    score = overall

    # Upside: aggressively reward commercial-readability cues.
    score += max(0.0, song_likeness - 75.0) * 0.35
    score += max(0.0, mix_sanity - 72.0) * 0.20
    score += max(0.0, structure - 75.0) * 0.15
    score += max(0.0, coherence - 75.0) * 0.10
    score += max(0.0, energy_arc - 70.0) * 0.10
    score += max(0.0, transition - 65.0) * 0.05
    score += max(0.0, groove - 60.0) * 0.05

    # Downside: heavily punish low-quality amateur-ish output.
    score -= max(0.0, 45.0 - groove) * 0.55
    score -= max(0.0, 45.0 - mix_sanity) * 0.45
    score -= max(0.0, 42.0 - song_likeness) * 0.60
    score -= max(0.0, 40.0 - overall) * 0.40

    # Bonus / malus anchors for practical chart-like calibration.
    if song_likeness >= 85.0 and mix_sanity >= 80.0 and structure >= 85.0:
        score += 6.0
    if groove < 35.0 and mix_sanity < 40.0:
        score -= 12.0

    return round(max(0.0, min(100.0, score)), 1)


def _run_git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout.strip()


def _latest_memory_file() -> Path | None:
    if not MEMORY_DIR.exists():
        return None
    files = sorted(MEMORY_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _extract_current_task() -> dict:
    fallback = {
        "summary": "Improving planner and evaluator quality toward stronger musical flow.",
        "source": None,
        "details": [],
    }

    memory_file = _latest_memory_file()
    if memory_file is None:
        return fallback

    try:
        text = memory_file.read_text(encoding="utf-8")
        lines = [line.rstrip() for line in text.splitlines()]
        bullets = [line.strip()[2:].strip() for line in lines if line.strip().startswith("- ")]
        recent_bullets = bullets[-4:]

        summary = fallback["summary"]
        priority_groups = [
            ("current quality bottleneck", "next quality gain"),
            ("planner", "evaluator", "listen", "musical flow", "structure intelligence"),
            ("working",),
        ]
        matched = False
        for keywords in priority_groups:
            for bullet in reversed(bullets):
                lower = bullet.lower()
                if any(keyword in lower for keyword in keywords):
                    summary = bullet
                    matched = True
                    break
            if matched:
                break
        if recent_bullets and not matched:
            summary = recent_bullets[-1]

        return {
            "summary": summary,
            "source": str(memory_file),
            "details": recent_bullets,
        }
    except Exception:
        return fallback


def _latest_commit() -> dict:
    try:
        out = _run_git("log", "-1", "--pretty=%H%n%h%n%s%n%ci")
        lines = [line for line in out.splitlines() if line.strip()]
        if len(lines) >= 4:
            return {
                "hash": lines[0],
                "short_hash": lines[1],
                "message": lines[2],
                "date": lines[3],
            }
    except Exception:
        pass
    return {}


def _changed_files() -> list[str]:
    try:
        out = _run_git("status", "--short")
        return [line for line in out.splitlines() if line.strip()][:12]
    except Exception:
        return []


def _relative_to_runs(path: Path) -> str:
    try:
        return str(path.relative_to(RUNS_DIR))
    except ValueError:
        return str(path)


def _artifact_entry(path: Path) -> dict:
    run_dir = path.parent.name if path.parent != RUNS_DIR else None
    return {
        "path": str(path),
        "name": path.name,
        "run_dir": run_dir,
        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds"),
        "relative_path": _relative_to_runs(path),
        "download_url": f"/api/artifact?path={path}",
    }


def _load_json_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _set_simple_fuse_job(job_id: str, **patch) -> dict:
    with _SIMPLE_FUSE_LOCK:
        current = dict(_SIMPLE_FUSE_JOBS.get(job_id) or {})
        current.update(patch)
        current.setdefault("job_id", job_id)
        current["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _SIMPLE_FUSE_JOBS[job_id] = current
        return dict(current)


def _get_simple_fuse_job(job_id: str) -> dict | None:
    with _SIMPLE_FUSE_LOCK:
        job = _SIMPLE_FUSE_JOBS.get(job_id)
        return dict(job) if job else None


def _prune_simple_fuse_jobs() -> None:
    cutoff = datetime.now().timestamp() - SIMPLE_FUSE_JOB_TTL_SECONDS
    with _SIMPLE_FUSE_LOCK:
        stale = []
        for job_id, payload in _SIMPLE_FUSE_JOBS.items():
            try:
                updated = datetime.fromisoformat(payload.get("updated_at", "1970-01-01T00:00:00")).timestamp()
            except Exception:
                updated = 0
            if updated < cutoff:
                stale.append(job_id)
        for job_id in stale:
            _SIMPLE_FUSE_JOBS.pop(job_id, None)


def _run_simple_fuse_job(job_id: str, song_a_path: Path, song_b_path: Path, run_id: str, outdir: Path, fusion_dir: Path) -> None:
    direct_cmd = [
        str(PROJECT_PYTHON if PROJECT_PYTHON.exists() else "python3"),
        str(BASE_DIR / "ai_dj.py"),
        "fusion",
        str(song_a_path),
        str(song_b_path),
        "--arrangement-mode",
        "pro",
        "--output",
        str(fusion_dir),
    ]
    _set_simple_fuse_job(
        job_id,
        status="running",
        progress=25,
        message="Pro fuse started. Rendering adaptive+baseline candidates and promoting the best output.",
        stage="fusion",
        product_mode="baseline_first",
        product_label="Baseline-first simple fuse",
    )
    try:
        proc = subprocess.run(
            direct_cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=SIMPLE_FUSE_DIRECT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _set_simple_fuse_job(
            job_id,
            status="error",
            progress=100,
            message="Deterministic fusion timed out.",
            error="Fuse failed: deterministic fusion timed out.",
            stage="failed",
        )
        return

    if proc.returncode != 0:
        _set_simple_fuse_job(
            job_id,
            status="error",
            progress=100,
            message="Deterministic fusion failed.",
            error="Fuse failed: deterministic fusion did not complete successfully.",
            stdout=proc.stdout,
            stderr=proc.stderr,
            stage="failed",
        )
        return

    try:
        result = _promote_direct_fusion_output(
            run_id=run_id,
            output_dir=outdir,
            fusion_dir=fusion_dir,
            fallback_reason="Finished. Returning the direct deterministic fuse result.",
            report_path=None,
        )
    except FileNotFoundError as exc:
        _set_simple_fuse_job(
            job_id,
            status="error",
            progress=100,
            message="Fusion finished but no playable audio artifact was found.",
            error=str(exc),
            stdout=proc.stdout,
            stderr=proc.stderr,
            stage="failed",
        )
        return

    _set_simple_fuse_job(
        job_id,
        status="done",
        progress=100,
        message="Fusion complete. Direct playable output ready.",
        stage="complete",
        output_dir=str(outdir),
        result=result,
        stdout=proc.stdout,
        stderr=proc.stderr,
        share_url=f"/share/{job_id}",
    )


def _find_latest_json(predicate) -> tuple[Path, dict] | None:
    try:
        candidates = sorted(
            [p for p in RUNS_DIR.rglob("*.json") if p.is_file() and "ui_uploads" not in p.parts],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            payload = _load_json_file(path)
            if isinstance(payload, dict) and predicate(path, payload):
                return path, payload
    except Exception:
        pass
    return None


def _summarize_manifest_diagnostics(manifest: dict) -> dict:
    sections = manifest.get("sections") or []
    work_orders = manifest.get("work_orders") or []
    warnings = [str(item) for item in (manifest.get("warnings") or [])]
    fallbacks = [str(item) for item in (manifest.get("fallbacks") or [])]
    overlap_sections = [section for section in sections if section.get("allowed_overlap")]
    stretch_risks = [section for section in sections if float(section.get("stretch_ratio") or 1.0) > 1.25 or float(section.get("stretch_ratio") or 1.0) < 0.75]
    return {
        "section_count": len(sections),
        "work_order_count": len(work_orders),
        "warning_count": len(warnings),
        "fallback_count": len(fallbacks),
        "overlap_section_count": len(overlap_sections),
        "stretch_risk_count": len(stretch_risks),
        "top_warnings": warnings[:3],
        "top_fallbacks": fallbacks[:3],
    }


def _latest_artifact() -> dict | None:
    try:
        preferred_suffixes = {".mp3", ".wav", ".json"}
        preferred_names = {"child_master.mp3", "child_master.wav", "child_raw.wav", "render_manifest.json", "arrangement_plan.json", "compatibility_report.json"}
        files = [
            p for p in RUNS_DIR.rglob("*")
            if p.is_file()
            and "ui_uploads" not in p.parts
            and p.suffix.lower() in preferred_suffixes
            and (p.name in preferred_names or p.parent != RUNS_DIR)
        ]
        if not files:
            return None
        latest = max(files, key=lambda p: p.stat().st_mtime)
        return _artifact_entry(latest)
    except Exception:
        return None


def _latest_listen_result() -> dict | None:
    match = _find_latest_json(lambda _path, payload: "overall_score" in payload and "verdict" in payload)
    if not match:
        return None
    path, payload = match
    result = _artifact_entry(path)
    result.update({
        "overall_score": payload.get("overall_score"),
        "verdict": payload.get("verdict"),
        "top_reasons": payload.get("top_reasons", [])[:3],
        "top_fixes": payload.get("top_fixes", [])[:3],
        "source_path": payload.get("source_path"),
    })
    return result


def _latest_compare_listen_result() -> dict | None:
    match = _find_latest_json(lambda _path, payload: "winner" in payload and "deltas" in payload and "summary" in payload)
    if not match:
        return None
    path, payload = match
    component_deltas = (payload.get("deltas") or {}).get("component_score_deltas") or {}
    biggest_component = None
    if component_deltas:
        biggest_component = max(component_deltas.items(), key=lambda item: abs(float(item[1] or 0.0)))
    result = _artifact_entry(path)
    result.update({
        "summary": payload.get("summary"),
        "overall_winner": (payload.get("winner") or {}).get("overall"),
        "overall_score_delta": (payload.get("deltas") or {}).get("overall_score_delta"),
        "biggest_component_delta": {
            "component": biggest_component[0],
            "delta": biggest_component[1],
        } if biggest_component else None,
    })
    return result


def _latest_benchmark_listen_result() -> dict | None:
    match = _find_latest_json(lambda _path, payload: "winner" in payload and "ranking" in payload and "comparisons" in payload)
    if not match:
        return None
    path, payload = match
    ranking = payload.get("ranking") or []
    top_entry = ranking[0] if ranking else None
    leader_gap = None
    if len(ranking) >= 2:
        leader_gap = round(float((ranking[0] or {}).get("net_score_delta") or 0.0) - float((ranking[1] or {}).get("net_score_delta") or 0.0), 1)
    ranking_preview = [
        {
            "rank": index + 1,
            "label": row.get("label"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "net_score_delta": row.get("net_score_delta"),
            "overall_score": row.get("overall_score"),
            "verdict": row.get("verdict"),
        }
        for index, row in enumerate(ranking[:3])
    ]
    result = _artifact_entry(path)
    result.update({
        "winner": payload.get("winner"),
        "entry_count": len(ranking),
        "comparison_count": len(payload.get("comparisons") or []),
        "leader_gap": leader_gap,
        "top_entry": {
            "label": top_entry.get("label"),
            "wins": top_entry.get("wins"),
            "ties": top_entry.get("ties"),
            "losses": top_entry.get("losses"),
            "net_score_delta": top_entry.get("net_score_delta"),
            "overall_score": top_entry.get("overall_score"),
            "verdict": top_entry.get("verdict"),
        } if top_entry else None,
        "ranking_preview": ranking_preview,
    })
    return result


def _latest_listener_agent_result() -> dict | None:
    match = _find_latest_json(
        lambda _path, payload: "listener_agent" in payload and "recommended_for_human_review" in payload and "rejected" in payload
    )
    if not match:
        return None
    path, payload = match
    recommended = payload.get("recommended_for_human_review") or []
    rejected = payload.get("rejected") or []
    borderline = payload.get("borderline") or []
    top_survivor = recommended[0] if recommended else None
    top_reject = rejected[0] if rejected else None
    empty_reason = "survivors_present" if recommended else "no_survivors" if borderline or rejected else "no_gate_run"
    result = _artifact_entry(path)
    result.update({
        "recommended_count": len(recommended),
        "borderline_count": len(borderline),
        "rejected_count": len(rejected),
        "winner": top_survivor.get("label") if top_survivor else None,
        "summary": list(payload.get("summary") or [])[:3],
        "survivor_preview": recommended[:3],
        "borderline_preview": borderline[:3],
        "reject_preview": rejected[:3],
        "empty_reason": empty_reason,
        "top_survivor": {
            "label": top_survivor.get("label"),
            "listener_rank": top_survivor.get("listener_rank"),
            "overall_score": top_survivor.get("overall_score"),
            "verdict": top_survivor.get("verdict"),
        } if top_survivor else None,
        "top_reject_reason": ((top_reject.get("hard_fail_reasons") or [None])[0] if top_reject else None),
    })
    return result



def _latest_auto_shortlist_result() -> dict | None:
    match = _find_latest_json(
        lambda path, payload: path.name == "auto_shortlist_report.json" and "recommended_shortlist" in payload and "candidates" in payload
    )
    if not match:
        return None
    path, payload = match
    recommended = list(payload.get("recommended_shortlist") or [])
    closest_misses = list(payload.get("closest_misses") or [])
    counts = (payload.get("listener_agent_report") or {}).get("counts") or {}
    result = _artifact_entry(path)
    pruning = payload.get("pruning") or {}
    result.update({
        "summary": list(payload.get("summary") or [])[:4],
        "candidate_count": len(payload.get("candidates") or []),
        "survivor_count": counts.get("survivors"),
        "borderline_count": counts.get("borderline"),
        "rejected_count": counts.get("rejected"),
        "winner": (recommended[0] or {}).get("candidate_id") if recommended else None,
        "top_survivor": recommended[0] if recommended else None,
        "closest_misses": closest_misses[:2],
        "pairwise_winner": ((payload.get("pairwise_pool") or {}).get("winner")),
        "pruning": pruning,
    })
    return result


def _latest_closed_loop_result() -> dict | None:
    match = _find_latest_json(
        lambda path, payload: path.name == "closed_loop_report.json" and "stop_reason" in payload and ("best_iteration" in payload or "loop_summary" in payload)
    )
    if not match:
        return None
    path, payload = match
    loop_summary = payload.get("loop_summary") or {}
    best = loop_summary.get("best_iteration") or payload.get("best_iteration") or {}
    iteration_summaries = list(loop_summary.get("iteration_summaries") or [])
    latest_iteration = iteration_summaries[-1] if iteration_summaries else None
    result = _artifact_entry(path)
    result.update({
        "stop_reason": payload.get("stop_reason"),
        "summary": list(payload.get("summary") or [])[:3],
        "total_iterations": loop_summary.get("total_iterations") if loop_summary else len(payload.get("iterations") or []),
        "net_improvement": loop_summary.get("net_improvement"),
        "score_trajectory": list(loop_summary.get("score_trajectory") or [])[:6],
        "best_iteration": {
            "iteration": best.get("iteration"),
            "candidate_overall_score": best.get("candidate_overall_score"),
            "candidate_verdict": best.get("candidate_verdict"),
            "candidate_listener_decision": best.get("candidate_listener_decision"),
            "candidate_input": best.get("candidate_input"),
        } if best else None,
        "latest_iteration": {
            "iteration": latest_iteration.get("iteration"),
            "candidate_overall_score": latest_iteration.get("candidate_overall_score"),
            "candidate_verdict": latest_iteration.get("candidate_verdict"),
            "candidate_listener_decision": latest_iteration.get("candidate_listener_decision"),
            "gap_to_target": latest_iteration.get("gap_to_target"),
            "plateau_count": latest_iteration.get("plateau_count"),
            "render_dir": latest_iteration.get("render_dir"),
        } if latest_iteration else None,
        "best_top_intervention": loop_summary.get("best_top_intervention"),
    })
    return result


def _product_flow_summary() -> dict:
    return {
        "mode": "baseline_first",
        "label": "Baseline-first simple fuse",
        "headline": "Ship one fast baseline output first, then use the critic loop to decide what to improve next.",
        "delivery_contract": "The simple product path returns one surfaced playable baseline file instead of waiting on a larger candidate batch.",
        "baseline_path": "direct deterministic fusion",
        "status_url": "/status",
    }


def _critic_loop_summary(*, latest_eval: dict | None, latest_compare: dict | None, latest_benchmark: dict | None, latest_listener_agent: dict | None, latest_closed_loop: dict | None, latest_auto_shortlist: dict | None) -> dict:
    status = "idle"
    headline = "No critic-loop results yet."
    decision = "Awaiting listen/eval artifacts."
    next_focus = None

    if latest_listener_agent:
        rejected = int(latest_listener_agent.get("rejected_count") or 0)
        survivors = int(latest_listener_agent.get("recommended_count") or 0)
        if survivors > 0:
            status = "survivor_found"
            headline = f"Listener gate kept {survivors} candidate(s) for human review."
            decision = f"Top survivor: {latest_listener_agent.get('winner') or 'unnamed candidate'}."
        elif rejected > 0:
            status = "needs_work"
            headline = "Listener gate rejected or held back the latest candidates."
            decision = latest_listener_agent.get("top_reject_reason") or "No candidate cleared the listener gate."

    if latest_closed_loop:
        best = latest_closed_loop.get("best_iteration") or {}
        top_intervention = latest_closed_loop.get("best_top_intervention") or {}
        if best.get("iteration"):
            headline = f"Critic loop best iteration: #{best.get('iteration')} at overall {best.get('candidate_overall_score', '—')}."
        if top_intervention.get("component"):
            next_focus = f"Top intervention target: {top_intervention.get('component')}"
            if top_intervention.get("problem"):
                next_focus += f" — {top_intervention.get('problem')}"
        if latest_closed_loop.get("stop_reason"):
            decision = f"Loop stop reason: {latest_closed_loop.get('stop_reason')}"

    if latest_auto_shortlist and latest_auto_shortlist.get("winner"):
        decision = f"Latest ranked winner: {latest_auto_shortlist.get('winner')}"
    elif latest_benchmark and latest_benchmark.get("winner"):
        decision = f"Latest benchmark winner: {latest_benchmark.get('winner')}"
    elif latest_compare and latest_compare.get("overall_winner"):
        decision = f"Latest compare winner: {latest_compare.get('overall_winner')}"
    elif latest_eval and latest_eval.get("verdict"):
        decision = f"Latest listen verdict: {latest_eval.get('verdict')}"

    return {
        "status": status,
        "headline": headline,
        "decision": decision,
        "next_focus": next_focus,
        "latest_listener_gate_winner": (latest_listener_agent or {}).get("winner"),
        "latest_listener_gate_reject_reason": (latest_listener_agent or {}).get("top_reject_reason"),
        "latest_closed_loop_stop_reason": (latest_closed_loop or {}).get("stop_reason"),
        "latest_benchmark_winner": (latest_benchmark or {}).get("winner"),
        "latest_compare_winner": (latest_compare or {}).get("overall_winner"),
        "latest_listen_verdict": (latest_eval or {}).get("verdict"),
    }


def _recent_render_inputs(limit: int = 4) -> list[str]:
    manifests = sorted(
        [
            path for path in RUNS_DIR.rglob("render_manifest.json")
            if path.is_file() and "ui_uploads" not in path.parts
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    seen: set[str] = set()
    results: list[str] = []
    for manifest in manifests:
        run_dir = manifest.parent.resolve()
        key = str(run_dir)
        if key in seen:
            continue
        seen.add(key)
        results.append(key)
        if len(results) >= limit:
            break
    return results


def _feedback_store() -> HumanFeedbackStore:
    return HumanFeedbackStore(HUMAN_FEEDBACK_DIR)


def _resolve_run_scoped_path(raw: str | Path, *, allow_music: bool = False) -> Path:
    path = Path(str(raw)).expanduser().resolve()
    try:
        path.relative_to(RUNS_DIR.resolve())
        return path
    except ValueError:
        if allow_music:
            path.relative_to(MUSIC_DIR.resolve())
            return path
        raise


def _load_run_manifest(run_dir: Path) -> dict | None:
    return _load_json_file(run_dir / "render_manifest.json")


def _candidate_master_audio(run_dir: Path) -> Path | None:
    candidates = [run_dir / "child_master.mp3", run_dir / "child_master.wav", run_dir / "child_raw.wav"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    manifest = _load_run_manifest(run_dir)
    outputs = (manifest or {}).get("outputs") or {}
    for key in ("master_mp3", "master_wav", "raw_wav"):
        value = outputs.get(key)
        if not value:
            continue
        candidate = Path(str(value)).expanduser().resolve()
        if candidate.exists():
            return candidate
    return None


def _resolve_existing_artifact_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw)).expanduser().resolve()
    try:
        path.relative_to(RUNS_DIR.resolve())
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path


def _select_promotable_fuse_candidate(report: dict) -> tuple[str, dict, Path] | None:
    candidate_groups = (
        ("survivor", list(report.get("recommended_shortlist") or [])),
        ("closest_miss", list(report.get("closest_misses") or [])),
        ("candidate", list(report.get("candidates") or [])),
    )
    for source, rows in candidate_groups:
        for row in rows:
            audio_path = _resolve_existing_artifact_path(row.get("audio_path"))
            if audio_path is None:
                run_dir = row.get("run_dir")
                if run_dir:
                    try:
                        resolved_run_dir = _resolve_run_scoped_path(run_dir)
                    except ValueError:
                        resolved_run_dir = None
                    if resolved_run_dir and resolved_run_dir.exists() and resolved_run_dir.is_dir():
                        audio_path = _candidate_master_audio(resolved_run_dir)
            if audio_path is not None:
                return source, row, audio_path
    return None


def _promote_fuse_candidate_output(*, run_id: str, output_dir: Path, report_path: Path, report: dict) -> dict:
    selected = _select_promotable_fuse_candidate(report)
    if selected is None:
        raise FileNotFoundError("No playable candidate audio was found in the fusion batch.")

    selection_source, candidate_row, source_audio = selected
    source_ext = source_audio.suffix.lower() or ".wav"
    promoted_audio = output_dir / f"fused_output{source_ext}"
    if source_audio.resolve() != promoted_audio.resolve():
        shutil.copy2(source_audio, promoted_audio)

    delivery_mode = "survivor" if selection_source == "survivor" else "fallback"
    summary_lines = list(report.get("summary") or [])
    if delivery_mode == "survivor":
        message = "Finished. Returning the top playable survivor from the advanced fuse pipeline."
    else:
        message = "Finished. No candidate fully cleared the listener gate, so this is the best playable fallback from the batch."

    result_payload = {
        "run_id": run_id,
        "delivery_mode": delivery_mode,
        "selection_source": selection_source,
        "selected_candidate_id": candidate_row.get("candidate_id") or candidate_row.get("label"),
        "selected_candidate_label": candidate_row.get("label") or candidate_row.get("candidate_id"),
        "audio_path": str(promoted_audio),
        "audio_url": f"/api/artifact?path={promoted_audio}",
        "source_audio_path": str(source_audio),
        "report_path": str(report_path),
        "top_reasons": list(candidate_row.get("top_reasons") or [])[:3],
        "top_fixes": list(candidate_row.get("top_fixes") or [])[:3],
        "summary": summary_lines[:4],
        "message": message,
    }
    result_path = output_dir / "fuse_result.json"
    result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")
    result_payload["result_path"] = str(result_path)
    result_payload["result_url"] = f"/api/artifact?path={result_path}"
    return result_payload


def _promote_direct_fusion_output(*, run_id: str, output_dir: Path, fusion_dir: Path, fallback_reason: str, report_path: Path | None = None) -> dict:
    source_audio = _candidate_master_audio(fusion_dir)
    if source_audio is None:
        raise FileNotFoundError("Deterministic fallback fusion did not produce a playable audio artifact.")

    source_ext = source_audio.suffix.lower() or ".wav"
    promoted_audio = output_dir / f"fused_output{source_ext}"
    if source_audio.resolve() != promoted_audio.resolve():
        shutil.copy2(source_audio, promoted_audio)

    product_flow = _product_flow_summary()
    critic_loop = _critic_loop_summary(
        latest_eval=_latest_evaluator_result(),
        latest_compare=_latest_compare_listen_result(),
        latest_benchmark=_latest_benchmark_listen_result(),
        latest_listener_agent=_latest_listener_agent_result(),
        latest_closed_loop=_latest_closed_loop_result(),
        latest_auto_shortlist=_latest_auto_shortlist_result(),
    )

    result_payload = {
        "run_id": run_id,
        "delivery_mode": "deterministic_fallback",
        "selection_source": "direct_fusion",
        "selected_candidate_id": "direct_fusion",
        "selected_candidate_label": "deterministic fusion fallback",
        "product_mode": product_flow["mode"],
        "product_label": product_flow["label"],
        "product_headline": product_flow["headline"],
        "audio_path": str(promoted_audio),
        "audio_url": f"/api/artifact?path={promoted_audio}",
        "source_audio_path": str(source_audio),
        "report_path": str(report_path) if report_path else None,
        "top_reasons": [],
        "top_fixes": [],
        "summary": [fallback_reason],
        "message": "Finished. The baseline-first fuse path returned one playable output immediately; critic-loop diagnostics stay visible on the status/share surfaces.",
        "fallback_reason": fallback_reason,
        "fallback_run_dir": str(fusion_dir),
        "critic_loop": critic_loop,
    }
    result_path = output_dir / "fuse_result.json"
    result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")
    result_payload["result_path"] = str(result_path)
    result_payload["result_url"] = f"/api/artifact?path={result_path}"
    return result_payload


def _run_listen_report_path(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*listen*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        payload = _load_json_file(candidate)
        if isinstance(payload, dict) and "overall_score" in payload and "transition" in payload:
            return candidate
    return None


def _derive_run_worst_moments(run_dir: Path) -> list[dict]:
    moments: list[dict] = []
    manifest = _load_run_manifest(run_dir) or {}
    listen_path = _run_listen_report_path(run_dir)
    listen_payload = _load_json_file(listen_path) if listen_path else None
    sections = list(manifest.get("sections") or [])
    for moment in (((listen_payload or {}).get("transition") or {}).get("details") or {}).get("worst_moments") or []:
        payload = dict(moment)
        payload.setdefault("source", "listen")
        moments.append(payload)
    risky_sections = (((listen_payload or {}).get("mix_sanity") or {}).get("details") or {}).get("manifest_metrics") or {}
    risky_sections = risky_sections.get("risky_sections") or []
    for row in risky_sections[:3]:
        idx = int(row.get("section_index", 0) or 0)
        section = sections[idx] if 0 <= idx < len(sections) else {}
        target = section.get("target") or {}
        start_sec = float(target.get("start_sec", section.get("start_sec", idx * 8.0)) or idx * 8.0)
        end_sec = float(target.get("end_sec", section.get("end_sec", start_sec + 8.0)) or (start_sec + 8.0))
        moments.append({
            "kind": "manifest_risky_section",
            "component": "mix_sanity",
            "section_index": idx,
            "label": str(section.get("label") or row.get("label") or f"section_{idx}"),
            "start_time": round(start_sec, 3),
            "end_time": round(end_sec, 3),
            "center_time": round((start_sec + end_sec) * 0.5, 3),
            "duration_seconds": round(max(0.0, end_sec - start_sec), 3),
            "severity": round(float(row.get("risk", 0.0) or 0.0), 3),
            "summary": f"{section.get('label') or row.get('label') or f'section_{idx}'} has elevated manifest risk",
            "evidence": row,
            "source": "manifest",
        })
    moments.sort(key=lambda item: float(item.get("severity", 0.0) or 0.0), reverse=True)
    deduped: list[dict] = []
    seen: set[tuple[str, float]] = set()
    for moment in moments:
        key = (str(moment.get("kind") or "unknown"), round(float(moment.get("center_time", 0.0) or 0.0), 1))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(moment)
        if len(deduped) >= 6:
            break
    return deduped


def _latest_manifest_summary() -> dict | None:
    match = _find_latest_json(lambda path, payload: path.name == "render_manifest.json" and "sections" in payload and "work_orders" in payload)
    if not match:
        return None
    path, payload = match
    result = _artifact_entry(path)
    result["diagnostics"] = _summarize_manifest_diagnostics(payload)
    outputs = payload.get("outputs") or {}
    result["outputs"] = {
        key: {
            "path": value,
            "download_url": f"/api/artifact?path={value}",
        }
        for key, value in outputs.items()
        if value
    }
    return result


def _latest_run_summary() -> dict | None:
    manifest = _latest_manifest_summary()
    listen = _latest_listen_result()
    if not manifest and not listen:
        return None
    reference = manifest or listen
    run_dir_name = reference.get("run_dir")
    if not run_dir_name:
        return reference
    run_dir = RUNS_DIR / run_dir_name
    if not run_dir.exists():
        return reference
    files = sorted([p for p in run_dir.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    summary = {
        "run_dir": run_dir_name,
        "path": str(run_dir),
        "artifact_count": len(files),
        "artifacts": [_artifact_entry(p) for p in files[:8]],
    }
    if manifest:
        summary["manifest"] = manifest
    if listen and listen.get("run_dir") == run_dir_name:
        summary["listen"] = listen
    return summary


def _latest_evaluator_result() -> dict | None:
    return _latest_listen_result()


def _active_sprint_status() -> dict | None:
    status_path = RUNS_DIR / "active_sprint_status.json"
    payload = _load_json_file(status_path)
    if not isinstance(payload, dict):
        return None
    active = bool(payload.get("active"))
    status = "active" if active else "idle"
    if payload.get("ended_at") and not active:
        status = "completed"
    return {
        "active": active,
        "status": status,
        "task": payload.get("task"),
        "started_at": payload.get("started_at"),
        "last_heartbeat": payload.get("last_heartbeat"),
        "ended_at": payload.get("ended_at"),
        "path": str(status_path),
        "relative_path": _relative_to_runs(status_path),
    }


def _build_workloop_visualization(*, latest_commit: dict, latest_manifest: dict | None, latest_eval: dict | None, latest_compare: dict | None, latest_benchmark: dict | None, latest_listener_agent: dict | None, latest_run: dict | None, active_sprint: dict | None) -> dict:
    manifest_done = latest_manifest is not None
    listen_done = latest_eval is not None
    compare_done = latest_compare is not None
    benchmark_done = latest_benchmark is not None
    listener_done = latest_listener_agent is not None
    run_done = latest_run is not None

    stages = [
        {
            "key": "code",
            "label": "Code checkpoint",
            "state": "done" if latest_commit.get("hash") else "pending",
            "detail": latest_commit.get("message") or "No git checkpoint yet.",
        },
        {
            "key": "render",
            "label": "Render manifest",
            "state": "done" if manifest_done else "pending",
            "detail": (latest_manifest or {}).get("relative_path") or "No render manifest yet.",
        },
        {
            "key": "listen",
            "label": "Listen eval",
            "state": "done" if listen_done else "pending",
            "detail": (
                f"score {(latest_eval or {}).get('overall_score', '—')} · {(latest_eval or {}).get('verdict', 'unknown')}"
                if listen_done else "No listen result yet."
            ),
        },
        {
            "key": "compare",
            "label": "Compare",
            "state": "done" if compare_done else "pending",
            "detail": (
                f"winner {(latest_compare or {}).get('overall_winner', 'tie')}"
                if compare_done else "No compare-listen result yet."
            ),
        },
        {
            "key": "benchmark",
            "label": "Benchmark",
            "state": "done" if benchmark_done else "pending",
            "detail": (
                f"winner {(latest_benchmark or {}).get('winner', '—')}"
                if benchmark_done else "No benchmark-listen result yet."
            ),
        },
        {
            "key": "gate",
            "label": "Listener gate",
            "state": "done" if listener_done else "pending",
            "detail": (
                f"survivors {(latest_listener_agent or {}).get('recommended_count', 0)} / rejected {(latest_listener_agent or {}).get('rejected_count', 0)}"
                if listener_done else "No listener-agent checkpoint yet."
            ),
        },
        {
            "key": "artifact",
            "label": "Run artifact",
            "state": "done" if run_done else "pending",
            "detail": (latest_run or {}).get("run_dir") or (latest_run or {}).get("relative_path") or "No run summary yet.",
        },
    ]

    completed = sum(1 for stage in stages if stage["state"] == "done")
    percent = round((completed / len(stages)) * 100) if stages else 0
    current_stage = next((stage for stage in stages if stage["state"] != "done"), stages[-1] if stages else None)

    return {
        "headline": "Closed-loop status UI",
        "status": (active_sprint or {}).get("status") or ("active" if completed and completed < len(stages) else "idle"),
        "progress_percent": percent,
        "completed_stage_count": completed,
        "total_stage_count": len(stages),
        "current_stage": current_stage,
        "stages": stages,
        "active_sprint": active_sprint,
    }


def _workloop_status() -> dict:
    task = _extract_current_task()
    latest_commit = _latest_commit()
    latest_eval = _latest_evaluator_result()
    latest_compare = _latest_compare_listen_result()
    latest_benchmark = _latest_benchmark_listen_result()
    latest_listener_agent = _latest_listener_agent_result()
    latest_auto_shortlist = _latest_auto_shortlist_result()
    latest_closed_loop = _latest_closed_loop_result()
    latest_artifact = _latest_artifact()
    latest_manifest = _latest_manifest_summary()
    latest_run = _latest_run_summary()
    active_sprint = _active_sprint_status()
    changed = _changed_files()
    product_flow = _product_flow_summary()
    critic_loop = _critic_loop_summary(
        latest_eval=latest_eval,
        latest_compare=latest_compare,
        latest_benchmark=latest_benchmark,
        latest_listener_agent=latest_listener_agent,
        latest_closed_loop=latest_closed_loop,
        latest_auto_shortlist=latest_auto_shortlist,
    )
    workloop_viz = _build_workloop_visualization(
        latest_commit=latest_commit,
        latest_manifest=latest_manifest,
        latest_eval=latest_eval,
        latest_compare=latest_compare,
        latest_benchmark=latest_benchmark,
        latest_listener_agent=latest_listener_agent,
        latest_run=latest_run,
        active_sprint=active_sprint,
    )

    return {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "current_task": task["summary"],
        "task_source": task["source"],
        "recent_progress_notes": task["details"],
        "currently_working_on": "baseline-first delivery plus critic-loop decisions/results that map back to the next quality improvement",
        "product_flow": product_flow,
        "critic_loop": critic_loop,
        "latest_changed_files": changed,
        "latest_commit": latest_commit,
        "last_artifact": latest_artifact,
        "latest_evaluator_result": latest_eval,
        "latest_compare_listen_result": latest_compare,
        "latest_benchmark_listen_result": latest_benchmark,
        "latest_listener_agent_result": latest_listener_agent,
        "latest_auto_shortlist_result": latest_auto_shortlist,
        "latest_closed_loop_result": latest_closed_loop,
        "latest_manifest": latest_manifest,
        "latest_run_summary": latest_run,
        "active_sprint": active_sprint,
        "workloop_visualization": workloop_viz,
        "links": {
            "fuse_ui": "/",
            "debug_ui": "/debug",
            "status_ui": "/status",
            "listener_agent_api": "/api/listener-agent",
            "auto_shortlist_api": "/api/auto-shortlist-fusion",
            "closed_loop_api": "/api/closed-loop",
            "benchmark_spec_api": "/api/benchmark-spec",
        },
    }


@app.route("/")
def index():
    return send_from_directory(TEMPLATES_DIR, "simple_fuse.html")


@app.route("/song-rater")
def song_rater_page():
    return send_from_directory(TEMPLATES_DIR, "song_rater.html")


@app.route("/debug")
def debug_index():
    return send_from_directory(TEMPLATES_DIR, "prototype_debug.html")


@app.route("/status")
def status_page():
    return send_from_directory(TEMPLATES_DIR, "status.html")


@app.route("/updates")
def updates_page():
    return send_from_directory(TEMPLATES_DIR, "updates.html")


@app.route("/api/songs")
def list_songs():
    songs = load_songs()
    return jsonify({"status": "success", "songs": songs, "count": len(songs)})


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/api/rate-song", methods=["POST"])
def api_rate_song():
    song = request.files.get("song")
    if song is None:
        return jsonify({"status": "error", "error": "One audio file is required."}), 400

    allowed = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
    ext = Path(song.filename or "song").suffix.lower()
    if ext not in allowed:
        return jsonify({"status": "error", "error": "Only MP3/WAV/FLAC/M4A/AAC files are supported."}), 400

    run_id = f"song_rating_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    upload_dir = UPLOADS_DIR / run_id
    report_dir = RUNS_DIR / "song_ratings" / run_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    song_path = upload_dir / f"song{ext}"
    song.save(song_path)

    try:
        dna = analyze_audio_file(str(song_path))
        report = evaluate_song(dna).to_dict()
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"status": "error", "error": f"Song rating failed: {exc}"}), 500

    overall_score = round(float(report.get("overall_score") or 0.0), 1)
    chart_calibrated_score = _chart_calibrated_score(report)
    report["chart_calibrated_score"] = chart_calibrated_score

    report_path = report_dir / "listen_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return jsonify(
        {
            "status": "success",
            "song_name": Path(song.filename or "song").name,
            "overall_score": overall_score,
            "chart_calibrated_score": chart_calibrated_score,
            "verdict": report.get("verdict"),
            "components": {key: report.get(key, {}) for key in ("structure", "groove", "energy_arc", "transition", "coherence", "mix_sanity", "song_likeness")},
            "report_path": str(report_path),
            "run_id": run_id,
        }
    )


@app.route("/fuse", methods=["POST"])
def start_simple_fuse_job():
    song_a = request.files.get("song_a")
    song_b = request.files.get("song_b")

    if song_a is None or song_b is None:
        return jsonify({"status": "error", "error": "Two audio files are required."}), 400

    allowed = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
    ext_a = Path(song_a.filename or "song_a").suffix.lower()
    ext_b = Path(song_b.filename or "song_b").suffix.lower()
    if ext_a not in allowed or ext_b not in allowed:
        return jsonify({"status": "error", "error": "Only common audio files like MP3/WAV/FLAC/M4A/AAC are supported."}), 400

    _prune_simple_fuse_jobs()
    job_id = uuid.uuid4().hex[:8]
    run_id = f"simple_fuse_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = RUNS_DIR / run_id
    upload_dir = UPLOADS_DIR / run_id
    fusion_dir = outdir / "fusion"
    outdir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)

    song_a_path = upload_dir / f"song_a{ext_a}"
    song_b_path = upload_dir / f"song_b{ext_b}"
    song_a.save(song_a_path)
    song_b.save(song_b_path)

    _set_simple_fuse_job(
        job_id,
        status="running",
        progress=10,
        message="Uploads received. Queueing pro fuse job.",
        stage="uploaded",
        run_id=run_id,
        output_dir=str(outdir),
        song_a_name=Path(song_a.filename or "song_a").name,
        song_b_name=Path(song_b.filename or "song_b").name,
        product_mode="baseline_first",
        product_label="Baseline-first simple fuse",
    )

    worker = threading.Thread(
        target=_run_simple_fuse_job,
        args=(job_id, song_a_path, song_b_path, run_id, outdir, fusion_dir),
        daemon=True,
    )
    worker.start()
    return jsonify({"status": "accepted", "job_id": job_id, "status_url": f"/status/{job_id}", "share_url": f"/share/{job_id}"})


@app.route("/status/<job_id>")
def simple_fuse_job_status(job_id: str):
    job = _get_simple_fuse_job(job_id)
    if not job:
        return jsonify({"status": "error", "error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/share/<job_id>")
def simple_fuse_share(job_id: str):
    job = _get_simple_fuse_job(job_id)
    if not job or job.get("status") != "done":
        return "Share link not ready or invalid.", 404
    result = job.get("result") or {}
    return render_template("simple_fuse_share.html", job=job, result=result)


@app.route("/api/status")
def api_status():
    return jsonify(_workloop_status())


@app.route("/api/updates")
def api_updates():
    entries = []
    try:
        if TOOLS_LOG_PATH.exists():
            lines = [line.strip() for line in TOOLS_LOG_PATH.read_text(encoding='utf-8').splitlines() if line.strip().startswith('- ')]
            for line in lines[-30:]:
                entries.append(line[2:])
    except Exception:
        entries = []
    return jsonify({"status": "ok", "entries": list(reversed(entries))})


@app.route("/api/listener-agent", methods=["POST"])
def api_listener_agent():
    data = request.get_json(silent=True) or {}
    raw_inputs = data.get("inputs") or []
    try:
        shortlist = max(1, min(int(data.get("shortlist", 3) or 3), 10))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "error": "listener-agent shortlist must be an integer between 1 and 10."}), 400

    if raw_inputs:
        inputs: list[str] = []
        for raw in raw_inputs:
            path = Path(str(raw)).expanduser().resolve()
            try:
                path.relative_to(RUNS_DIR.resolve())
            except ValueError:
                return jsonify({"status": "error", "error": "listener-agent inputs must stay inside runs/"}), 403
            if not path.exists():
                return jsonify({"status": "error", "error": f"listener-agent input not found: {path}"}), 404
            inputs.append(str(path))
    else:
        inputs = _recent_render_inputs(limit=max(shortlist + 1, 3))

    if not inputs:
        return jsonify({"status": "error", "error": "No render outputs were found for listener-agent evaluation."}), 404

    report_dir = RUNS_DIR / "listener_agent"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"listener_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    cmd = [
        str(PROJECT_PYTHON if PROJECT_PYTHON.exists() else "python3"),
        str(BASE_DIR / "ai_dj.py"),
        "listener-agent",
        *inputs,
        "--shortlist",
        str(shortlist),
        "--output",
        str(output_path),
    ]

    try:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "error": "Listener agent timed out.", "inputs": inputs}), 500

    if proc.returncode != 0:
        return jsonify(
            {
                "status": "error",
                "error": "Listener agent failed.",
                "inputs": inputs,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        ), 500

    payload = _load_json_file(output_path) or {}
    return jsonify(
        {
            "status": "success",
            "inputs": inputs,
            "input_count": len(inputs),
            "shortlist": shortlist,
            "report_path": str(output_path),
            "report": payload,
            "stdout": proc.stdout,
        }
    )


@app.route("/api/recent-renders")
def api_recent_renders():
    runs: list[dict] = []
    for raw in _recent_render_inputs(limit=20):
        run_dir = Path(raw)
        audio = _candidate_master_audio(run_dir)
        runs.append({
            "run_dir": str(run_dir),
            "name": run_dir.name,
            "audio_path": str(audio) if audio else None,
            "audio_url": f"/api/artifact?path={audio}" if audio else None,
            "listen_report_path": str(_run_listen_report_path(run_dir)) if _run_listen_report_path(run_dir) else None,
            "worst_moments": _derive_run_worst_moments(run_dir)[:3],
        })
    return jsonify({"status": "success", "runs": runs, "count": len(runs)})


@app.route("/api/run-diagnostics")
def api_run_diagnostics():
    raw = request.args.get("run_dir", "")
    if not raw:
        return jsonify({"status": "error", "error": "run_dir is required"}), 400
    try:
        run_dir = _resolve_run_scoped_path(raw)
    except ValueError:
        return jsonify({"status": "error", "error": "run_dir must stay inside runs/"}), 403
    if not run_dir.exists() or not run_dir.is_dir():
        return jsonify({"status": "error", "error": "run_dir not found"}), 404
    manifest = _load_run_manifest(run_dir)
    listen_path = _run_listen_report_path(run_dir)
    listen_payload = _load_json_file(listen_path) if listen_path else None
    return jsonify({
        "status": "success",
        "run_dir": str(run_dir),
        "audio_path": str(_candidate_master_audio(run_dir)) if _candidate_master_audio(run_dir) else None,
        "audio_url": f"/api/artifact?path={_candidate_master_audio(run_dir)}" if _candidate_master_audio(run_dir) else None,
        "manifest_path": str(run_dir / 'render_manifest.json') if (run_dir / 'render_manifest.json').exists() else None,
        "listen_report_path": str(listen_path) if listen_path else None,
        "manifest": manifest,
        "listen_report": listen_payload,
        "worst_moments": _derive_run_worst_moments(run_dir),
    })


@app.route("/api/compare-listen", methods=["POST"])
def api_compare_listen():
    data = request.get_json(silent=True) or {}
    raw_left = data.get("left")
    raw_right = data.get("right")
    if not raw_left or not raw_right:
        return jsonify({"status": "error", "error": "left and right are required."}), 400
    try:
        left = _resolve_run_scoped_path(raw_left)
        right = _resolve_run_scoped_path(raw_right)
    except ValueError:
        return jsonify({"status": "error", "error": "compare inputs must stay inside runs/"}), 403
    if not left.exists() or not right.exists():
        return jsonify({"status": "error", "error": "compare input not found."}), 404
    compare_dir = RUNS_DIR / "compare_listen"
    compare_dir.mkdir(parents=True, exist_ok=True)
    output_path = compare_dir / f"listen_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    cmd = [
        str(PROJECT_PYTHON if PROJECT_PYTHON.exists() else "python3"),
        str(BASE_DIR / "ai_dj.py"),
        "compare-listen",
        str(left),
        str(right),
        "--output",
        str(output_path),
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "error": "compare-listen timed out."}), 500
    if proc.returncode != 0:
        return jsonify({"status": "error", "error": "compare-listen failed.", "stdout": proc.stdout, "stderr": proc.stderr}), 500
    payload = _load_json_file(output_path) or {}
    return jsonify({"status": "success", "left": str(left), "right": str(right), "report_path": str(output_path), "report": payload, "stdout": proc.stdout})


@app.route("/api/human-feedback/render", methods=["POST"])
def api_human_feedback_render():
    data = request.get_json(silent=True) or {}
    raw_run_dir = data.get("run_dir")
    if not raw_run_dir:
        return jsonify({"status": "error", "error": "run_dir is required."}), 400
    try:
        run_dir = _resolve_run_scoped_path(raw_run_dir)
    except ValueError:
        return jsonify({"status": "error", "error": "run_dir must stay inside runs/."}), 403
    if not run_dir.exists() or not run_dir.is_dir():
        return jsonify({"status": "error", "error": "run_dir not found."}), 404
    overall_label = str(data.get("overall_label") or data.get("decision") or "borderline").strip().lower()
    if overall_label not in {"reject", "borderline", "promising", "favorite", "keep"}:
        return jsonify({"status": "error", "error": "overall_label must be reject, borderline, promising, favorite, or keep."}), 400
    tags = [str(tag).strip() for tag in (data.get("tags") or []) if str(tag).strip()]
    event = _feedback_store().append_event({
        "type": "render",
        "reviewer": str(data.get("reviewer") or "human"),
        "run_dir": str(run_dir),
        "artifact_path": str(data.get("artifact_path") or _candidate_master_audio(run_dir) or ""),
        "overall_label": overall_label,
        "tags": tags,
        "note": str(data.get("note") or "").strip(),
        "timestamp_sec": data.get("timestamp_sec"),
        "component_ratings": data.get("component_ratings") or {},
    })
    per_run_path = run_dir / "human_feedback.json"
    per_run_path.write_text(json.dumps(event, indent=2, sort_keys=True), encoding="utf-8")
    return jsonify({"status": "success", "feedback": event, "feedback_path": str(per_run_path)})


@app.route("/api/human-feedback/pairwise", methods=["POST"])
def api_human_feedback_pairwise():
    data = request.get_json(silent=True) or {}
    raw_left = data.get("left_run_dir")
    raw_right = data.get("right_run_dir")
    winner = str(data.get("winner") or "").strip().lower()
    if not raw_left or not raw_right:
        return jsonify({"status": "error", "error": "left_run_dir and right_run_dir are required."}), 400
    if winner not in {"left", "right", "tie"}:
        return jsonify({"status": "error", "error": "winner must be left, right, or tie."}), 400
    try:
        left = _resolve_run_scoped_path(raw_left)
        right = _resolve_run_scoped_path(raw_right)
    except ValueError:
        return jsonify({"status": "error", "error": "pairwise inputs must stay inside runs/."}), 403
    event = _feedback_store().append_event({
        "type": "pairwise",
        "reviewer": str(data.get("reviewer") or "human"),
        "left_run_dir": str(left),
        "right_run_dir": str(right),
        "winner": winner,
        "confidence": str(data.get("confidence") or "normal"),
        "tags": [str(tag).strip() for tag in (data.get("tags") or []) if str(tag).strip()],
        "note": str(data.get("note") or "").strip(),
        "component_preferences": data.get("component_preferences") or {},
        "compare_artifact_path": str(data.get("compare_artifact_path") or "").strip(),
    })
    return jsonify({"status": "success", "feedback": event})


@app.route("/api/human-feedback")
def api_human_feedback_list():
    feedback_type = request.args.get("type") or None
    run_dir = request.args.get("run_dir") or None
    reviewer = request.args.get("reviewer") or None
    limit = int(request.args.get("limit", "100") or 100)
    events = _feedback_store().list_events(feedback_type=feedback_type, run_dir=run_dir, reviewer=reviewer, limit=limit)
    return jsonify({"status": "success", "events": events, "count": len(events), "summary": _feedback_store().summarize()})


@app.route("/api/human-feedback/learning")
def api_human_feedback_learning():
    limit = int(request.args.get("limit", "5000") or 5000)
    payload = build_feedback_learning_summary(HUMAN_FEEDBACK_DIR, limit=limit)
    return jsonify({"status": "success", "learning": payload})


@app.route("/api/human-feedback/learning/distill", methods=["POST"])
def api_human_feedback_learning_distill():
    limit = int((request.get_json(silent=True) or {}).get("limit", 5000) or 5000)
    output_path = HUMAN_FEEDBACK_DIR / "learning_snapshot.json"
    payload = write_feedback_learning_summary(HUMAN_FEEDBACK_DIR, output_path, limit=limit)
    return jsonify({"status": "success", "learning": payload, "output_path": str(output_path)})


@app.route("/api/benchmark-spec", methods=["POST"])
def api_benchmark_spec():
    data = request.get_json(silent=True) or {}

    case_keys = {
        "cases": "cases_raw",
        "reference_cases": "reference_cases_raw",
        "good_cases": "good_cases_raw",
        "review_cases": "review_cases_raw",
        "bad_cases": "bad_cases_raw",
    }

    def _normalize_case_collection(raw_values, *, bucket: str) -> list[str] | tuple[dict, int]:
        if raw_values in (None, []):
            return []
        if not isinstance(raw_values, list):
            return jsonify({"status": "error", "error": f"{bucket} must be a JSON array."}), 400
        normalized: list[str] = []
        for index, item in enumerate(raw_values, start=1):
            if not isinstance(item, dict):
                return jsonify({"status": "error", "error": f"{bucket}[{index}] must be an object with label and path."}), 400
            label = str(item.get("label") or "").strip()
            raw_path = item.get("path")
            if not label or raw_path in (None, ""):
                return jsonify({"status": "error", "error": f"{bucket}[{index}] requires non-empty label and path."}), 400
            path = Path(str(raw_path)).expanduser().resolve()
            try:
                path.relative_to(RUNS_DIR.resolve())
            except ValueError:
                return jsonify({"status": "error", "error": f"{bucket}[{index}] path must stay inside runs/: {path}"}), 403
            if not path.exists():
                return jsonify({"status": "error", "error": f"{bucket}[{index}] path not found: {path}"}), 404
            normalized.append(f"{label}={path}")
        return normalized

    spec_kwargs: dict[str, object] = {}
    for key, target in case_keys.items():
        normalized = _normalize_case_collection(data.get(key), bucket=key)
        if isinstance(normalized, tuple):
            return normalized
        spec_kwargs[target] = normalized

    for request_key, target_key in (
        ("expected_order", "expected_order_raw"),
        ("gating_expectations", "gating_expectations"),
        ("verdict_expectations", "verdict_expectations"),
        ("overall_at_least", "overall_at_least"),
        ("overall_at_most", "overall_at_most"),
        ("component_at_least", "component_at_least"),
        ("component_at_most", "component_at_most"),
        ("metric_at_least", "metric_at_least"),
        ("metric_at_most", "metric_at_most"),
        ("better_than", "better_than_raw"),
    ):
        value = data.get(request_key)
        if request_key == "expected_order":
            if value is None:
                spec_kwargs[target_key] = None
            elif isinstance(value, list):
                spec_kwargs[target_key] = ",".join(str(item).strip() for item in value if str(item).strip())
            elif isinstance(value, str):
                spec_kwargs[target_key] = value
            else:
                return jsonify({"status": "error", "error": "expected_order must be a JSON array or comma-separated string."}), 400
            continue
        if value is None:
            spec_kwargs[target_key] = []
        elif isinstance(value, list):
            spec_kwargs[target_key] = [str(item) for item in value]
        else:
            return jsonify({"status": "error", "error": f"{request_key} must be a JSON array."}), 400

    try:
        payload = build_spec(**spec_kwargs)
    except SpecBuildError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400

    output_dir = RUNS_DIR / "benchmark_specs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"benchmark_spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return jsonify(
        {
            "status": "success",
            "report_path": str(output_path),
            "case_count": len(payload.get("cases") or []),
            "expected_order": payload.get("expected_order") or [],
            "report": payload,
        }
    )


@app.route("/api/closed-loop", methods=["POST"])
def api_closed_loop():
    data = request.get_json(silent=True) or {}
    raw_song_a = data.get("song_a")
    raw_song_b = data.get("song_b")
    raw_references = data.get("references") or []

    if not raw_song_a or not raw_song_b:
        return jsonify({"status": "error", "error": "closed-loop requires song_a and song_b paths."}), 400
    if not isinstance(raw_references, list) or not raw_references:
        return jsonify({"status": "error", "error": "closed-loop requires one or more reference paths."}), 400

    def _resolve_audio_input(raw: str, *, label: str) -> Path | tuple[dict, int]:
        path = Path(str(raw)).expanduser().resolve()
        try:
            path.relative_to(MUSIC_DIR.resolve())
        except ValueError:
            try:
                path.relative_to(RUNS_DIR.resolve())
            except ValueError:
                return jsonify({"status": "error", "error": f"{label} must stay inside music/ or runs/"}), 403
        if not path.exists():
            return jsonify({"status": "error", "error": f"{label} not found: {path}"}), 404
        return path

    song_a_path = _resolve_audio_input(raw_song_a, label="song_a")
    if isinstance(song_a_path, tuple):
        return song_a_path
    song_b_path = _resolve_audio_input(raw_song_b, label="song_b")
    if isinstance(song_b_path, tuple):
        return song_b_path

    reference_roots: list[str] = []
    for index, raw in enumerate(raw_references, start=1):
        resolved = _resolve_audio_input(raw, label=f"reference[{index}]")
        if isinstance(resolved, tuple):
            return resolved
        reference_roots.append(str(resolved))

    try:
        references = normalize_reference_inputs(reference_roots)
    except ReferenceInputError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400

    for index, raw in enumerate(references, start=1):
        reference_label, reference_value = _split_labeled_path(raw)
        path = Path(reference_value).expanduser().resolve()
        try:
            path.relative_to(MUSIC_DIR.resolve())
        except ValueError:
            try:
                path.relative_to(RUNS_DIR.resolve())
            except ValueError:
                label = f"reference[{index}]"
                if reference_label:
                    label = f"reference[{index}] ({reference_label})"
                return jsonify({"status": "error", "error": f"{label} must stay inside music/ or runs/"}), 403

    max_iterations = max(1, min(int(data.get("max_iterations", 3) or 3), 10))
    quality_gate = float(data.get("quality_gate", 85.0) or 85.0)
    plateau_limit = max(1, min(int(data.get("plateau_limit", 2) or 2), 10))
    min_improvement = float(data.get("min_improvement", 0.5) or 0.5)
    target_score = float(data.get("target_score", 99.0) or 99.0)
    change_command = data.get("change_command")
    test_command = data.get("test_command")
    change_dispatch = data.get("change_dispatch")
    test_dispatch = data.get("test_dispatch")
    if change_command and change_dispatch:
        return jsonify({"status": "error", "error": "Provide either change_command or change_dispatch, not both."}), 400
    if test_command and test_dispatch:
        return jsonify({"status": "error", "error": "Provide either test_command or test_dispatch, not both."}), 400
    if change_dispatch is not None and not isinstance(change_dispatch, dict):
        return jsonify({"status": "error", "error": "change_dispatch must be a JSON object."}), 400
    if test_dispatch is not None and not isinstance(test_dispatch, dict):
        return jsonify({"status": "error", "error": "test_dispatch must be a JSON object."}), 400

    report_root = RUNS_DIR / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_root.mkdir(parents=True, exist_ok=True)

    change_dispatch_path = None
    test_dispatch_path = None
    if change_dispatch is not None:
        change_dispatch_path = report_root / "change_dispatch.json"
        change_dispatch_path.write_text(json.dumps(change_dispatch, indent=2, sort_keys=True), encoding="utf-8")
    if test_dispatch is not None:
        test_dispatch_path = report_root / "test_dispatch.json"
        test_dispatch_path.write_text(json.dumps(test_dispatch, indent=2, sort_keys=True), encoding="utf-8")

    cmd = [
        str(PROJECT_PYTHON if PROJECT_PYTHON.exists() else "python3"),
        str(BASE_DIR / "ai_dj.py"),
        "closed-loop",
        str(song_a_path),
        str(song_b_path),
        *references,
        "--output",
        str(report_root),
        "--max-iterations",
        str(max_iterations),
        "--quality-gate",
        str(quality_gate),
        "--plateau-limit",
        str(plateau_limit),
        "--min-improvement",
        str(min_improvement),
        "--target-score",
        str(target_score),
    ]
    if change_command:
        cmd.extend(["--change-command", str(change_command)])
    if test_command:
        cmd.extend(["--test-command", str(test_command)])
    if change_dispatch_path is not None:
        cmd.extend(["--change-dispatch", str(change_dispatch_path)])
    if test_dispatch_path is not None:
        cmd.extend(["--test-dispatch", str(test_dispatch_path)])

    try:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return jsonify({
            "status": "error",
            "error": "Closed loop timed out.",
            "song_a": str(song_a_path),
            "song_b": str(song_b_path),
            "references": references,
        }), 500

    if proc.returncode != 0:
        return jsonify(
            {
                "status": "error",
                "error": "Closed loop failed.",
                "song_a": str(song_a_path),
                "song_b": str(song_b_path),
                "references": references,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        ), 500

    report_path = report_root / "closed_loop_report.json"
    payload = _load_json_file(report_path) or {}
    return jsonify(
        {
            "status": "success",
            "song_a": str(song_a_path),
            "song_b": str(song_b_path),
            "references": references,
            "config": {
                "max_iterations": max_iterations,
                "quality_gate": quality_gate,
                "plateau_limit": plateau_limit,
                "min_improvement": min_improvement,
                "target_score": target_score,
                "change_command": change_command,
                "test_command": test_command,
                "change_dispatch": change_dispatch,
                "test_dispatch": test_dispatch,
            },
            "report_path": str(report_path),
            "report": payload,
            "stdout": proc.stdout,
        }
    )


@app.route("/api/auto-shortlist-fusion", methods=["POST"])
def api_auto_shortlist_fusion():
    song_a = request.files.get("song_a")
    song_b = request.files.get("song_b")

    if song_a is None or song_b is None:
        return jsonify({"status": "error", "error": "Two audio files are required."}), 400

    allowed = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
    ext_a = Path(song_a.filename or "song_a").suffix.lower()
    ext_b = Path(song_b.filename or "song_b").suffix.lower()
    if ext_a not in allowed or ext_b not in allowed:
        return jsonify({"status": "error", "error": "Only common audio files like MP3/WAV/FLAC/M4A/AAC are supported."}), 400

    try:
        batch_size = max(1, min(int(request.form.get("batch_size", "6") or 6), 12))
        shortlist = max(1, min(int(request.form.get("shortlist", "1") or 1), 6))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "error": "batch_size and shortlist must be integers."}), 400
    variant_mode = str(request.form.get("variant_mode", "safe") or "safe").strip() or "safe"
    keep_non_survivors = str(request.form.get("keep_non_survivors", "false") or "false").strip().lower() in {"1", "true", "yes", "on"}

    run_id = f"auto_shortlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = RUNS_DIR / run_id
    upload_dir = UPLOADS_DIR / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    song_a_path = upload_dir / f"song_a{ext_a}"
    song_b_path = upload_dir / f"song_b{ext_b}"
    song_a.save(song_a_path)
    song_b.save(song_b_path)

    cmd = [
        str(PROJECT_PYTHON if PROJECT_PYTHON.exists() else "python3"),
        str(BASE_DIR / "ai_dj.py"),
        "auto-shortlist-fusion",
        str(song_a_path),
        str(song_b_path),
        "--output",
        str(outdir),
        "--batch-size",
        str(batch_size),
        "--shortlist",
        str(shortlist),
        "--variant-mode",
        variant_mode,
    ]
    if keep_non_survivors:
        cmd.append("--keep-non-survivors")

    try:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "error": "Auto shortlist fusion timed out.", "run_id": run_id, "output_dir": str(outdir)}), 500

    if proc.returncode != 0:
        return jsonify(
            {
                "status": "error",
                "error": "Auto shortlist fusion failed.",
                "run_id": run_id,
                "output_dir": str(outdir),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        ), 500

    report_path = outdir / "auto_shortlist_report.json"
    payload = _load_json_file(report_path) or {}
    recommended = list(payload.get("recommended_shortlist") or [])
    closest_misses = list(payload.get("closest_misses") or [])
    for row in recommended + closest_misses:
        audio_path = row.get("audio_path")
        audio_candidate = Path(str(audio_path)).expanduser().resolve() if audio_path else None
        row["audio_url"] = f"/api/artifact?path={audio_candidate}" if audio_candidate and audio_candidate.exists() else None
        run_dir = row.get("run_dir")
        run_candidate = Path(str(run_dir)).expanduser().resolve() if run_dir else None
        row["diagnostics_url"] = f"/api/run-diagnostics?run_dir={run_candidate}" if run_candidate and run_candidate.exists() else None
    return jsonify(
        {
            "status": "success",
            "run_id": run_id,
            "output_dir": str(outdir),
            "report_path": str(report_path),
            "report": payload,
            "recommended_shortlist": recommended,
            "closest_misses": closest_misses,
            "pruning": payload.get("pruning") or {"enabled": not keep_non_survivors},
            "stdout": proc.stdout,
        }
    )


@app.route("/api/fuse-upload", methods=["POST"])
def fuse_upload():
    song_a = request.files.get("song_a")
    song_b = request.files.get("song_b")

    if song_a is None or song_b is None:
        return jsonify({"status": "error", "error": "Two audio files are required."}), 400

    allowed = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
    ext_a = Path(song_a.filename or "song_a").suffix.lower()
    ext_b = Path(song_b.filename or "song_b").suffix.lower()
    if ext_a not in allowed or ext_b not in allowed:
        return jsonify({"status": "error", "error": "Only common audio files like MP3/WAV/FLAC/M4A/AAC are supported."}), 400

    run_id = f"simple_fuse_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = RUNS_DIR / run_id
    upload_dir = UPLOADS_DIR / run_id
    direct_fusion_dir = outdir / "fusion"
    outdir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)
    direct_fusion_dir.mkdir(parents=True, exist_ok=True)

    song_a_path = upload_dir / f"song_a{ext_a}"
    song_b_path = upload_dir / f"song_b{ext_b}"
    song_a.save(song_a_path)
    song_b.save(song_b_path)

    direct_cmd = [
        str(PROJECT_PYTHON if PROJECT_PYTHON.exists() else "python3"),
        str(BASE_DIR / "ai_dj.py"),
        "fusion",
        str(song_a_path),
        str(song_b_path),
        "--arrangement-mode",
        "pro",
        "--output",
        str(direct_fusion_dir),
    ]

    try:
        direct_proc = subprocess.run(
            direct_cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=SIMPLE_FUSE_DIRECT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return jsonify(
            {
                "status": "error",
                "error": "Fuse failed: deterministic fusion timed out.",
                "run_id": run_id,
                "output_dir": str(outdir),
            }
        ), 500

    if direct_proc.returncode != 0:
        return jsonify(
            {
                "status": "error",
                "error": "Fuse failed: deterministic fusion did not complete successfully.",
                "run_id": run_id,
                "output_dir": str(outdir),
                "stdout": direct_proc.stdout,
                "stderr": direct_proc.stderr,
            }
        ), 500

    try:
        result = _promote_direct_fusion_output(
            run_id=run_id,
            output_dir=outdir,
            fusion_dir=direct_fusion_dir,
            fallback_reason="Finished. Returning the direct deterministic fuse result.",
            report_path=None,
        )
    except FileNotFoundError as exc:
        return jsonify(
            {
                "status": "error",
                "error": str(exc),
                "run_id": run_id,
                "output_dir": str(outdir),
                "stdout": direct_proc.stdout,
                "stderr": direct_proc.stderr,
            }
        ), 500

    return jsonify(
        {
            "status": "success",
            "run_id": run_id,
            "output_dir": str(outdir),
            "report_path": None,
            "report": None,
            "result": result,
            "stdout": direct_proc.stdout,
        }
    )


@app.route("/api/artifact")
def artifact():
    path_str = request.args.get("path", "")
    if not path_str:
        return jsonify({"status": "error", "error": "path is required"}), 400
    path = Path(path_str).expanduser().resolve()
    try:
        path.relative_to(RUNS_DIR.resolve())
    except ValueError:
        return jsonify({"status": "error", "error": "artifact path must stay inside runs/"}), 403
    if not path.exists() or not path.is_file():
        return jsonify({"status": "error", "error": "artifact not found"}), 404
    return send_file(path)


@app.route("/api/prototype", methods=["POST"])
def prototype():
    data = request.get_json() or {}
    song_a_id = data.get("song_a_id", "")
    song_b_id = data.get("song_b_id", "")

    if not song_a_id or not song_b_id:
        return jsonify({"status": "error", "error": "Two songs are required."}), 400

    song_a_path = resolve_song_path(song_a_id)
    song_b_path = resolve_song_path(song_b_id)

    if song_a_path is None or song_b_path is None:
        return jsonify({"status": "error", "error": "Could not resolve one or both song paths."}), 404

    run_id = f"prototype_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = RUNS_DIR / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    stages: list[dict] = []
    artifacts: dict[str, str] = {}

    try:
        stages.append({"name": "resolve_inputs", "status": "completed", "details": {"song_a": str(song_a_path), "song_b": str(song_b_path)}})

        song_a_dna = analyze_audio_file(song_a_path)
        song_a_payload = song_a_dna.to_dict()
        song_a_file = outdir / "song_a_dna.json"
        song_a_file.write_text(json.dumps(song_a_payload, indent=2, sort_keys=True), encoding="utf-8")
        artifacts["song_a_dna"] = str(song_a_file)
        stages.append({
            "name": "analyze_song_a",
            "status": "completed",
            "details": {
                "tempo_bpm": song_a_payload["tempo_bpm"],
                "key": song_a_payload["key"],
                "duration_seconds": song_a_payload["duration_seconds"],
                "artifact": str(song_a_file),
            },
        })

        song_b_dna = analyze_audio_file(song_b_path)
        song_b_payload = song_b_dna.to_dict()
        song_b_file = outdir / "song_b_dna.json"
        song_b_file.write_text(json.dumps(song_b_payload, indent=2, sort_keys=True), encoding="utf-8")
        artifacts["song_b_dna"] = str(song_b_file)
        stages.append({
            "name": "analyze_song_b",
            "status": "completed",
            "details": {
                "tempo_bpm": song_b_payload["tempo_bpm"],
                "key": song_b_payload["key"],
                "duration_seconds": song_b_payload["duration_seconds"],
                "artifact": str(song_b_file),
            },
        })

        compatibility = build_compatibility_report(song_a_dna, song_b_dna).to_dict()
        compatibility_file = outdir / "compatibility_report.json"
        compatibility_file.write_text(json.dumps(compatibility, indent=2, sort_keys=True), encoding="utf-8")
        artifacts["compatibility_report"] = str(compatibility_file)
        stages.append({
            "name": "compatibility_report",
            "status": "completed",
            "details": {
                "overall": compatibility["factors"]["overall"],
                "factors": compatibility["factors"],
                "artifact": str(compatibility_file),
            },
        })

        arrangement = build_stub_arrangement_plan(song_a_dna, song_b_dna).to_dict()
        arrangement_file = outdir / "arrangement_plan.json"
        arrangement_file.write_text(json.dumps(arrangement, indent=2, sort_keys=True), encoding="utf-8")
        artifacts["arrangement_plan"] = str(arrangement_file)
        stages.append({
            "name": "arrangement_plan",
            "status": "completed",
            "details": {
                "section_count": len(arrangement.get("sections", [])),
                "sections": arrangement.get("sections", []),
                "artifact": str(arrangement_file),
            },
        })

        return jsonify(
            {
                "status": "success",
                "run_id": run_id,
                "output_dir": str(outdir),
                "stages": stages,
                "artifacts": artifacts,
                "artifact_payloads": {
                    "song_a_dna": song_a_payload,
                    "song_b_dna": song_b_payload,
                    "compatibility_report": compatibility,
                    "arrangement_plan": arrangement,
                },
            }
        )
    except Exception as exc:
        stages.append(
            {
                "name": "error",
                "status": "failed",
                "details": {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        return jsonify(
            {
                "status": "error",
                "error": str(exc),
                "run_id": run_id,
                "output_dir": str(outdir),
                "stages": stages,
            }
        ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("VF_DEBUG", "0") == "1"
    print(f"🎛️ VocalFusion prototype debug UI running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
