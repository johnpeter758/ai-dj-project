#!/usr/bin/env python3
"""VocalFusion prototype debug server."""

from __future__ import annotations

import json
import os
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from src.core.analysis import analyze_audio_file
from src.core.planner import build_compatibility_report, build_stub_arrangement_plan

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
    top_survivor = recommended[0] if recommended else None
    top_reject = rejected[0] if rejected else None
    result = _artifact_entry(path)
    result.update({
        "recommended_count": len(recommended),
        "rejected_count": len(rejected),
        "winner": top_survivor.get("label") if top_survivor else None,
        "summary": list(payload.get("summary") or [])[:3],
        "top_survivor": {
            "label": top_survivor.get("label"),
            "listener_rank": top_survivor.get("listener_rank"),
            "overall_score": top_survivor.get("overall_score"),
            "verdict": top_survivor.get("verdict"),
        } if top_survivor else None,
        "top_reject_reason": ((top_reject.get("hard_fail_reasons") or [None])[0] if top_reject else None),
    })
    return result


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


def _workloop_status() -> dict:
    task = _extract_current_task()
    latest_commit = _latest_commit()
    latest_eval = _latest_evaluator_result()
    latest_compare = _latest_compare_listen_result()
    latest_benchmark = _latest_benchmark_listen_result()
    latest_listener_agent = _latest_listener_agent_result()
    latest_artifact = _latest_artifact()
    latest_manifest = _latest_manifest_summary()
    latest_run = _latest_run_summary()
    changed = _changed_files()

    return {
        "status": "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "current_task": task["summary"],
        "task_source": task["source"],
        "recent_progress_notes": task["details"],
        "currently_working_on": "planner/evaluator quality, regression safety, and usable local checkpoints",
        "latest_changed_files": changed,
        "latest_commit": latest_commit,
        "last_artifact": latest_artifact,
        "latest_evaluator_result": latest_eval,
        "latest_compare_listen_result": latest_compare,
        "latest_benchmark_listen_result": latest_benchmark,
        "latest_listener_agent_result": latest_listener_agent,
        "latest_manifest": latest_manifest,
        "latest_run_summary": latest_run,
        "links": {
            "fuse_ui": "/",
            "debug_ui": "/debug",
            "status_ui": "/status",
            "listener_agent_api": "/api/listener-agent",
        },
    }


@app.route("/")
def index():
    return send_from_directory(TEMPLATES_DIR, "simple_fuse.html")


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
    shortlist = max(1, min(int(data.get("shortlist", 3) or 3), 10))

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
    outdir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    song_a_path = upload_dir / f"song_a{ext_a}"
    song_b_path = upload_dir / f"song_b{ext_b}"
    song_a.save(song_a_path)
    song_b.save(song_b_path)

    cmd = [
        str(PROJECT_PYTHON if PROJECT_PYTHON.exists() else "python3"),
        str(BASE_DIR / "ai_dj.py"),
        "fusion",
        str(song_a_path),
        str(song_b_path),
        "--output",
        str(outdir),
    ]

    try:
        proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "error": "Fusion timed out.", "run_id": run_id, "output_dir": str(outdir)}), 500

    if proc.returncode != 0:
        return jsonify(
            {
                "status": "error",
                "error": "Fusion failed.",
                "run_id": run_id,
                "output_dir": str(outdir),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        ), 500

    artifacts = {
        "raw_wav": str(outdir / "child_raw.wav"),
        "master_wav": str(outdir / "child_master.wav"),
        "master_mp3": str(outdir / "child_master.mp3"),
        "manifest": str(outdir / "render_manifest.json"),
    }
    return jsonify(
        {
            "status": "success",
            "run_id": run_id,
            "output_dir": str(outdir),
            "artifacts": artifacts,
            "stdout": proc.stdout,
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
