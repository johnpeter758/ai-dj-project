#!/usr/bin/env python3
"""VocalFusion prototype debug server."""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src.core.analysis import analyze_audio_file
from src.core.planner import build_compatibility_report, build_stub_arrangement_plan

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
MUSIC_DIR = BASE_DIR / "music"
RUNS_DIR = BASE_DIR / "runs"


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


@app.route("/")
def index():
    return send_from_directory(TEMPLATES_DIR, "prototype_debug.html")


@app.route("/api/songs")
def list_songs():
    songs = load_songs()
    return jsonify({"status": "success", "songs": songs, "count": len(songs)})


@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})


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
    print(f"🎛️ VocalFusion prototype debug UI running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
