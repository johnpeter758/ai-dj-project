from __future__ import annotations

from pathlib import Path

import numpy as np

from src import perceptual_reference as pr


class _DummyEmbedder:
    backend_name = "dummy"

    def __init__(self, embeddings: dict[str, np.ndarray]) -> None:
        self._embeddings = embeddings

    def embed_audio_file(self, path: str | Path) -> np.ndarray:
        return self._embeddings[Path(path).name]


def test_score_candidate_against_references_returns_unavailable_when_backend_missing(tmp_path: Path, monkeypatch) -> None:
    candidate = tmp_path / "candidate.wav"
    reference = tmp_path / "reference.wav"
    candidate.write_bytes(b"wave")
    reference.write_bytes(b"wave")

    monkeypatch.setattr(pr, "load_embedder", lambda preferred_backend=None: (_ for _ in ()).throw(RuntimeError("missing backend")))

    result = pr.score_candidate_against_references(candidate, [reference]).to_dict()
    assert result["available"] is False
    assert "missing backend" in result["notes"][0]


def test_score_candidate_against_references_ranks_reference_similarity(tmp_path: Path, monkeypatch) -> None:
    candidate = tmp_path / "candidate.wav"
    strong = tmp_path / "strong.wav"
    weak = tmp_path / "weak.wav"
    for path in (candidate, strong, weak):
        path.write_bytes(b"wave")

    embeddings = {
        "candidate.wav": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "strong.wav": np.array([0.9, 0.1, 0.0], dtype=np.float32),
        "weak.wav": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    monkeypatch.setattr(pr, "load_embedder", lambda preferred_backend=None: _DummyEmbedder(embeddings))

    result = pr.score_candidate_against_references(candidate, [strong, weak]).to_dict()
    assert result["available"] is True
    assert result["backend"] == "dummy"
    assert result["reference_count"] == 2
    assert result["top_reference_similarities"][0]["label"] == "strong.wav"
    assert result["candidate_similarity"] > 0.4


def test_load_embedder_uses_env_preference_and_cache(monkeypatch) -> None:
    class _EnvEmbedder:
        backend_name = "env"

    calls: list[str] = []

    def _fake_msclap_embedder():
        calls.append("msclap")
        return _EnvEmbedder()

    monkeypatch.setenv("AI_DJ_PERCEPTUAL_BACKEND", "msclap")
    pr._load_cached_embedder.cache_clear()
    monkeypatch.setattr(pr, "_MsClapEmbedder", _fake_msclap_embedder)

    first = pr.load_embedder()
    second = pr.load_embedder()

    assert first is second
    assert calls == ["msclap"]
