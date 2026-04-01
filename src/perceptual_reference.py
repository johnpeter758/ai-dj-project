from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_audio_mono(path: str | Path, *, sr: int = 24000, duration: float = 60.0) -> np.ndarray:
    import librosa

    waveform, _ = librosa.load(str(path), sr=sr, mono=True, duration=duration)
    if waveform.size == 0:
        return np.zeros(1, dtype=np.float32)
    return np.asarray(waveform, dtype=np.float32)


class _BaseEmbedder:
    backend_name = "base"

    def embed_audio_file(self, path: str | Path) -> np.ndarray:
        raise NotImplementedError


class _MsClapEmbedder(_BaseEmbedder):
    backend_name = "msclap"

    def __init__(self) -> None:
        from msclap import CLAP

        self._model = CLAP(version="2023", use_cuda=False)

    def embed_audio_file(self, path: str | Path) -> np.ndarray:
        import os
        import soundfile as sf
        import tempfile

        waveform = _load_audio_mono(path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name
        try:
            sf.write(temp_path, waveform, 24000)
            embedding = self._model.get_audio_embeddings([temp_path])[0]
            return np.asarray(embedding, dtype=np.float32)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


class _LaionClapEmbedder(_BaseEmbedder):
    backend_name = "laion-clap"

    def __init__(self) -> None:
        import laion_clap

        self._model = laion_clap.CLAP_Module(enable_fusion=False)
        self._model.load_ckpt()

    def embed_audio_file(self, path: str | Path) -> np.ndarray:
        import os
        import soundfile as sf
        import tempfile

        waveform = _load_audio_mono(path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = handle.name
        try:
            sf.write(temp_path, waveform, 24000)
            embedding = self._model.get_audio_embedding_from_filelist([temp_path], use_tensor=False)[0]
            return np.asarray(embedding, dtype=np.float32)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


class _MERTEmbedder(_BaseEmbedder):
    backend_name = "mert"

    def __init__(self) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self._torch = torch
        self._extractor = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")
        self._model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M")
        self._model.eval()

    def embed_audio_file(self, path: str | Path) -> np.ndarray:
        waveform = _load_audio_mono(path)
        inputs = self._extractor(
            waveform,
            sampling_rate=24000,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        hidden = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return np.asarray(hidden, dtype=np.float32)


def available_backends() -> list[str]:
    backends: list[str] = []
    try:
        import msclap  # noqa: F401

        backends.append("msclap")
    except Exception:
        pass
    try:
        import laion_clap  # noqa: F401

        backends.append("laion-clap")
    except Exception:
        pass
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        backends.append("mert")
    except Exception:
        pass
    return backends


def load_embedder(preferred_backend: str | None = None) -> _BaseEmbedder:
    backend_order = []
    if preferred_backend:
        backend_order.append(preferred_backend)
    backend_order.extend([name for name in ("msclap", "laion-clap", "mert") if name not in backend_order])

    errors: list[str] = []
    for backend in backend_order:
        try:
            if backend == "msclap":
                return _MsClapEmbedder()
            if backend == "laion-clap":
                return _LaionClapEmbedder()
            if backend == "mert":
                return _MERTEmbedder()
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
    error_text = "; ".join(errors) if errors else "no perceptual backend available"
    raise RuntimeError(error_text)


def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        return np.zeros_like(array)
    return array / norm


@dataclass(slots=True)
class PerceptualReferenceResult:
    available: bool
    backend: str | None
    reference_count: int
    candidate_similarity: float | None
    top_reference_similarities: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "backend": self.backend,
            "reference_count": self.reference_count,
            "candidate_similarity": self.candidate_similarity,
            "top_reference_similarities": self.top_reference_similarities,
            "notes": self.notes,
        }


def score_candidate_against_references(
    candidate_audio_path: str | Path,
    reference_audio_paths: list[str | Path],
    *,
    preferred_backend: str | None = None,
    top_k: int = 3,
) -> PerceptualReferenceResult:
    reference_paths = [Path(path).expanduser().resolve() for path in reference_audio_paths]
    candidate_path = Path(candidate_audio_path).expanduser().resolve()

    if not candidate_path.exists():
        return PerceptualReferenceResult(
            available=False,
            backend=None,
            reference_count=0,
            candidate_similarity=None,
            top_reference_similarities=[],
            notes=[f"candidate audio missing: {candidate_path}"],
        )
    if not reference_paths:
        return PerceptualReferenceResult(
            available=False,
            backend=None,
            reference_count=0,
            candidate_similarity=None,
            top_reference_similarities=[],
            notes=["no reference audio paths available"],
        )

    try:
        embedder = load_embedder(preferred_backend)
    except Exception as exc:
        return PerceptualReferenceResult(
            available=False,
            backend=None,
            reference_count=len(reference_paths),
            candidate_similarity=None,
            top_reference_similarities=[],
            notes=[f"perceptual backend unavailable: {exc}"],
        )

    candidate_embedding = _normalize_embedding(embedder.embed_audio_file(candidate_path))
    reference_rows: list[dict[str, Any]] = []
    for path in reference_paths:
        if not path.exists():
            continue
        embedding = _normalize_embedding(embedder.embed_audio_file(path))
        similarity = float(np.dot(candidate_embedding, embedding))
        reference_rows.append(
            {
                "path": str(path),
                "label": path.name,
                "similarity": round(similarity, 4),
            }
        )

    reference_rows.sort(key=lambda item: (-float(item["similarity"]), item["label"]))
    if not reference_rows:
        return PerceptualReferenceResult(
            available=False,
            backend=embedder.backend_name,
            reference_count=0,
            candidate_similarity=None,
            top_reference_similarities=[],
            notes=["no readable reference audio paths"],
        )

    mean_similarity = sum(float(item["similarity"]) for item in reference_rows) / len(reference_rows)
    notes = [
        "Perceptual similarity is an optional second-opinion signal, not the primary musical judge.",
    ]
    return PerceptualReferenceResult(
        available=True,
        backend=embedder.backend_name,
        reference_count=len(reference_rows),
        candidate_similarity=round(mean_similarity, 4),
        top_reference_similarities=reference_rows[: max(1, int(top_k))],
        notes=notes,
    )
