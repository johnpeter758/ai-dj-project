from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_INPUT_LABEL_DELIMITERS = ("=", "::")


class ReferenceInputError(ValueError):
    """Raised when a reference input collection cannot be normalized."""


_COLLECTION_KEYS = ("references", "inputs", "items")
_ENTRY_PATH_KEYS = (
    "input_path",
    "path",
    "report_path",
    "render_manifest_path",
    "resolved_audio_path",
)


def _split_labeled_input(raw: str) -> tuple[str | None, str]:
    text = str(raw).strip()
    for delimiter in _INPUT_LABEL_DELIMITERS:
        if delimiter not in text:
            continue
        label, candidate = text.split(delimiter, 1)
        label = label.strip()
        candidate = candidate.strip()
        if not label or not candidate:
            continue
        if candidate.startswith(("/", "./", "../", "~/")):
            return label, candidate
        if len(candidate) >= 3 and candidate[1] == ":" and candidate[2] in {"/", "\\"}:
            return label, candidate
    return None, text


def _looks_like_path_string(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    if any(sep in text for sep in ("/", "\\")):
        return True
    return Path(text).suffix.lower() in {".json", ".wav", ".mp3", ".flac", ".m4a", ".aac", ".txt", ".lst", ".list", ".refs"}


def _extract_collection_entries(payload: Any) -> list[str]:
    if isinstance(payload, str):
        text = payload.strip()
        return [text] if _looks_like_path_string(text) else []
    if isinstance(payload, list):
        entries: list[str] = []
        for item in payload:
            entries.extend(_extract_collection_entries(item))
        return entries
    if isinstance(payload, dict):
        for key in _ENTRY_PATH_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]

        entries: list[str] = []
        for key in _COLLECTION_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    entries.extend(_extract_collection_entries(item))
                if entries:
                    return entries

        for value in payload.values():
            if isinstance(value, list) and any(isinstance(item, dict) for item in value):
                for item in value:
                    entries.extend(_extract_collection_entries(item))
        return entries
    return []


def _iter_collection_entries(payload: Any) -> list[str] | None:
    entries = _extract_collection_entries(payload)
    return entries or None


def _load_collection_file(path: Path) -> list[str] | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _iter_collection_entries(payload)
    if suffix in {".txt", ".lst", ".list", ".refs"}:
        entries: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
        return entries
    return None


def _flatten_reference_tokens(raw_inputs: list[str]) -> list[str]:
    tokens: list[str] = []
    for raw in raw_inputs:
        text = str(raw).strip()
        if not text:
            continue
        _label, candidate_text = _split_labeled_input(text)
        candidate = Path(candidate_text).expanduser()
        if candidate.exists() and candidate.is_file():
            nested = _load_collection_file(candidate)
            if nested is not None:
                base_dir = candidate.parent
                for item in _flatten_reference_tokens(nested):
                    nested_label, nested_value = _split_labeled_input(item)
                    nested_path = Path(nested_value).expanduser()
                    if not nested_path.is_absolute():
                        nested_path = (base_dir / nested_path).expanduser()
                    if nested_label:
                        tokens.append(f"{nested_label}={nested_path}")
                    else:
                        tokens.append(str(nested_path))
                continue
        parts = [part.strip() for part in text.split(",")]
        tokens.extend(part for part in parts if part)
    return tokens


def normalize_reference_inputs(raw_inputs: list[str]) -> list[str]:
    flattened = _flatten_reference_tokens(raw_inputs)
    if not flattened:
        raise ReferenceInputError("at least one reference is required")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in flattened:
        label, raw_path = _split_labeled_input(raw)
        path = Path(raw_path).expanduser().resolve()
        key = f"{label}={path}" if label else str(path)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized
