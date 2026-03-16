from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReferenceInputError(ValueError):
    """Raised when a reference input collection cannot be normalized."""


_COLLECTION_KEYS = ("references", "inputs", "items")


def _iter_collection_entries(payload: Any) -> list[str] | None:
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        for key in _COLLECTION_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
    return None


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
        candidate = Path(text).expanduser()
        if candidate.exists() and candidate.is_file():
            nested = _load_collection_file(candidate)
            if nested is not None:
                base_dir = candidate.parent
                for item in _flatten_reference_tokens(nested):
                    nested_path = Path(item).expanduser()
                    if not nested_path.is_absolute():
                        nested_path = (base_dir / nested_path).expanduser()
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
        path = Path(raw).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized
