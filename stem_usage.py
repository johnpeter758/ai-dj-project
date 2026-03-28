"""Deterministic stem-selection and focal-priority helpers."""

from __future__ import annotations

from typing import Any


ROLE_ORDER = ("drums", "bass", "vocals", "music", "fx")


def _sorted_stems(stems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(s) for s in stems),
        key=lambda s: (
            -float(s.get("priority", 0)),
            -float(s.get("energy", 0)),
            str(s.get("name", "")),
        ),
    )


def _pick_one(stems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = _sorted_stems(stems)
    return ordered[:1]


def enforce_stem_conflict_rules(active: dict) -> dict:
    """Enforce anti-mud rules and stem conflicts deterministically."""
    out: dict[str, Any] = {k: [dict(s) for s in v] if isinstance(v, list) else v for k, v in active.items()}

    # hard limits for low-end and lead conflicts
    out["bass"] = _pick_one(out.get("bass", []))

    drums = _sorted_stems(out.get("drums", []))
    kicks = [s for s in drums if s.get("is_kick")]
    non_kicks = [s for s in drums if not s.get("is_kick")]
    selected_drums: list[dict[str, Any]] = []
    if kicks:
        selected_drums.extend(kicks[:1])
    selected_drums.extend(non_kicks[:2])
    out["drums"] = selected_drums

    vocals = _sorted_stems(out.get("vocals", []))
    out["vocals"] = vocals[:1]

    music = _sorted_stems(out.get("music", []))
    # Anti-mud: max two tonal music layers; avoid duplicate register collision.
    chosen_music: list[dict[str, Any]] = []
    used_registers: set[str] = set()
    for stem in music:
        register = str(stem.get("register", "mid"))
        if register in used_registers:
            continue
        chosen_music.append(stem)
        used_registers.add(register)
        if len(chosen_music) >= 2:
            break
    out["music"] = chosen_music

    # Keep fx sparse for intelligibility.
    out["fx"] = _sorted_stems(out.get("fx", []))[:1]

    return out


def apply_focal_priority(active: dict, focus: str) -> dict:
    """Apply one-primary-focus policy and duck competing focal elements."""
    focus_norm = (focus or "").lower().strip()
    out: dict[str, Any] = {k: [dict(s) for s in v] if isinstance(v, list) else v for k, v in active.items()}

    candidates: list[tuple[str, int, dict[str, Any]]] = []
    for role in ROLE_ORDER:
        for idx, stem in enumerate(out.get(role, [])):
            name = str(stem.get("name", "")).lower()
            is_focus = bool(focus_norm and (focus_norm in role or focus_norm in name))
            score = 1000 if is_focus else 0
            score += int(float(stem.get("priority", 0)) * 100)
            score += int(float(stem.get("energy", 0)) * 10)
            candidates.append((role, idx, {**stem, "_score": score, "_is_focus": is_focus}))

    primary_key: tuple[str, int] | None = None
    if candidates:
        best = sorted(candidates, key=lambda t: (-t[2]["_score"], str(t[2].get("name", ""))))[0]
        primary_key = (best[0], best[1])

    for role in ROLE_ORDER:
        stems = out.get(role, [])
        for idx, stem in enumerate(stems):
            is_primary = primary_key == (role, idx)
            stem["primary_focus"] = bool(is_primary)
            if is_primary:
                stem["gain_db"] = max(float(stem.get("gain_db", 0)), 0)
            else:
                if role in {"vocals", "music", "fx"}:
                    stem["gain_db"] = min(float(stem.get("gain_db", 0)), -4)
                else:
                    stem["gain_db"] = min(float(stem.get("gain_db", 0)), -1)

    out["focus"] = focus_norm
    return out


def select_active_stems(section_plan: dict, candidate_stems: dict) -> dict:
    """Select active stems from candidates with deterministic anti-mud logic."""
    max_layers = int(section_plan.get("max_layers", 6))
    focus = str(section_plan.get("focus", "")).lower().strip()

    active: dict[str, list[dict[str, Any]]] = {
        "drums": _sorted_stems(candidate_stems.get("drums", []))[:3],
        "bass": _sorted_stems(candidate_stems.get("bass", []))[:2],
        "vocals": _sorted_stems(candidate_stems.get("vocals", []))[:2],
        "music": _sorted_stems(candidate_stems.get("music", []))[:3],
        "fx": _sorted_stems(candidate_stems.get("fx", []))[:2],
    }

    # Apply conflict/anti-mud first, then focus policy.
    active = enforce_stem_conflict_rules(active)
    active = apply_focal_priority(active, focus)

    # Global layer cap (deterministic role priority order), while guaranteeing
    # one primary focus survives when max_layers >= 1.
    if max_layers <= 0:
        for role in ROLE_ORDER:
            active[role] = []
        return active

    flattened: list[tuple[str, dict[str, Any]]] = []
    for role in ROLE_ORDER:
        for stem in active.get(role, []):
            flattened.append((role, stem))

    primary = [item for item in flattened if item[1].get("primary_focus")]
    others = [item for item in flattened if not item[1].get("primary_focus")]
    selected = primary[:1] + others[: max(0, max_layers - len(primary[:1]))]

    for role in ROLE_ORDER:
        active[role] = [stem for r, stem in selected if r == role]

    return active
