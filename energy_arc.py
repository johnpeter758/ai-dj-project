from __future__ import annotations

from copy import deepcopy


_PROFILE_ANCHORS: dict[str, list[float]] = {
    "standard": [0.30, 0.45, 0.62, 0.78, 0.88],
    "slowburn": [0.22, 0.34, 0.50, 0.72, 0.90],
    "peak_early": [0.35, 0.68, 0.82, 0.74, 0.86],
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _interpolate_anchors(section_count: int, anchors: list[float]) -> list[float]:
    if section_count <= 0:
        return []
    if section_count == 1:
        return [_clamp01(anchors[-1])]

    out: list[float] = []
    max_idx = len(anchors) - 1
    for i in range(section_count):
        x = i / (section_count - 1)
        pos = x * max_idx
        lo = int(pos)
        hi = min(lo + 1, max_idx)
        frac = pos - lo
        val = anchors[lo] * (1.0 - frac) + anchors[hi] * frac
        out.append(_clamp01(round(val, 3)))
    return out


def build_energy_arc_template(section_count: int, profile: str = "standard") -> list[dict]:
    """Build a deterministic per-section target-energy template."""
    if section_count <= 0:
        return []

    anchors = _PROFILE_ANCHORS.get(profile, _PROFILE_ANCHORS["standard"])
    curve = _interpolate_anchors(section_count=section_count, anchors=anchors)

    out: list[dict] = []
    for i, target in enumerate(curve):
        idx = i + 1
        if idx == 1:
            role = "intro"
        elif idx == section_count:
            role = "finale"
        elif target >= 0.8:
            role = "peak"
        elif i > 0 and target > curve[i - 1]:
            role = "lift"
        else:
            role = "groove"

        out.append(
            {
                "section_index": idx,
                "target_energy": target,
                "role": role,
            }
        )
    return out


def apply_energy_arc_rules(sections: list[dict]) -> list[dict]:
    """Apply anti-flatness and anti-chaos corrections to section energy targets."""
    if not sections:
        return []

    normalized = deepcopy(sections)
    energies: list[float] = []
    for section in normalized:
        source = section.get("target_energy", section.get("energy", 0.5))
        energies.append(_clamp01(round(float(source), 3)))

    # Anti-flatness: avoid long near-identical runs.
    min_step = 0.06
    for i in range(1, len(energies)):
        if abs(energies[i] - energies[i - 1]) < min_step:
            direction = 1.0 if i < len(energies) - 1 else -1.0
            nudged = energies[i - 1] + direction * min_step
            energies[i] = _clamp01(round(nudged, 3))

    # Anti-chaos: cap violent jumps between adjacent sections.
    max_step = 0.22
    for i in range(1, len(energies)):
        delta = energies[i] - energies[i - 1]
        if abs(delta) > max_step:
            energies[i] = _clamp01(round(energies[i - 1] + (max_step if delta > 0 else -max_step), 3))

    # Ensure landing has payoff: last section should not be weaker than previous by much.
    if len(energies) >= 2 and energies[-1] + 0.04 < energies[-2]:
        energies[-1] = _clamp01(round(energies[-2] - 0.04, 3))

    for i, section in enumerate(normalized):
        section["target_energy"] = energies[i]
        section["energy_delta"] = 0.0 if i == 0 else round(energies[i] - energies[i - 1], 3)

    return normalized
