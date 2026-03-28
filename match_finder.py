"""Deterministic compatibility scoring for song section descriptors.

The scorer consumes dictionary-like section descriptors (typically from song_dna)
and returns explainable category scores, penalties, and a final score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}

CATEGORY_WEIGHTS: Dict[str, float] = {
    "tempo": 0.14,
    "key_harmony": 0.18,
    "rhythmic": 0.14,
    "energy": 0.14,
    "structural_role": 0.10,
    "transition_compatibility": 0.14,
    "contrast_value": 0.08,
    "payoff_potential": 0.08,
}

PENALTY_WEIGHTS: Dict[str, float] = {
    "clashing_hooks": 1.0,
    "dense_overlap": 0.9,
    "conflicting_bass_movement": 0.9,
    "rhythmic_fighting": 1.0,
    "weak_payoff": 0.8,
    "awkward_phrasing": 1.0,
}

ROLE_COMPATIBILITY: Dict[Tuple[str, str], float] = {
    ("intro", "intro"): 95,
    ("intro", "verse"): 88,
    ("intro", "pre_chorus"): 70,
    ("intro", "chorus"): 60,
    ("verse", "verse"): 90,
    ("verse", "pre_chorus"): 92,
    ("verse", "chorus"): 82,
    ("pre_chorus", "chorus"): 95,
    ("pre_chorus", "verse"): 70,
    ("chorus", "chorus"): 90,
    ("bridge", "chorus"): 88,
    ("bridge", "bridge"): 82,
    ("breakdown", "drop"): 88,
    ("outro", "outro"): 96,
    ("chorus", "outro"): 76,
}

PAIRING_PROFILES: Dict[str, Dict[str, Any]] = {
    "intro_options": {
        "source_roles": {"intro"},
        "target_roles": {"intro", "verse", "pre_chorus"},
        "weights": {"tempo": 1.1, "transition_compatibility": 1.2, "energy": 0.8},
    },
    "first_vocal_entry_options": {
        "source_roles": {"intro", "verse"},
        "target_roles": {"verse", "pre_chorus"},
        "weights": {"structural_role": 1.2, "rhythmic": 1.1, "awkward_phrasing": 1.2},
    },
    "chorus_payoff_options": {
        "source_roles": {"pre_chorus", "verse", "build"},
        "target_roles": {"chorus", "drop"},
        "weights": {"payoff_potential": 1.4, "transition_compatibility": 1.2, "weak_payoff": 1.3},
    },
    "swap_moments": {
        "source_roles": {"verse", "chorus", "bridge", "breakdown"},
        "target_roles": {"verse", "chorus", "bridge", "breakdown"},
        "weights": {"contrast_value": 1.3, "energy": 1.0, "rhythmic": 1.0},
    },
    "ending_options": {
        "source_roles": {"chorus", "outro", "bridge"},
        "target_roles": {"outro", "chorus", "tag"},
        "weights": {"transition_compatibility": 1.3, "structural_role": 1.1, "tempo": 0.9},
    },
}


@dataclass(frozen=True)
class CompatibilityResult:
    total_score: float
    category_scores: Dict[str, float]
    penalties: Dict[str, float]
    weighted_base: float
    penalty_total: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "category_scores": self.category_scores,
            "penalties": self.penalties,
            "weighted_base": self.weighted_base,
            "penalty_total": self.penalty_total,
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _num(section: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    val = section.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _text(section: Mapping[str, Any], key: str, default: str = "") -> str:
    val = section.get(key, default)
    return str(val).strip().lower() if val is not None else default


def _parse_key(section: Mapping[str, Any]) -> Tuple[Optional[int], str]:
    raw_key = section.get("key")
    raw_mode = section.get("mode")

    if isinstance(raw_key, str) and raw_key:
        token = raw_key.strip().upper().replace("MIN", "M").replace("MAJ", "")
        mode = "minor" if token.endswith("M") and len(token) > 1 else "major"
        tonic = token[:-1] if mode == "minor" else token
        tonic = tonic.replace("♯", "#").replace("♭", "B")
        return PITCH_CLASS.get(tonic), mode

    if raw_key is not None:
        try:
            pitch = int(raw_key) % 12
            mode = str(raw_mode or "major").strip().lower()
            return pitch, ("minor" if mode.startswith("min") else "major")
        except (TypeError, ValueError):
            return None, "major"

    return None, "major"


def _interval_distance(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


def _tempo_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    bpm_a, bpm_b = _num(a, "tempo_bpm"), _num(b, "tempo_bpm")
    if bpm_a <= 0 or bpm_b <= 0:
        return 50.0
    diff = abs(bpm_a - bpm_b)
    half_time_diff = min(abs((2 * bpm_a) - bpm_b), abs(bpm_a - (2 * bpm_b)))
    effective = min(diff, half_time_diff * 0.75)
    return _clamp(100.0 - (effective * 1.8))


def _key_harmony_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    pitch_a, mode_a = _parse_key(a)
    pitch_b, mode_b = _parse_key(b)
    if pitch_a is None or pitch_b is None:
        return 55.0

    interval = _interval_distance(pitch_a, pitch_b)
    if interval == 0 and mode_a == mode_b:
        base = 100.0
    elif interval == 0:
        base = 88.0
    elif interval in {5, 7}:  # fourth/fifth relation
        base = 90.0 if mode_a == mode_b else 84.0
    elif interval in {3, 4, 8, 9}:  # relative/mediant-ish
        base = 78.0 if mode_a == mode_b else 70.0
    elif interval in {1, 11}:
        base = 40.0
    else:
        base = 58.0
    return _clamp(base)


def _rhythmic_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    groove_gap = abs(_num(a, "groove_density") - _num(b, "groove_density"))
    sync_gap = abs(_num(a, "syncopation") - _num(b, "syncopation"))
    swing_gap = abs(_num(a, "swing") - _num(b, "swing"))
    ts_match = 1.0 if _text(a, "time_signature", "4/4") == _text(b, "time_signature", "4/4") else 0.75
    score = 100.0 - (groove_gap * 35.0 + sync_gap * 28.0 + swing_gap * 18.0)
    return _clamp(score * ts_match)


def _energy_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    energy_gap = abs(_num(a, "energy") - _num(b, "energy"))
    loud_gap = abs(_num(a, "loudness") - _num(b, "loudness"))
    return _clamp(100.0 - (energy_gap * 60.0 + loud_gap * 22.0))


def _structural_role_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    role_a = _text(a, "role", "unknown")
    role_b = _text(b, "role", "unknown")
    if (role_a, role_b) in ROLE_COMPATIBILITY:
        return ROLE_COMPATIBILITY[(role_a, role_b)]
    if role_a == role_b and role_a != "unknown":
        return 85.0
    return 62.0


def _transition_compatibility_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    end_cadence = _num(a, "cadence_strength")
    start_stability = _num(b, "entry_stability")
    tail = _num(a, "tail_sustain")
    attack = _num(b, "attack")
    phrase_a = max(1.0, _num(a, "phrase_bars", 4.0))
    phrase_b = max(1.0, _num(b, "phrase_bars", 4.0))
    phrase_ratio = min(phrase_a, phrase_b) / max(phrase_a, phrase_b)

    flow = 100.0 - abs((end_cadence * 0.7 + tail * 0.3) - (start_stability * 0.6 + attack * 0.4)) * 55.0
    phrase_fit = 60.0 + phrase_ratio * 40.0
    return _clamp((flow * 0.6) + (phrase_fit * 0.4))


def _contrast_value_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    e_diff = abs(_num(a, "energy") - _num(b, "energy"))
    tex_diff = abs(_num(a, "arrangement_density") - _num(b, "arrangement_density"))
    target_contrast = 0.30
    distance = abs(((e_diff + tex_diff) / 2.0) - target_contrast)
    return _clamp(100.0 - distance * 220.0)


def _payoff_potential_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    build = (_num(a, "tension") * 0.65) + (_num(a, "cadence_strength") * 0.35)
    release = (_num(b, "impact") * 0.7) + (_num(b, "energy") * 0.3)
    role_bonus = 12.0 if (_text(a, "role") in {"pre_chorus", "build"} and _text(b, "role") in {"chorus", "drop"}) else 0.0
    score = 55.0 + min(build, release) * 45.0 + role_bonus - abs(build - release) * 20.0
    return _clamp(score)


def _penalties(a: Mapping[str, Any], b: Mapping[str, Any], payoff_score: float) -> Dict[str, float]:
    clashing_hooks = _clamp((_num(a, "hook_density") + _num(b, "hook_density") - 1.2) * 80.0)
    dense_overlap = _clamp((_num(a, "arrangement_density") + _num(b, "arrangement_density") - 1.35) * 95.0)

    bass_a = _text(a, "bass_movement", "")
    bass_b = _text(b, "bass_movement", "")
    bass_conflict = 0.0
    if bass_a and bass_b and bass_a != bass_b:
        bass_conflict = _clamp((_num(a, "bass_activity") + _num(b, "bass_activity")) * 42.0)

    rhythmic_fighting = _clamp(
        (abs(_num(a, "offbeat_emphasis") - _num(b, "offbeat_emphasis")) * 65.0)
        + ((_num(a, "polyrhythm") * _num(b, "polyrhythm")) * 55.0)
    )

    weak_payoff = _clamp((68.0 - payoff_score) * 1.15)

    phrase_a = max(1.0, _num(a, "phrase_bars", 4.0))
    phrase_b = max(1.0, _num(b, "phrase_bars", 4.0))
    mod = max(phrase_a, phrase_b) % min(phrase_a, phrase_b)
    awkward_phrasing = 0.0 if mod == 0 else _clamp(28.0 + (mod / max(phrase_a, phrase_b)) * 62.0)

    return {
        "clashing_hooks": clashing_hooks,
        "dense_overlap": dense_overlap,
        "conflicting_bass_movement": bass_conflict,
        "rhythmic_fighting": rhythmic_fighting,
        "weak_payoff": weak_payoff,
        "awkward_phrasing": awkward_phrasing,
    }


def section_to_section_score(section_a: Mapping[str, Any], section_b: Mapping[str, Any]) -> Dict[str, Any]:
    """Score compatibility of two section descriptors."""
    categories = {
        "tempo": _tempo_score(section_a, section_b),
        "key_harmony": _key_harmony_score(section_a, section_b),
        "rhythmic": _rhythmic_score(section_a, section_b),
        "energy": _energy_score(section_a, section_b),
        "structural_role": _structural_role_score(section_a, section_b),
        "transition_compatibility": _transition_compatibility_score(section_a, section_b),
        "contrast_value": _contrast_value_score(section_a, section_b),
        "payoff_potential": _payoff_potential_score(section_a, section_b),
    }

    weighted_base = sum(categories[k] * CATEGORY_WEIGHTS[k] for k in CATEGORY_WEIGHTS)
    penalties = _penalties(section_a, section_b, categories["payoff_potential"])
    penalty_total = sum(penalties[k] * PENALTY_WEIGHTS[k] for k in PENALTY_WEIGHTS)
    total = _clamp(weighted_base - (penalty_total * 0.35))

    return CompatibilityResult(
        total_score=round(total, 3),
        category_scores={k: round(v, 3) for k, v in categories.items()},
        penalties={k: round(v, 3) for k, v in penalties.items()},
        weighted_base=round(weighted_base, 3),
        penalty_total=round(penalty_total, 3),
    ).to_dict()


def phrase_to_phrase_score(phrase_a: Mapping[str, Any], phrase_b: Mapping[str, Any]) -> Dict[str, Any]:
    """Phrase-level scoring emphasizing rhythmic and phrasing alignment."""
    result = section_to_section_score(phrase_a, phrase_b)
    # Phrase matching should reward tight phrasing and rhythm slightly more.
    adjusted = (
        result["total_score"]
        + 0.06 * result["category_scores"]["rhythmic"]
        + 0.06 * result["category_scores"]["transition_compatibility"]
        - 0.08 * result["penalties"]["awkward_phrasing"]
    )
    result["total_score"] = round(_clamp(adjusted), 3)
    return result


def stem_to_stem_score(stem_a: Mapping[str, Any], stem_b: Mapping[str, Any]) -> Dict[str, Any]:
    """Stem-level compatibility emphasizing harmonic and density conflicts."""
    result = section_to_section_score(stem_a, stem_b)
    adjusted = (
        result["total_score"]
        + 0.08 * result["category_scores"]["key_harmony"]
        - 0.10 * result["penalties"]["dense_overlap"]
        - 0.08 * result["penalties"]["clashing_hooks"]
    )
    result["total_score"] = round(_clamp(adjusted), 3)
    return result


def transition_to_transition_score(transition_a: Mapping[str, Any], transition_b: Mapping[str, Any]) -> Dict[str, Any]:
    """Transition-level score emphasizing flow and payoff behavior."""
    result = section_to_section_score(transition_a, transition_b)
    adjusted = (
        result["total_score"]
        + 0.10 * result["category_scores"]["transition_compatibility"]
        + 0.08 * result["category_scores"]["payoff_potential"]
        - 0.10 * result["penalties"]["weak_payoff"]
    )
    result["total_score"] = round(_clamp(adjusted), 3)
    return result


def _scored_pairs(
    source_sections: Sequence[Mapping[str, Any]],
    target_sections: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for i, source in enumerate(source_sections):
        for j, target in enumerate(target_sections):
            score = section_to_section_score(source, target)
            pairs.append(
                {
                    "source_index": i,
                    "target_index": j,
                    "source_role": _text(source, "role", "unknown"),
                    "target_role": _text(target, "role", "unknown"),
                    "score": score,
                }
            )
    return pairs


def _apply_profile(pair: Dict[str, Any], profile: Mapping[str, Any]) -> Optional[float]:
    source_role = pair["source_role"]
    target_role = pair["target_role"]
    if source_role not in profile["source_roles"] or target_role not in profile["target_roles"]:
        return None

    base = pair["score"]["total_score"]
    cat = pair["score"]["category_scores"]
    pen = pair["score"]["penalties"]

    boosts = profile.get("weights", {})
    weighted = base
    for key, factor in boosts.items():
        if key in cat:
            weighted += (factor - 1.0) * cat[key] * 0.12
        elif key in pen:
            weighted -= (factor - 1.0) * pen[key] * 0.12
    return round(_clamp(weighted), 3)


def ranked_candidate_pairings(
    source_sections: Sequence[Mapping[str, Any]],
    target_sections: Sequence[Mapping[str, Any]],
    top_n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return ranked candidate pairings for common arrangement decisions."""
    all_pairs = _scored_pairs(source_sections, target_sections)
    output: Dict[str, List[Dict[str, Any]]] = {}

    for profile_name, profile in PAIRING_PROFILES.items():
        ranked: List[Dict[str, Any]] = []
        for pair in all_pairs:
            profile_score = _apply_profile(pair, profile)
            if profile_score is None:
                continue
            ranked.append(
                {
                    "source_index": pair["source_index"],
                    "target_index": pair["target_index"],
                    "source_role": pair["source_role"],
                    "target_role": pair["target_role"],
                    "score": profile_score,
                    "base_score": pair["score"]["total_score"],
                }
            )
        ranked.sort(key=lambda x: (-x["score"], x["source_index"], x["target_index"]))
        output[profile_name] = ranked[: max(0, top_n)]

    return output
