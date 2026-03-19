from __future__ import annotations

SECTION_RECIPE_POLICIES: dict[str, dict[str, str]] = {
    "intro": {
        "tension_target": "establish_and_tease",
        "rhythmic_constraint": "light_pulse_without_kick_commitment",
        "harmonic_constraint": "establish_tonal_center",
        "timbral_anchor": "backbone_palette_anchor",
    },
    "verse": {
        "tension_target": "steady_forward_motion",
        "rhythmic_constraint": "backbone_pocket_locked",
        "harmonic_constraint": "stable_center_support_topline",
        "timbral_anchor": "backbone_palette_anchor",
    },
    "build": {
        "tension_target": "controlled_tension_rise",
        "rhythmic_constraint": "density_rise_without_kick_conflict",
        "harmonic_constraint": "pre_payoff_tension_without_false_resolution",
        "timbral_anchor": "hybrid_riser_anchor",
    },
    "payoff": {
        "tension_target": "release_and_conviction",
        "rhythmic_constraint": "kick_and_snare_authority",
        "harmonic_constraint": "clear_resolution_and_root_conviction",
        "timbral_anchor": "feature_anchor",
    },
    "bridge": {
        "tension_target": "contrast_then_reset",
        "rhythmic_constraint": "groove_reset_then_relock",
        "harmonic_constraint": "contrast_without_key_break",
        "timbral_anchor": "contrast_palette_anchor",
    },
    "outro": {
        "tension_target": "resolve_and_decay",
        "rhythmic_constraint": "release_density_keep_grid_clear",
        "harmonic_constraint": "resolution_with_decay",
        "timbral_anchor": "backbone_palette_anchor",
    },
}

_DEFAULT_POLICY = {
    "tension_target": "balanced_motion",
    "rhythmic_constraint": "pocket_preservation",
    "harmonic_constraint": "local_tonal_continuity",
    "timbral_anchor": "backbone_palette_anchor",
}


def recipe_policy_for_label(label: str) -> dict[str, str]:
    return dict(SECTION_RECIPE_POLICIES.get(str(label or ""), _DEFAULT_POLICY))
