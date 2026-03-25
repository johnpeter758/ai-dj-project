from __future__ import annotations

from .models import ChildSectionRecipe
from .policies import recipe_policy_for_label


def build_child_section_recipe(
    *,
    section_label: str,
    backbone_parent: str,
    chosen_parent: str,
    chosen_label: str | None,
    support_recipe: dict | None,
    primary_mi_summary: dict | None,
    support_mi_summary: dict | None,
    arrangement_mode: str = "adaptive",
) -> ChildSectionRecipe:
    policy = recipe_policy_for_label(section_label)
    support_parent = support_recipe.get("parent_id") if support_recipe else None
    support_mode = (support_recipe or {}).get("mode")
    primary_melodic = float((primary_mi_summary or {}).get("melodic_identity_strength", 0.0) or 0.0)
    support_melodic = float((support_mi_summary or {}).get("melodic_identity_strength", 0.0) or 0.0)

    motif_anchor_parent = chosen_parent
    motif_anchor_label = chosen_label
    if (
        support_recipe
        and support_parent
        and support_mode == "foreground_counterlayer"
        and support_melodic >= max(0.45, primary_melodic + 0.05)
    ):
        motif_anchor_parent = support_parent
        motif_anchor_label = support_recipe.get("window_label") or chosen_label

    motif_recurrence_strength = round(max(primary_melodic, support_melodic), 3)
    donor_support_required = bool(support_recipe)
    integration_strength = 0.0
    if support_recipe:
        support_gain_db = float((support_recipe or {}).get("gain_db", -12.0) or -12.0)
        # Map typical support-layer gain into an interpretable 0..1 integration score.
        # -18 dB or quieter reads as a light/background layer, while -6 dB or louder
        # is effectively strongly integrated into the section texture.
        gain_floor_db = -18.0
        gain_ceiling_db = -6.0
        normalized_gain = (support_gain_db - gain_floor_db) / (gain_ceiling_db - gain_floor_db)
        integration_strength = max(0.0, min(1.0, normalized_gain))
        if support_mode == "foreground_counterlayer":
            integration_strength = min(1.0, integration_strength + 0.1)
        integration_strength = round(integration_strength, 3)
    timbral_anchor_policy = str(policy["timbral_anchor"])
    if timbral_anchor_policy == "feature_anchor":
        timbral_anchor = f"{chosen_parent}_feature_anchor"
    elif timbral_anchor_policy == "hybrid_riser_anchor":
        timbral_anchor = f"{backbone_parent}_to_{support_parent or chosen_parent}_hybrid_anchor"
    elif timbral_anchor_policy == "contrast_palette_anchor":
        timbral_anchor = f"{support_parent or chosen_parent}_contrast_palette_anchor"
    else:
        timbral_anchor = f"{backbone_parent}_palette_anchor"
    if arrangement_mode == "baseline":
        timbral_anchor = f"baseline_{timbral_anchor}"
    return ChildSectionRecipe(
        backbone_owner=backbone_parent if chosen_parent != backbone_parent else chosen_parent,
        donor_support_required=donor_support_required,
        motif_anchor_parent=motif_anchor_parent,
        motif_anchor_label=motif_anchor_label,
        motif_recurrence_strength=motif_recurrence_strength,
        tension_target=policy["tension_target"],
        rhythmic_constraint=policy["rhythmic_constraint"],
        harmonic_constraint=policy["harmonic_constraint"],
        timbral_anchor=timbral_anchor,
        support_parent=support_parent,
        support_mode=support_mode,
        support_gain_db=(support_recipe or {}).get("gain_db"),
        integration_strength=integration_strength,
        policy_id="section_recipe_v1_baseline" if arrangement_mode == "baseline" else "section_recipe_v1",
    )
