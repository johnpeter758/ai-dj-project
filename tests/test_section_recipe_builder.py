from src.core.intelligence import build_child_section_recipe


def test_filtered_support_keeps_motif_anchor_on_primary_owner():
    recipe = build_child_section_recipe(
        section_label='verse',
        backbone_parent='A',
        chosen_parent='A',
        chosen_label='phrase_4_6',
        support_recipe={
            'parent_id': 'B',
            'window_label': 'phrase_8_10',
            'gain_db': -10.5,
            'mode': 'filtered_counterlayer',
        },
        primary_mi_summary={'melodic_identity_strength': 0.61},
        support_mi_summary={'melodic_identity_strength': 0.82},
    )

    assert recipe.motif_anchor_parent == 'A'
    assert recipe.motif_anchor_label == 'phrase_4_6'
    assert recipe.donor_support_required is True


def test_foreground_support_can_claim_motif_anchor_when_it_is_the_stronger_feature_voice():
    recipe = build_child_section_recipe(
        section_label='payoff',
        backbone_parent='A',
        chosen_parent='A',
        chosen_label='phrase_12_14',
        support_recipe={
            'parent_id': 'B',
            'window_label': 'phrase_20_22',
            'gain_db': -8.0,
            'mode': 'foreground_counterlayer',
        },
        primary_mi_summary={'melodic_identity_strength': 0.48},
        support_mi_summary={'melodic_identity_strength': 0.71},
    )

    assert recipe.motif_anchor_parent == 'B'
    assert recipe.motif_anchor_label == 'phrase_20_22'
    assert recipe.integration_strength == 0.933


def test_baseline_arrangement_mode_marks_recipe_policy_and_anchor():
    recipe = build_child_section_recipe(
        section_label='verse',
        backbone_parent='A',
        chosen_parent='A',
        chosen_label='phrase_2_4',
        support_recipe=None,
        primary_mi_summary={'melodic_identity_strength': 0.55},
        support_mi_summary=None,
        arrangement_mode='baseline',
    )

    assert recipe.policy_id == 'section_recipe_v1_baseline'
    assert recipe.timbral_anchor.startswith('baseline_')



def test_support_integration_strength_tracks_support_gain_and_stays_clamped():
    quiet_recipe = build_child_section_recipe(
        section_label='build',
        backbone_parent='A',
        chosen_parent='B',
        chosen_label='phrase_6_8',
        support_recipe={
            'parent_id': 'A',
            'window_label': 'phrase_2_4',
            'gain_db': -18.0,
            'mode': 'filtered_counterlayer',
        },
        primary_mi_summary={'melodic_identity_strength': 0.57},
        support_mi_summary={'melodic_identity_strength': 0.43},
    )
    mid_recipe = build_child_section_recipe(
        section_label='build',
        backbone_parent='A',
        chosen_parent='B',
        chosen_label='phrase_6_8',
        support_recipe={
            'parent_id': 'A',
            'window_label': 'phrase_2_4',
            'gain_db': -12.0,
            'mode': 'filtered_counterlayer',
        },
        primary_mi_summary={'melodic_identity_strength': 0.57},
        support_mi_summary={'melodic_identity_strength': 0.43},
    )
    hot_recipe = build_child_section_recipe(
        section_label='build',
        backbone_parent='A',
        chosen_parent='B',
        chosen_label='phrase_6_8',
        support_recipe={
            'parent_id': 'A',
            'window_label': 'phrase_2_4',
            'gain_db': -3.0,
            'mode': 'filtered_counterlayer',
        },
        primary_mi_summary={'melodic_identity_strength': 0.57},
        support_mi_summary={'melodic_identity_strength': 0.43},
    )

    assert quiet_recipe.integration_strength == 0.0
    assert mid_recipe.integration_strength == 0.5
    assert hot_recipe.integration_strength == 1.0
    assert quiet_recipe.integration_strength < mid_recipe.integration_strength < hot_recipe.integration_strength
