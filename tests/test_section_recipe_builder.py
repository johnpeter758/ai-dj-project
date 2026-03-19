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
    assert recipe.integration_strength == 1.0


def test_support_integration_strength_is_clamped_to_unit_interval():
    recipe = build_child_section_recipe(
        section_label='build',
        backbone_parent='A',
        chosen_parent='B',
        chosen_label='phrase_6_8',
        support_recipe={
            'parent_id': 'A',
            'window_label': 'phrase_2_4',
            'gain_db': -9.0,
            'mode': 'filtered_counterlayer',
        },
        primary_mi_summary={'melodic_identity_strength': 0.57},
        support_mi_summary={'melodic_identity_strength': 0.43},
    )

    assert 0.0 <= recipe.integration_strength <= 1.0
    assert recipe.integration_strength == 1.0
