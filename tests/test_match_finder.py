import unittest

from match_finder import (
    phrase_to_phrase_score,
    ranked_candidate_pairings,
    section_to_section_score,
    stem_to_stem_score,
    transition_to_transition_score,
)


SECTION_A = {
    "tempo_bpm": 124,
    "key": "A",
    "mode": "major",
    "time_signature": "4/4",
    "groove_density": 0.52,
    "syncopation": 0.34,
    "swing": 0.08,
    "energy": 0.66,
    "loudness": 0.58,
    "role": "verse",
    "cadence_strength": 0.62,
    "entry_stability": 0.45,
    "tail_sustain": 0.42,
    "attack": 0.40,
    "phrase_bars": 8,
    "arrangement_density": 0.55,
    "tension": 0.52,
    "impact": 0.57,
    "hook_density": 0.41,
    "bass_movement": "ascending",
    "bass_activity": 0.48,
    "offbeat_emphasis": 0.45,
    "polyrhythm": 0.16,
}

SECTION_B = {
    "tempo_bpm": 126,
    "key": "E",
    "mode": "major",
    "time_signature": "4/4",
    "groove_density": 0.57,
    "syncopation": 0.30,
    "swing": 0.10,
    "energy": 0.72,
    "loudness": 0.62,
    "role": "pre_chorus",
    "cadence_strength": 0.70,
    "entry_stability": 0.52,
    "tail_sustain": 0.44,
    "attack": 0.50,
    "phrase_bars": 8,
    "arrangement_density": 0.60,
    "tension": 0.66,
    "impact": 0.63,
    "hook_density": 0.47,
    "bass_movement": "ascending",
    "bass_activity": 0.52,
    "offbeat_emphasis": 0.43,
    "polyrhythm": 0.12,
}

CLASHY = {
    "tempo_bpm": 92,
    "key": "A#",
    "mode": "minor",
    "time_signature": "7/8",
    "groove_density": 0.92,
    "syncopation": 0.92,
    "swing": 0.45,
    "energy": 0.95,
    "loudness": 0.95,
    "role": "bridge",
    "cadence_strength": 0.10,
    "entry_stability": 0.15,
    "tail_sustain": 0.93,
    "attack": 0.88,
    "phrase_bars": 5,
    "arrangement_density": 0.94,
    "tension": 0.11,
    "impact": 0.10,
    "hook_density": 0.97,
    "bass_movement": "descending",
    "bass_activity": 0.90,
    "offbeat_emphasis": 0.93,
    "polyrhythm": 0.87,
}


class TestMatchFinder(unittest.TestCase):
    def test_section_to_section_score_is_explainable_and_deterministic(self):
        out = section_to_section_score(SECTION_A, SECTION_B)

        self.assertEqual(
            set(out["category_scores"]),
            {
                "tempo",
                "key_harmony",
                "rhythmic",
                "energy",
                "structural_role",
                "transition_compatibility",
                "contrast_value",
                "payoff_potential",
            },
        )
        self.assertEqual(
            set(out["penalties"]),
            {
                "clashing_hooks",
                "dense_overlap",
                "conflicting_bass_movement",
                "rhythmic_fighting",
                "weak_payoff",
                "awkward_phrasing",
            },
        )

        self.assertEqual(out["category_scores"]["key_harmony"], 90.0)
        self.assertEqual(out["penalties"]["conflicting_bass_movement"], 0.0)
        self.assertAlmostEqual(out["total_score"], 88.693, places=3)

    def test_penalties_trigger_for_conflicts(self):
        out = section_to_section_score(SECTION_A, CLASHY)
        penalties = out["penalties"]

        self.assertGreater(penalties["clashing_hooks"], 0)
        self.assertGreater(penalties["dense_overlap"], 0)
        self.assertGreater(penalties["conflicting_bass_movement"], 0)
        self.assertGreater(penalties["rhythmic_fighting"], 0)
        self.assertGreater(penalties["awkward_phrasing"], 0)
        self.assertLess(out["total_score"], 35)

    def test_api_variants_apply_different_biases(self):
        base = section_to_section_score(SECTION_A, SECTION_B)
        phrase = phrase_to_phrase_score(SECTION_A, SECTION_B)
        stem = stem_to_stem_score(SECTION_A, SECTION_B)
        transition = transition_to_transition_score(SECTION_A, SECTION_B)

        self.assertNotEqual(phrase["total_score"], base["total_score"])
        self.assertNotEqual(stem["total_score"], base["total_score"])
        self.assertNotEqual(transition["total_score"], base["total_score"])
        self.assertGreaterEqual(transition["total_score"], phrase["total_score"])

    def test_ranked_candidate_pairings_returns_expected_buckets(self):
        source = [
            {**SECTION_A, "role": "intro", "energy": 0.42, "phrase_bars": 8},
            {**SECTION_A, "role": "verse", "energy": 0.63, "phrase_bars": 8},
            {**SECTION_A, "role": "pre_chorus", "energy": 0.76, "tension": 0.81, "phrase_bars": 8},
            {**SECTION_A, "role": "chorus", "energy": 0.88, "impact": 0.90, "phrase_bars": 8},
            {**SECTION_A, "role": "outro", "energy": 0.40, "phrase_bars": 8},
        ]
        target = [
            {**SECTION_B, "role": "intro", "energy": 0.46, "phrase_bars": 8},
            {**SECTION_B, "role": "verse", "energy": 0.66, "phrase_bars": 8},
            {**SECTION_B, "role": "pre_chorus", "energy": 0.73, "phrase_bars": 8},
            {**SECTION_B, "role": "chorus", "energy": 0.91, "impact": 0.93, "phrase_bars": 8},
            {**SECTION_B, "role": "outro", "energy": 0.38, "phrase_bars": 8},
        ]

        ranked = ranked_candidate_pairings(source, target, top_n=3)

        self.assertEqual(
            set(ranked),
            {
                "intro_options",
                "first_vocal_entry_options",
                "chorus_payoff_options",
                "swap_moments",
                "ending_options",
            },
        )

        for bucket in ranked.values():
            self.assertLessEqual(len(bucket), 3)
            self.assertTrue(
                all(bucket[i]["score"] >= bucket[i + 1]["score"] for i in range(len(bucket) - 1))
            )

        self.assertEqual(ranked["intro_options"][0]["source_role"], "intro")
        self.assertEqual(ranked["chorus_payoff_options"][0]["target_role"], "chorus")
        self.assertIn(ranked["ending_options"][0]["target_role"], {"outro", "chorus"})


if __name__ == "__main__":
    unittest.main()
