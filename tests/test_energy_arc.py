from energy_arc import apply_energy_arc_rules, build_energy_arc_template


def test_build_energy_arc_template_is_deterministic_and_shaped() -> None:
    first = build_energy_arc_template(6, profile="standard")
    second = build_energy_arc_template(6, profile="standard")

    assert first == second
    assert len(first) == 6
    assert first[0]["role"] == "intro"
    assert first[-1]["role"] == "finale"
    assert first[-1]["target_energy"] >= first[0]["target_energy"]


def test_apply_energy_arc_rules_enforces_anti_flatness_and_anti_chaos() -> None:
    sections = [
        {"section_index": 1, "target_energy": 0.50},
        {"section_index": 2, "target_energy": 0.50},
        {"section_index": 3, "target_energy": 0.51},
        {"section_index": 4, "target_energy": 0.95},
    ]

    fixed = apply_energy_arc_rules(sections)
    energies = [s["target_energy"] for s in fixed]

    # anti-flatness: small differences are nudged apart
    assert abs(energies[1] - energies[0]) >= 0.06
    # anti-chaos: jumps are capped
    assert abs(energies[3] - energies[2]) <= 0.22
    # deterministic and annotated
    assert all("energy_delta" in s for s in fixed)
    assert fixed == apply_energy_arc_rules(sections)
