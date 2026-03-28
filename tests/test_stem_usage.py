from stem_usage import (
    apply_focal_priority,
    enforce_stem_conflict_rules,
    select_active_stems,
)


def test_enforce_stem_conflict_rules_limits_muddy_layers():
    active = {
        "drums": [
            {"name": "kick_main", "is_kick": True, "priority": 5},
            {"name": "kick_alt", "is_kick": True, "priority": 1},
            {"name": "hat", "priority": 2},
            {"name": "perc", "priority": 1},
        ],
        "bass": [
            {"name": "sub", "priority": 3},
            {"name": "mid_bass", "priority": 2},
        ],
        "vocals": [
            {"name": "lead_vocal", "priority": 3},
            {"name": "backing", "priority": 2},
        ],
        "music": [
            {"name": "chords", "priority": 3, "register": "mid"},
            {"name": "arp", "priority": 2, "register": "mid"},
            {"name": "lead", "priority": 1, "register": "high"},
        ],
        "fx": [{"name": "noise", "priority": 1}, {"name": "impact", "priority": 2}],
    }
    out = enforce_stem_conflict_rules(active)
    assert len(out["bass"]) == 1
    assert len([d for d in out["drums"] if d.get("is_kick")]) == 1
    assert len(out["vocals"]) == 1
    assert len(out["music"]) <= 2
    assert len(out["fx"]) == 1


def test_apply_focal_priority_sets_exactly_one_primary_focus():
    active = {
        "drums": [{"name": "kick", "priority": 2}],
        "bass": [{"name": "sub", "priority": 2}],
        "vocals": [{"name": "lead_vocal", "priority": 3}],
        "music": [{"name": "chords", "priority": 2}],
        "fx": [],
    }
    out = apply_focal_priority(active, "vocal")
    primaries = []
    for stems in [out["drums"], out["bass"], out["vocals"], out["music"], out["fx"]]:
        primaries.extend([s for s in stems if s.get("primary_focus")])
    assert len(primaries) == 1
    assert primaries[0]["name"] == "lead_vocal"


def test_select_active_stems_obeys_max_layers_and_focus():
    section_plan = {"focus": "vocal", "max_layers": 4}
    candidates = {
        "drums": [
            {"name": "kick", "is_kick": True, "priority": 5},
            {"name": "hat", "priority": 4},
            {"name": "perc", "priority": 3},
        ],
        "bass": [{"name": "sub", "priority": 5}],
        "vocals": [{"name": "lead_vocal", "priority": 5}],
        "music": [
            {"name": "chords", "priority": 4, "register": "mid"},
            {"name": "lead", "priority": 3, "register": "high"},
        ],
        "fx": [{"name": "impact", "priority": 2}],
    }
    out = select_active_stems(section_plan, candidates)

    total = sum(len(out[k]) for k in ["drums", "bass", "vocals", "music", "fx"])
    assert total <= 4

    primary = [
        s
        for k in ["drums", "bass", "vocals", "music", "fx"]
        for s in out[k]
        if s.get("primary_focus")
    ]
    assert len(primary) == 1
    assert "vocal" in primary[0]["name"]
