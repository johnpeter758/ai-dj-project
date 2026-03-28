from dsp import (
    TRANSITION_TYPES,
    build_transition_instruction,
    choose_transition_type,
    post_transition_stabilization,
    pre_transition_cleanup,
)


def test_choose_transition_type_supported_and_deterministic():
    context = {
        "from_has_vocal": False,
        "to_has_vocal": True,
        "energy_delta": 0.4,
        "bpm_delta": 0,
        "bars": 8,
    }
    t1 = choose_transition_type(context)
    t2 = choose_transition_type(context)
    assert t1 == t2
    assert t1 in TRANSITION_TYPES
    assert t1 == "acapella_spotlight"


def test_build_transition_instruction_contains_valid_type_and_actions():
    from_section = {"name": "verse", "energy": 0.45, "bpm": 124, "has_vocal": False}
    to_section = {
        "name": "drop",
        "energy": 0.85,
        "bpm": 124,
        "has_vocal": False,
        "dramatic": True,
        "transition_bars": 4,
    }
    instruction = build_transition_instruction(from_section, to_section)
    assert instruction["type"] in TRANSITION_TYPES
    assert instruction["bars"] == 4
    assert isinstance(instruction["actions"], list)
    assert instruction["type"] == "riser_drop"


def test_pre_transition_cleanup_applies_anti_mud_and_bass_dropout():
    stems = [
        {"name": "bass_a", "role": "bass", "priority": 2},
        {"name": "bass_b", "role": "bass", "priority": 1},
        {"name": "pad_1", "role": "music"},
        {"name": "pad_2", "role": "music"},
        {"name": "pad_3", "role": "music"},
        {"name": "lead", "role": "music"},
    ]
    cleaned = pre_transition_cleanup(
        stems,
        {
            "energy_delta": -0.5,
            "dramatic": True,
            "bars": 8,
            "from_has_vocal": False,
            "to_has_vocal": False,
        },
    )
    bass_muted = [s for s in cleaned if s["role"] == "bass" and s.get("state") == "muted"]
    assert len(bass_muted) == 2

    ducked_music = [s for s in cleaned if s["role"] == "music" and s.get("state") == "ducked"]
    assert len(ducked_music) >= 1


def test_post_transition_stabilization_restores_backbone_and_focus():
    stems = [
        {"name": "kick", "role": "drums", "state": "muted", "gain_db": -96},
        {"name": "sub", "role": "bass", "state": "muted", "gain_db": -96},
        {"name": "vox", "role": "vocal", "state": "active", "gain_db": -1},
        {"name": "chords", "role": "music", "state": "active", "gain_db": 0},
    ]
    out = post_transition_stabilization(stems, {"focus": "vocal"})

    kick = next(s for s in out if s["name"] == "kick")
    sub = next(s for s in out if s["name"] == "sub")
    vox = next(s for s in out if s["name"] == "vox")
    chords = next(s for s in out if s["name"] == "chords")

    assert kick["state"] == "active"
    assert sub["state"] == "active"
    assert vox["primary_focus"] is True
    assert chords["state"] == "ducked"
