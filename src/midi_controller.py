"""
MIDI Controller Mapping for AI DJ Project

Hardware mappings for DJ controllers (Pioneer DDJ, Numark, etc.)
Supports: play/pause, faders, EQ knobs, jog wheels, pads, transport controls

Usage:
    from midi_controller import MIDIMapping, create_controller
    
    controller = create_controller("pioneer_ddj_1000")
    controller.handle_note_on(channel=0, note=0, velocity=127)
"""

from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from dataclasses import dataclass


class ControlType(Enum):
    """Types of MIDI controls"""
    BUTTON = "button"           # On/off toggle
    KNOB = "knob"               # Rotary encoder (0-127)
    FADER = "fader"             # Linear slider (0-127)
    JOG_WHEEL = "jog_wheel"     # Platter/jog
    PAD = "pad"                 # Drum pad
    TOUCH_STRIP = "touch_strip" # Touch-sensitive strip


class Action(Enum):
    """Actions triggered by MIDI controls"""
    # Transport
    PLAY = "play"
    PAUSE = "pause"
    CUE = "cue"
    LOOP = "loop"
    SYNC = "sync"
    TAP_TEMPO = "tap_tempo"
    
    # Deck controls
    LOAD_TRACK = "load_track"
    EJECT = "eject"
    JOG_SCRATCH = "jog_scratch"
    JOG_SEEK = "jog_seek"
    PITCH_SLIDE = "pitch_slide"
    
    # Mixer
    VOLUME = "volume"
    EQ_HIGH = "eq_high"
    EQ_MID = "eq_mid"
    EQ_LOW = "eq_low"
    GAIN = "gain"
    FILTER = "filter"
    
    # Effects
    FX_1 = "fx_1"
    FX_2 = "fx_2"
    FX_3 = "fx_3"
    FX_WET_DRY = "fx_wet_dry"
    
    # Performance
    HOT_CUE_1 = "hot_cue_1"
    HOT_CUE_2 = "hot_cue_2"
    HOT_CUE_3 = "hot_cue_3"
    HOT_CUE_4 = "hot_cue_4"
    HOT_CUE_5 = "hot_cue_5"
    HOT_CUE_6 = "hot_cue_6"
    SLICER = "slicer"
    SAMPLER = "sampler"
    
    # AI DJ specific
    AI_ANALYZE = "ai_analyze"
    AI_SUGGEST_TRANSITION = "ai_suggest_transition"
    AI_GENERATE_DROP = "ai_generate_drop"
    AI_MATCH_BPM = "ai_match_bpm"
    AI_ENHANCE = "ai_enhance"


@dataclass
class MIDIControl:
    """Single MIDI control mapping"""
    channel: int           # MIDI channel (0-15)
    note: int              # Note number (0-127) or CC number for CC messages
    control_type: ControlType
    action: Action
    deck: int = 1          # Which deck (1-4)
    min_value: int = 0     # Value range
    max_value: int = 127
    label: str = ""        # Human-readable label
    
    # For advanced mapping
    secondary_channel: Optional[int] = None  # For CC (control change)
    is_toggle: bool = False  # Button toggle behavior
    is_led: bool = True      # Has LED feedback


@dataclass
class DeckMapping:
    """Complete mapping for a single deck"""
    deck_number: int
    controls: List[MIDIControl] = field(default_factory=list)


class MIDIMapping:
    """MIDI controller mapping manager"""
    
    def __init__(self, controller_name: str):
        self.controller_name = controller_name
        self.decks: Dict[int, DeckMapping] = {}
        self.control_map: Dict[tuple, MIDIControl] = {}  # (channel, note) -> control
        self.callbacks: Dict[Action, List[Callable]] = {}
        self._initialized = False
    
    def add_control(self, control: MIDIControl) -> None:
        """Add a control to the mapping"""
        key = (control.channel, control.note)
        self.control_map[key] = control
        
        # Add to deck
        if control.deck not in self.decks:
            self.decks[control.deck] = DeckMapping(deck_number=control.deck)
        self.decks[control.deck].controls.append(control)
    
    def register_callback(self, action: Action, callback: Callable) -> None:
        """Register callback for an action"""
        if action not in self.callbacks:
            self.callbacks[action] = []
        self.callbacks[action].append(callback)
    
    def handle_midi_message(self, status: int, note: int, velocity: int) -> Optional[Any]:
        """
        Handle incoming MIDI message
        Returns: action result or None
        """
        channel = status & 0x0F
        message_type = status & 0xF0
        
        key = (channel, note)
        control = self.control_map.get(key)
        
        if control is None:
            return None
        
        # Call registered callbacks
        if control.action in self.callbacks:
            for callback in self.callbacks[control.action]:
                result = callback(control, velocity)
                if result is not None:
                    return result
        
        return control.action
    
    def get_led_value(self, action: Action, deck: int = 1) -> Optional[tuple]:
        """Get LED value for feedback (channel, note, value)"""
        for deck_map in self.decks.values():
            for ctrl in deck_map.controls:
                if ctrl.action == action and ctrl.deck == deck and ctrl.is_led:
                    return (ctrl.channel, ctrl.note, 127 if ctrl.is_toggle else 64)
        return None


# ============================================================
# Controller-Specific Mappings
# ============================================================

def create_pioneer_ddj_1000() -> MIDIMapping:
    """Pioneer DDJ-1000 controller mapping"""
    m = MIDIMapping("Pioneer DDJ-1000")
    
    # Deck 1
    deck1_controls = [
        # Transport
        MIDIControl(channel=0, note=0, control_type=ControlType.BUTTON, action=Action.PLAY, deck=1, label="Play"),
        MIDIControl(channel=0, note=1, control_type=ControlType.BUTTON, action=Action.CUE, deck=1, label="Cue"),
        MIDIControl(channel=0, note=2, control_type=ControlType.JOG_WHEEL, action=Action.JOG_SEEK, deck=1, label="Jog Wheel"),
        
        # Mixer
        MIDIControl(channel=0, note=7, control_type=ControlType.FADER, action=Action.VOLUME, deck=1, label="Volume"),
        MIDIControl(channel=0, note=20, control_type=ControlType.KNOB, action=Action.EQ_HIGH, deck=1, label="EQ High"),
        MIDIControl(channel=0, note=21, control_type=ControlType.KNOB, action=Action.EQ_MID, deck=1, label="EQ Mid"),
        MIDIControl(channel=0, note=22, control_type=ControlType.KNOB, action=Action.EQ_LOW, deck=1, label="EQ Low"),
        MIDIControl(channel=0, note=23, control_type=ControlType.KNOB, action=Action.FILTER, deck=1, label="Filter"),
        
        # Performance Pads
        MIDIControl(channel=0, note=36, control_type=ControlType.PAD, action=Action.HOT_CUE_1, deck=1, label="Hot Cue 1"),
        MIDIControl(channel=0, note=37, control_type=ControlType.PAD, action=Action.HOT_CUE_2, deck=1, label="Hot Cue 2"),
        MIDIControl(channel=0, note=38, control_type=ControlType.PAD, action=Action.HOT_CUE_3, deck=1, label="Hot Cue 3"),
        MIDIControl(channel=0, note=39, control_type=ControlType.PAD, action=Action.HOT_CUE_4, deck=1, label="Hot Cue 4"),
        MIDIControl(channel=0, note=40, control_type=ControlType.PAD, action=Action.SLICER, deck=1, label="Slicer"),
        MIDIControl(channel=0, note=41, control_type=ControlType.PAD, action=Action.SAMPLER, deck=1, label="Sampler"),
        
        # Pitch fader
        MIDIControl(channel=0, note=77, control_type=ControlType.FADER, action=Action.PITCH_SLIDE, deck=1, label="Pitch"),
        
        # AI DJ Actions
        MIDIControl(channel=0, note=50, control_type=ControlType.BUTTON, action=Action.AI_ANALYZE, deck=1, label="AI Analyze"),
        MIDIControl(channel=0, note=51, control_type=ControlType.BUTTON, action=Action.AI_MATCH_BPM, deck=1, label="AI Match BPM"),
    ]
    
    # Deck 2 (same layout, different channel)
    deck2_controls = [
        MIDIControl(channel=1, note=0, control_type=ControlType.BUTTON, action=Action.PLAY, deck=2, label="Play"),
        MIDIControl(channel=1, note=1, control_type=ControlType.BUTTON, action=Action.CUE, deck=2, label="Cue"),
        MIDIControl(channel=1, note=2, control_type=ControlType.JOG_WHEEL, action=Action.JOG_SEEK, deck=2, label="Jog Wheel"),
        MIDIControl(channel=1, note=7, control_type=ControlType.FADER, action=Action.VOLUME, deck=2, label="Volume"),
        MIDIControl(channel=1, note=20, control_type=ControlType.KNOB, action=Action.EQ_HIGH, deck=2, label="EQ High"),
        MIDIControl(channel=1, note=21, control_type=ControlType.KNOB, action=Action.EQ_MID, deck=2, label="EQ Mid"),
        MIDIControl(channel=1, note=22, control_type=ControlType.KNOB, action=Action.EQ_LOW, deck=2, label="EQ Low"),
        MIDIControl(channel=1, note=23, control_type=ControlType.KNOB, action=Action.FILTER, deck=2, label="Filter"),
        MIDIControl(channel=1, note=36, control_type=ControlType.PAD, action=Action.HOT_CUE_1, deck=2, label="Hot Cue 1"),
        MIDIControl(channel=1, note=37, control_type=ControlType.PAD, action=Action.HOT_CUE_2, deck=2, label="Hot Cue 2"),
        MIDIControl(channel=1, note=38, control_type=ControlType.PAD, action=Action.HOT_CUE_3, deck=2, label="Hot Cue 3"),
        MIDIControl(channel=1, note=39, control_type=ControlType.PAD, action=Action.HOT_CUE_4, deck=2, label="Hot Cue 4"),
        MIDIControl(channel=1, note=77, control_type=ControlType.FADER, action=Action.PITCH_SLIDE, deck=2, label="Pitch"),
        MIDIControl(channel=1, note=50, control_type=ControlType.BUTTON, action=Action.AI_ANALYZE, deck=2, label="AI Analyze"),
        MIDIControl(channel=1, note=51, control_type=ControlType.BUTTON, action=Action.AI_MATCH_BPM, deck=2, label="AI Match BPM"),
    ]
    
    for ctrl in deck1_controls + deck2_controls:
        m.add_control(ctrl)
    
    return m


def create_numark_mixed_in_key() -> MIDIMapping:
    """Numark Mixed In Key controller mapping"""
    m = MIDIMapping("Numark Mixdeck")
    
    # Simplified 2-deck setup
    for deck in [1, 2]:
        ch = deck - 1
        m.add_control(MIDIControl(channel=ch, note=0x00, control_type=ControlType.BUTTON, action=Action.PLAY, deck=deck))
        m.add_control(MIDIControl(channel=ch, note=0x01, control_type=ControlType.BUTTON, action=Action.CUE, deck=deck))
        m.add_control(MIDIControl(channel=ch, note=0x04, control_type=ControlType.JOG_WHEEL, action=Action.JOG_SEEK, deck=deck))
        m.add_control(MIDIControl(channel=ch, note=0x0B, control_type=ControlType.FADER, action=Action.VOLUME, deck=deck))
        m.add_control(MIDIControl(channel=ch, note=0x10, control_type=ControlType.KNOB, action=Action.EQ_HIGH, deck=deck))
        m.add_control(MIDIControl(channel=ch, note=0x11, control_type=ControlType.KNOB, action=Action.EQ_MID, deck=deck))
        m.add_control(MIDIControl(channel=ch, note=0x12, control_type=ControlType.KNOB, action=Action.EQ_LOW, deck=deck))
        m.add_control(MIDIControl(channel=ch, note=0x4D, control_type=ControlType.FADER, action=Action.PITCH_SLIDE, deck=deck))
    
    return m


def create_generic_2deck() -> MIDIMapping:
    """Generic 2-deck MIDI controller (fallback)"""
    m = MIDIMapping("Generic 2-Deck Controller")
    
    # Standard MIDI CC mappings for basic DJ controller
    for deck in [1, 2]:
        ch = deck - 1
        
        # Channel faders (volume)
        m.add_control(MIDIControl(channel=ch, note=0x07, control_type=ControlType.FADER, 
                                    action=Action.VOLUME, deck=deck, label="Channel Fader"))
        
        # EQ knobs (use notes 20-22 for simplicity)
        m.add_control(MIDIControl(channel=ch, note=0x14, control_type=ControlType.KNOB,
                                    action=Action.EQ_HIGH, deck=deck, label="High EQ"))
        m.add_control(MIDIControl(channel=ch, note=0x15, control_type=ControlType.KNOB,
                                    action=Action.EQ_MID, deck=deck, label="Mid EQ"))
        m.add_control(MIDIControl(channel=ch, note=0x16, control_type=ControlType.KNOB,
                                    action=Action.EQ_LOW, deck=deck, label="Low EQ"))
        
        # Transport
        m.add_control(MIDIControl(channel=ch, note=0x00, control_type=ControlType.BUTTON,
                                    action=Action.PLAY, deck=deck, label="Play"))
        m.add_control(MIDIControl(channel=ch, note=0x01, control_type=ControlType.BUTTON,
                                    action=Action.CUE, deck=deck, label="Cue"))
        
        # Jog wheel
        m.add_control(MIDIControl(channel=ch, note=0x0C, control_type=ControlType.JOG_WHEEL,
                                    action=Action.JOG_SEEK, deck=deck, label="Jog"))
        
        # Pitch fader
        m.add_control(MIDIControl(channel=ch, note=0x0F, control_type=ControlType.FADER,
                                    action=Action.PITCH_SLIDE, deck=deck, label="Pitch"))
        
        # Performance pads (4 pads per deck)
        for i, pad_note in enumerate([0x24, 0x25, 0x26, 0x27]):
            action = [Action.HOT_CUE_1, Action.HOT_CUE_2, Action.HOT_CUE_3, Action.HOT_CUE_4][i]
            m.add_control(MIDIControl(channel=ch, note=pad_note, control_type=ControlType.PAD,
                                        action=action, deck=deck, label=f"Pad {i+1}"))
    
    return m


# ============================================================
# Factory Function
# ============================================================

CONTROLLER_PRESETS: Dict[str, Callable[[], MIDIMapping]] = {
    "pioneer_ddj_1000": create_pioneer_ddj_1000,
    "pioneer_ddj_800": create_pioneer_ddj_1000,  # Similar layout
    "pioneer_ddj_400": create_generic_2deck,
    "numark_mixdeck": create_numark_mixed_in_key,
    "numark_nd4000": create_numark_mixed_in_key,
    "generic": create_generic_2deck,
    "2deck": create_generic_2deck,
}


def create_controller(controller_type: str = "generic") -> MIDIMapping:
    """
    Create a MIDI controller mapping
    
    Args:
        controller_type: Name of controller preset
                         Options: pioneer_ddj_1000, pioneer_ddj_800, 
                                 numark_mixdeck, generic, 2deck
    
    Returns:
        MIDIMapping instance
    """
    controller_type = controller_type.lower().replace(" ", "_")
    
    if controller_type in CONTROLLER_PRESETS:
        return CONTROLLER_PRESETS[controller_type]()
    
    # Try partial match
    for name, factory in CONTROLLER_PRESETS.items():
        if name in controller_type or controller_type in name:
            return factory()
    
    # Default to generic
    print(f"Unknown controller '{controller_type}', using generic mapping")
    return create_generic_2deck()


# ============================================================
# MIDI Input Handler (for real-time input)
# ============================================================

class MIDIInputHandler:
    """Handles real-time MIDI input from controller"""
    
    def __init__(self, mapping: MIDIMapping):
        self.mapping = mapping
        self.last_values: Dict[tuple, int] = {}  # (channel, note) -> last value
    
    def process_message(self, status: int, note: int, velocity: int) -> Optional[Any]:
        """Process a raw MIDI message"""
        return self.mapping.handle_midi_message(status, note, velocity)
    
    def process_cc(self, channel: int, cc: int, value: int) -> Optional[Any]:
        """Process a MIDI CC message"""
        return self.mapping.handle_midi_message(0xB0 | channel, cc, value)
    
    def is_changed(self, channel: int, note: int, value: int) -> bool:
        """Check if value changed from last read"""
        key = (channel, note)
        if key not in self.last_values:
            self.last_values[key] = value
            return True
        
        changed = self.last_values[key] != value
        self.last_values[key] = value
        return changed


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Create controller
    controller = create_controller("pioneer_ddj_1000")
    print(f"Created mapping for: {controller.controller_name}")
    print(f"Total controls: {len(controller.control_map)}")
    print(f"Decks: {list(controller.decks.keys())}")
    
    # Register a callback
    def on_play(control, velocity):
        print(f"Play pressed on deck {control.deck}!")
        return "playing"
    
    controller.register_callback(Action.PLAY, on_play)
    
    # Simulate MIDI message
    result = controller.handle_midi_message(0x90, 0, 127)  # Note On, Channel 0, Note 0, Velocity 127
    print(f"Result: {result}")
