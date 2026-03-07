"""
MIDI Sequencer for AI DJ Project

Pattern-based MIDI sequencer with timing, quantization, and event management.
Supports: step sequencing, note editing, pattern chaining, MIDI export.

Usage:
    from midi_sequencer import MIDISequencer, Pattern, Track, Note
    
    sequencer = MIDISequencer(bpm=128)
    pattern = sequencer.create_pattern(length=16, time_signature=(4, 4))
    track = pattern.add_track(channel=1, instrument="synth")
    track.add_note(pitch=60, start=0, duration=4, velocity=100)
    midi_data = sequencer.export_midi()
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable, Any, Tuple
from pathlib import Path
import struct
import time


class NoteLength(Enum):
    """Standard note lengths in ticks"""
    WHOLE = 3840
    HALF = 1920
    QUARTER = 960
    EIGHTH = 480
    SIXTEENTH = 240
    THIRTY_SECOND = 120
    SIXTY_FOURTH = 60


@dataclass
class Note:
    """Single MIDI note event"""
    pitch: int              # MIDI note number (0-127)
    start: int              # Start time in ticks
    duration: int           # Duration in ticks
    velocity: int           # Note velocity (0-127)
    track: int = 0          # Track index
    
    def __post_init__(self):
        self.pitch = max(0, min(127, self.pitch))
        self.start = max(0, self.start)
        self.duration = max(1, self.duration)
        self.velocity = max(0, min(127, self.velocity))
    
    @property
    def end(self) -> int:
        """End time in ticks"""
        return self.start + self.duration
    
    @property
    def note_name(self) -> str:
        """Convert MIDI pitch to note name"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return f"{notes[self.pitch % 12]}{self.pitch // 12 - 1}"
    
    def transpose(self, semitones: int) -> 'Note':
        """Transpose note by semitones"""
        return Note(
            pitch=max(0, min(127, self.pitch + semitones)),
            start=self.start,
            duration=self.duration,
            velocity=self.velocity,
            track=self.track
        )


@dataclass
class ControlChange:
    """MIDI CC event"""
    cc_number: int          # CC number (0-127)
    value: int              # CC value (0-127)
    time: int               # Time in ticks
    track: int = 0
    
    def __post_init__(self):
        self.cc_number = max(0, min(127, self.cc_number))
        self.value = max(0, min(127, self.value))


@dataclass
class Track:
    """MIDI track containing notes and CC events"""
    channel: int            # MIDI channel (0-15)
    instrument: str = ""    # Instrument name
    notes: List[Note] = field(default_factory=list)
    cc_events: List[ControlChange] = field(default_factory=list)
    volume: int = 100        # Track volume (0-127)
    pan: int = 64            # Track pan (0=Left, 64=Center, 127=Right)
    
    def __post_init__(self):
        self.channel = max(0, min(15, self.channel))
    
    def add_note(self, pitch: int, start: int, duration: int = 480, 
                 velocity: int = 100) -> Note:
        """Add a note to the track"""
        note = Note(
            pitch=pitch,
            start=start,
            duration=duration,
            velocity=velocity,
            track=self.channel
        )
        self.notes.append(note)
        return note
    
    def add_cc(self, cc_number: int, value: int, time: int) -> ControlChange:
        """Add a CC event to the track"""
        cc = ControlChange(
            cc_number=cc_number,
            value=value,
            time=time,
            track=self.channel
        )
        self.cc_events.append(cc)
        return cc
    
    def quantize(self, grid_size: int = 240) -> None:
        """Quantize all notes to grid"""
        for note in self.notes:
            note.start = round(note.start / grid_size) * grid_size
    
    def clear(self) -> None:
        """Clear all events from track"""
        self.notes.clear()
        self.cc_events.clear()
    
    @property
    def note_count(self) -> int:
        return len(self.notes)


@dataclass
class Pattern:
    """MIDI pattern with multiple tracks"""
    name: str = "Untitled"
    length: int = 3840          # Pattern length in ticks (1 bar @ 960 tps)
    time_signature: Tuple[int, int] = (4, 4)
    tracks: List[Track] = field(default_factory=list)
    color: int = 0              # Pattern color for UI
    
    def __post_init__(self):
        self.tracks = list(self.tracks) if self.tracks else [Track(channel=0)]
    
    @property
    def ticks_per_beat(self) -> int:
        """Standard MIDI ticks per beat (PPQ)"""
        return 960
    
    @property
    def bars(self) -> float:
        """Pattern length in bars"""
        return self.length / (self.ticks_per_beat * self.time_signature[0])
    
    def add_track(self, channel: int, instrument: str = "") -> Track:
        """Add a new track"""
        track = Track(channel=channel, instrument=instrument)
        self.tracks.append(track)
        return track
    
    def get_track(self, channel: int) -> Optional[Track]:
        """Get track by channel"""
        for track in self.tracks:
            if track.channel == channel:
                return track
        return None
    
    def quantize_all(self, grid_size: int = 240) -> None:
        """Quantize all tracks"""
        for track in self.tracks:
            track.quantize(grid_size)
    
    def clear(self) -> None:
        """Clear all tracks"""
        for track in self.tracks:
            track.clear()
    
    def transpose_all(self, semitones: int) -> None:
        """Transpose all notes"""
        for track in self.tracks:
            track.notes = [n.transpose(semitones) for n in track.notes]
    
    def scale_velocity(self, factor: float) -> None:
        """Scale all velocities by factor"""
        for track in self.tracks:
            for note in track.notes:
                note.velocity = int(max(0, min(127, note.velocity * factor)))


class MIDISequencer:
    """
    Main MIDI sequencer class for pattern-based sequencing.
    """
    
    DEFAULT_PPQ = 960  # Pulses per quarter note
    
    def __init__(self, bpm: float = 120.0, ppq: int = DEFAULT_PPQ):
        self.bpm = bpm
        self.ppq = ppq
        self.patterns: List[Pattern] = []
        self.current_pattern: Optional[Pattern] = None
        self.current_step = 0
        self.is_playing = False
        self.loop_enabled = True
        self.swing_amount = 0  # Swing percentage (0-100)
        
        # Callbacks for real-time playback
        self.on_note_on: Optional[Callable[[int, int, int], None]] = None
        self.on_note_off: Optional[Callable[[int, int], None]] = None
        self.on_cc: Optional[Callable[[int, int, int], None]] = None
        self.on_step: Optional[Callable[[int], None]] = None
        
        # Create default pattern
        self.create_pattern()
    
    def create_pattern(self, length: int = 3840, 
                      time_signature: Tuple[int, int] = (4, 4),
                      name: str = "Pattern") -> Pattern:
        """Create a new pattern"""
        pattern = Pattern(
            name=name,
            length=length,
            time_signature=time_signature
        )
        pattern.add_track(channel=0, instrument="Lead")
        self.patterns.append(pattern)
        self.current_pattern = pattern
        return pattern
    
    def select_pattern(self, index: int) -> Optional[Pattern]:
        """Select a pattern by index"""
        if 0 <= index < len(self.patterns):
            self.current_pattern = self.patterns[index]
            self.current_step = 0
            return self.current_pattern
        return None
    
    def delete_pattern(self, index: int) -> bool:
        """Delete a pattern by index"""
        if 0 <= index < len(self.patterns) and len(self.patterns) > 1:
            del self.patterns[index]
            if self.current_pattern == self.patterns[index] if index < len(self.patterns) else True:
                self.current_pattern = self.patterns[0]
            return True
        return False
    
    @property
    def tick_duration_ms(self) -> float:
        """Calculate tick duration in milliseconds"""
        return (60000.0 / self.bpm) / self.ppq
    
    @property
    def step_duration_ticks(self) -> int:
        """Get step duration for 16-step sequence"""
        return self.ppq // 4  # 16th notes
    
    def tick_to_beat(self, ticks: int) -> float:
        """Convert ticks to beats"""
        return ticks / self.ppq
    
    def beat_to_tick(self, beats: float) -> int:
        """Convert beats to ticks"""
        return int(beats * self.ppq)
    
    def beat_to_step(self, beat: float, steps_per_beat: int = 4) -> int:
        """Convert beat to step number"""
        return int(beat * steps_per_beat)
    
    def quantize_time(self, time: int, grid: int = 240) -> int:
        """Quantize time to grid"""
        return round(time / grid) * grid
    
    # ----- Pattern Building Helpers -----
    
    def add_drum_pattern(self, channel: int = 9, 
                        kick: str = "x---x---x---x---",
                        snare: str = "----x-------x---",
                        hihat: str = "x-x-x-x-x-x-x-x") -> Pattern:
        """Add a drum pattern from string notation"""
        pattern = self.current_pattern
        if not pattern:
            pattern = self.create_pattern()
        
        track = pattern.get_track(channel) or pattern.add_track(channel, "Drums")
        
        for step, char in enumerate(kick):
            if char == 'x':
                track.add_note(36, step * 240, 120, 127)  # Kick on channel 10
        for step, char in enumerate(snare):
            if char == 'x':
                track.add_note(38, step * 240, 120, 127)  # Snare
        for step, char in enumerate(hihat):
            if char == 'x':
                track.add_note(42, step * 240, 60, 80)   # Closed hi-hat
        
        return pattern
    
    def add_chord_progression(self, channel: int = 0,
                              progression: List[Tuple[int, ...]] = None,
                              start_beat: float = 0,
                              duration_beats: float = 4) -> None:
        """Add a chord progression"""
        if progression is None:
            progression = [
                (60, 64, 67),   # C major
                (65, 69, 72),   # F major
                (67, 71, 74),   # G major
                (60, 64, 67),   # C major
            ]
        
        pattern = self.current_pattern
        if not pattern:
            return
        
        track = pattern.get_track(channel) or pattern.add_track(channel, "Chords")
        
        chord_length = int(duration_beats * self.ppq)
        current_tick = int(start_beat * self.ppq)
        
        for chord in progression:
            for pitch in chord:
                track.add_note(pitch, current_tick, chord_length - 60, 90)
            current_tick += chord_length
    
    def add_arpeggio(self, channel: int = 0,
                     root: int = 60,
                     pattern: str = "up",
                     octave_range: int = 1,
                     speed: str = "16th") -> None:
        """Add an arpeggio pattern"""
        pattern_obj = self.current_pattern
        if not pattern_obj:
            return
        
        track = pattern_obj.get_track(channel) or pattern_obj.add_track(channel, "Arp")
        
        # Note patterns
        patterns = {
            "up": [0, 4, 7, 12],
            "down": [12, 7, 4, 0],
            "updown": [0, 4, 7, 12, 7, 4],
            "random": [0, 7, 12, 4],
        }
        
        notes = patterns.get(pattern, patterns["up"])
        step_duration = {"16th": 240, "8th": 480, "4th": 960}[speed]
        
        for octave in range(octave_range):
            for i, interval in enumerate(notes):
                pitch = root + interval + (octave * 12)
                if pitch <= 127:
                    track.add_note(pitch, i * step_duration, step_duration // 2, 100)
    
    # ----- MIDI Export -----
    
    def export_midi(self, filepath: str = None) -> bytes:
        """Export sequencer data to MIDI file bytes"""
        # MIDI file header
        header = b'MThd'
        header += struct.pack('>IHHH', 6, 1, len(self.patterns), self.ppq)
        
        # Build track data
        track_data = bytearray()
        
        # Add tempo track (Meta event 0x51 = tempo)
        microseconds_per_beat = int(60000000 / self.bpm)
        # Delta time (0) + FF (meta) + 51 (tempo) + 03 (length) + 3 bytes tempo
        tempo_event = bytes([0, 0xFF, 0x51, 0x03, 
                            (microseconds_per_beat >> 16) & 0xFF,
                            (microseconds_per_beat >> 8) & 0xFF,
                            microseconds_per_beat & 0xFF])
        track_data += tempo_event
        track_data += struct.pack('>IBH', 0, 0x2F, 0)  # End of track
        
        # Add note events for each pattern
        for pattern in self.patterns:
            for track in pattern.tracks:
                # Set instrument/program
                track_data += struct.pack('>IBH', 0, 0xC0 | track.channel, 0)
                
                # Set volume and pan
                track_data += struct.pack('>IBH', 0, 0xB0 | track.channel, 7)
                track_data += struct.pack('>IBH', 0, track.volume)
                track_data += struct.pack('>IBH', 0, 0xB0 | track.channel, 10)
                track_data += struct.pack('>IBH', 0, track.pan)
                
                # Sort notes by start time
                sorted_notes = sorted(track.notes, key=lambda n: n.start)
                
                # Add note events
                for note in sorted_notes:
                    # Note on
                    track_data += struct.pack('>IBHB', 
                                              note.start, 
                                              0x90 | track.channel, 
                                              note.pitch, 
                                              note.velocity)
                    # Note off
                    track_data += struct.pack('>IBHB', 
                                              note.end, 
                                              0x80 | track.channel, 
                                              note.pitch, 
                                              0)
                
                # Add CC events
                sorted_cc = sorted(track.cc_events, key=lambda c: c.time)
                for cc in sorted_cc:
                    track_data += struct.pack('>IBHB',
                                             cc.time,
                                             0xB0 | track.channel,
                                             cc.cc_number,
                                             cc.value)
                
                # End of track
                track_data += struct.pack('>IBH', 0, 0x2F, 0)
        
        # Assemble file
        midi_file = header + b'MTrk' + struct.pack('>I', len(track_data)) + bytes(track_data)
        
        if filepath:
            Path(filepath).write_bytes(midi_file)
        
        return midi_file
    
    def import_midi(self, filepath: str) -> bool:
        """Import MIDI file into sequencer"""
        try:
            data = Path(filepath).read_bytes()
            
            # Parse header
            if data[:4] != b'MThd':
                return False
            
            header = struct.unpack('>IHHH', data[4:14])
            format_type, num_tracks, ppq = header[1], header[2], header[3]
            
            self.ppq = ppq
            self.patterns.clear()
            
            offset = 14
            for _ in range(num_tracks):
                if data[offset:offset+4] != b'MTrk':
                    break
                
                track_len = struct.unpack('>I', data[offset+4:offset+8])[0]
                track_data = data[offset+8:offset+8+track_len]
                offset += 8 + track_len
                
                # Parse track events
                track = Track(channel=0)
                time = 0
                i = 0
                
                while i < len(track_data):
                    # Read delta time
                    delta = 0
                    while i < len(track_data):
                        delta = (delta << 7) | (track_data[i] & 0x7F)
                        i += 1
                        if not (track_data[i-1] & 0x80):
                            break
                    
                    time += delta
                    
                    if i >= len(track_data):
                        break
                    
                    status = track_data[i]
                    i += 1
                    
                    if status & 0x80:
                        channel = status & 0x0F
                        event_type = status & 0xF0
                    else:
                        # Running status
                        event_type = last_status & 0xF0
                        i -= 1
                    
                    last_status = status
                    
                    if event_type == 0x90:  # Note on
                        if i + 2 <= len(track_data):
                            pitch = track_data[i]
                            velocity = track_data[i + 1]
                            i += 2
                            if velocity > 0:
                                track.add_note(pitch, time, 240, velocity)
                    
                    elif event_type == 0x80:  # Note off
                        if i + 2 <= len(track_data):
                            pitch = track_data[i]
                            i += 2
                    
                    elif event_type == 0xB0:  # CC
                        if i + 2 <= len(track_data):
                            cc_num = track_data[i]
                            value = track_data[i + 1]
                            i += 2
                            track.add_cc(cc_num, value, time)
                    
                    elif event_type == 0xC0:  # Program change
                        if i < len(track_data):
                            program = track_data[i]
                            i += 1
                    
                    elif event_type == 0xFF:  # Meta event
                        if i < len(track_data):
                            meta_type = track_data[i]
                            i += 1
                            if i < len(track_data):
                                length = track_data[i]
                                i += 1 + length
                
                if track.notes or track.cc_events:
                    track.channel = channel
                    pattern = self.create_pattern(name=f"Imported Track {channel}")
                    pattern.tracks = [track]
            
            return True
            
        except Exception as e:
            print(f"Error importing MIDI: {e}")
            return False
    
    # ----- Sequencer Operations -----
    
    def clear_all(self) -> None:
        """Clear all patterns"""
        self.patterns.clear()
        self.current_pattern = None
        self.create_pattern()
    
    def duplicate_pattern(self, index: int) -> Optional[Pattern]:
        """Duplicate a pattern"""
        if 0 <= index < len(self.patterns):
            original = self.patterns[index]
            new_pattern = Pattern(
                name=f"{original.name} (copy)",
                length=original.length,
                time_signature=original.time_signature,
                tracks=[Track(channel=t.channel, 
                             instrument=t.instrument,
                             notes=list(t.notes),
                             cc_events=list(t.cc_events),
                             volume=t.volume,
                             pan=t.pan) 
                       for t in original.tracks]
            )
            self.patterns.append(new_pattern)
            return new_pattern
        return None
    
    def merge_patterns(self, indices: List[int]) -> Optional[Pattern]:
        """Merge multiple patterns into one"""
        if not indices:
            return None
        
        merged = Pattern(name="Merged")
        
        for idx in indices:
            if 0 <= idx < len(self.patterns):
                pattern = self.patterns[idx]
                for track in pattern.tracks:
                    existing = merged.get_track(track.channel)
                    if existing:
                        existing.notes.extend(track.notes)
                        existing.cc_events.extend(track.cc_events)
                    else:
                        merged.tracks.append(Track(
                            channel=track.channel,
                            instrument=track.instrument,
                            notes=list(track.notes),
                            cc_events=list(track.cc_events)
                        ))
        
        self.patterns.append(merged)
        return merged
    
    def get_pattern_data(self) -> Dict[str, Any]:
        """Get pattern data as dictionary"""
        return {
            "bpm": self.bpm,
            "ppq": self.ppq,
            "patterns": [
                {
                    "name": p.name,
                    "length": p.length,
                    "time_signature": p.time_signature,
                    "tracks": [
                        {
                            "channel": t.channel,
                            "instrument": t.instrument,
                            "note_count": t.note_count,
                            "volume": t.volume,
                            "pan": t.pan
                        }
                        for t in p.tracks
                    ]
                }
                for p in self.patterns
            ]
        }
    
    # ----- Preset Patterns -----
    
    def load_preset(self, preset: str) -> Pattern:
        """Load a preset pattern"""
        presets = {
            "techno_kick": self._preset_techno_kick,
            "house_kick": self._preset_house_kick,
            "breakbeat": self._preset_breakbeat,
            "bassline": self._preset_bassline,
            "arp_chords": self._preset_arp_chords,
        }
        
        if preset in presets:
            return presets[preset]()
        
        return self.create_pattern(name=preset)
    
    def _preset_techno_kick(self) -> Pattern:
        """Techno kick pattern"""
        self.create_pattern(length=3840, name="Techno Kick")
        track = self.current_pattern.tracks[0]
        
        # Driving 4-on-the-floor with variations
        for bar in range(4):
            beat = bar * 3840
            track.add_note(36, beat, 480, 127)           # Kick
            track.add_note(36, beat + 1920, 480, 110)    # Kick on &
        
        return self.current_pattern
    
    def _preset_house_kick(self) -> Pattern:
        """House kick pattern"""
        self.create_pattern(length=3840, name="House Kick")
        track = self.current_pattern.tracks[0]
        
        for bar in range(4):
            beat = bar * 3840
            track.add_note(36, beat, 960, 127)
        
        return self.current_pattern
    
    def _preset_breakbeat(self) -> Pattern:
        """Breakbeat pattern"""
        self.create_pattern(length=7680, name="Breakbeat")  # 2 bars
        
        # Kick on 1 and 3
        for bar in range(2):
            track = self.current_pattern.add_track(9, "Drums")
            track.add_note(36, bar * 3840, 480, 127)
            track.add_note(36, bar * 3840 + 1920, 480, 100)
        
        # Snare on 2 and 4
        track = self.current_pattern.get_track(9)
        if track:
            track.add_note(38, 1920, 480, 120)
            track.add_note(38, 5760, 480, 120)
        
        return self.current_pattern
    
    def _preset_bassline(self) -> Pattern:
        """Basic bassline"""
        self.create_pattern(length=3840, name="Bassline")
        track = self.current_pattern.tracks[0]
        
        # Offbeat bass
        for step in range(8):
            track.add_note(36 + (step % 4), step * 480, 240, 100)
        
        return self.current_pattern
    
    def _preset_arp_chords(self) -> Pattern:
        """Arpeggiated chords"""
        self.create_pattern(length=3840, name="Arp Chords")
        
        # Chord track
        chord_track = self.current_pattern.tracks[0]
        chords = [(60, 64, 67), (65, 69, 72), (67, 71, 74), (60, 64, 67)]
        
        for i, chord in enumerate(chords):
            tick = i * 960
            for pitch in chord:
                chord_track.add_note(pitch, tick, 720, 80)
        
        # Arp track
        arp_track = self.current_pattern.add_track(1, "Arp")
        arp_notes = [60, 64, 67, 72, 67, 64]
        
        for i, pitch in enumerate(arp_notes):
            arp_track.add_note(pitch, i * 320, 160, 90)
        
        return self.current_pattern


# ----- Convenience Functions -----

def create_sequencer(bpm: float = 128) -> MIDISequencer:
    """Create a new sequencer with default settings"""
    return MIDISequencer(bpm=bpm)


def load_midi_file(filepath: str, bpm: float = 120) -> Optional[MIDISequencer]:
    """Load a MIDI file into a new sequencer"""
    sequencer = MIDISequencer(bpm=bpm)
    if sequencer.import_midi(filepath):
        return sequencer
    return None


def save_midi_file(sequencer: MIDISequencer, filepath: str) -> bool:
    """Save sequencer to MIDI file"""
    try:
        sequencer.export_midi(filepath)
        return True
    except Exception as e:
        print(f"Error saving MIDI: {e}")
        return False


# ----- Example Usage -----

if __name__ == "__main__":
    # Create sequencer
    seq = MIDISequencer(bpm=128)
    
    # Load a preset pattern
    pattern = seq.load_preset("techno_kick")
    
    # Add a drum track
    seq.add_drum_pattern(
        kick="x---x---x---x---",
        snare="----x-------x---",
        hihat="x-x-x-x-x-x-x-x"
    )
    
    # Export to MIDI
    midi_bytes = seq.export_midi()
    print(f"Generated {len(midi_bytes)} bytes of MIDI data")
    print(f"Pattern: {pattern.name}, {pattern.bars} bars")
    print(f"Total notes: {sum(t.note_count for t in pattern.tracks)}")
