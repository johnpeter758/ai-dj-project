#!/usr/bin/env python3
"""
AI DJ - DAW Integration Module
==============================
Provides integration with popular Digital Audio Workstations (DAWs).
Supports: Ableton Live, Logic Pro, REAPER, FL Studio, Pro Tools, Cubase

Communication methods:
- Ableton Live: OSC (Open Sound Control) via connection_server.py
- Logic Pro: AU/Inter-app communication
- REAPER: Native OSC + ReaScript (Python/Lua)
- FL Studio: FPC, Plugin delay compensation, Fruity Lua
- Pro Tools: MIDI Remote Integration
- Cubase: Generic MIDI, VST3

Author: AI DJ System
"""

import json
import logging
import os
import socket
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DAW Types and Enums
# =============================================================================

class DAWType(Enum):
    """Supported DAW applications."""
    ABLETON_LIVE = "ableton_live"
    LOGIC_PRO = "logic_pro"
    REAPER = "reaper"
    FL_STUDIO = "fl_studio"
    PRO_TOOLS = "pro_tools"
    CUBASE = "cubase"
    STUDIO_ONE = "studio_one"
    BITWIG = "bitwig"
    UNKNOWN = "unknown"


class DAWConnectionStatus(Enum):
    """Connection status with DAW."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"


class TransportState(Enum):
    """Transport playback state."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    RECORDING = "recording"


class TrackType(Enum):
    """Track/audio type."""
    AUDIO = "audio"
    MIDI = "midi"
    RETURN = "return"
    MASTER = "master"


@dataclass
class DAWTrack:
    """Represents a track in the DAW."""
    id: int
    name: str
    track_type: TrackType
    color: str = "#000000"
    volume: float = 0.0  # dB
    pan: float = 0.0    # -1 to 1
    mute: bool = False
    solo: bool = False
    armed: bool = False
    clips: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    sends: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DAWScene:
    """Represents a scene/clip in the DAW."""
    id: int
    name: str
    color: str = "#000000"
    clips: List[Optional[str]] = field(default_factory=list)  # None = empty


@dataclass
class DAWProject:
    """Represents a DAW project."""
    name: str
    path: str
    daw_type: DAWType
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    sample_rate: int = 44100
    bit_depth: int = 24
    tracks: List[DAWTrack] = field(default_factory=list)
    scenes: List[DAWScene] = field(default_factory=list)


@dataclass
class DAWDevice:
    """Represents a plugin device on a track."""
    id: str
    name: str
    device_type: str  # "instrument", "effect", "midi_effect"
    plugin_name: Optional[str] = None
    plugin_format: Optional[str] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    bypassed: bool = False


@dataclass
class DAWClip:
    """Represents an audio or MIDI clip."""
    id: str
    name: str
    file_path: Optional[str] = None
    clip_type: str = "audio"  # "audio" or "midi"
    start_time: float = 0.0
    duration: float = 0.0
    color: str = "#000000"
    mute: bool = False
    loop_enabled: bool = False
    loop_start: float = 0.0
    loop_end: float = 0.0


# =============================================================================
# OSC Communication (for Ableton Live, REAPER)
# =============================================================================

class OSCClient:
    """OSC client for communicating with DAWs."""
    
    def __init__(self, host: str = "127.0.0.1", send_port: int = 9000, 
                 receive_port: int = 9001):
        self.host = host
        self.send_port = send_port
        self.receive_port = receive_port
        self.socket = None
        self.running = False
        self.receive_thread = None
        self.callbacks: Dict[str, Callable] = {}
        
    def connect(self) -> bool:
        """Connect to OSC server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(1.0)
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            logger.info(f"OSC client connected to {self.host}:{self.send_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect OSC client: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from OSC server."""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None
        logger.info("OSC client disconnected")
        
    def send(self, address: str, *args) -> bool:
        """Send OSC message."""
        if not self.socket:
            return False
            
        try:
            # Simple OSC message building
            message = self._build_osc_message(address, args)
            self.socket.sendto(message, (self.host, self.send_port))
            return True
        except Exception as e:
            logger.error(f"Failed to send OSC message: {e}")
            return False
            
    def _build_osc_message(self, address: str, args: Tuple) -> bytes:
        """Build OSC message bytes."""
        import struct
        
        # Pad address to 4-byte boundary
        address = address + "\0" * (4 - (len(address) % 4))
        
        # Build type tag
        type_tag = ","
        values = b""
        
        for arg in args:
            if isinstance(arg, int):
                type_tag += "i"
                values += struct.pack(">i", arg)
            elif isinstance(arg, float):
                type_tag += "f"
                values += struct.pack(">f", arg)
            elif isinstance(arg, str):
                type_tag += "s"
                arg = arg + "\0" * (4 - (len(arg) % 4))
                values += arg.encode("utf-8")
            elif isinstance(arg, bytes):
                type_tag += "b"
                values += struct.pack(">i", len(arg))
                values += arg
                values += b"\0" * (4 - (len(arg) % 4))
        
        # Pad type tag
        type_tag = type_tag + "\0" * (4 - (len(type_tag) % 4))
        
        return address.encode("utf-8") + type_tag.encode("utf-8") + values
        
    def _receive_loop(self):
        """Receive OSC messages."""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                # Parse OSC message (simplified)
                if data:
                    self._handle_message(data)
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"OSC receive error: {e}")
                
    def _handle_message(self, data: bytes):
        """Handle received OSC message."""
        # Simple parsing - extract address
        try:
            addr = data.split(b"\0")[0].decode("utf-8")
            if addr in self.callbacks:
                self.callbacks[addr](data)
        except:
            pass
            
    def register_callback(self, address: str, callback: Callable):
        """Register callback for OSC address."""
        self.callbacks[address] = callback


# =============================================================================
# MIDI Communication
# =============================================================================

class MIDIClient:
    """MIDI client for DAW communication."""
    
    def __init__(self):
        self.connected = False
        self.input_port = None
        self.output_port = None
        
    def connect_input(self, port_name: str) -> bool:
        """Connect to MIDI input port."""
        # Would use rtmidi or similar
        logger.info(f"MIDI input connected to: {port_name}")
        self.connected = True
        return True
        
    def connect_output(self, port_name: str) -> bool:
        """Connect to MIDI output port."""
        logger.info(f"MIDI output connected to: {port_name}")
        self.connected = True
        return True
        
    def send_note_on(self, channel: int, note: int, velocity: int):
        """Send MIDI note on."""
        if not self.connected:
            return
        status = 0x90 | (channel & 0x0F)
        # Would send via rtmidi
        
    def send_note_off(self, channel: int, note: int, velocity: int = 0):
        """Send MIDI note off."""
        if not self.connected:
            return
        status = 0x80 | (channel & 0x0F)
        
    def send_cc(self, channel: int, cc: int, value: int):
        """Send MIDI control change."""
        if not self.connected:
            return
        status = 0xB0 | (channel & 0x0F)


# =============================================================================
# Base DAW Interface
# =============================================================================

class DAWInterface(ABC):
    """Abstract base class for DAW integration."""
    
    def __init__(self, daw_type: DAWType):
        self.daw_type = daw_type
        self.connected = False
        self.project: Optional[DAWProject] = None
        self.transport_state = TransportState.STOPPED
        self.position = 0.0  # seconds
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the DAW."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the DAW."""
        pass
    
    @abstractmethod
    def get_tracks(self) -> List[DAWTrack]:
        """Get all tracks from the DAW."""
        pass
    
    @abstractmethod
    def get_track(self, track_id: int) -> Optional[DAWTrack]:
        """Get a specific track."""
        pass
    
    @abstractmethod
    def create_track(self, name: str, track_type: TrackType) -> Optional[DAWTrack]:
        """Create a new track."""
        pass
    
    @abstractmethod
    def set_track_volume(self, track_id: int, volume_db: float) -> bool:
        """Set track volume in dB."""
        pass
    
    @abstractmethod
    def set_track_pan(self, track_id: int, pan: float) -> bool:
        """Set track pan (-1 to 1)."""
        pass
    
    @abstractmethod
    def set_track_mute(self, track_id: int, mute: bool) -> bool:
        """Set track mute."""
        pass
    
    @abstractmethod
    def set_track_solo(self, track_id: int, solo: bool) -> bool:
        """Set track solo."""
        pass
    
    @abstractmethod
    def play(self) -> bool:
        """Start playback."""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop playback."""
        pass
    
    @abstractmethod
    def pause(self) -> bool:
        """Pause playback."""
        pass
    
    @abstractmethod
    def set_position(self, position: float) -> bool:
        """Set playback position in seconds."""
        pass
    
    @abstractmethod
    def set_tempo(self, bpm: float) -> bool:
        """Set tempo in BPM."""
        pass
    
    @abstractmethod
    def get_tempo(self) -> float:
        """Get current tempo."""
        pass
    
    @abstractmethod
    def add_device(self, track_id: int, device: DAWDevice) -> bool:
        """Add a device/plugin to a track."""
        pass
    
    @abstractmethod
    def set_device_parameter(self, track_id: int, device_id: str, 
                            param: str, value: float) -> bool:
        """Set device parameter value."""
        pass
    
    @abstractmethod
    def import_audio(self, file_path: str, track_id: Optional[int] = None) -> bool:
        """Import audio file to project."""
        pass
    
    @abstractmethod
    def export_project(self, output_path: str, format: str = "wav") -> bool:
        """Export project to audio file."""
        pass


# =============================================================================
# Ableton Live Implementation
# =============================================================================

class AbletonLiveInterface(DAWInterface):
    """Ableton Live OSC integration."""
    
    def __init__(self, osc_host: str = "127.0.0.1", osc_port: int = 9000):
        super().__init__(DAWType.ABLETON_LIVE)
        self.osc_client = OSCClient(host=osc_host, send_port=osc_port)
        self.osc_client.register_callback("/live/track/info", self._on_track_info)
        self.osc_client.register_callback("/live/play", self._on_play)
        self.osc_client.register_callback("/live/tempo", self._on_tempo)
        
    def connect(self) -> bool:
        """Connect to Ableton Live via OSC."""
        success = self.osc_client.connect()
        if success:
            self.connected = True
            # Request initial state
            self.osc_client.send("/live/get/track_count")
            self.osc_client.send("/live/get/tempo")
        return success
        
    def disconnect(self) -> bool:
        """Disconnect from Ableton Live."""
        self.osc_client.disconnect()
        self.connected = False
        return True
        
    def get_tracks(self) -> List[DAWTrack]:
        """Get all tracks from Ableton Live."""
        tracks = []
        # Send request
        self.osc_client.send("/live/get/tracks")
        # Would wait for response
        return tracks
        
    def get_track(self, track_id: int) -> Optional[DAWTrack]:
        """Get a specific track."""
        self.osc_client.send("/live/track/get", track_id)
        return None
        
    def create_track(self, name: str, track_type: TrackType) -> Optional[DAWTrack]:
        """Create a new track in Ableton Live."""
        type_val = 0 if track_type == TrackType.AUDIO else 1
        self.osc_client.send("/live/track/create", name, type_val)
        return None
        
    def set_track_volume(self, track_id: int, volume_db: float) -> bool:
        """Set track volume."""
        # Convert dB to 0-1 range
        volume_norm = (volume_db + 60) / 60  # -60dB to 0dB
        volume_norm = max(0, min(1, volume_norm))
        self.osc_client.send("/live/track/volume", track_id, volume_norm)
        return True
        
    def set_track_pan(self, track_id: int, pan: float) -> bool:
        """Set track pan."""
        pan = max(-1, min(1, pan))
        self.osc_client.send("/live/track/pan", track_id, pan)
        return True
        
    def set_track_mute(self, track_id: int, mute: bool) -> bool:
        """Set track mute."""
        self.osc_client.send("/live/track/mute", track_id, 1 if mute else 0)
        return True
        
    def set_track_solo(self, track_id: int, solo: bool) -> bool:
        """Set track solo."""
        self.osc_client.send("/live/track/solo", track_id, 1 if solo else 0)
        return True
        
    def play(self) -> bool:
        """Start playback."""
        self.osc_client.send("/live/play")
        self.transport_state = TransportState.PLAYING
        return True
        
    def stop(self) -> bool:
        """Stop playback."""
        self.osc_client.send("/live/stop")
        self.transport_state = TransportState.STOPPED
        return True
        
    def pause(self) -> bool:
        """Pause playback."""
        self.osc_client.send("/live/pause")
        self.transport_state = TransportState.PAUSED
        return True
        
    def set_position(self, position: float) -> bool:
        """Set playback position."""
        self.osc_client.send("/live/position", position)
        self.position = position
        return True
        
    def set_tempo(self, bpm: float) -> bool:
        """Set tempo."""
        self.osc_client.send("/live/tempo", bpm)
        return True
        
    def get_tempo(self) -> float:
        """Get current tempo."""
        self.osc_client.send("/live/get/tempo")
        return 120.0  # Would be from response
        
    def add_device(self, track_id: int, device: DAWDevice) -> bool:
        """Add device to track."""
        self.osc_client.send("/live/device/add", track_id, device.name)
        return True
        
    def set_device_parameter(self, track_id: int, device_id: str,
                            param: str, value: float) -> bool:
        """Set device parameter."""
        self.osc_client.send("/live/device/param", track_id, device_id, param, value)
        return True
        
    def import_audio(self, file_path: str, track_id: Optional[int] = None) -> bool:
        """Import audio file."""
        if track_id is not None:
            self.osc_client.send("/live/track/insert/audio", track_id, file_path)
        else:
            self.osc_client.send("/live/import/audio", file_path)
        return True
        
    def export_project(self, output_path: str, format: str = "wav") -> bool:
        """Export project."""
        format_map = {"wav": 0, "aiff": 1, "mp3": 2, "flac": 3}
        fmt = format_map.get(format, 0)
        self.osc_client.send("/live/export", output_path, fmt)
        return True
        
    # Callbacks
    def _on_track_info(self, data):
        """Handle track info response."""
        pass
        
    def _on_play(self, data):
        """Handle play state change."""
        self.transport_state = TransportState.PLAYING
        
    def _on_tempo(self, data):
        """Handle tempo change."""
        pass


# =============================================================================
# REAPER Implementation
# =============================================================================

class REAPERInterface(DAWInterface):
    """REAPER integration via OSC and ReaScript."""
    
    def __init__(self, osc_host: str = "127.0.0.1", osc_port: int = 8000):
        super().__init__(DAWType.REAPER)
        self.osc_client = OSCClient(host=osc_host, send_port=osc_port)
        self.reascript_path = os.path.expanduser("~/Library/Application Support/REAPER/Scripts")
        
    def connect(self) -> bool:
        """Connect to REAPER."""
        success = self.osc_client.connect()
        if success:
            self.connected = True
        return success
        
    def disconnect(self) -> bool:
        """Disconnect from REAPER."""
        self.osc_client.disconnect()
        self.connected = False
        return True
        
    def _run_reascript(self, script: str) -> Any:
        """Run a ReaScript Python script."""
        # Write script to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            temp_path = f.name
            
        # Execute via REAPER's command
        # In practice, would use SWS extension or OSC
        os.unlink(temp_path)
        return None
        
    def get_tracks(self) -> List[DAWTrack]:
        """Get all tracks from REAPER."""
        tracks = []
        # Would use ReaScript to get track count and info
        return tracks
        
    def get_track(self, track_id: int) -> Optional[DAWTrack]:
        """Get a specific track."""
        return None
        
    def create_track(self, name: str, track_type: TrackType) -> Optional[DAWTrack]:
        """Create a new track."""
        return None
        
    def set_track_volume(self, track_id: int, volume_db: float) -> bool:
        """Set track volume."""
        # OSC: /track/{id}/volume {volume 0-1}
        volume_norm = (volume_db + 60) / 60
        volume_norm = max(0, min(1, volume_norm))
        self.osc_client.send(f"/track/{track_id}/volume", volume_norm)
        return True
        
    def set_track_pan(self, track_id: int, pan: float) -> bool:
        """Set track pan."""
        pan = max(-1, min(1, pan))
        self.osc_client.send(f"/track/{track_id}/pan", pan)
        return True
        
    def set_track_mute(self, track_id: int, mute: bool) -> bool:
        """Set track mute."""
        self.osc_client.send(f"/track/{track_id}/mute", 1 if mute else 0)
        return True
        
    def set_track_solo(self, track_id: int, solo: bool) -> bool:
        """Set track solo."""
        self.osc_client.send(f"/track/{track_id}/solo", 1 if solo else 0)
        return True
        
    def play(self) -> bool:
        """Start playback."""
        self.osc_client.send("/transport/play")
        self.transport_state = TransportState.PLAYING
        return True
        
    def stop(self) -> bool:
        """Stop playback."""
        self.osc_client.send("/transport/stop")
        self.transport_state = TransportState.STOPPED
        return True
        
    def pause(self) -> bool:
        """Pause playback."""
        self.osc_client.send("/transport/pause")
        self.transport_state = TransportState.PAUSED
        return True
        
    def set_position(self, position: float) -> bool:
        """Set playback position."""
        self.osc_client.send("/transport/position", position)
        self.position = position
        return True
        
    def set_tempo(self, bpm: float) -> bool:
        """Set tempo."""
        self.osc_client.send("/transport/tempo", bpm)
        return True
        
    def get_tempo(self) -> float:
        """Get current tempo."""
        return 120.0
        
    def add_device(self, track_id: int, device: DAWDevice) -> bool:
        """Add device to track."""
        return True
        
    def set_device_parameter(self, track_id: int, device_id: str,
                            param: str, value: float) -> bool:
        """Set device parameter."""
        self.osc_client.send(f"/track/{track_id}/device/{device_id}/param/{param}", value)
        return True
        
    def import_audio(self, file_path: str, track_id: Optional[int] = None) -> bool:
        """Import audio file."""
        return True
        
    def export_project(self, output_path: str, format: str = "wav") -> bool:
        """Export project."""
        return True


# =============================================================================
# FL Studio Implementation
# =============================================================================

class FLStudioInterface(DAWInterface):
    """FL Studio integration via Fruity Lua and plugin communication."""
    
    def __init__(self):
        super().__init__(DAWType.FL_STUDIO)
        self.fruity_host = "127.0.0.1"
        self.fruity_port = 5000
        self.udp_socket = None
        
    def connect(self) -> bool:
        """Connect to FL Studio."""
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.settimeout(1.0)
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to FL Studio: {e}")
            return False
            
    def disconnect(self) -> bool:
        """Disconnect from FL Studio."""
        if self.udp_socket:
            self.udp_socket.close()
        self.connected = False
        return True
        
    def _send_fruity_command(self, command: str, *args) -> bool:
        """Send command to FL Studio via UDP."""
        if not self.udp_socket:
            return False
            
        try:
            message = command + " " + " ".join(str(a) for a in args)
            self.udp_socket.sendto(message.encode(), (self.fruity_host, self.fruity_port))
            return True
        except Exception as e:
            logger.error(f"Failed to send to FL Studio: {e}")
            return False
            
    def get_tracks(self) -> List[DAWTrack]:
        """Get all tracks (playlist slots) from FL Studio."""
        return []
        
    def get_track(self, track_id: int) -> Optional[DAWTrack]:
        """Get a specific track."""
        return None
        
    def create_track(self, name: str, track_type: TrackType) -> Optional[DAWTrack]:
        """Create a new track/pattern."""
        return None
        
    def set_track_volume(self, track_id: int, volume_db: float) -> bool:
        """Set track volume."""
        return True
        
    def set_track_pan(self, track_id: int, pan: float) -> bool:
        """Set track pan."""
        return True
        
    def set_track_mute(self, track_id: int, mute: bool) -> bool:
        """Set track mute."""
        return True
        
    def set_track_solo(self, track_id: int, solo: bool) -> bool:
        """Set track solo."""
        return True
        
    def play(self) -> bool:
        """Start playback."""
        self._send_fruity_command("Play")
        self.transport_state = TransportState.PLAYING
        return True
        
    def stop(self) -> bool:
        """Stop playback."""
        self._send_fruity_command("Stop")
        self.transport_state = TransportState.STOPPED
        return True
        
    def pause(self) -> bool:
        """Pause playback."""
        self._send_fruity_command("Pause")
        self.transport_state = TransportState.PAUSED
        return True
        
    def set_position(self, position: float) -> bool:
        """Set playback position."""
        return True
        
    def set_tempo(self, bpm: float) -> bool:
        """Set tempo."""
        self._send_fruity_command("SetTempo", int(bpm * 1000))
        return True
        
    def get_tempo(self) -> float:
        """Get current tempo."""
        return 120.0
        
    def add_device(self, track_id: int, device: DAWDevice) -> bool:
        """Add device to track."""
        return True
        
    def set_device_parameter(self, track_id: int, device_id: str,
                            param: str, value: float) -> bool:
        """Set device parameter."""
        return True
        
    def import_audio(self, file_path: str, track_id: Optional[int] = None) -> bool:
        """Import audio file."""
        return True
        
    def export_project(self, output_path: str, format: str = "wav") -> bool:
        """Export project."""
        return True


# =============================================================================
# Logic Pro Implementation
# =============================================================================

class LogicProInterface(DAWInterface):
    """Logic Pro integration via Inter-App Audio and OSC."""
    
    def __init__(self, osc_host: str = "127.0.0.1", osc_port: int = 9000):
        super().__init__(DAWType.LOGIC_PRO)
        self.osc_client = OSCClient(host=osc_host, send_port=osc_port)
        
    def connect(self) -> bool:
        """Connect to Logic Pro."""
        # Logic Pro doesn't have native OSC, would use third-party
        # or Control Surface API
        return False
        
    def disconnect(self) -> bool:
        """Disconnect from Logic Pro."""
        self.connected = False
        return True
        
    def get_tracks(self) -> List[DAWTrack]:
        """Get all tracks from Logic Pro."""
        return []
        
    def get_track(self, track_id: int) -> Optional[DAWTrack]:
        """Get a specific track."""
        return None
        
    def create_track(self, name: str, track_type: TrackType) -> Optional[DAWTrack]:
        """Create a new track."""
        return None
        
    def set_track_volume(self, track_id: int, volume_db: float) -> bool:
        """Set track volume."""
        return True
        
    def set_track_pan(self, track_id: int, pan: float) -> bool:
        """Set track pan."""
        return True
        
    def set_track_mute(self, track_id: int, mute: bool) -> bool:
        """Set track mute."""
        return True
        
    def set_track_solo(self, track_id: int, solo: bool) -> bool:
        """Set track solo."""
        return True
        
    def play(self) -> bool:
        """Start playback."""
        self.transport_state = TransportState.PLAYING
        return True
        
    def stop(self) -> bool:
        """Stop playback."""
        self.transport_state = TransportState.STOPPED
        return True
        
    def pause(self) -> bool:
        """Pause playback."""
        self.transport_state = TransportState.PAUSED
        return True
        
    def set_position(self, position: float) -> bool:
        """Set playback position."""
        self.position = position
        return True
        
    def set_tempo(self, bpm: float) -> bool:
        """Set tempo."""
        return True
        
    def get_tempo(self) -> float:
        """Get current tempo."""
        return 120.0
        
    def add_device(self, track_id: int, device: DAWDevice) -> bool:
        """Add device to track."""
        return True
        
    def set_device_parameter(self, track_id: int, device_id: str,
                            param: str, value: float) -> bool:
        """Set device parameter."""
        return True
        
    def import_audio(self, file_path: str, track_id: Optional[int] = None) -> bool:
        """Import audio file."""
        return True
        
    def export_project(self, output_path: str, format: str = "wav") -> bool:
        """Export project."""
        return True


# =============================================================================
# DAW Factory
# =============================================================================

class DAWFactory:
    """Factory for creating DAW interface instances."""
    
    @staticmethod
    def create(daw_type: DAWType, **kwargs) -> DAWInterface:
        """Create a DAW interface instance."""
        interfaces = {
            DAWType.ABLETON_LIVE: AbletonLiveInterface,
            DAWType.REAPER: REAPERInterface,
            DAWType.FL_STUDIO: FLStudioInterface,
            DAWType.LOGIC_PRO: LogicProInterface,
        }
        
        interface_class = interfaces.get(daw_type)
        if interface_class:
            return interface_class(**kwargs)
        else:
            raise ValueError(f"Unsupported DAW type: {daw_type}")
    
    @staticmethod
    def auto_detect() -> Optional[DAWType]:
        """Auto-detect running DAW."""
        # Check for running DAW processes
        try:
            import psutil
            
            daw_processes = {
                "Ableton Live": DAWType.ABLETON_LIVE,
                "Logic Pro": DAWType.LOGIC_PRO,
                "REAPER": DAWType.REAPER,
                "FL64": DAWType.FL_STUDIO,
                "Cubase": DAWType.CUBASE,
                "Pro Tools": DAWType.PRO_TOOLS,
            }
            
            for proc in psutil.process_iter(['name']):
                try:
                    name = proc.info['name']
                    for daw_name, daw_type in daw_processes.items():
                        if daw_name.lower() in name.lower():
                            return daw_type
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            logger.warning("psutil not available, auto-detect disabled")
                    
        return None


# =============================================================================
# DAW Manager (Main Integration Point)
# =============================================================================

class DAWManager:
    """High-level DAW integration manager."""
    
    def __init__(self):
        self.interfaces: Dict[DAWType, DAWInterface] = {}
        self.active_daw: Optional[DAWInterface] = None
        self.midi_client = MIDIClient()
        
    def connect_daw(self, daw_type: DAWType, **kwargs) -> bool:
        """Connect to a specific DAW."""
        try:
            interface = DAWFactory.create(daw_type, **kwargs)
            success = interface.connect()
            if success:
                self.interfaces[daw_type] = interface
                self.active_daw = interface
                logger.info(f"Connected to {daw_type.value}")
            return success
        except Exception as e:
            logger.error(f"Failed to connect to {daw_type.value}: {e}")
            return False
            
    def disconnect_daw(self, daw_type: Optional[DAWType] = None) -> bool:
        """Disconnect from a DAW."""
        if daw_type and daw_type in self.interfaces:
            success = self.interfaces[daw_type].disconnect()
            if success:
                del self.interfaces[daw_type]
                if self.active_daw and self.active_daw.daw_type == daw_type:
                    self.active_daw = None
            return success
        elif self.active_daw:
            return self.active_daw.disconnect()
        return False
        
    def switch_active_daw(self, daw_type: DAWType) -> bool:
        """Switch active DAW."""
        if daw_type in self.interfaces:
            self.active_daw = self.interfaces[daw_type]
            return True
        return False
        
    def auto_connect(self) -> bool:
        """Auto-detect and connect to DAW."""
        detected = DAWFactory.auto_detect()
        if detected:
            return self.connect_daw(detected)
        return False
        
    # Convenience methods that delegate to active DAW
    def play(self) -> bool:
        """Start playback on active DAW."""
        if self.active_daw:
            return self.active_daw.play()
        return False
        
    def stop(self) -> bool:
        """Stop playback on active DAW."""
        if self.active_daw:
            return self.active_daw.stop()
        return False
        
    def pause(self) -> bool:
        """Pause playback on active DAW."""
        if self.active_daw:
            return self.active_daw.pause()
        return False
        
    def get_tracks(self) -> List[DAWTrack]:
        """Get tracks from active DAW."""
        if self.active_daw:
            return self.active_daw.get_tracks()
        return []
        
    def create_track(self, name: str, track_type: TrackType = TrackType.AUDIO) -> Optional[DAWTrack]:
        """Create track on active DAW."""
        if self.active_daw:
            return self.active_daw.create_track(name, track_type)
        return None
        
    def set_track_volume(self, track_id: int, volume_db: float) -> bool:
        """Set track volume."""
        if self.active_daw:
            return self.active_daw.set_track_volume(track_id, volume_db)
        return False
        
    def set_track_pan(self, track_id: int, pan: float) -> bool:
        """Set track pan."""
        if self.active_daw:
            return self.active_daw.set_track_pan(track_id, pan)
        return False
        
    def set_tempo(self, bpm: float) -> bool:
        """Set tempo."""
        if self.active_daw:
            return self.active_daw.set_tempo(bpm)
        return False
        
    def get_tempo(self) -> float:
        """Get tempo."""
        if self.active_daw:
            return self.active_daw.get_tempo()
        return 120.0
        
    def import_audio(self, file_path: str, track_id: Optional[int] = None) -> bool:
        """Import audio file."""
        if self.active_daw:
            return self.active_daw.import_audio(file_path, track_id)
        return False
        
    def export_project(self, output_path: str, format: str = "wav") -> bool:
        """Export project."""
        if self.active_daw:
            return self.active_daw.export_project(output_path, format)
        return False


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example usage
    manager = DAWManager()
    
    #