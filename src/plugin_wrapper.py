#!/usr/bin/env python3
"""
AI DJ Plugin Wrapper
====================
VST3-compatible plugin wrapper for AI music generation and processing.
Provides automation support and integrates AI components as audio plugins.
"""

import asyncio
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Plugin Types and Enums
# =============================================================================

class PluginType(Enum):
    """Types of AI DJ plugins."""
    GENERATOR = auto()      # Melody, bass, drum generation
    PROCESSOR = auto()      # Effects, mixing, mastering
    ANALYZER = auto()       # BPM, key, energy detection
    ARRANGER = auto()       # Structure, transitions
    EFFECTS = auto()        # Reverb, delay, compression


class PluginState(Enum):
    """Plugin lifecycle states."""
    UNLOADED = auto()
    LOADED = auto()
    ACTIVE = auto()
    PROCESSING = auto()
    ERROR = auto()


class AutomationSource(Enum):
    """Sources for parameter automation."""
    MIDI = auto()
    OSC = auto()
    HOST = auto()           # DAW automation
    INTERNAL = auto()       # AI-driven automation
    MIDI_CC = auto()


@dataclass
class PluginParameter:
    """Represents a VST3-style parameter with automation support."""
    id: str
    name: str
    default: float = 0.0
    min_value: float = 0.0
    max_value: float = 1.0
    unit: str = ""
    automation_source: AutomationSource = AutomationSource.HOST
    
    # Current value with automation
    value: float = field(default=0.0)
    _automation_curve: List[tuple] = field(default_factory=list)
    
    def __post_init__(self):
        self.value = self.default
    
    def set_value(self, value: float, source: AutomationSource = AutomationSource.HOST):
        """Set parameter value with automation tracking."""
        clamped = max(self.min_value, min(self.max_value, value))
        self.value = clamped
        self.automation_source = source
        # Record automation point
        self._automation_curve.append((time.time(), clamped, source))
    
    def get_value(self) -> float:
        return self.value
    
    def reset_automation(self):
        """Clear automation curve."""
        self._automation_curve.clear()
        self.value = self.default


@dataclass
class AudioBuffer:
    """Audio buffer for plugin processing."""
    samples: List[List[float]]  # [channel][sample]
    sample_rate: int = 44100
    channels: int = 2
    frames: int = 0
    
    def __post_init__(self):
        if not self.samples:
            self.samples = [[] for _ in range(self.channels)]


@dataclass
class MidiEvent:
    """MIDI event for plugin communication."""
    status: int = 0
    channel: int = 0
    note: int = 0
    velocity: int = 0
    delta_frames: int = 0


# =============================================================================
# Base Plugin Interface
# =============================================================================

class AIPlugin(ABC):
    """
    Base class for AI DJ plugins.
    Implements VST3-compatible plugin interface.
    """
    
    def __init__(self, plugin_id: str, name: str, plugin_type: PluginType):
        self.plugin_id = plugin_id
        self.name = name
        self.plugin_type = plugin_type
        self.state = PluginState.UNLOADED
        self._parameters: Dict[str, PluginParameter] = {}
        self._sample_rate = 44100
        self._block_size = 512
        self._is_processing = False
        
        # Initialize default parameters
        self._init_default_parameters()
    
    @abstractmethod
    def _init_default_parameters(self):
        """Initialize plugin-specific parameters."""
        pass
    
    @abstractmethod
    async def process(
        self, 
        audio_in: AudioBuffer, 
        midi_in: List[MidiEvent]
    ) -> AudioBuffer:
        """Process audio through the plugin."""
        pass
    
    @abstractmethod
    async def initialize(self, sample_rate: int, block_size: int) -> bool:
        """Initialize plugin with audio settings."""
        pass
    
    def initialize_sync(self, sample_rate: int, block_size: int) -> bool:
        """Synchronous initialization wrapper."""
        # Store settings, actual async init happens on first process call
        self._sample_rate = sample_rate
        self._block_size = block_size
        return True
    
    @abstractmethod
    def get_editor_size(self) -> tuple:
        """Return (width, height) for plugin editor."""
        return (400, 300)
    
    # -------------------------------------------------------------------------
    # Parameter Management
    # -------------------------------------------------------------------------
    
    def add_parameter(self, param: PluginParameter):
        """Add a parameter to the plugin."""
        self._parameters[param.id] = param
    
    def get_parameter(self, param_id: str) -> Optional[PluginParameter]:
        """Get a parameter by ID."""
        return self._parameters.get(param_id)
    
    def set_parameter(self, param_id: str, value: float, source: AutomationSource = AutomationSource.HOST):
        """Set parameter value with automation."""
        param = self._parameters.get(param_id)
        if param:
            param.set_value(value, source)
    
    def get_all_parameters(self) -> Dict[str, PluginParameter]:
        """Get all parameters."""
        return self._parameters.copy()
    
    def get_parameter_values(self) -> Dict[str, float]:
        """Get current parameter values as dict."""
        return {pid: p.value for pid, p in self._parameters.items()}
    
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    def load(self) -> bool:
        """Load the plugin."""
        try:
            self.state = PluginState.LOADED
            logger.info(f"Plugin '{self.name}' loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load plugin '{self.name}': {e}")
            self.state = PluginState.ERROR
            return False
    
    def unload(self):
        """Unload the plugin."""
        self.state = PluginState.UNLOADED
        logger.info(f"Plugin '{self.name}' unloaded")
    
    def activate(self):
        """Activate the plugin for processing."""
        self.state = PluginState.ACTIVE
    
    def deactivate(self):
        """Deactivate the plugin."""
        self.state = PluginState.LOADED


# =============================================================================
# Concrete Plugin Implementations
# =============================================================================

class MelodyGeneratorPlugin(AIPlugin):
    """AI Melody Generation Plugin."""
    
    def __init__(self):
        super().__init__("ai_dj.melody_gen", "AI Melody Generator", PluginType.GENERATOR)
        self._melody_model = None
    
    def _init_default_parameters(self):
        self.add_parameter(PluginParameter("tempo", "Tempo (BPM)", 120, 60, 200, "BPM"))
        self.add_parameter(PluginParameter("key", "Musical Key", 0, 0, 23, ""))
        self.add_parameter(PluginParameter("scale", "Scale Type", 0, 0, 10, ""))
        self.add_parameter(PluginParameter("complexity", "Complexity", 0.5, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("mood", "Mood/Energy", 0.5, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("seed", "Random Seed", 0, 0, 999999, ""))
        self.add_parameter(PluginParameter("mix_dry_wet", "Dry/Wet Mix", 1.0, 0.0, 1.0, ""))
    
    async def initialize(self, sample_rate: int, block_size: int) -> bool:
        self._sample_rate = sample_rate
        self._block_size = block_size
        # Initialize melody generation model here
        # self._melody_model = await load_model(...)
        return True
    
    async def process(self, audio_in: AudioBuffer, midi_in: List[MidiEvent]) -> AudioBuffer:
        """Generate melody based on parameters."""
        if not self._parameters:
            return audio_in
        
        # Get current parameter values
        tempo = self.get_parameter("tempo").value
        complexity = self.get_parameter("complexity").value
        mood = self.get_parameter("mood").value
        
        # Process audio (placeholder for actual generation)
        # In real implementation, this would call the melody generator
        output = AudioBuffer(
            sample_rate=audio_in.sample_rate,
            channels=audio_in.channels,
            frames=audio_in.frames
        )
        
        return output
    
    def get_editor_size(self) -> tuple:
        return (500, 400)


class DrumGeneratorPlugin(AIPlugin):
    """AI Drum Pattern Generation Plugin."""
    
    def __init__(self):
        super().__init__("ai_dj.drum_gen", "AI Drum Generator", PluginType.GENERATOR)
    
    def _init_default_parameters(self):
        self.add_parameter(PluginParameter("tempo", "Tempo (BPM)", 120, 60, 200, "BPM"))
        self.add_parameter(PluginParameter("pattern", "Pattern Select", 0, 0, 50, ""))
        self.add_parameter(PluginParameter("swing", "Swing Amount", 0.0, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("kick_level", "Kick Level", 1.0, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("snare_level", "Snare Level", 1.0, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("hihat_level", "Hi-Hat Level", 0.8, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("genre", "Genre Style", 0, 0, 20, ""))
        self.add_parameter(PluginParameter("fill_probability", "Fill Probability", 0.3, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("mix_dry_wet", "Dry/Wet Mix", 1.0, 0.0, 1.0, ""))
    
    async def initialize(self, sample_rate: int, block_size: int) -> bool:
        self._sample_rate = sample_rate
        self._block_size = block_size
        return True
    
    async def process(self, audio_in: AudioBuffer, midi_in: List[MidiEvent]) -> AudioBuffer:
        output = AudioBuffer(
            sample_rate=audio_in.sample_rate,
            channels=audio_in.channels,
            frames=audio_in.frames
        )
        return output
    
    def get_editor_size(self) -> tuple:
        return (450, 350)


class EffectsProcessorPlugin(AIPlugin):
    """AI Effects Processing Plugin with automation."""
    
    def __init__(self):
        super().__init__("ai_dj.effects", "AI Effects Processor", PluginType.PROCESSOR)
    
    def _init_default_parameters(self):
        # Reverb
        self.add_parameter(PluginParameter("reverb_mix", "Reverb Mix", 0.3, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("reverb_size", "Reverb Size", 0.5, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("reverb_damping", "Reverb Damping", 0.5, 0.0, 1.0, ""))
        
        # Delay
        self.add_parameter(PluginParameter("delay_mix", "Delay Mix", 0.2, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("delay_time", "Delay Time", 0.5, 0.0, 1.0, "ms"))
        self.add_parameter(PluginParameter("delay_feedback", "Delay Feedback", 0.4, 0.0, 0.95, ""))
        
        # Compression
        self.add_parameter(PluginParameter("comp_threshold", "Compressor Threshold", 0.5, 0.0, 1.0, "dB"))
        self.add_parameter(PluginParameter("comp_ratio", "Compressor Ratio", 4.0, 1.0, 20.0, ":1"))
        self.add_parameter(PluginParameter("comp_attack", "Compressor Attack", 10, 0.1, 100, "ms"))
        self.add_parameter(PluginParameter("comp_release", "Compressor Release", 100, 10, 1000, "ms"))
        
        # EQ
        self.add_parameter(PluginParameter("eq_low", "Low EQ", 0.0, -12, 12, "dB"))
        self.add_parameter(PluginParameter("eq_mid", "Mid EQ", 0.0, -12, 12, "dB"))
        self.add_parameter(PluginParameter("eq_high", "High EQ", 0.0, -12, 12, "dB"))
        
        # AI Automation
        self.add_parameter(PluginParameter("ai_intensity", "AI Intensity", 0.5, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("auto_transitions", "Auto Transitions", 0.0, 0.0, 1.0, ""))
    
    async def initialize(self, sample_rate: int, block_size: int) -> bool:
        self._sample_rate = sample_rate
        self._block_size = block_size
        return True
    
    async def process(self, audio_in: AudioBuffer, midi_in: List[MidiEvent]) -> AudioBuffer:
        """Process audio with effects and AI automation."""
        output = AudioBuffer(
            sample_rate=audio_in.sample_rate,
            channels=audio_in.channels,
            frames=audio_in.frames
        )
        
        # Copy input to output (in real impl, apply effects)
        for ch in range(min(audio_in.channels, output.channels)):
            output.samples[ch] = audio_in.samples[ch][:]
        
        return output
    
    def get_editor_size(self) -> tuple:
        return (600, 450)


class AnalyzerPlugin(AIPlugin):
    """Audio Analysis Plugin (BPM, Key, Energy)."""
    
    def __init__(self):
        super().__init__("ai_dj.analyzer", "AI Audio Analyzer", PluginType.ANALYZER)
        self._analysis_results = {}
    
    def _init_default_parameters(self):
        self.add_parameter(PluginParameter("sensitivity", "Analysis Sensitivity", 0.7, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("tempo_weight", "Tempo Detection Weight", 0.8, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("key_weight", "Key Detection Weight", 0.8, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("energy_weight", "Energy Detection Weight", 0.6, 0.0, 1.0, ""))
        self.add_parameter(PluginParameter("output_json", "Output JSON", 0.0, 0.0, 1.0, ""))
    
    async def initialize(self, sample_rate: int, block_size: int) -> bool:
        self._sample_rate = sample_rate
        self._block_size = block_size
        return True
    
    async def process(self, audio_in: AudioBuffer, midi_in: List[MidiEvent]) -> AudioBuffer:
        """Analyze audio and store results."""
        # Analysis would happen here
        self._analysis_results = {
            "bpm": 120.0,
            "key": "Cm",
            "energy": 0.7,
            "timbre": "warm",
            "danceability": 0.85
        }
        return audio_in
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Return latest analysis results."""
        return self._analysis_results.copy()
    
    def get_editor_size(self) -> tuple:
        return (350, 250)


# =============================================================================
# Plugin Host / Manager
# =============================================================================

class PluginHost:
    """
    VST3-compatible plugin host that manages multiple AI plugins.
    Provides automation, MIDI, and OSC support.
    """
    
    def __init__(self, sample_rate: int = 44100, block_size: int = 512):
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._plugins: Dict[str, AIPlugin] = {}
        self._plugin_chain: List[str] = []  # Processing order
        
        # Automation
        self._automation_curves: Dict[str, Dict[str, List]] = {}
        self._automation_callbacks: List[Callable] = []
        
        # MIDI/OSC
        self._midi_handlers: List[Callable] = []
        self._osc_handlers: List[Callable] = []
        
        # Processing state
        self._is_running = False
        self._processing_thread: Optional[threading.Thread] = None
        
        # Register built-in plugins
        self._register_builtin_plugins()
    
    def _register_builtin_plugins(self):
        """Register all built-in AI plugins."""
        self.register_plugin(MelodyGeneratorPlugin())
        self.register_plugin(DrumGeneratorPlugin())
        self.register_plugin(EffectsProcessorPlugin())
        self.register_plugin(AnalyzerPlugin())

    def _init_async(self):
        """Initialize async components after event loop exists."""
        for plugin in self._plugins.values():
            plugin.initialize_sync(self._sample_rate, self._block_size)
    
    # -------------------------------------------------------------------------
    # Plugin Management
    # -------------------------------------------------------------------------
    
    def register_plugin(self, plugin: AIPlugin) -> bool:
        """Register a plugin with the host."""
        if plugin.plugin_id in self._plugins:
            logger.warning(f"Plugin {plugin.plugin_id} already registered")
            return False
        
        plugin.load()
        self._plugins[plugin.plugin_id] = plugin
        
        # Try async initialization if we have an event loop, otherwise defer
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, schedule initialization
            async def init_plugin():
                await plugin.initialize(self._sample_rate, self._block_size)
            asyncio.create_task(init_plugin())
        except RuntimeError:
            # No running loop, defer initialization until later
            pass
        
        logger.info(f"Registered plugin: {plugin.name}")
        return True
    
    def unregister_plugin(self, plugin_id: str):
        """Unregister a plugin."""
        if plugin_id in self._plugins:
            self._plugins[plugin_id].unload()
            del self._plugins[plugin_id]
            if plugin_id in self._plugin_chain:
                self._plugin_chain.remove(plugin_id)
    
    def get_plugin(self, plugin_id: str) -> Optional[AIPlugin]:
        """Get a plugin by ID."""
        return self._plugins.get(plugin_id)
    
    def get_all_plugins(self) -> List[AIPlugin]:
        """Get all registered plugins."""
        return list(self._plugins.values())
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[AIPlugin]:
        """Get plugins filtered by type."""
        return [p for p in self._plugins.values() if p.plugin_type == plugin_type]
    
    # -------------------------------------------------------------------------
    # Plugin Chain
    # -------------------------------------------------------------------------
    
    def add_to_chain(self, plugin_id: str):
        """Add plugin to processing chain."""
        if plugin_id in self._plugins and plugin_id not in self._plugin_chain:
            self._plugin_chain.append(plugin_id)
    
    def remove_from_chain(self, plugin_id: str):
        """Remove plugin from processing chain."""
        if plugin_id in self._plugin_chain:
            self._plugin_chain.remove(plugin_id)
    
    def reorder_chain(self, new_order: List[str]):
        """Reorder the plugin chain."""
        if set(new_order) == set(self._plugin_chain):
            self._plugin_chain = new_order
    
    def get_chain(self) -> List[str]:
        """Get current plugin chain."""
        return self._plugin_chain.copy()
    
    # -------------------------------------------------------------------------
    # Automation
    # -------------------------------------------------------------------------
    
    def set_automation(
        self, 
        plugin_id: str, 
        param_id: str, 
        value: float,
        source: AutomationSource = AutomationSource.HOST
    ):
        """Set parameter value with automation."""
        plugin = self._plugins.get(plugin_id)
        if plugin:
            plugin.set_parameter(param_id, value, source)
            
            # Record automation
            if plugin_id not in self._automation_curves:
                self._automation_curves[plugin_id] = {}
            if param_id not in self._automation_curves[plugin_id]:
                self._automation_curves[plugin_id][param_id] = []
            
            self._automation_curves[plugin_id][param_id].append({
                "time": time.time(),
                "value": value,
                "source": source.name
            })
            
            # Notify callbacks
            for callback in self._automation_callbacks:
                try:
                    callback(plugin_id, param_id, value, source)
                except Exception as e:
                    logger.error(f"Automation callback error: {e}")
    
    def get_automation_curve(
        self, 
        plugin_id: str, 
        param_id: str
    ) -> List[Dict]:
        """Get recorded automation curve for a parameter."""
        if plugin_id in self._automation_curves:
            return self._automation_curves[plugin_id].get(param_id, [])
        return []
    
    def add_automation_callback(self, callback: Callable):
        """Add a callback for automation changes."""
        self._automation_callbacks.append(callback)
    
    def load_automation_curve(self, plugin_id: str, param_id: str, curve: List[Dict]):
        """Load a pre-recorded automation curve."""
        if plugin_id not in self._automation_curves:
            self._automation_curves[plugin_id] = {}
        self._automation_curves[plugin_id][param_id] = curve
    
    def export_automation(self, filepath: str):
        """Export all automation curves to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self._automation_curves, f, indent=2)
        logger.info(f"Automation exported to {filepath}")
    
    def import_automation(self, filepath: str):
        """Import automation curves from JSON."""
        with open(filepath, 'r') as f:
            self._automation_curves = json.load(f)
        logger.info(f"Automation imported from {filepath}")
    
    # -------------------------------------------------------------------------
    # MIDI / OSC
    # -------------------------------------------------------------------------
    
    def send_midi(self, event: MidiEvent):
        """Send MIDI event to plugins."""
        for handler in self._midi_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"MIDI handler error: {e}")
    
    def add_midi_handler(self, handler: Callable[[MidiEvent], None]):
        """Add a MIDI event handler."""
        self._midi_handlers.append(handler)
    
    def send_osc(self, address: str, *args):
        """Send OSC message to plugins."""
        for handler in self._osc_handlers:
            try:
                handler(address, args)
            except Exception as e:
                logger.error(f"OSC handler error: {e}")
    
    def add_osc_handler(self, handler: Callable):
        """Add an OSC message handler."""
        self._osc_handlers.append(handler)
    
    # -------------------------------------------------------------------------
    # Audio Processing
    # -------------------------------------------------------------------------
    
    async def process_audio(self, audio_in: AudioBuffer) -> AudioBuffer:
        """Process audio through the plugin chain."""
        current_audio = audio_in
        
        for plugin_id in self._plugin_chain:
            plugin = self._plugins.get(plugin_id)
            if plugin and plugin.state == PluginState.ACTIVE:
                plugin.state = PluginState.PROCESSING
                current_audio = await plugin.process(current_audio, [])
                plugin.state = PluginState.ACTIVE
        
        return current_audio
    
    def process_block(self, audio_data: List[List[float]]) -> List[List[float]]:
        """Synchronous block processing."""
        audio_buffer = AudioBuffer(
            samples=audio_data,
            frames=len(audio_data[0]) if audio_data else 0
        )
        
        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.process_audio(audio_buffer))
            return result.samples
        finally:
            loop.close()
    
    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    def start(self):
        """Start the plugin host."""
        self._is_running = True
        for plugin in self._plugins.values():
            plugin.activate()
        logger.info("Plugin host started")
    
    def stop(self):
        """Stop the plugin host."""
        self._is_running = False
        for plugin in self._plugins.values():
            plugin.deactivate()
        logger.info("Plugin host stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get host status."""
        return {
            "running": self._is_running,
            "sample_rate": self._sample_rate,
            "block_size": self._block_size,
            "plugins": {
                pid: {
                    "name": p.name,
                    "type": p.plugin_type.name,
                    "state": p.state.name,
                    "parameters": p.get_parameter_values()
                }
                for pid, p in self._plugins.items()
            },
            "chain": self._plugin_chain
        }


# =============================================================================
# Plugin Factory
# =============================================================================

class PluginFactory:
    """Factory for creating AI plugins."""
    
    _plugin_classes: Dict[str, type] = {}
    
    @classmethod
    def register(cls, plugin_id: str, plugin_class: type):
        """Register a plugin class."""
        cls._plugin_classes[plugin_id] = plugin_class
    
    @classmethod
    def create(cls, plugin_id: str) -> Optional[AIPlugin]:
        """Create a plugin instance."""
        plugin_class = cls._plugin_classes.get(plugin_id)
        if plugin_class:
            return plugin_class()
        return None
    
    @classmethod
    def get_available_plugins(cls) -> List[str]:
        """Get list of available plugin IDs."""
        return list(cls._plugin_classes.keys())


# Register built-in plugins
PluginFactory.register("ai_dj.melody_gen", MelodyGeneratorPlugin)
PluginFactory.register("ai_dj.drum_gen", DrumGeneratorPlugin)
PluginFactory.register("ai_dj.effects", EffectsProcessorPlugin)
PluginFactory.register("ai_dj.analyzer", AnalyzerPlugin)


# =============================================================================
# Main Entry Point / Example Usage
# =============================================================================

async def main():
    """Example usage of the plugin wrapper."""
    # Create host
    host = PluginHost(sample_rate=48000, block_size=512)
    
    # Initialize async components
    host._init_async()
    
    # Get available plugins
    plugins = host.get_all_plugins()
    print(f"Available plugins: {[p.name for p in plugins]}")
    
    # Setup chain
    host.add_to_chain("ai_dj.analyzer")
    host.add_to_chain("ai_dj.effects")
    
    # Set some automation
    host.set_automation("ai_dj.effects", "reverb_mix", 0.5)
    host.set_automation("ai_dj.effects", "delay_mix", 0.3)
    
    # Add automation callback
    def on_automation_change(plugin_id, param_id, value, source):
        print(f"Automation: {plugin_id}.{param_id} = {value} (source: {source.name})")
    
    host.add_automation_callback(on_automation_change)
    
    # Start host
    host.start()
    
    # Get status
    status = host.get_status()
    print(json.dumps(status, indent=2))
    
    # Export automation
    host.export_automation("automation.json")
    
    # Stop
    host.stop()
    
    print("Plugin host demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
