#!/usr/bin/env python3
"""
Radar Display System with Audio Visualization
Real-time radar sweep visualization that reacts to audio analysis
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum
import math
import time


class RadarMode(Enum):
    """Radar display modes."""
    CLASSIC = "classic"           # Traditional radar sweep
    AUDIO_REACTIVE = "audio"      # Reacts to audio frequency bands
    SPECTRUM = "spectrum"         # Frequency spectrum display
    BEAT_DETECT = "beat"          # Beat-pulsing radar


@dataclass
class RadarConfig:
    """Configuration for radar display."""
    resolution: int = 360          # Number of angular positions
    sweep_speed: float = 2.0      # Rotations per second
    decay_rate: float = 0.95      # Trail decay factor
    sensitivity: float = 1.0      # Audio sensitivity multiplier
    noise_floor: float = 0.1       # Minimum signal threshold
    num_bands: int = 8            # Number of frequency bands
    min_freq: float = 20.0        # Minimum frequency (Hz)
    max_freq: float = 20000.0     # Maximum frequency (Hz)


@dataclass
class AudioFrame:
    """Audio frame data for visualization."""
    samples: np.ndarray
    sample_rate: int
    timestamp: float


class FrequencyBands:
    """Extract frequency bands from audio for radar visualization."""
    
    def __init__(self, num_bands: int = 8, sample_rate: int = 44100):
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.window_size = 2048
        self._band_ranges = self._calculate_band_ranges()
    
    def _calculate_band_ranges(self) -> List[Tuple[float, float]]:
        """Calculate frequency band ranges (logarithmic scale)."""
        min_freq = 20.0
        max_freq = 20000.0
        bands = []
        
        for i in range(self.num_bands):
            low = min_freq * (max_freq / min_freq) ** (i / self.num_bands)
            high = min_freq * (max_freq / min_freq) ** ((i + 1) / self.num_bands)
            bands.append((low, high))
        
        return bands
    
    def extract_bands(self, audio: np.ndarray) -> np.ndarray:
        """Extract energy levels from each frequency band."""
        if len(audio) < self.window_size:
            # Pad if needed
            audio = np.pad(audio, (0, self.window_size - len(audio)))
        
        # Apply Hann window
        window = np.hanning(self.window_size)
        windowed = audio[:self.window_size] * window
        
        # FFT
        fft = np.fft.rfft(windowed)
        magnitudes = np.abs(fft)
        
        # Calculate frequency bins
        freqs = np.fft.rfftfreq(self.window_size, 1.0 / self.sample_rate)
        
        # Extract band energies
        band_energies = np.zeros(self.num_bands)
        
        for i, (low, high) in enumerate(self._band_ranges):
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_energies[i] = np.mean(magnitudes[mask])
            else:
                band_energies[i] = 0.0
        
        # Normalize to 0-1 range
        max_energy = np.max(band_energies) + 1e-10
        return band_energies / max_energy


class RadarBuffer:
    """Circular buffer for radar display data."""
    
    def __init__(self, resolution: int = 360):
        self.resolution = resolution
        self.buffer = np.zeros(resolution)
        self._head = 0
    
    def update(self, position: int, value: float) -> None:
        """Update a single position in the buffer."""
        if 0 <= position < self.resolution:
            self.buffer[position] = value
    
    def decay(self, factor: float = 0.95) -> None:
        """Apply decay to entire buffer."""
        self.buffer *= factor
    
    def get_current_sweep(self) -> np.ndarray:
        """Get the current sweep data (recent values)."""
        return self.buffer.copy()
    
    def fill_from_bands(self, band_energies: np.ndarray, 
                        base_angle: float, band_width: float = 45.0) -> None:
        """Fill buffer positions based on frequency band energies."""
        num_bands = len(band_energies)
        angles_per_band = band_width / num_bands
        
        for i, energy in enumerate(band_energies):
            angle = int(base_angle + i * angles_per_band) % self.resolution
            self.update(angle, energy)


class RadarDisplay:
    """
    Main radar display system with audio visualization.
    
    Creates a radar-like display that can:
    - Sweep continuously like a radar
    - React to audio frequency bands
    - Display beat detection pulses
    - Show spectrum analysis
    """
    
    def __init__(self, config: Optional[RadarConfig] = None,
                 sample_rate: int = 44100):
        self.config = config or RadarConfig()
        self.sample_rate = sample_rate
        
        # Components
        self.frequency_bands = FrequencyBands(
            num_bands=self.config.num_bands,
            sample_rate=sample_rate
        )
        self.radar_buffer = RadarBuffer(resolution=self.config.resolution)
        
        # State
        self.mode = RadarMode.CLASSIC
        self._current_angle = 0.0
        self._last_update = time.time()
        self._beat_threshold = 0.7
        self._last_beat_time = 0.0
        self._beat_cooldown = 0.1  # seconds
        
        # Callbacks
        self.on_frame_ready: Optional[Callable[[np.ndarray], None]] = None
        
        # Statistics
        self.frames_rendered = 0
        self.beats_detected = 0
    
    def set_mode(self, mode: RadarMode) -> None:
        """Change the radar display mode."""
        self.mode = mode
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio and return radar display data.
        
        Args:
            audio: Audio samples (numpy array)
            
        Returns:
            Radar display data as numpy array
        """
        # Update sweep angle
        self._update_sweep()
        
        # Process based on mode
        if self.mode == RadarMode.AUDIO_REACTIVE:
            display_data = self._process_audio_reactive(audio)
        elif self.mode == RadarMode.SPECTRUM:
            display_data = self._process_spectrum(audio)
        elif self.mode == RadarMode.BEAT_DETECT:
            display_data = self._process_beat_detect(audio)
        else:  # CLASSIC
            display_data = self._process_classic()
        
        self.frames_rendered += 1
        return display_data
    
    def _update_sweep(self) -> None:
        """Update the current sweep angle based on time."""
        current_time = time.time()
        elapsed = current_time - self._last_update
        self._last_update = current_time
        
        # Calculate angle increment based on sweep speed
        angle_increment = elapsed * self.config.sweep_speed * 360.0
        self._current_angle = (self._current_angle + angle_increment) % 360.0
    
    def _process_classic(self) -> np.ndarray:
        """Process in classic radar mode (sweep only)."""
        # Add a point at current sweep position
        position = int(self._current_angle) % self.config.resolution
        
        # Classic radar shows full sweep circle
        sweep_data = np.sin(np.linspace(0, 2 * np.pi, self.config.resolution)) * 0.3 + 0.5
        
        # Add bright spot at current position
        sweep_data[position] = 1.0
        
        # Decay the buffer
        self.radar_buffer.decay(self.config.decay_rate)
        
        return sweep_data
    
    def _process_audio_reactive(self) -> np.ndarray:
        """Process audio-reactive radar mode."""
        # This would be called with actual audio in real-time
        # For now, return the current buffer state
        return self.radar_buffer.get_current_sweep()
    
    def _process_spectrum(self) -> np.ndarray:
        """Process spectrum analysis mode."""
        # Return spectrum as radial display
        spectrum = np.zeros(self.config.resolution)
        
        # Create a simple spectrum visualization
        for i in range(self.config.num_bands):
            angle = int(i * (self.config.resolution / self.config.num_bands))
            # Use buffer values as spectrum
            spectrum[angle] = self.radar_buffer.buffer[angle]
        
        return spectrum
    
    def _process_beat_detect(self) -> np.ndarray:
        """Process beat detection mode."""
        current_time = time.time()
        
        # Check if we're in beat cooldown
        if current_time - self._last_beat_time < self._beat_cooldown:
            # Show pulsing effect
            pulse = np.exp(-((current_time - self._last_beat_time) / 0.1))
            beat_display = np.ones(self.config.resolution) * pulse
            return beat_display
        
        return self.radar_buffer.get_current_sweep()
    
    def detect_beat(self, audio: np.ndarray) -> bool:
        """
        Detect beats in audio using energy comparison.
        
        Args:
            audio: Audio samples
            
        Returns:
            True if a beat is detected
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio ** 2))
        
        # Normalize
        normalized_energy = min(energy / 32768.0, 1.0)
        
        current_time = time.time()
        
        # Check if energy exceeds threshold
        if (normalized_energy > self._beat_threshold and 
            current_time - self._last_beat_time > self._beat_cooldown):
            self._last_beat_time = current_time
            self.beats_detected += 1
            
            # Trigger beat pulse in radar
            position = int(self._current_angle) % self.config.resolution
            self.radar_buffer.update(position, 1.0)
            
            return True
        
        return False
    
    def update_from_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Full update cycle: extract bands, update buffer, return display.
        
        Args:
            audio: Audio samples
            
        Returns:
            Radar display data
        """
        # Extract frequency bands
        band_energies = self.frequency_bands.extract_bands(audio)
        
        # Apply sensitivity
        band_energies = np.clip(
            band_energies * self.config.sensitivity,
            0.0, 1.0
        )
        
        # Update radar buffer with band energies
        current_pos = int(self._current_angle) % self.config.resolution
        
        for i, energy in enumerate(band_energies):
            # Spread bands around current position
            offset = int((i - len(band_energies) / 2) * 5)
            pos = (current_pos + offset) % self.config.resolution
            self.radar_buffer.update(pos, energy)
        
        # Detect beats
        self.detect_beat(audio)
        
        # Decay buffer
        self.radar_buffer.decay(self.config.decay_rate)
        
        # Return current display
        return self.radar_buffer.get_current_sweep()
    
    def get_polar_display(self, data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert radial data to polar coordinate display.
        
        Args:
            data: Radial data (if None, uses current buffer)
            
        Returns:
            2D polar representation
        """
        if data is None:
            data = self.radar_buffer.get_current_sweep()
        
        # Create polar grid
        num_rings = 10
        display = np.zeros((num_rings, self.config.resolution))
        
        for ring in range(num_rings):
            radius = (ring + 1) / num_rings
            for angle in range(self.config.resolution):
                value = data[angle]
                # Apply radial falloff
                display[ring, angle] = value * radius
        
        return display
    
    def get_visualization_matrix(self, width: int = 100, 
                                  height: int = 100) -> np.ndarray:
        """
        Generate a 2D visualization matrix for rendering.
        
        Args:
            width: Output width
            height: Output height
            
        Returns:
            2D numpy array representing the display
        """
        data = self.radar_buffer.get_current_sweep()
        
        # Create coordinate grids
        cx, cy = width / 2, height / 2
        max_radius = min(cx, cy) - 1
        
        display = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                # Calculate distance and angle from center
                dx, dy = x - cx, y - cy
                distance = np.sqrt(dx ** 2 + dy ** 2)
                
                if distance < max_radius:
                    # Normalize distance to 0-1
                    norm_dist = distance / max_radius
                    
                    # Calculate angle (0-360)
                    angle = (np.arctan2(dy, dx) * 180 / np.pi + 360) % 360
                    angle_idx = int(angle) % self.config.resolution
                    
                    # Get value from radar data
                    value = data[angle_idx]
                    
                    # Apply radial profile
                    display[y, x] = value * (1 - norm_dist * 0.5)
        
        return display


class RadarVisualizer:
    """
    High-level visualizer that wraps RadarDisplay with additional features.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.radar = RadarDisplay(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        
        # Color schemes
        self.color_schemes = {
            "radar": {"bg": (0, 0, 0), "sweep": (0, 255, 0), "trail": (0, 128, 0)},
            "neon": {"bg": (0, 0, 0), "sweep": (255, 0, 255), "trail": (128, 0, 128)},
            "ocean": {"bg": (0, 0, 32), "sweep": (0, 255, 255), "trail": (0, 128, 128)},
            "fire": {"bg": (32, 0, 0), "sweep": (255, 128, 0), "trail": (128, 64, 0)},
        }
        self.current_scheme = "radar"
        
        # Animation state
        self.animation_enabled = True
        self.pulse_intensity = 1.0
    
    def set_color_scheme(self, scheme: str) -> None:
        """Set the color scheme."""
        if scheme in self.color_schemes:
            self.current_scheme = scheme
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """Set audio sensitivity (0.1 - 5.0)."""
        self.radar.config.sensitivity = np.clip(sensitivity, 0.1, 5.0)
    
    def set_sweep_speed(self, speed: float) -> None:
        """Set sweep speed (rotations per second)."""
        self.radar.config.sweep_speed = np.clip(speed, 0.1, 10.0)
    
    def visualize(self, audio_chunk: np.ndarray) -> dict:
        """
        Generate complete visualization data from audio chunk.
        
        Args:
            audio_chunk: Audio samples
            
        Returns:
            Dictionary with various visualization data
        """
        # Process audio through radar
        radar_data = self.radar.update_from_audio(audio_chunk)
        
        # Get additional visualizations
        polar_display = self.radar.get_polar_display()
        matrix = self.radar.get_visualization_matrix()
        
        # Detect beat
        beat_detected = self.radar.detect_beat(audio_chunk)
        
        return {
            "radar": radar_data,
            "polar": polar_display,
            "matrix": matrix,
            "beat": beat_detected,
            "angle": self.radar._current_angle,
            "scheme": self.current_scheme,
            "intensity": self.pulse_intensity,
        }
    
    def get_frame(self) -> np.ndarray:
        """Get current visualization frame."""
        return self.radar.get_visualization_matrix()
    
    def reset(self) -> None:
        """Reset the visualizer state."""
        self.radar = RadarDisplay(sample_rate=self.sample_rate)
        self.pulse_intensity = 1.0


# Demo/Testing
if __name__ == "__main__":
    # Create visualizer
    viz = RadarVisualizer()
    
    # Generate test audio (white noise with varying energy)
    sample_rate = 44100
    duration = 5  # seconds
    samples = int(sample_rate * duration)
    
    # Simulate audio with some structure
    t = np.linspace(0, duration, samples)
    
    # Base tone
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    
    # Add varying energy (simulating beats)
    for i in range(10):
        beat_time = 0.5 + i * 0.5
        beat_start = int(beat_time * sample_rate)
        beat_end = int((beat_time + 0.1) * sample_rate)
        if beat_end < samples:
            audio[beat_start:beat_end] += np.sin(2 * np.pi * 100 * t[beat_start:beat_end])
    
    # Add some random noise
    audio += np.random.randn(samples) * 0.1
    
    # Normalize
    audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    
    # Process in chunks
    chunk_size = 2048
    num_chunks = len(audio) // chunk_size
    
    print("Radar Display System Test")
    print("=" * 40)
    
    for i in range(num_chunks):
        chunk = audio[i * chunk_size:(i + 1) * chunk_size].astype(np.float32)
        
        # Visualize
        result = viz.visualize(chunk)
        
        if result["beat"]:
            print(f"Beat detected at chunk {i}!")
        
        # Print some stats every 50 chunks
        if i % 50 == 0:
            print(f"Chunk {i}: angle={result['angle']:.1f}, "
                  f"max={result['radar'].max():.3f}")
    
    print(f"\nTotal frames: {viz.radar.frames_rendered}")
    print(f"Total beats: {viz.radar.beats_detected}")
    print("\nTest complete!")
