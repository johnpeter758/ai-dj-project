"""
Audio Visualization System
Real-time audio visualization for AI DJ project.
Provides spectrum analyzers, waveform displays, beat visualizations, and circular patterns.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

# Audio libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class VisualizationStyle(Enum):
    """Visualization style presets."""
    BARS = "bars"
    WAVE = "wave"
    CIRCULAR = "circular"
    RADIAL = "radial"
    MIRROR = "mirror"
    PARTICLE = "particle"
    SPECTROGRAM = "spectrogram"


class ColorScheme(Enum):
    """Color scheme presets."""
    FIRE = "fire"
    OCEAN = "ocean"
    NEON = "neon"
    RAINBOW = "rainbow"
    MONOCHROME = "monochrome"
    PLASMA = "plasma"
    VIRIDIS = "viridis"


@dataclass
class VisualizationConfig:
    """Configuration for audio visualization."""
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_bands: int = 64
    smoothing: float = 0.8
    min_db: float = -80
    max_db: float = 0
    style: VisualizationStyle = VisualizationStyle.BARS
    color_scheme: ColorScheme = ColorScheme.FIRE
    fps: int = 30
    mirror: bool = True
    gain: float = 1.0


# =============================================================================
# COLOR UTILITIES
# =============================================================================

class ColorPalette:
    """Color palette generation for visualizations."""
    
    @staticmethod
    def fire(n_colors: int) -> np.ndarray:
        """Generate fire color palette (reds, oranges, yellows)."""
        colors = np.zeros((n_colors, 3))
        for i in range(n_colors):
            t = i / max(n_colors - 1, 1)
            if t < 0.33:
                colors[i] = [1.0, t * 3, 0.0]
            elif t < 0.66:
                colors[i] = [1.0 - (t - 0.33) * 3, 1.0, 0.0]
            else:
                colors[i] = [1.0, 1.0, (t - 0.66) * 3]
        return colors
    
    @staticmethod
    def ocean(n_colors: int) -> np.ndarray:
        """Generate ocean color palette (blues, cyans)."""
        colors = np.zeros((n_colors, 3))
        for i in range(n_colors):
            t = i / max(n_colors - 1, 1)
            colors[i] = [0.0, t * 0.7 + 0.3, t * 0.5 + 0.5]
        return colors
    
    @staticmethod
    def neon(n_colors: int) -> np.ndarray:
        """Generate neon color palette (bright pinks, cyans)."""
        colors = np.zeros((n_colors, 3))
        for i in range(n_colors):
            t = i / max(n_colors - 1, 1)
            colors[i] = [
                np.sin(t * np.pi * 2) * 0.5 + 0.5,
                np.sin(t * np.pi * 2 + np.pi / 3) * 0.5 + 0.5,
                np.sin(t * np.pi * 2 + 2 * np.pi / 3) * 0.5 + 0.5
            ]
        return colors
    
    @staticmethod
    def rainbow(n_colors: int) -> np.ndarray:
        """Generate rainbow color palette."""
        colors = np.zeros((n_colors, 3))
        for i in range(n_colors):
            t = i / max(n_colors - 1, 1)
            hue = t
            # HSV to RGB conversion
            h = hue * 6
            x = 1 - abs(h % 2 - 1)
            if h < 1:
                colors[i] = [1, x, 0]
            elif h < 2:
                colors[i] = [x, 1, 0]
            elif h < 3:
                colors[i] = [0, 1, x]
            elif h < 4:
                colors[i] = [0, x, 1]
            elif h < 5:
                colors[i] = [x, 0, 1]
            else:
                colors[i] = [1, 0, x]
        return colors
    
    @staticmethod
    def plasma(n_colors: int) -> np.ndarray:
        """Generate plasma color palette (purple to yellow)."""
        colors = np.zeros((n_colors, 3))
        for i in range(n_colors):
            t = i / max(n_colors - 1, 1)
            colors[i] = [
                0.5 * np.sin(t * np.pi * 2) + 0.5 + 0.5 * np.cos(t * np.pi),
                0.5 * np.sin(t * np.pi * 2 + np.pi) + 0.5,
                1.0 - t
            ]
        return colors
    
    @staticmethod
    def get_palette(name: ColorScheme, n_colors: int) -> np.ndarray:
        """Get color palette by name."""
        palettes = {
            ColorScheme.FIRE: ColorPalette.fire,
            ColorScheme.OCEAN: ColorPalette.ocean,
            ColorScheme.NEON: ColorPalette.neon,
            ColorScheme.RAINBOW: ColorPalette.rainbow,
            ColorScheme.PLASMA: ColorPalette.plasma,
            ColorScheme.MONOCHROME: lambda n: np.ones((n, 3)) * np.linspace(0.2, 1, n)[:, np.newaxis],
        }
        return palettes.get(name, ColorPalette.fire)(n_colors)


# =============================================================================
# FREQUENCY BAND EXTRACTOR
# =============================================================================

class FrequencyBands:
    """Extract frequency band energies from audio."""
    
    # Standard frequency bands in Hz
    BANDS = {
        'sub_bass': (20, 60),
        'bass': (60, 250),
        'low_mid': (250, 500),
        'mid': (500, 2000),
        'upper_mid': (2000, 4000),
        'presence': (4000, 6000),
        'brilliance': (6000, 20000)
    }
    
    @staticmethod
    def get_band_indices(freqs: np.ndarray, low_freq: float, high_freq: float) -> Tuple[int, int]:
        """Get array indices for frequency range."""
        low_idx = np.searchsorted(freqs, low_freq)
        high_idx = np.searchsorted(freqs, high_freq)
        return max(0, low_idx), min(len(freqs), high_idx)
    
    @staticmethod
    def extract_bands(
        magnitudes: np.ndarray,
        freq_bins: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract energy from each frequency band.
        
        Args:
            magnitudes: Magnitude spectrogram (freq_bins x frames)
            freq_bins: Frequency values for each bin
            
        Returns:
            Dictionary of band name to energy value
        """
        energies = {}
        for band_name, (low, high) in FrequencyBands.BANDS.items():
            low_idx, high_idx = FrequencyBands.get_band_indices(freq_bins, low, high)
            if high_idx > low_idx:
                energies[band_name] = np.mean(magnitudes[low_idx:high_idx])
            else:
                energies[band_name] = 0.0
        return energies


# =============================================================================
# SPECTRUM ANALYZER
# =============================================================================

class SpectrumAnalyzer:
    """Real-time spectrum analysis for visualization."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize spectrum analyzer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.sample_rate = self.config.sample_rate
        self.n_fft = self.config.n_fft
        self.hop_length = self.config.hop_length
        self.n_bands = self.config.n_bands
        
        # Smoothing buffer
        self.smooth_buffer = np.zeros(n_bands)
        self.smoothing = self.config.smoothing
        
        # Pre-compute frequency bins
        self.freq_bins = librosa.fft_frequencies(
            sr=self.sample_rate,
            n_fft=self.n_fft
        ) if LIBROSA_AVAILABLE else np.fft.fftfreq(self.n_fft, 1/self.sample_rate)
        
        # Window function
        self.window = np.hanning(self.n_fft)
        
        # Color palette
        self.colors = ColorPalette.get_palette(
            self.config.color_scheme,
            self.n_bands
        )
    
    def compute_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute magnitude spectrum from audio frame.
        
        Args:
            audio: Audio samples (mono)
            
        Returns:
            Magnitude spectrum in dB
        """
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        elif len(audio) > self.n_fft:
            audio = audio[:self.n_fft]
        
        # Apply window
        windowed = audio * self.window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # Clip to range
        magnitude_db = np.clip(
            magnitude_db,
            self.config.min_db,
            self.config.max_db
        )
        
        # Normalize to 0-1
        normalized = (magnitude_db - self.config.min_db) / (
            self.config.max_db - self.config.min_db
        )
        
        return normalized
    
    def compute_band_spectrum(
        self,
        audio: np.ndarray,
        n_bands: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute energy in logarithmically spaced frequency bands.
        
        Args:
            audio: Audio samples
            n_bands: Number of bands (uses config default if None)
            
        Returns:
            Array of band energies
        """
        n_bands = n_bands or self.n_bands
        
        # Compute full spectrum
        spectrum = self.compute_spectrum(audio)
        
        # Create logarithmic frequency band boundaries
        min_freq = self.freq_bins[1] if len(self.freq_bins) > 1 else 20
        max_freq = self.freq_bins[-1] if len(self.freq_bins) > 1 else 20000
        
        log_freqs = np.logspace(
            np.log10(min_freq),
            np.log10(max_freq),
            n_bands + 1
        )
        
        # Compute energy in each band
        bands = np.zeros(n_bands)
        for i in range(n_bands):
            low_idx = np.searchsorted(self.freq_bins, log_freqs[i])
            high_idx = np.searchsorted(self.freq_bins, log_freqs[i + 1])
            if high_idx > low_idx:
                bands[i] = np.mean(spectrum[low_idx:high_idx])
        
        # Apply smoothing
        self.smooth_buffer = (
            self.smoothing * self.smooth_buffer +
            (1 - self.smoothing) * bands
        )
        
        return self.smooth_buffer
    
    def get_band_energies(self, audio: np.ndarray) -> Dict[str, float]:
        """Get energy for standard frequency bands."""
        spectrum = self.compute_spectrum(audio)
        return FrequencyBands.extract_bands(
            spectrum[np.newaxis, :],
            self.freq_bins
        )


# =============================================================================
# WAVEFORM RENDERER
# =============================================================================

class WaveformRenderer:
    """Renders waveform visualizations from audio."""
    
    def __init__(
        self,
        width: int = 800,
        height: int = 200,
        sample_rate: int = 22050,
        color: Tuple[int, int, int] = (0, 255, 128)
    ):
        """
        Initialize waveform renderer.
        
        Args:
            width: Display width in pixels
            height: Display height in pixels
            sample_rate: Audio sample rate
            color: RGB color tuple
        """
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.color = np.array(color) / 255.0
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file for visualization."""
        if LIBROSA_AVAILABLE:
            return librosa.load(file_path, sr=self.sample_rate, mono=True)[0]
        elif SOUNDFILE_AVAILABLE:
            audio, _ = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return audio
        else:
            raise ImportError("No audio library available")
    
    def downsample(self, audio: np.ndarray, target_points: int) -> np.ndarray:
        """
        Downsample audio to target number of points.
        
        Args:
            audio: Audio samples
            target_points: Number of output points
            
        Returns:
            Downsampled audio
        """
        samples_per_point = len(audio) // target_points
        if samples_per_point == 0:
            return audio
        
        # Reshape and compute RMS for each segment
        n_segments = len(audio) // samples_per_point
        truncated = audio[:n_segments * samples_per_point]
        reshaped = truncated.reshape(n_segments, samples_per_point)
        
        # Use max absolute value for visualization
        downsampled = np.max(np.abs(reshaped), axis=1)
        
        # Pad if needed
        if len(downsampled) < target_points:
            downsampled = np.pad(
                downsampled,
                (0, target_points - len(downsampled))
            )
        
        return downsampled
    
    def render(
        self,
        audio: np.ndarray,
        mirror: bool = True
    ) -> np.ndarray:
        """
        Render waveform as image array.
        
        Args:
            audio: Audio samples
            mirror: Whether to mirror waveform
            
        Returns:
            Waveform image as numpy array (height x width x 3)
        """
        # Downsample to display resolution
        waveform = self.downsample(audio, self.width)
        
        # Normalize
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)
        
        # Create output image
        image = np.zeros((self.height, self.width, 3))
        
        # Center line
        center_y = self.height // 2
        amplitude = (self.height // 2) - 2
        
        # Draw waveform
        for x in range(len(waveform)):
            y = int(center_y - waveform[x] * amplitude)
            y_mirror = int(center_y + waveform[x] * amplitude)
            
            if 0 <= y < self.height:
                image[y, x] = self.color
            
            if mirror and 0 <= y_mirror < self.height:
                image[y_mirror, x] = self.color
        
        return (image * 255).astype(np.uint8)
    
    def render_stereo(
        self,
        audio_left: np.ndarray,
        audio_right: np.ndarray
    ) -> np.ndarray:
        """Render stereo waveform with separate channels."""
        waveform_left = self.downsample(audio_left, self.width)
        waveform_right = self.downsample(audio_right, self.width)
        
        # Normalize both
        max_val = max(
            np.max(np.abs(waveform_left)),
            np.max(np.abs(waveform_right))
        ) + 1e-10
        
        waveform_left /= max_val
        waveform_right /= max_val
        
        image = np.zeros((self.height, self.width, 3))
        center_y = self.height // 2
        amplitude = (self.height // 2) - 2
        
        for x in range(len(waveform_left)):
            # Left channel (top)
            y_l = int(center_y // 2 - waveform_left[x] * (center_y // 2 - 2))
            if 0 <= y_l < center_y:
                image[y_l, x] = [1, 0.3, 0.3]  # Red-ish
            
            # Right channel (bottom)
            y_r = int(center_y + center_y // 2 - waveform_right[x] * (center_y // 2 - 2))
            if center_y <= y_r < self.height:
                image[y_r, x] = [0.3, 0.3, 1]  # Blue-ish
        
        return (image * 255).astype(np.uint8)


# =============================================================================
# CIRCULAR VISUALIZER
# =============================================================================

class CircularVisualizer:
    """Creates circular/radial audio visualizations."""
    
    def __init__(
        self,
        size: int = 400,
        n_bands: int = 64,
        color_scheme: ColorScheme = ColorScheme.PLASMA,
        smoothing: float = 0.7
    ):
        """
        Initialize circular visualizer.
        
        Args:
            size: Canvas size (square)
            n_bands: Number of frequency bands
            color_scheme: Color scheme
            smoothing: Smoothing factor for animation
        """
        self.size = size
        self.n_bands = n_bands
        self.smoothing = smoothing
        
        self.colors = ColorPalette.get_palette(color_scheme, n_bands)
        
        # Animation state
        self.prev_bands = np.zeros(n_bands)
        self.rotation = 0.0
    
    def render(
        self,
        bands: np.ndarray,
        rotation_speed: float = 0.5,
        glow: bool = True
    ) -> np.ndarray:
        """
        Render circular visualization.
        
        Args:
            bands: Frequency band energies (0-1 normalized)
            rotation_speed: Rotation speed multiplier
            glow: Whether to add glow effect
            
        Returns:
            Image array
        """
        # Smooth bands
        bands = self.smoothing * self.prev_bands + (1 - self.smoothing) * bands
        self.prev_bands = bands.copy()
        
        # Update rotation
        self.rotation += rotation_speed * np.mean(bands) + 0.01
        
        # Create canvas
        center = self.size // 2
        max_radius = self.size // 2 - 10
        min_radius = max_radius * 0.2
        
        image = np.zeros((self.size, self.size, 3))
        
        # Draw each frequency band
        angle_step = 2 * np.pi / self.n_bands
        
        for i, energy in enumerate(bands):
            angle = i * angle_step + self.rotation
            
            # Outer radius based on energy
            radius = min_radius + energy * (max_radius - min_radius)
            
            # Calculate position
            x = int(center + radius * np.cos(angle))
            y = int(center + radius * np.sin(angle))
            
            if 0 <= x < self.size and 0 <= y < self.size:
                # Draw point with color
                color = self.colors[i]
                
                # Add glow
                if glow:
                    glow_radius = int(energy * 15 + 3)
                    for dx in range(-glow_radius, glow_radius + 1):
                        for dy in range(-glow_radius, glow_radius + 1):
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= glow_radius:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.size and 0 <= ny < self.size:
                                    alpha = 1 - dist / glow_radius
                                    alpha *= energy * 0.5
                                    image[ny, nx] += color * alpha
                else:
                    image[y, x] = color
        
        # Draw center circle
        center_radius = int(min_radius * 0.8)
        for angle in np.linspace(0, 2 * np.pi, 100):
            x = int(center + center_radius * np.cos(angle))
            y = int(center + center_radius * np.sin(angle))
            if 0 <= x < self.size and 0 <= y < self.size:
                avg_energy = np.mean(bands)
                image[y, x] = [avg_energy, avg_energy, avg_energy]
        
        return np.clip(image, 0, 1)


# =============================================================================
# BEAT VISUALIZER
# =============================================================================

class BeatVisualizer:
    """Visualizes beat and rhythm patterns."""
    
    def __init__(
        self,
        width: int = 400,
        height: int = 100,
        sample_rate: int = 22050,
        hop_length: int = 512
    ):
        """
        Initialize beat visualizer.
        
        Args:
            width: Display width
            height: Display height
            sample_rate: Audio sample rate
            hop_length: Hop length for analysis
        """
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Energy history for beat detection
        self.energy_history = []
        self.history_size = 100
        
        # Beat state
        self.is_beat = False
        self.beat_decay = 0.0
    
    def compute_energy(self, audio: np.ndarray) -> float:
        """Compute RMS energy of audio frame."""
        return np.sqrt(np.mean(audio ** 2))
    
    def detect_beat(self, audio: np.ndarray, threshold: float = 1.5) -> bool:
        """
        Detect if current frame is a beat.
        
        Args:
            audio: Audio frame
            threshold: Beat detection threshold
            
        Returns:
            True if beat detected
        """
        energy = self.compute_energy(audio)
        
        # Update history
        self.energy_history.append(energy)
        if len(self.energy_history) > self.history_size:
            self.energy_history.pop(0)
        
        if len(self.energy_history) < 10:
            return False
        
        # Compare to local average
        local_avg = np.mean(self.energy_history[:-1])
        if local_avg > 0:
            ratio = energy / local_avg
            is_beat = ratio > threshold
            
            if is_beat:
                self.beat_decay = 1.0
            
            return is_beat
        
        return False
    
    def render(
        self,
        audio: np.ndarray,
        show_beats: bool = True
    ) -> np.ndarray:
        """
        Render beat visualization.
        
        Args:
            audio: Audio samples
            show_beats: Whether to show beat markers
            
        Returns:
            Image array
        """
        # Detect beat
        is_beat = self.detect_beat(audio)
        
        # Update decay
        self.beat_decay *= 0.9
        
        # Compute waveform for display
        samples_per_pixel = len(audio) // self.width
        if samples_per_pixel > 0:
            waveform = np.array([
                np.max(np.abs(audio[i*samples_per_pixel:(i+1)*samples_per_pixel]))
                for i in range(self.width)
            ])
        else:
            waveform = np.abs(audio[:self.width])
        
        waveform = waveform / (np.max(waveform) + 1e-10)
        
        # Create image
        image = np.zeros((self.height, self.width, 3))
        center_y = self.height // 2
        
        # Draw waveform
        for x in range(len(waveform)):
            y = int(center_y - waveform[x] * (center_y - 5))
            if 0 <= y < self.height:
                color = [0.2, 0.8, 0.4] if not is_beat else [1.0, 0.3, 0.3]
                image[y, x] = color
                if is_beat and show_beats:
                    # Draw beat pulse
                    for dy in range(-3, 4):
                        ny = y + dy
                        if 0 <= ny < self.height:
                            alpha = 1 - abs(dy) / 4
                            image[ny, x] = color * alpha + image[ny, x] * (1 - alpha)
        
        return np.clip(image, 0, 1)


# =============================================================================
# SPECTROGRAM RENDERER
# =============================================================================

class SpectrogramRenderer:
    """Renders spectrogram visualizations."""
    
    def __init__(
        self,
        width: int = 800,
        height: int = 400,
        n_fft: int = 2048,
        hop_length: int = 512,
        color_scheme: ColorScheme = ColorScheme.PLASMA
    ):
        """
        Initialize spectrogram renderer.
        
        Args:
            width: Display width
            height: Display height
            n_fft: FFT size
            hop_length: Hop length
            color_scheme: Color scheme
        """
        self.width = width
        self.height = height
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Color lookup table
        n_colors = 256
        self.color_lut = ColorPalette.get_palette(color_scheme, n_colors)
    
    def compute_spectrogram(
        self,
        audio: np.ndarray
    ) -> np.ndarray:
        """Compute spectrogram from audio."""
        if not LIBROSA_AVAILABLE:
            # Manual STFT
            n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
            window = np.hanning(self.n_fft)
            spectrogram = np.zeros((self.n_fft // 2 + 1, n_frames))
            
            for i in range(n_frames):
                start = i * self.hop_length
                frame = audio[start:start + self.n_fft]
                if len(frame) == self.n_fft:
                    frame = frame * window
                    spectrogram[:, i] = np.abs(np.fft.rfft(frame))
            
            return spectrogram
        else:
            return np.abs(librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            ))
    
    def render(
        self,
        audio: np.ndarray,
        db_range: Tuple[float, float] = (-80, 0)
    ) -> np.ndarray:
        """
        Render spectrogram.
        
        Args:
            audio: Audio samples
            db_range: dB range for display
            
        Returns:
            Spectrogram image
        """
        # Compute spectrogram
        spec = self.compute_spectrogram(audio)
        
        # Convert to dB
        spec_db = 20 * np.log10(spec + 1e-10)
        spec_db = np.clip(spec_db, db_range[0], db_range[1])
        
        # Normalize to 0-1
        spec_norm = (spec_db - db_range[0]) / (db_range[1] - db_range[0])
        
        # Resize to display dimensions
        # Time axis
        if spec_norm.shape[1] > self.width:
            # Downsample time
            indices = np.linspace(0, spec_norm.shape[1] - 1, self.width).astype(int)
            spec_norm = spec_norm[:, indices]
        elif spec_norm.shape[1] < self.width:
            # Pad
            spec_norm = np.pad(
                spec_norm,
                ((0, 0), (0, self.width - spec_norm.shape[1]))
            )
        
        # Frequency axis
        if spec_norm.shape[0] > self.height:
            indices = np.linspace(0, spec_norm.shape[0] - 1, self.height).astype(int)
            spec_norm = spec_norm[indices, :]
        elif spec_norm.shape[0] < self.height:
            spec_norm = np.pad(
                spec_norm,
                ((0, self.height - spec_norm.shape[0]), (0, 0))
            )
        
        # Flip vertically (low freq at bottom)
        spec_norm = np.flipud(spec_norm)
        
        # Apply color mapping
        image = np.zeros((self.height, self.width, 3))
        indices = (spec_norm * 255).astype(int)
        indices = np.clip(indices, 0, 255)
        
        for i in range(3):
            image[:, :, i] = self.color_lut[indices, i]
        
        return image


# =============================================================================
# MAIN VISUALIZATION CONTROLLER
# =============================================================================

class AudioVisualizer:
    """
    Main audio visualization controller.
    Coordinates all visualization components.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize audio visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Initialize components
        self.spectrum_analyzer = SpectrumAnalyzer(self.config)
        
        self.waveform_renderer = WaveformRenderer(
            width=800,
            height=200,
            sample_rate=self.config.sample_rate
        )
        
        self.circular_visualizer = CircularVisualizer(
            size=400,
            n_bands=self.config.n_bands,
            color_scheme=self.config.color_scheme,
            smoothing=self.config.smoothing
        )
        
        self.beat_visualizer = BeatVisualizer(
            width=400,
            height=100,
            sample_rate=self.config.sample_rate,
            hop_length=self.config.hop_length
        )
        
        self.spectrogram_renderer = SpectrogramRenderer(
            width=800,
            height=400,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            color_scheme=self.config.color_scheme
        )
        
        # State
        self.is_playing = False
        self.current_frame = 0
    
    def process_frame(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process a single audio frame and generate all visualizations.
        
        Args:
            audio: Audio samples for current frame
            
        Returns:
            Dictionary of visualization outputs
        """
        results = {}
        
        # Spectrum bars
        results['spectrum'] = self.spectrum_analyzer.compute_band_spectrum(audio)
        
        # Band energies
        results['bands'] = self.spectrum_analyzer.get_band_energies(audio)
        
        # Circular visualization
        results['circular'] = self.circular_visualizer.render(results['spectrum'])
        
        # Beat detection
        results['beat'] = self.beat_visualizer.detect_beat(audio)
        results['beat_decay'] = self.beat_visualizer.beat_decay
        
        return results
    
    def generate_waveform(self, audio: np.ndarray) -> np.ndarray:
        """Generate waveform visualization."""
        return self.waveform_renderer.render(audio, mirror=self.config.mirror)
    
    def generate_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Generate spectrogram visualization."""
        return self.spectrogram_renderer.render(audio)
    
    def generate_all(
        self,
        audio: np.ndarray,
        frame_duration: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Generate all visualizations from audio file.
        
        Args:
            audio: Full audio array
            frame_duration: Duration per frame in seconds
            
        Returns:
            Dictionary of all visualizations
        """
        samples_per_frame = int(self.config.sample_rate * frame_duration)
        n_frames = len(audio) // samples_per_frame
        
        visualizations = {
            'frames': [],
            'waveform': self.generate_waveform(audio),
            'spectrogram': self.generate_spectrogram(audio)
        }
        
        for i in range(n_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            frame = audio[start:end]
            
            frame_viz = self.process_frame(frame)
            visualizations['frames'].append(frame_viz)
        
        return visualizations


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def visualize_audio_file(
    file_path: str,
    output_dir: Optional[str] = None,
    config: Optional[VisualizationConfig] = None
) -> Dict[str, str]:
    """
    Generate visualizations for an audio file.
    
    Args:
        file_path: Path to audio file
        output_dir: Output directory (uses temp if None)
        config: Visualization configuration
        
    Returns:
        Dictionary of output paths
    """
    import tempfile
    from pathlib import Path
    
    # Initialize visualizer
    visualizer = AudioVisualizer(config)
    
    # Load audio
    if LIBROSA_AVAILABLE:
        audio, sr = librosa.load(file_path, sr=visualizer.config.sample_rate)
    else:
        raise ImportError("librosa required for audio loading")
    
    # Generate visualizations
    visualizations = visualizer.generate_all(audio)
    
    # Save outputs
    output_paths = {}
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(tempfile.gettempdir()) / "audio_viz"
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Save spectrogram
    spec_path = output_path / "spectrogram.png"
    # Note: Would need PIL to save, returning data for now
    output_paths['spectrogram'] = str(spec_path)
    
    # Save waveform path
    output_paths['waveform'] = str(output_path / "waveform.png")
    
    return output_paths


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    config = VisualizationConfig(
        sample_rate=22050,
        n_fft=2048,
        n_bands=64,
        style=VisualizationStyle.BARS,
        color_scheme=ColorScheme.PLASMA,
        smoothing=0.8
    )
    
    visualizer = AudioVisualizer(config)
    
    print("AudioVisualizer initialized successfully!")
    print(f"  - Sample rate: {config.sample_rate} Hz")
    print(f"  - FFT size: {config.n_fft}")
    print(f"  - Bands: {config.n_bands}")
    print(f"  - Color scheme: {config.color_scheme.value}")
