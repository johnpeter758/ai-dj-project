"""
Spatial Audio System for AR/VR
Binaural, Ambisonics, and 360° audio processing for immersive experiences.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy import signal
from scipy.interpolate import interp1d
import math


class SpatialAudioEngine:
    """
    Core spatial audio processor supporting multiple spatialization techniques.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.hrtf_database = None
        self.ambisonic_decoder = None
        self._init_hrtf_database()
    
    def _init_hrtf_database(self):
        """Initialize default HRTF database for binaural processing."""
        # Default HRTF angles (azimuth in degrees)
        self.hrtf_angles = list(range(-90, 91, 5))
        self.hrtf_database = self._generate_default_hrtf()
    
    def _generate_default_hrtf(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Generate default HRTF filters for common angles."""
        hrtf_db = {}
        
        for angle in self.hrtf_angles:
            # Generate simplified HRTF using modified JIRC model
            freq = np.fft.rfftfreq(2048, 1.0 / self.sample_rate)
            
            # Frequency-dependent interaural level difference
            illd = 20 * np.log10(1 + 0.3 * np.abs(np.sin(np.radians(angle))))
            
            # Frequency-dependent interaural time difference
            itd = 0.00003 * angle  # Simplified ITD
            
            # Create magnitude response
            mag_l = np.ones_like(freq)
            mag_r = np.ones_like(freq) * 10 ** (-illd / 20)
            
            # Add pinna cues (high-frequency elevation cues)
            elevation_factor = np.sin(2 * np.pi * freq * 0.001) * 0.1
            
            # Frequency shaping for spatialization
            # Low frequencies need less filtering
            low_freq_mask = freq < 500
            mag_l[low_freq_mask] = 1.0
            mag_r[low_freq_mask] = 1.0
            
            # Mid and high frequencies get more spatial processing
            mid_freq_mask = (freq >= 500) & (freq < 4000)
            mag_l[mid_freq_mask] *= 1 + elevation_factor[mid_freq_mask]
            mag_r[mid_freq_mask] *= 1 - elevation_factor[mid_freq_mask]
            
            high_freq_mask = freq >= 4000
            mag_l[high_freq_mask] *= 1.2
            mag_r[high_freq_mask] *= 0.9
            
            # Convert to time domain using minimum phase
            hrtf_l = self._minimum_phase_ir(mag_l, 128)
            hrtf_r = self._minimum_phase_ir(mag_r, 128)
            
            hrtf_db[angle] = (hrtf_l, hrtf_r)
        
        return hrtf_db
    
    def _minimum_phase_ir(self, magnitude: np.ndarray, 
                          length: int) -> np.ndarray:
        """Convert magnitude response to minimum phase impulse response."""
        # Log magnitude forcepstrum
        log_mag = np.log(magnitude + 1e-10)
        n = len(log_mag)
        
        # Compute cepstrum
        cepstrum = np.fft.irfft(log_mag)
        
        # Apply Hilbert window (causal part only)
        window = np.ones(n)
        window[:n//2] = 2.0
        window[0] = 1.0
        cepstrum *= window
        
        # Reconstruct minimum phase
        min_phase_mag = np.exp(np.fft.rfft(cepstrum))
        ir = np.fft.irfft(min_phase_mag)
        
        return ir[:length]
    
    def _interpolate_hrtf(self, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate HRTF for arbitrary angle."""
        angle = np.clip(angle, -90, 90)
        
        if angle in self.hrtf_database:
            return self.hrtf_database[int(angle)]
        
        # Find surrounding angles
        angles = sorted(self.hrtf_database.keys())
        lower = max([a for a in angles if a <= angle], default=min(angles))
        upper = min([a for a in angles if a >= angle], default=max(angles))
        
        if lower == upper:
            return self.hrtf_database[lower]
        
        # Linear interpolation
        weight = (angle - lower) / (upper - lower)
        
        h_l1, h_r1 = self.hrtf_database[lower]
        h_l2, h_r2 = self.hrtf_database[upper]
        
        h_l = h_l1 * (1 - weight) + h_l2 * weight
        h_r = h_r1 * (1 - weight) + h_r2 * weight
        
        return h_l, h_r


class BinauralRenderer(SpatialAudioEngine):
    """
    Binaural audio renderer for headphone-based 3D audio.
    Uses HRTF (Head-Related Transfer Functions) for accurate spatialization.
    """
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__(sample_rate)
        self.head_radius = 0.087  # Average head radius in meters
        self.use_crossfeed = True  # Enable crossfeed for speaker emulation
    
    def render_binaural(self, audio: np.ndarray, 
                        azimuth: float, 
                        elevation: float = 0,
                        distance: float = 1.0) -> np.ndarray:
        """
        Render mono audio to binaural (stereo) using HRTF.
        
        Args:
            audio: Input mono audio signal
            azimuth: Horizontal angle in degrees (-180 to 180)
            elevation: Vertical angle in degrees (-90 to 90)
            distance: Distance from listener (affects volume and filtering)
        
        Returns:
            Stereo audio (2 channels)
        """
        # Handle stereo input by mixing to mono first
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Apply distance attenuation
        distance_gain = 1.0 / (1.0 + 0.5 * (distance - 1.0))
        audio = audio * distance_gain
        
        # Get HRTF for this position
        h_l, h_r = self._interpolate_hrtf(azimuth)
        
        # Apply HRTF via convolution
        left = signal.fftconvolve(audio, h_l, mode='full')
        right = signal.fftconvolve(audio, h_r, mode='full')
        
        # Stack to stereo
        stereo = np.stack([left, right], axis=-1)
        
        # Trim to original length
        stereo = stereo[:len(audio)]
        
        # Apply head shadowing for extreme azimuths
        if abs(azimuth) > 45:
            shadow = 1.0 - 0.3 * (abs(azimuth) - 45) / 45
            if azimuth > 0:
                stereo[:, 0] *= shadow
            else:
                stereo[:, 1] *= shadow
        
        return self._normalize(stereo)
    
    def render_multiple_sources(self, sources: List[Tuple[np.ndarray, float, float, float]]
                                ) -> np.ndarray:
        """
        Render multiple audio sources at different positions.
        
        Args:
            sources: List of (audio, azimuth, elevation, distance) tuples
        
        Returns:
            Mixed binaural stereo output
        """
        outputs = []
        
        for audio, az, el, dist in sources:
            binaural = self.render_binaural(audio, az, el, dist)
            outputs.append(binaural)
        
        # Mix all sources
        if outputs:
            return np.sum(outputs, axis=0)
        return np.zeros((1, 2))
    
    def apply_head_tracking(self, audio: np.ndarray,
                           reference_angle: float,
                           current_angle: float) -> np.ndarray:
        """
        Apply head tracking transformation to binaural audio.
        
        Args:
            audio: Original binaural audio
            reference_angle: Angle when audio was rendered
            current_angle: Current head orientation
        
        Returns:
            Re-rendered binaural audio for new orientation
        """
        angle_diff = current_angle - reference_angle
        
        # Rotate the sound stage
        rotated = self._rotate_binaural(audio, angle_diff)
        return rotated
    
    def _rotate_binaural(self, audio: np.ndarray, angle: float) -> np.ndarray:
        """Rotate binaural audio by specified angle."""
        if abs(angle) < 1:
            return audio
        
        # Use all-pass filter for rotation (simplified)
        # For production, use proper Ambisonic rotation
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        
        rotated = audio.copy()
        
        # Simple rotation approximation using HRTF
        h_l, h_r = self._interpolate_hrtf(angle)
        
        # Crossfade to rotated position
        fade_len = min(1024, len(audio) // 4)
        fade = np.linspace(0, 1, fade_len)
        
        left = signal.fftconvolve(audio[:, 0], h_l, mode='same')
        right = signal.fftconvolve(audio[:, 1], h_r, mode='same')
        
        rotated[:, 0] = audio[:, 0] * (1 - fade[-1]) + left * fade[-1]
        rotated[:, 1] = audio[:, 1] * (1 - fade[-1]) + right * fade[-1]
        
        return rotated
    
    def enable_crossfeed(self, enable: bool = True):
        """Enable/disable crossfeed for natural speaker listening."""
        self.use_crossfeed = enable
    
    def apply_crossfeed(self, audio: np.ndarray) -> np.ndarray:
        """Apply headphone crossfeed for speaker emulation."""
        if not self.use_crossfeed or audio.ndim < 2:
            return audio
        
        # Simple crossfeed: mix small amount of opposite channel
        crossfeed_level = 0.3
        output = audio.copy()
        output[:, 0] = audio[:, 0] + audio[:, 1] * crossfeed_level
        output[:, 1] = audio[:, 1] + audio[:, 0] * crossfeed_level
        output *= 1.0 / (1 + crossfeed_level)
        
        return output
    
    def _normalize(self, audio: np.ndarray, 
                   target_level: float = 0.9) -> np.ndarray:
        """Normalize audio to target peak level."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio


class AmbisonicProcessor(SpatialAudioEngine):
    """
    Ambisonic audio processor for higher-order Ambisonic rendering.
    Supports Ambisonic encoding, decoding, and rotation.
    """
    
    # Ambisonic channel ordering
    ACN_CHANNEL_MAP = {
        0: (0, 0),   # W (omnidirectional)
        1: (1, 0),   # X (front)
        2: (0, 1),   # Y (right)
        3: (0, 0),   # Z (up) - actually (0,0) for 3D
    }
    
    def __init__(self, sample_rate: int = 48000, order: int = 1):
        super().__init__(sample_rate)
        self.order = order
        self.num_channels = (order + 1) ** 2
        self._init_ambisonic_matrices()
    
    def _init_ambisonic_matrices(self):
        """Initialize Ambisonic encoding/decoding matrices."""
        # Spherical harmonic weights for first order
        # N3D normalization
        self.sh_weights = {
            0: 1.0,  # W
            1: 1.0,  # X
            2: 1.0,  # Y
            3: 1.0,  # Z
        }
        
        # Decoder weights for common layouts
        self.decoder_weights = self._compute_ea_speaker_weights()
    
    def _compute_ea_speaker_weights(self) -> np.ndarray:
        """Compute equal-angle speaker decoding weights."""
        # 3D speaker layout (cube)
        speaker_dirs = self._get_cube_speaker_directions()
        
        # Compute pseudo-inverse for MVP decoder
        Y = self._compute_spherical_harmonics(speaker_dirs)
        
        # Regularized pseudo-inverse
        reg = 0.001
        Y_pinv = np.linalg.inv(Y.T @ Y + reg * np.eye(Y.shape[1])) @ Y.T
        
        return Y_pinv
    
    def _get_cube_speaker_directions(self) -> np.ndarray:
        """Get speaker directions for cube layout."""
        # 8 corners of a cube (8-channel Ambisonic)
        dirs = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    dir_vec = np.array([x, y, z])
                    dir_vec /= np.linalg.norm(dir_vec)
                    dirs.append(dir_vec)
        return np.array(dirs)
    
    def _compute_spherical_harmonics(self, directions: np.ndarray) -> np.ndarray:
        """Compute spherical harmonics for given directions."""
        # Convert to spherical coordinates
        r, theta, phi = self._cartesian_to_spherical(directions)
        
        # Compute first-order harmonics (N3D normalization)
        Y = np.zeros((len(directions), 4))
        
        # W (omnidirectional)
        Y[:, 0] = 0.5 * np.sqrt(1 / np.pi) * np.ones(len(directions))
        
        # X (front-back)
        Y[:, 1] = 0.5 * np.sqrt(3 / np.pi) * np.sin(theta) * np.cos(phi)
        
        # Y (left-right)
        Y[:, 2] = 0.5 * np.sqrt(3 / np.pi) * np.sin(theta) * np.sin(phi)
        
        # Z (up-down)
        Y[:, 3] = 0.5 * np.sqrt(3 / np.pi) * np.cos(theta)
        
        return Y
    
    def _cartesian_to_spherical(self, vectors: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Convert Cartesian to spherical coordinates."""
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(np.clip(z / (r + 1e-10), -1, 1))  # elevation
        phi = np.arctan2(y, x)  # azimuth
        
        return r, theta, phi
    
    def encode_mono(self, audio: np.ndarray, 
                    azimuth: float, 
                    elevation: float = 0) -> np.ndarray:
        """
        Encode mono source to Ambisonic (first-order).
        
        Args:
            audio: Input mono audio
            azimuth: Source azimuth in degrees
            elevation: Source elevation in degrees
        
        Returns:
            Ambisonic B-format (4 channels: W, X, Y, Z)
        """
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Compute spherical harmonic components
        cos_el = np.cos(el_rad)
        sin_el = np.sin(el_rad)
        cos_az = np.cos(az_rad)
        sin_az = np.sin(az_rad)
        
        # Encode to B-format
        W = audio * 0.5 * np.sqrt(1 / np.pi)
        X = audio * 0.5 * np.sqrt(3 / np.pi) * cos_el * cos_az
        Y = audio * 0.5 * np.sqrt(3 / np.pi) * cos_el * sin_az
        Z = audio * 0.5 * np.sqrt(3 / np.pi) * sin_el
        
        return np.stack([W, X, Y, Z], axis=-1 if audio.ndim > 1 else 0)
    
    def encode_stereo(self, audio: np.ndarray,
                      width: float = 1.0) -> np.ndarray:
        """
        Encode stereo to Ambisonic with stereo width control.
        
        Args:
            audio: Stereo audio (2 channels)
            width: Stereo width (0 = mono, 1 = full stereo)
        
        Returns:
            Ambisonic B-format
        """
        if audio.ndim < 2:
            return self.encode_mono(audio, 0, 0)
        
        # Simple stereo encoding: assume L/R at ±30 degrees
        left = self.encode_mono(audio[:, 0], -30 * width, 0)
        right = self.encode_mono(audio[:, 1], 30 * width, 0)
        
        return (left + right) / 2
    
    def decode_binaural(self, bformat: np.ndarray,
                        head_azimuth: float = 0,
                        head_elevation: float = 0) -> np.ndarray:
        """
        Decode Ambisonic B-format to binaural stereo.
        
        Args:
            bformat: Ambisonic B-format (4+ channels)
            head_azimuth: Listener head azimuth
            head_elevation: Listener head elevation
        
        Returns:
            Binaural stereo
        """
        # Ensure we have at least 4 channels
        if bformat.shape[-1] < 4:
            bformat = self._extend_bformat(bformat)
        
        # Get HRTF for head orientation
        # Use binaural renderer for the decoding
        binaural = BinauralRenderer(self.sample_rate)
        
        # Sum weighted B-format components
        # W contributes equally to both ears
        # X, Y, Z provide spatial information
        if bformat.ndim == 1:
            W = bformat[0]
            X = bformat[1] if len(bformat) > 1 else 0
            Y = bformat[2] if len(bformat) > 2 else 0
            Z = bformat[3] if len(bformat) > 3 else 0
        else:
            W = bformat[:, 0]
            X = bformat[:, 1] if bformat.shape[1] > 1 else 0
            Y = bformat[:, 2] if bformat.shape[1] > 2 else 0
            Z = bformat[:, 3] if bformat.shape[1] > 3 else 0
        
        # Simple binaural decoding using VBAP-like panning
        left = W + 0.7 * X - 0.5 * Y
        right = W - 0.7 * X - 0.5 * Y
        
        # Stack to stereo
        if left.ndim == 1:
            stereo = np.stack([left, right], axis=-1)
        else:
            stereo = np.column_stack([left, right])
        
        return self._normalize(stereo)
    
    def _extend_bformat(self, bformat: np.ndarray) -> np.ndarray:
        """Extend B-format to 4 channels if needed."""
        if bformat.ndim == 1:
            extended = np.zeros(4)
            extended[:len(bformat)] = bformat
            return extended
        else:
            extended = np.zeros((bformat.shape[0], 4))
            extended[:, :bformat.shape[1]] = bformat
            return extended
    
    def decode_speaker(self, bformat: np.ndarray,
                       speaker_positions: List[Tuple[float, float]]
                       ) -> np.ndarray:
        """
        Decode Ambisonic to physical speaker layout.
        
        Args:
            bformat: Ambisonic B-format
            speaker_positions: List of (azimuth, elevation) tuples
        
        Returns:
            Multi-channel speaker output
        """
        # Compute spherical harmonics for speaker positions
        dirs = []
        for az, el in speaker_positions:
            az_rad = np.radians(az)
            el_rad = np.radians(el)
            x = np.cos(el_rad) * np.cos(az_rad)
            y = np.cos(el_rad) * np.sin(az_rad)
            z = np.sin(el_rad)
            dirs.append([x, y, z])
        
        dirs = np.array(dirs)
        Y = self._compute_spherical_harmonics(dirs)
        
        # Apply decoder weights
        if bformat.ndim == 1:
            output = Y @ bformat[:4]
        else:
            output = Y @ bformat[:, :4]
        
        return output
    
    def rotate_bformat(self, bformat: np.ndarray,
                       yaw: float = 0,
                       pitch: float = 0,
                       roll: float = 0) -> np.ndarray:
        """
        Rotate Ambisonic B-format (rotation of sound field).
        
        Args:
            bformat: Ambisonic B-format
            yaw: Rotation around vertical axis (degrees)
            pitch: Rotation around lateral axis (degrees)
            roll: Rotation around forward axis (degrees)
        
        Returns:
            Rotated B-format
        """
        # Compute rotation matrices
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Rotation matrices for first-order Ambisonic
        # Yaw (Y-axis rotation)
        Ry = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad), 0],
            [0, 0, 0, 1]
        ])
        
        # Pitch (X-axis rotation)
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(pitch_rad), 0, -np.sin(pitch_rad)],
            [0, 0, 1, 0],
            [0, np.sin(pitch_rad), 0, np.cos(pitch_rad)]
        ])
        
        # Roll (Z-axis rotation)
        Rz = np.array([
            [np.cos(roll_rad), 0, 0, np.sin(roll_rad)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-np.sin(roll_rad), 0, 0, np.cos(roll_rad)]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Apply rotation
        if bformat.ndim == 1:
            return R @ bformat[:4]
        else:
            # Apply to each sample
            result = np.zeros_like(bformat)
            for i in range(len(bformat)):
                result[i] = R @ bformat[i, :4]
            return result
    
    def _normalize(self, audio: np.ndarray,
                   target_level: float = 0.9) -> np.ndarray:
        """Normalize audio to target peak level."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio


class Audio360Renderer:
    """
    360° audio renderer for VR/AR applications.
    Combines binaural and Ambisonic techniques with head tracking.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.binaural = BinauralRenderer(sample_rate)
        self.ambisonic = AmbisonicProcessor(sample_rate)
        self.head_orientation = (0, 0, 0)  # yaw, pitch, roll
        self.source_positions = []
        self.render_mode = 'binaural'  # 'binaural', 'ambisonic', 'automatic'
    
    def set_head_orientation(self, yaw: float, pitch: float = 0, roll: float = 0):
        """Update head orientation for spatial rendering."""
        self.head_orientation = (yaw, pitch, roll)
    
    def add_source(self, audio: np.ndarray, 
                   azimuth: float,
                   elevation: float = 0,
                   distance: float = 1.0):
        """Add an audio source at a specific position."""
        self.source_positions.append({
            'audio': audio,
            'azimuth': azimuth,
            'elevation': elevation,
            'distance': distance
        })
    
    def clear_sources(self):
        """Remove all audio sources."""
        self.source_positions = []
    
    def render(self, output_mode: str = 'stereo') -> np.ndarray:
        """
        Render all sources to output.
        
        Args:
            output_mode: 'stereo' (binaural) or 'multichannel'
        
        Returns:
            Rendered audio
        """
        if not self.source_positions:
            return np.zeros((1, 2))
        
        if self.render_mode == 'binaural' or output_mode == 'stereo':
            return self._render_binaural()
        elif self.render_mode == 'ambisonic':
            return self._render_ambisonic()
        else:
            # Automatic selection based on source count
            if len(self.source_positions) <= 2:
                return self._render_binaural()
            else:
                return self._render_ambisonic()
    
    def _render_binaural(self) -> np.ndarray:
        """Render using binaural processing."""
        outputs = []
        
        yaw, pitch, roll = self.head_orientation
        
        for source in self.source_positions:
            # Calculate relative angle
            rel_azimuth = source['azimuth'] - yaw
            rel_elevation = source['elevation'] - pitch
            
            # Normalize azimuth
            while rel_azimuth > 180:
                rel_azimuth -= 360
            while rel_azimuth < -180:
                rel_azimuth += 360
            
            # Render to binaural
            binaural = self.binaural.render_binaural(
                source['audio'],
                rel_azimuth,
                rel_elevation,
                source['distance']
            )
            outputs.append(binaural)
        
        # Mix all sources
        if outputs:
            max_len = max(len(o) for o in outputs)
            mixed = np.zeros((max_len, 2))
            for o in outputs:
                if len(o) < max_len:
                    padded = np.zeros((max_len, 2))
                    padded[:len(o)] = o
                    mixed += padded
                else:
                    mixed += o
            return self._normalize(mixed)
        
        return np.zeros((1, 2))
    
    def _render_ambisonic(self) -> np.ndarray:
        """Render using Ambisonic processing."""
        # Encode all sources to B-format
        bformat = None
        
        for source in self.source_positions:
            encoded = self.ambisonic.encode_mono(
                source['audio'],
                source['azimuth'],
                source['elevation']
            )
            
            # Distance attenuation
            dist_gain = 1.0 / (1.0 + 0.5 * (source['distance'] - 1.0))
            encoded *= dist_gain
            
            if bformat is None:
                bformat = encoded
            else:
                bformat += encoded
        
        if bformat is None:
            return np.zeros((1, 2))
        
        # Apply head rotation
        yaw, pitch, roll = self.head_orientation
        bformat = self.ambisonic.rotate_bformat(bformat, yaw, pitch, roll)
        
        # Decode to binaural
        return self.ambisonic.decode_binaural(bformat)
    
    def set_render_mode(self, mode: str):
        """Set rendering mode: 'binaural', 'ambisonic', or 'automatic'."""
        if mode in ['binaural', 'ambisonic', 'automatic']:
            self.render_mode = mode
    
    def _normalize(self, audio: np.ndarray,
                   target_level: float = 0.9) -> np.ndarray:
        """Normalize audio to target peak level."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio


class Panner3D:
    """
    3D panner for precise spatial positioning of audio sources.
    Supports multiple panning algorithms.
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.position = (0, 0, 1)  # Default position (front)
        self.panning_method = 'hrtf'  # 'hrtf', 'vbap', 'stereo平衡'
    
    def set_position(self, azimuth: float, elevation: float = 0, distance: float = 1.0):
        """Set source position in 3D space."""
        self.position = (azimuth, elevation, distance)
    
    def pan_mono(self, audio: np.ndarray) -> np.ndarray:
        """Pan mono audio to stereo based on current position."""
        az, el, dist = self.position
        
        # Simple stereo panning based on azimuth
        az_rad = np.radians(az)
        
        # Equal power panning
        pan_angle = (az + 90) / 180 * np.pi  # 0 to PI
        left_gain = np.cos(pan_angle)
        right_gain = np.sin(pan_angle)
        
        # Apply distance attenuation
        dist_gain = 1.0 / (1.0 + 0.3 * (dist - 1.0))
        
        if audio.ndim > 1:
            # Already stereo, apply gains
            output = audio.copy()
            output[:, 0] *= left_gain * dist_gain
            output[:, 1] *= right_gain * dist_gain
        else:
            # Mono to stereo
            output = np.zeros((len(audio), 2))
            output[:, 0] = audio * left_gain * dist_gain
            output[:, 1] = audio * right_gain * dist_gain
        
        return output
    
    def pan_vbap(self, audio: np.ndarray, 
                 speaker_dirs: List[Tuple[float, float]]) -> np.ndarray:
        """
        Vector Base Amplitude Panning (VBAP).
        
        Args:
            audio: Mono audio
            speaker_dirs: List of (azimuth, elevation) tuples for speakers
        
        Returns:
            Multi-channel output
        """
        az, el, _ = self.position
        
        # Convert to 3D vector
        source_vec = self._az_el_to_vector(az, el)
        
        # Find closest 3 speakers
        speaker_vectors = [self._az_el_to_vector(az_, el_) for az_, el_ in speaker_dirs]
        
        # Compute VBAP gains
        gains = self._compute_vbap_gains(source_vec, speaker_vectors)
        
        # Apply panning
        output = np.zeros((len(audio), len(speaker_dirs)))
        for i, gain in enumerate(gains):
            output[:, i] = audio * gain
        
        return output
    
    def _az_el_to_vector(self, azimuth: float, elevation: float) -> np.ndarray:
        """Convert azimuth/elevation to 3D unit vector."""
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        x = np.cos(el_rad) * np.cos(az_rad)
        y = np.cos(el_rad) * np.sin(az_rad)
        z = np.sin(el_rad)
        
        return np.array([x, y, z])
    
    def _compute_vbap_gains(self, source: np.ndarray,
                           speakers: List[np.ndarray]) -> List[float]:
        """Compute VBAP panning gains for 3D."""
        if len(speakers) < 3:
            return [1.0 / len(speakers)] * len(speakers) if speakers else [1.0]
        
        # Use first 3 speakers for simplicity
        S = np.column_stack(speakers[:3])
        
        try:
            gains = np.linalg.solve(S, source)
            # Normalize
            gain_sum = np.sum(np.abs(gains))
            if gain_sum > 0:
                gains /= gain_sum
            return list(gains)
        except np.linalg.LinAlgError:
            return [1.0 / 3] * 3


# Utility functions

def create_spatial_audio_engine(mode: str = 'binaural', 
                                sample_rate: int = 48000) -> SpatialAudioEngine:
    """
    Factory function to create spatial audio engine.
    
    Args:
        mode: 'binaural', 'ambisonic', or '360'
        sample_rate: Audio sample rate
    
    Returns:
        Spatial audio engine
    """
    if mode == 'binaural':
        return BinauralRenderer(sample_rate)
    elif mode == 'ambisonic':
        return AmbisonicProcessor(sample_rate)
    elif mode == '360':
        return Audio360Renderer(sample_rate)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def estimate_rtf(audio: np.ndarray, 
                 sample_rate: int = 48000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate Room Transfer Function from impulse response.
    
    Args:
        audio: Impulse response recording
        sample_rate: Sample rate
    
    Returns:
        (magnitude_response, phase_response)
    """
    # Compute frequency response
    freq_response = np.fft.rfft(audio)
    magnitude = np.abs(freq_response)
    phase = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    
    return magnitude, phase
