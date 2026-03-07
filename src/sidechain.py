"""
Sidechain Compressor Module for AI DJ Project
Advanced sidechain compression techniques for music production
"""

import numpy as np
from typing import Optional, Tuple, Callable
from scipy import signal
from scipy.signal import butter, lfilter, firwin


class SidechainCompressor:
    """
    Sidechain compressor with multiple modes:
    - Standard sidechain (kick → bass/synths)
    - Ghost kick (synthetic sidechain trigger)
    - Parallel sidechain (New York style)
    - Mid/Side sidechain (stereo width control)
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._envelope = 0.0
        self._gain_buffer = None
    
    # ==================== CORE SIDECHAIN ====================
    
    def sidechain_compress(
        self,
        audio: np.ndarray,
        sidechain: np.ndarray,
        threshold: float = -18.0,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.15,
        makeup_gain: float = 0.0,
        knee: float = 6.0
    ) -> np.ndarray:
        """
        Standard sidechain compression.
        
        Parameters:
            audio: Audio to be compressed (target)
            sidechain: Trigger signal (e.g., kick drum)
            threshold: Threshold in dB
            ratio: Compression ratio (4:1, 8:1, etc.)
            attack: Attack time in seconds
            release: Release time in seconds
            makeup_gain: Makeup gain in dB
            knee: Knee width in dB
            
        Returns:
            Compressed audio
        """
        # Ensure same length
        min_len = min(len(audio), len(sidechain))
        audio = audio[:min_len]
        sidechain = sidechain[:min_len]
        
        # Convert sidechain to dB for envelope detection
        sidechain_db = 20 * np.log10(np.abs(sidechain) + 1e-10)
        
        # Envelope follower
        attack_coeff = np.exp(-1.0 / (attack * self.sample_rate))
        release_coeff = np.exp(-1.0 / (release * self.sample_rate))
        
        envelope = np.zeros(min_len)
        self._envelope = 0.0
        
        for i in range(min_len):
            # Envelope detection
            if sidechain_db[i] > self._envelope:
                self._envelope = attack_coeff * self._envelope + (1 - attack_coeff) * sidechain_db[i]
            else:
                self._envelope = release_coeff * self._envelope + (1 - release_coeff) * sidechain_db[i]
            envelope[i] = self._envelope
        
        # Calculate gain reduction
        gain_reduction = np.zeros(min_len)
        
        for i in range(min_len):
            env = envelope[i]
            
            # Soft knee
            if env > threshold + knee / 2:
                over = env - (threshold + knee / 2)
                gain_reduction[i] = -over * (1 - 1 / ratio)
            elif env > threshold - knee / 2:
                over = env - threshold
                gain_reduction[i] = -(over ** 2) / (2 * knee) * (1 - 1 / ratio)
            else:
                gain_reduction[i] = 0
        
        # Apply gain reduction with makeup
        gain_linear = 10 ** ((gain_reduction + makeup_gain) / 20)
        return audio * gain_linear
    
    # ==================== GHOST KICK ====================
    
    def generate_ghost_kick(
        self,
        duration: float,
        bpm: float = 128.0,
        pattern: Optional[str] = None,
        attack: float = 0.001,
        decay: float = 0.3,
        pitch: float = 60.0
    ) -> np.ndarray:
        """
        Generate synthetic ghost kick for sidechain triggering.
        
        Parameters:
            duration: Total duration in seconds
            bpm: Beats per minute
            pattern: Rhythm pattern ('four_on_floor', 'syncopated', 'half_time')
            attack: Kick attack time
            decay: Kick decay time
            pitch: Kick pitch in Hz
            
        Returns:
            Ghost kick audio signal
        """
        num_samples = int(duration * self.sample_rate)
        kick = np.zeros(num_samples)
        
        # Beat interval in samples
        beat_samples = int(self.sample_rate * 60.0 / bpm)
        
        # Default pattern: four on the floor
        if pattern is None:
            pattern = 'four_on_floor'
        
        patterns = {
            'four_on_floor': [0, 1, 2, 3],
            'syncopated': [0, 1.5, 2, 3],
            'half_time': [0, 2],
            ' triplet': [0, 0.66, 1.33, 2],
            'stomping': [0, 0, 1, 1, 2, 3, 3],
        }
        
        kicks = patterns.get(pattern, patterns['four_on_floor'])
        
        for kick_pos in kicks:
            # Calculate position in samples
            pos = int(kick_pos * beat_samples)
            
            if pos < num_samples:
                # Generate kick envelope
                env_len = int(self.sample_rate * (attack + decay))
                t = np.linspace(0, 1, env_len)
                
                # Exponential decay envelope
                env = np.exp(-t * (3 + 5 * decay))
                
                # Add pitch envelope (pitch drops)
                freq_env = pitch * np.exp(-t * 2)
                
                # Generate sine wave
                kick_partial = np.sin(2 * np.pi * freq_env * t / self.sample_rate * np.arange(env_len))
                
                # Apply envelope
                kick_partial *= env
                
                # Mix into output
                end_pos = min(pos + env_len, num_samples)
                kick[pos:end_pos] += kick_partial[:end_pos - pos]
        
        return kick
    
    def apply_ghost_kick_sidechain(
        self,
        audio: np.ndarray,
        bpm: float = 128.0,
        pattern: str = 'four_on_floor',
        threshold: float = -18.0,
        ratio: float = 6.0,
        attack: float = 0.003,
        release: float = 0.1
    ) -> np.ndarray:
        """
        Apply sidechain compression using synthetic ghost kick.
        
        Parameters:
            audio: Audio to compress
            bpm: Target BPM
            pattern: Kick pattern
            threshold: Threshold in dB
            ratio: Compression ratio
            attack: Attack time
            release: Release time
            
        Returns:
            Compressed audio
        """
        duration = len(audio) / self.sample_rate
        ghost_kick = self.generate_ghost_kick(duration, bpm, pattern)
        
        return self.sidechain_compress(
            audio, ghost_kick,
            threshold=threshold,
            ratio=ratio,
            attack=attack,
            release=release
        )
    
    # ==================== PARALLEL SIDECHAIN ====================
    
    def parallel_sidechain(
        self,
        audio: np.ndarray,
        sidechain: np.ndarray,
        blend: float = 0.5,
        ratio: float = 10.0,
        threshold: float = -20.0,
        attack: float = 0.001,
        release: float = 0.05
    ) -> np.ndarray:
        """
        Parallel (New York style) sidechain compression.
        Heavy compression with wet/dry blend.
        
        Parameters:
            audio: Original audio
            sidechain: Sidechain trigger
            blend: Wet/dry mix (0=dry, 1=full wet)
            ratio: High ratio for parallel (10:1 or infinite)
            threshold: Lower threshold for heavy compression
            attack: Fast attack
            release: Fast release
            
        Returns:
            Mixed dry + compressed signal
        """
        # Heavy compressed signal
        compressed = self.sidechain_compress(
            audio, sidechain,
            threshold=threshold,
            ratio=ratio,
            attack=attack,
            release=release,
            makeup_gain=6.0  # Compensate for heavy compression
        )
        
        # Mix dry and wet
        return audio * (1 - blend) + compressed * blend
    
    # ==================== MID/SIDE SIDECHAIN ====================
    
    def mid_side_encode(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode stereo to mid/side.
        
        Parameters:
            audio: Stereo audio [samples, 2]
            
        Returns:
            (mid, side) arrays
        """
        if audio.ndim == 1:
            # Mono - duplicate
            mid = audio
            side = np.zeros_like(audio)
        else:
            mid = (audio[:, 0] + audio[:, 1]) / np.sqrt(2)
            side = (audio[:, 0] - audio[:, 1]) / np.sqrt(2)
        
        return mid, side
    
    def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
        """
        Decode mid/side to stereo.
        
        Parameters:
            mid: Mid channel
            side: Side channel
            
        Returns:
            Stereo audio [samples, 2]
        """
        left = (mid + side) * np.sqrt(2) / 2
        right = (mid - side) * np.sqrt(2) / 2
        
        return np.stack([left, right], axis=1)
    
    def mid_side_sidechain(
        self,
        audio: np.ndarray,
        sidechain: np.ndarray,
        mid_threshold: float = -18.0,
        side_threshold: float = -24.0,
        mid_ratio: float = 4.0,
        side_ratio: float = 2.0,
        attack: float = 0.01,
        release: float = 0.2
    ) -> np.ndarray:
        """
        Mid/Side specific sidechain compression.
        - Mid: Controls bass/center (more aggressive)
        - Side: Controls stereo width (less aggressive)
        
        Parameters:
            audio: Stereo audio to process
            sidechain: Sidechain trigger
            mid_threshold: Threshold for mid channel
            side_threshold: Threshold for side channel
            mid_ratio: Ratio for mid channel
            side_ratio: Ratio for side channel
            attack: Attack time
            release: Release time
            
        Returns:
            Processed stereo audio
        """
        # Encode to mid/side
        mid, side = self.mid_side_encode(audio)
        
        # Apply different compression to each
        mid_compressed = self.sidechain_compress(
            mid, sidechain,
            threshold=mid_threshold,
            ratio=mid_ratio,
            attack=attack,
            release=release
        )
        
        side_compressed = self.sidechain_compress(
            side, sidechain,
            threshold=side_threshold,
            ratio=side_ratio,
            attack=attack,
            release=release
        )
        
        # Decode back to stereo
        return self.mid_side_decode(mid_compressed, side_compressed)
    
    # ==================== ADVANCED MODES ====================
    
    def multiband_sidechain(
        self,
        audio: np.ndarray,
        sidechain: np.ndarray,
        crossover_frequencies: Tuple[float, float] = (200, 2000),
        thresholds: Tuple[float, float, float] = (-15.0, -18.0, -21.0),
        ratios: Tuple[float, float, float] = (2.0, 4.0, 6.0),
        attack: float = 0.01,
        release: float = 0.15
    ) -> np.ndarray:
        """
        Multiband sidechain compression.
        Different compression settings per frequency band.
        
        Parameters:
            audio: Audio to process
            sidechain: Sidechain trigger
            crossover_frequencies: (low_mid, mid_high) crossover points
            thresholds: (low, mid, high) thresholds in dB
            ratios: (low, mid, high) ratios
            attack: Attack time
            release: Release time
            
        Returns:
            Compressed audio
        """
        # Split into bands
        low, mid, high = self._split_bands(audio, crossover_frequencies)
        
        # Process each band with different settings
        low_compressed = self.sidechain_compress(
            low, sidechain,
            threshold=thresholds[0],
            ratio=ratios[0],
            attack=attack,
            release=release
        )
        
        mid_compressed = self.sidechain_compress(
            mid, sidechain,
            threshold=thresholds[1],
            ratio=ratios[1],
            attack=attack,
            release=release
        )
        
        high_compressed = self.sidechain_compress(
            high, sidechain,
            threshold=thresholds[2],
            ratio=ratios[2],
            attack=attack,
            release=release
        )
        
        # Recombine
        return low_compressed + mid_compressed + high_compressed
    
    def _split_bands(
        self,
        audio: np.ndarray,
        crossover_frequencies: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split audio into three frequency bands."""
        nyquist = self.sample_rate / 2
        
        # Lowpass for low band
        b, a = butter(4, crossover_frequencies[0] / nyquist, 'low')
        low = lfilter(b, a, audio)
        
        # Bandpass for mid band
        b, a = butter(4, [
            crossover_frequencies[0] / nyquist,
            crossover_frequencies[1] / nyquist
        ], 'band')
        mid = lfilter(b, a, audio)
        
        # Highpass for high band
        b, a = butter(4, crossover_frequencies[1] / nyquist, 'high')
        high = lfilter(b, a, audio)
        
        return low, mid, high
    
    def ducking(
        self,
        audio: np.ndarray,
        sidechain: np.ndarray,
        depth: float = 12.0,
        attack: float = 0.01,
        release: float = 0.3
    ) -> np.ndarray:
        """
        Simple volume ducking (like radio talk over music).
        More subtle than full compression.
        
        Parameters:
            audio: Audio to duck
            sidechain: Trigger signal
            depth: Ducking depth in dB
            attack: Attack time
            release: Release time
            
        Returns:
            Ducked audio
        """
        # Convert sidechain to envelope
        sidechain_db = 20 * np.log10(np.abs(sidechain) + 1e-10)
        
        # Smooth envelope
        attack_coeff = np.exp(-1.0 / (attack * self.sample_rate))
        release_coeff = np.exp(-1.0 / (release * self.sample_rate))
        
        envelope = np.zeros_like(sidechain_db)
        env = 0.0
        
        for i in range(len(sidechain_db)):
            if sidechain_db[i] > env:
                env = attack_coeff * env + (1 - attack_coeff) * sidechain_db[i]
            else:
                env = release_coeff * env + (1 - release_coeff) * sidechain_db[i]
            envelope[i] = env
        
        # Normalize envelope and apply ducking
        envelope = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-10)
        duck_gain = 1.0 - envelope * (1 - 10 ** (-depth / 20))
        
        return audio * duck_gain
    
    # ==================== PRESETS ====================
    
    @staticmethod
    def edm_preset() -> dict:
        """EDM-style sidechain settings."""
        return {
            'threshold': -18.0,
            'ratio': 6.0,
            'attack': 0.005,
            'release': 0.15,
            'makeup_gain': 0.0,
            'knee': 6.0
        }
    
    @staticmethod
    def house_preset() -> dict:
        """House music sidechain settings."""
        return {
            'threshold': -15.0,
            'ratio': 4.0,
            'attack': 0.01,
            'release': 0.2,
            'makeup_gain': 0.0,
            'knee': 6.0
        }
    
    @staticmethod
    def dubstep_preset() -> dict:
        """Dubstep-style aggressive sidechain."""
        return {
            'threshold': -21.0,
            'ratio': 8.0,
            'attack': 0.001,
            'release': 0.1,
            'makeup_gain': 0.0,
            'knee': 3.0
        }
    
    @staticmethod
    def vocal_dip_preset() -> dict:
        """Subtle vocal ducking preset."""
        return {
            'threshold': -20.0,
            'ratio': 3.0,
            'attack': 0.02,
            'release': 0.3,
            'makeup_gain': 2.0,
            'knee': 9.0
        }


class SidechainProcessor:
    """
    High-level sidechain processor with presets and automation.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.compressor = SidechainCompressor(sample_rate)
        self._current_preset = None
    
    def process(
        self,
        audio: np.ndarray,
        sidechain: np.ndarray,
        mode: str = 'standard',
        preset: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Process audio with sidechain compression.
        
        Parameters:
            audio: Target audio
            sidechain: Trigger signal
            mode: Processing mode ('standard', 'parallel', 'mid_side', 'multiband', 'ducking')
            preset: Preset name ('edm', 'house', 'dubstep', 'vocal_dip')
            **kwargs: Override preset settings
            
        Returns:
            Processed audio
        """
        # Load preset
        if preset:
            settings = self._get_preset(preset)
            settings.update(kwargs)
            kwargs = settings
        
        # Route to appropriate processor
        if mode == 'standard':
            return self.compressor.sidechain_compress(
                audio, sidechain, **kwargs
            )
        elif mode == 'parallel':
            return self.compressor.parallel_sidechain(
                audio, sidechain, **kwargs
            )
        elif mode == 'mid_side':
            return self.compressor.mid_side_sidechain(
                audio, sidechain, **kwargs
            )
        elif mode == 'multiband':
            return self.compressor.multiband_sidechain(
                audio, sidechain, **kwargs
            )
        elif mode == 'ducking':
            return self.compressor.ducking(
                audio, sidechain, **kwargs
            )
        elif mode == 'ghost_kick':
            return self.compressor.apply_ghost_kick_sidechain(
                audio, **kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _get_preset(self, name: str) -> dict:
        """Get preset settings by name."""
        presets = {
            'edm': self.compressor.edm_preset(),
            'house': self.compressor.house_preset(),
            'dubstep': self.compressor.dubstep_preset(),
            'vocal_dip': self.compressor.vocal_dip_preset(),
        }
        
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}")
        
        return presets[name]
    
    def process_from_stems(
        self,
        stems: dict,
        trigger_stem: str = 'drums',
        target_stems: list = None,
        mode: str = 'standard',
        **kwargs
    ) -> dict:
        """
        Process multiple stems with sidechain.
        
        Parameters:
            stems: Dict of {stem_name: audio_array}
            trigger_stem: Stem to use as sidechain trigger
            target_stems: Stems to apply sidechain to (default: all except trigger)
            mode: Processing mode
            **kwargs: Additional settings
            
        Returns:
            Dict of processed stems
        """
        if trigger_stem not in stems:
            raise ValueError(f"Trigger stem '{trigger_stem}' not found")
        
        if target_stems is None:
            target_stems = [k for k in stems.keys() if k != trigger_stem]
        
        sidechain = stems[trigger_stem]
        processed = {}
        
        for stem_name in target_stems:
            if stem_name in stems:
                processed[stem_name] = self.process(
                    stems[stem_name],
                    sidechain,
                    mode=mode,
                    **kwargs
                )
        
        # Keep trigger unchanged
        processed[trigger_stem] = stems[trigger_stem]
        
        return processed


# Convenience functions

def sidechain_compress(
    audio: np.ndarray,
    sidechain: np.ndarray,
    **kwargs
) -> np.ndarray:
    """Quick sidechain compression."""
    sc = SidechainCompressor()
    return sc.sidechain_compress(audio, sidechain, **kwargs)


def ghost_kick_sidechain(
    audio: np.ndarray,
    bpm: float = 128.0,
    pattern: str = 'four_on_floor',
    **kwargs
) -> np.ndarray:
    """Sidechain using synthetic ghost kick."""
    sc = SidechainCompressor()
    return sc.apply_ghost_kick_sidechain(audio, bpm, pattern, **kwargs)


def parallel_compress(
    audio: np.ndarray,
    sidechain: np.ndarray,
    blend: float = 0.5,
    **kwargs
) -> np.ndarray:
    """Parallel sidechain compression."""
    sc = SidechainCompressor()
    return sc.parallel_sidechain(audio, sidechain, blend, **kwargs)
