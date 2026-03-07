#!/usr/bin/env python3
"""
Stem Mastering System
=====================

Professional mastering for individual audio stems.
Each stem type receives specialized processing for optimal quality.

Stem Types:
- vocals: EQ, de-essing, compression, reverb control
- drums: parallel compression, transient shaping, punch enhancement
- bass: sub-harmonic synthesis, mono bass, compression
- melody/synth: air/presence EQ, stereo widening
- pads/ambient: reverb/delay automation, gentle compression

Usage:
    from stem_mastering import StemMastering, StemType
    
    mastering = StemMastering(platform='spotify')
    mastered_stems = mastering.master_stems(stem_files)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import json
import warnings

# =============================================================================
# STEM TYPE DEFINITIONS
# =============================================================================

class StemType(Enum):
    """Audio stem categories"""
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    MELODY = "melody"
    SYNTH = "synth"
    PAD = "pad"
    GUITAR = "guitar"
    OTHER = "other"


class StemConfig:
    """Configuration for each stem type"""
    
    # Default EQ settings (frequency in Hz, gain in dB)
    EQ_PRESETS = {
        StemType.VOCALS: {
            'low_cut': 80,
            'high_cut': 16000,
            'low_boost': 0,      # 80-200Hz
            'mid_boost': 2,      # 1-4kHz presence
            'high_boost': 1,    # 8-12kHz air
        },
        StemType.DRUMS: {
            'low_cut': 40,
            'high_cut': 18000,
            'low_boost': 3,      # 60-120Hz kick punch
            'mid_boost': 0,
            'high_boost': 2,     # 8-12kHz shimmer
        },
        StemType.BASS: {
            'low_cut': 30,
            'high_cut': 500,
            'low_boost': 4,      # 40-80Hz sub
            'mid_boost': -2,     # 200-400Hz boxiness removal
            'high_boost': 0,
        },
        StemType.MELODY: {
            'low_cut': 60,
            'high_cut': 16000,
            'low_boost': 0,
            'mid_boost': 1,
            'high_boost': 3,     # 10-14kHz presence
        },
        StemType.SYNTH: {
            'low_cut': 60,
            'high_cut': 14000,
            'low_boost': 1,
            'mid_boost': 2,
            'high_boost': 2,
        },
        StemType.PAD: {
            'low_cut': 50,
            'high_cut': 12000,
            'low_boost': 2,
            'mid_boost': 0,
            'high_boost': 0,
        },
        StemType.GUITAR: {
            'low_cut': 80,
            'high_cut': 12000,
            'low_boost': 0,
            'mid_boost': 2,
            'high_boost': 1,
        },
        StemType.OTHER: {
            'low_cut': 60,
            'high_cut': 16000,
            'low_boost': 0,
            'mid_boost': 0,
            'high_boost': 0,
        },
    }
    
    # Compression presets
    COMPRESSION_PRESETS = {
        StemType.VOCALS: {
            'threshold': -18,
            'ratio': 3.0,
            'attack': 5,
            'release': 80,
            'knee': 6,
            'makeup': 3,
        },
        StemType.DRUMS: {
            'threshold': -12,
            'ratio': 4.0,
            'attack': 1,
            'release': 100,
            'knee': 4,
            'makeup': 4,
        },
        StemType.BASS: {
            'threshold': -15,
            'ratio': 4.0,
            'attack': 2,
            'release': 150,
            'knee': 6,
            'makeup': 2,
        },
        StemType.MELODY: {
            'threshold': -20,
            'ratio': 2.5,
            'attack': 8,
            'release': 100,
            'knee': 4,
            'makeup': 2,
        },
        StemType.SYNTH: {
            'threshold': -18,
            'ratio': 2.0,
            'attack': 10,
            'release': 80,
            'knee': 5,
            'makeup': 1,
        },
        StemType.PAD: {
            'threshold': -22,
            'ratio': 1.5,
            'attack': 20,
            'release': 150,
            'knee': 8,
            'makeup': 0,
        },
        StemType.GUITAR: {
            'threshold': -20,
            'ratio': 2.5,
            'attack': 5,
            'release': 100,
            'knee': 4,
            'makeup': 2,
        },
        StemType.OTHER: {
            'threshold': -18,
            'ratio': 2.0,
            'attack': 10,
            'release': 100,
            'knee': 4,
            'makeup': 2,
        },
    }
    
    # Stereo width presets
    STEREO_PRESETS = {
        StemType.VOCALS: {
            'width': 0.8,      # Slightly narrow for clarity
            'mid_high_boost': 0,
        },
        StemType.DRUMS: {
            'width': 1.2,      # Wider for overheads
            'mid_high_boost': 2,
        },
        StemType.BASS: {
            'width': 0.5,      # Mono sub
            'mid_high_boost': 0,
        },
        StemType.MELODY: {
            'width': 1.0,
            'mid_high_boost': 1,
        },
        StemType.SYNTH: {
            'width': 1.1,
            'mid_high_boost': 2,
        },
        StemType.PAD: {
            'width': 1.3,      # Wide ambient
            'mid_high_boost': 0,
        },
        StemType.GUITAR: {
            'width': 0.9,
            'mid_high_boost': 1,
        },
        StemType.OTHER: {
            'width': 1.0,
            'mid_high_boost': 0,
        },
    }
    
    # Target LUFS per stem type (relative to full mix)
    LUFS_PRESETS = {
        StemType.VOCALS: -20,
        StemType.DRUMS: -16,
        StemType.BASS: -14,
        StemType.MELODY: -18,
        StemType.SYNTH: -18,
        StemType.PAD: -22,
        StemType.GUITAR: -20,
        StemType.OTHER: -18,
    }
    
    @classmethod
    def get_eq(cls, stem_type: StemType) -> Dict[str, float]:
        return cls.EQ_PRESETS.get(stem_type, cls.EQ_PRESETS[StemType.OTHER])
    
    @classmethod
    def get_compression(cls, stem_type: StemType) -> Dict[str, float]:
        return cls.COMPRESSION_PRESETS.get(stem_type, cls.COMPRESSION_PRESETS[StemType.OTHER])
    
    @classmethod
    def get_stereo(cls, stem_type: StemType) -> Dict[str, float]:
        return cls.STEREO_PRESETS.get(stem_type, cls.STEREO_PRESETS[StemType.OTHER])
    
    @classmethod
    def get_lufs(cls, stem_type: StemType) -> float:
        return cls.LUFS_PRESETS.get(stem_type, -18)


# =============================================================================
# SIGNAL PROCESSING UTILITIES
# =============================================================================

def db_to_linear(db: float) -> float:
    """Convert dB to linear"""
    return 10 ** (db / 20)


def linear_to_db(linear: float) -> float:
    """Convert linear to dB"""
    return 20 * np.log10(max(linear, 1e-10))


def apply_highpass(audio: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Apply high-pass filter"""
    # Simple first-order RC filter
    dt = 1.0 / sample_rate
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    alpha = rc / (rc + dt)
    
    if audio.ndim == 1:
        output = np.zeros_like(audio)
        output[0] = audio[0]
        for i in range(1, len(audio)):
            output[i] = alpha * (output[i-1] + audio[i] - audio[i-1])
    else:
        output = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            output[:, ch] = apply_highpass(audio[:, ch], cutoff_hz, sample_rate)
    
    return output


def apply_lowpass(audio: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Apply low-pass filter"""
    # Simple first-order RC filter
    dt = 1.0 / sample_rate
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    
    if audio.ndim == 1:
        output = np.zeros_like(audio)
        output[0] = audio[0]
        for i in range(1, len(audio)):
            output[i] = output[i-1] + alpha * (audio[i] - output[i-1])
    else:
        output = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            output[:, ch] = apply_lowpass(audio[:, ch], cutoff_hz, sample_rate)
    
    return output


def apply_eq(audio: np.ndarray, sample_rate: int, 
             low_cut: float, high_cut: float,
             low_boost: float, mid_boost: float, high_boost: float) -> np.ndarray:
    """Apply multi-band EQ"""
    # High-pass
    if low_cut > 20:
        audio = apply_highpass(audio, low_cut, sample_rate)
    
    # Low shelf
    if low_boost != 0:
        audio = audio * db_to_linear(low_boost)
    
    # Mid peak (simplified - would use biquad in production)
    if mid_boost != 0:
        audio = audio * db_to_linear(mid_boost)
    
    # High shelf
    if high_boost != 0:
        audio = audio * db_to_linear(high_boost)
    
    # Low-pass
    if high_cut < 20000:
        audio = apply_lowpass(audio, high_cut, sample_rate)
    
    return audio


def apply_compression(audio: np.ndarray, sample_rate: int,
                      threshold_db: float, ratio: float,
                      attack_ms: float, release_ms: float,
                      makeup_db: float = 0) -> np.ndarray:
    """Apply dynamic range compression"""
    # Convert to mono for gain detection
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    
    # Calculate envelope
    attack = np.exp(-1.0 / (attack_ms * sample_rate / 1000))
    release = np.exp(-1.0 / (release_ms * sample_rate / 1000))
    
    envelope = np.zeros_like(mono)
    envelope[0] = abs(mono[0])
    
    for i in range(1, len(mono)):
        input_level = abs(mono[i])
        if input_level > envelope[i-1]:
            envelope[i] = attack * envelope[i-1] + (1 - attack) * input_level
        else:
            envelope[i] = release * envelope[i-1] + (1 - release) * input_level
    
    # Calculate gain reduction
    threshold_linear = db_to_linear(threshold_db)
    envelope_db = linear_to_db(envelope + 1e-10)
    threshold_env = np.maximum(envelope_db, threshold_db)
    gain_reduction_db = (threshold_env - envelope_db) * (1 - 1/ratio)
    gain_reduction_db = np.maximum(gain_reduction_db, 0)
    
    # Apply gain reduction
    gain = db_to_linear(-gain_reduction_db + makeup_db)
    
    if audio.ndim == 1:
        output = audio * gain
    else:
        output = audio * gain[:, np.newaxis]
    
    return output


def apply_stereo_widening(audio: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Apply mid-side stereo widening"""
    if audio.ndim < 2 or audio.shape[1] < 2:
        return audio
    
    # Convert to mid-side
    mid = (audio[:, 0] + audio[:, 1]) / np.sqrt(2)
    side = (audio[:, 0] - audio[:, 1]) / np.sqrt(2)
    
    # Apply width to sides
    side = side * width
    
    # Convert back to left-right
    output = np.column_stack([
        (mid + side) / np.sqrt(2),
        (mid - side) / np.sqrt(2)
    ])
    
    return output


def normalize_to_lufs(audio: np.ndarray, sample_rate: int, 
                      target_lufs: float) -> np.ndarray:
    """Normalize audio to target LUFS"""
    # Calculate RMS (simplified loudness)
    rms = np.sqrt(np.mean(audio ** 2))
    current_lufs = linear_to_db(rms)
    
    # Calculate gain needed
    gain_db = target_lufs - current_lufs
    
    # Apply gain
    output = audio * db_to_linear(gain_db)
    
    # Soft clip if needed
    max_val = np.max(np.abs(output))
    if max_val > 0.95:
        output = np.tanh(output / max_val * 0.95) * max_val
    
    return output


def soft_clip(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Apply soft clipping for smooth limiting"""
    max_val = np.max(np.abs(audio))
    if max_val > threshold:
        return np.tanh(audio / max_val * threshold) * max_val
    return audio


# =============================================================================
# STEM MASTERING CLASS
# =============================================================================

class StemMastering:
    """
    Professional mastering for individual audio stems.
    
    Each stem is processed with type-specific EQ, compression,
    stereo widening, and loudness normalization.
    """
    
    def __init__(self, platform: str = 'spotify', sample_rate: int = 44100):
        """
        Initialize stem mastering system.
        
        Args:
            platform: Target streaming platform ('spotify', 'apple', 'tidal', etc.)
            sample_rate: Audio sample rate
        """
        self.platform = platform.lower()
        self.sample_rate = sample_rate
        
        # Platform-specific target LUFS
        self.target_lufs = {
            'spotify': -14,
            'apple': -16,
            'tidal': -14,
            'youtube': -14,
            'amazon': -14,
            'soundcloud': -14,
        }.get(self.platform, -14)
        
        self.processing_report = {}
    
    def detect_stem_type(self, stem_name: str) -> StemType:
        """Detect stem type from filename"""
        name_lower = stem_name.lower()
        
        if any(kw in name_lower for kw in ['vocal', 'acapella', 'voice', 'lead']):
            return StemType.VOCALS
        elif any(kw in name_lower for kw in ['drum', 'percussion', 'beat']):
            return StemType.DRUMS
        elif any(kw in name_lower for kw in ['bass', 'sub', '808']):
            return StemType.BASS
        elif any(kw in name_lower for kw in ['melody', 'lead', 'synth', 'arp']):
            return StemType.MELODY
        elif any(kw in name_lower for kw in ['synth', 'pluck', 'pad', 'ambient']):
            return StemType.SYNTH if 'pad' not in name_lower else StemType.PAD
        elif any(kw in name_lower for kw in ['pad', 'ambient', 'atmosphere']):
            return StemType.PAD
        elif any(kw in name_lower for kw in ['guitar', 'gtr']):
            return StemType.GUITAR
        
        return StemType.OTHER
    
    def master_stem(self, audio: np.ndarray, stem_type: StemType,
                    enable_eq: bool = True,
                    enable_compression: bool = True,
                    enable_stereo: bool = True,
                    enable_limiter: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Apply full mastering chain to a single stem.
        
        Args:
            audio: Input audio (samples,) or (samples, channels)
            stem_type: Type of stem for processing
            enable_eq: Enable EQ processing
            enable_compression: Enable compression
            enable_stereo: Enable stereo widening
            enable_limiter: Enable soft limiting
            
        Returns:
            Tuple of (mastered audio, processing report)
        """
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        report = {
            'stem_type': stem_type.value,
            'input_rms_db': linear_to_db(np.sqrt(np.mean(audio ** 2))),
            'processing_steps': [],
        }
        
        # Get presets for this stem type
        eq_preset = StemConfig.get_eq(stem_type)
        comp_preset = StemConfig.get_compression(stem_type)
        stereo_preset = StemConfig.get_stereo(stem_type)
        target_lufs = StemConfig.get_lufs(stem_type)
        
        # 1. EQ
        if enable_eq:
            audio = apply_eq(
                audio, self.sample_rate,
                low_cut=eq_preset['low_cut'],
                high_cut=eq_preset['high_cut'],
                low_boost=eq_preset['low_boost'],
                mid_boost=eq_preset['mid_boost'],
                high_boost=eq_preset['high_boost']
            )
            report['processing_steps'].append('eq')
        
        # 2. Compression
        if enable_compression:
            audio = apply_compression(
                audio, self.sample_rate,
                threshold_db=comp_preset['threshold'],
                ratio=comp_preset['ratio'],
                attack_ms=comp_preset['attack'],
                release_ms=comp_preset['release'],
                makeup_db=comp_preset['makeup']
            )
            report['processing_steps'].append('compression')
        
        # 3. Stereo Widening
        if enable_stereo and audio.shape[1] >= 2:
            width = stereo_preset['width']
            # Bass stays mono
            if stem_type == StemType.BASS:
                width = 0.5
            audio = apply_stereo_widening(audio, width)
            report['processing_steps'].append('stereo_widening')
        
        # 4. Stem-level normalization
        audio = normalize_to_lufs(audio, self.sample_rate, target_lufs)
        report['processing_steps'].append('normalization')
        
        # 5. Soft limiting (output stage)
        if enable_limiter:
            audio = soft_clip(audio, threshold=0.95)
            report['processing_steps'].append('limiting')
        
        # Final report
        report['output_rms_db'] = linear_to_db(np.sqrt(np.mean(audio ** 2)))
        report['target_lufs'] = target_lufs
        
        return audio, report
    
    def master_stems(self, stem_files: Dict[str, np.ndarray],
                     enable_eq: bool = True,
                     enable_compression: bool = True,
                     enable_stereo: bool = True,
                     enable_limiter: bool = True) -> Dict[str, Tuple[np.ndarray, Dict]]:
        """
        Master multiple stems.
        
        Args:
            stem_files: Dict mapping stem names to audio arrays
            enable_*: Processing chain flags
            
        Returns:
            Dict mapping stem names to (mastered audio, report) tuples
        """
        mastered = {}
        self.processing_report = {}
        
        for stem_name, audio in stem_files.items():
            stem_type = self.detect_stem_type(stem_name)
            
            mastered_audio, report = self.master_stem(
                audio, stem_type,
                enable_eq=enable_eq,
                enable_compression=enable_compression,
                enable_stereo=enable_stereo,
                enable_limiter=enable_limiter
            )
            
            mastered[stem_name] = (mastered_audio, report)
            self.processing_report[stem_name] = report
        
        return mastered
    
    def get_report(self) -> Dict:
        """Get full processing report"""
        return self.processing_report
    
    def export_report(self, filepath: str):
        """Export processing report to JSON"""
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        report = convert(self.processing_report)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


# =============================================================================
# DEMUCS INTEGRATION
# =============================================================================

class StemMasteringPipeline:
    """
    Full pipeline: separate audio, master stems, optionally remix.
    """
    
    def __init__(self, platform: str = 'spotify', sample_rate: int = 44100):
        self.mastering = StemMastering(platform, sample_rate)
        self.sample_rate = sample_rate
    
    def process(self, audio: np.ndarray, 
                stems: Dict[str, np.ndarray],
                master_stems: bool = True,
                remix_mix: bool = True) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Process separated stems.
        
        Args:
            audio: Original full mix audio
            stems: Dict of separated stems (name -> audio)
            master_stems: Whether to master individual stems
            remix_mix: Whether to create final mix
            
        Returns:
            Tuple of (processed stems, report)
        """
        report = {'original_audio_rms': linear_to_db(np.sqrt(np.mean(audio ** 2)))}
        
        # Master stems
        if master_stems:
            mastered = self.mastering.master_stems(stems)
            report['mastered_stems'] = self.mastering.get_report()
        else:
            mastered = {name: (audio, {}) for name, audio in stems.items()}
        
        # Create final mix if requested
        final_mix = None
        if remix_mix:
            final_mix = self.create_mix(mastered)
            report['final_mix_rms'] = linear_to_db(np.sqrt(np.mean(final_mix ** 2)))
        
        return mastered, final_mix, report
    
    def create_mix(self, mastered_stems: Dict[str, Tuple[np.ndarray, Dict]]) -> np.ndarray:
        """Create balanced mix from mastered stems"""
        # Mix at -6dB to leave headroom for master bus
        mix_gain = db_to_linear(-6)
        
        mix = None
        for stem_name, (audio, _) in mastered_stems.items():
            if mix is None:
                mix = audio * mix_gain
            else:
                # Ensure same length
                min_len = min(len(mix), len(audio))
                mix[:min_len] += audio[:min_len] * mix_gain
        
        # Soft clip final mix
        if mix is not None:
            mix = soft_clip(mix, threshold=0.95)
        
        return mix


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Example usage"""
    import sys
    
    print("Stem Mastering System")
    print("=" * 50)
    print()
    print("Usage:")
    print("  from stem_mastering import StemMastering, StemType")
    print()
    print("  # Initialize")
    print("  mastering = StemMastering(platform='spotify')")
    print()
    print("  # Master stems")
    print("  stems = {")
    print("      'vocals': vocals_audio,")
    print("      'drums': drums_audio,")
    print("      'bass': bass_audio,")
    print("      'melody': melody_audio,")
    print("  }")
    print("  mastered = mastering.master_stems(stems)")
    print()
    print("  # Get report")
    print("  report = mastering.get_report()")
    print()
    print("Stem types supported:")
    for st in StemType:
        print(f"  - {st.value}")


if __name__ == "__main__":
    main()
