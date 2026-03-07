"""
Loudness Meter - Measures LUFS (Loudness Units Full Scale)

Uses pyloudnorm library which implements EBU R128 standard for loudness measurement.
Supports:
- Integrated loudness (overall program loudness)
- True peak measurement
- Loudness range (LRA)
"""

import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    import pyloudnorm as pyln
except ImportError as e:
    print(f"Error: Missing required package. Install with: pip install numpy soundfile pyloudnorm")
    print(f"Import error: {e}")
    sys.exit(1)


def measure_loudness(audio_path: str) -> dict:
    """
    Measure LUFS loudness metrics for an audio file.
    
    Args:
        audio_path: Path to audio file (WAV, FLAC, MP3, OGG, etc.)
    
    Returns:
        Dictionary with loudness metrics:
        - integrated_loudness: Program loudness in LUFS
        - true_peak: Maximum true peak in dBTP
        - loudness_range: Loudness range in LU
        - momentary: Momentary loudness (400ms window)
        - short_term: Short-term loudness (3s window)
    """
    # Load audio file
    data, sr = sf.read(audio_path)
    
    # Handle stereo by converting to mono for integrated loudness
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Create loudness meter (EBU R128 standard)
    meter = pyln.Meter(sr)
    
    # Measure different loudness metrics
    integrated_loudness = meter.integrated_loudness(data)
    true_peak = meter.true_peak(data)
    loudness_range = meter.loudness_range(data)
    
    # Get momentary and short-term (perceptual)
    momentary = meter.momentary(data)
    short_term = meter.short_term(data)
    
    return {
        "integrated_loudness": integrated_loudness,
        "true_peak": true_peak,
        "loudness_range": loudness_range,
        "momentary": momentary,
        "short_term": short_term,
    }


def print_loudness_report(metrics: dict, filename: str = ""):
    """Print a formatted loudness report."""
    if filename:
        print(f"\n📊 Loudness Report: {filename}")
        print("=" * 40)
    
    print(f"  Integrated Loudness: {metrics['integrated_loudness']:.1f} LUFS")
    print(f"  True Peak:           {metrics['true_peak']:.1f} dBTP")
    print(f"  Loudness Range:      {metrics['loudness_range']:.1f} LU")
    print(f"  Momentary:           {metrics['momentary']:.1f} LUFS")
    print(f"  Short-term:          {metrics['short_term']:.1f} LUFS")
    print("=" * 40)


def normalize_to_target(input_path: str, output_path: str, target_lufs: float = -14.0) -> None:
    """
    Normalize audio to target loudness.
    
    Args:
        input_path: Source audio file
        output_path: Output file path
        target_lufs: Target integrated loudness (default: -14 LUFS)
    """
    # Load audio
    data, sr = sf.read(input_path)
    
    # Handle stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Create meter and measure
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(data)
    
    # Calculate gain adjustment
    gain_db = target_lufs - loudness
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain
    normalized_data = data * gain_linear
    
    # Clip to prevent true peak > -1dBTP
    if np.max(np.abs(normalized_data)) > 0.99:
        normalized_data = np.clip(normalized_data, -0.99, 0.99)
    
    # Save
    sf.write(output_path, normalized_data, sr)
    print(f"Normalized to {target_lufs} LUFS (gain: {gain_db:.1f} dB)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Loudness Meter (LUFS)")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--normalize", "-n", type=float, metavar="LUFS",
                        help="Normalize to target loudness (default: -14 LUFS)")
    parser.add_argument("--output", "-o", help="Output file for normalized audio")
    
    args = parser.parse_args()
    
    if not Path(args.audio_file).exists():
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)
    
    # Measure and display
    metrics = measure_loudness(args.audio_file)
    print_loudness_report(metrics, args.audio_file)
    
    # Optional normalization
    if args.normalize is not None:
        output_path = args.output or Path(args.audio_file).with_stem(
            Path(args.audio_file).stem + "_normalized"
        ).with_suffix(".wav")
        normalize_to_target(args.audio_file, str(output_path), args.normalize)
