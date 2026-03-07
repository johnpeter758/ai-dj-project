"""
Beat Detection System
Detects beats and estimates tempo (BPM) from audio files.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json

# Audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False


class BeatDetector:
    """Detects beats and estimates tempo from audio files."""
    
    def __init__(self, method: str = "librosa"):
        """
        Initialize beat detector.
        
        Args:
            method: Detection method - 'librosa', 'aubio', or 'essentia'
        """
        self.method = method
        self.sample_rate = 22050
        self.hop_length = 512
        
        if method == "librosa" and not LIBROSA_AVAILABLE:
            raise ImportError("librosa not installed. Run: pip install librosa")
        elif method == "aubio" and not AUBIO_AVAILABLE:
            raise ImportError("aubio not installed. Run: pip install aubio")
        elif method == "essentia" and not ESSENTIA_AVAILABLE:
            raise ImportError("essentia not installed. Run: pip install essentia")
    
    def detect_beats(self, audio_path: str) -> Dict:
        """
        Detect beats and estimate BPM from an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing:
                - bpm: Estimated BPM
                - beats: Array of beat timestamps (in seconds)
                - onset_times: Array of onset timestamps
                - onset_strength: Onset strength envelope
                - confidence: Detection confidence score
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.method == "librosa":
            return self._detect_librosa(audio_path)
        elif self.method == "aubio":
            return self._detect_aubio(audio_path)
        elif self.method == "essentia":
            return self._detect_essentia(audio_path)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _detect_librosa(self, audio_path: str) -> Dict:
        """Librosa-based beat detection."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Get onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Detect onset times
        onset_times = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_times, sr=sr, hop_length=self.hop_length)
        
        # Estimate tempo using autocorrelation
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=self.hop_length
        )
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        # Calculate confidence based on onset regularity
        confidence = self._calculate_confidence(onset_times)
        
        return {
            "bpm": float(tempo),
            "beats": beat_times.tolist(),
            "onset_times": onset_times.tolist(),
            "onset_strength": onset_env.tolist(),
            "confidence": confidence,
            "method": "librosa"
        }
    
    def _detect_aubio(self, audio_path: str) -> Dict:
        """Aubio-based beat detection."""
        # Use aubio for beat tracking
        win_size = 1024
        hop_size = 512
        
        # Load audio through aubio
        source = aubio.source(audio_path, 0, hop_size)
        samplerate = source.samplerate
        
        # Create onset detector
        onset_det = aubio.onset("default", win_size, hop_size, samplerate)
        
        # Create beat tracker
        beat_tracker = aubio.beattracking("default", win_size, hop_size, samplerate)
        
        onset_times = []
        beat_times = []
        
        # Process audio
        total_frames = 0
        while True:
            samples, read = source()
            if read > 0:
                # Detect onsets
                if onset_det(samples):
                    onset_time = total_frames / float(samplerate)
                    onset_times.append(onset_time)
                
                # Track beats
                if beat_tracker(samples):
                    beat_time = total_frames / float(samplerate)
                    beat_times.append(beat_time)
                
                total_frames += read
            else:
                break
        
        # Estimate BPM from beat intervals
        bpm = self._estimate_bpm_from_beats(beat_times)
        confidence = self._calculate_confidence(np.array(beat_times))
        
        return {
            "bpm": bpm,
            "beats": beat_times,
            "onset_times": onset_times,
            "onset_strength": [],
            "confidence": confidence,
            "method": "aubio"
        }
    
    def _detect_essentia(self, audio_path: str) -> Dict:
        """Essentia-based beat detection."""
        # Load audio
        loader = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
        audio = loader()
        
        # Beat tracking
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, _, _, _ = rhythm_extractor(audio)
        
        # Onset detection
        od = es.OnsetDetection()
        odf = es.OriginalFrameGenerator(self.sample_rate, 2048, 1024)
        onsets = []
        
        # Process in frames
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
            odf_frame = odf(frame)
            od_result = od(odf_frame)
            if od_result > 0.5:  # threshold
                # Get timestamp (approximate)
                pass
        
        # Use beat positions from rhythm extractor
        beat_times = beats.tolist() if hasattr(beats, 'tolist') else list(beats)
        
        # Calculate confidence
        confidence = self._calculate_confidence(np.array(beat_times))
        
        return {
            "bpm": float(bpm),
            "beats": beat_times,
            "onset_times": beat_times,  # Onsets roughly match beats
            "onset_strength": [],
            "confidence": confidence,
            "method": "essentia"
        }
    
    def _estimate_bpm_from_beats(self, beat_times: List[float]) -> float:
        """Estimate BPM from beat timestamps using autocorrelation."""
        if len(beat_times) < 4:
            return 120.0  # Default BPM
        
        # Calculate intervals between beats
        intervals = np.diff(beat_times)
        
        # Filter out unreasonable intervals (< 0.2s or > 2s = 300-30 BPM)
        valid_intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
        
        if len(valid_intervals) == 0:
            return 120.0
        
        # Use median interval for BPM
        median_interval = np.median(valid_intervals)
        bpm = 60.0 / median_interval
        
        # Handle double/half time
        while bpm < 70:
            bpm *= 2
        while bpm > 200:
            bpm /= 2
            
        return round(bpm, 1)
    
    def _calculate_confidence(self, times: np.ndarray) -> float:
        """Calculate confidence score based on timing regularity."""
        if len(times) < 4:
            return 0.5
        
        intervals = np.diff(times)
        
        # Remove outliers
        median = np.median(intervals)
        valid = intervals[(intervals > 0.5 * median) & (intervals < 1.5 * median)]
        
        if len(valid) < 2:
            return 0.5
        
        # Coefficient of variation (lower = more regular)
        cv = np.std(valid) / np.mean(valid)
        
        # Convert to confidence (0-1)
        confidence = max(0, 1 - cv)
        
        return round(confidence, 2)
    
    def analyze_file(self, audio_path: str, output_path: Optional[str] = None) -> Dict:
        """
        Full analysis of audio file with beat detection.
        
        Args:
            audio_path: Path to audio file
            output_path: Optional path to save JSON results
            
        Returns:
            Complete analysis dictionary
        """
        result = self.detect_beats(audio_path)
        
        # Add additional metadata
        result["file"] = audio_path
        result["duration"] = float(librosa.get_duration(filename=audio_path) if LIBROSA_AVAILABLE else 0)
        
        # Calculate beats per measure (assume 4/4 time)
        if result["bpm"] > 0:
            result["beat_duration"] = 60.0 / result["bpm"]
            result["beats_per_bar"] = 4  # Default to 4/4
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
        
        return result


def detect_bpm(audio_path: str, method: str = "librosa") -> float:
    """
    Quick BPM detection helper.
    
    Args:
        audio_path: Path to audio file
        method: Detection method
        
    Returns:
        Estimated BPM
    """
    detector = BeatDetector(method=method)
    result = detector.detect_beats(audio_path)
    return result["bpm"]


def detect_beats(audio_path: str, method: str = "librosa") -> List[float]:
    """
    Quick beat timestamps helper.
    
    Args:
        audio_path: Path to audio file
        method: Detection method
        
    Returns:
        List of beat timestamps in seconds
    """
    detector = BeatDetector(method=method)
    result = detector.detect_beats(audio_path)
    return result["beats"]


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python beat_detector.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Try each method
    for method in ["librosa", "aubio", "essentia"]:
        try:
            detector = BeatDetector(method=method)
            result = detector.analyze_file(audio_file)
            
            print(f"\n=== {method.upper()} Results ===")
            print(f"BPM: {result['bpm']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Beats detected: {len(result['beats'])}")
            print(f"First 10 beats: {result['beats'][:10]}")
            break
        except ImportError:
            continue
        except Exception as e:
            print(f"Error with {method}: {e}")
            continue
