"""
Mood Detector for Music
Detects emotional mood from audio files using audio feature analysis.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

# Audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import essentia
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False


# Mood categories with associated musical characteristics
MOOD_CATEGORIES = {
    "happy": {
        "energy": "high",
        "valence": "high",
        "danceability": "high",
        "tempo_range": (100, 150),
        "description": "Upbeat, cheerful, major key feel"
    },
    "sad": {
        "energy": "low",
        "valence": "low",
        "danceability": "low",
        "tempo_range": (60, 100),
        "description": "Melancholic, emotional, minor key feel"
    },
    "energetic": {
        "energy": "high",
        "valence": "medium",
        "danceability": "high",
        "tempo_range": (120, 180),
        "description": "High energy, driving, powerful"
    },
    "calm": {
        "energy": "low",
        "valence": "medium",
        "danceability": "low",
        "tempo_range": (60, 100),
        "description": "Relaxed, peaceful, ambient"
    },
    "angry": {
        "energy": "high",
        "valence": "low",
        "danceability": "medium",
        "tempo_range": (100, 160),
        "description": "Intense, aggressive, distorted"
    },
    "romantic": {
        "energy": "low",
        "valence": "high",
        "danceability": "medium",
        "tempo_range": (70, 110),
        "description": "Love, intimacy, smooth"
    },
    "mysterious": {
        "energy": "medium",
        "valence": "low",
        "danceability": "low",
        "tempo_range": (60, 100),
        "description": "Dark, atmospheric, suspenseful"
    },
    "uplifting": {
        "energy": "medium",
        "valence": "high",
        "danceability": "high",
        "tempo_range": (100, 140),
        "description": "Inspiring, hopeful, positive"
    }
}


class MoodDetector:
    """Detects musical mood from audio files."""
    
    def __init__(self, method: str = "librosa"):
        """
        Initialize mood detector.
        
        Args:
            method: Analysis method - 'librosa' or 'essentia'
        """
        self.method = method
        self.sample_rate = 22050
        self.hop_length = 512
        
        if method == "librosa" and not LIBROSA_AVAILABLE:
            raise ImportError("librosa not installed. Run: pip install librosa")
        elif method == "essentia" and not ESSENTIA_AVAILABLE:
            raise ImportError("essentia not installed. Run: pip install essentia")
    
    def detect_mood(self, audio_path: str, duration: Optional[float] = None) -> Dict:
        """
        Detect mood from an audio file.
        
        Args:
            audio_path: Path to audio file
            duration: Optional duration to analyze (seconds from start)
            
        Returns:
            Dictionary containing:
                - primary_mood: Dominant mood category
                - secondary_mood: Secondary mood (if detected)
                - moods: List of all moods with confidence scores
                - features: Extracted audio features
                - valence: Estimated valence (0-1, sad to happy)
                - energy: Estimated energy level (0-1)
                - danceability: Estimated danceability (0-1)
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.method == "librosa":
            return self._detect_librosa(audio_path, duration)
        elif self.method == "essentia":
            return self._detect_essentia(audio_path, duration)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _detect_librosa(self, audio_path: str, duration: Optional[float] = None) -> Dict:
        """Librosa-based mood detection."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
        
        # Extract features
        features = self._extract_features_librosa(y, sr)
        
        # Calculate mood scores
        mood_scores = self._calculate_mood_scores(features)
        
        # Sort moods by score
        sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_mood": sorted_moods[0][0],
            "secondary_mood": sorted_moods[1][0] if len(sorted_moods) > 1 else None,
            "moods": [{"mood": m, "confidence": round(s, 3)} for m, s in sorted_moods],
            "features": features,
            "valence": round(features["valence"], 3),
            "energy": round(features["energy"], 3),
            "danceability": round(features["danceability"], 3),
            "tempo": round(features["tempo"], 1)
        }
    
    def _extract_features_librosa(self, y: np.ndarray, sr: int) -> Dict:
        """Extract audio features using librosa."""
        # Compute STFT
        S = np.abs(librosa.stft(y, hop_length=self.hop_length))
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=S))
        
        # RMS energy (overall amplitude)
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Chroma features (for key detection)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Major/minor key detection based on chroma
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
        is_major = major_corr > minor_corr
        
        # MFCCs for timbre
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Normalize features to 0-1 range
        energy = min(1.0, rms * 10)
        danceability = min(1.0, (spectral_contrast / 7.0) * 0.5 + (zcr * 10) * 0.3 + (tempo / 180) * 0.2)
        
        # Valence: major key + higher energy = more positive
        valence = 0.5
        if is_major:
            valence += 0.3
        valence += energy * 0.2
        valence = min(1.0, max(0.0, valence))
        
        return {
            "tempo": float(tempo),
            "energy": float(energy),
            "valence": float(valence),
            "danceability": float(danceability),
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth),
            "spectral_rolloff": float(spectral_rolloff),
            "spectral_contrast": float(spectral_contrast),
            "zero_crossing_rate": float(zcr),
            "is_major": bool(is_major),
            "mfccs": mfccs_mean.tolist()
        }
    
    def _detect_essentia(self, audio_path: str, duration: Optional[float] = None) -> Dict:
        """Essentia-based mood detection."""
        # Load audio
        loader = es.MonoLoader(filename=audio_path, sampleRate=self.sample_rate)
        if duration:
            # Approximate frame loading by duration
            pass
        y = loader()
        
        # Extract features using Essentia
        features = {}
        
        # Rhythm features
        rhythm = es.RhythmExtractor2013()(y)
        features["tempo"] = rhythm[0]
        
        # Energy features
        energy = es.Energy()(y)
        features["energy"] = min(1.0, energy / 10.0)
        
        # Danceability
        danceable = es.Danceability()(y)
        features["danceability"] = float(danceable["danceability"])
        
        # Spectral features
        spec = es.SpectralCentroid()(y)
        features["spectral_centroid"] = float(spec)
        
        # Key detection
        key = es.Key()(y)
        features["key"] = key["key"]
        features["scale"] = key["scale"]
        features["is_major"] = key["scale"] == "major"
        
        # Valence estimation
        # Higher energy + major key = higher valence
        valence = 0.5
        if features["is_major"]:
            valence += 0.3
        valence += features["energy"] * 0.2
        features["valence"] = min(1.0, max(0.0, valence))
        
        # Calculate mood scores
        mood_scores = self._calculate_mood_scores(features)
        sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_mood": sorted_moods[0][0],
            "secondary_mood": sorted_moods[1][0] if len(sorted_moods) > 1 else None,
            "moods": [{"mood": m, "confidence": round(s, 3)} for m, s in sorted_moods],
            "features": features,
            "valence": round(features["valence"], 3),
            "energy": round(features["energy"], 3),
            "danceability": round(features["danceability"], 3),
            "tempo": round(features["tempo"], 1)
        }
    
    def _calculate_mood_scores(self, features: Dict) -> Dict[str, float]:
        """Calculate confidence scores for each mood category."""
        scores = {}
        
        tempo = features.get("tempo", 100)
        energy = features.get("energy", 0.5)
        valence = features.get("valence", 0.5)
        danceability = features.get("danceability", 0.5)
        
        for mood, criteria in MOOD_CATEGORIES.items():
            score = 0.0
            
            # Energy score
            if criteria["energy"] == "high":
                score += energy * 0.35
            elif criteria["energy"] == "low":
                score += (1 - energy) * 0.35
            else:
                score += (1 - abs(energy - 0.5) * 2) * 0.35
            
            # Valence score
            if criteria["valence"] == "high":
                score += valence * 0.25
            elif criteria["valence"] == "low":
                score += (1 - valence) * 0.25
            else:
                score += (1 - abs(valence - 0.5) * 2) * 0.25
            
            # Danceability score
            if criteria["danceability"] == "high":
                score += danceability * 0.2
            elif criteria["danceability"] == "low":
                score += (1 - danceability) * 0.2
            else:
                score += (1 - abs(danceability - 0.5) * 2) * 0.2
            
            # Tempo score
            min_tempo, max_tempo = criteria["tempo_range"]
            if min_tempo <= tempo <= max_tempo:
                score += 0.2
            else:
                dist = min(abs(tempo - min_tempo), abs(tempo - max_tempo))
                score += max(0, 0.2 - dist / 100)
            
            scores[mood] = score
        
        # Normalize scores to 0-1
        max_score = max(scores.values()) if scores else 1
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def get_mood_description(self, mood: str) -> str:
        """Get description for a mood category."""
        return MOOD_CATEGORIES.get(mood, {}).get("description", "Unknown mood")
    
    def get_all_moods(self) -> List[str]:
        """Get list of all supported mood categories."""
        return list(MOOD_CATEGORIES.keys())


def batch_detect_mood(audio_paths: List[str], method: str = "librosa") -> List[Dict]:
    """
    Detect mood for multiple audio files.
    
    Args:
        audio_paths: List of paths to audio files
        method: Analysis method
        
    Returns:
        List of mood detection results
    """
    detector = MoodDetector(method=method)
    results = []
    
    for path in audio_paths:
        try:
            result = detector.detect_mood(path)
            result["file"] = path
            results.append(result)
        except Exception as e:
            results.append({"file": path, "error": str(e)})
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mood_detector.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    detector = MoodDetector()
    result = detector.detect_mood(audio_file)
    
    print(f"\n🎵 Mood Detection Results for: {audio_file}")
    print(f"   Primary Mood: {result['primary_mood'].upper()}")
    print(f"   Secondary Mood: {result['secondary_mood']}")
    print(f"   Valence: {result['valence']} (sad ← → happy)")
    print(f"   Energy: {result['energy']}")
    print(f"   Danceability: {result['danceability']}")
    print(f"   Tempo: {result['tempo']} BPM")
    print(f"\n   All moods:")
    for m in result["moods"]:
        print(f"     - {m['mood']}: {m['confidence']:.2%}")
