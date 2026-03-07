"""
Genre Classifier for AI DJ Project

Classifies music genres using audio analysis and metadata extraction.
Works with the genre_db.py module for genre definitions.
"""

import json
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Try to import audio libraries, fallback to metadata-only mode
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import mutagen
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

from genre_db import GENRES, Genre, EnergyLevel, Mood


@dataclass
class AudioFeatures:
    """Extracted audio features for classification."""
    bpm: float
    key: str
    energy: float  # 0.0 to 1.0
    danceability: float  # 0.0 to 1.0
    bass_intensity: float  # 0.0 to 1.0
    vocal_present: float  # 0.0 to 1.0
    spectral_centroid: float  # brightness
    zero_crossing_rate: float  # noisiness
    tempo_stability: float  # how steady the tempo is
    dynamic_range: float


@dataclass
class ClassificationResult:
    """Result of genre classification."""
    genre: str
    confidence: float
    alternative_genres: List[Tuple[str, float]]  # (genre, confidence)
    features: AudioFeatures
    method: str  # "audio", "metadata", "hybrid"


class GenreClassifier:
    """
    Classifies music genres using audio analysis and metadata.
    
    Supports three modes:
    - audio: Full audio analysis (requires librosa)
    - metadata: Genre extraction from file metadata
    - hybrid: Combines both for best accuracy
    """
    
    # Key detection mapping
    KEY_PITCHES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Genre fingerprint weights
    FINGERPRINT_WEIGHTS = {
        'bpm_range': 0.25,
        'energy': 0.20,
        'danceability': 0.15,
        'bass_intensity': 0.15,
        'vocal_present': 0.10,
        'key': 0.08,
        'spectral': 0.07,
    }
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_path(self, file_path: str) -> Path:
        """Get cache file path for a track."""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
        return self.cache_dir / f"genre_{file_hash}.json"
    
    def _extract_metadata(self, file_path: str) -> Dict:
        """Extract genre and metadata from audio file tags."""
        if not MUTAGEN_AVAILABLE:
            return {}
        
        try:
            audio = mutagen.File(file_path)
            if audio is None:
                return {}
            
            metadata = {}
            
            # Extract genre from tags
            if hasattr(audio, 'tags'):
                for key in ['genre', 'GENRE', 'TCON']:
                    if key in audio.tags:
                        genre_tags = audio.tags[key]
                        if genre_tags:
                            metadata['tag_genre'] = str(genre_tags[0])
                            break
            
            # Extract BPM if available
            if hasattr(audio, 'info') and hasattr(audio.info, 'bitrate'):
                metadata['bitrate'] = audio.info.bitrate
            
            # Duration
            if hasattr(audio.info, 'length'):
                metadata['duration'] = audio.info.length
            
            return metadata
        except Exception:
            return {}
    
    def _detect_bpm(self, y: np.ndarray, sr: int) -> float:
        """Detect BPM using onset strength envelope."""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, _ = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            return float(tempo)
        except Exception:
            return 120.0  # Default BPM
    
    def _detect_key(self, y: np.ndarray, sr: int) -> str:
        """Detect musical key using chroma features."""
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1)
            key_idx = chroma_mean.argmax()
            return self.KEY_PITCHES[key_idx]
        except Exception:
            return "C"  # Default key
    
    def _analyze_audio(self, file_path: str) -> AudioFeatures:
        """Extract comprehensive audio features."""
        if not LIBROSA_AVAILABLE:
            return self._default_features()
        
        try:
            # Load audio (limit to 30 seconds for speed)
            y, sr = librosa.load(file_path, duration=30, mono=True)
            
            # BPM detection
            bpm = self._detect_bpm(y, sr)
            
            # Key detection
            key = self._detect_key(y, sr)
            
            # Energy (RMS amplitude)
            rms = librosa.feature.rms(y=y)[0]
            energy = float(np.clip(rms.mean() * 10, 0, 1))
            
            # Danceability (based on beat regularity and tempo)
            tempo, _ = librosa.beat.tempo(y=y, sr=sr)
            danceability = self._calculate_danceability(y, sr, tempo)
            
            # Bass intensity (low frequency energy)
            bass = librosa.feature.spectral_bandwidth(y=y, sr=sr, p=1)[0]
            bass_intensity = float(np.clip(1 - (bass.mean() / 2000), 0, 1))
            
            # Vocal presence detection
            vocal_present = self._detect_vocals(y, sr)
            
            # Spectral centroid (brightness)
            spectral = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid = float(spectral.mean() / 4000)  # Normalize
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zero_crossing_rate = float(zcr.mean())
            
            # Tempo stability
            tempo_stability = self._calculate_tempo_stability(y, sr)
            
            # Dynamic range
            dynamic_range = float(np.clip((rms.max() - rms.min()) / (rms.max() + 1e-6), 0, 1))
            
            return AudioFeatures(
                bpm=bpm,
                key=key,
                energy=energy,
                danceability=danceability,
                bass_intensity=bass_intensity,
                vocal_present=vocal_present,
                spectral_centroid=spectral_centroid,
                zero_crossing_rate=zero_crossing_rate,
                tempo_stability=tempo_stability,
                dynamic_range=dynamic_range
            )
        except Exception as e:
            print(f"Audio analysis error: {e}")
            return self._default_features()
    
    def _calculate_danceability(self, y: np.ndarray, sr: int, tempo: float) -> float:
        """Calculate danceability score based on rhythm characteristics."""
        try:
            # Get onset envelope
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Danceability: combine tempo suitability and beat regularity
            # Ideal dance tempo is 120-130 BPM
            tempo_score = 1.0 - abs(tempo - 125) / 60
            tempo_score = max(0, min(1, tempo_score))
            
            # Beat regularity
            beat_scores = []
            for _ in range(5):
                onset_sub = onset_env[10 * _:10 * _ + 20]
                if len(onset_sub) > 5:
                    regularity = np.std(onset_sub) / (np.mean(onset_sub) + 1e-6)
                    beat_scores.append(1 / (1 + regularity))
            
            beat_regularity = np.mean(beat_scores) if beat_scores else 0.5
            
            return (tempo_score * 0.6 + beat_regularity * 0.4)
        except Exception:
            return 0.5
    
    def _detect_vocals(self, y: np.ndarray, sr: int) -> float:
        """Detect vocal presence using spectral characteristics."""
        try:
            # Vocals tend to have strong mid-frequency content
            # and specific harmonic patterns
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
            
            # Vocal detection heuristic
            vocal_score = 0.0
            
            # MFCC variance (vocals have distinctive MFCC patterns)
            mfcc_variance = np.var(mfcc, axis=1).mean()
            vocal_score += min(mfcc_variance / 50, 1.0) * 0.5
            
            # Harmonic to percussive ratio
            y_harm, y_perc = librosa.decompose.hpss(librosa.stft(y))
            harmonic_energy = np.sum(np.abs(y_harm) ** 2)
            percussive_energy = np.sum(np.abs(y_perc) ** 2)
            if harmonic_energy + percussive_energy > 0:
                h_p_ratio = harmonic_energy / (harmonic_energy + percussive_energy)
                vocal_score += h_p_ratio * 0.5
            
            return float(np.clip(vocal_score, 0, 1))
        except Exception:
            return 0.3
    
    def _calculate_tempo_stability(self, y: np.ndarray, sr: int) -> float:
        """Calculate how stable the tempo is throughout the track."""
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo, beats = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, units='frames')
            
            if len(beats) < 10:
                return 0.5
            
            # Analyze tempo variations
            beat_times = librosa.frames_to_time(beats, sr=sr)
            intervals = np.diff(beat_times)
            
            if len(intervals) > 0:
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                stability = 1 / (1 + (interval_std / interval_mean) * 10)
                return float(np.clip(stability, 0, 1))
            
            return 0.5
        except Exception:
            return 0.5
    
    def _default_features(self) -> AudioFeatures:
        """Return default features when audio analysis is unavailable."""
        return AudioFeatures(
            bpm=120.0,
            key="C",
            energy=0.5,
            danceability=0.5,
            bass_intensity=0.5,
            vocal_present=0.3,
            spectral_centroid=0.5,
            zero_crossing_rate=0.3,
            tempo_stability=0.7,
            dynamic_range=0.5
        )
    
    def _calculate_genre_score(self, features: AudioFeatures, genre: Genre) -> float:
        """Calculate similarity score between audio features and genre."""
        score = 0.0
        
        # BPM match (within genre's typical range)
        bpm_min, bpm_max = genre.bpm_range
        if bpm_min <= features.bpm <= bpm_max:
            bpm_score = 1.0
        else:
            bpm_diff = min(abs(features.bpm - bpm_min), abs(features.bpm - bpm_max))
            bpm_score = max(0, 1 - bpm_diff / 50)
        score += bpm_score * self.FINGERPRINT_WEIGHTS['bpm_range']
        
        # Energy match
        energy_target = {
            EnergyLevel.LOW: 0.3,
            EnergyLevel.MEDIUM: 0.5,
            EnergyLevel.HIGH: 0.8,
            EnergyLevel.BUILDUP: 0.7,
            EnergyLevel.DROPCENTER: 0.9,
        }.get(genre.energy, 0.5)
        energy_diff = abs(features.energy - energy_target)
        score += (1 - energy_diff) * self.FINGERPRINT_WEIGHTS['energy']
        
        # Danceability match
        dance_diff = abs(features.danceability - genre.danceability)
        score += (1 - dance_diff) * self.FINGERPRINT_WEIGHTS['danceability']
        
        # Bass intensity match
        bass_diff = abs(features.bass_intensity - genre.bass_intensity)
        score += (1 - bass_diff) * self.FINGERPRINT_WEIGHTS['bass_intensity']
        
        # Vocal presence match
        vocal_diff = abs(features.vocal_present - genre.vocal_present)
        score += (1 - vocal_diff) * self.FINGERPRINT_WEIGHTS['vocal_present']
        
        # Key compatibility
        if features.key in genre.common_keys:
            score += self.FINGERPRINT_WEIGHTS['key']
        
        # Spectral characteristics
        if features.spectral_centroid > 0.6:
            if 'bright' in genre.tags or 'energetic' in genre.tags:
                score += self.FINGERPRINT_WEIGHTS['spectral'] * 0.5
        
        return score
    
    def _match_metadata_genre(self, metadata_genre: str) -> Optional[str]:
        """Match metadata genre string to database genre."""
        if not metadata_genre:
            return None
        
        metadata_genre = metadata_genre.lower().strip()
        
        # Direct matches
        for genre_name in GENRES.keys():
            if metadata_genre == genre_name.replace('_', ' '):
                return genre_name
        
        # Fuzzy matching
        for genre_name, genre in GENRES.items():
            if metadata_genre in genre_name.replace('_', ' '):
                return genre_name
            if any(metadata_genre in tag for tag in genre.tags):
                return genre_name
            if any(metadata_genre in sim for sim in genre.similar_genres):
                return genre_name
        
        # Common genre aliases
        aliases = {
            'edm': 'house',
            'electronic': 'house',
            'dance': 'house',
            'dubstep': 'dubstep',
            'dnb': 'drum_and_bass',
            'drumnbass': 'drum_and_bass',
            'jungle': 'drum_and_bass',
            'hip hop': 'hip_hop',
            'hiphop': 'hip_hop',
            'r&b': 'rnb',
            'rn b': 'rnb',
            'pop': 'pop',
            'rock': 'rock',
            'metal': 'metal',
            'classical': 'classical',
            'jazz': 'jazz',
            'blues': 'blues',
            'country': 'country',
            'folk': 'folk',
            'ambient': 'ambient',
            'chill': 'chillout',
            'lounge': 'chillout',
        }
        
        for alias, genre_name in aliases.items():
            if alias in metadata_genre:
                return genre_name
        
        return None
    
    def classify(self, file_path: str, method: str = "hybrid") -> ClassificationResult:
        """
        Classify the genre of an audio file.
        
        Args:
            file_path: Path to audio file
            method: Classification method - "audio", "metadata", or "hybrid"
        
        Returns:
            ClassificationResult with genre, confidence, and features
        """
        # Check cache
        cache_path = self._get_cache_path(file_path)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                    return ClassificationResult(
                        genre=cached['genre'],
                        confidence=cached['confidence'],
                        alternative_genres=cached['alternatives'],
                        features=AudioFeatures(**cached['features']),
                        method=cached['method']
                    )
            except Exception:
                pass
        
        # Extract metadata
        metadata = self._extract_metadata(file_path)
        metadata_genre = metadata.get('tag_genre')
        
        # Match metadata genre to database
        matched_genre = self._match_metadata_genre(metadata_genre) if metadata_genre else None
        
        if method == "metadata":
            features = self._default_features()
            if matched_genre:
                result = ClassificationResult(
                    genre=matched_genre,
                    confidence=0.7,
                    alternative_genres=[],
                    features=features,
                    method="metadata"
                )
            else:
                result = ClassificationResult(
                    genre="house",  # Default
                    confidence=0.3,
                    alternative_genres=[],
                    features=features,
                    method="metadata"
                )
        else:
            # Audio analysis
            features = self._analyze_audio(file_path)
            
            # Calculate scores for all genres
            scores = {}
            for genre_name, genre in GENRES.items():
                scores[genre_name] = self._calculate_genre_score(features, genre)
            
            # Sort by score
            sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Boost score if metadata matches
            if matched_genre and matched_genre in scores:
                scores[matched_genre] = scores[matched_genre] * 1.2  # 20% boost
                sorted_genres = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            top_genre = sorted_genres[0][0]
            top_score = sorted_genres[0][1]
            
            # Normalize confidence
            max_possible = sum(self.FINGERPRINT_WEIGHTS.values())
            confidence = min(top_score / max_possible, 1.0)
            
            # Get alternatives
            alternatives = [(g, s / max_possible) for g, s in sorted_genres[1:4]]
            
            result = ClassificationResult(
                genre=top_genre,
                confidence=confidence,
                alternative_genres=alternatives,
                features=features,
                method=method if method != "hybrid" else "audio"
            )
        
        # Cache result
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'genre': result.genre,
                    'confidence': result.confidence,
                    'alternatives': result.alternative_genres,
                    'features': {
                        'bpm': result.features.bpm,
                        'key': result.features.key,
                        'energy': result.features.energy,
                        'danceability': result.features.danceability,
                        'bass_intensity': result.features.bass_intensity,
                        'vocal_present': result.features.vocal_present,
                        'spectral_centroid': result.features.spectral_centroid,
                        'zero_crossing_rate': result.features.zero_crossing_rate,
                        'tempo_stability': result.features.tempo_stability,
                        'dynamic_range': result.features.dynamic_range,
                    },
                    'method': result.method,
                }, f, indent=2)
        except Exception:
            pass
        
        return result
    
    def classify_batch(self, file_paths: List[str], method: str = "hybrid") -> List[ClassificationResult]:
        """Classify multiple files."""
        return [self.classify(fp, method) for fp in file_paths]
    
    def get_compatible_genres(self, genre: str) -> List[str]:
        """Get genres compatible for mixing with given genre."""
        if genre in GENRES:
            return GENRES[genre].compatible_genres
        return list(GENRES.keys())[:5]  # Return some defaults
    
    def get_similar_genres(self, genre: str) -> List[str]:
        """Get similar genres to the given genre."""
        if genre in GENRES:
            return GENRES[genre].similar_genres
        return []


def main():
    """CLI for genre classification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify music genres")
    parser.add_argument("file", help="Audio file to classify")
    parser.add_argument("--method", choices=["audio", "metadata", "hybrid"], default="hybrid")
    parser.add_argument("--show-features", action="store_true", help="Show extracted features")
    
    args = parser.parse_args()
    
    classifier = GenreClassifier()
    result = classifier.classify(args.file, args.method)
    
    print(f"\n🎵 Genre Classification Results")
    print(f"=" * 40)
    print(f"Genre: {result.genre}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Method: {result.method}")
    
    if result.alternative_genres:
        print(f"\nAlternatives:")
        for genre, conf in result.alternative_genres:
            print(f"  - {genre}: {conf:.1%}")
    
    if args.show_features:
        f = result.features
        print(f"\n📊 Audio Features:")
        print(f"  BPM: {f.bpm:.1f}")
        print(f"  Key: {f.key}")
        print(f"  Energy: {f.energy:.2f}")
        print(f"  Danceability: {f.danceability:.2f}")
        print(f"  Bass Intensity: {f.bass_intensity:.2f}")
        print(f"  Vocal Presence: {f.vocal_present:.2f}")
        print(f"  Spectral Centroid: {f.spectral_centroid:.2f}")


if __name__ == "__main__":
    main()
