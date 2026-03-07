#!/usr/bin/env python3
"""
Quality Evaluator - Audio Quality Analysis for AI-Generated Music

Analyzes audio files for:
- LUFS (integrated loudness)
- Peak levels
- Dynamic range
- Creativity scoring
- Improvement suggestions
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class AudioMetrics:
    """Audio measurement results"""
    integrated_lufs: float = -70.0  # LUFS integrated loudness
    true_peak_db: float = -70.0     # True peak in dB
    dynamic_range: float = 0.0      # DR (dynamic range) in dB
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 2
    duration_seconds: float = 0.0


@dataclass
class CreativityScore:
    """Creativity assessment results"""
    overall: float = 0.0           # 0-100 overall score
    originality: float = 0.0       # 0-100
    arrangement: float = 0.0       # 0-100
    harmonic_richness: float = 0.0 # 0-100
    rhythmic_variety: float = 0.0 # 0-100
    production_quality: float = 0.0  # 0-100


@dataclass
class QualitySuggestions:
    """Improvement suggestions"""
    critical: list[str] = field(default_factory=list)
    recommended: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Complete quality evaluation report"""
    file_path: str
    audio_metrics: AudioMetrics
    creativity: CreativityScore
    suggestions: QualitySuggestions
    overall_score: float = 0.0
    grade: str = "F"
    timestamp: str = ""


class QualityEvaluator:
    """
    Evaluates audio quality for AI-generated music.
    
    Supports multiple analysis methods:
    - File-based analysis (requires audio library)
    - Metadata-based analysis (for generated tracks)
    - Heuristic analysis (based on generation parameters)
    """
    
    # LUFS targets for different contexts
    LUFS_TARGETS = {
        "streaming": -14.0,   # Spotify, Apple Music
        "broadcast": -24.0,  # EBU R128
        "mastering": -12.0,   # Loud but not crushed
        "ambient": -20.0,     # More dynamic
    }
    
    # Peak thresholds
    PEAK_THRESHOLDS = {
        "optimal": -1.0,      # Slightly below 0dB
        "acceptable": -0.3,   # Just below digital max
        "warning": 0.0,       # At or above 0dB (clipping)
    }
    
    # Dynamic range ratings
    DR_RATINGS = {
        "excellent": 14.0,    # High dynamic range
        "good": 10.0,
        "fair": 6.0,
        "poor": 3.0,
    }
    
    def __init__(self, target_platform: str = "streaming"):
        """
        Initialize evaluator.
        
        Args:
            target_platform: Target loudness standard ('streaming', 'broadcast', 'mastering', 'ambient')
        """
        self.target_platform = target_platform
        self.target_lufs = self.LUFS_TARGETS.get(target_platform, -14.0)
        self._audio_lib_available = self._check_audio_library()
    
    def _check_audio_library(self) -> bool:
        """Check if audio analysis library is available"""
        try:
            import librosa
            return True
        except ImportError:
            pass
        
        try:
            import pydub
            return True
        except ImportError:
            pass
        
        return False
    
    def evaluate_file(self, file_path: str) -> QualityReport:
        """
        Evaluate an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            QualityReport with complete analysis
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Try to load and analyze with library
        if self._audio_lib_available:
            metrics = self._analyze_audio_file(file_path)
        else:
            # Fallback to basic file analysis
            metrics = self._basic_file_analysis(file_path)
        
        # Score creativity (would use ML model in production)
        creativity = self._score_creativity(file_path, metrics)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(metrics, creativity)
        
        # Calculate overall score
        overall = self._calculate_overall_score(metrics, creativity)
        grade = self._get_grade(overall)
        
        from datetime import datetime
        
        return QualityReport(
            file_path=file_path,
            audio_metrics=metrics,
            creativity=creativity,
            suggestions=suggestions,
            overall_score=overall,
            grade=grade,
            timestamp=datetime.now().isoformat()
        )
    
    def evaluate_metadata(self, metadata: dict) -> QualityReport:
        """
        Evaluate based on generation metadata (for AI-generated tracks).
        
        Args:
            metadata: Dict with generation parameters
            
        Returns:
            QualityReport with estimated quality
        """
        metrics = self._estimate_from_metadata(metadata)
        creativity = self._score_from_metadata(metadata)
        suggestions = self._generate_suggestions(metrics, creativity)
        overall = self._calculate_overall_score(metrics, creativity)
        grade = self._get_grade(overall)
        
        from datetime import datetime
        
        return QualityReport(
            file_path=metadata.get("file_path", "unknown"),
            audio_metrics=metrics,
            creativity=creativity,
            suggestions=suggestions,
            overall_score=overall,
            grade=grade,
            timestamp=datetime.now().isoformat()
        )
    
    def _analyze_audio_file(self, file_path: str) -> AudioMetrics:
        """Analyze audio file using available library"""
        metrics = AudioMetrics()
        
        try:
            import librosa
            import numpy as np
            
            # Load audio
            y, sr = librosa.load(file_path, sr=None)
            
            # Get basic info
            metrics.sample_rate = sr
            metrics.channels = 2 if len(y.shape) > 1 else 1
            metrics.duration_seconds = float(len(y)) / sr
            
            # Convert to mono for analysis
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Calculate true peak
            samples = np.abs(y)
            max_sample = np.max(samples)
            metrics.true_peak_db = 20 * np.log10(max_sample) if max_sample > 0 else -70.0
            
            # Calculate LUFS (approximation using RMS)
            # Real implementation would use pyloudnorm or similar
            frame_length = int(0.4 * sr)  # 400ms blocks
            hop_length = int(0.1 * sr)    # 100ms overlap
            
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = 20 * np.log10(rms + 1e-10)
            
            # Mean RMS (approximate integrated LUFS)
            metrics.integrated_lufs = float(np.mean(rms_db))
            
            # Dynamic range (approximate)
            metrics.dynamic_range = float(np.max(rms_db) - np.min(rms_db))
            
        except Exception as e:
            # Fallback to basic analysis
            return self._basic_file_analysis(file_path)
        
        return metrics
    
    def _basic_file_analysis(self, file_path: str) -> AudioMetrics:
        """Basic file analysis without audio library"""
        metrics = AudioMetrics()
        
        # Get file info
        stat = os.stat(file_path)
        metrics.duration_seconds = stat.st_size / 170000  # Rough estimate
        
        # Default values (unknown)
        metrics.integrated_lufs = -14.0
        metrics.true_peak_db = -3.0
        metrics.dynamic_range = 8.0
        
        return metrics
    
    def _estimate_from_metadata(self, metadata: dict) -> AudioMetrics:
        """Estimate metrics from generation metadata"""
        metrics = AudioMetrics()
        
        # Extract or estimate values
        metrics.integrated_lufs = metadata.get("lufs", -14.0)
        metrics.true_peak_db = metadata.get("peak_db", -3.0)
        metrics.dynamic_range = metadata.get("dynamic_range", 8.0)
        metrics.duration_seconds = metadata.get("duration", 180.0)
        metrics.sample_rate = metadata.get("sample_rate", 44100)
        
        return metrics
    
    def _score_creativity(self, file_path: str, metrics: AudioMetrics) -> CreativityScore:
        """Score creativity aspects (placeholder for ML-based scoring)"""
        # In production, this would use an ML model
        # For now, return reasonable defaults based on metrics
        
        creativity = CreativityScore(
            originality=75.0,
            arrangement=70.0,
            harmonic_richness=72.0,
            rhythmic_variety=68.0,
            production_quality=75.0
        )
        
        # Adjust based on dynamic range
        if metrics.dynamic_range >= 10:
            creativity.production_quality += 5
            creativity.rhythmic_variety += 5
        
        # Adjust based on loudness
        if -16 <= metrics.integrated_lufs <= -12:
            creativity.production_quality += 5
        
        # Normalize to 0-100
        for attr in ["originality", "arrangement", "harmonic_richness", "rhythmic_variety", "production_quality"]:
            value = getattr(creativity, attr)
            setattr(creativity, attr, min(100.0, max(0.0, value)))
        
        # Overall is weighted average
        creativity.overall = (
            creativity.originality * 0.25 +
            creativity.arrangement * 0.20 +
            creativity.harmonic_richness * 0.15 +
            creativity.rhythmic_variety * 0.15 +
            creativity.production_quality * 0.25
        )
        
        return creativity
    
    def _score_from_metadata(self, metadata: dict) -> CreativityScore:
        """Score creativity from generation metadata"""
        creativity = CreativityScore(
            originality=metadata.get("originality_score", 70.0),
            arrangement=metadata.get("arrangement_score", 70.0),
            harmonic_richness=metadata.get("harmonic_score", 70.0),
            rhythmic_variety=metadata.get("rhythmic_score", 70.0),
            production_quality=metadata.get("production_score", 70.0)
        )
        
        # Calculate overall
        creativity.overall = (
            creativity.originality * 0.25 +
            creativity.arrangement * 0.20 +
            creativity.harmonic_richness * 0.15 +
            creativity.rhythmic_variety * 0.15 +
            creativity.production_quality * 0.25
        )
        
        return creativity
    
    def _generate_suggestions(self, metrics: AudioMetrics, creativity: CreativityScore) -> QualitySuggestions:
        """Generate improvement suggestions based on analysis"""
        suggestions = QualitySuggestions()
        
        # LUFS suggestions
        lufs_diff = abs(metrics.integrated_lufs - self.target_lufs)
        if metrics.integrated_lufs > self.target_lufs + 2:
            suggestions.critical.append(
                f"Loudness too high ({metrics.integrated_lufs:.1f} LUFS). "
                f"Target for {self.target_platform} is {self.target_lufs:.1f} LUFS. "
                "Consider reducing gain or applying limiter."
            )
        elif metrics.integrated_lufs < self.target_lufs - 4:
            suggestions.recommended.append(
                f"Loudness low ({metrics.integrated_lufs:.1f} LUFS). "
                f"Consider mastering to {self.target_lufs:.1f} LUFS for {self.target_platform}."
            )
        
        # Peak suggestions
        if metrics.true_peak_db >= self.PEAK_THRESHOLDS["warning"]:
            suggestions.critical.append(
                f"Clipping detected! True peak at {metrics.true_peak_db:.1f} dB. "
                "Apply brickwall limiting to prevent distortion."
            )
        elif metrics.true_peak_db > self.PEAK_THRESHOLDS["acceptable"]:
            suggestions.recommended.append(
                f"Peak level high ({metrics.true_peak_db:.1f} dB). "
                "Leave 0.5-1dB headroom for mastering."
            )
        
        # Dynamic range suggestions
        if metrics.dynamic_range < self.DR_RATINGS["fair"]:
            suggestions.critical.append(
                f"Low dynamic range ({metrics.dynamic_range:.1f} dB). "
                "Track sounds overly compressed. Add more dynamics for impact."
            )
        elif metrics.dynamic_range > self.DR_RATINGS["excellent"]:
            suggestions.optional.append(
                f"High dynamic range ({metrics.dynamic_range:.1f} dB). "
                "Great for classical/ambient, may need compression for club use."
            )
        
        # Creativity suggestions
        if creativity.originality < 60:
            suggestions.recommended.append(
                "Originality score low. Try more unique chord progressions or sound design."
            )
        if creativity.harmonic_richness < 60:
            suggestions.optional.append(
                "Harmonic richness could improve. Add richer chords or counter-melodies."
            )
        if creativity.rhythmic_variety < 60:
            suggestions.recommended.append(
                "Rhythmic variety limited. Experiment with syncopation or polyrhythms."
            )
        
        # Overall quality
        if creativity.production_quality < 60:
            suggestions.critical.append(
                "Production quality needs attention. Check mixing and arrangement."
            )
        
        return suggestions
    
    def _calculate_overall_score(self, metrics: AudioMetrics, creativity: CreativityScore) -> float:
        """Calculate weighted overall quality score (0-100)"""
        # Technical score (40%)
        technical_score = 0.0
        
        # LUFS accuracy (closer to target = better)
        lufs_accuracy = max(0, 100 - abs(metrics.integrated_lufs - self.target_lufs) * 10)
        
        # Peak headroom
        if metrics.true_peak_db <= self.PEAK_THRESHOLDS["optimal"]:
            peak_score = 100.0
        elif metrics.true_peak_db <= self.PEAK_THRESHOLDS["acceptable"]:
            peak_score = 80.0
        else:
            peak_score = max(0, 50 - (metrics.true_peak_db - self.PEAK_THRESHOLDS["acceptable"]) * 50)
        
        # Dynamic range
        dr_score = min(100, metrics.dynamic_range * 7)  # 14 DR = 100
        
        technical_score = (lufs_accuracy * 0.4 + peak_score * 0.3 + dr_score * 0.3)
        
        # Creativity score (60%)
        creative_score = creativity.overall
        
        # Weighted overall
        overall = technical_score * 0.4 + creative_score * 0.6
        
        return round(overall, 1)
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def generate_report_text(self, report: QualityReport) -> str:
        """Generate human-readable report"""
        lines = [
            "=" * 50,
            "AUDIO QUALITY EVALUATION REPORT",
            "=" * 50,
            f"\nFile: {report.file_path}",
            f"Date: {report.timestamp}",
            f"\n{'='*50}",
            "TECHNICAL METRICS",
            f"{'='*50}",
            f"Integrated Loudness: {report.audio_metrics.integrated_lufs:.1f} LUFS (target: {self.target_lufs:.1f})",
            f"True Peak: {report.audio_metrics.true_peak_db:.1f} dB",
            f"Dynamic Range: {report.audio_metrics.dynamic_range:.1f} dB",
            f"Duration: {report.audio_metrics.duration_seconds:.1f}s",
            f"Sample Rate: {report.audio_metrics.sample_rate} Hz",
            f"\n{'='*50}",
            "CREATIVITY SCORES",
            f"{'='*50}",
            f"Overall: {report.creativity.overall:.1f}/100",
            f"  Originality: {report.creativity.originality:.1f}/100",
            f"  Arrangement: {report.creativity.arrangement:.1f}/100",
            f"  Harmonic Richness: {report.creativity.harmonic_richness:.1f}/100",
            f"  Rhythmic Variety: {report.creativity.rhythmic_variety:.1f}/100",
            f"  Production Quality: {report.creativity.production_quality:.1f}/100",
            f"\n{'='*50}",
            "SUGGESTIONS",
            f"{'='*50}",
        ]
        
        if report.suggestions.critical:
            lines.append("\n🔴 CRITICAL:")
            for s in report.suggestions.critical:
                lines.append(f"  • {s}")
        
        if report.suggestions.recommended:
            lines.append("\n🟡 RECOMMENDED:")
            for s in report.suggestions.recommended:
                lines.append(f"  • {s}")
        
        if report.suggestions.optional:
            lines.append("\n🟢 OPTIONAL:")
            for s in report.suggestions.optional:
                lines.append(f"  • {s}")
        
        lines.extend([
            f"\n{'='*50}",
            "FINAL GRADE",
            f"{'='*50}",
            f"Overall Score: {report.overall_score:.1f}/100",
            f"Grade: {report.grade}",
            "=" * 50,
        ])
        
        return "\n".join(lines)


# Convenience function for quick evaluation
def evaluate_audio(file_path: str, target: str = "streaming") -> QualityReport:
    """
    Quick evaluation function.
    
    Args:
        file_path: Path to audio file
        target: Target platform ('streaming', 'broadcast', 'mastering', 'ambient')
    
    Returns:
        QualityReport with complete analysis
    """
    evaluator = QualityEvaluator(target_platform=target)
    return evaluator.evaluate_file(file_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_evaluator.py <audio_file> [target_platform]")
        print("\nTarget platforms: streaming, broadcast, mastering, ambient")
        print("\nExample: python quality_evaluator.py track.wav streaming")
        sys.exit(1)
    
    file_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "streaming"
    
    evaluator = QualityEvaluator(target_platform=target)
    
    try:
        report = evaluator.evaluate_file(file_path)
        print(evaluator.generate_report_text(report))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Analysis error: {e}")
        sys.exit(1)
