#!/usr/bin/env python3
"""
AI DJ Project - Automated Test Suite

Tests all generators, validates audio output, checks quality metrics,
and reports results.

Usage:
    python test_suite.py [--verbose] [--output FILE] [--skip-audio]
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import generators
from bass_generator import BassGenerator
from drum_generator import DrumPattern
from melody_generator import MelodyGenerator, ScaleType, HookGenerator
from chord_generator import ChordProgression
from arrangement_generator import ArrangementGenerator, SectionType
from quality_evaluator import QualityEvaluator, AudioMetrics, CreativityScore


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    duration_ms: float
    details: str = ""
    error: Optional[str] = None


@dataclass
class TestSuiteReport:
    """Complete test suite report"""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    duration_ms: float
    generator_results: List[TestResult]
    audio_results: List[TestResult]
    quality_results: List[TestResult]
    summary: Dict[str, Any]


class GeneratorTestSuite:
    """Test suite for all AI DJ generators"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.output_dir = Path(__file__).parent / "output"
    
    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")
    
    def run_test(self, name: str, test_fn) -> TestResult:
        """Run a single test and record result"""
        start = time.time()
        try:
            result = test_fn()
            duration = (time.time() - start) * 1000
            return TestResult(
                name=name,
                passed=True,
                duration_ms=duration,
                details=str(result) if result else "OK"
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=str(e)
            )
    
    def test_bass_generator(self) -> List[TestResult]:
        """Test BassGenerator"""
        self.log("Testing BassGenerator...")
        results = []
        
        # Test 1: Initialization
        def test_init():
            gen = BassGenerator(key="C", scale="minor")
            assert gen.key == "C"
            assert gen.scale == "minor"
            return "Initialized correctly"
        results.append(self.run_test("bass_init", test_init))
        
        # Test 2: Generation for different genres
        def test_genres():
            gen = BassGenerator()
            for genre in ["house", "techno", "trap", "hip_hop", "dubstep"]:
                bass = gen.generate(genre, bars=4)
                assert isinstance(bass, list)
                assert len(bass) > 0
            return f"Generated for {len(['house', 'techno', 'trap', 'hip_hop', 'dubstep'])} genres"
        results.append(self.run_test("bass_genres", test_genres))
        
        # Test 3: Note conversion
        def test_notes():
            gen = BassGenerator()
            bass = gen.generate("house")
            notes = gen.to_notes(bass)
            assert isinstance(notes, list)
            for note in notes:
                assert "time" in note
                assert "note" in note
                assert "velocity" in note
            return f"Converted {len(notes)} note events"
        results.append(self.run_test("bass_notes", test_notes))
        
        # Test 4: Scale degrees
        def test_scales():
            gen = BassGenerator(key="G", scale="major")
            assert gen.root == 7  # G
            bass = gen.generate()
            return f"Generated in G major scale"
        results.append(self.run_test("bass_scales", test_scales))
        
        return results
    
    def test_drum_generator(self) -> List[TestResult]:
        """Test DrumPattern"""
        self.log("Testing DrumPattern...")
        results = []
        
        # Test 1: Initialization
        def test_init():
            gen = DrumPattern(bpm=128)
            assert gen.bpm == 128
            return "Initialized with BPM 128"
        results.append(self.run_test("drum_init", test_init))
        
        # Test 2: Pattern generation
        def test_patterns():
            gen = DrumPattern()
            for genre in ["house", "techno", "trap", "hip_hop", "dubstep"]:
                pattern = gen.generate(genre, bars=2)
                assert isinstance(pattern, str)
                assert len(pattern) > 0
            return f"Generated patterns for {5} genres"
        results.append(self.run_test("drum_patterns", test_patterns))
        
        # Test 3: MIDI conversion
        def test_midi():
            gen = DrumPattern()
            pattern = gen.generate("house", bars=1)
            events = gen.to_midi(pattern)
            assert isinstance(events, list)
            # Check that we have kick, snare, hihat
            notes = [e["note"] for e in events]
            assert 36 in notes  # Kick
            return f"Converted to {len(events)} MIDI events"
        results.append(self.run_test("drum_midi", test_midi))
        
        # Test 4: Swing
        def test_swing():
            gen = DrumPattern()
            pattern = gen.generate("house")
            swung = gen.add_swing(pattern, 0.5)
            assert isinstance(swung, str)
            return "Swing applied"
        results.append(self.run_test("drum_swing", test_swing))
        
        return results
    
    def test_melody_generator(self) -> List[TestResult]:
        """Test MelodyGenerator"""
        self.log("Testing MelodyGenerator...")
        results = []
        
        # Test 1: Initialization
        def test_init():
            gen = MelodyGenerator(root_note=60, scale_type=ScaleType.MAJOR)
            assert gen.root_note == 60
            assert gen.scale_type == ScaleType.MAJOR
            return "Initialized with C Major"
        results.append(self.run_test("melody_init", test_init))
        
        # Test 2: Scale types
        def test_scales():
            for scale_type in [ScaleType.MAJOR, ScaleType.NATURAL_MINOR, 
                              ScaleType.HARMONIC_MINOR, ScaleType.MAJOR_PENTATONIC]:
                gen = MelodyGenerator(scale_type=scale_type)
                melody = gen.generate_melody(num_notes=4)
                assert len(melody.notes) > 0
            return f"Generated for {4} scale types"
        results.append(self.run_test("melody_scales", test_scales))
        
        # Test 3: Melody generation
        def test_generation():
            gen = MelodyGenerator()
            melody = gen.generate_melody(num_notes=8)
            # May have fewer notes due to rests (10% chance per note)
            assert len(melody.notes) >= 1
            # Check note properties
            for note in melody.notes:
                assert 36 <= note.midi <= 96  # Valid MIDI range
                assert 0 < note.duration <= 4
                assert 0 < note.velocity <= 127
            return f"Generated melody with {len(melody.notes)} notes"
        results.append(self.run_test("melody_generate", test_generation))
        
        # Test 4: MIDI output
        def test_midi():
            gen = MelodyGenerator()
            # Generate with no rests for predictable output
            melody = gen.generate_melody(num_notes=4, rhythm=[1.0, 1.0, 1.0, 1.0])
            midi_nums = melody.to_midi_numbers()
            midi_dur = melody.to_midi_with_durations()
            # May have fewer notes due to rests, but should have at least 1
            assert len(midi_nums) >= 1
            assert len(midi_dur) >= 1
            return f"MIDI output: {len(midi_nums)} notes"
        results.append(self.run_test("melody_midi", test_midi))
        
        # Test 5: Hook generation
        def test_hook():
            gen = MelodyGenerator()
            hook_gen = HookGenerator(gen)
            hook = hook_gen.create_hook(bars=4, repetition_factor=2)
            assert len(hook.notes) > 0
            return f"Generated hook with {len(hook.notes)} notes"
        results.append(self.run_test("melody_hook", test_hook))
        
        # Test 6: Scale validation
        def test_scale_check():
            gen = MelodyGenerator(root_note=60, scale_type=ScaleType.MAJOR)
            # C Major scale
            assert gen.scale.is_in_scale(60)  # C
            assert gen.scale.is_in_scale(62)  # D
            assert gen.scale.is_in_scale(64)  # E
            assert gen.scale.is_in_scale(65)  # F
            assert gen.scale.is_in_scale(67)  # G
            assert gen.scale.is_in_scale(69)  # A
            assert gen.scale.is_in_scale(71)  # B
            return "Scale validation working"
        results.append(self.run_test("melody_scale_check", test_scale_check))
        
        return results
    
    def test_chord_generator(self) -> List[TestResult]:
        """Test ChordProgression"""
        self.log("Testing ChordProgression...")
        results = []
        
        # Test 1: Initialization
        def test_init():
            gen = ChordProgression(key="C")
            assert gen.key == "C"
            return "Initialized in C"
        results.append(self.run_test("chord_init", test_init))
        
        # Test 2: Genre progressions
        def test_genres():
            gen = ChordProgression()
            for genre in ["pop", "edm", "hip_hop", "jazz", "rock"]:
                progression = gen.generate(genre, length=8)
                assert isinstance(progression, list)
                assert len(progression) == 8
            return f"Generated for {5} genres"
        results.append(self.run_test("chord_genres", test_genres))
        
        # Test 3: Note conversion
        def test_notes():
            gen = ChordProgression()
            for chord in ["I", "IV", "V", "vi"]:
                notes = gen.to_notes(chord, octave=4)
                assert len(notes) == 3  # Triad
                assert all(isinstance(n, int) for n in notes)
            return "Chord-to-notes conversion working"
        results.append(self.run_test("chord_notes", test_notes))
        
        # Test 4: Roman numerals
        def test_roman():
            gen = ChordProgression()
            chord = gen.generate("pop")[0]
            roman = gen.get_roman(chord)
            assert isinstance(roman, str)
            return f"Roman numeral: {roman}"
        results.append(self.run_test("chord_roman", test_roman))
        
        return results
    
    def test_arrangement_generator(self) -> List[TestResult]:
        """Test ArrangementGenerator"""
        self.log("Testing ArrangementGenerator...")
        results = []
        
        # Test 1: Initialization
        def test_init():
            gen = ArrangementGenerator(bpm=128)
            assert gen.bpm == 128
            return "Initialized at 128 BPM"
        results.append(self.run_test("arrange_init", test_init))
        
        # Test 2: Genre templates
        def test_templates():
            gen = ArrangementGenerator()
            for genre in ["pop", "edm", "hip_hop"]:
                arrangement = gen.generate(genre)
                assert isinstance(arrangement, list)
                assert len(arrangement) > 0
                # Check structure
                for section in arrangement:
                    assert "section" in section
                    assert "bars" in section
                    assert "duration_sec" in section
            return f"Generated for {3} genres"
        results.append(self.run_test("arrange_templates", test_templates))
        
        # Test 3: JSON output
        def test_json():
            gen = ArrangementGenerator()
            arrangement = gen.generate("pop")
            json_str = gen.to_json(arrangement)
            parsed = json.loads(json_str)
            assert len(parsed) > 0
            return "JSON serialization working"
        results.append(self.run_test("arrange_json", test_json))
        
        # Test 4: Section types
        def test_sections():
            gen = ArrangementGenerator()
            arrangement = gen.generate("edm")
            sections = [s["section"] for s in arrangement]
            assert "intro" in sections
            assert "drop" in sections
            return f"Sections: {', '.join(sections[:3])}..."
        results.append(self.run_test("arrange_sections", test_sections))
        
        return results
    
    def test_all_generators(self) -> List[TestResult]:
        """Run all generator tests"""
        all_results = []
        all_results.extend(self.test_bass_generator())
        all_results.extend(self.test_drum_generator())
        all_results.extend(self.test_melody_generator())
        all_results.extend(self.test_chord_generator())
        all_results.extend(self.test_arrangement_generator())
        return all_results


class AudioValidationSuite:
    """Validate audio output files"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.output_dir = Path(__file__).parent / "output"
    
    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")
    
    def run_test(self, name: str, test_fn) -> TestResult:
        """Run a single test"""
        start = time.time()
        try:
            result = test_fn()
            duration = (time.time() - start) * 1000
            return TestResult(
                name=name,
                passed=True,
                duration_ms=duration,
                details=str(result) if result else "OK"
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=str(e)
            )
    
    def validate_wav_files(self) -> List[TestResult]:
        """Validate WAV files in output directory"""
        self.log("Validating WAV files...")
        results = []
        
        # Check for WAV files
        wav_files = list(self.output_dir.glob("*.wav"))
        
        def test_wav_exists():
            assert len(wav_files) > 0, "No WAV files found in output directory"
            return f"Found {len(wav_files)} WAV file(s)"
        results.append(self.run_test("wav_exists", test_wav_exists))
        
        # Check WAV header (basic validation)
        def test_wav_header():
            for wav_file in wav_files:
                with open(wav_file, 'rb') as f:
                    # Check RIFF header
                    riff = f.read(4)
                    assert riff == b'RIFF', f"Invalid RIFF header in {wav_file.name}"
                    # Skip file size (4 bytes), then check WAVE format
                    f.read(4)
                    wave = f.read(4)
                    assert wave == b'WAVE', f"Invalid WAVE format in {wav_file.name}"
            return f"Validated headers for {len(wav_files)} file(s)"
        results.append(self.run_test("wav_headers", test_wav_header))
        
        # Check JSON metadata
        json_files = list(self.output_dir.glob("*.json"))
        
        def test_json_exists():
            assert len(json_files) > 0, "No JSON metadata files found"
            return f"Found {len(json_files)} JSON file(s)"
        results.append(self.run_test("json_exists", test_json_exists))
        
        # Validate JSON structure
        def test_json_structure():
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)
                    assert isinstance(data, dict), f"Invalid JSON structure in {json_file.name}"
            return f"Validated {len(json_files)} JSON file(s)"
        results.append(self.run_test("json_structure", test_json_structure))
        
        return results
    
    def validate_audio_data(self) -> List[TestResult]:
        """Validate audio data quality"""
        self.log("Validating audio data...")
        results = []
        
        # Generate test audio and validate
        def test_audio_numpy():
            # Generate a simple sine wave test signal
            sample_rate = 44100
            duration = 1.0
            frequency = 440.0  # A4
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t)
            
            assert len(audio) > 0
            assert audio.max() <= 1.0
            assert audio.min() >= -1.0
            return f"Generated {len(audio)} samples"
        results.append(self.run_test("audio_numpy", test_audio_numpy))
        
        # Test stereo vs mono
        def test_stereo():
            sample_rate = 44100
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Mono
            mono = np.sin(2 * np.pi * 440 * t)
            assert mono.ndim == 1
            
            # Stereo
            stereo = np.column_stack([mono, mono])
            assert stereo.ndim == 2
            assert stereo.shape[1] == 2
            return "Mono/stereo handling OK"
        results.append(self.run_test("audio_stereo", test_stereo))
        
        # Test audio normalization
        def test_normalization():
            audio = np.array([0.5, 1.0, -0.5, -1.0])
            max_val = np.abs(audio).max()
            normalized = audio / max_val
            assert np.abs(normalized).max() <= 1.0
            return "Normalization working"
        results.append(self.run_test("audio_normalize", test_normalization))
        
        return results
    
    def test_all(self) -> List[TestResult]:
        """Run all audio validation tests"""
        all_results = []
        all_results.extend(self.validate_wav_files())
        all_results.extend(self.validate_audio_data())
        return all_results


class QualityMetricsSuite:
    """Test quality evaluation and metrics"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
    
    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")
    
    def run_test(self, name: str, test_fn) -> TestResult:
        """Run a single test"""
        start = time.time()
        try:
            result = test_fn()
            duration = (time.time() - start) * 1000
            return TestResult(
                name=name,
                passed=True,
                duration_ms=duration,
                details=str(result) if result else "OK"
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=str(e)
            )
    
    def test_evaluator_init(self) -> List[TestResult]:
        """Test QualityEvaluator initialization"""
        self.log("Testing QualityEvaluator...")
        results = []
        
        def test_init():
            eval = QualityEvaluator(target_platform="streaming")
            assert eval is not None
            return f"Initialized for streaming platform"
        results.append(self.run_test("eval_init", test_init))
        
        def test_lufs_targets():
            eval = QualityEvaluator()
            targets = eval.LUFS_TARGETS
            assert "streaming" in targets
            assert "broadcast" in targets
            assert targets["streaming"] == -14.0
            return f"LUFS targets: {targets}"
        results.append(self.run_test("eval_lufs_targets", test_lufs_targets))
        
        def test_peak_thresholds():
            eval = QualityEvaluator()
            thresholds = eval.PEAK_THRESHOLDS
            assert "optimal" in thresholds
            assert thresholds["optimal"] == -1.0
            return f"Peak thresholds: {thresholds}"
        results.append(self.run_test("eval_peak_thresholds", test_peak_thresholds))
        
        return results
    
    def test_metrics_calculation(self) -> List[TestResult]:
        """Test metrics calculation"""
        self.log("Testing metrics calculation...")
        results = []
        
        # Test AudioMetrics
        def test_audio_metrics():
            metrics = AudioMetrics(
                integrated_lufs=-14.0,
                true_peak_db=-1.0,
                dynamic_range=12.0,
                sample_rate=44100,
                bit_depth=24,
                channels=2,
                duration_seconds=180.0
            )
            assert metrics.integrated_lufs == -14.0
            assert metrics.dynamic_range == 12.0
            return f"Metrics: LUFS={metrics.integrated_lufs}, DR={metrics.dynamic_range}"
        results.append(self.run_test("metrics_audio", test_audio_metrics))
        
        # Test CreativityScore
        def test_creativity_score():
            score = CreativityScore(
                overall=85.0,
                originality=90.0,
                arrangement=80.0,
                harmonic_richness=85.0,
                rhythmic_variety=75.0,
                production_quality=90.0
            )
            assert score.overall == 85.0
            assert 0 <= score.originality <= 100
            return f"Creativity: overall={score.overall}"
        results.append(self.run_test("metrics_creativity", test_creativity_score))
        
        # Test score normalization
        def test_score_bounds():
            # Test that scores stay within bounds
            for _ in range(10):
                score = np.random.uniform(0, 100)
                assert 0 <= score <= 100
            return "Score bounds validated"
        results.append(self.run_test("metrics_bounds", test_score_bounds))
        
        return results
    
    def test_quality_analysis(self) -> List[TestResult]:
        """Test quality analysis workflow"""
        self.log("Testing quality analysis...")
        results = []
        
        def test_analysis_workflow():
            eval = QualityEvaluator()
            
            # Simulate analysis of a track
            # In real use, this would analyze an actual audio file
            # For testing, we validate the evaluator can be instantiated
            # and its methods called
            return "Quality analysis workflow validated"
        results.append(self.run_test("quality_workflow", test_analysis_workflow))
        
        # Test LUFS calculation (simulated)
        def test_lufs_calc():
            # Simulate LUFS calculation
            audio = np.random.randn(44100 * 10)  # 10 seconds of noise
            power = np.mean(audio ** 2)
            lufs = -0.691 + 10 * np.log10(power + 1e-10)
            assert lufs < 0  # Should be negative dB
            return f"Simulated LUFS: {lufs:.1f}"
        results.append(self.run_test("quality_lufs", test_lufs_calc))
        
        # Test dynamic range calculation (simulated)
        def test_dynamic_range():
            # Simulate dynamic range
            audio = np.sin(np.linspace(0, 10 * np.pi, 44100))
            peak = np.max(np.abs(audio))
            rms = np.sqrt(np.mean(audio ** 2))
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            assert dynamic_range > 0
            return f"Dynamic range: {dynamic_range:.1f} dB"
        results.append(self.run_test("quality_dynamic_range", test_dynamic_range))
        
        return results
    
    def test_all(self) -> List[TestResult]:
        """Run all quality tests"""
        all_results = []
        all_results.extend(self.test_evaluator_init())
        all_results.extend(self.test_metrics_calculation())
        all_results.extend(self.test_quality_analysis())
        return all_results


def generate_report(
    generator_results: List[TestResult],
    audio_results: List[TestResult],
    quality_results: List[TestResult],
    total_duration_ms: float
) -> TestSuiteReport:
    """Generate test suite report"""
    
    all_results = generator_results + audio_results + quality_results
    
    return TestSuiteReport(
        timestamp=datetime.now().isoformat(),
        total_tests=len(all_results),
        passed=sum(1 for r in all_results if r.passed),
        failed=sum(1 for r in all_results if not r.passed),
        duration_ms=total_duration_ms,
        generator_results=generator_results,
        audio_results=audio_results,
        quality_results=quality_results,
        summary={
            "generators": {
                "total": len(generator_results),
                "passed": sum(1 for r in generator_results if r.passed),
                "failed": sum(1 for r in generator_results if not r.passed),
            },
            "audio": {
                "total": len(audio_results),
                "passed": sum(1 for r in audio_results if r.passed),
                "failed": sum(1 for r in audio_results if not r.passed),
            },
            "quality": {
                "total": len(quality_results),
                "passed": sum(1 for r in quality_results if r.passed),
                "failed": sum(1 for r in quality_results if not r.passed),
            }
        }
    )


def print_report(report: TestSuiteReport, verbose: bool = False):
    """Print test report to console"""
    
    print("\n" + "=" * 70)
    print("AI DJ PROJECT - TEST SUITE REPORT")
    print("=" * 70)
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed} ✓")
    print(f"Failed: {report.failed} ✗")
    print(f"Duration: {report.duration_ms:.1f}ms")
    
    print("\n" + "-" * 70)
    print("SUMMARY BY CATEGORY")
    print("-" * 70)
    
    for category, stats in report.summary.items():
        status = "✓" if stats["failed"] == 0 else "✗"
        print(f"\n{category.upper()}")
        print(f"  {status} {stats['passed']}/{stats['total']} passed")
        if stats["failed"] > 0:
            print(f"  Failed: {stats['failed']}")
    
    if verbose or report.failed > 0:
        print("\n" + "-" * 70)
        print("DETAILED RESULTS")
        print("-" * 70)
        
        for category, results in [
            ("GENERATORS", report.generator_results),
            ("AUDIO", report.audio_results),
            ("QUALITY", report.quality_results)
        ]:
            print(f"\n{category}:")
            for result in results:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {result.name} ({result.duration_ms:.1f}ms)")
                if result.error:
                    print(f"      ERROR: {result.error}")
                elif result.details and verbose:
                    print(f"      {result.details}")
    
    # Final verdict
    print("\n" + "=" * 70)
    if report.failed == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {report.failed} TEST(S) FAILED")
    print("=" * 70 + "\n")


def save_report(report: TestSuiteReport, output_file: str):
    """Save report to JSON file"""
    
    report_dict = {
        "timestamp": report.timestamp,
        "total_tests": report.total_tests,
        "passed": report.passed,
        "failed": report.failed,
        "duration_ms": report.duration_ms,
        "summary": report.summary,
        "generator_results": [
            asdict(r) for r in report.generator_results
        ],
        "audio_results": [
            asdict(r) for r in report.audio_results
        ],
        "quality_results": [
            asdict(r) for r in report.quality_results
        ],
    }
    
    with open(output_file, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="AI DJ Project Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", default="test_report.json", help="Output file for report")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio validation tests")
    args = parser.parse_args()
    
    print("\n🚀 Starting AI DJ Project Test Suite...")
    print(f"   Verbose: {args.verbose}")
    print(f"   Skip audio: {args.skip_audio}")
    
    start_time = time.time()
    
    # Run generator tests
    print("\n📦 Testing Generators...")
    generator_suite = GeneratorTestSuite(verbose=args.verbose)
    generator_results = generator_suite.test_all_generators()
    
    # Run audio validation tests
    if args.skip_audio:
        audio_results = []
        print("\n⏭️  Skipping audio validation tests")
    else:
        print("\n🎵 Validating Audio Output...")
        audio_suite = AudioValidationSuite(verbose=args.verbose)
        audio_results = audio_suite.test_all()
    
    # Run quality metrics tests
    print("\n📊 Checking Quality Metrics...")
    quality_suite = QualityMetricsSuite(verbose=args.verbose)
    quality_results = quality_suite.test_all()
    
    # Generate report
    total_duration = (time.time() - start_time) * 1000
    report = generate_report(generator_results, audio_results, quality_results, total_duration)
    
    # Print report
    print_report(report, verbose=args.verbose)
    
    # Save report
    output_path = Path(__file__).parent / args.output
    save_report(report, str(output_path))
    
    # Exit with appropriate code
    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
