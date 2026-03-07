#!/usr/bin/env python3
"""
Data Pipeline - Audio Preprocessing, Augmentation, and Training Data Preparation

This module handles:
1. Audio preprocessing (loading, normalization, resampling)
2. Audio augmentation (pitch shift, time stretch, noise, effects)
3. Training data preparation (feature extraction, dataset creation)
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import random

# Audio processing imports
try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not installed. Install with: pip install librosa")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("Warning: soundfile not installed. Install with: pip install soundfile")

try:
    import scipy.signal as signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Feature extraction
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 20
    fmin: float = 0
    fmax: float = 22050
    duration: Optional[float] = None  # Max duration in seconds, None = full
    normalize: bool = True
    mono: bool = True


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation."""
    pitch_shift_steps: Tuple[float, float] = (-2, 2)  # Semitones
    time_stretch_factor: Tuple[float, float] = (0.9, 1.1)
    noise_level: float = 0.005
    reverb_room_size: float = 0.5
    reverb_wet: float = 0.3
    low_pass_freq: Optional[int] = None
    high_pass_freq: Optional[int] = None
    volume_range: Tuple[float, float] = (0.8, 1.2)


@dataclass
class TrainingDataConfig:
    """Configuration for training data preparation."""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    seed: int = 42
    batch_size: int = 32
    sequence_length: float = 4.0  # seconds per sample
    overlap: float = 0.5  # overlap between segments


class AudioPreprocessor:
    """Handles audio loading, normalization, and basic transformations."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform and sample rate."""
        if not HAS_LIBROSA:
            raise RuntimeError("librosa required for audio loading")

        logger.info(f"Loading audio: {file_path}")

        # Load with specified sample rate
        y, sr = librosa.load(
            file_path,
            sr=self.config.sample_rate,
            mono=self.config.mono,
            duration=self.config.duration
        )

        logger.info(f"Loaded audio: {len(y)} samples at {sr}Hz")

        return y, sr

    def normalize(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
        return y

    def resample(self, y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return y

        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

    def trim_silence(self, y: np.ndarray, top_db: int = 30) -> np.ndarray:
        """Trim leading and trailing silence."""
        return librosa.effects.trim(y, top_db=top_db)[0]

    def split_channels(self, y: np.ndarray) -> List[np.ndarray]:
        """Split stereo audio into individual channels."""
        if len(y.shape) == 1:
            return [y]
        return [y[i] for i in range(y.shape[0])]

    def process(self, file_path: str) -> np.ndarray:
        """Full preprocessing pipeline."""
        y, sr = self.load_audio(file_path)

        if self.config.normalize:
            y = self.normalize(y)

        y = self.trim_silence(y)

        return y


class AudioAugmentor:
    """Applies various augmentations to audio for data augmentation."""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()

    def pitch_shift(self, y: np.ndarray, sr: int, steps: Optional[float] = None) -> np.ndarray:
        """Shift pitch by specified semitones."""
        if steps is None:
            steps = random.uniform(*self.config.pitch_shift_steps)

        logger.debug(f"Pitch shift: {steps} semitones")
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

    def time_stretch(self, y: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
        """Time stretch audio by factor."""
        if rate is None:
            rate = random.uniform(*self.config.time_stretch_factor)

        logger.debug(f"Time stretch: {rate}x")
        return librosa.effects.time_stretch(y, rate=rate)

    def add_noise(self, y: np.ndarray, level: Optional[float] = None) -> np.ndarray:
        """Add white noise at specified level."""
        if level is None:
            level = self.config.noise_level

        noise = np.random.randn(len(y)) * level
        return y + noise

    def add_reverb(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Add simple reverb effect."""
        # Simple convolution-based reverb
        room_size = int(self.config.reverb_room_size * sr * 0.05)  # 0-50ms room
        impulse = np.exp(-np.linspace(0, 5, room_size))
        impulse = np.concatenate([impulse, np.zeros(room_size)])

        reverb_wet = self.config.reverb_wet
        reverb_dry = 1 - reverb_wet

        reverb = signal.convolve(y, impulse, mode='same')
        return reverb_dry * y + reverb_wet * reverb

    def low_pass_filter(self, y: np.ndarray, sr: int, freq: Optional[int] = None) -> np.ndarray:
        """Apply low-pass filter."""
        if freq is None:
            if self.config.low_pass_freq is None:
                return y
            freq = self.config.low_pass_freq

        nyquist = sr / 2
        if freq >= nyquist:
            return y

        b, a = signal.butter(8, freq / nyquist, btype='low')
        return signal.filtfilt(b, a, y)

    def high_pass_filter(self, y: np.ndarray, sr: int, freq: Optional[int] = None) -> np.ndarray:
        """Apply high-pass filter."""
        if freq is None:
            if self.config.high_pass_freq is None:
                return y
            freq = self.config.high_pass_freq

        nyquist = sr / 2
        if freq >= nyquist:
            return y

        b, a = signal.butter(8, freq / nyquist, btype='high')
        return signal.filtfilt(b, a, y)

    def adjust_volume(self, y: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """Adjust volume by factor."""
        if factor is None:
            factor = random.uniform(*self.config.volume_range)

        return y * factor

    def random_augment(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Apply random combination of augmentations."""
        augs = [
            ('pitch', random.random() > 0.5),
            ('time', random.random() > 0.5),
            ('noise', random.random() > 0.7),
            ('volume', random.random() > 0.3),
            ('filter', random.random() > 0.6),
        ]

        for aug_name, apply_it in augs:
            if apply_it:
                try:
                    if aug_name == 'pitch':
                        y = self.pitch_shift(y, sr)
                    elif aug_name == 'time':
                        y = self.time_stretch(y)
                    elif aug_name == 'noise':
                        y = self.add_noise(y)
                    elif aug_name == 'volume':
                        y = self.adjust_volume(y)
                    elif aug_name == 'filter':
                        if random.random() > 0.5:
                            y = self.low_pass_filter(y, sr)
                        else:
                            y = self.high_pass_filter(y, sr)
                except Exception as e:
                    logger.warning(f"Augmentation {aug_name} failed: {e}")

        return y


class FeatureExtractor:
    """Extract audio features for training data."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def extract_melspectrogram(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )
        return librosa.power_to_db(mel, ref=np.max)

    def extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features."""
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )
        return mfcc

    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features."""
        features = {}

        # Spectral centroid
        features['spectral_centroid'] = float(
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length))
        )

        # Spectral rolloff
        features['spectral_rolloff'] = float(
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length))
        )

        # Spectral bandwidth
        features['spectral_bandwidth'] = float(
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.config.n_fft, hop_length=self.config.hop_length))
        )

        # Zero crossing rate
        features['zcr'] = float(
            np.mean(librosa.feature.zero_crossing_rate(y, n_fft=self.config.n_fft, hop_length=self.config.hop_length))
        )

        return features

    def extract_tempo(self, y: np.ndarray, sr: int) -> float:
        """Extract tempo (BPM)."""
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)

    def extract_key(self, y: np.ndarray, sr: int) -> Dict[str, any]:
        """Extract musical key."""
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Estimate key from chroma
        chroma_mean = np.mean(chroma, axis=1)

        # Major keys: more even distribution
        # Minor keys: more concentrated
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]

        is_major = major_corr > minor_corr
        key_idx = np.argmax(chroma_mean)

        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_name = keys[key_idx] + (' major' if is_major else ' minor')

        return {'key': key_name, 'root': keys[key_idx], 'mode': 'major' if is_major else 'minor'}

    def extract_all(self, y: np.ndarray, sr: int) -> Dict[str, any]:
        """Extract all features."""
        features = {}

        # Mel spectrogram
        features['melspectrogram'] = self.extract_melspectrogram(y, sr)

        # MFCC
        features['mfcc'] = self.extract_mfcc(y, sr)

        # Spectral
        features.update(self.extract_spectral_features(y, sr))

        # Tempo
        features['tempo'] = self.extract_tempo(y, sr)

        # Key
        features['key'] = self.extract_key(y, sr)

        return features


class TrainingDataPreparator:
    """Prepares training data from audio files."""

    def __init__(
        self,
        audio_config: Optional[AudioConfig] = None,
        aug_config: Optional[AugmentationConfig] = None,
        train_config: Optional[TrainingDataConfig] = None
    ):
        self.audio_config = audio_config or AudioConfig()
        self.aug_config = aug_config or AugmentationConfig()
        self.train_config = train_config or TrainingDataConfig()

        self.preprocessor = AudioPreprocessor(self.audio_config)
        self.augmentor = AudioAugmentor(self.aug_config)
        self.feature_extractor = FeatureExtractor(self.audio_config)

    def load_dataset(self, source_dir: str, extensions: List[str] = None) -> List[str]:
        """Load all audio files from directory."""
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

        source_path = Path(source_dir)
        audio_files = []

        for ext in extensions:
            audio_files.extend(source_path.glob(f"**/*{ext}"))

        logger.info(f"Found {len(audio_files)} audio files in {source_dir}")

        return [str(f) for f in audio_files]

    def segment_audio(
        self,
        y: np.ndarray,
        sr: int,
        segment_length: Optional[float] = None
    ) -> List[Tuple[np.ndarray, float]]:
        """Segment audio into overlapping chunks."""
        if segment_length is None:
            segment_length = self.train_config.sequence_length

        segment_samples = int(segment_length * sr)
        overlap_samples = int(segment_length * self.train_config.overlap * sr)
        step = segment_samples - overlap_samples

        segments = []
        for start in range(0, len(y) - segment_samples + 1, step):
            segment = y[start:start + segment_samples]
            timestamp = start / sr
            segments.append((segment, timestamp))

        return segments

    def create_samples(
        self,
        audio_files: List[str],
        output_dir: str,
        extract_features: bool = True,
        augment: bool = False,
        num_augmented: int = 3
    ) -> Dict[str, List[str]]:
        """Create training samples from audio files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples = {
            'original': [],
            'augmented': [],
            'features': []
        }

        for audio_file in audio_files:
            try:
                # Load and preprocess
                y = self.preprocessor.process(audio_file)
                sr = self.audio_config.sample_rate

                # Extract segments
                segments = self.segment_audio(y, sr)

                for idx, (segment, timestamp) in enumerate(segments):
                    # Save original segment
                    original_name = f"{Path(audio_file).stem}_seg{idx:04d}.npy"
                    original_path = output_path / 'original' / original_name
                    original_path.parent.mkdir(parents=True, exist_ok=True)

                    np.save(original_path, segment)
                    samples['original'].append(str(original_path))

                    # Extract and save features
                    if extract_features:
                        features = self.feature_extractor.extract_all(segment, sr)
                        features_name = f"{Path(audio_file).stem}_seg{idx:04d}_features.json"
                        features_path = output_path / 'features' / features_name
                        features_path.parent.mkdir(parents=True, exist_ok=True)

                        # Convert numpy arrays to lists for JSON serialization
                        features_serializable = {}
                        for k, v in features.items():
                            if isinstance(v, np.ndarray):
                                features_serializable[k] = v.tolist()
                            else:
                                features_serializable[k] = v

                        with open(features_path, 'w') as f:
                            json.dump(features_serializable, f, indent=2)
                        samples['features'].append(str(features_path))

                    # Create augmented versions
                    if augment:
                        for aug_idx in range(num_augmented):
                            try:
                                augmented = self.augmentor.random_augment(segment, sr)

                                aug_name = f"{Path(audio_file).stem}_seg{idx:04d}_aug{aug_idx}.npy"
                                aug_path = output_path / 'augmented' / aug_name
                                aug_path.parent.mkdir(parents=True, exist_ok=True)

                                np.save(aug_path, augmented)
                                samples['augmented'].append(str(aug_path))
                            except Exception as e:
                                logger.warning(f"Augmentation failed for {audio_file} segment {idx}: {e}")

            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")

        logger.info(f"Created {len(samples['original'])} original samples, {len(samples['augmented'])} augmented")
        return samples

    def split_dataset(
        self,
        samples: List[str],
        metadata: Optional[Dict] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split dataset into train/val/test sets."""
        if self.train_config.shuffle:
            random.seed(self.train_config.seed)
            random.shuffle(samples)

        n = len(samples)
        train_size = int(n * self.train_config.train_split)
        val_size = int(n * self.train_config.val_split)

        train = samples[:train_size]
        val = samples[train_size:train_size + val_size]
        test = samples[train_size + val_size:]

        logger.info(f"Dataset split: train={len(train)}, val={len(val)}, test={len(test)}")

        return train, val, test

    def save_metadata(
        self,
        train: List[str],
        val: List[str],
        test: List[str],
        output_dir: str
    ):
        """Save dataset metadata."""
        metadata = {
            'train': train,
            'val': val,
            'test': test,
            'config': {
                'sample_rate': self.audio_config.sample_rate,
                'sequence_length': self.train_config.sequence_length,
                'train_split': self.train_config.train_split,
                'val_split': self.train_config.val_split,
                'test_split': self.train_config.test_split,
            }
        }

        metadata_path = Path(output_dir) / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")


class DataPipeline:
    """Main data pipeline orchestrator."""

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        audio_config: Optional[AudioConfig] = None,
        aug_config: Optional[AugmentationConfig] = None,
        train_config: Optional[TrainingDataConfig] = None
    ):
        self.source_dir = source_dir
        self.output_dir = output_dir

        self.preparator = TrainingDataPreparator(
            audio_config=audio_config,
            aug_config=aug_config,
            train_config=train_config
        )

    def run(
        self,
        extract_features: bool = True,
        augment: bool = True,
        num_augmented: int = 3,
        create_splits: bool = True
    ):
        """Run the complete data pipeline."""
        logger.info("Starting data pipeline...")

        # Load audio files
        audio_files = self.preparator.load_dataset(self.source_dir)
        if not audio_files:
            logger.warning(f"No audio files found in {self.source_dir}")
            return

        # Create samples
        samples = self.preparator.create_samples(
            audio_files,
            self.output_dir,
            extract_features=extract_features,
            augment=augment,
            num_augmented=num_augmented
        )

        # Create train/val/test splits
        if create_splits:
            all_samples = samples['original'] + samples['augmented']
            train, val, test = self.preparator.split_dataset(all_samples)
            self.preparator.save_metadata(train, val, test, self.output_dir)

        logger.info("Data pipeline complete!")

        return {
            'original': len(samples['original']),
            'augmented': len(samples['augmented']),
            'features': len(samples['features'])
        }


def main():
    """CLI for data pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='AI DJ Data Pipeline')
    parser.add_argument('source', help='Source directory with audio files')
    parser.add_argument('output', help='Output directory for processed data')
    parser.add_argument('--sr', '--sample-rate', type=int, default=44100, dest='sample_rate')
    parser.add_argument('--seq-len', '--sequence-length', type=float, default=4.0, dest='sequence_length')
    parser.add_argument('--no-augment', action='store_true', help='Disable augmentation')
    parser.add_argument('--num-aug', type=int, default=3, dest='num_augmented', help='Number of augmented versions per sample')
    parser.add_argument('--no-features', action='store_true', help='Skip feature extraction')

    args = parser.parse_args()

    # Configure pipeline
    audio_config = AudioConfig(
        sample_rate=args.sample_rate,
        duration=30.0  # Max 30 seconds per file for training
    )

    train_config = TrainingDataConfig(
        sequence_length=args.sequence_length,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )

    # Run pipeline
    pipeline = DataPipeline(
        source_dir=args.source,
        output_dir=args.output,
        audio_config=audio_config,
        train_config=train_config
    )

    result = pipeline.run(
        extract_features=not args.no_features,
        augment=not args.no_augment,
        num_augmented=args.num_augmented
    )

    print(f"\nPipeline Results:")
    print(f"  Original samples: {result['original']}")
    print(f"  Augmented samples: {result['augmented']}")
    print(f"  Feature files: {result['features']}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
