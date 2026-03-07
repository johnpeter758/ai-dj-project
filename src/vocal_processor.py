"""
Vocal Processing Chain
A numpy-based audio processing pipeline for vocals.

Chain:
1. De-esser (first)
2. EQ (high-pass, presence boost)
3. Compressor (gentle, 2-3dB reduction)
4. Reverb (small room)
5. Delay
6. De-esser (second)
7. Example usage
"""

import numpy as np
from scipy import signal


class DeEsser:
    """Dynamic de-esser to reduce sibilance."""
    
    def __init__(self, sample_rate: int = 44100, threshold: float = -20.0, 
                 ratio: float = 4.0, attack_ms: float = 5.0, release_ms: float = 50.0):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.ratio = ratio
        self.attack = np.exp(-1 / (attack_ms * sample_rate / 1000))
        self.release = np.exp(-1 / (release_ms * sample_rate / 1000))
        
        # High-pass filter to detect sibilance (5-10kHz)
        self.highpass_b, self.highpass_a = signal.butter(4, 5000, 'high', fs=sample_rate)
        
        # Smoothing filter for gain envelope
        self.smooth_b, self.smooth_a = signal.butter(2, 100, 'low', fs=sample_rate)
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply de-essing to audio signal."""
        # Detect sibilance with high-pass
        sibilance = signal.filtfilt(self.highpass_b, self.highpass_a, audio)
        
        # Compute envelope
        envelope = np.abs(sibilance)
        
        # Apply smoothing to envelope
        envelope = signal.filtfilt(self.smooth_b, self.smooth_a, envelope)
        
        # Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)
        
        # Compute gain reduction
        over_threshold = np.maximum(envelope_db - self.threshold, 0)
        gain_reduction_db = over_threshold * (1 - 1 / self.ratio)
        
        # Convert back to linear gain
        gain = 10 ** (-gain_reduction_db / 20)
        
        # Apply envelope following
        output = np.zeros_like(audio)
        gain_env = 0
        
        for i in range(len(audio)):
            if envelope[i] > envelope[i-1] if i > 0 else False:
                gain_env = self.attack * gain_env + (1 - self.attack) * gain[i]
            else:
                gain_env = self.release * gain_env + (1 - self.release) * gain[i]
            output[i] = audio[i] * gain_env
            
        return output


class EQ:
    """Parametric EQ with high-pass filter and presence boost."""
    
    def __init__(self, sample_rate: int = 44100, 
                 high_pass_hz: float = 80.0,
                 presence_hz: float = 3000.0,
                 presence_boost_db: float = 3.0):
        self.sample_rate = sample_rate
        
        # High-pass filter
        self.highpass_b, self.highpass_a = signal.butter(4, high_pass_hz, 'high', fs=sample_rate)
        
        # Presence boost - peak filter (using iirpeak)
        # Center frequency, Q factor, gain in dB
        Q = 1.0
        w0 = 2 * np.pi * presence_hz / sample_rate
        alpha = np.sin(w0) / (2 * Q)
        A = 10 ** (presence_boost_db / 40)
        
        b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
        a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
        self.presence_b, self.presence_a = np.array(b), np.array(a)
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply EQ processing."""
        # Apply high-pass
        audio = signal.filtfilt(self.highpass_b, self.highpass_a, audio)
        
        # Apply presence boost
        audio = signal.filtfilt(self.presence_b, self.presence_a, audio)
        
        return audio


class Compressor:
    """Gentle dynamics compressor (2-3dB reduction target)."""
    
    def __init__(self, sample_rate: int = 44100, 
                 threshold_db: float = -18.0,
                 ratio: float = 2.0,
                 attack_ms: float = 10.0,
                 release_ms: float = 100.0,
                 knee_db: float = 6.0,
                 makeup_gain_db: float = 2.0):
        self.sample_rate = sample_rate
        self.threshold = 10 ** (threshold_db / 20)
        self.ratio = ratio
        self.knee = 10 ** (knee_db / 20)
        self.attack = np.exp(-1 / (attack_ms * sample_rate / 1000))
        self.release = np.exp(-1 / (release_ms * sample_rate / 1000))
        self.makeup = 10 ** (makeup_gain_db / 20)
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply compression to audio."""
        # Compute input envelope
        envelope = np.abs(audio)
        
        # Apply soft knee compression
        output = np.zeros_like(audio)
        gain_env = 1.0
        
        for i in range(len(audio)):
            # Input level in linear
            level = envelope[i]
            
            # Soft knee computation
            if level < self.threshold - self.knee / 2:
                # Below knee - no compression
                target_gain = 1.0
            elif level > self.threshold + self.knee / 2:
                # Above knee - full ratio
                excess = level - self.threshold
                target_gain = 1.0 / (1 + (self.ratio - 1) * excess / level)
            else:
                # In knee region
                excess = level - (self.threshold - self.knee / 2)
                soft_ratio = 1 + (self.ratio - 1) * (excess / self.knee) ** 2
                target_gain = 1.0 / soft_ratio
            
            # Envelope following
            if target_gain < gain_env:
                gain_env = self.attack * gain_env + (1 - self.attack) * target_gain
            else:
                gain_env = self.release * gain_env + (1 - self.release) * target_gain
            
            output[i] = audio[i] * gain_env * self.makeup
            
        return output


class Reverb:
    """Small room reverb using Schroeder reverb algorithm."""
    
    def __init__(self, sample_rate: int = 44100, 
                 room_size: float = 0.5,
                 decay: float = 0.4,
                 wet_dry: float = 0.3):
        self.sample_rate = sample_rate
        self.wet_dry = wet_dry
        
        # Comb filter delays (in samples) - typical for small room
        self.comb_delays = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
        self.comb_delays = [int(d * room_size) for d in self.comb_delays]
        
        # Allpass filter delays
        self.allpass_delays = [225, 556, 441, 341]
        
        # Initialize comb filter buffers
        self.comb_buffers = [np.zeros(d) for d in self.comb_delays]
        self.comb_indices = [0] * len(self.comb_delays)
        
        # Initialize allpass filter buffers
        self.allpass_buffers = [np.zeros(d) for d in self.allpass_delays]
        self.allpass_indices = [0] * len(self.allpass_delays)
        
        # Decay factor for comb filters
        self.decay = decay
        
        # Feedback for allpass
        self.allpass_feedback = 0.5
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply reverb to audio."""
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            input_sample = audio[i]
            
            # Parallel comb filters
            comb_sum = 0
            for j, (buffer, idx) in enumerate(zip(self.comb_buffers, self.comb_indices)):
                delayed = buffer[idx]
                buffer[idx] = input_sample + delayed * self.decay
                comb_sum += delayed
                self.comb_indices[j] = (idx + 1) % len(buffer)
            
            comb_out = comb_sum / len(self.comb_delays)
            
            # Series allpass filters
            allpass_out = comb_out
            for j, (buffer, idx) in enumerate(zip(self.allpass_buffers, self.allpass_indices)):
                delayed = buffer[idx]
                buffer[idx] = allpass_out + delayed * self.allpass_feedback
                allpass_out = delayed - allpass_out * self.allpass_feedback
                self.allpass_indices[j] = (idx + 1) % len(buffer)
            
            # Mix wet and dry
            output[i] = input_sample * (1 - self.wet_dry) + allpass_out * self.wet_dry
            
        return output


class Delay:
    """Simple delay effect."""
    
    def __init__(self, sample_rate: int = 44100, 
                 delay_ms: float = 150.0,
                 feedback: float = 0.3,
                 wet_dry: float = 0.25):
        self.sample_rate = sample_rate
        self.delay_samples = int(delay_ms * sample_rate / 1000)
        self.feedback = feedback
        self.wet_dry = wet_dry
        
        self.buffer = np.zeros(self.delay_samples)
        self.index = 0
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply delay to audio."""
        output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Read from buffer
            delayed = self.buffer[self.index]
            
            # Write to buffer with feedback
            self.buffer[self.index] = audio[i] + delayed * self.feedback
            
            # Mix wet and dry
            output[i] = audio[i] * (1 - self.wet_dry) + delayed * self.wet_dry
            
            # Advance buffer index
            self.index = (self.index + 1) % self.delay_samples
            
        return output


class VocalProcessor:
    """Complete vocal processing chain."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # Initialize processors in order
        self.deesser1 = DeEsser(sample_rate)
        self.eq = EQ(sample_rate)
        self.compressor = Compressor(sample_rate)
        self.reverb = Reverb(sample_rate, room_size=0.4, wet_dry=0.2)
        self.delay = Delay(sample_rate, delay_ms=120, feedback=0.25, wet_dry=0.2)
        self.deesser2 = DeEsser(sample_rate, threshold=-15, ratio=3.0)
        
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through the full chain."""
        # 1. First De-esser
        audio = self.deesser1.process(audio)
        
        # 2. EQ (high-pass + presence boost)
        audio = self.eq.process(audio)
        
        # 3. Compressor (gentle, 2-3dB reduction)
        audio = self.compressor.process(audio)
        
        # 4. Reverb (small room)
        audio = self.reverb.process(audio)
        
        # 5. Delay
        audio = self.delay.process(audio)
        
        # 6. Second De-esser
        audio = self.deesser2.process(audio)
        
        return audio
    
    def process_file(self, input_path: str, output_path: str):
        """Process an audio file."""
        try:
            import soundfile as sf
            audio, sr = sf.read(input_path)
            
            # Convert to stereo if mono
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
                audio = np.concatenate([audio, audio], axis=1)
            
            # Process each channel
            processed = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                processed[:, ch] = self.process(audio[:, ch])
            
            # Normalize to prevent clipping
            max_val = np.abs(processed).max()
            if max_val > 0.95:
                processed = processed * 0.95 / max_val
                
            sf.write(output_path, processed, sr)
            print(f"Processed audio saved to {output_path}")
            
        except ImportError:
            print("soundfile not installed. Install with: pip install soundfile")
        except Exception as e:
            print(f"Error processing file: {e}")


def generate_test_tone(duration: float = 2.0, sample_rate: int = 44100) -> np.ndarray:
    """Generate a test audio signal with speech-like frequencies."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Fundamental frequency (male voice ~120Hz)
    fundamental = 120
    
    # Mix of harmonics
    audio = (np.sin(2 * np.pi * fundamental * t) +
             0.5 * np.sin(2 * np.pi * fundamental * 2 * t) +
             0.3 * np.sin(2 * np.pi * fundamental * 3 * t) +
             0.2 * np.sin(2 * np.pi * fundamental * 4 * t))
    
    # Add sibilance (high frequency content)
    sibilance = 0.3 * np.sin(2 * np.pi * 7000 * t) * (np.random.rand(len(t)) * 0.5 + 0.5)
    audio += sibilance
    
    # Apply envelope (like speech)
    envelope = np.exp(-t * 2) + 0.3 * np.sin(2 * np.pi * 2 * t) ** 2
    audio = audio * envelope
    
    # Normalize
    audio = audio / np.abs(audio).max() * 0.7
    
    return audio.astype(np.float32)


def example_usage():
    """Demonstrate the vocal processor."""
    print("=" * 50)
    print("Vocal Processing Chain - Example Usage")
    print("=" * 50)
    
    sample_rate = 44100
    
    # Generate test signal
    print("\n1. Generating test audio...")
    test_audio = generate_test_tone(duration=3.0, sample_rate=sample_rate)
    print(f"   Generated {len(test_audio)} samples ({len(test_audio)/sample_rate:.2f}s)")
    
    # Process through chain
    print("\n2. Creating vocal processor...")
    processor = VocalProcessor(sample_rate=sample_rate)
    
    print("\n3. Processing audio through chain:")
    print("   Step 1: De-esser (first)")
    print("   Step 2: EQ (high-pass @ 80Hz, presence boost @ 3kHz)")
    print("   Step 3: Compressor (threshold: -18dB, ratio: 2:1)")
    print("   Step 4: Reverb (small room)")
    print("   Step 5: Delay (120ms)")
    print("   Step 6: De-esser (second)")
    
    processed_audio = processor.process(test_audio)
    
    # Analyze results
    print("\n4. Analysis:")
    print(f"   Input RMS:  {np.sqrt(np.mean(test_audio**2)):.4f}")
    print(f"   Output RMS: {np.sqrt(np.mean(processed_audio**2)):.4f}")
    print(f"   Peak Input:  {np.abs(test_audio).max():.4f}")
    print(f"   Peak Output: {np.abs(processed_audio).max():.4f}")
    
    # Save output (if soundfile available)
    try:
        import soundfile as sf
        sf.write('/tmp/vocal_processed.wav', processed_audio, sample_rate)
        print("\n5. Saved processed audio to /tmp/vocal_processed.wav")
    except ImportError:
        print("\n5. (Install soundfile to save output: pip install soundfile)")
    
    print("\n" + "=" * 50)
    print("Example complete!")
    print("=" * 50)
    
    return processed_audio


if __name__ == "__main__":
    example_usage()
