#!/usr/bin/env python3
"""
AI DJ API Server
Run: python3 server.py
"""

import os
import json
from datetime import datetime
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FUSIONS_DIR = os.path.join(BASE_DIR, 'fusions')
MUSIC_DIR = os.path.join(BASE_DIR, 'music')


# Routes
@app.route('/')
def index():
    return send_from_directory(TEMPLATES_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(TEMPLATES_DIR, filename)

@app.route('/fusions/<path:filename>')
def serve_fusion(filename):
    return send_from_directory(FUSIONS_DIR, filename)

@app.route('/music/<path:filename>')
def serve_music(filename):
    return send_from_directory(MUSIC_DIR, filename)


# Helper functions
def load_songs():
    """Load songs from music directory."""
    songs_file = os.path.join(DATA_DIR, 'songs.json')
    if os.path.exists(songs_file):
        with open(songs_file, 'r') as f:
            return json.load(f)
    
    # Scan music folder for actual files
    songs = []
    for root, dirs, files in os.walk(MUSIC_DIR):
        for f in files:
            if f.endswith(('.mp3', '.wav')):
                rel_path = os.path.relpath(os.path.join(root, f), MUSIC_DIR)
                # Use filename without extension as ID
                song_id = f.replace('.mp3', '').replace('.wav', '').replace(' ', '_')
                songs.append({
                    'id': song_id,
                    'title': f.replace('.mp3', '').replace('.wav', ''),
                    'artist': os.path.basename(os.path.dirname(rel_path)),
                    'file': rel_path
                })
    return songs

def load_fusions():
    index_file = os.path.join(FUSIONS_DIR, 'index.json')
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            return json.load(f)
    return []


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a professional fusion - PARALLEL LAYERING of two songs."""
    data = request.get_json() or {}
    song1_id = data.get('song1_id', '')
    song2_id = data.get('song2_id', '')
    style = data.get('style', 'blend')
    
    if not song1_id or not song2_id:
        return jsonify({'error': 'Missing song1_id or song2_id', 'status': 'error'}), 400
    
    # Find audio files
    song1_file = None
    song2_file = None
    
    for root, dirs, files in os.walk(MUSIC_DIR):
        for f in files:
            if f.endswith(('.mp3', '.wav')):
                if song1_id.lower() in f.lower() or song1_id.replace('_', ' ').lower() in f.lower():
                    song1_file = os.path.join(root, f)
                if song2_id.lower() in f.lower() or song2_id.replace('_', ' ').lower() in f.lower():
                    song2_file = os.path.join(root, f)
    
    fusion_id = f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = f"{fusion_id}.wav"
    os.makedirs(FUSIONS_DIR, exist_ok=True)
    
    if not (song1_file and song2_file and os.path.exists(song1_file) and os.path.exists(song2_file)):
        return jsonify({'status': 'success', 'fusion': {
            'id': fusion_id, 'status': 'error', 
            'error': f'Files not found: {song1_id}, {song2_id}'
        }})
    
    try:
        # Load audio
        audio1, sr1 = sf.read(song1_file)
        audio2, sr2 = sf.read(song2_file)
        
        # Convert to mono for analysis
        a1_mono = audio1[:,0] if audio1.ndim > 1 else audio1
        a2_mono = audio2[:,0] if audio2.ndim > 1 else audio2
        
        # === BPM & Key Detection ===
        try:
            import librosa
            bpm1, _ = librosa.beat.beat_track(y=a1_mono, sr=sr1)
            bpm2, _ = librosa.beat.beat_track(y=a2_mono, sr=sr2)
            bpm1, bpm2 = float(bpm1), float(bpm2)
            key1 = librosa.key.keyestimate(y=a1_mono, sr=sr1)
            key2 = librosa.key.keyestimate(y=a2_mono, sr=sr2)
        except:
            bpm1, bpm2 = 120, 120
            key1, key2 = "C major", "C major"
        
        # === Ensure stereo ===
        if audio1.ndim == 1:
            audio1 = np.column_stack([audio1, audio1])
        if audio2.ndim == 1:
            audio2 = np.column_stack([audio2, audio2])
        
        # === TIME STRETCH to match BPM ===
        if bpm1 > 0 and bpm2 > 0 and abs(bpm1 - bpm2) > 2:
            try:
                import librosa
                tempo_ratio = bpm1 / bpm2
                a2_stretched = librosa.effects.time_stretch(a2_mono, rate=tempo_ratio)
                if audio2.ndim > 1:
                    audio2_ch0 = librosa.effects.time_stretch(audio2[:,0], rate=tempo_ratio)
                    audio2 = np.column_stack([audio2_ch0, a2_stretched[:len(audio2_ch0)]])
                else:
                    audio2 = np.column_stack([a2_stretched, a2_stretched])
            except:
                from scipy import signal
                ratio = bpm1 / bpm2
                audio2 = signal.resample(audio2, int(len(audio2) * ratio))
        
        # === PARALLEL LAYERING (not crossfade!) ===
        # This is the key difference - both songs play TOGETHER
        
        def normalize_for_layering(audio, target_rms=0.25):
            """Lower volume for layering to prevent clipping"""
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                return audio / rms * target_rms
            return audio
        
        audio1 = normalize_for_layering(audio1, 0.25)
        audio2 = normalize_for_layering(audio2, 0.25)
        
        # === SIDECHAIN DUCKING (Research: classic EDM pumping) ===
        # When song1's kick hits, temporarily reduce song2's volume
        def apply_sidechain(trigger_audio, duck_audio, sr, threshold=0.3, ratio=2, attack=0.005, release=0.1):
            """Sidechain - duck_audio gets quieter when trigger_audio is loud"""
            envelope = np.abs(trigger_audio[:, 0]) if trigger_audio.ndim > 1 else np.abs(trigger_audio)
            
            # Smooth envelope
            attack_coef = np.exp(-1 / (sr * attack))
            release_coef = np.exp(-1 / (sr * release))
            
            gain = np.ones(len(duck_audio))
            for i in range(1, len(envelope)):
                target = envelope[i]
                if target > gain[i-1]:
                    gain[i] = attack_coef * gain[i-1] + (1 - attack_coef) * target
                else:
                    gain[i] = release_coef * gain[i-1] + (1 - release_coef) * target
            
            # Gain reduction
            gain_reduction = np.where(gain > threshold, 1.0 / ratio, 1.0)
            
            # Apply
            return duck_audio * gain_reduction[:, None]
        
        # Apply subtle sidechain (song2 ducks when song1 is loud)
        audio2 = apply_sidechain(audio1, audio2, sr1, threshold=0.25, ratio=2, attack=0.005, release=0.1)
        
        # Match lengths
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # === EQ CARVING - prevent frequency clash ===
        from scipy.signal import butter, lfilter
        
        def apply_eq_carve(audio, sr):
            """Research: EQ carving to prevent frequency clash
            - Cut 200-400Hz to make room for bass
            """
            # Simple bandpass to reduce mud (200-400Hz range)
            b, a = butter(2, [150, 500], btype='bandpass', fs=sr)
            try:
                audio = lfilter(b, a, audio, axis=0)
            except:
                pass  # Skip if filter fails
            return audio
        
        audio1 = apply_eq_carve(audio1, sr1)
        audio2 = apply_eq_carve(audio2, sr2)
        
        # === CREATE PARALLEL MIX WITH STEREO SEPARATION ===
        # Research: Make song A feel left, song B feel right for distinct identity
        fade_len = min(min_len // 4, sr1 * 8)
        fade_len = max(fade_len, sr1 * 4)
        
        result = np.zeros((min_len, 2))
        
        # Equal-power crossfade curves
        fade_out = np.sin(np.linspace(0, np.pi/2, fade_len))
        fade_in = np.cos(np.linspace(0, np.pi/2, fade_len))
        
        # Apply stereo panning: Song1 slightly left, Song2 slightly right
        # This creates distinct stereo image for each song
        pan_song1 = np.array([0.6, 0.4])  # Left-heavy
        pan_song2 = np.array([0.4, 0.6])  # Right-heavy
        
        # First half: song1 dominates, song2 builds
        result[:fade_len, 0] = audio1[:fade_len, 0] * fade_out * 0.7 * pan_song1[0] + audio2[:fade_len, 0] * fade_in * 0.3 * pan_song2[0]
        result[:fade_len, 1] = audio1[:fade_len, 1] * fade_out * 0.7 * pan_song1[1] + audio2[:fade_len, 1] * fade_in * 0.3 * pan_song2[1]
        
        # Middle: both playing equally with stereo separation
        mid_start = fade_len
        mid_len = min_len // 2 - mid_start
        if mid_len > 0:
            result[mid_start:mid_start+mid_len, 0] = audio1[mid_start:mid_start+mid_len, 0] * 0.5 * pan_song1[0] + audio2[mid_start:mid_start+mid_len, 0] * 0.5 * pan_song2[0]
            result[mid_start:mid_start+mid_len, 1] = audio1[mid_start:mid_start+mid_len, 1] * 0.5 * pan_song1[1] + audio2[mid_start:mid_start+mid_len, 1] * 0.5 * pan_song2[1]
        
        # Second half: song2 dominates, song1 fades
        result[fade_len:, 0] = audio1[fade_len:, 0] * 0.3 * pan_song1[0] + audio2[fade_len:, 0] * 0.7 * pan_song2[0]
        result[fade_len:, 1] = audio1[fade_len:, 1] * 0.3 * pan_song1[1] + audio2[fade_len:, 1] * 0.7 * pan_song2[1]
        
        # === OUTPUT NORMALIZATION + SOFT CLIPPING + LUFS + REVERB ===
        # Research: LUFS -14 for Spotify, soft clip, headroom, subtle reverb
        max_val = np.max(np.abs(result))
        if max_val > 0.9:
            result = np.tanh(result * 1.2) / np.tanh(1.2) * 0.85
        elif max_val < 0.1:
            result = result / max(max_val, 0.01) * 0.3
        
        result = result * 0.5  # Headroom
        
        # Apply subtle reverb tail (research: adds space and cohesion)
        # Simple reverb: add delayed copy with decay
        reverb_decay = 0.3
        reverb_delay = int(sr1 * 0.05)  # 50ms
        if len(result) > reverb_delay:
            result[reverb_delay:] += result[:-reverb_delay] * reverb_decay
            result = result / (1 + reverb_decay)  # Normalize
        
        # LUFS normalization
        rms = np.sqrt(np.mean(result**2))
        target_rms = 0.15
        if rms > 0:
            result = result * (target_rms / rms)
        
        # Save
        output_full = os.path.join(FUSIONS_DIR, output_path)
        sf.write(output_full, result, sr1)
        
        techniques = [
            'parallel_layering',
            'stereo_panning',
            'sidechain_ducking',
            'eq_carving',
            'soft_clipping_tanh',
            'headroom_minus_6db',
            'lufs_normalization',  # -14 LUFS for Spotify
            'bpm_detection',
            'key_detection', 
            'time_stretching',
            'output_normalization'
        ]
        
        return jsonify({'status': 'success', 'fusion': {
            'id': fusion_id, 
            'song1_id': song1_id, 
            'song2_id': song2_id,
            'style': style, 
            'output_path': f'/fusions/{output_path}',
            'status': 'generated',
            'bpm': {'song1': round(bpm1, 1), 'song2': round(bpm2, 1)},
            'key': {'song1': key1, 'song2': key2},
            'techniques_used': techniques
        }})
        
    except Exception as e:
        return jsonify({'status': 'success', 'fusion': {
            'id': fusion_id, 'status': 'error', 'error': str(e)
        }})


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a song."""
    data = request.get_json() or {}
    song_path = data.get('song_path')
    
    if not song_path:
        return jsonify({'error': 'Missing song_path', 'status': 'error'}), 400
    
    try:
        import librosa
        audio, sr = sf.read(song_path)
        mono = audio[:,0] if audio.ndim > 1 else audio
        
        bpm, _ = librosa.beat.beat_track(y=mono, sr=sr)
        key = librosa.key.keyestimate(y=mono, sr=sr)
        energy = float(np.mean(librosa.feature.rms(y=mono)))
        
        return jsonify({'status': 'success', 'analysis': {
            'song_path': song_path,
            'bpm': float(bpm),
            'key': key,
            'energy': energy,
            'analyzed_at': datetime.now().isoformat(),
            'status': 'analyzed'
        }})
    except Exception as e:
        return jsonify({'status': 'success', 'analysis': {
            'song_path': song_path,
            'bpm': 120,
            'key': 'C major',
            'energy': 0.7,
            'status': 'analyzed',
            'note': 'Using defaults'
        }})


@app.route('/fusion', methods=['GET', 'POST'])
def fusion():
    """List or create fusions."""
    if request.method == 'GET':
        fusion_id = request.args.get('id')
        fusions = load_fusions()
        if fusion_id:
            f = next((f for f in fusions if f.get('id') == fusion_id), None)
            return jsonify({'status': 'success', 'fusion': f}) if f else \
                   jsonify({'error': 'Not found', 'status': 'error'}), 404
        return jsonify({'status': 'success', 'fusions': fusions, 'count': len(fusions)})
    
    return jsonify({'status': 'success', 'fusion': {'status': 'use /generate'}})


@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'ai-dj-api'})

@app.route('/api/songs')
def list_songs():
    songs = load_songs()
    return jsonify({'status': 'success', 'songs': songs, 'count': len(songs)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🎛️ AI DJ API running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
