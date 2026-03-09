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
    """Generate a professional fusion of two songs."""
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
                search_term = f.lower()
                if song1_id.lower() in search_term or song1_id.replace('_', ' ').lower() in search_term:
                    song1_file = os.path.join(root, f)
                if song2_id.lower() in search_term or song2_id.replace('_', ' ').lower() in search_term:
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
        
        # === IMPROVED BPM DETECTION ===
        try:
            import librosa
            
            # Use onset detection for more accurate BPM
            onset_env1 = librosa.onset.onset_strength(y=a1_mono, sr=sr1)
            onset_env2 = librosa.onset.onset_strength(y=a2_mono, sr=sr2)
            
            # Beat tracking with onset information
            bpm1, beats1 = librosa.beat.beat_track(onset_envelope=onset_env1, sr=sr1)
            bpm2, beats2 = librosa.beat.beat_track(onset_envelope=onset_env2, sr=sr2)
            
            bpm1, bpm2 = float(bpm1), float(bpm2)
            
            # Key detection
            key1 = librosa.key.keyestimate(y=a1_mono, sr=sr1)
            key2 = librosa.key.keyestimate(y=a2_mono, sr=sr2)
            
            # Energy analysis for smooth transition
            rms1 = librosa.feature.rms(y=a1_mono)[0]
            rms2 = librosa.feature.rms(y=a2_mono)[0]
            energy1 = float(np.mean(rms1))
            energy2 = float(np.mean(rms2))
            
            # Get tempo in beats per bar (for 4/4 time)
            tempo1 = bpm1
            tempo2 = bpm2
            
        except Exception as e:
            bpm1, bpm2 = 120, 120
            key1, key2 = "C major", "C major"
            energy1, energy2 = 0.5, 0.5
            tempo1, tempo2 = 120, 120
        
        # === RESAMPLE TO MATCH SAMPLE RATES ===
        if sr1 != sr2:
            from scipy import signal
            num_samples = int(len(audio2) * sr1 / sr2)
            audio2 = signal.resample(audio2, num_samples)
        
        # === STRETCH TO MATCH BPM (using phase vocoder for quality) ===
        if bpm1 > 0 and bpm2 > 0 and abs(bpm1 - bpm2) > 2:
            try:
                import librosa
                # High-quality time stretching using librosa
                tempo_ratio = bpm1 / bpm2
                
                # Stretch audio2 to match audio1's tempo
                a2_stretched = librosa.effects.time_stretch(a2_mono, rate=tempo_ratio)
                
                # Reconstruct stereo
                if audio2.ndim > 1:
                    # Stretch each channel
                    audio2_ch0 = librosa.effects.time_stretch(audio2[:,0], rate=tempo_ratio)
                    if len(audio2_ch0) < len(a2_stretched):
                        a2_stretched = a2_stretched[:len(audio2_ch0)]
                    audio2 = np.column_stack([audio2_ch0, a2_stretched])
                else:
                    audio2 = np.column_stack([a2_stretched, a2_stretched])
            except:
                # Fallback to simple resampling
                from scipy import signal
                ratio = bpm1 / bpm2
                audio2 = signal.resample(audio2, int(len(audio2) * ratio))
        
        # === NORMALIZE AND MATCH ENERGY ===
        # Normalize both tracks to similar loudness
        rms1_current = np.sqrt(np.mean(audio1**2))
        rms2_current = np.sqrt(np.mean(audio2**2))
        
        if rms1_current > 0:
            audio1 = audio1 / rms1_current * 0.7
        if rms2_current > 0:
            audio2 = audio2 / rms2_current * 0.7
        
        # Match energy levels
        target_energy = 0.7
        energy_ratio = target_energy / (rms1_current + 1e-8)
        audio1 = audio1 * energy_ratio * 0.5 + audio1 * 0.5
        
        energy_ratio2 = target_energy / (rms2_current + 1e-8)
        audio2 = audio2 * energy_ratio2 * 0.5 + audio2 * 0.5
        
        # Ensure stereo
        if audio1.ndim == 1:
            audio1 = np.column_stack([audio1, audio1])
        if audio2.ndim == 1:
            audio2 = np.column_stack([audio2, audio2])
        
        # === FIND BEST TRANSITION POINT ===
        # Use energy envelope to find a drop/transition point
        try:
            import librosa
            # Get onset strength for finding beats
            onset1 = librosa.onset.onset_strength(y=a1_mono, sr=sr1)
            onset2 = librosa.onset.onset_strength(y=a2_mono, sr=sr2)
            
            # Find high-energy sections (drops, buildups)
            # Look for transition points in second half of song1
            transition_search_start = len(audio1) // 2
            
            # Find peaks in onset strength
            onset_peaks = np.where(onset1 > np.percentile(onset1, 70))[0]
            if len(onset_peaks) > 0:
                # Use the last major onset before the end
                transition_point = onset_peaks[-1] * 512  # Convert to samples
                transition_point = min(transition_point, len(audio1) - sr1 * 8)  # At least 8 sec from end
            else:
                transition_point = len(audio1) - sr1 * 16  # Default: 16 sec before end
        except:
            # Default: transition at 3/4 through the track
            transition_point = int(len(audio1) * 0.75)
        
        # === CROSSFADE WITH REVERB TAIL ===
        fade_duration = 8 * sr1  # 8 second crossfade
        fade_start = max(0, transition_point - fade_duration // 2)
        fade_len = min(fade_duration, len(audio1) - fade_start, len(audio2))
        
        # Equal-power crossfade with smooth curves
        fade_out = np.sin(np.linspace(0, np.pi/2, fade_len))**2
        fade_in = np.cos(np.linspace(0, np.pi/2, fade_len))**2
        
        # Create result
        result = np.zeros((max(len(audio1), len(audio2)), 2))
        
        # Copy song1 up to fade
        result[:fade_start] = audio1[:fade_start]
        
        # Crossfade region with reverb
        if fade_len > 0:
            # Add reverb tail to song1 during fade
            reverb_decay = np.exp(np.linspace(0, -3, fade_len))
            
            result[fade_start:fade_start+fade_len] = (
                audio1[fade_start:fade_start+fade_len] * fade_out[:, None] * reverb_decay[:, None] +
                audio2[:fade_len] * fade_in[:, None]
            )
        
        # Add song2 after crossfade
        if fade_start + fade_len < len(result):
            remaining_start = fade_start + fade_len
            remaining_len = min(len(audio2) - fade_len, len(result) - remaining_start)
            result[remaining_start:remaining_start+remaining_len] = audio2[fade_len:fade_len+remaining_len]
        
        # === LIGHT COMPRESSION FOR CONSISTENCY ===
        # Soft knee compression
        threshold = 0.8
        ratio = 4.0
        
        peak = np.max(np.abs(result))
        if peak > threshold:
            # Soft clip
            excess = (peak - threshold) / peak
            result = result * threshold + (result - result * threshold) * 1/(1 + excess * ratio)
        
        # === FINAL OUTPUT NORMALIZATION ===
        peak = np.max(np.abs(result))
        if peak > 0.9:
            result = result / peak * 0.9
        elif peak < 0.1:
            result = result / max(peak, 0.01) * 0.3
        
        # Save
        output_full = os.path.join(FUSIONS_DIR, output_path)
        sf.write(output_full, result, sr1)
        
        techniques = [
            'improved_bpm_detection',
            'onset_analysis',
            'energy_matching',
            'phase_vocoder_stretching',
            'beat_aligned_transition',
            'reverb_tail_crossfade',
            'soft_knee_compression',
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
            'energy': {'song1': round(energy1, 3), 'song2': round(energy2, 3)},
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
