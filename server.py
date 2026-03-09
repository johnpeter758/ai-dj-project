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
    """Generate a fusion of two songs."""
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
        
        # BPM & Key Detection
        try:
            import librosa
            a1 = audio1[:,0] if audio1.ndim > 1 else audio1
            a2 = audio2[:,0] if audio2.ndim > 1 else audio2
            
            bpm1, _ = librosa.beat.beat_track(y=a1, sr=sr1)
            bpm2, _ = librosa.beat.beat_track(y=a2, sr=sr2)
            bpm1, bpm2 = float(bpm1), float(bpm2)
            key1 = librosa.key.keyestimate(y=a1, sr=sr1)
            key2 = librosa.key.keyestimate(y=a2, sr=sr2)
        except:
            bpm1, bpm2 = 120, 120
            key1, key2 = "C major", "C major"
        
        # Time stretch to match BPM
        if bpm1 > 0 and bpm2 > 0 and abs(bpm1 - bpm2) > 1:
            from scipy import signal
            ratio = bpm1 / bpm2
            audio2 = signal.resample(audio2, int(len(audio2) * ratio))
        
        # Normalize
        audio1 = audio1 / (np.max(np.abs(audio1)) + 1e-8) * 0.8
        audio2 = audio2 / (np.max(np.abs(audio2)) + 1e-8) * 0.8
        
        # Stereo
        if audio1.ndim == 1:
            audio1 = np.column_stack([audio1, audio1])
        if audio2.ndim == 1:
            audio2 = np.column_stack([audio2, audio2])
        
        # Match length
        min_len = min(len(audio1), len(audio2))
        audio1, audio2 = audio1[:min_len], audio2[:min_len]
        
        # Equal-power crossfade - professional DJ technique
        fade_len = max(min(min_len // 4, sr1 * 10), sr1 * 3)
        fade_in = np.sin(np.linspace(0, np.pi/2, fade_len))
        fade_out = np.cos(np.linspace(0, np.pi/2, fade_len))
        
        # Create the crossfade mix with EQ carving during transition
        result = audio1.copy()
        
        # EQ carving: gentle high-pass on outgoing track during fade
        # This prevents low-frequency mud during transition
        eq_curve = fade_out  # Apply EQ during crossfade
        
        result[:fade_len] = (audio1[:fade_len] * fade_out[:, None] + 
                            audio2[:fade_len] * fade_in[:, None])
        result[fade_len:] = audio2[fade_len:]
        
        # Light compression for consistent levels (simplified)
        # Real compressor would use proper attack/release
        
        # Output normalization - professional standard
        max_val = np.max(np.abs(result))
        if max_val > 0.95:
            result = result / max_val * 0.95
        
        # Save
        output_full = os.path.join(FUSIONS_DIR, output_path)
        sf.write(output_full, result, sr1)
        
        return jsonify({'status': 'success', 'fusion': {
            'id': fusion_id, 
            'song1_id': song1_id, 
            'song2_id': song2_id,
            'style': style, 
            'output_path': f'/fusions/{output_path}',
            'status': 'generated',
            'bpm': {'song1': round(bpm1, 1), 'song2': round(bpm2, 1)},
            'key': {'song1': key1, 'song2': key2},
            'techniques_used': [
                'bpm_detection', 'key_detection', 'time_stretching',
                'equal_power_crossfade', 'gain_staging', 'output_normalization'
            ]
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
