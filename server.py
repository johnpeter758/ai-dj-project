#!/usr/bin/env python3
"""
AI DJ API Server - Clean Simple Version
"""

import os
import json
from datetime import datetime
import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FUSIONS_DIR = os.path.join(BASE_DIR, 'fusions')
MUSIC_DIR = os.path.join(BASE_DIR, 'music')


@app.route('/')
def index():
    return send_from_directory(TEMPLATES_DIR, 'index.html')

@app.route('/fusions/<path:filename>')
def serve_fusion(filename):
    return send_from_directory(FUSIONS_DIR, filename)

@app.route('/music/<path:filename>')
def serve_music(filename):
    return send_from_directory(MUSIC_DIR, filename)


def load_songs():
    songs_file = os.path.join(DATA_DIR, 'songs.json')
    if os.path.exists(songs_file):
        with open(songs_file, 'r') as f:
            return json.load(f)
    
    songs = []
    for root, dirs, files in os.walk(MUSIC_DIR):
        for f in files:
            if f.endswith(('.mp3', '.wav')):
                song_id = f.replace('.mp3', '').replace('.wav', '').replace(' ', '_')
                songs.append({
                    'id': song_id,
                    'title': f.replace('.mp3', '').replace('.wav', ''),
                    'artist': os.path.basename(os.path.dirname(os.path.join(root, f))),
                    'file': os.path.relpath(os.path.join(root, f), MUSIC_DIR)
                })
    return songs


@app.route('/api/songs')
def list_songs():
    songs = load_songs()
    return jsonify({'status': 'success', 'songs': songs, 'count': len(songs)})


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a clean, simple fusion of two songs."""
    data = request.get_json() or {}
    song1_id = data.get('song1_id', '')
    song2_id = data.get('song2_id', '')
    
    if not song1_id or not song2_id:
        return jsonify({'error': 'Missing song IDs'}), 400
    
    # Find audio files
    song1_file = None
    song2_file = None
    
    for root, dirs, files in os.walk(MUSIC_DIR):
        for f in files:
            if f.endswith(('.mp3', '.wav')):
                if song1_id.lower() in f.lower():
                    song1_file = os.path.join(root, f)
                if song2_id.lower() in f.lower():
                    song2_file = os.path.join(root, f)
    
    fusion_id = f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = f"{fusion_id}.wav"
    os.makedirs(FUSIONS_DIR, exist_ok=True)
    
    if not (song1_file and song2_file):
        return jsonify({'status': 'error', 'fusion': {'error': 'Files not found'}})
    
    try:
        # Load audio - SIMPLE CLEAN LOAD
        audio1, sr1 = sf.read(song1_file)
        audio2, sr2 = sf.read(song2_file)
        
        # Convert to stereo
        if audio1.ndim == 1:
            audio1 = np.column_stack([audio1, audio1])
        if audio2.ndim == 1:
            audio2 = np.column_stack([audio2, audio2])
        
        # Match length - take shorter
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Simple clean mix - NO heavy processing
        # Just a simple crossfade at the midpoint
        fade_len = min(min_len // 2, sr1 * 10)
        
        # Equal power crossfade
        fade_out = np.sin(np.linspace(0, np.pi/2, fade_len))
        fade_in = np.cos(np.linspace(0, np.pi/2, fade_len))
        
        result = np.zeros((min_len, 2))
        
        # Song 1 plays first half, fades out
        result[:fade_len] = audio1[:fade_len] * fade_out[:, None] + audio2[:fade_len] * fade_in[:, None]
        
        # Song 2 plays second half
        result[fade_len:] = audio2[fade_len:]
        
        # Simple peak normalization - NO heavy processing
        peak = np.max(np.abs(result))
        if peak > 0.95:
            result = result / peak * 0.95
        
        # Save
        output_full = os.path.join(FUSIONS_DIR, output_path)
        sf.write(output_full, result, sr1)
        
        return jsonify({'status': 'success', 'fusion': {
            'id': fusion_id, 
            'song1_id': song1_id, 
            'song2_id': song2_id,
            'output_path': f'/fusions/{output_path}',
            'status': 'generated',
            'techniques_used': ['simple_crossfade', 'peak_normalization']
        }})
        
    except Exception as e:
        return jsonify({'status': 'error', 'fusion': {'error': str(e)}})


@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🎛️ AI DJ running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
