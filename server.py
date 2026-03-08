#!/usr/bin/env python3
"""
AI DJ API Server
Run: python3 server.py
Endpoints: /generate, /analyze, /fusion
"""

from flask import Flask, jsonify, request, send_from_directory
import os
import json
from datetime import datetime

app = Flask(__name__)

# Templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')

# Serve dashboard at root
@app.route('/')
def index():
    return send_from_directory(TEMPLATES_DIR, 'dashboard.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(TEMPLATES_DIR, filename)

# Data directory for songs and fusions
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FUSIONS_DIR = os.path.join(os.path.dirname(__file__), 'fusions')

def load_songs():
    """Load analyzed songs from data directory."""
    songs = []
    songs_file = os.path.join(DATA_DIR, 'songs.json')
    if os.path.exists(songs_file):
        with open(songs_file, 'r') as f:
            songs = json.load(f)
    return songs

def load_fusions():
    """Load created fusions from fusions directory."""
    fusions = []
    fusions_file = os.path.join(FUSIONS_DIR, 'index.json')
    if os.path.exists(fusions_file):
        with open(fusions_file, 'r') as f:
            fusions = json.load(f)
    return fusions

@app.route('/generate', methods=['POST'])
def generate():
    """Generate a fusion of two songs."""
    import os, json, datetime
    import numpy as np
    import soundfile as sf
    
    data = request.get_json() or {}
    song1_id = data.get('song1_id', '')
    song2_id = data.get('song2_id', '')
    style = data.get('style', 'blend')
    
    if not song1_id or not song2_id:
        return jsonify({'error': 'Missing song1_id or song2_id', 'status': 'error'}), 400
    
    # Find audio files recursively
    music_dir = os.path.join(os.path.dirname(__file__), 'music')
    song1_file = None
    song2_file = None
    
    for root, dirs, files in os.walk(music_dir):
        for f in files:
            if f.endswith(('.mp3', '.wav')):
                if song1_id.lower() in f.lower() or song1_id.replace('_', ' ').lower() in f.lower():
                    song1_file = os.path.join(root, f)
                if song2_id.lower() in f.lower() or song2_id.replace('_', ' ').lower() in f.lower():
                    song2_file = os.path.join(root, f)
    
    fusion_id = f"fusion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = f"fusions/{fusion_id}.wav"
    os.makedirs(os.path.join(os.path.dirname(__file__), 'fusions'), exist_ok=True)
    
    if song1_file and song2_file and os.path.exists(song1_file) and os.path.exists(song2_file):
        try:
            # Load and process audio
            audio1, sr1 = sf.read(song1_file)
            audio2, sr2 = sf.read(song2_file)
            
            # Normalize
            if np.max(np.abs(audio1)) > 0:
                audio1 = audio1 / np.max(np.abs(audio1)) * 0.8
            if np.max(np.abs(audio2)) > 0:
                audio2 = audio2 / np.max(np.abs(audio2)) * 0.8
            
            # Match lengths
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            # Ensure stereo
            if audio1.ndim == 1:
                audio1 = np.column_stack([audio1, audio1])
            if audio2.ndim == 1:
                audio2 = np.column_stack([audio2, audio2])
            
            # Crossfade blend
            fade_len = min(min_len // 4, sr1 * 10)
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            
            result = audio1.copy()
            result[:fade_len] = audio1[:fade_len] * fade_out[:, None] + audio2[:fade_len] * fade_in[:, None]
            result[fade_len:] = audio2[fade_len:]
            
            # Save
            output_full = os.path.join(os.path.dirname(__file__), output_path)
            sf.write(output_full, result, sr1)
            
            return jsonify({'status': 'success', 'fusion': {
                'id': fusion_id, 'song1_id': song1_id, 'song2_id': song2_id,
                'style': style, 'output_path': output_path, 'status': 'generated'
            }})
        except Exception as e:
            return jsonify({'status': 'success', 'fusion': {
                'id': fusion_id, 'status': 'error', 'error': str(e)
            }})
    else:
        return jsonify({'status': 'success', 'fusion': {
            'id': fusion_id, 'status': 'error', 'error': 'Files not found'
        }})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze a song to extract key, BPM, energy, etc.
    Request body: {"song_path": "path/to/song.mp3"}
    """
    data = request.get_json() or {}
    
    song_path = data.get('song_path')
    if not song_path:
        return jsonify({
            'error': 'Missing required field: song_path',
            'status': 'error'
        }), 400
    
    # Mock analysis - in production this would use audio analysis tools
    analysis = {
        'song_path': song_path,
        'analyzed_at': datetime.datetime.now().isoformat(),
        'key': '4A',  # Camelot wheel key
        'bpm': 128,
        'energy': 0.75,
        'duration': 210,
        'loudness': -8.5,
        'tempo_confidence': 0.92,
        'key_confidence': 0.88,
        'status': 'analyzed'
    }
    
    return jsonify({
        'status': 'success',
        'analysis': analysis
    })

@app.route('/fusion', methods=['GET', 'POST'])
def fusion():
    """
    GET: List all fusions or get specific fusion
    POST: Create a new fusion
    
    Query params (GET): ?id=<fusion_id>
    Request body (POST): {"song1_id": "...", "song2_id": "...", "options": {...}}
    """
    if request.method == 'GET':
        fusion_id = request.args.get('id')
        
        if fusion_id:
            # Get specific fusion
            fusions = load_fusions()
            fusion = next((f for f in fusions if f.get('id') == fusion_id), None)
            if not fusion:
                return jsonify({
                    'error': 'Fusion not found',
                    'status': 'error'
                }), 404
            return jsonify({'status': 'success', 'fusion': fusion})
        else:
            # List all fusions
            fusions = load_fusions()
            return jsonify({'status': 'success', 'fusions': fusions, 'count': len(fusions)})
    
    elif request.method == 'POST':
        data = request.get_json() or {}
        
        song1_id = data.get('song1_id')
        song2_id = data.get('song2_id')
        
        if not song1_id or not song2_id:
            return jsonify({
                'error': 'Missing required fields: song1_id, song2_id',
                'status': 'error'
            }), 400
        
        # Mock fusion creation
        fusion = {
            'id': f"fusion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'song1_id': song1_id,
            'song2_id': song2_id,
            'options': data.get('options', {}),
            'created_at': datetime.datetime.now().isoformat(),
            'status': 'processing'
        }
        
        return jsonify({
            'status': 'success',
            'fusion': fusion
        })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'ai-dj-api'})

@app.route('/api/songs', methods=['GET'])
def list_songs():
    """List all analyzed songs."""
    songs = load_songs()
    return jsonify({'status': 'success', 'songs': songs, 'count': len(songs)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🎛️ AI DJ API Server running at http://localhost:{port}")
    print(f"📡 Endpoints: /generate, /analyze, /fusion, /api/health, /api/songs")
    app.run(host='0.0.0.0', port=port, debug=True)
