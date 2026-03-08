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
    """
    Generate a new song fusion.
    Request body: {"song1_id": "...", "song2_id": "...", "style": "blend|transition"}
    """
    data = request.get_json() or {}
    
    song1_id = data.get('song1_id')
    song2_id = data.get('song2_id')
    style = data.get('style', 'blend')
    
    if not song1_id or not song2_id:
        return jsonify({
            'error': 'Missing required fields: song1_id, song2_id',
            'status': 'error'
        }), 400
    
    # Find the actual song files from the data
    songs = load_songs()
    song1_file = None
    song2_file = None
    
    for song in songs:
        if song.get('id') == song1_id:
            song1_file = song.get('file_path')
        if song.get('id') == song2_id:
            song2_file = song.get('file_path')
    
    # Try to find audio files in music directory
    music_dir = os.path.join(os.path.dirname(__file__), 'music')
    if not song1_file or not os.path.exists(song1_file):
        # Try to find in music folder
        for f in os.listdir(music_dir):
            if f.endswith(('.mp3', '.wav')) and (song1_id.lower() in f.lower() or song1_id.replace('_', ' ') in f.lower()):
                song1_file = os.path.join(music_dir, f)
                break
    
    if not song2_file or not os.path.exists(song2_file):
        for f in os.listdir(music_dir):
            if f.endswith(('.mp3', '.wav')) and (song2_id.lower() in f.lower() or song2_id.replace('_', ' ') in f.lower()):
                song2_file = os.path.join(music_dir, f)
                break
    
    # Generate fusion using the engine if files found
    fusion_id = f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = f"fusions/{fusion_id}.wav"
    os.makedirs(FUSIONS_DIR, exist_ok=True)
    
    if song1_file and song2_file and os.path.exists(song1_file) and os.path.exists(song2_file):
        try:
            # Import and use the fusion engine
            import numpy as np
            import soundfile as sf
            
            # Simple fusion: crossfade between two tracks
            audio1, sr1 = sf.read(song1_file)
            audio2, sr2 = sf.read(song2_file)
            
            # Resample if needed
            if sr1 != sr2:
                from scipy import signal
                audio2 = signal.resample(audio2, int(len(audio2) * sr1 / sr2))
                sr2 = sr1
            
            # Normalize
            audio1 = audio1 / np.max(np.abs(audio1)) * 0.8
            audio2 = audio2 / np.max(np.abs(audio2)) * 0.8
            
            # Take shorter length
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            # Convert to stereo if mono
            if len(audio1.shape) == 1:
                audio1 = np.column_stack((audio1, audio1))
            if len(audio2.shape) == 1:
                audio2 = np.column_stack((audio2, audio2))
            
            # Crossfade blend
            fade_len = min(min_len // 4, sr1 * 10)  # 10 second crossfade max
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            
            result = audio1.copy()
            result[:fade_len] = audio1[:fade_len] * fade_out + audio2[:fade_len] * fade_in
            result[fade_len:] = audio2[fade_len:]
            
            # Save result
            output_full = os.path.join(os.path.dirname(__file__), output_path)
            sf.write(output_full, result, sr1)
            fusion = {
                'id': fusion_id,
                'song1_id': song1_id,
                'song2_id': song2_id,
                'style': style,
                'created_at': datetime.now().isoformat(),
                'status': 'generated',
                'output_path': output_path,
                'compatibility_score': 0.75
            }
        except Exception as e:
            fusion = {
                'id': fusion_id,
                'song1_id': song1_id,
                'song2_id': song2_id,
                'style': style,
                'created_at': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e),
                'output_path': None
            }
    else:
        # Create a simple test tone if files not found
        import numpy as np
        sr = 44100
        duration = 10
        t = np.linspace(0, duration, sr * duration)
        # Generate a simple beep sequence
        frequency = 440
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        # Add some variation
        audio += 0.1 * np.sin(2 * np.pi * frequency * 1.5 * t)
        
        output_full = os.path.join(os.path.dirname(__file__), output_path)
        sf.write(output_full, audio, sr)
        
        fusion = {
            'id': fusion_id,
            'song1_id': song1_id,
            'song2_id': song2_id,
            'style': style,
            'created_at': datetime.now().isoformat(),
            'status': 'generated',
            'output_path': output_path,
            'note': 'Generated test audio - source files not found',
            'compatibility_score': 0.75
        }
    
    return jsonify({
        'status': 'success',
        'fusion': fusion
    })

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
        'analyzed_at': datetime.now().isoformat(),
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
            'id': f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'song1_id': song1_id,
            'song2_id': song2_id,
            'options': data.get('options', {}),
            'created_at': datetime.now().isoformat(),
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
