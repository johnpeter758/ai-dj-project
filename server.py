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
    
    # Mock generation - in production this would call the fusion engine
    fusion = {
        'id': f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'song1_id': song1_id,
        'song2_id': song2_id,
        'style': style,
        'created_at': datetime.now().isoformat(),
        'status': 'generated',
        'output_path': f"fusions/{song1_id}_x_{song2_id}.wav",
        'compatibility_score': 0.75  # Would be calculated by engine
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
