#!/usr/bin/env python3
"""
AI DJ Dashboard Server
Run: python3 server.py
Then open http://localhost:5000
"""

from flask import Flask, render_template, jsonify
import os

app = Flask(__name__, template_folder='templates')

# Mock data - in production, this would come from the fusion engine
DATA = {
    'songs_analyzed': 13,
    'fusions_created': 540,
    'user_ratings': 0,
    'uptime': '2h 15m',
    'songs': [
        {'name': 'Drake - In My Feelings', 'key': '2A', 'bpm': 185, 'energy': 0.15},
        {'name': 'Travis Scott - Sicko Mode', 'key': '3A', 'bpm': 152, 'energy': 0.27},
        {'name': 'Marshmello - Happier', 'key': '6A', 'bpm': 99, 'energy': 0.12},
        {'name': 'Rick Astley - Never Gonna Give You Up', 'key': '9B', 'bpm': 112, 'energy': 0.13},
    ],
    'recent_fusions': [
        {'name': 'Fusion #1', 'songs': 'Drake x Travis Scott', 'score': 0.66},
        {'name': 'Fusion #2', 'songs': 'Marshmello x EDM 1', 'score': 0.65},
    ],
    'system_status': {
        'fusion_engine': 'Active',
        'demucs': 'Installed',
        'key_detection': 'Working',
        'bpm_detection': 'Working',
        'self_learning': 'Waiting for ratings'
    }
}

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/stats')
def stats():
    return jsonify(DATA)

@app.route('/api/songs')
def songs():
    return jsonify(DATA['songs'])

@app.route('/api/fusions')
def fusions():
    return jsonify(DATA['recent_fusions'])

@app.route('/api/status')
def status():
    return jsonify(DATA['system_status'])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🎛️ AI DJ Dashboard running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
