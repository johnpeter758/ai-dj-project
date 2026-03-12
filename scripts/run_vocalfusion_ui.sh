#!/usr/bin/env bash
set -euo pipefail
cd /Users/johnpeter/Code/ai-dj-project
export PORT=5055
export VF_DEBUG=0
exec /Users/johnpeter/venvs/vocalfusion-env/bin/python server.py
