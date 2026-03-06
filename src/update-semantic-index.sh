#!/bin/bash
# Semantic Search Index Update - Runs at 3:00 AM

VAULT_PATH="/Users/johnpeter/obsidian-vault"
SCRIPT="/Users/johnpeter/ai-dj-project/src/semantic-search.py"

python3 "$SCRIPT" index --vault "$VAULT_PATH" >> /Users/johnpeter/ai-dj-project/logs/semantic-index.log 2>&1

echo "Semantic index updated at $(date)" >> /Users/johnpeter/ai-dj-project/logs/semantic-index.log
