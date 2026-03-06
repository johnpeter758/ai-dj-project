#!/usr/bin/env python3
"""
Obsidian Semantic Search
Uses sentence-transformers for semantic search over markdown files
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

INDEX_FILE = "/Users/johnpeter/ai-dj-project/.semantic-index/index.pkl"
DEFAULT_VAULT = "/Users/johnpeter/obsidian-vault"
EXCLUDE_DIRS = {'.obsidian', '.trash', 'Templates', '.git', '__pycache__'}

def get_files(vault_path):
    """Get all .md files excluding certain directories"""
    files = []
    for root, dirs, filenames in os.walk(vault_path):
        # Filter excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in filenames:
            if f.endswith('.md'):
                files.append(os.path.join(root, f))
    return files

def load_files(vault_path):
    """Load file contents"""
    files = get_files(vault_path)
    documents = []
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Get relative path from vault
                rel_path = os.path.relpath(filepath, vault_path)
                documents.append({
                    'path': filepath,
                    'relative_path': rel_path,
                    'content': content[:10000]  # Limit content length
                })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return documents

def build_index(vault_path, model_name='all-MiniLM-L6-v2'):
    """Build semantic index"""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Loading files from: {vault_path}")
    documents = load_files(vault_path)
    print(f"Found {len(documents)} files")
    
    if not documents:
        print("No documents found!")
        return
    
    print("Generating embeddings...")
    texts = [doc['content'] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Save index
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    index_data = {
        'documents': documents,
        'embeddings': embeddings,
        'vault_path': vault_path,
        'model': model_name
    }
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"Index saved to: {INDEX_FILE}")

def search(query, top_k=5):
    """Semantic search"""
    # Load index
    with open(INDEX_FILE, 'rb') as f:
        index_data = pickle.load(f)
    
    model = SentenceTransformer(index_data['model'])
    documents = index_data['documents']
    embeddings = index_data['embeddings']
    
    # Encode query
    query_embedding = model.encode([query])[0]
    
    # Calculate similarities
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return results
    results = []
    for idx in top_indices:
        doc = documents[idx]
        results.append({
            'path': doc['path'],
            'relative_path': doc['relative_path'],
            'score': float(similarities[idx]),
            'excerpt': doc['content'][:500]
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Obsidian Semantic Search')
    parser.add_argument('command', choices=['index', 'search'], help='Command to run')
    parser.add_argument('--vault', help='Path to Obsidian vault (for index command)')
    parser.add_argument('--query', help='Search query (for search command)')
    parser.add_argument('--top', type=int, default=5, help='Number of results')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        if not args.vault:
            print("Error: --vault required for index command")
            return
        build_index(args.vault)
    
    elif args.command == 'search':
        if not args.query:
            print("Error: --query required for search command")
            return
        results = search(args.query, args.top)
        print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
