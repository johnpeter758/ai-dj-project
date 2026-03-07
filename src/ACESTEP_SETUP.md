# ACE-Step 1.5 Setup

## Status: ✅ Ready

## Installation
- Cloned to: /Users/johnpeter/ACE-Step-1.5/
- Packages: 123 installed via uv sync

## Usage
```bash
cd /Users/johnpeter/ACE-Step-1.5

# Generate music
uv run python test_generate.py --prompt "A relaxing ambient piano piece" --duration 30

# Or launch Gradio UI
uv run acestep
```

## First Run
- Auto-downloads models (~1.2GB)
- Subsequent runs: ~10s on RTX 3090, ~2s on A100

## Models
- DiT only mode: 4GB VRAM
- With LM: 6-8GB VRAM
- Recommended: 12-16GB VRAM
