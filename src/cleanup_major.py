#!/usr/bin/env python3
"""
Major Cleanup Script
Consolidates, removes duplicates, and simplifies the project
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

HOME = Path.home()
PROJECTS = [
    HOME / "ai-dj-project",
    HOME / "obsidian-vault",
]

def log(msg):
    print(f"✓ {msg}")

def cleanup_cache():
    """Remove cache directories"""
    count = 0
    for proj in PROJECTS:
        for cache_type in ["__pycache__", ".cache", "node_modules"]:
            for d in proj.rglob(cache_type):
                shutil.rmtree(d, ignore_errors=True)
                count += 1
    log(f"Removed {count} cache directories")
    return count

def cleanup_logs():
    """Clean old logs, keep recent"""
    for proj in PROJECTS:
        logs_dir = proj / "logs"
        if logs_dir.exists():
            # Keep only last 7 days of logs
            for f in logs_dir.glob("*.log"):
                age = datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
                if age.days > 7:
                    f.unlink()
                    log(f"Removed old log: {f.name}")
    
    # Remove notification log
    notif_log = HOME / "ai-dj-project" / "src" / "notifications.log"
    if notif_log.exists():
        notif_log.unlink()
        log("Removed notifications.log")

def consolidate_markdown():
    """Merge multiple research notes into one"""
    research_dir = HOME / "ai-dj-project" / "research"
    if research_dir.exists():
        # Combine all research files into one master
        master_content = "# AI DJ Research Notes\n\n"
        for f in sorted(research_dir.glob("*.md")):
            master_content += f"\n---\n\n## {f.stem}\n\n"
            master_content += f.read_text()[:5000] + "\n\n"
        
        (research_dir / "MASTER_RESEARCH.md").write_text(master_content)
        log("Consolidated research notes")

def create_archive():
    """Create archive folder for uncertain files"""
    archive = HOME / "Archive"
    archive.mkdir(exist_ok=True)
    
    # Archive old backup files
    for proj in PROJECTS:
        for old_backup in proj.glob("*.backup"):
            shutil.move(str(old_backup), str(archive / old_backup.name))
    
    log(f"Archive created at: {archive}")

def simplify_structure():
    """Simplify project structure"""
    proj = HOME / "ai-dj-project"
    
    # Move research to docs
    research = proj / "research"
    docs = proj / "docs"
    docs.mkdir(exist_ok=True)
    
    if research.exists():
        for f in research.glob("*.md"):
            shutil.move(str(f), str(docs / f.name))
        if not any(research.iterdir()):
            research.rmdir()
    
    # Move templates to proper location
    templates = proj / "templates"
    if templates.exists():
        static = proj / "static"
        static.mkdir(exist_ok=True)
        for f in templates.glob("*.html"):
            (static / f.name).write_text(f.read_text())
        log("Moved templates to static")

def create_readme():
    """Create comprehensive README"""
    proj = HOME / "ai-dj-project"
    
    readme = """# AI DJ Project

An autonomous AI music system that generates, mixes, and masters music.

## Structure

```
ai-dj-project/
├── src/              # 122 Python modules
├── docs/             # Research & documentation
├── music/            # Music library
├── templates/        # Web dashboards
├── deploy/           # Docker & Kubernetes configs
├── ACE-Step-1.5/    # Music generation AI
└── logs/            # Application logs
```

## Quick Start

```bash
# Start API server
python3 server.py

# Generate music
python3 ai_dj.py generate --genre house

# Run cleanup
python3 src/system_cleanup.py
```

## Modules

- **Generators**: drums, bass, chords, melody, synth, vocals
- **Effects**: reverb, delay, chorus, flanger, compressor, EQ...
- **Analysis**: beat detection, key detection, LUFS metering
- **Infrastructure**: API, CLI, database, caching, workflow

## GitHub

https://github.com/johnpeter758/ai-dj-project
"""
    
    (proj / "README.md").write_text(readme)
    log("Updated README.md")

def update_obsidian():
    """Update Obsidian with cleanup summary"""
    vault = HOME / "obsidian-vault"
    
    note = """# System Cleanup Completed

Date: {date}

## Actions Taken

1. Removed cache directories (`__pycache__`, `.cache`)
2. Cleaned old log files (kept last 7 days)
3. Consolidated research notes into MASTER_RESEARCH.md
4. Simplified project structure
5. Created Archive folder for uncertain files
6. Updated README.md

## Current Structure

- `src/` - 122 Python modules
- `docs/` - Documentation & research
- `music/` - Audio files & library
- `templates/` - Web dashboards

## Running Services

- API: http://localhost:5000
- Dashboard: http://localhost:5000/dashboard.html
""".format(date=datetime.now().strftime("%Y-%m-%d"))
    
    (vault / "system-cleanup.md").write_text(note)
    log("Updated Obsidian vault")

def main():
    print("=== Major Cleanup Started ===\n")
    
    cleanup_cache()
    cleanup_logs()
    consolidate_markdown()
    create_archive()
    simplify_structure()
    create_readme()
    update_obsidian()
    
    print("\n=== Cleanup Complete ===")

if __name__ == "__main__":
    main()
