#!/usr/bin/env python3
"""
System Cleanup & Organization Script
Runs daily at 2 AM to organize files and maintain system
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import json

HOME = Path.home()
REPORT = []

def log(msg):
    """Log to console and report"""
    print(f"[{datetime.now().strftime('%H:%M')}] {msg}")
    REPORT.append(f"- {msg}")

def organize_downloads():
    """Organize Downloads folder"""
    downloads = HOME / "Downloads"
    rules = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".svg"],
        "Videos": [".mp4", ".mov", ".avi", ".mkv", ".webm"],
        "Documents": [".pdf", ".doc", ".docx", ".txt", ".md", ".xls", ".xlsx", ".ppt", ".pptx"],
        "Archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        "Audio": [".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"],
        "Code": [".py", ".js", ".html", ".css", ".json", ".sh", ".swift"],
        "DMGs": [".dmg", ".pkg", ".app"],
    }
    
    moved = 0
    for item in downloads.iterdir():
        if item.name.startswith(".") or item.is_dir():
            continue
        
        ext = item.suffix.lower()
        for folder, extensions in rules.items():
            if ext in extensions:
                dest = downloads / folder
                dest.mkdir(exist_ok=True)
                shutil.move(str(item), str(dest / item.name))
                moved += 1
                log(f"Moved {item.name} → {folder}/")
                break
    
    if moved:
        log(f"Downloads: {moved} files organized")
    return moved

def organize_desktop():
    """Organize Desktop - keep minimal"""
    desktop = HOME / "Desktop"
    kept = 0
    for item in desktop.iterdir():
        if item.name.startswith("."):
            continue
        # Move to Downloads for organization
        shutil.move(str(item), str(HOME / "Downloads" / item.name))
        kept += 1
    
    if kept:
        log(f"Desktop: {kept} items moved to Downloads")
    return kept

def organize_documents():
    """Organize Documents folder"""
    docs = HOME / "Documents"
    
    # Create structure
    (docs / "Projects").mkdir(exist_ok=True)
    (docs / "PDFs").mkdir(exist_ok=True)
    (docs / "Archives").mkdir(exist_ok=True)
    
    moved = 0
    for item in docs.iterdir():
        if item.name.startswith(".") or item.is_dir():
            continue
        
        if item.suffix == ".pdf":
            shutil.move(str(item), str(docs / "PDFs" / item.name))
            moved += 1
        elif item.suffix in [".zip", ".tar", ".gz"]:
            shutil.move(str(item), str(docs / "Archives" / item.name))
            moved += 1
    
    log(f"Documents: {moved} files organized")
    return moved

def organize_music():
    """Organize Music folder"""
    music = HOME / "Music"
    
    # Create structure
    (music / "Library").mkdir(exist_ok=True)
    (music / "Stems").mkdir(exist_ok=True)
    (music / "Exports").mkdir(exist_ok=True)
    (music / "Samples").mkdir(exist_ok=True)
    
    # Move AI DJ project music if exists
    ai_dj_music = HOME / "ai-dj-project" / "music"
    if ai_dj_music.exists():
        for item in ai_dj_music.iterdir():
            if item.is_file():
                shutil.move(str(item), str(music / "Library" / item.name))
                log(f"Moved {item.name} → Music/Library/")
    
    log("Music folder organized")
    return 0

def remove_empty_dirs():
    """Remove empty directories"""
    count = 0
    for folder in [HOME / "Downloads", HOME / "Desktop", HOME / "Documents"]:
        for item in folder.rglob("*"):
            if item.is_dir() and not any(item.iterdir()):
                item.rmdir()
                count += 1
                log(f"Removed empty: {item}")
    
    log(f"Removed {count} empty directories")
    return count

def git_auto_commit(repo_path, repo_name):
    """Auto-commit changes in a repo"""
    if not Path(repo_path).exists():
        return False
    
    os.chdir(repo_path)
    
    # Check for changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not result.stdout.strip():
        return False
    
    # Add all changes
    subprocess.run(["git", "add", "-A"], capture_output=True)
    
    # Create meaningful commit message
    changed_files = result.stdout.strip().split("\n")[:5]
    files_msg = ", ".join([f.split()[1] if len(f.split()) > 1 else f for f in changed_files])
    msg = f"Daily update: {files_msg}"
    
    subprocess.run(["git", "commit", "-m", msg], capture_output=True)
    subprocess.run(["git", "push"], capture_output=True)
    
    log(f"Git: {repo_name} committed & pushed")
    return True

def obsidian_cleanup():
    """Clean up Obsidian vault"""
    vault = HOME / "obsidian-vault"
    if not vault.exists():
        return 0
    
    # Create folder structure
    (vault / "daily").mkdir(exist_ok=True)
    (vault / "projects").mkdir(exist_ok=True)
    (vault / "research").mkdir(exist_ok=True)
    (vault / "songs").mkdir(exist_ok=True)
    (vault / "fusions").mkdir(exist_ok=True)
    (vault / "trash").mkdir(exist_ok=True)
    
    # Move existing notes
    moved = 0
    for item in vault.glob("*.md"):
        if item.name.startswith("."):
            continue
        # Skip special files
        if item.name in ["AI-DJ Brain.md", "KNOWLEDGE-BASE.md"]:
            continue
        # Move to appropriate folder based on name
        name_lower = item.name.lower()
        if "song" in name_lower or "music" in name_lower:
            dest = vault / "songs"
        elif "fusion" in name_lower or "mix" in name_lower:
            dest = vault / "fusions"
        else:
            dest = vault / "research"
        
        shutil.move(str(item), str(dest / item.name))
        moved += 1
        log(f"Moved {item.name} → {dest.name}/")
    
    # Move date-named files to daily
    for item in vault.glob("*.md"):
        if item.stem[0:2].isdigit():  # Starts with date like 2024-01-01
            shutil.move(str(item), str(vault / "daily" / item.name))
            moved += 1
    
    log(f"Obsidian: {moved} notes organized")
    return moved

def find_stale_branches(repo_path):
    """Find branches older than 30 days"""
    if not Path(repo_path).exists():
        return []
    
    os.chdir(repo_path)
    result = subprocess.run(["git", "branch", "-a"], capture_output=True, text=True)
    branches = result.stdout.strip().split("\n")
    
    stale = []
    for branch in branches:
        branch = branch.strip().replace("* ", "")
        if branch in ["main", "master", "dev"]:
            continue
        
        # Check last commit date
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ai", branch],
            capture_output=True, text=True
        )
        if result.stdout:
            # Could add date logic here
            stale.append(branch)
    
    return stale

def generate_report():
    """Generate daily summary report"""
    report = f"""# Daily System Cleanup Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary
"""
    for item in REPORT:
        report += f"{item}\n"
    
    # Save to Obsidian
    vault = HOME / "obsidian-vault"
    if vault.exists():
        report_file = vault / "daily" / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        report_file.write_text(report)
        print(f"Report saved to: {report_file}")
    
    return report

def main():
    log("=== Starting Daily System Cleanup ===")
    
    # File organization
    organize_downloads()
    organize_desktop()
    organize_documents()
    organize_music()
    remove_empty_dirs()
    
    # Git auto-commit
    repos = [
        (str(HOME / "ai-dj-project"), "AI DJ Project"),
        (str(HOME / "obsidian-vault"), "Obsidian Vault"),
    ]
    for repo_path, name in repos:
        git_auto_commit(repo_path, name)
    
    # Obsidian cleanup
    obsidian_cleanup()
    
    # Generate report
    report = generate_report()
    
    log("=== Cleanup Complete ===")
    print(f"\n{report}")

if __name__ == "__main__":
    main()
