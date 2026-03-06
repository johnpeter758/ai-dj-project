#!/bin/bash
# Daily Backup Script - Runs at 4:30 AM
# Backs up critical OpenClaw files to private GitHub repo

BACKUP_REPO="https://github.com/johnpeter758/ai-dj-backup.git"
BACKUP_DIR="/Users/johnpeter/ai-dj-backup"
CHANNEL_ID="1479541701923180576"
LOG_FILE="/Users/johnpeter/ai-dj-project/logs/backup-$(date +\%Y-\%m-\%d).log"
DATE=$(date +\%Y-\%m-\%d)
DISCORD_TOKEN=$(cat /Users/johnpeter/.openclaw/openclaw.json | python3 -c "import json,sys; print(json.load(sys.stdin)['channels']['discord']['token'])")

echo "=== Backup $DATE ===" | tee -a "$LOG_FILE"

# Clone or update backup repo
if [ -d "$BACKUP_DIR/.git" ]; then
    cd "$BACKUP_DIR"
    # Ensure backup remote exists
    git remote get-url backup >> "$LOG_FILE" 2>&1 || git remote add backup "$BACKUP_REPO" >> "$LOG_FILE" 2>&1
    git pull backup main >> "$LOG_FILE" 2>&1
else
    rm -rf "$BACKUP_DIR"
    git clone "$BACKUP_REPO" "$BACKUP_DIR" >> "$LOG_FILE" 2>&1
    cd "$BACKUP_DIR"
    # Rename origin to backup for consistency
    git remote rename origin backup >> "$LOG_FILE" 2>&1 || true
fi

if [ $? -ne 0 ]; then
    curl -s -X POST "https://discord.com/api/v10/channels/$CHANNEL_ID/messages" \
        -H "Authorization: Bot $DISCORD_TOKEN" \
        -H "Content-Type: application/json" \
        --data-raw '{"content": "❌ Backup failed: could not clone/pull repo"}'
    exit 1
fi

# Files to back up
BACKUP_FILES=(
    "/Users/johnpeter/.openclaw/workspace/SOUL.md"
    "/Users/johnpeter/.openclaw/workspace/MEMORY.md"
    "/Users/johnpeter/.openclaw/workspace/USER.md"
    "/Users/johnpeter/.openclaw/workspace/AGENTS.md"
    "/Users/johnpeter/.openclaw/workspace/TOOLS.md"
    "/Users/johnpeter/.openclaw/workspace/IDENTITY.md"
    "/Users/johnpeter/.openclaw/workspace/HEARTBEAT.md"
    "/Users/johnpeter/.openclaw/openclaw.json"
    "/Users/johnpeter/ai-dj-project/src/daily-maintenance.sh"
    "/Users/johnpeter/ai-dj-project/src/daily-backup.sh"
)

# Create backup directory structure
mkdir -p "$BACKUP_DIR/workspace"
mkdir -p "$BACKUP_DIR/config"
mkdir -p "$BACKUP_DIR/scripts"

MISSING_FILES=()
COPIED=0

for file in "${BACKUP_FILES[@]}"; do
    if [ -f "$file" ]; then
        dest="$BACKUP_DIR/$(echo $file | sed 's|/Users/johnpeter/||')"
        mkdir -p "$(dirname "$dest")"
        cp "$file" "$dest"
        
        # Replace secrets with placeholders
        /usr/bin/perl -i -pe 's/(token[":]=?)["\s]*[a-zA-Z0-9_\-]{20,}/$1[REDACTED]/g' "$dest" 2>/dev/null
        /usr/bin/perl -i -pe 's/(password[":]=?)["\s]*[^"\s]{4,}/$1[REDACTED]/g' "$dest" 2>/dev/null
        /usr/bin/perl -i -pe 's/(api[_-]?key[":]=?)["\s]*[a-zA-Z0-9_\-]{10,}/$1[API_KEY]/g' "$dest" 2>/dev/null
        
        COPIED=$((COPIED + 1))
    else
        MISSING_FILES+=("$file")
    fi
done

# Export crontab
crontab -l > "$BACKUP_DIR/crontab.txt" 2>/dev/null || echo "# No crontab" > "$BACKUP_DIR/crontab.txt"

# Commit and push
cd "$BACKUP_DIR"
git add -A
git diff --cached --quiet
if [ $? -eq 0 ]; then
    echo "No changes to commit" | tee -a "$LOG_FILE"
    curl -s -X POST "https://discord.com/api/v10/channels/$CHANNEL_ID/messages" \
        -H "Authorization: Bot $DISCORD_TOKEN" \
        -H "Content-Type: application/json" \
        --data-raw '{"content": "✅ Backup complete - no changes since last backup"}'
else
    git commit -m "Backup $DATE - $COPIED files" >> "$LOG_FILE" 2>&1
    git push backup main >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Backup successful" | tee -a "$LOG_FILE"
        curl -s -X POST "https://discord.com/api/v10/channels/$CHANNEL_ID/messages" \
            -H "Authorization: Bot $DISCORD_TOKEN" \
            -H "Content-Type: application/json" \
            --data-raw "{\"content\": \"✅ Backup complete - $COPIED files synced\"}"
    else
        echo "Push failed" | tee -a "$LOG_FILE"
        curl -s -X POST "https://discord.com/api/v10/channels/$CHANNEL_ID/messages" \
            -H "Authorization: Bot $DISCORD_TOKEN" \
            -H "Content-Type: application/json" \
            --data-raw '{"content": "❌ Backup failed - push error"}'
    fi
fi

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "Missing files: ${MISSING_FILES[@]}" >> "$LOG_FILE"
fi

echo "=== Done ===" | tee -a "$LOG_FILE"
