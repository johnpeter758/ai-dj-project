#!/bin/bash
# Daily OpenClaw Maintenance Script
# Runs at 4:00 AM

CHANNEL_ID="1479541701923180576"
OPENCLAW_BIN="/Users/johnpeter/.nvm/versions/node/v22.22.0/bin/openclaw"
LOG_FILE="/Users/johnpeter/ai-dj-project/logs/maintenance-$(date +\%Y-\%m-\%d).log"

echo "=== OpenClaw Daily Maintenance $(date) ===" | tee -a "$LOG_FILE"

# Get current version before update
BEFORE_VERSION=$($OPENCLAW_BIN --version 2>&1 | head -1)
echo "Version before: $BEFORE_VERSION" | tee -a "$LOG_FILE"

# Run update
echo "Running update..." | tee -a "$LOG_FILE"
UPDATE_OUTPUT=$($OPENCLAW_BIN update --yes --json 2>&1)
UPDATE_EXIT=$?

echo "Update exit code: $UPDATE_EXIT" | tee -a "$LOG_FILE"
echo "$UPDATE_OUTPUT" >> "$LOG_FILE"

# Get version after update
AFTER_VERSION=$($OPENCLAW_BIN --version 2>&1 | head -1)
echo "Version after: $AFTER_VERSION" | tee -a "$LOG_FILE"

# Build Discord message
if [ $UPDATE_EXIT -eq 0 ]; then
    MESSAGE="✅ **Daily Maintenance Complete**
- Before: $BEFORE_VERSION
- After: $AFTER_VERSION
- Updates applied successfully"
else
    MESSAGE="❌ **Maintenance Failed**
- Exit code: $UPDATE_EXIT
- Error: $(echo "$UPDATE_OUTPUT" | head -3)
- Check logs: $LOG_FILE"
fi

# Send to Discord
curl -s -X POST "https://discord.com/api/v10/channels/$CHANNEL_ID/messages" \
    -H "Authorization: Bot $(cat /Users/johnpeter/.openclaw/openclaw.json | python3 -c "import json,sys; print(json.load(sys.stdin)['channels']['discord']['token'])")" \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"$MESSAGE\"}" 2>&1

echo "=== Done ===" | tee -a "$LOG_FILE"
