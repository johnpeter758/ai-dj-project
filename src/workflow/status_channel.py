"""
Workflow Status Channel - Posts updates to Discord
"""

import time
import json
from datetime import datetime

STATUS_FILE = "/Users/johnpeter/ai-dj-project/src/workflow/current_status.json"

def update_status(task, progress=0, details=""):
    """Update current workflow status"""
    status = {
        "task": task,
        "progress": progress,
        "details": details,
        "updated": datetime.now().strftime("%H:%M:%S"),
        "timestamp": time.time()
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    return status

def get_status():
    """Get current status"""
    try:
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"task": "Idle", "progress": 0, "details": ""}

# Initial status
update_status("System ready", 100, "Waiting for tasks")
print("Status system ready!")
