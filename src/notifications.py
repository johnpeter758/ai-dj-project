#!/usr/bin/env python3
"""
AI DJ Notification System
Provides alerts for song generation, fusions, analysis, and system events.
"""

import os
import json
from datetime import datetime
from enum import Enum
from typing import Callable, Optional
from dataclasses import dataclass, field


class NotificationLevel(Enum):
    """Notification priority levels"""
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class NotificationType(Enum):
    """Types of notifications"""
    SONG_GENERATED = "song_generated"
    FUSION_CREATED = "fusion_created"
    ANALYSIS_COMPLETE = "analysis_complete"
    QUALITY_EVAL_COMPLETE = "quality_eval_complete"
    CACHE_CLEARED = "cache_cleared"
    ERROR = "error"
    SYSTEM_STATUS = "system_status"
    COLLABORATION = "collaboration"
    TREND_ALERT = "trend_alert"


@dataclass
class Notification:
    """Notification message"""
    level: NotificationLevel
    type: NotificationType
    title: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: dict = field(default_factory=dict)
    sound: bool = True


class NotificationChannel:
    """Base class for notification channels"""
    
    def send(self, notification: Notification) -> bool:
        """Send notification - override in subclass"""
        raise NotImplementedError


class ConsoleChannel(NotificationChannel):
    """Console/terminal notifications with color"""
    
    COLORS = {
        NotificationLevel.DEBUG: "\033[36m",     # Cyan
        NotificationLevel.INFO: "\033[34m",      # Blue
        NotificationLevel.SUCCESS: "\033[32m",   # Green
        NotificationLevel.WARNING: "\033[33m",  # Yellow
        NotificationLevel.ERROR: "\033[31m",     # Red
    }
    RESET = "\033[0m"
    ICONS = {
        NotificationLevel.DEBUG: "🔍",
        NotificationLevel.INFO: "ℹ️",
        NotificationLevel.SUCCESS: "✅",
        NotificationLevel.WARNING: "⚠️",
        NotificationLevel.ERROR: "❌",
    }
    
    def send(self, notification: Notification) -> bool:
        """Print notification to console"""
        color = self.COLORS.get(notification.level, "")
        icon = self.ICONS.get(notification.level, "📢")
        
        # Format the message
        lines = [
            f"{color}{icon} [{notification.level.value.upper()}] {notification.title}{self.RESET}",
            f"   {notification.message}",
        ]
        
        # Add data if present
        if notification.data:
            for key, value in notification.data.items():
                lines.append(f"   └─ {key}: {value}")
        
        print("\n".join(lines))
        return True


class FileChannel(NotificationChannel):
    """File-based notifications (log file)"""
    
    def __init__(self, log_path: str = "notifications.log"):
        self.log_path = log_path
    
    def send(self, notification: Notification) -> bool:
        """Write notification to log file"""
        entry = json.dumps({
            "timestamp": notification.timestamp,
            "level": notification.level.value,
            "type": notification.type.value,
            "title": notification.title,
            "message": notification.message,
            "data": notification.data,
        })
        
        with open(self.log_path, "a") as f:
            f.write(entry + "\n")
        return True


class JSONChannel(NotificationChannel):
    """JSON file for programmatic consumption"""
    
    def __init__(self, json_path: str = "notifications.json"):
        self.json_path = json_path
        self.notifications = []
    
    def send(self, notification: Notification) -> bool:
        """Add to JSON file"""
        self.notifications.append({
            "timestamp": notification.timestamp,
            "level": notification.level.value,
            "type": notification.type.value,
            "title": notification.title,
            "message": notification.message,
            "data": notification.data,
        })
        
        # Keep last 100 notifications
        self.notifications = self.notifications[-100:]
        
        with open(self.json_path, "w") as f:
            json.dump(self.notifications, f, indent=2)
        return True


class WebhookChannel(NotificationChannel):
    """Webhooks for external services (Discord, Slack, etc.)"""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url
    
    def send(self, notification: Notification) -> bool:
        """Send to webhook (requires webhook_url)"""
        if not self.webhook_url:
            return False
        
        # Prepare payload
        payload = {
            "content": self._format_message(notification),
            "embeds": [{
                "title": notification.title,
                "description": notification.message,
                "color": self._get_color(notification.level),
                "fields": [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in notification.data.items()
                ] if notification.data else [],
                "timestamp": notification.timestamp,
            }]
        }
        
        # Would use requests.post here in production
        print(f"🔗 Would send webhook to {self.webhook_url}")
        return True
    
    def _format_message(self, notification: Notification) -> str:
        """Format message for Discord/Slack"""
        level_emoji = {
            NotificationLevel.SUCCESS: "✅",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.INFO: "ℹ️",
        }
        emoji = level_emoji.get(notification.level, "📢")
        return f"{emoji} **{notification.type.value}**"
    
    def _get_color(self, level: NotificationLevel) -> int:
        """Get embed color for level"""
        colors = {
            NotificationLevel.SUCCESS: 0x00FF00,
            NotificationLevel.WARNING: 0xFFFF00,
            NotificationLevel.ERROR: 0xFF0000,
            NotificationLevel.INFO: 0x0000FF,
        }
        return colors.get(level, 0x808080)


class NotificationManager:
    """Central notification manager"""
    
    def __init__(self):
        self.channels: list[NotificationChannel] = [ConsoleChannel()]
        self.handlers: list[Callable[[Notification], None]] = []
        self.stats = {
            "sent": 0,
            "by_level": {},
            "by_type": {},
        }
    
    def add_channel(self, channel: NotificationChannel) -> "NotificationManager":
        """Add a notification channel"""
        self.channels.append(channel)
        return self
    
    def add_handler(self, handler: Callable[[Notification], None]) -> "NotificationManager":
        """Add custom handler function"""
        self.handlers.append(handler)
        return self
    
    def notify(
        self,
        level: NotificationLevel,
        type: NotificationType,
        title: str,
        message: str,
        data: dict = None,
        sound: bool = True,
    ) -> Notification:
        """Send a notification"""
        notification = Notification(
            level=level,
            type=type,
            title=title,
            message=message,
            data=data or {},
            sound=sound,
        )
        
        # Send to all channels
        for channel in self.channels:
            try:
                channel.send(notification)
            except Exception as e:
                print(f"Channel error: {e}")
        
        # Call custom handlers
        for handler in self.handlers:
            try:
                handler(notification)
            except Exception as e:
                print(f"Handler error: {e}")
        
        # Update stats
        self.stats["sent"] += 1
        level_key = level.value
        self.stats["by_level"][level_key] = self.stats["by_level"].get(level_key, 0) + 1
        type_key = type.value
        self.stats["by_type"][type_key] = self.stats["by_type"].get(type_key, 0) + 1
        
        return notification
    
    # Convenience methods
    
    def info(self, type: NotificationType, title: str, message: str, **data) -> Notification:
        return self.notify(NotificationLevel.INFO, type, title, message, data)
    
    def success(self, type: NotificationType, title: str, message: str, **data) -> Notification:
        return self.notify(NotificationLevel.SUCCESS, type, title, message, data)
    
    def warning(self, type: NotificationType, title: str, message: str, **data) -> Notification:
        return self.notify(NotificationLevel.WARNING, type, title, message, data)
    
    def error(self, type: NotificationType, title: str, message: str, **data) -> Notification:
        return self.notify(NotificationLevel.ERROR, type, title, message, data)
    
    # AI DJ Specific alerts
    
    def song_generated(self, prompt: str, genre: str, output_path: str) -> Notification:
        """Notify when a song is generated"""
        return self.success(
            NotificationType.SONG_GENERATED,
            "🎵 Song Generated",
            f'Generated new {genre} track for: "{prompt}"',
            output=output_path,
            genre=genre,
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
        )
    
    def fusion_created(self, song1: str, song2: str, output_path: str) -> Notification:
        """Notify when a fusion is created"""
        return self.success(
            NotificationType.FUSION_CREATED,
            "🎛️ Fusion Created",
            f"Blended '{song1}' with '{song2}'",
            song1=song1,
            song2=song2,
            output=output_path,
        )
    
    def analysis_complete(self, file: str, bpm: int, key: str, energy: float) -> Notification:
        """Notify when analysis completes"""
        return self.info(
            NotificationType.ANALYSIS_COMPLETE,
            "🔍 Analysis Complete",
            f"Analyzed: {file}",
            bpm=bpm,
            key=key,
            energy=energy,
        )
    
    def quality_evaluation(self, score: float, passed: bool, details: dict) -> Notification:
        """Notify about quality evaluation results"""
        level = NotificationLevel.SUCCESS if passed else NotificationLevel.WARNING
        data = {"score": score, "passed": passed}
        data.update(details)
        return self.notify(
            level,
            NotificationType.QUALITY_EVAL_COMPLETE,
            "📊 Quality Check",
            f"Quality score: {score:.1f}/100 - {'PASSED' if passed else 'NEEDS REVIEW'}",
            data=data,
        )
    
    def cache_cleared(self, files_removed: int, space_freed: str) -> Notification:
        """Notify when cache is cleared"""
        return self.info(
            NotificationType.CACHE_CLEARED,
            "🗑️ Cache Cleared",
            f"Removed {files_removed} cached files ({space_freed})",
            files_removed=files_removed,
            space_freed=space_freed,
        )
    
    def error_alert(self, error_type: str, details: str, recoverable: bool = True) -> Notification:
        """Notify about an error"""
        return self.error(
            NotificationType.ERROR,
            f"⚠️ Error: {error_type}",
            details,
            recoverable=recoverable,
        )
    
    def system_status(self, component: str, status: str, details: dict = None) -> Notification:
        """System status notification"""
        return self.info(
            NotificationType.SYSTEM_STATUS,
            f"📡 {component} Status",
            status,
            **(details or {}),
        )
    
    def trend_alert(self, trend: str, change: float) -> Notification:
        """Alert about trending topic/genre"""
        emoji = "📈" if change > 0 else "📉"
        return self.info(
            NotificationType.TREND_ALERT,
            f"{emoji} Trend Alert",
            f"{trend}: {change:+.1f}% change",
            trend=trend,
            change_percent=change,
        )
    
    def get_stats(self) -> dict:
        """Get notification statistics"""
        return self.stats.copy()


# Global instance
_notifications: Optional[NotificationManager] = None

def get_notifier() -> NotificationManager:
    """Get or create global notification manager"""
    global _notifications
    if _notifications is None:
        _notifications = NotificationManager()
        # Add file channel by default
        _notifications.add_channel(FileChannel())
    return _notifications


def configure_notifications(
    log_path: str = None,
    json_path: str = None,
    webhook_url: str = None,
) -> NotificationManager:
    """Configure notification system"""
    notifier = get_notifier()
    
    if log_path:
        notifier.add_channel(FileChannel(log_path))
    
    if json_path:
        notifier.add_channel(JSONChannel(json_path))
    
    if webhook_url:
        notifier.add_channel(WebhookChannel(webhook_url))
    
    return notifier


# Example usage
if __name__ == "__main__":
    notifier = get_notifier()
    
    # Add JSON output
    notifier.add_channel(JSONChannel("notifications.json"))
    
    # Test notifications
    print("=" * 50)
    print("Testing Notification System")
    print("=" * 50)
    
    notifier.song_generated(
        prompt="Upbeat summer pop with tropical vibes",
        genre="pop",
        output_path="/output/summer_hit.wav"
    )
    
    notifier.fusion_created(
        song1="Techno Beat",
        song2="Jazz Piano",
        output_path="/output/techno_jazz_fusion.wav"
    )
    
    notifier.analysis_complete(
        file="track.wav",
        bpm=128,
        key="8A",
        energy=0.85
    )
    
    notifier.quality_evaluation(
        score=87.5,
        passed=True,
        details={"clarity": 90, "bass": 85, "vocals": 88}
    )
    
    notifier.cache_cleared(
        files_removed=42,
        space_freed="1.2GB"
    )
    
    notifier.error_alert(
        error_type="Model Loading",
        details="Failed to load voice model, using fallback"
    )
    
    notifier.trend_alert(
        trend="Hyperpop",
        change=15.3
    )
    
    print("\n" + "=" * 50)
    print(f"Stats: {notifier.get_stats()}")
    print("=" * 50)
