#!/usr/bin/env python3
"""
AI DJ Webhooks System
=====================
External HTTP webhook system for notifying third-party services
about AI DJ events and state changes.

Features:
- Register webhook endpoints for specific events
- Support for multiple webhook providers (Discord, Slack, custom URLs)
- HMAC signature verification for security
- Retry logic with exponential backoff
- Webhook delivery history and status tracking
- Async delivery for non-blocking operation

Usage:
    from webhooks import WebhookManager, WebhookEvent, WebhookPayload
    
    # Register a webhook
    WebhookManager.register(
        url="https://example.com/webhook",
        events=[WebhookEvent.SONG_GENERATED, WebhookEvent.FUSION_CREATED],
        secret="your-webhook-secret"
    )
    
    # Trigger webhooks for an event
    await WebhookManager.trigger(
        event=WebhookEvent.SONG_GENERATED,
        data={"song_id": "123", "title": "My Song"}
    )
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

# Optional dependencies
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# WEBHOOK EVENTS
# =============================================================================

class WebhookEvent(Enum):
    """Events that can trigger webhooks."""
    # Song lifecycle
    SONG_GENERATION_START = auto()
    SONG_GENERATED = auto()
    SONG_GENERATION_FAILED = auto()
    SONG_EXPORTED = auto()
    SONG_LOADED = auto()
    
    # Fusion events
    FUSION_START = auto()
    FUSION_CREATED = auto()
    FUSION_FAILED = auto()
    
    # Analysis
    ANALYSIS_COMPLETE = auto()
    BPM_DETECTED = auto()
    KEY_DETECTED = auto()
    GENRE_CLASSIFIED = auto()
    
    # Playback
    PLAYBACK_START = auto()
    PLAYBACK_PAUSE = auto()
    PLAYBACK_STOP = auto()
    PLAYBACK_COMPLETE = auto()
    CROSSFADE_START = auto()
    CROSSFADE_COMPLETE = auto()
    
    # Effects & Processing
    STEM_PROCESSED = auto()
    MASTERING_COMPLETE = auto()
    EFFECTS_APPLIED = auto()
    
    # System
    SYSTEM_READY = auto()
    SYSTEM_ERROR = auto()
    SYSTEM_SHUTDOWN = auto()
    CONFIG_CHANGED = auto()
    
    # Playlists & Queue
    QUEUE_UPDATED = auto()
    PLAYLIST_CREATED = auto()
    PLAYLIST_UPDATED = auto()
    
    # Social/Collaboration
    COLLABORATION_INVITE = auto()
    COLLABORATION_UPDATE = auto()
    SHARE_CREATED = auto()
    
    # Custom
    CUSTOM = auto()


class WebhookStatus(Enum):
    """Delivery status for webhooks."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    TIMEOUT = "timeout"
    INVALID_PAYLOAD = "invalid_payload"


class WebhookProvider(Enum):
    """Pre-configured webhook providers."""
    CUSTOM = "custom"
    DISCORD = "discord"
    SLACK = "slack"
    TELEGRAM = "telegram"
    WEBHOOKS = "webhooks"  # Generic


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WebhookPayload:
    """Payload sent to webhook endpoints."""
    event: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary."""
        return {
            "event": self.event,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "data": self.data,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert payload to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def sign(self, secret: str) -> str:
        """Generate HMAC signature for payload."""
        message = self.to_json()
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature


@dataclass
class WebhookRegistration:
    """Registration for a webhook endpoint."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    provider: WebhookProvider = WebhookProvider.CUSTOM
    events: Set[WebhookEvent] = field(default_factory=set)
    secret: Optional[str] = None
    enabled: bool = True
    name: str = ""
    description: str = ""
    
    # Delivery settings
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0  # Base delay for exponential backoff
    
    # Headers to send with requests
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Custom transform function (stored as string reference)
    transform: Optional[str] = None
    
    # Stats
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    last_delivery_at: Optional[float] = None
    last_delivery_status: Optional[WebhookStatus] = None
    
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def get_success_rate(self) -> float:
        """Calculate delivery success rate."""
        if self.total_deliveries == 0:
            return 0.0
        return (self.successful_deliveries / self.total_deliveries) * 100
    
    def is_event_registered(self, event: WebhookEvent) -> bool:
        """Check if an event is registered for this webhook."""
        return event in self.events or WebhookEvent.CUSTOM in self.events


@dataclass
class DeliveryAttempt:
    """Record of a single delivery attempt."""
    webhook_id: str
    event_id: str
    attempt_number: int
    status: WebhookStatus
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass 
class WebhookHistory:
    """History of webhook deliveries."""
    webhook_id: str
    event: WebhookEvent
    payload: WebhookPayload
    attempts: List[DeliveryAttempt] = field(default_factory=list)
    final_status: WebhookStatus = WebhookStatus.PENDING
    completed_at: Optional[float] = None


# =============================================================================
# WEBHOOK MANAGER
# =============================================================================

class WebhookManager:
    """
    Central webhook management system.
    Handles registration, delivery, retries, and tracking of webhooks.
    """
    
    _instance: Optional['WebhookManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._webhooks: Dict[str, WebhookRegistration] = {}
        self._event_subscriptions: Dict[WebhookEvent, Set[str]] = {}
        self._delivery_history: Dict[str, List[WebhookHistory]] = {}
        self._pending_deliveries: Dict[str, asyncio.Task] = {}
        
        # Configuration
        self._default_timeout = 30.0
        self._default_max_retries = 3
        self._enable_async = True
        
        # Rate limiting
        self._rate_limit: Dict[str, float] = {}  # webhook_id -> last delivery time
        self._min_delivery_interval = 0.1  # 100ms minimum between deliveries
        
        # Custom event handlers
        self._custom_handlers: Dict[str, Callable] = {}
        
        self._initialized = True
        logger.info("WebhookManager initialized")
    
    # -------------------------------------------------------------------------
    # Registration Methods
    # -------------------------------------------------------------------------
    
    def register(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        provider: WebhookProvider = WebhookProvider.CUSTOM,
        name: str = "",
        description: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: Optional[Dict[str, str]] = None,
        enabled: bool = True
    ) -> WebhookRegistration:
        """
        Register a new webhook endpoint.
        
        Args:
            url: The webhook URL to send requests to
            events: List of events to subscribe to
            secret: Optional secret for HMAC signing
            provider: Pre-configured provider type
            name: Friendly name for the webhook
            description: Description of the webhook
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            headers: Additional headers to send
            enabled: Whether the webhook is active
        
        Returns:
            WebhookRegistration object
        """
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid webhook URL: {url}")
        
        # Create registration
        webhook = WebhookRegistration(
            url=url,
            events=set(events),
            secret=secret,
            provider=provider,
            name=name or url,
            description=description,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers or {},
            enabled=enabled
        )
        
        # Store webhook
        self._webhooks[webhook.id] = webhook
        
        # Subscribe to events
        for event in events:
            if event not in self._event_subscriptions:
                self._event_subscriptions[event] = set()
            self._event_subscriptions[event].add(webhook.id)
        
        logger.info(f"Registered webhook: {webhook.name} ({webhook.id}) for events: {[e.name for e in events]}")
        return webhook
    
    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        if webhook_id not in self._webhooks:
            return False
        
        webhook = self._webhooks[webhook_id]
        
        # Remove from event subscriptions
        for event in webhook.events:
            if event in self._event_subscriptions:
                self._event_subscriptions[event].discard(webhook_id)
        
        # Remove webhook
        del self._webhooks[webhook_id]
        logger.info(f"Unregistered webhook: {webhook_id}")
        return True
    
    def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[WebhookEvent]] = None,
        secret: Optional[str] = None,
        enabled: Optional[bool] = None,
        **kwargs
    ) -> Optional[WebhookRegistration]:
        """Update an existing webhook registration."""
        if webhook_id not in self._webhooks:
            return None
        
        webhook = self._webhooks[webhook_id]
        
        # Update fields
        if url is not None:
            webhook.url = url
        if secret is not None:
            webhook.secret = secret
        if enabled is not None:
            webhook.enabled = enabled
        
        # Update events if provided
        if events is not None:
            # Remove old subscriptions
            for event in webhook.events:
                if event in self._event_subscriptions:
                    self._event_subscriptions[event].discard(webhook_id)
            
            # Add new subscriptions
            webhook.events = set(events)
            for event in events:
                if event not in self._event_subscriptions:
                    self._event_subscriptions[event] = set()
                self._event_subscriptions[event].add(webhook_id)
        
        webhook.updated_at = time.time()
        return webhook
    
    def enable(self, webhook_id: str) -> bool:
        """Enable a webhook."""
        if webhook_id in self._webhooks:
            self._webhooks[webhook_id].enabled = True
            return True
        return False
    
    def disable(self, webhook_id: str) -> bool:
        """Disable a webhook."""
        if webhook_id in self._webhooks:
            self._webhooks[webhook_id].enabled = False
            return True
        return False
    
    # -------------------------------------------------------------------------
    # Delivery Methods
    # -------------------------------------------------------------------------
    
    async def trigger(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, WebhookHistory]:
        """
        Trigger webhooks for an event.
        
        Args:
            event: The event that occurred
            data: Event data to include in payload
            metadata: Additional metadata
        
        Returns:
            Dictionary of webhook_id -> WebhookHistory
        """
        # Get subscribed webhooks
        subscribed_ids = self._event_subscriptions.get(event, set())
        
        # Filter to enabled webhooks
        active_webhooks = [
            self._webhooks[wid] for wid in subscribed_ids
            if wid in self._webhooks and self._webhooks[wid].enabled
        ]
        
        if not active_webhooks:
            logger.debug(f"No active webhooks for event: {event.name}")
            return {}
        
        # Create payload
        payload = WebhookPayload(
            event=event.name,
            data=data,
            metadata=metadata or {}
        )
        
        # Deliver to all webhooks concurrently
        results = {}
        tasks = []
        
        for webhook in active_webhooks:
            task = asyncio.create_task(
                self._deliver_webhook(webhook, payload, event)
            )
            tasks.append((webhook.id, task))
        
        # Wait for all deliveries
        for webhook_id, task in tasks:
            try:
                history = await task
                results[webhook_id] = history
            except Exception as e:
                logger.error(f"Webhook delivery error for {webhook_id}: {e}")
        
        return results
    
    def trigger_sync(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, WebhookHistory]:
        """Synchronous trigger for non-async contexts."""
        return asyncio.run(self.trigger(event, data, metadata))
    
    async def _deliver_webhook(
        self,
        webhook: WebhookRegistration,
        payload: WebhookPayload,
        event: WebhookEvent
    ) -> WebhookHistory:
        """Deliver a webhook with retry logic."""
        history = WebhookHistory(
            webhook_id=webhook.id,
            event=event,
            payload=payload
        )
        
        # Check rate limit
        await self._check_rate_limit(webhook.id)
        
        # Attempt delivery with retries
        for attempt in range(webhook.max_retries + 1):
            attempt_record = DeliveryAttempt(
                webhook_id=webhook.id,
                event_id=payload.event_id,
                attempt_number=attempt + 1,
                status=WebhookStatus.RETRYING
            )
            
            start_time = time.time()
            
            try:
                success, status_code, response_body = await self._send_request(
                    webhook, payload
                )
                
                attempt_record.duration_ms = (time.time() - start_time) * 1000
                attempt_record.status_code = status_code
                attempt_record.response_body = response_body
                
                if success:
                    attempt_record.status = WebhookStatus.SUCCESS
                    history.final_status = WebhookStatus.SUCCESS
                    webhook.successful_deliveries += 1
                    break
                else:
                    attempt_record.status = WebhookStatus.FAILED
                    attempt_record.error_message = f"HTTP {status_code}: {response_body[:200]}"
                    
            except asyncio.TimeoutError:
                attempt_record.status = WebhookStatus.TIMEOUT
                attempt_record.error_message = "Request timed out"
                attempt_record.duration_ms = (time.time() - start_time) * 1000
                
            except Exception as e:
                attempt_record.status = WebhookStatus.FAILED
                attempt_record.error_message = str(e)
                attempt_record.duration_ms = (time.time() - start_time) * 1000
            
            history.attempts.append(attempt_record)
            
            # Retry with exponential backoff
            if attempt < webhook.max_retries:
                delay = webhook.retry_delay * (2 ** attempt)
                logger.debug(f"Retrying webhook {webhook.id} in {delay}s")
                await asyncio.sleep(delay)
        
        # Update webhook stats
        webhook.total_deliveries += 1
        webhook.last_delivery_at = time.time()
        webhook.last_delivery_status = history.final_status
        
        if history.final_status != WebhookStatus.SUCCESS:
            webhook.failed_deliveries += 1
        
        # Store in history
        if webhook.id not in self._delivery_history:
            self._delivery_history[webhook.id] = []
        self._delivery_history[webhook.id].append(history)
        
        # Keep only last 100 entries per webhook
        if len(self._delivery_history[webhook.id]) > 100:
            self._delivery_history[webhook.id] = \
                self._delivery_history[webhook.id][-100:]
        
        history.completed_at = time.time()
        
        return history
    
    async def _send_request(
        self,
        webhook: WebhookRegistration,
        payload: WebhookPayload
    ) -> tuple[bool, Optional[int], Optional[str]]:
        """Send HTTP request to webhook endpoint."""
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-DJ-Webhooks/1.0",
            "X-Webhook-Event": payload.event,
            "X-Webhook-Event-ID": payload.event_id
        }
        
        # Add custom headers
        headers.update(webhook.headers)
        
        # Add HMAC signature if secret is configured
        if webhook.secret:
            signature = payload.sign(webhook.secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        # Prepare body
        body = payload.to_json()
        
        # Send request - prefer aiohttp if available, fallback to requests
        if AIOHTTP_AVAILABLE:
            return await self._send_request_aiohttp(webhook, body, headers)
        elif REQUESTS_AVAILABLE:
            return self._send_request_requests(webhook, body, headers)
        else:
            raise RuntimeError("Neither aiohttp nor requests is available. Install one to enable webhooks.")
    
    async def _send_request_aiohttp(
        self,
        webhook: WebhookRegistration,
        body: str,
        headers: Dict[str, str]
    ) -> tuple[bool, Optional[int], Optional[str]]:
        """Send request using aiohttp."""
        timeout = aiohttp.ClientTimeout(total=webhook.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                "POST", webhook.url, data=body, headers=headers
            ) as response:
                response_body = await response.text()
                
                success = 200 <= response.status < 300
                return success, response.status, response_body
    
    def _send_request_requests(
        self,
        webhook: WebhookRegistration,
        body: str,
        headers: Dict[str, str]
    ) -> tuple[bool, Optional[int], Optional[str]]:
        """Send request using requests library (sync fallback)."""
        try:
            response = requests.post(
                webhook.url,
                data=body,
                headers=headers,
                timeout=webhook.timeout
            )
            success = 200 <= response.status_code < 300
            return success, response.status_code, response.text
        except requests.Timeout:
            return False, None, "Request timed out"
        except Exception as e:
            return False, None, str(e)
    
    async def _check_rate_limit(self, webhook_id: str):
        """Enforce minimum delivery interval."""
        last_delivery = self._rate_limit.get(webhook_id, 0)
        elapsed = time.time() - last_delivery
        
        if elapsed < self._min_delivery_interval:
            await asyncio.sleep(self._min_delivery_interval - elapsed)
        
        self._rate_limit[webhook_id] = time.time()
    
    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------
    
    def get_webhook(self, webhook_id: str) -> Optional[WebhookRegistration]:
        """Get a webhook by ID."""
        return self._webhooks.get(webhook_id)
    
    def list_webhooks(
        self,
        event: Optional[WebhookEvent] = None,
        enabled: Optional[bool] = None,
        provider: Optional[WebhookProvider] = None
    ) -> List[WebhookRegistration]:
        """List webhooks with optional filters."""
        webhooks = list(self._webhooks.values())
        
        if event is not None:
            webhooks = [w for w in webhooks if w.is_event_registered(event)]
        if enabled is not None:
            webhooks = [w for w in webhooks if w.enabled == enabled]
        if provider is not None:
            webhooks = [w for w in webhooks if w.provider == provider]
        
        return webhooks
    
    def get_history(
        self,
        webhook_id: str,
        limit: int = 10
    ) -> List[WebhookHistory]:
        """Get delivery history for a webhook."""
        history = self._delivery_history.get(webhook_id, [])
        return history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get webhook system statistics."""
        total_webhooks = len(self._webhooks)
        enabled_webhooks = sum(1 for w in self._webhooks.values() if w.enabled)
        
        total_deliveries = sum(w.total_deliveries for w in self._webhooks.values())
        successful = sum(w.successful_deliveries for w in self._webhooks.values())
        failed = sum(w.failed_deliveries for w in self._webhooks.values())
        
        return {
            "total_webhooks": total_webhooks,
            "enabled_webhooks": enabled_webhooks,
            "disabled_webhooks": total_webhooks - enabled_webhooks,
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful,
            "failed_deliveries": failed,
            "success_rate": (successful / total_deliveries * 100) if total_deliveries > 0 else 0,
            "subscribed_events": len(self._event_subscriptions)
        }
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def verify_signature(
        self,
        payload: str,
        signature: str,
        secret: str
    ) -> bool:
        """Verify webhook HMAC signature."""
        expected = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected, signature)
    
    def clear_history(self, webhook_id: Optional[str] = None):
        """Clear delivery history."""
        if webhook_id:
            self._delivery_history.pop(webhook_id, None)
        else:
            self._delivery_history.clear()
    
    def test_webhook(self, webhook_id: str) -> bool:
        """Send a test webhook to verify configuration."""
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            return False
        
        payload = WebhookPayload(
            event="test",
            data={"message": "This is a test webhook from AI DJ"},
            metadata={"test": True}
        )
        
        history = asyncio.run(self._deliver_webhook(
            webhook, payload, WebhookEvent.CUSTOM
        ))
        
        return history.final_status == WebhookStatus.SUCCESS


# =============================================================================
# PROVIDER-SPECIFIC HELPERS
# =============================================================================

def create_discord_webhook(
    url: str,
    events: List[WebhookEvent],
    secret: Optional[str] = None,
    name: str = "Discord"
) -> WebhookRegistration:
    """Create a Discord webhook."""
    manager = WebhookManager()
    
    # Discord webhooks need special embed formatting
    headers = {
        "Content-Type": "application/json"
    }
    
    return manager.register(
        url=url,
        events=events,
        secret=secret,
        provider=WebhookProvider.DISCORD,
        name=name,
        description="Discord channel webhook",
        headers=headers
    )


def create_slack_webhook(
    url: str,
    events: List[WebhookEvent],
    secret: Optional[str] = None,
    name: str = "Slack"
) -> WebhookRegistration:
    """Create a Slack webhook."""
    manager = WebhookManager()
    
    headers = {
        "Content-Type": "application/json"
    }
    
    return manager.register(
        url=url,
        events=events,
        secret=secret,
        provider=WebhookProvider.SLACK,
        name=name,
        description="Slack channel webhook",
        headers=headers
    )


def create_telegram_webhook(
    bot_token: str,
    chat_id: str,
    events: List[WebhookEvent],
    name: str = "Telegram"
) -> WebhookRegistration:
    """Create a Telegram bot webhook."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    manager = WebhookManager()
    
    # Telegram needs chat_id in the payload
    headers = {
        "Content-Type": "application/json"
    }
    
    return manager.register(
        url=url,
        events=events,
        provider=WebhookProvider.TELEGRAM,
        name=name,
        description="Telegram bot webhook",
        headers=headers
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_webhook_manager() -> WebhookManager:
    """Get the global WebhookManager instance."""
    return WebhookManager()


async def notify(
    event: WebhookEvent,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, WebhookHistory]:
    """Trigger webhooks for an event (async)."""
    return await get_webhook_manager().trigger(event, data, metadata)


def notify_sync(
    event: WebhookEvent,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, WebhookHistory]:
    """Trigger webhooks for an event (sync)."""
    return get_webhook_manager().trigger_sync(event, data, metadata)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Register webhooks
    
    manager = WebhookManager()
    
    # Register a custom webhook
    custom_webhook = manager.register(
        url="https://example.com/webhook",
        events=[WebhookEvent.SONG_GENERATED, WebhookEvent.FUSION_CREATED],
        secret="my-secret-key",
        name="My Custom Webhook",
        description="Receives notifications about generated songs"
    )
    
    # Register a Discord webhook
    discord_webhook = create_discord_webhook(
        url="https://discord.com/api/webhooks/xxx/yyy",
        events=[WebhookEvent.PLAYBACK_COMPLETE, WebhookEvent.SYSTEM_ERROR],
        name="Discord Notifications"
    )
    
    # Register a Slack webhook
    slack_webhook = create_slack_webhook(
        url="https://hooks.slack.com/services/xxx/yyy/zzz",
        events=[WebhookEvent.ANALYSIS_COMPLETE],
        name="Slack Alerts"
    )
    
    # List registered webhooks
    print("Registered webhooks:")
    for wh in manager.list_webhooks():
        print(f"  - {wh.name} ({wh.provider.value}): {wh.url}")
        print(f"    Events: {[e.name for e in wh.events]}")
        print(f"    Success rate: {wh.get_success_rate():.1f}%")
    
    # Trigger an event (sync for demo)
    results = manager.trigger_sync(
        event=WebhookEvent.SONG_GENERATED,
        data={
            "song_id": "abc123",
            "title": "My Awesome Track",
            "bpm": 128,
            "key": "Am",
            "duration": 180
        },
        metadata={"source": "test"}
    )
    
    print(f"\nDelivery results: {len(results)} webhooks notified")
    
    # Print stats
    print(f"\nStats: {manager.get_stats()}")
