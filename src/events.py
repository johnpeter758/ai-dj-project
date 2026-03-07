#!/usr/bin/env python3
"""
AI DJ Event System
==================
Central event handling system for the AI DJ project.
Provides a pub/sub event bus for loose coupling between components.

Usage:
    from events import EventBus, Event, EventType
    
    # Subscribe to events
    def on_song_complete(event):
        print(f"Song completed: {event.data}")
    
    EventBus.subscribe(EventType.SONG_GENERATED, on_song_complete)
    
    # Emit events
    EventBus.emit(Event(EventType.SONG_GENERATED, {"title": "My Song"}))
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from functools import wraps
import threading
import weakref
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    """Core event types for AI DJ system"""
    # Song lifecycle
    SONG_GENERATION_START = "song_generation_start"
    SONG_GENERATED = "song_generated"
    SONG_GENERATION_FAILED = "song_generation_failed"
    SONG_EXPORTED = "song_exported"
    SONG_LOADED = "song_loaded"
    
    # Fusion events
    FUSION_START = "fusion_start"
    FUSION_CREATED = "fusion_created"
    FUSION_FAILED = "fusion_failed"
    
    # Analysis events
    ANALYSIS_START = "analysis_start"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_FAILED = "analysis_failed"
    
    # Audio processing
    STEM_PROCESSING_START = "stem_processing_start"
    STEM_PROCESSING_COMPLETE = "stem_processing_complete"
    AUDIO_EFFECTS_APPLIED = "audio_effects_applied"
    MASTERING_COMPLETE = "mastering_complete"
    
    # Playback
    PLAYBACK_START = "playback_start"
    PLAYBACK_PAUSE = "playback_pause"
    PLAYBACK_STOP = "playback_stop"
    PLAYBACK_COMPLETE = "playback_complete"
    BEAT_SYNC = "beat_sync"
    CROSSFADE_START = "crossfade_start"
    CROSSFADE_COMPLETE = "crossfade_complete"
    
    # System
    SYSTEM_READY = "system_ready"
    SYSTEM_ERROR = "system_error"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"
    
    # MIDI/Controller
    MIDI_CONTROLLER_CONNECTED = "midi_controller_connected"
    MIDI_CONTROLLER_DISCONNECTED = "midi_controller_disconnected"
    MIDI_NOTE_RECEIVED = "midi_note_received"
    MIDI_CC_RECEIVED = "midi_cc_received"
    
    # UI/Interaction
    USER_ACTION = "user_action"
    UI_UPDATE = "ui_update"
    HOTKEY_TRIGGERED = "hotkey_triggered"
    
    # Notifications (mirrors NotificationType)
    NOTIFICATION = "notification"
    TREND_ALERT = "trend_alert"
    COLLABORATION_UPDATE = "collaboration_update"
    
    # Custom events can be added dynamically
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event handler priority levels"""
    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


# =============================================================================
# EVENT DATA STRUCTURE
# =============================================================================

@dataclass
class Event:
    """Event object passed to handlers"""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: EventPriority = EventPriority.NORMAL
    bubble: bool = True  # Whether event bubbles up to parent handlers
    
    def __str__(self):
        return f"Event({self.type.value}, source={self.source})"


# =============================================================================
# EVENT HANDLER
# =============================================================================

@dataclass
class EventHandler:
    """Wrapper for event handler functions"""
    callback: Callable[[Event], Any]
    priority: EventPriority = EventPriority.NORMAL
    filter_fn: Optional[Callable[[Event], bool]] = None
    once: bool = False
    async_execute: bool = False
    
    def __post_init__(self):
        # Sort by priority (higher first)
        self.sort_key = -self.priority.value


# =============================================================================
# EVENT BUS (Singleton)
# =============================================================================

class EventBus:
    """
    Central event bus implementing observer pattern.
    Supports synchronous and asynchronous handlers, priorities, and filtering.
    """
    
    _instance: Optional['EventBus'] = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []  # Handle all events
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._subscribers: Set[weakref.ref] = set()
        
        self._initialized = True
        logger.info("EventBus initialized")
    
    # -------------------------------------------------------------------------
    # Subscription Methods
    # -------------------------------------------------------------------------
    
    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], Any],
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[Event], bool]] = None,
        once: bool = False
    ) -> Callable[[], None]:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: Type of event to listen for
            handler: Callback function to execute
            priority: Handler priority (higher = called first)
            filter_fn: Optional filter function
            once: If True, handler runs only once then auto-unsubscribes
            
        Returns:
            Unsubscribe function
        """
        event_handler = EventHandler(
            callback=handler,
            priority=priority,
            filter_fn=filter_fn,
            once=once
        )
        
        self._handlers[event_type].append(event_handler)
        # Sort by priority (descending)
        self._handlers[event_type].sort(key=lambda x: x.sort_key)
        
        logger.debug(f"Subscribed to {event_type.value}: {handler.__name__}")
        
        # Return unsubscribe function
        def unsubscribe():
            self.unsubscribe(event_type, handler)
        
        return unsubscribe
    
    def subscribe_all(
        self,
        handler: Callable[[Event], Any],
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[Event], bool]] = None
    ) -> Callable[[], None]:
        """
        Subscribe to all events (global handler).
        
        Args:
            handler: Callback function to execute for all events
            priority: Handler priority
            filter_fn: Optional filter function
            
        Returns:
            Unsubscribe function
        """
        event_handler = EventHandler(
            callback=handler,
            priority=priority,
            filter_fn=filter_fn
        )
        
        self._global_handlers.append(event_handler)
        self._global_handlers.sort(key=lambda x: x.sort_key)
        
        def unsubscribe():
            self._global_handlers.remove(event_handler)
        
        return unsubscribe
    
    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], Any]) -> bool:
        """Unsubscribe a handler from an event type"""
        handlers = self._handlers.get(event_type, [])
        for i, h in enumerate(handlers):
            if h.callback == handler:
                handlers.pop(i)
                logger.debug(f"Unsubscribed from {event_type.value}: {handler.__name__}")
                return True
        return False
    
    def clear_handlers(self, event_type: Optional[EventType] = None):
        """Clear all handlers for an event type, or all if None"""
        if event_type:
            self._handlers[event_type].clear()
        else:
            self._handlers.clear()
            self._global_handlers.clear()
    
    # -------------------------------------------------------------------------
    # Event Emission
    # -------------------------------------------------------------------------
    
    def emit(self, event: Event) -> List[Any]:
        """
        Emit an event synchronously.
        
        Args:
            event: Event object to emit
            
        Returns:
            List of handler return values
        """
        # Add to history
        self._add_to_history(event)
        
        results = []
        
        # Call type-specific handlers
        handlers = self._handlers.get(event.type, [])
        
        # Also check for CUSTOM handlers if it's a custom event
        if event.type == EventType.CUSTOM:
            handlers = handlers + self._handlers.get(EventType.CUSTOM, [])
        
        for handler in handlers[:]:  # Copy list to allow modification during iteration
            try:
                # Check filter
                if handler.filter_fn and not handler.filter_fn(event):
                    continue
                    
                # Execute handler
                result = handler.callback(event)
                results.append(result)
                
                # Handle once subscriptions
                if handler.once:
                    self.unsubscribe(event.type, handler.callback)
                    
            except Exception as e:
                logger.error(f"Error in event handler for {event.type.value}: {e}")
        
        # Call global handlers
        for handler in self._global_handlers[:]:
            try:
                if handler.filter_fn and not handler.filter_fn(event):
                    continue
                result = handler.callback(event)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in global event handler: {e}")
        
        return results
    
    async def emit_async(self, event: Event) -> List[Any]:
        """Emit an event asynchronously"""
        self._add_to_history(event)
        
        results = []
        handlers = self._handlers.get(event.type, [])
        
        async def run_handler(handler: EventHandler):
            try:
                if handler.filter_fn and not handler.filter_fn(event):
                    return
                if asyncio.iscoroutinefunction(handler.callback):
                    return await handler.callback(event)
                return handler.callback(event)
            except Exception as e:
                logger.error(f"Error in async event handler: {e}")
        
        # Run handlers concurrently
        tasks = [run_handler(h) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Global handlers
        global_tasks = [run_handler(h) for h in self._global_handlers]
        global_results = await asyncio.gather(*global_tasks, return_exceptions=True)
        results.extend(global_results)
        
        return results
    
    def emit_simple(self, event_type: EventType, data: Dict[str, Any] = None, 
                    source: str = None) -> List[Any]:
        """Convenience method to create and emit an event"""
        event = Event(
            type=event_type,
            data=data or {},
            source=source
        )
        return self.emit(event)
    
    # -------------------------------------------------------------------------
    # History & Debugging
    # -------------------------------------------------------------------------
    
    def _add_to_history(self, event: Event):
        """Add event to history buffer"""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
    
    def get_history(self, event_type: Optional[EventType] = None, 
                    limit: int = 100) -> List[Event]:
        """Get event history"""
        if event_type:
            return [e for e in self._event_history if e.type == event_type][-limit:]
        return self._event_history[-limit:]
    
    def get_handlers(self, event_type: EventType) -> List[str]:
        """Get list of handler names for an event type"""
        return [h.callback.__name__ for h in self._handlers.get(event_type, [])]
    
    # -------------------------------------------------------------------------
    # Async Processing
    # -------------------------------------------------------------------------
    
    def start_async_processing(self):
        """Start async event processing loop"""
        if self._running:
            return
            
        self._running = True
        
        def run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        logger.info("EventBus async processing started")
    
    def stop_async_processing(self):
        """Stop async event processing"""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("EventBus async processing stopped")
    
    def queue_event(self, event: Event):
        """Queue event for async processing"""
        if not self._running:
            self.start_async_processing()
        self._event_queue.put_nowait(event)


# =============================================================================
# DECORATOR SUPPORT
# =============================================================================

def on_event(event_type: EventType, priority: EventPriority = EventPriority.NORMAL):
    """Decorator for event handlers"""
    def decorator(func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        @wraps(func)
        def wrapper(event: Event):
            return func(event)
        
        # Register with EventBus
        EventBus.subscribe(event_type, wrapper, priority)
        
        return wrapper
    return decorator


def on_any_event(priority: EventPriority = EventPriority.NORMAL):
    """Decorator for handlers that receive all events"""
    def decorator(func: Callable[[Event], Any]) -> Callable[[Event], Any]:
        @wraps(func)
        def wrapper(event: Event):
            return func(event)
        
        EventBus.subscribe_all(wrapper, priority)
        
        return wrapper
    return decorator


# =============================================================================
# EVENT EMITTER MIXIN
# =============================================================================

class EventEmitter:
    """Mixin class that adds event emission capability to any class"""
    
    def __init__(self):
        self._event_bus = EventBus()
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._emitter_name = self.__class__.__name__
    
    def emit(self, event_type: EventType, data: Dict[str, Any] = None) -> List[Any]:
        """Emit an event from this emitter"""
        event = Event(
            type=event_type,
            data=data or {},
            source=self._emitter_name
        )
        return self._event_bus.emit(event)
    
    def on(self, event_type: EventType, handler: Callable) -> Callable:
        """Subscribe to an event"""
        return self._event_bus.subscribe(event_type, handler, source=self._emitter_name)
    
    def off(self, event_type: EventType, handler: Callable):
        """Unsubscribe from an event"""
        self._event_bus.unsubscribe(event_type, handler)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Default instance
_bus = None

def get_event_bus() -> EventBus:
    """Get the global EventBus instance"""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def emit(event_type: EventType, data: Dict[str, Any] = None, 
         source: str = None) -> List[Any]:
    """Emit an event using the global EventBus"""
    return get_event_bus().emit_simple(event_type, data, source)


def subscribe(event_type: EventType, handler: Callable,
              priority: EventPriority = EventPriority.NORMAL) -> Callable:
    """Subscribe to an event using the global EventBus"""
    return get_event_bus().subscribe(event_type, handler, priority)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Basic usage
    
    # Define handlers
    def on_song_generated(event: Event):
        print(f"🎵 Song generated: {event.data.get('title', 'Unknown')}")
    
    def on_system_error(event: Event):
        print(f"❌ System error: {event.data.get('message', 'Unknown error')}")
    
    # Subscribe to events
    bus = EventBus()
    bus.subscribe(EventType.SONG_GENERATED, on_song_generated)
    bus.subscribe(EventType.SYSTEM_ERROR, on_system_error, priority=EventPriority.HIGH)
    
    # Emit events
    bus.emit_simple(EventType.SONG_GENERATED, {"title": "My Awesome Track", "bpm": 128})
    bus.emit_simple(EventType.SYSTEM_ERROR, {"message": "Out of memory"})
    
    # Show history
    print("\n📜 Event History:")
    for event in bus.get_history():
        print(f"  {event.timestamp} - {event.type.value}")
