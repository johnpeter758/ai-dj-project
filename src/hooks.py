#!/usr/bin/env python3
"""
AI DJ Plugin Hooks System
=========================
Event-driven hooks system for plugin extensibility.
Allows plugins to react to events, modify audio, and extend functionality.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# =============================================================================
# Hook Types and Events
# =============================================================================

class HookEvent(Enum):
    """Events that plugins can hook into."""
    # Lifecycle
    PLUGIN_LOAD = auto()
    PLUGIN_UNLOAD = auto()
    PLUGIN_ENABLE = auto()
    PLUGIN_DISABLE = auto()
    
    # Playback
    PLAYBACK_START = auto()
    PLAYBACK_STOP = auto()
    PLAYBACK_PAUSE = auto()
    PLAYBACK_RESUME = auto()
    SONG_CHANGE = auto()
    TRANSITION_START = auto()
    TRANSITION_END = auto()
    
    # Audio Pipeline
    AUDIO_INPUT = auto()
    AUDIO_OUTPUT = auto()
    AUDIO_ANALYZE = auto()
    STEM_PROCESSED = auto()
    MIX_COMPLETE = auto()
    
    # Beat/Grid
    BEAT = auto()
    BAR = auto()
    PHRASE = auto()
    DROP = auto()
    BREAK = auto()
    
    # Generation
    MELODY_GENERATE = auto()
    BASS_GENERATE = auto()
    DRUMS_GENERATE = auto()
    ARRANGEMENT_GENERATE = auto()
    FUSION_CREATE = auto()
    
    # Analysis
    BPM_DETECTED = auto()
    KEY_DETECTED = auto()
    GENRE_CLASSIFIED = auto()
    ENERGY_ANALYZED = auto()
    MOOD_ANALYZED = auto()
    
    # Effects
    EFFECT_APPLY = auto()
    EFFECT_BYPASS = auto()
    EFFECT_PARAMETER_CHANGE = auto()
    
    # MIDI/Controller
    MIDI_RECEIVED = auto()
    CONTROLLER_CONNECT = auto()
    CONTROLLER_DISCONNECT = auto()
    
    # UI
    UI_UPDATE = auto()
    DISPLAY_RENDER = auto()
    
    # Error/Status
    ERROR = auto()
    WARNING = auto()
    STATUS_CHANGE = auto()


class HookPriority(Enum):
    """Priority levels for hook execution (lower = runs first)."""
    CRITICAL = 0    # System-critical hooks
    HIGH = 25       # Important preprocessing
    NORMAL = 50     # Default priority
    LOW = 75        # Post-processing
    MONITOR = 100   # Observation only, no modification


@dataclass
class HookContext:
    """Context passed to hook handlers."""
    event: HookEvent
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Flow control
    stop_propagation: bool = False
    handled: bool = False
    
    # For audio hooks
    audio_buffer: Any = None
    sample_rate: int = 44100
    channels: int = 2
    
    # For generation hooks
    prompt: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    
    def set_data(self, key: str, value: Any):
        """Set data in the context."""
        self.data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from the context."""
        return self.data.get(key, default)
    
    def modify_audio(self, buffer: Any):
        """Replace the audio buffer."""
        self.audio_buffer = buffer
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the context."""
        self.metadata[key] = value


# =============================================================================
# Hook Handler Types
# =============================================================================

HookHandler = Callable[[HookContext], Optional[HookContext]]
FilterHandler = Callable[[HookContext], HookContext]
AsyncHookHandler = Callable[[HookContext], asyncio.Optional[HookContext]]
AsyncFilterHandler = Callable[[HookContext], asyncio.Awaitable[HookContext]]

T = TypeVar('T')


@dataclass
class HookRegistration:
    """Represents a registered hook."""
    plugin_id: str
    event: HookEvent
    handler: HookHandler
    priority: HookPriority = HookPriority.NORMAL
    is_async: bool = False
    is_filter: bool = False
    description: str = ""
    enabled: bool = True
    
    def __lt__(self, other):
        """Sort by priority (lower runs first)."""
        return self.priority.value < other.priority.value


# =============================================================================
# Hook Manager
# =============================================================================

class HookManager:
    """
    Central hook management system.
    Manages registration, execution, and flow of plugin hooks.
    """
    
    _instance: Optional['HookManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._hooks: Dict[HookEvent, List[HookRegistration]] = {}
        self._global_hooks: List[HookRegistration] = []
        self._plugin_hooks: Dict[str, List[HookRegistration]] = {}
        self._event_counts: Dict[HookEvent, int] = {}
        self._execution_log: List[Dict[str, Any]] = []
        
        self._initialized = True
        logger.info("HookManager initialized")
    
    def register(
        self,
        plugin_id: str,
        event: HookEvent,
        handler: HookHandler,
        priority: HookPriority = HookPriority.NORMAL,
        description: str = "",
        async_handler: bool = False,
        is_filter: bool = False
    ) -> HookRegistration:
        """
        Register a hook handler for an event.
        
        Args:
            plugin_id: Unique identifier for the plugin
            event: The event to hook into
            handler: Callback function to execute
            priority: Execution priority (lower = runs first)
            description: Human-readable description
            async_handler: Whether handler is async
            is_filter: Whether handler modifies context (filter) vs reacts (hook)
        
        Returns:
            HookRegistration for later management
        """
        registration = HookRegistration(
            plugin_id=plugin_id,
            event=event,
            handler=handler,
            priority=priority,
            is_async=async_handler,
            is_filter=is_filter,
            description=description
        )
        
        # Add to event-specific hooks
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(registration)
        self._hooks[event].sort()  # Sort by priority
        
        # Add to plugin's hooks
        if plugin_id not in self._plugin_hooks:
            self._plugin_hooks[plugin_id] = []
        self._plugin_hooks[plugin_id].append(registration)
        
        # Add to global hooks list
        self._global_hooks.append(registration)
        
        logger.debug(f"Registered hook: {plugin_id} -> {event.name} (priority: {priority.value})")
        return registration
    
    def unregister(
        self,
        plugin_id: str,
        event: Optional[HookEvent] = None
    ) -> int:
        """
        Unregister hooks for a plugin or specific event.
        
        Args:
            plugin_id: Plugin to unregister
            event: Optional specific event (unregisters all if None)
        
        Returns:
            Number of hooks unregistered
        """
        count = 0
        
        if event is None:
            # Unregister all hooks for plugin
            for ev, hooks in self._hooks.items():
                self._hooks[ev] = [h for h in hooks if h.plugin_id != plugin_id]
            self._plugin_hooks.pop(plugin_id, None)
            self._global_hooks = [h for h in self._global_hooks if h.plugin_id != plugin_id]
            count = len(self._global_hooks)
        else:
            # Unregister specific event
            if event in self._hooks:
                before = len(self._hooks[event])
                self._hooks[event] = [h for h in self._hooks[event] if h.plugin_id != plugin_id]
                count = before - len(self._hooks[event])
            
            if plugin_id in self._plugin_hooks:
                self._plugin_hooks[plugin_id] = [
                    h for h in self._plugin_hooks[plugin_id]
                    if h.event != event
                ]
        
        logger.debug(f"Unregistered {count} hooks for {plugin_id}")
        return count
    
    def enable(
        self,
        plugin_id: str,
        event: Optional[HookEvent] = None
    ):
        """Enable hooks for a plugin."""
        if event is None:
            for reg in self._plugin_hooks.get(plugin_id, []):
                reg.enabled = True
        else:
            for reg in self._hooks.get(event, []):
                if reg.plugin_id == plugin_id:
                    reg.enabled = True
    
    def disable(
        self,
        plugin_id: str,
        event: Optional[HookEvent] = None
    ):
        """Disable hooks for a plugin."""
        if event is None:
            for reg in self._plugin_hooks.get(plugin_id, []):
                reg.enabled = False
        else:
            for reg in self._hooks.get(event, []):
                if reg.plugin_id == plugin_id:
                    reg.enabled = False
    
    async def trigger(
        self,
        event: HookEvent,
        context: Optional[HookContext] = None,
        **kwargs
    ) -> HookContext:
        """
        Trigger all handlers for an event.
        
        Args:
            event: The event to trigger
            context: Existing context (creates new if None)
            **kwargs: Additional context data
        
        Returns:
            Modified HookContext after all handlers run
        """
        if context is None:
            context = HookContext(event=event, **kwargs)
        else:
            context.data.update(kwargs)
        
        # Track event count
        self._event_counts[event] = self._event_counts.get(event, 0) + 1
        
        hooks = self._hooks.get(event, [])
        
        for registration in hooks:
            if not registration.enabled:
                continue
            
            try:
                if registration.is_async:
                    if asyncio.iscoroutinefunction(registration.handler):
                        context = await registration.handler(context)
                    else:
                        # Run sync handler in executor
                        loop = asyncio.get_event_loop()
                        context = await loop.run_in_executor(
                            None, registration.handler, context
                        )
                else:
                    result = registration.handler(context)
                    if result is not None:
                        context = result
                
                # Log execution
                self._log_execution(registration, context)
                
                # Check for stop propagation
                if context.stop_propagation:
                    logger.debug(f"Hook propagation stopped by {registration.plugin_id}")
                    break
                    
            except Exception as e:
                logger.error(
                    f"Hook error in {registration.plugin_id} for {event.name}: {e}"
                )
                context.set_data('error', str(e))
                context.set_data('error_plugin', registration.plugin_id)
        
        context.handled = True
        return context
    
    def trigger_sync(
        self,
        event: HookEvent,
        context: Optional[HookContext] = None,
        **kwargs
    ) -> HookContext:
        """
        Synchronous trigger for events (non-async context).
        """
        if context is None:
            context = HookContext(event=event, **kwargs)
        else:
            context.data.update(kwargs)
        
        self._event_counts[event] = self._event_counts.get(event, 0) + 1
        hooks = self._hooks.get(event, [])
        
        for registration in hooks:
            if not registration.enabled:
                continue
            
            try:
                result = registration.handler(context)
                if result is not None:
                    context = result
                
                self._log_execution(registration, context)
                
                if context.stop_propagation:
                    break
                    
            except Exception as e:
                logger.error(
                    f"Hook error in {registration.plugin_id} for {event.name}: {e}"
                )
                context.set_data('error', str(e))
        
        context.handled = True
        return context
    
    def _log_execution(self, registration: HookRegistration, context: HookContext):
        """Log hook execution for debugging."""
        self._execution_log.append({
            'timestamp': context.timestamp,
            'plugin': registration.plugin_id,
            'event': registration.event.name,
            'priority': registration.priority.value,
            'data_keys': list(context.data.keys())
        })
        
        # Keep only last 1000 entries
        if len(self._execution_log) > 1000:
            self._execution_log = self._execution_log[-1000:]
    
    def get_registered_hooks(
        self,
        plugin_id: Optional[str] = None,
        event: Optional[HookEvent] = None
    ) -> List[HookRegistration]:
        """Get registered hooks, filtered by plugin and/or event."""
        if plugin_id and event:
            return [
                h for h in self._global_hooks
                if h.plugin_id == plugin_id and h.event == event
            ]
        elif plugin_id:
            return self._plugin_hooks.get(plugin_id, [])
        elif event:
            return self._hooks.get(event, [])
        return self._global_hooks.copy()
    
    def get_event_count(self, event: HookEvent) -> int:
        """Get how many times an event has been triggered."""
        return self._event_counts.get(event, 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hook system statistics."""
        return {
            'total_hooks': len(self._global_hooks),
            'total_events': len(self._hooks),
            'event_counts': {e.name: c for e, c in self._event_counts.items()},
            'plugins': list(self._plugin_hooks.keys()),
            'recent_executions': len(self._execution_log)
        }
    
    def clear(self):
        """Clear all registered hooks (for testing)."""
        self._hooks.clear()
        self._global_hooks.clear()
        self._plugin_hooks.clear()
        self._event_counts.clear()
        self._execution_log.clear()
        logger.info("HookManager cleared")


# =============================================================================
# Decorators for Easy Hook Registration
# =============================================================================

def hook(
    event: HookEvent,
    priority: HookPriority = HookPriority.NORMAL,
    description: str = "",
    is_filter: bool = False
):
    """
    Decorator to register a function as a hook handler.
    
    Usage:
        @hook(HookEvent.BEAT, priority=HookPriority.HIGH)
        def on_beat(context: HookContext):
            print("Beat hit!")
            return context
    """
    def decorator(func: HookHandler) -> HookHandler:
        # Store hook metadata on function
        func._hook_metadata = {
            'event': event,
            'priority': priority,
            'description': description,
            'is_filter': is_filter
        }
        return func
    return decorator


def filter_hook(event: HookEvent, priority: HookPriority = HookPriority.NORMAL):
    """
    Decorator for hooks that modify context (filters).
    
    Usage:
        @filter_hook(HookEvent.AUDIO_OUTPUT)
        def modify_output(context: HookContext):
            # Modify audio in context
            context.set_data('modified', True)
            return context
    """
    return hook(event, priority, is_filter=True)


# =============================================================================
# Plugin Hook Interface (for plugins to inherit)
# =============================================================================

class PluginHooks(ABC):
    """
    Abstract base class for plugins that want to use hooks.
    Provides convenient methods for hook registration.
    """
    
    def __init__(self, plugin_id: str):
        self.plugin_id = plugin_id
        self._registered_hooks: List[HookRegistration] = []
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the plugin name."""
        pass
    
    def register_hook(
        self,
        event: HookEvent,
        handler: HookHandler,
        priority: HookPriority = HookPriority.NORMAL,
        description: str = ""
    ) -> HookRegistration:
        """Register a hook for this plugin."""
        manager = HookManager()
        reg = manager.register(
            plugin_id=self.plugin_id,
            event=event,
            handler=handler,
            priority=priority,
            description=description
        )
        self._registered_hooks.append(reg)
        return reg
    
    def register_filter(
        self,
        event: HookEvent,
        handler: FilterHandler,
        priority: HookPriority = HookPriority.NORMAL,
        description: str = ""
    ) -> HookRegistration:
        """Register a filter hook for this plugin."""
        manager = HookManager()
        reg = manager.register(
            plugin_id=self.plugin_id,
            event=event,
            handler=handler,
            priority=priority,
            description=description,
            is_filter=True
        )
        self._registered_hooks.append(reg)
        return reg
    
    def unregister_all(self):
        """Unregister all hooks for this plugin."""
        manager = HookManager()
        manager.unregister(self.plugin_id)
        self._registered_hooks.clear()
    
    def enable_hooks(self):
        """Enable all hooks for this plugin."""
        manager = HookManager()
        manager.enable(self.plugin_id)
    
    def disable_hooks(self):
        """Disable all hooks for this plugin."""
        manager = HookManager()
        manager.disable(self.plugin_id)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_hook_manager() -> HookManager:
    """Get the global HookManager instance."""
    return HookManager()


def trigger_event(event: HookEvent, **kwargs) -> HookContext:
    """Trigger an event synchronously."""
    return get_hook_manager().trigger_sync(event, **kwargs)


async def trigger_event_async(event: HookEvent, **kwargs) -> HookContext:
    """Trigger an event asynchronously."""
    return await get_hook_manager().trigger(event, **kwargs)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Register hooks
    
    manager = HookManager()
    
    # Example hook handler
    def on_playback_start(context: HookContext) -> HookContext:
        print(f"Playback started: {context.get_data('song_name', 'Unknown')}")
        return context
    
    # Example filter hook
    def modify_audio_output(context: HookContext) -> HookContext:
        print("Modifying audio output...")
        context.set_data('applied_effect', 'custom_filter')
        return context
    
    # Register hooks
    manager.register(
        plugin_id="example_plugin",
        event=HookEvent.PLAYBACK_START,
        handler=on_playback_start,
        priority=HookPriority.NORMAL,
        description="Logs playback start"
    )
    
    manager.register(
        plugin_id="example_plugin",
        event=HookEvent.AUDIO_OUTPUT,
        handler=modify_audio_output,
        priority=HookPriority.LOW,
        is_filter=True,
        description="Applies custom filter"
    )
    
    # Trigger an event
    context = manager.trigger_sync(
        HookEvent.PLAYBACK_START,
        song_name="Test Song",
        bpm=128
    )
    
    print(f"\nContext data: {context.data}")
    print(f"\nStats: {manager.get_stats()}")
