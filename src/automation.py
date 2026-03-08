#!/usr/bin/env python3
"""
Track Automation System for AI DJ Project

Provides automated track control including:
- Scheduled track playback automation
- Automatic transition management
- Effect automation (volume, effects, EQ)
- Event-based triggers and actions
- Automation presets and timelines
- Integration with scheduler and queue systems
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from copy import deepcopy

from queue import TrackQueue, QueuedTrack, RepeatMode, QueueEvent
from scheduler import TaskScheduler, TaskPriority, ScheduledTask

logger = logging.getLogger(__name__)


class AutomationEvent(Enum):
    """Events that can trigger automation"""
    TRACK_START = auto()
    TRACK_END = auto()
    TRANSITION_START = auto()
    TRANSITION_END = auto()
    BPM_MATCH = auto()
    ENERGY_CHANGE = auto()
    MANUAL_TRIGGER = auto()
    SCHEDULED_TIME = auto()
    QUEUE_EMPTY = auto()


class TransitionType(Enum):
    """Types of track transitions"""
    CUT = auto()
    CROSSFADE = auto()
    FADE_IN = auto()
    FADE_OUT = auto()
    BEAT_MATCH = auto()
    KEY_MIX = auto()
    TEMPO_GLIDE = auto()


@dataclass
class AutomationAction:
    """An action to perform in automation"""
    action_type: str  # e.g., "set_volume", "set_effect", "play", "pause", "skip"
    params: Dict[str, Any] = field(default_factory=dict)
    delay_seconds: float = 0.0


@dataclass
class AutomationTrigger:
    """Condition that triggers automation"""
    event: AutomationEvent
    condition: Optional[Callable[[Any], bool]] = None
    track_filter: Optional[Callable[[QueuedTrack], bool]] = None  # Filter which tracks trigger
    
    def __hash__(self):
        # Only hash the event (callable fields can't be hashed)
        return hash(self.event)


@dataclass
class AutomationStep:
    """A single step in an automation timeline"""
    time_offset: float  # Seconds from trigger
    actions: List[AutomationAction]
    description: str = ""


@dataclass
class AutomationPreset:
    """Saved automation preset"""
    name: str
    description: str = ""
    transition_type: TransitionType = TransitionType.CROSSFADE
    crossfade_duration: float = 5.0
    volume_curve: List[float] = field(default_factory=lambda: [0.8, 0.8])  # Start, end
    effects: Dict[str, float] = field(default_factory=dict)
    energy_flow: List[float] = field(default_factory=list)  # Energy levels over time
    timeline: List[AutomationStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class TrackAutomation:
    """Main track automation controller"""
    
    def __init__(
        self,
        track_queue: Optional[TrackQueue] = None,
        scheduler: Optional[TaskScheduler] = None,
        persist_path: Optional[str] = None,
    ):
        self._queue = track_queue
        self._scheduler = scheduler or TaskScheduler()
        self._persist_path = persist_path or "/Users/johnpeter/ai-dj-project/src/automation.json"
        
        # Automation state
        self._current_track: Optional[QueuedTrack] = None
        self._is_playing = False
        self._is_paused = False
        self._current_volume = 0.8
        self._target_volume = 0.8
        self._crossfade_duration = 5.0
        self._transition_type = TransitionType.CROSSFADE
        
        # Active automations
        self._triggers: Dict[AutomationTrigger, List[AutomationAction]] = {}
        self._scheduled_automations: List[tuple[datetime, AutomationAction]] = []
        self._active_timeline: List[AutomationStep] = []
        self._timeline_start_time: Optional[datetime] = None
        
        # Presets
        self._presets: Dict[str, AutomationPreset] = {}
        self._active_preset: Optional[AutomationPreset] = None
        
        # Callbacks
        self._callbacks: Dict[AutomationEvent, List[Callable]] = {
            event: [] for event in AutomationEvent
        }
        
        # Effects state
        self._effects_state: Dict[str, float] = {
            "reverb_mix": 0.0,
            "delay_mix": 0.0,
            "filter_cutoff": 1.0,
            "low_gain": 1.0,
            "mid_gain": 1.0,
            "high_gain": 1.0,
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load persisted state
        self._load_state()
        
        # Register default triggers
        self._register_default_triggers()
    
    def _register_default_triggers(self):
        """Register default automation triggers"""
        # Auto-crossfade at track end
        self.add_trigger(
            AutomationTrigger(
                event=AutomationEvent.TRACK_END,
                condition=lambda _: self._is_playing and len(self._queue.queue) > 0
            ),
            [AutomationAction(action_type="transition_next", params={})]
        )
        
        # Auto-continue when queue empty but we have history
        self.add_trigger(
            AutomationTrigger(event=AutomationEvent.QUEUE_EMPTY),
            [AutomationAction(action_type="repeat_or_shuffle", params={})]
        )
    
    def _load_state(self):
        """Load persisted automation state"""
        try:
            if Path(self._persist_path).exists():
                with open(self._persist_path, 'r') as f:
                    data = json.load(f)
                    
                # Load effects state
                if "effects" in data:
                    self._effects_state.update(data["effects"])
                    
                # Load active preset
                if "active_preset" in data:
                    preset_data = data["active_preset"]
                    if preset_data and preset_data in self._presets:
                        self._active_preset = self._presets[preset_data]
                
                logger.info(f"Loaded automation state from {self._persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load automation state: {e}")
    
    def _save_state(self):
        """Persist automation state"""
        try:
            data = {
                "effects": self._effects_state,
                "active_preset": self._active_preset.name if self._active_preset else None,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self._persist_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save automation state: {e}")
    
    # ============ Queue Integration ============
    
    def set_queue(self, queue: TrackQueue):
        """Set the track queue to control"""
        self._queue = queue
        
        # Register queue callbacks
        if queue:
            queue.on(QueueEvent.TRACK_PLAYING, self._on_track_playing)
            queue.on(QueueEvent.QUEUE_CLEARED, self._on_queue_cleared)
    
    def _on_track_playing(self, track: Optional[QueuedTrack], extra: Any):
        """Handle track playing event"""
        if track:
            with self._lock:
                self._current_track = track
                self._timeline_start_time = datetime.now()
                self._start_timeline()
            self._fire_callback(AutomationEvent.TRACK_START, track)
    
    def _on_queue_cleared(self, track: Optional[QueuedTrack], extra: Any):
        """Handle queue cleared"""
        self._fire_callback(AutomationEvent.QUEUE_EMPTY)
    
    # ============ Playback Control ============
    
    def play(self) -> bool:
        """Start playback"""
        with self._lock:
            if self._is_paused:
                self._is_paused = False
            self._is_playing = True
            logger.info("Automation: Play")
            return True
        return False
    
    def pause(self) -> bool:
        """Pause playback"""
        with self._lock:
            self._is_paused = True
            logger.info("Automation: Pause")
            return True
    
    def stop(self) -> bool:
        """Stop playback"""
        with self._lock:
            self._is_playing = False
            self._is_paused = False
            self._active_timeline = []
            logger.info("Automation: Stop")
            return True
    
    def skip(self) -> bool:
        """Skip to next track"""
        with self._lock:
            if self._queue and len(self._queue.queue) > 0:
                self._queue.pop()
                self._fire_callback(AutomationEvent.TRACK_END)
                return True
        return False
    
    def previous(self) -> bool:
        """Go to previous track (from history)"""
        with self._lock:
            if self._queue and len(self._queue.history) > 0:
                last = self._queue.history[-1]
                self._queue.add(last)
                return True
        return False
    
    # ============ Volume Control ============
    
    def set_volume(self, volume: float) -> bool:
        """Set volume (0.0 to 1.0)"""
        with self._lock:
            self._target_volume = max(0.0, min(1.0, volume))
            logger.info(f"Automation: Volume set to {self._target_volume}")
            self._save_state()
            return True
    
    def fade_volume(self, target: float, duration: float = 2.0) -> threading.Thread:
        """Fade volume over time"""
        def _fade():
            start_vol = self._current_volume
            steps = int(duration * 10)  # 10 steps per second
            for i in range(steps + 1):
                if not self._is_playing:
                    break
                t = i / steps
                vol = start_vol + (target - start_vol) * t
                with self._lock:
                    self._current_volume = vol
                time.sleep(duration / steps)
        
        thread = threading.Thread(target=_fade, daemon=True)
        thread.start()
        return thread
    
    def get_volume(self) -> float:
        """Get current volume"""
        return self._current_volume
    
    # ============ Effects Control ============
    
    def set_effect(self, effect_name: str, value: float) -> bool:
        """Set an effect parameter"""
        with self._lock:
            self._effects_state[effect_name] = value
            logger.info(f"Automation: {effect_name} = {value}")
            self._save_state()
            return True
    
    def set_effects(self, effects: Dict[str, float]) -> bool:
        """Set multiple effect parameters"""
        with self._lock:
            self._effects_state.update(effects)
            self._save_state()
            return True
    
    def get_effects(self) -> Dict[str, float]:
        """Get all effect values"""
        return deepcopy(self._effects_state)
    
    # ============ Transition Control ============
    
    def set_transition_type(self, transition_type: Union[str, TransitionType]) -> bool:
        """Set transition type"""
        if isinstance(transition_type, str):
            try:
                transition_type = TransitionType[transition_type.upper()]
            except KeyError:
                logger.error(f"Unknown transition type: {transition_type}")
                return False
        
        with self._lock:
            self._transition_type = transition_type
            logger.info(f"Automation: Transition type set to {transition_type.name}")
            return True
    
    def set_crossfade_duration(self, seconds: float) -> bool:
        """Set crossfade duration"""
        with self._lock:
            self._crossfade_duration = max(0.0, seconds)
            return True
    
    def trigger_transition(self, transition_type: Optional[TransitionType] = None) -> bool:
        """Manually trigger a transition"""
        with self._lock:
            tt = transition_type or self._transition_type
            logger.info(f"Automation: Triggering {tt.name} transition")
            self._fire_callback(AutomationEvent.TRANSITION_START)
            
            # Execute transition based on type
            if tt == TransitionType.CROSSFADE:
                self._execute_crossfade()
            elif tt == TransitionType.FADE_OUT:
                self._execute_fade_out()
            elif tt == TransitionType.FADE_IN:
                self._execute_fade_in()
            elif tt == TransitionType.CUT:
                self._execute_cut()
            else:
                # Default to crossfade
                self._execute_crossfade()
            
            self._fire_callback(AutomationEvent.TRANSITION_END)
            return True
    
    def _execute_crossfade(self):
        """Execute crossfade transition"""
        if not self._queue or len(self._queue.queue) < 1:
            return
        
        # Fade out current
        self.fade_volume(0.0, self._crossfade_duration / 2)
        time.sleep(self._crossfade_duration / 2)
        
        # Get next track
        next_track = self._queue.peek()
        
        # Skip current
        self._queue.pop()
        
        # Fade in next
        self._current_volume = 0.0
        self.fade_volume(self._target_volume, self._crossfade_duration / 2)
    
    def _execute_fade_out(self):
        """Execute fade out transition"""
        self.fade_volume(0.0, self._crossfade_duration)
    
    def _execute_fade_in(self):
        """Execute fade in transition"""
        self._current_volume = 0.0
        self.fade_volume(self._target_volume, self._crossfade_duration)
    
    def _execute_cut(self):
        """Execute instant cut transition"""
        self.set_volume(0.0)
        if self._queue and len(self._queue.queue) > 0:
            self._queue.pop()
        self.set_volume(self._target_volume)
    
    # ============ Automation Triggers ============
    
    def add_trigger(
        self,
        trigger: AutomationTrigger,
        actions: List[AutomationAction],
    ) -> None:
        """Add an automation trigger"""
        with self._lock:
            self._triggers[trigger] = actions
    
    def remove_trigger(self, trigger: AutomationTrigger) -> bool:
        """Remove an automation trigger"""
        with self._lock:
            if trigger in self._triggers:
                del self._triggers[trigger]
                return True
        return False
    
    def _check_triggers(self, event: AutomationEvent, context: Any = None) -> None:
        """Check and execute matching triggers"""
        with self._lock:
            for trigger, actions in self._triggers.items():
                if trigger.event != event:
                    continue
                
                # Check condition
                if trigger.condition and not trigger.condition(context):
                    continue
                
                # Check track filter
                if trigger.track_filter and self._current_track:
                    if not trigger.track_filter(self._current_track):
                        continue
                
                # Execute actions
                for action in actions:
                    self._execute_action(action)
    
    def _execute_action(self, action: AutomationAction) -> None:
        """Execute a single automation action"""
        action_type = action.action_type
        
        if action.delay_seconds > 0:
            time.sleep(action.delay_seconds)
        
        if action_type == "play":
            self.play()
        elif action_type == "pause":
            self.pause()
        elif action_type == "stop":
            self.stop()
        elif action_type == "skip":
            self.skip()
        elif action_type == "previous":
            self.previous()
        elif action_type == "set_volume":
            self.set_volume(action.params.get("volume", 0.8))
        elif action_type == "fade_volume":
            self.fade_volume(
                action.params.get("target", 0.8),
                action.params.get("duration", 2.0)
            )
        elif action_type == "set_effect":
            self.set_effect(
                action.params.get("name", ""),
                action.params.get("value", 0.0)
            )
        elif action_type == "set_transition":
            self.set_transition_type(action.params.get("type", "CROSSFADE"))
        elif action_type == "transition_next":
            self.trigger_transition()
        elif action_type == "repeat_or_shuffle":
            self._handle_repeat_or_shuffle()
        elif action_type == "load_preset":
            self.load_preset(action.params.get("preset", ""))
        elif action_type == "apply_preset":
            self.apply_preset(action.params.get("preset", ""))
        else:
            logger.warning(f"Unknown automation action: {action_type}")
    
    def _handle_repeat_or_shuffle(self):
        """Handle repeat or shuffle when queue is empty"""
        if not self._queue:
            return
        
        repeat_mode = self._queue.get_repeat_mode()
        
        if repeat_mode == RepeatMode.ALL and self._queue.history:
            # Re-add history to queue
            for track in reversed(self._queue.history):
                self._queue.add(deepcopy(track))
        elif repeat_mode == RepeatMode.OFF:
            # Try to shuffle library or notify
            logger.info("Queue empty, repeat off")
    
    # ============ Timeline Automation ============
    
    def load_timeline(self, steps: List[AutomationStep]) -> None:
        """Load an automation timeline"""
        with self._lock:
            self._active_timeline = sorted(steps, key=lambda s: s.time_offset)
            logger.info(f"Loaded timeline with {len(steps)} steps")
    
    def _start_timeline(self) -> None:
        """Start executing loaded timeline"""
        if not self._active_timeline:
            return
        
        def _run_timeline():
            with self._lock:
                timeline = deepcopy(self._active_timeline)
                start_time = self._timeline_start_time
            
            for step in timeline:
                # Wait until step time
                elapsed = (datetime.now() - start_time).total_seconds()
                wait_time = step.time_offset - elapsed
                
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # Execute actions
                for action in step.actions:
                    self._execute_action(action)
        
        thread = threading.Thread(target=_run_timeline, daemon=True)
        thread.start()
    
    # ============ Presets ============
    
    def create_preset(
        self,
        name: str,
        description: str = "",
        transition_type: TransitionType = TransitionType.CROSSFADE,
        crossfade_duration: float = 5.0,
        effects: Optional[Dict[str, float]] = None,
    ) -> AutomationPreset:
        """Create a new automation preset"""
        preset = AutomationPreset(
            name=name,
            description=description,
            transition_type=transition_type,
            crossfade_duration=crossfade_duration,
            effects=effects or deepcopy(self._effects_state),
        )
        
        with self._lock:
            self._presets[name] = preset
            logger.info(f"Created preset: {name}")
        
        return preset
    
    def load_preset(self, name: str) -> bool:
        """Load a preset (applies settings but doesn't activate)"""
        with self._lock:
            if name not in self._presets:
                logger.warning(f"Preset not found: {name}")
                return False
            
            preset = self._presets[name]
            self._transition_type = preset.transition_type
            self._crossfade_duration = preset.crossfade_duration
            
            if preset.effects:
                self._effects_state.update(preset.effects)
            
            logger.info(f"Loaded preset: {name}")
            return True
    
    def apply_preset(self, name: str) -> bool:
        """Apply and activate a preset"""
        if not self.load_preset(name):
            return False
        
        with self._lock:
            self._active_preset = self._presets[name]
        
        self._save_state()
        return True
    
    def delete_preset(self, name: str) -> bool:
        """Delete a preset"""
        with self._lock:
            if name in self._presets:
                del self._presets[name]
                if self._active_preset and self._active_preset.name == name:
                    self._active_preset = None
                return True
        return False
    
    def list_presets(self) -> List[str]:
        """List all preset names"""
        return list(self._presets.keys())
    
    def get_current_preset(self) -> Optional[str]:
        """Get name of active preset"""
        return self._active_preset.name if self._active_preset else None
    
    # ============ Scheduled Automation ============
    
    def schedule_action(
        self,
        action: AutomationAction,
        at: datetime,
    ) -> str:
        """Schedule an action to run at a specific time"""
        schedule_id = f"sched_{int(at.timestamp())}_{action.action_type}"
        
        with self._lock:
            self._scheduled_automations.append((at, action))
        
        # Schedule with task scheduler
        if self._scheduler:
            self._scheduler.schedule_once(
                name=schedule_id,
                func=self._execute_action,
                args=(action,),
                scheduled_time=at,
            )
        
        logger.info(f"Scheduled action: {schedule_id} at {at}")
        return schedule_id
    
    def schedule_recurring(
        self,
        action: AutomationAction,
        interval: timedelta,
        start: Optional[datetime] = None,
    ) -> str:
        """Schedule an action to run repeatedly"""
        start = start or datetime.now()
        schedule_id = f"recurring_{int(start.timestamp())}_{action.action_type}"
        
        if self._scheduler:
            self._scheduler.schedule_interval(
                name=schedule_id,
                func=self._execute_action,
                interval=interval,
                args=(action,),
            )
        
        logger.info(f"Scheduled recurring action: {schedule_id} every {interval}")
        return schedule_id
    
    def cancel_scheduled(self, schedule_id: str) -> bool:
        """Cancel a scheduled action"""
        if self._scheduler:
            return self._scheduler.cancel(schedule_id)
        return False
    
    # ============ Callbacks ============
    
    def on(self, event: AutomationEvent, callback: Callable) -> None:
        """Register event callback"""
        self._callbacks[event].append(callback)
    
    def off(self, event: AutomationEvent, callback: Callable) -> None:
        """Unregister event callback"""
        if callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    def _fire_callback(self, event: AutomationEvent, context: Any = None) -> None:
        """Fire callbacks for an event"""
        for callback in self._callbacks[event]:
            try:
                callback(event, context)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
        
        # Also check triggers
        self._check_triggers(event, context)
    
    # ============ Status ============
    
    def get_status(self) -> Dict[str, Any]:
        """Get current automation status"""
        return {
            "is_playing": self._is_playing,
            "is_paused": self._is_paused,
            "volume": self._current_volume,
            "target_volume": self._target_volume,
            "transition_type": self._transition_type.name,
            "crossfade_duration": self._crossfade_duration,
            "current_track": self._current_track.name if self._current_track else None,
            "queue_length": len(self._queue.queue) if self._queue else 0,
            "active_preset": self._active_preset.name if self._active_preset else None,
            "effects": deepcopy(self._effects_state),
        }
    
    def __repr__(self) -> str:
        status = self.get_status()
        return f"TrackAutomation(playing={status['is_playing']}, volume={status['volume']:.2f}, track={status['current_track']})"


# ============ Factory Functions ============

def create_automation(
    queue: Optional[TrackQueue] = None,
    persist_path: Optional[str] = None,
) -> TrackAutomation:
    """Create and initialize track automation"""
    scheduler = TaskScheduler(persist_path="/Users/johnpeter/ai-dj-project/src/scheduler_state.json")
    automation = TrackAutomation(
        track_queue=queue,
        scheduler=scheduler,
        persist_path=persist_path,
    )
    
    # Create default presets
    automation.create_preset(
        name="smooth_jazz",
        description="Gentle transitions for jazz",
        transition_type=TransitionType.CROSSFADE,
        crossfade_duration=8.0,
        effects={"reverb_mix": 0.3, "delay_mix": 0.1},
    )
    
    automation.create_preset(
        name="edm_drops",
        description="Hard cuts and quick fades for EDM",
        transition_type=TransitionType.CUT,
        crossfade_duration=2.0,
        effects={"reverb_mix": 0.1, "delay_mix": 0.2},
    )
    
    automation.create_preset(
        name="chill_vibes",
        description="Long fades for ambient music",
        transition_type=TransitionType.FADE_IN,
        crossfade_duration=10.0,
        effects={"reverb_mix": 0.5, "delay_mix": 0.2, "filter_cutoff": 0.8},
    )
    
    return automation


# ============ Example Usage ============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create automation with a test queue
    queue = TrackQueue(persist_path=Path("/Users/johnpeter/ai-dj-project/src/queue_state.json"))
    automation = create_automation(queue)
    
    # Add some test tracks
    test_tracks = [
        QueuedTrack(filename="test1.wav", artist="Artist 1", genre="Pop", bpm=120, key="4A", energy=0.7),
        QueuedTrack(filename="test2.wav", artist="Artist 2", genre="Rock", bpm=130, key="5A", energy=0.8),
        QueuedTrack(filename="test3.wav", artist="Artist 3", genre="EDM", bpm=128, key="6A", energy=0.9),
    ]
    
    for track in test_tracks:
        queue.add(track)
    
    # Start scheduler
    automation._scheduler.start()
    
    # Example: Schedule volume fade in 5 seconds
    automation.schedule_action(
        action=AutomationAction(
            action_type="fade_volume",
            params={"target": 0.5, "duration": 3.0}
        ),
        at=datetime.now() + timedelta(seconds=5)
    )
    
    # Example: Schedule skip in 10 seconds
    automation.schedule_action(
        action=AutomationAction(action_type="skip", params={}),
        at=datetime.now() + timedelta(seconds=10)
    )
    
    print(f"Automation initialized: {automation}")
    print(f"Available presets: {automation.list_presets()}")
    
    # Apply a preset
    automation.apply_preset("smooth_jazz")
    print(f"Active preset: {automation.get_current_preset()}")
    
    # Get status
    print(f"Status: {automation.get_status()}")
    
    try:
        # Let it run for a bit
        time.sleep(15)
    except KeyboardInterrupt:
        pass
    finally:
        automation._scheduler.shutdown()
        print("Automation shutdown complete")
