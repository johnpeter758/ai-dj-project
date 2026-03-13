#!/usr/bin/env python3
"""
Track Queue System for AI DJ Project

A robust, thread-safe queue management system featuring:
- Thread-safe operations for concurrent access
- Priority queue support (bump tracks to front)
- Play history with undo functionality
- Shuffle and repeat modes
- Queue persistence to JSON
- Event callbacks for UI integration
"""

import json
import threading
import random
import importlib.util
import sysconfig
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any
from datetime import datetime
from enum import Enum
from collections import deque
from copy import deepcopy


def _load_stdlib_queue_module():
    """Load the stdlib queue module without recursing into this local file."""
    stdlib_path = Path(sysconfig.get_path("stdlib")) / "queue.py"
    spec = importlib.util.spec_from_file_location("_vocalfusion_stdlib_queue", stdlib_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load stdlib queue module from {stdlib_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_queue = _load_stdlib_queue_module()

# Re-export stdlib queue symbols so internal Python modules (for example
# concurrent.futures / multiprocessing) still work even if this file is
# imported as top-level `queue` because the repo's src directory is on sys.path.
Empty = _stdlib_queue.Empty
Full = _stdlib_queue.Full
Queue = _stdlib_queue.Queue
PriorityQueue = _stdlib_queue.PriorityQueue
LifoQueue = _stdlib_queue.LifoQueue
SimpleQueue = getattr(_stdlib_queue, "SimpleQueue", None)


class RepeatMode(Enum):
    """Queue repeat modes"""
    OFF = "off"
    ALL = "all"
    ONE = "one"


class QueueEvent(Enum):
    """Events emitted by the queue"""
    TRACK_ADDED = "track_added"
    TRACK_REMOVED = "track_removed"
    TRACK_PLAYING = "track_playing"
    QUEUE_CLEARED = "queue_cleared"
    QUEUE_SHUFFLED = "queue_shuffled"
    HISTORY_UPDATED = "history_updated"


@dataclass
class QueuedTrack:
    """A track in the queue with metadata"""
    filename: str
    artist: str
    genre: str
    bpm: float
    key: str
    energy: float = 0.5
    duration: float = 180.0
    added_at: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Higher = plays sooner
    requested_by: Optional[str] = None
    
    @property
    def name(self) -> str:
        return self.filename.replace(".wav", "").replace("_", " ").title()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['added_at'] = self.added_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'QueuedTrack':
        """Create from dictionary"""
        if 'added_at' in data and isinstance(data['added_at'], str):
            data['added_at'] = datetime.fromisoformat(data['added_at'])
        return cls(**data)


class TrackQueue:
    """Thread-safe track queue with advanced features"""
    
    def __init__(self, max_history: int = 50, persist_path: Optional[Path] = None):
        self._queue: deque[QueuedTrack] = deque()
        self._history: deque[QueuedTrack] = deque()
        self._max_history = max_history
        self._persist_path = persist_path
        self._lock = threading.RLock()
        self._repeat_mode = RepeatMode.OFF
        self._shuffle_enabled = False
        self._current_index: Optional[int] = None
        self._callbacks: dict[QueueEvent, list[Callable]] = {
            event: [] for event in QueueEvent
        }
        
        # Load persisted queue if available
        if persist_path and persist_path.exists():
            self._load()
    
    # ============ Event System ============
    
    def on(self, event: QueueEvent, callback: Callable[[QueuedTrack, Any], None]) -> None:
        """Register event callback"""
        self._callbacks[event].append(callback)
    
    def _emit(self, event: QueueEvent, track: Optional[QueuedTrack] = None, extra: Any = None) -> None:
        """Emit event to all registered callbacks"""
        for callback in self._callbacks[event]:
            try:
                callback(track, extra)
            except Exception as e:
                print(f"Queue callback error: {e}")
    
    # ============ Core Queue Operations ============
    
    def add(self, track: QueuedTrack) -> int:
        """Add track to queue, returns position"""
        with self._lock:
            if track.priority > 0:
                # Insert at priority position (higher priority = earlier)
                inserted = False
                for i, t in enumerate(self._queue):
                    if track.priority > t.priority:
                        self._queue.insert(i, track)
                        inserted = True
                        break
                if not inserted:
                    self._queue.append(track)
            else:
                self._queue.append(track)
            
            position = list(self._queue).index(track)
            self._emit(QueueEvent.TRACK_ADDED, track, position)
            self._persist_async()
            return position
    
    def add_many(self, tracks: list[QueuedTrack]) -> list[int]:
        """Add multiple tracks to queue"""
        positions = []
        for track in tracks:
            positions.append(self.add(track))
        return positions
    
    def remove(self, index: int) -> Optional[QueuedTrack]:
        """Remove track at index"""
        with self._lock:
            if 0 <= index < len(self._queue):
                track = self._queue[index]
                self._queue.remove(track)
                self._emit(QueueEvent.TRACK_REMOVED, track, index)
                self._persist_async()
                return track
            return None
    
    def remove_by_filename(self, filename: str) -> bool:
        """Remove track by filename"""
        with self._lock:
            for i, track in enumerate(self._queue):
                if track.filename == filename:
                    self.remove(i)
                    return True
        return False
    
    def get(self, index: int) -> Optional[QueuedTrack]:
        """Get track at index without removing"""
        with self._lock:
            if 0 <= index < len(self._queue):
                return self._queue[index]
            return None
    
    def peek(self) -> Optional[QueuedTrack]:
        """View next track without removing"""
        return self.get(0)
    
    def pop(self) -> Optional[QueuedTrack]:
        """Remove and return next track"""
        with self._lock:
            if self._queue:
                track = self._queue.popleft()
                self._emit(QueueEvent.TRACK_REMOVED, track, 0)
                self._persist_async()
                return track
            return None
    
    def clear(self) -> list[QueuedTrack]:
        """Clear entire queue, returns cleared tracks"""
        with self._lock:
            cleared = list(self._queue)
            self._queue.clear()
            self._current_index = None
            self._emit(QueueEvent.QUEUE_CLEARED)
            self._persist_async()
            return cleared
    
    # ============ Queue Manipulation ============
    
    def move(self, from_index: int, to_index: int) -> bool:
        """Move track from one position to another"""
        with self._lock:
            if not (0 <= from_index < len(self._queue) and 0 <= to_index < len(self._queue)):
                return False
            
            track = self._queue[from_index]
            self._queue.remove(track)
            self._queue.insert(to_index, track)
            self._persist_async()
            return True
    
    def bump(self, index: int) -> Optional[QueuedTrack]:
        """Move track to front of queue"""
        return self.move(index, 0)
    
    def shuffle(self) -> None:
        """Shuffle queue (preserves current track position)"""
        with self._lock:
            if self._current_index is not None and self._current_index < len(self._queue):
                current = self._queue[self._current_index]
                remaining = list(self._queue)
                remaining.pop(self._current_index)
                random.shuffle(remaining)
                self._queue = deque([current] + remaining)
            else:
                self._queue = deque(random.sample(list(self._queue), len(self._queue)))
            
            self._shuffle_enabled = True
            self._emit(QueueEvent.QUEUE_SHUFFLED)
            self._persist_async()
    
    def sort_by_bpm(self, ascending: bool = True) -> None:
        """Sort queue by BPM"""
        with self._lock:
            self._queue = deque(sorted(self._queue, key=lambda t: t.bpm, reverse=not ascending))
            self._persist_async()
    
    def sort_by_energy(self, ascending: bool = True) -> None:
        """Sort queue by energy level"""
        with self._lock:
            self._queue = deque(sorted(self._queue, key=lambda t: t.energy, reverse=not ascending))
            self._persist_async()
    
    def sort_by_key(self) -> None:
        """Sort queue by musical key (Camelot wheel)"""
        with self._lock:
            # Simple key ordering - could be enhanced for harmonic mixing
            key_order = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', 
                        '5A', '5B', '6A', '6B', '7A', '7B', '8A', '8B',
                        '9A', '9B', '10A', '10B', '11A', '11B', '12A', '12B']
            self._queue = deque(sorted(self._queue, key=lambda t: key_order.index(t.key) if t.key in key_order else 99))
            self._persist_async()
    
    # ============ Playback Control ============
    
    def next(self) -> Optional[QueuedTrack]:
        """Get next track (handles repeat modes)"""
        with self._lock:
            if not self._queue:
                if self._repeat_mode == RepeatMode.ALL and self._history:
                    # Restart from beginning
                    self._queue = deepcopy(self._history)
                    self._history.clear()
                else:
                    return None
            
            track = self.pop()
            self._add_to_history(track)
            self._emit(QueueEvent.TRACK_PLAYING, track)
            return track
    
    def previous(self) -> Optional[QueuedTrack]:
        """Get previous track from history"""
        with self._lock:
            if self._history:
                track = self._history.pop()
                # Add back to front of queue
                self._queue.appendleft(track)
                self._emit(QueueEvent.TRACK_PLAYING, track)
                return track
            return None
    
    def set_repeat(self, mode: RepeatMode) -> None:
        """Set repeat mode"""
        self._repeat_mode = mode
    
    def get_repeat(self) -> RepeatMode:
        """Get current repeat mode"""
        return self._repeat_mode
    
    def set_shuffle(self, enabled: bool) -> None:
        """Enable/disable shuffle"""
        self._shuffle_enabled = enabled
    
    def is_shuffle(self) -> bool:
        """Check if shuffle is enabled"""
        return self._shuffle_enabled
    
    # ============ History Management ============
    
    def _add_to_history(self, track: QueuedTrack) -> None:
        """Add track to history"""
        self._history.append(track)
        while len(self._history) > self._max_history:
            self._history.popleft()
        self._emit(QueueEvent.HISTORY_UPDATED, track)
    
    def get_history(self) -> list[QueuedTrack]:
        """Get play history"""
        with self._lock:
            return list(self._history)
    
    def clear_history(self) -> None:
        """Clear play history"""
        with self._lock:
            self._history.clear()
            self._emit(QueueEvent.HISTORY_UPDATED)
    
    # ============ Queue Status ============
    
    @property
    def length(self) -> int:
        """Number of tracks in queue"""
        return len(self._queue)
    
    @property
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self._queue) == 0
    
    @property
    def total_duration(self) -> float:
        """Total duration of all tracks in queue (seconds)"""
        return sum(t.duration for t in self._queue)
    
    def __len__(self) -> int:
        return self.length
    
    def __iter__(self):
        """Iterate over queue tracks"""
        return iter(list(self._queue))
    
    def __getitem__(self, index: int) -> QueuedTrack:
        return self._queue[index]
    
    # ============ Persistence ============
    
    def _persist_async(self) -> None:
        """Save queue to disk asynchronously"""
        if not self._persist_path:
            return
        
        def _save():
            try:
                # Ensure directory exists
                self._persist_path.parent.mkdir(parents=True, exist_ok=True)
                self._save()
            except Exception as e:
                print(f"Failed to save queue: {e}")
        
        threading.Thread(target=_save, daemon=True).start()
    
    def _save(self) -> None:
        """Save queue to disk"""
        if not self._persist_path:
            return
        
        # Ensure directory exists
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'queue': [t.to_dict() for t in self._queue],
            'history': [t.to_dict() for t in self._history],
            'repeat_mode': self._repeat_mode.value,
            'shuffle_enabled': self._shuffle_enabled,
            'saved_at': datetime.now().isoformat()
        }
        
        # Write directly (simpler than atomic rename for cross-platform)
        with open(self._persist_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> None:
        """Load queue from disk"""
        if not self._persist_path or not self._persist_path.exists():
            return
        
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            
            self._queue = deque(QueuedTrack.from_dict(t) for t in data.get('queue', []))
            self._history = deque(QueuedTrack.from_dict(t) for t in data.get('history', []))
            self._repeat_mode = RepeatMode(data.get('repeat_mode', 'off'))
            self._shuffle_enabled = data.get('shuffle_enabled', False)
            
        except Exception as e:
            print(f"Failed to load queue: {e}")
    
    def save(self) -> bool:
        """Manually save queue (blocking)"""
        try:
            self._save()
            return True
        except Exception as e:
            print(f"Failed to save queue: {e}")
            return False
    
    # ============ Queue Display ============
    
    def preview(self, lines: int = 10) -> str:
        """Generate preview string of queue"""
        with self._lock:
            if not self._queue:
                return "Queue is empty"
            
            total_time = int(self.total_duration)
            mins, secs = divmod(total_time, 60)
            
            parts = [f"📜 Queue ({len(self._queue)} tracks, ~{mins}m {secs}s)"]
            
            for i, track in enumerate(list(self._queue)[:lines]):
                emoji = "▶️" if i == 0 else "  "
                priority_indicator = "⭐" if track.priority > 0 else " "
                parts.append(f"{emoji} {i+1}. {priority_indicator}{track.name} - {track.artist}")
                parts.append(f"    BPM: {track.bpm} | Key: {track.key} | Energy: {track.energy:.2f}")
            
            if len(self._queue) > lines:
                parts.append(f"  ... and {len(self._queue) - lines} more")
            
            return "\n".join(parts)


# ============ Helper Functions ============

def create_queued_track(
    filename: str,
    artist: str = "Unknown",
    genre: str = "Unknown",
    bpm: float = 120.0,
    key: str = "1A",
    energy: float = 0.5,
    duration: float = 180.0,
    priority: int = 0,
    requested_by: Optional[str] = None
) -> QueuedTrack:
    """Factory function to create a QueuedTrack"""
    return QueuedTrack(
        filename=filename,
        artist=artist,
        genre=genre,
        bpm=bpm,
        key=key,
        energy=energy,
        duration=duration,
        priority=priority,
        requested_by=requested_by
    )


# ============ Example Usage ============

if __name__ == "__main__":
    # Demo
    queue = TrackQueue(persist_path=Path("/Users/johnpeter/ai-dj-project/data/queue.json"))
    
    # Add some tracks
    tracks = [
        create_queued_track("drake_in_my_feings.wav", "Drake", "Hip-Hop", 185, "2A", 0.7, 210),
        create_queued_track("travis_sicko_mode.wav", "Travis Scott", "Trap", 152, "3A", 0.8, 240),
        create_queued_track("marshmello_happier.wav", "Marshmello", "EDM", 99, "6A", 0.6, 210),
    ]
    
    for track in tracks:
        queue.add(track)
    
    print(queue.preview())
    print(f"\nTotal: {queue.length} tracks, {int(queue.total_duration)}s")
    
    # Test shuffle
    queue.shuffle()
    print("\nAfter shuffle:")
    print(queue.preview(3))
    
    # Test priority
    queue.add(create_queued_track("urgent_track.wav", "VIP", priority=10))
    print("\nAfter priority track:")
    print(queue.preview(3))
