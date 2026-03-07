#!/usr/bin/env python3
"""
AI DJ Collaboration System - Real-time Music Collaboration
"""

import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Callable, Optional
from dataclasses import dataclass, field, asdict
import threading


class CollaborationEvent(Enum):
    """Types of collaboration events"""
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    PLAYLIST_UPDATED = "playlist_updated"
    COMMENT_ADDED = "comment_added"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    PLAYBACK_SYNC = "playback_sync"
    TRACK_ADDED = "track_added"
    TRACK_REMOVED = "track_removed"
    MIX_STARTED = "mix_started"
    MIX_PROGRESS = "mix_progress"
    MIX_COMPLETED = "mix_completed"


@dataclass
class User:
    """Collaborator user"""
    id: str
    name: str
    avatar: str = ""
    color: str = "#6366f1"  # Default accent color
    is_online: bool = True
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    role: str = "collaborator"  # owner, collaborator, viewer


@dataclass
class Track:
    """Shared track in collaboration"""
    id: str
    title: str
    artist: str
    bpm: int = 128
    key: str = "8A"
    duration: int = 180  # seconds
    added_by: str = ""
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())
    audio_url: str = ""
    stem_urls: dict = field(default_factory=dict)


@dataclass
class Comment:
    """Comment on a track or session"""
    id: str
    user_id: str
    user_name: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    track_id: Optional[str] = None
    position: Optional[int] = None  # Timestamp position in seconds


@dataclass
class PlaybackState:
    """Current playback state for sync"""
    track_id: str
    position: float  # seconds
    is_playing: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_by: str = ""


class CollaborationSession:
    """Real-time collaboration session"""
    
    def __init__(self, session_id: str, owner: User, name: str = "New Session"):
        self.id = session_id
        self.name = name
        self.owner = owner
        self.created_at = datetime.now().isoformat()
        
        # Participants
        self.participants: dict[str, User] = {owner.id: owner}
        
        # Shared playlist
        self.tracks: dict[str, Track] = {}
        self.playlist_order: list[str] = []
        
        # Comments
        self.comments: list[Comment] = []
        
        # Playback
        self.playback_state: Optional[PlaybackState] = None
        self.current_track_index: int = 0
        
        # Mix state
        self.active_mix: Optional[dict] = None
        self.mix_progress: float = 0.0
        
        # Event callbacks
        self._event_handlers: dict[CollaborationEvent, list[Callable]] = {
            event: [] for event in CollaborationEvent
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Polling state for real-time updates
        self._last_poll = datetime.now().isoformat()
    
    def add_participant(self, user: User) -> bool:
        """Add a user to the session"""
        with self._lock:
            if user.id in self.participants:
                return False
            
            self.participants[user.id] = user
            self._emit(CollaborationEvent.USER_JOINED, {
                "user": asdict(user),
                "session_id": self.id,
                "timestamp": datetime.now().isoformat(),
            })
            return True
    
    def remove_participant(self, user_id: str) -> bool:
        """Remove a user from the session"""
        with self._lock:
            if user_id not in self.participants:
                return False
            
            if user_id == self.owner.id:
                return False  # Can't remove owner
            
            user = self.participants.pop(user_id)
            self._emit(CollaborationEvent.USER_LEFT, {
                "user_id": user_id,
                "user_name": user.name,
                "session_id": self.id,
                "timestamp": datetime.now().isoformat(),
            })
            return True
    
    def add_track(self, track: Track, user_id: str) -> str:
        """Add a track to the playlist"""
        with self._lock:
            track.added_by = user_id
            self.tracks[track.id] = track
            self.playlist_order.append(track.id)
            
            self._emit(CollaborationEvent.TRACK_ADDED, {
                "track": asdict(track),
                "position": len(self.playlist_order) - 1,
                "added_by": user_id,
                "timestamp": datetime.now().isoformat(),
            })
            return track.id
    
    def remove_track(self, track_id: str, user_id: str) -> bool:
        """Remove a track from the playlist"""
        with self._lock:
            if track_id not in self.tracks:
                return False
            
            del self.tracks[track_id]
            self.playlist_order.remove(track_id)
            
            self._emit(CollaborationEvent.TRACK_REMOVED, {
                "track_id": track_id,
                "removed_by": user_id,
                "timestamp": datetime.now().isoformat(),
            })
            return True
    
    def reorder_tracks(self, new_order: list[str], user_id: str) -> bool:
        """Reorder the playlist"""
        with self._lock:
            if set(new_order) != set(self.playlist_order):
                return False
            
            self.playlist_order = new_order
            self._emit(CollaborationEvent.PLAYLIST_UPDATED, {
                "new_order": new_order,
                "reordered_by": user_id,
                "timestamp": datetime.now().isoformat(),
            })
            return True
    
    def add_comment(self, comment: Comment) -> str:
        """Add a comment"""
        with self._lock:
            self.comments.append(comment)
            self._emit(CollaborationEvent.COMMENT_ADDED, {
                "comment": asdict(comment),
                "timestamp": datetime.now().isoformat(),
            })
            return comment.id
    
    def update_playback(self, playback_state: PlaybackState) -> None:
        """Update playback state (sync)"""
        with self._lock:
            self.playback_state = playback_state
            self._emit(CollaborationEvent.PLAYBACK_SYNC, {
                "playback": asdict(playback_state),
                "timestamp": datetime.now().isoformat(),
            })
    
    def start_mix(self, track1_id: str, track2_id: str, user_id: str) -> str:
        """Start a new mix between two tracks"""
        with self._lock:
            mix_id = str(uuid.uuid4())[:8]
            self.active_mix = {
                "id": mix_id,
                "track1_id": track1_id,
                "track2_id": track2_id,
                "started_by": user_id,
                "started_at": datetime.now().isoformat(),
                "progress": 0.0,
            }
            self._emit(CollaborationEvent.MIX_STARTED, {
                "mix": self.active_mix,
                "timestamp": datetime.now().isoformat(),
            })
            return mix_id
    
    def update_mix_progress(self, progress: float) -> None:
        """Update mix progress"""
        with self._lock:
            if self.active_mix:
                self.active_mix["progress"] = progress
                self.mix_progress = progress
                self._emit(CollaborationEvent.MIX_PROGRESS, {
                    "mix_id": self.active_mix["id"],
                    "progress": progress,
                    "timestamp": datetime.now().isoformat(),
                })
    
    def complete_mix(self, output_url: str) -> dict:
        """Complete the current mix"""
        with self._lock:
            if not self.active_mix:
                return {}
            
            self.active_mix["completed_at"] = datetime.now().isoformat()
            self.active_mix["output_url"] = output_url
            self.active_mix["progress"] = 100.0
            
            result = self.active_mix.copy()
            self._emit(CollaborationEvent.MIX_COMPLETED, {
                "mix": result,
                "timestamp": datetime.now().isoformat(),
            })
            
            self.active_mix = None
            self.mix_progress = 0.0
            return result
    
    def on(self, event: CollaborationEvent, callback: Callable) -> None:
        """Register event handler"""
        self._event_handlers[event].append(callback)
    
    def _emit(self, event: CollaborationEvent, data: dict) -> None:
        """Emit event to all handlers"""
        for callback in self._event_handlers[event]:
            try:
                callback(data)
            except Exception as e:
                print(f"Event handler error: {e}")
    
    def get_state(self) -> dict:
        """Get full session state"""
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "owner": asdict(self.owner),
                "created_at": self.created_at,
                "participants": {k: asdict(v) for k, v in self.participants.items()},
                "tracks": {k: asdict(v) for k, v in self.tracks.items()},
                "playlist_order": self.playlist_order,
                "comments": [asdict(c) for c in self.comments],
                "playback_state": asdict(self.playback_state) if self.playback_state else None,
                "current_track_index": self.current_track_index,
                "active_mix": self.active_mix,
                "mix_progress": self.mix_progress,
            }
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.get_state(), indent=2)


class CollaborationManager:
    """Manages multiple collaboration sessions"""
    
    def __init__(self):
        self.sessions: dict[str, CollaborationSession] = {}
        self.user_sessions: dict[str, set[str]] = {}  # user_id -> session_ids
        self._lock = threading.RLock()
    
    def create_session(self, owner: User, name: str = "New Session") -> CollaborationSession:
        """Create a new collaboration session"""
        with self._lock:
            session_id = str(uuid.uuid4())[:8]
            session = CollaborationSession(session_id, owner, name)
            self.sessions[session_id] = session
            
            if owner.id not in self.user_sessions:
                self.user_sessions[owner.id] = set()
            self.user_sessions[owner.id].add(session_id)
            
            return session
    
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def join_session(self, session_id: str, user: User) -> bool:
        """Join an existing session"""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            success = session.add_participant(user)
            if success:
                if user.id not in self.user_sessions:
                    self.user_sessions[user.id] = set()
                self.user_sessions[user.id].add(session_id)
            
            return success
    
    def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a session"""
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            success = session.remove_participant(user_id)
            if success and user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
            
            return success
    
    def get_user_sessions(self, user_id: str) -> list[CollaborationSession]:
        """Get all sessions for a user"""
        with self._lock:
            session_ids = self.user_sessions.get(user_id, set())
            return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
    
    def list_sessions(self) -> list[dict]:
        """List all available sessions"""
        with self._lock:
            return [
                {
                    "id": s.id,
                    "name": s.name,
                    "owner": s.owner.name,
                    "participant_count": len(s.participants),
                    "track_count": len(s.tracks),
                    "created_at": s.created_at,
                }
                for s in self.sessions.values()
            ]


# Real-time polling helper
class CollaborationPoller:
    """Poll for changes in collaboration sessions"""
    
    def __init__(self, manager: CollaborationManager, poll_interval: float = 1.0):
        self.manager = manager
        self.poll_interval = poll_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_states: dict[str, dict] = {}
    
    async def start(self):
        """Start polling"""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
    
    async def stop(self):
        """Stop polling"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _poll_loop(self):
        """Poll for changes"""
        while self._running:
            try:
                await self._check_for_changes()
            except Exception as e:
                print(f"Polling error: {e}")
            
            await asyncio.sleep(self.poll_interval)
    
    async def _check_for_changes(self):
        """Check for changes in all sessions"""
        for session in self.manager.sessions.values():
            current_state = session.get_state()
            session_id = session.id
            
            if session_id in self._last_states:
                last_state = self._last_states[session_id]
                
                # Check for new participants
                current_participants = set(current_state["participants"].keys())
                last_participants = set(last_state["participants"].keys())
                added = current_participants - last_participants
                removed = last_participants - current_participants
                
                # Check for track changes
                current_tracks = set(current_state["tracks"].keys())
                last_tracks = set(last_state["tracks"].keys())
                tracks_added = current_tracks - last_tracks
                tracks_removed = last_tracks - current_tracks
                
                # Check for comment changes
                current_comments = len(current_state["comments"])
                last_comments = len(last_state["comments"])
                
                # Check for playback changes
                current_playback = current_state.get("playback_state")
                last_playback = last_state.get("playback_state")
                
                # Emit callbacks for changes
                # (In a real implementation, this would trigger callbacks)
            
            self._last_states[session_id] = current_state


# Example usage
def main():
    """Demo the collaboration system"""
    print("=" * 50)
    print("🎛️ AI DJ Collaboration System")
    print("=" * 50)
    
    # Create manager
    manager = CollaborationManager()
    
    # Create users
    owner = User(
        id="user_1",
        name="Peter",
        avatar="🎧",
        color="#6366f1",
        role="owner"
    )
    
    collaborator = User(
        id="user_2", 
        name="Alex",
        avatar="🎵",
        color="#10b981",
        role="collaborator"
    )
    
    # Create session
    session = manager.create_session(owner, "Friday Mix Session")
    print(f"\n📁 Created session: {session.name} (ID: {session.id})")
    
    # Add collaborator
    session.add_participant(collaborator)
    print(f"👥 Added {collaborator.name} to session")
    
    # Add tracks
    track1 = Track(
        id="track_1",
        title="Summer Vibes",
        artist="DJ Peter",
        bpm=128,
        key="8A",
        duration=240,
    )
    track2 = Track(
        id="track_2",
        title="Night Drive",
        artist="Alex Beats",
        bpm=126,
        key="10A", 
        duration=210,
    )
    
    session.add_track(track1, owner.id)
    session.add_track(track2, collaborator.id)
    print(f"🎵 Added tracks: {track1.title}, {track2.title}")
    
    # Add comment
    comment = Comment(
        id="comment_1",
        user_id=collaborator.id,
        user_name=collaborator.name,
        content="This drop is fire! 🔥",
        track_id=track1.id,
        position=120,
    )
    session.add_comment(comment)
    print(f"💬 {collaborator.name}: {comment.content}")
    
    # Simulate playback sync
    playback = PlaybackState(
        track_id=track1.id,
        position=45.5,
        is_playing=True,
        updated_by=owner.id,
    )
    session.update_playback(playback)
    print(f"▶️  Playback synced: {track1.title} at {playback.position}s")
    
    # Start a mix
    mix_id = session.start_mix(track1.id, track2.id, owner.id)
    print(f"🎚️ Started mix: {mix_id}")
    
    # Update mix progress
    session.update_mix_progress(50.0)
    print(f"⏳ Mix progress: 50%")
    
    # Complete mix
    mix_result = session.complete_mix("/output/mix_001.mp3")
    print(f"✅ Mix completed: {mix_result['output_url']}")
    
    # Print session state
    print("\n📊 Session State:")
    print(session.to_json())
    
    print("\n✨ Collaboration system ready!")


if __name__ == "__main__":
    main()
