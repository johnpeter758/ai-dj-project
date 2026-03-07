"""
Social Music Platform - AI DJ Project
A collaborative music sharing and voting platform for AI-generated music.
"""

import json
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from enum import Enum


class CollaborationRole(Enum):
    """Roles in a music collaboration."""
    CREATOR = "creator"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"


@dataclass
class UserProfile:
    """User profile for the social platform."""
    user_id: str
    username: str
    display_name: str
    bio: str = ""
    avatar_url: str = ""
    website: str = ""
    joined_at: datetime = field(default_factory=datetime.now)
    tracks_shared: int = 0
    collaborations: int = 0
    total_votes_received: int = 0
    badges: List[str] = field(default_factory=list)
    favorite_genres: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['joined_at'] = self.joined_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        data['joined_at'] = datetime.fromisoformat(data['joined_at'])
        return cls(**data)


@dataclass
class Track:
    """A shared music track."""
    track_id: str
    title: str
    artist_id: str
    artist_name: str
    description: str = ""
    genre: str = ""
    bpm: int = 128
    duration_seconds: int = 180
    file_path: str = ""
    cover_art_url: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    upvotes: int = 0
    downvotes: int = 0
    play_count: int = 0
    tags: List[str] = field(default_factory=list)
    
    @property
    def vote_score(self) -> int:
        return self.upvotes - self.downvotes
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class Collaboration:
    """A music collaboration between users."""
    collab_id: str
    title: str
    description: str = ""
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "open"  # open, in_progress, completed
    contributors: List[Dict] = field(default_factory=list)
    tracks: List[str] = field(default_factory=list)
    votes: int = 0
    
    def add_contributor(self, user_id: str, role: CollaborationRole = CollaborationRole.CONTRIBUTOR) -> None:
        self.contributors.append({
            'user_id': user_id,
            'role': role.value,
            'joined_at': datetime.now().isoformat()
        })
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


class SocialPlatform:
    """
    Main social music platform class.
    Features: User profiles, music sharing, collaboration, voting.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.users: Dict[str, UserProfile] = {}
        self.tracks: Dict[str, Track] = {}
        self.collaborations: Dict[str, Collaboration] = {}
        self.votes: Dict[str, List[str]] = {}  # track_id -> [user_ids]
        
    # ==================== USER PROFILES ====================
    
    def create_user(self, username: str, display_name: str, bio: str = "", 
                    avatar_url: str = "") -> UserProfile:
        """Create a new user profile."""
        user_id = str(uuid.uuid4())[:8]
        user = UserProfile(
            user_id=user_id,
            username=username,
            display_name=display_name,
            bio=bio,
            avatar_url=avatar_url
        )
        self.users[user_id] = user
        self._save_users()
        return user
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get a user by ID."""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """Get a user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def update_user(self, user_id: str, **kwargs) -> Optional[UserProfile]:
        """Update user profile fields."""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        self._save_users()
        return user
    
    def get_leaderboard(self, limit: int = 10) -> List[UserProfile]:
        """Get top users by vote score."""
        sorted_users = sorted(
            self.users.values(), 
            key=lambda u: u.total_votes_received, 
            reverse=True
        )
        return sorted_users[:limit]
    
    # ==================== MUSIC SHARING ====================
    
    def share_track(self, title: str, artist_id: str, artist_name: str,
                    description: str = "", genre: str = "", bpm: int = 128,
                    duration_seconds: int = 180, file_path: str = "",
                    tags: List[str] = None) -> Track:
        """Share a new track."""
        track_id = str(uuid.uuid4())[:8]
        track = Track(
            track_id=track_id,
            title=title,
            artist_id=artist_id,
            artist_name=artist_name,
            description=description,
            genre=genre,
            bpm=bpm,
            duration_seconds=duration_seconds,
            file_path=file_path,
            tags=tags or []
        )
        self.tracks[track_id] = track
        
        # Update user stats
        if artist_id in self.users:
            self.users[artist_id].tracks_shared += 1
        
        self._save_tracks()
        return track
    
    def get_track(self, track_id: str) -> Optional[Track]:
        """Get a track by ID."""
        return self.tracks.get(track_id)
    
    def get_user_tracks(self, user_id: str) -> List[Track]:
        """Get all tracks by a user."""
        return [t for t in self.tracks.values() if t.artist_id == user_id]
    
    def get_trending_tracks(self, limit: int = 10) -> List[Track]:
        """Get trending tracks by vote score."""
        sorted_tracks = sorted(
            self.tracks.values(),
            key=lambda t: t.vote_score,
            reverse=True
        )
        return sorted_tracks[:limit]
    
    def get_recent_tracks(self, limit: int = 10) -> List[Track]:
        """Get most recent tracks."""
        sorted_tracks = sorted(
            self.tracks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )
        return sorted_tracks[:limit]
    
    def increment_play_count(self, track_id: str) -> None:
        """Increment play count for a track."""
        if track_id in self.tracks:
            self.tracks[track_id].play_count += 1
            self._save_tracks()
    
    # ==================== VOTING ====================
    
    def vote(self, track_id: str, user_id: str, upvote: bool = True) -> bool:
        """Vote on a track. Returns True if successful."""
        if track_id not in self.tracks:
            return False
        
        if user_id not in self.users:
            return False
        
        # Check if user already voted
        if track_id not in self.votes:
            self.votes[track_id] = []
        
        track = self.tracks[track_id]
        
        # Toggle vote
        if user_id in self.votes[track_id]:
            # Remove previous vote
            prev_upvote = getattr(track, '_last_vote', None)
            if prev_upvote:
                if prev_upvote:
                    track.upvotes -= 1
                else:
                    track.downvotes -= 1
            self.votes[track_id].remove(user_id)
        
        # Add new vote
        if upvote:
            track.upvotes += 1
        else:
            track.downvotes += 1
        
        track._last_vote = upvote
        self.votes[track_id].append(user_id)
        
        # Update artist stats
        if track.artist_id in self.users:
            self.users[track.artist_id].total_votes_received += 1
        
        self._save_tracks()
        return True
    
    def get_user_vote(self, track_id: str, user_id: str) -> Optional[bool]:
        """Get user's vote on a track."""
        if track_id not in self.votes:
            return None
        # This is simplified - would need to track upvote/downvote separately
        return user_id in self.votes[track_id]
    
    # ==================== COLLABORATION ====================
    
    def create_collaboration(self, title: str, created_by: str,
                             description: str = "") -> Collaboration:
        """Create a new collaboration."""
        collab_id = str(uuid.uuid4())[:8]
        collab = Collaboration(
            collab_id=collab_id,
            title=title,
            description=description,
            created_by=created_by
        )
        collab.add_contributor(created_by, CollaborationRole.CREATOR)
        
        self.collaborations[collab_id] = collab
        
        # Update user stats
        if created_by in self.users:
            self.users[created_by].collaborations += 1
        
        self._save_collaborations()
        return collab
    
    def get_collaboration(self, collab_id: str) -> Optional[Collaboration]:
        """Get a collaboration by ID."""
        return self.collaborations.get(collab_id)
    
    def join_collaboration(self, collab_id: str, user_id: str,
                           role: CollaborationRole = CollaborationRole.CONTRIBUTOR) -> bool:
        """Join a collaboration."""
        if collab_id not in self.collaborations:
            return False
        
        if user_id not in self.users:
            return False
        
        collab = self.collaborations[collab_id]
        
        # Check if already a contributor
        for c in collab.contributors:
            if c['user_id'] == user_id:
                return False
        
        collab.add_contributor(user_id, role)
        
        # Update user stats
        self.users[user_id].collaborations += 1
        
        self._save_collaborations()
        return True
    
    def add_track_to_collaboration(self, collab_id: str, track_id: str) -> bool:
        """Add a track to a collaboration."""
        if collab_id not in self.collaborations:
            return False
        
        if track_id not in self.tracks:
            return False
        
        collab = self.collaborations[collab_id]
        if track_id not in collab.tracks:
            collab.tracks.append(track_id)
            self._save_collaborations()
        
        return True
    
    def vote_collaboration(self, collab_id: str) -> bool:
        """Vote on a collaboration."""
        if collab_id not in self.collaborations:
            return False
        
        self.collaborations[collab_id].votes += 1
        self._save_collaborations()
        return True
    
    def get_open_collaborations(self) -> List[Collaboration]:
        """Get all open collaborations."""
        return [c for c in self.collaborations.values() if c.status == "open"]
    
    def get_user_collaborations(self, user_id: str) -> List[Collaboration]:
        """Get collaborations a user is part of."""
        return [
            c for c in self.collaborations.values()
            if any(contrib['user_id'] == user_id for contrib in c.contributors)
        ]
    
    # ==================== PERSISTENCE ====================
    
    def _save_users(self) -> None:
        """Save users to JSON."""
        try:
            data = {uid: u.to_dict() for uid, u in self.users.items()}
            with open(f"{self.data_dir}/users.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")
    
    def _save_tracks(self) -> None:
        """Save tracks to JSON."""
        try:
            data = {tid: t.to_dict() for tid, t in self.tracks.items()}
            with open(f"{self.data_dir}/tracks.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving tracks: {e}")
    
    def _save_collaborations(self) -> None:
        """Save collaborations to JSON."""
        try:
            data = {cid: c.to_dict() for cid, c in self.collaborations.items()}
            with open(f"{self.data_dir}/collaborations.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving collaborations: {e}")
    
    def load_data(self) -> None:
        """Load all data from JSON files."""
        import os
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load users
        try:
            with open(f"{self.data_dir}/users.json", "r") as f:
                data = json.load(f)
                self.users = {uid: UserProfile.from_dict(d) for uid, d in data.items()}
        except FileNotFoundError:
            pass
        
        # Load tracks
        try:
            with open(f"{self.data_dir}/tracks.json", "r") as f:
                data = json.load(f)
                for tid, d in data.items():
                    d['created_at'] = datetime.fromisoformat(d['created_at'])
                    self.tracks[tid] = Track(**d)
        except FileNotFoundError:
            pass
        
        # Load collaborations
        try:
            with open(f"{self.data_dir}/collaborations.json", "r") as f:
                data = json.load(f)
                for cid, d in data.items():
                    d['created_at'] = datetime.fromisoformat(d['created_at'])
                    self.collaborations[cid] = Collaboration(**d)
        except FileNotFoundError:
            pass


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize platform
    platform = SocialPlatform()
    
    # Create users
    user1 = platform.create_user("musicfan", "DJ Awesome", "AI music enthusiast")
    user2 = platform.create_user("beatmaker", "Beat Master", "Producer & collaborator")
    
    print(f"Created user: {user1.display_name} ({user1.username})")
    print(f"Created user: {user2.display_name} ({user2.username})")
    
    # Share tracks
    track1 = platform.share_track(
        title="AI Sunset",
        artist_id=user1.user_id,
        artist_name=user1.display_name,
        genre="Electronic",
        bpm=128,
        description="A chill electronic track"
    )
    
    track2 = platform.share_track(
        title="Neon Dreams",
        artist_id=user2.user_id,
        artist_name=user2.display_name,
        genre="Synthwave",
        bpm=110,
        description="Retro synth vibes"
    )
    
    print(f"\nShared track: {track1.title} by {track1.artist_name}")
    print(f"Shared track: {track2.title} by {track2.artist_name}")
    
    # Vote on tracks
    platform.vote(track1.track_id, user2.user_id, upvote=True)
    platform.vote(track1.track_id, "test_user", upvote=True)  # Another user
    platform.vote(track2.track_id, user1.user_id, upvote=True)
    
    print(f"\n{track1.title} votes: {track1.upvotes} up, {track1.downvotes} down (score: {track1.vote_score})")
    print(f"{track2.title} votes: {track2.upvotes} up, {track2.downvotes} down (score: {track2.vote_score})")
    
    # Create collaboration
    collab = platform.create_collaboration(
        title="Summer Vibes EP",
        created_by=user1.user_id,
        description="A collaborative summer-themed EP"
    )
    
    print(f"\nCreated collaboration: {collab.title}")
    
    # Add contributor
    platform.join_collaboration(collab.collab_id, user2.user_id)
    platform.add_track_to_collaboration(collab.collab_id, track1.track_id)
    platform.add_track_to_collaboration(collab.collab_id, track2.track_id)
    
    print(f"Collaborators: {len(collab.contributors)}")
    print(f"Tracks in collab: {len(collab.tracks)}")
    
    # Get trending
    print("\n--- Trending Tracks ---")
    for track in platform.get_trending_tracks():
        print(f"  {track.title} by {track.artist_name} (score: {track.vote_score})")
