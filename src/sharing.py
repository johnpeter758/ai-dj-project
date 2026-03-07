"""
AI DJ Track Sharing System
A comprehensive system for sharing tracks via links, QR codes, and external platforms.
"""

import json
import hashlib
import uuid
import os
import base64
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from enum import Enum
import urllib.parse


class SharePermission(Enum):
    """Permission levels for shared tracks."""
    VIEW_ONLY = "view"
    DOWNLOAD = "download"
    REMIX = "remix"
    COLLABORATE = "collab"


class SharePlatform(Enum):
    """Platforms for sharing."""
    LINK = "link"
    DISCORD = "discord"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    QR_CODE = "qr"


@dataclass
class ShareLink:
    """A shareable link for a track."""
    link_id: str
    track_id: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    permission: SharePermission = SharePermission.VIEW_ONLY
    max_uses: Optional[int] = None
    uses_count: int = 0
    password: Optional[str] = None
    is_active: bool = True
    title: str = ""
    description: str = ""
    
    def is_valid(self) -> bool:
        """Check if link is still valid."""
        if not self.is_active:
            return False
        if self.max_uses and self.uses_count >= self.max_uses:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        data['permission'] = self.permission.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ShareLink':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        data['permission'] = SharePermission(data['permission'])
        return cls(**data)


@dataclass
class ShareAnalytics:
    """Analytics for a shared track."""
    share_id: str
    track_id: str
    views: int = 0
    downloads: int = 0
    plays: int = 0
    remixes: int = 0
    shares: int = 0
    unique_viewers: List[str] = field(default_factory=list)
    referrers: Dict[str, int] = field(default_factory=dict)
    countries: Dict[str, int] = field(default_factory=dict)
    first_shared: Optional[datetime] = None
    last_viewed: Optional[datetime] = None
    
    def record_view(self, viewer_id: str = "", referrer: str = "", country: str = "") -> None:
        """Record a view event."""
        self.views += 1
        self.last_viewed = datetime.now()
        if not self.first_shared:
            self.first_shared = datetime.now()
        
        if viewer_id and viewer_id not in self.unique_viewers:
            self.unique_viewers.append(viewer_id)
        
        if referrer:
            self.referrers[referrer] = self.referrers.get(referrer, 0) + 1
        
        if country:
            self.countries[country] = self.countries.get(country, 0) + 1
    
    def record_download(self) -> None:
        """Record a download event."""
        self.downloads += 1
    
    def record_play(self) -> None:
        """Record a play event."""
        self.plays += 1
    
    def record_remix(self) -> None:
        """Record a remix event."""
        self.remixes += 1
    
    def record_share(self, platform: str = "") -> None:
        """Record a share event."""
        self.shares += 1
        if platform:
            self.referrers[platform] = self.referrers.get(platform, 0) + 1
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['first_shared'] = self.first_shared.isoformat() if self.first_shared else None
        data['last_viewed'] = self.last_viewed.isoformat() if self.last_viewed else None
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ShareAnalytics':
        if data.get('first_shared'):
            data['first_shared'] = datetime.fromisoformat(data['first_shared'])
        if data.get('last_viewed'):
            data['last_viewed'] = datetime.fromisoformat(data['last_viewed'])
        return cls(**data)


@dataclass
class ShareCard:
    """Visual share card for a track."""
    track_id: str
    title: str
    artist: str
    genre: str
    bpm: int
    duration: str
    cover_url: str
    waveform_url: str = ""
    background_color: str = "#1a1a2e"
    text_color: str = "#ffffff"
    accent_color: str = "#6366f1"
    branding: str = "AI DJ"
    
    def to_dict(self) -> dict:
        return asdict(self)


class SharingSystem:
    """
    Main sharing system for AI DJ tracks.
    Features: Share links, QR codes, platform integration, analytics.
    """
    
    def __init__(self, data_dir: str = "data/sharing"):
        self.data_dir = data_dir
        self.share_links: Dict[str, ShareLink] = {}
        self.analytics: Dict[str, ShareAnalytics] = {}
        self.track_shares: Dict[str, List[str]] = {}  # track_id -> [share_ids]
        
        # Base URL for share links
        self.base_url = "https://ai-dj.app/share"
        
        self._ensure_data_dir()
        self._load_data()
    
    def _ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    # ==================== SHARE LINKS ====================
    
    def create_share_link(
        self,
        track_id: str,
        created_by: str,
        permission: SharePermission = SharePermission.VIEW_ONLY,
        expires_in_days: Optional[int] = None,
        max_uses: Optional[int] = None,
        password: Optional[str] = None,
        title: str = "",
        description: str = ""
    ) -> ShareLink:
        """Create a new shareable link for a track."""
        link_id = self._generate_link_id(track_id)
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        share_link = ShareLink(
            link_id=link_id,
            track_id=track_id,
            created_by=created_by,
            expires_at=expires_at,
            permission=permission,
            max_uses=max_uses,
            password=password,
            title=title,
            description=description
        )
        
        self.share_links[link_id] = share_link
        
        # Track which shares belong to which track
        if track_id not in self.track_shares:
            self.track_shares[track_id] = []
        self.track_shares[track_id].append(link_id)
        
        # Initialize analytics
        self.analytics[link_id] = ShareAnalytics(
            share_id=link_id,
            track_id=track_id
        )
        
        self._save_data()
        return share_link
    
    def _generate_link_id(self, track_id: str) -> str:
        """Generate a unique share link ID."""
        unique_str = f"{track_id}{uuid.uuid4().hex[:8]}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    def get_share_link(self, link_id: str) -> Optional[ShareLink]:
        """Get a share link by ID."""
        return self.share_links.get(link_id)
    
    def get_track_shares(self, track_id: str) -> List[ShareLink]:
        """Get all share links for a track."""
        link_ids = self.track_shares.get(track_id, [])
        return [self.share_links[lid] for lid in link_ids if lid in self.share_links]
    
    def generate_share_url(self, link_id: str) -> str:
        """Generate the full share URL."""
        return f"{self.base_url}/{link_id}"
    
    def validate_share_link(
        self, 
        link_id: str, 
        password: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a share link.
        Returns (is_valid, error_message).
        """
        link = self.share_links.get(link_id)
        
        if not link:
            return False, "Link not found"
        
        if not link.is_valid():
            return False, "Link expired or no longer valid"
        
        if link.password and link.password != password:
            return False, "Invalid password"
        
        return True, None
    
    def use_share_link(self, link_id: str, viewer_id: str = "") -> bool:
        """Record usage of a share link."""
        link = self.share_links.get(link_id)
        
        if not link or not link.is_valid():
            return False
        
        link.uses_count += 1
        
        # Record analytics
        if link_id in self.analytics:
            self.analytics[link_id].record_view(viewer_id=viewer_id)
        
        self._save_data()
        return True
    
    def revoke_share_link(self, link_id: str) -> bool:
        """Revoke a share link."""
        link = self.share_links.get(link_id)
        
        if not link:
            return False
        
        link.is_active = False
        self._save_data()
        return True
    
    def extend_share_link(
        self, 
        link_id: str, 
        days: Optional[int] = None,
        uses: Optional[int] = None
    ) -> bool:
        """Extend a share link's expiration or uses."""
        link = self.share_links.get(link_id)
        
        if not link:
            return False
        
        if days:
            if link.expires_at:
                link.expires_at = link.expires_at + timedelta(days=days)
            else:
                link.expires_at = datetime.now() + timedelta(days=days)
        
        if uses and link.max_uses:
            link.max_uses += uses
        elif uses:
            link.max_uses = uses
        
        self._save_data()
        return True
    
    # ==================== QR CODE ====================
    
    def generate_qr_data(self, link_id: str) -> str:
        """Generate QR code data for a share link."""
        share_url = self.generate_share_url(link_id)
        # QR data is just the URL - actual QR generation would be done client-side
        return share_url
    
    # ==================== PLATFORM SHARING ====================
    
    def get_share_text(
        self,
        track_id: str,
        platform: SharePlatform = SharePlatform.LINK
    ) -> str:
        """Generate share text for a platform."""
        # This would need track metadata - return placeholder
        return f"Check out this AI-generated track! {self.base_url}/{track_id}"
    
    def get_platform_share_url(
        self,
        link_id: str,
        platform: SharePlatform
    ) -> str:
        """Get a share URL for a specific platform."""
        share_url = self.generate_share_url(link_id)
        encoded_url = urllib.parse.quote(share_url)
        
        platform_urls = {
            SharePlatform.DISCORD: f"https://discord.com/share?url={encoded_url}",
            SharePlatform.TWITTER: f"https://twitter.com/intent/tweet?url={encoded_url}",
            SharePlatform.FACEBOOK: f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}",
            SharePlatform.WHATSAPP: f"https://wa.me/?text={encoded_url}",
            SharePlatform.EMAIL: f"mailto:?subject=Check%20out%20this%20track&body={encoded_url}",
        }
        
        return platform_urls.get(platform, share_url)
    
    # ==================== SHARE CARDS ====================
    
    def create_share_card(
        self,
        track_id: str,
        title: str,
        artist: str,
        genre: str,
        bpm: int,
        duration_seconds: int,
        cover_url: str = "",
        **kwargs
    ) -> ShareCard:
        """Create a share card for a track."""
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        duration_str = f"{minutes}:{seconds:02d}"
        
        return ShareCard(
            track_id=track_id,
            title=title,
            artist=artist,
            genre=genre,
            bpm=bpm,
            duration=duration_str,
            cover_url=cover_url,
            **{k: v for k, v in kwargs.items() if k in ShareCard.__dataclass_fields__}
        )
    
    def get_share_card_html(self, card: ShareCard) -> str:
        """Generate HTML for a share card."""
        return f"""
        <div class="share-card" style="background: {card.background_color}; color: {card.text_color};">
            <img src="{card.cover_url}" alt="{card.title}" class="cover-art"/>
            <div class="track-info">
                <h3>{card.title}</h3>
                <p class="artist">{card.artist}</p>
                <div class="meta">
                    <span class="genre">{card.genre}</span>
                    <span class="bpm">{card.bpm} BPM</span>
                    <span class="duration">{card.duration}</span>
                </div>
            </div>
            <div class="branding">{card.branding}</div>
        </div>
        """
    
    # ==================== ANALYTICS ====================
    
    def get_analytics(self, link_id: str) -> Optional[ShareAnalytics]:
        """Get analytics for a share link."""
        return self.analytics.get(link_id)
    
    def get_track_analytics(self, track_id: str) -> Dict:
        """Get combined analytics for all shares of a track."""
        shares = self.get_track_shares(track_id)
        
        total_views = 0
        total_downloads = 0
        total_plays = 0
        total_shares = 0
        unique_viewers = []
        
        for share in shares:
            if share.link_id in self.analytics:
                a = self.analytics[share.link_id]
                total_views += a.views
                total_downloads += a.downloads
                total_plays += a.plays
                total_shares += a.shares
                unique_viewers.extend(a.unique_viewers)
        
        return {
            "total_shares": len(shares),
            "total_views": total_views,
            "total_downloads": total_downloads,
            "total_plays": total_plays,
            "total_shares": total_shares,
            "unique_viewers": len(set(unique_viewers))
        }
    
    def record_download(self, link_id: str) -> None:
        """Record a download for analytics."""
        if link_id in self.analytics:
            self.analytics[link_id].record_download()
            self._save_data()
    
    def record_play(self, link_id: str) -> None:
        """Record a play for analytics."""
        if link_id in self.analytics:
            self.analytics[link_id].record_play()
            self._save_data()
    
    def record_remix(self, link_id: str) -> None:
        """Record a remix for analytics."""
        if link_id in self.analytics:
            self.analytics[link_id].record_remix()
            self._save_data()
    
    # ==================== BULK OPERATIONS ====================
    
    def get_all_active_shares(self) -> List[ShareLink]:
        """Get all active share links."""
        return [s for s in self.share_links.values() if s.is_active]
    
    def get_expiring_shares(self, days: int = 7) -> List[ShareLink]:
        """Get shares expiring within specified days."""
        cutoff = datetime.now() + timedelta(days=days)
        return [
            s for s in self.share_links.values()
            if s.is_active and s.expires_at and s.expires_at <= cutoff
        ]
    
    def cleanup_expired_links(self) -> int:
        """Remove expired links from storage."""
        expired = [
            link_id for link_id, link in self.share_links.items()
            if not link.is_valid()
        ]
        
        for link_id in expired:
            del self.share_links[link_id]
        
        if expired:
            self._save_data()
        
        return len(expired)
    
    # ==================== PERSISTENCE ====================
    
    def _save_data(self) -> None:
        """Save all data to JSON files."""
        try:
            # Save share links
            links_data = {lid: link.to_dict() for lid, link in self.share_links.items()}
            with open(f"{self.data_dir}/links.json", "w") as f:
                json.dump(links_data, f, indent=2)
            
            # Save analytics
            analytics_data = {aid: a.to_dict() for aid, a in self.analytics.items()}
            with open(f"{self.data_dir}/analytics.json", "w") as f:
                json.dump(analytics_data, f, indent=2)
            
            # Save track shares mapping
            with open(f"{self.data_dir}/track_shares.json", "w") as f:
                json.dump(self.track_shares, f, indent=2)
                
        except Exception as e:
            print(f"Error saving sharing data: {e}")
    
    def _load_data(self) -> None:
        """Load all data from JSON files."""
        # Load share links
        try:
            with open(f"{self.data_dir}/links.json", "r") as f:
                data = json.load(f)
                self.share_links = {lid: ShareLink.from_dict(d) for lid, d in data.items()}
        except FileNotFoundError:
            pass
        
        # Load analytics
        try:
            with open(f"{self.data_dir}/analytics.json", "r") as f:
                data = json.load(f)
                self.analytics = {aid: ShareAnalytics.from_dict(d) for aid, d in data.items()}
        except FileNotFoundError:
            pass
        
        # Load track shares mapping
        try:
            with open(f"{self.data_dir}/track_shares.json", "r") as f:
                self.track_shares = json.load(f)
        except FileNotFoundError:
            pass


# ==================== UTILITY FUNCTIONS ====================

def format_duration(seconds: int) -> str:
    """Format duration in seconds to MM:SS."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def generate_short_code(length: int = 8) -> str:
    """Generate a short random code."""
    return uuid.uuid4().hex[:length]


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize sharing system
    sharing = SharingSystem()
    
    # Create a share link
    share_link = sharing.create_share_link(
        track_id="track_abc123",
        created_by="user_xyz",
        permission=SharePermission.DOWNLOAD,
        expires_in_days=30,
        max_uses=100,
        title="My Awesome Track",
        description="Check out this AI-generated banger!"
    )
    
    print(f"Created share link: {share_link.link_id}")
    print(f"Share URL: {sharing.generate_share_url(share_link.link_id)}")
    
    # Test link validation
    is_valid, error = sharing.validate_share_link(share_link.link_id)
    print(f"Link valid: {is_valid}")
    
    # Record some analytics
    sharing.use_share_link(share_link.link_id, viewer_id="viewer_1")
    sharing.use_share_link(share_link.link_id, viewer_id="viewer_2")
    sharing.record_download(share_link.link_id)
    sharing.record_play(share_link.link_id)
    
    # Get analytics
    analytics = sharing.get_analytics(share_link.link_id)
    if analytics:
        print(f"Views: {analytics.views}")
        print(f"Downloads: {analytics.downloads}")
        print(f"Plays: {analytics.plays}")
        print(f"Unique viewers: {len(analytics.unique_viewers)}")
    
    # Get platform share URLs
    discord_url = sharing.get_platform_share_url(
        share_link.link_id, 
        SharePlatform.DISCORD
    )
    print(f"Discord share URL: {discord_url}")
    
    # Create a share card
    card = sharing.create_share_card(
        track_id="track_abc123",
        title="Neon Sunset",
        artist="AI DJ",
        genre="Electronic",
        bpm=128,
        duration_seconds=210,
        cover_url="https://example.com/cover.jpg"
    )
    print(f"Share card created: {card.title} by {card.artist}")
