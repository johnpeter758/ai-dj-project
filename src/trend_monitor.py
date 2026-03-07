"""
Trend Monitor for AI DJ Project

Tracks music trends, monitors genre popularity, and recommends styles
based on current market data.
"""

import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

# Trend data storage path
TREND_DATA_DIR = Path(__file__).parent.parent / "data"
TREND_DATA_FILE = TREND_DATA_DIR / "trends.json"


@dataclass
class GenreTrend:
    """Represents a genre's trend data."""
    name: str
    popularity_score: float  # 0-100
    growth_rate: float        # -1 to 1 (declining to exploding)
    viral_potential: float    # 0-1
    last_updated: str
    tags: list[str]


@dataclass
class StyleRecommendation:
    """Represents a style recommendation."""
    name: str
    genre: str
    trend_score: float        # 0-100
    confidence: float         # 0-1
    keywords: list[str]
    target_bpm_range: tuple[int, int]
    energy_level: str         # low, medium, high
    description: str


class TrendMonitor:
    """Main class for tracking music trends and recommending styles."""

    # Default genre trends (can be updated via web scraping/API)
    DEFAULT_GENRES = {
        "afrobeats": GenreTrend(
            name="afrobeats",
            popularity_score=92,
            growth_rate=0.8,
            viral_potential=0.85,
            last_updated="2025-01-01",
            tags=["dance", "feel-good", "party", "west-africa"]
        ),
        "phonk": GenreTrend(
            name="phonk",
            popularity_score=88,
            growth_rate=0.9,
            viral_potential=0.95,
            last_updated="2025-01-01",
            tags=["drift", "hype", "808s", "cowbell", "aggressive"]
        ),
        "hyperpop": GenreTrend(
            name="hyperpop",
            popularity_score=75,
            growth_rate=0.6,
            viral_potential=0.8,
            last_updated="2025-01-01",
            tags=["glitch", "extreme", "experimental", "catchy"]
        ),
        "amapiano": GenreTrend(
            name="amapiano",
            popularity_score=85,
            growth_rate=0.7,
            viral_potential=0.75,
            last_updated="2025-01-01",
            tags=["south-africa", "deep-house", "groovy", "log-drumming"]
        ),
        "regional-mexican": GenreTrend(
            name="regional-mexican",
            popularity_score=90,
            growth_rate=0.75,
            viral_potential=0.8,
            last_updated="2025-01-01",
            tags=["corridos", "tumbados", "band", "acoustic"]
        ),
        "lo-fi-hip-hop": GenreTrend(
            name="lo-fi-hip-hop",
            popularity_score=78,
            growth_rate=0.2,
            viral_potential=0.6,
            last_updated="2025-01-01",
            tags=["chill", "study", "relaxing", "vinyl"]
        ),
        "pop-punk": GenreTrend(
            name="pop-punk",
            popularity_score=72,
            growth_rate=0.5,
            viral_potential=0.7,
            last_updated="2025-01-01",
            tags=["punk", "emo", "guitar", " energetic"]
        ),
        "drill": GenreTrend(
            name="drill",
            popularity_score=82,
            growth_rate=0.4,
            viral_potential=0.85,
            last_updated="2025-01-01",
            tags=["trap", "aggressive", "dark", "flow"]
        ),
        "trap-soul": GenreTrend(
            name="trap-soul",
            popularity_score=76,
            growth_rate=0.55,
            viral_potential=0.7,
            last_updated="2025-01-01",
            tags=["r&b", "melodic", "moody", "808s"]
        ),
        "techno": GenreTrend(
            name="techno",
            popularity_score=70,
            growth_rate=0.45,
            viral_potential=0.6,
            last_updated="2025-01-01",
            tags=["hard", "dark", "warehouse", "rhythmic"]
        ),
        "bass-house": GenreTrend(
            name="bass-house",
            popularity_score=74,
            growth_rate=0.5,
            viral_potential=0.7,
            last_updated="2025-01-01",
            tags=["heavy-bass", "four-on-floor", "festival"]
        ),
        "cloud-rap": GenreTrend(
            name="cloud-rap",
            popularity_score=68,
            growth_rate=0.6,
            viral_potential=0.75,
            last_updated="2025-01-01",
            tags=["atmospheric", "dreamy", "melodic", "lo-fi"]
        ),
    }

    # Style templates for recommendations
    STYLE_TEMPLATES = {
        "viral-tiktok": {
            "bpm_range": (120, 140),
            "energy": "high",
            "keywords": ["hook", "dance", "short", "catchy", "remixable"]
        },
        "chill-lofi": {
            "bpm_range": (70, 90),
            "energy": "low",
            "keywords": ["relaxing", "study", "sleep", "ambient"]
        },
        "festival-hype": {
            "bpm_range": (128, 140),
            "energy": "high",
            "keywords": ["drop", "buildup", "bass", "crowd"]
        },
        "bedroom-producer": {
            "bpm_range": (80, 120),
            "energy": "medium",
            "keywords": ["lo-fi", "chill", "sample", "vintage"]
        },
        "trap-heat": {
            "bpm_range": (140, 180),
            "energy": "high",
            "keywords": ["808s", "hi-hats", "dark", "aggressive"]
        },
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the trend monitor."""
        self.data_dir = data_dir or TREND_DATA_DIR
        self.data_file = self.data_dir / "trends.json"
        self.genres: dict[str, GenreTrend] = {}
        self.last_fetch: Optional[datetime] = None
        
        self._ensure_data_dir()
        self._load_trends()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_trends(self):
        """Load trends from file or use defaults."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    for genre_data in data.get('genres', []):
                        genre = GenreTrend(**genre_data)
                        self.genres[genre.name] = genre
                    self.last_fetch = datetime.fromisoformat(
                        data.get('last_fetch', datetime.now().isoformat())
                    )
            except Exception as e:
                print(f"Error loading trends: {e}, using defaults")
                self._load_defaults()
        else:
            self._load_defaults()

    def _load_defaults(self):
        """Load default genre trends."""
        self.genres = self.DEFAULT_GENRES.copy()
        self.last_fetch = datetime.now()
        self._save_trends()

    def _save_trends(self):
        """Save trends to file."""
        data = {
            'genres': [asdict(g) for g in self.genres.values()],
            'last_fetch': self.last_fetch.isoformat() if self.last_fetch else None
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def update_trends(self, genre_name: str, popularity: float = None, 
                      growth: float = None, viral: float = None):
        """Update trend data for a specific genre."""
        if genre_name not in self.genres:
            # Add new genre
            self.genres[genre_name] = GenreTrend(
                name=genre_name,
                popularity_score=popularity or 50,
                growth_rate=growth or 0,
                viral_potential=viral or 0.5,
                last_updated=datetime.now().isoformat(),
                tags=[]
            )
        else:
            genre = self.genres[genre_name]
            if popularity is not None:
                genre.popularity_score = popularity
            if growth is not None:
                genre.growth_rate = growth
            if viral is not None:
                genre.viral_potential = viral
            genre.last_updated = datetime.now().isoformat()
        
        self.last_fetch = datetime.now()
        self._save_trends()

    def get_genre_trends(self, min_popularity: float = 0, 
                        sort_by: str = "popularity") -> list[GenreTrend]:
        """Get genre trends, optionally filtered and sorted."""
        trends = [g for g in self.genres.values() 
                  if g.popularity_score >= min_popularity]
        
        if sort_by == "popularity":
            trends.sort(key=lambda x: x.popularity_score, reverse=True)
        elif sort_by == "growth":
            trends.sort(key=lambda x: x.growth_rate, reverse=True)
        elif sort_by == "viral":
            trends.sort(key=lambda x: x.viral_potential, reverse=True)
        
        return trends

    def get_trending_genres(self, limit: int = 5) -> list[GenreTrend]:
        """Get the top trending genres based on growth rate."""
        trends = sorted(self.genres.values(), 
                       key=lambda x: x.growth_rate, 
                       reverse=True)[:limit]
        return trends

    def get_viral_genres(self, limit: int = 5) -> list[GenreTrend]:
        """Get genres with highest viral potential."""
        trends = sorted(self.genres.values(), 
                       key=lambda x: x.viral_potential, 
                       reverse=True)[:limit]
        return trends

    def recommend_styles(self, context: str = "general", 
                        target_energy: str = None,
                        bpm_range: tuple = None,
                        limit: int = 5) -> list[StyleRecommendation]:
        """Recommend styles based on current trends and context."""
        recommendations = []
        
        # Get relevant genres based on context
        if context == "tiktok-viral":
            relevant_genres = self.get_viral_genres(limit=8)
        elif context == "festival":
            relevant_genres = [g for g in self.genres.values() 
                              if "house" in g.name or "techno" in g.name]
            relevant_genres.sort(key=lambda x: x.popularity_score, reverse=True)
        elif context == "chill":
            relevant_genres = [g for g in self.genres.values() 
                              if "lo-fi" in g.name or "chill" in g.name]
        else:
            relevant_genres = self.get_trending_genres(limit=8)

        # Generate recommendations
        for genre in relevant_genres[:limit]:
            # Calculate trend score
            trend_score = (
                genre.popularity_score * 0.4 +
                (genre.growth_rate + 1) * 50 * 0.3 +  # Normalize -1:1 to 0:100
                genre.viral_potential * 100 * 0.3
            )
            
            # Get style default
            template = self.STYLE_TEMPLATES.get(genre.name, 
                self.STYLE_TEMPLATES.get("bedroom-producer"))
            
            # Override BPM if specified
            bpm = bpm_range or template["bpm_range"]
            
            # Override energy if specified
            energy = target_energy or template["energy"]
            
            # Calculate confidence based on data freshness
            last_updated = datetime.fromisoformat(genre.last_updated)
            days_old = (datetime.now() - last_updated).days
            confidence = max(0.3, 1.0 - (days_old / 90))  # Decay over 90 days
            
            recommendation = StyleRecommendation(
                name=f"{genre.name}-{context}",
                genre=genre.name,
                trend_score=trend_score,
                confidence=confidence,
                keywords=genre.tags + template["keywords"],
                target_bpm_range=bpm,
                energy_level=energy,
                description=f"Trending {genre.name} with {energy} energy"
            )
            recommendations.append(recommendation)
        
        # Sort by trend score
        recommendations.sort(key=lambda x: x.trend_score, reverse=True)
        return recommendations[:limit]

    def get_production_tips(self, genre: str = None) -> dict:
        """Get production tips for a genre or general tips."""
        general_tips = [
            "Sidechain compression to kick for pump",
            "Wide stereo on synths and pads",
            "Quick punchy intros (8 bars max)",
            "Big simple drops",
            "Mid-side EQ for better stereo image"
        ]
        
        genre_tips = {
            "phonk": [
                "Cowbell prominence",
                "Aggressive 808s with long release",
                "Heavy reverb on vocals",
                "Drift-style modulation"
            ],
            "afrobeats": [
                "Polyrhythmic percussion",
                "Synth hooks with groove",
                "Ample space for vocal ad-libs",
                "Four-on-floor kick pattern"
            ],
            "hyperpop": [
                "Extreme vocal processing",
                "Pitch-shifted hooks",
                "Glitchy transitions",
                "High-energy arrangements"
            ],
            "amapiano": [
                "Log drum patterns",
                "Deep bass",
                "Warm piano chords",
                "Groovy percussion loops"
            ],
            "lo-fi-hip-hop": [
                "Vinyl crackle samples",
                "Dusty drum breaks",
                "Warm saturation",
                "Mellow jazz samples"
            ],
            "trap": [
                "Triplet hi-hat patterns",
                "Reverb tails on 808s",
                "Dark melodic elements",
                "Aggressive mixing"
            ]
        }
        
        return {
            "general": general_tips,
            "genre_specific": genre_tips.get(genre, []),
            "viral_tips": [
                "Instant hook in 0-5 seconds",
                "15-second extractability",
                "Dance/trend potential",
                "Emotional resonance",
                "Remix-friendly stems"
            ]
        }

    def needs_update(self, max_age_days: int = 7) -> bool:
        """Check if trends need updating."""
        if not self.last_fetch:
            return True
        return (datetime.now() - self.last_fetch).days >= max_age_days

    def refresh_from_web(self):
        """Placeholder for web-based trend fetching."""
        # This would integrate with chart APIs (Spotify, Billboard, etc.)
        # For now, returns info about manual update
        return {
            "status": "manual",
            "message": "Web refresh not configured. Use update_trends() to manually update.",
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None
        }


# Convenience function for quick access
def get_recommendations(context: str = "general", limit: int = 5) -> list[StyleRecommendation]:
    """Quick way to get style recommendations."""
    monitor = TrendMonitor()
    return monitor.recommend_styles(context=context, limit=limit)


def get_trending_genres(limit: int = 5) -> list[GenreTrend]:
    """Quick way to get trending genres."""
    monitor = TrendMonitor()
    return monitor.get_trending_genres(limit=limit)


if __name__ == "__main__":
    # Demo usage
    print("=" * 50)
    print("AI DJ Trend Monitor")
    print("=" * 50)
    
    monitor = TrendMonitor()
    
    print("\n📈 TRENDING GENRES:")
    for genre in monitor.get_trending_genres(5):
        print(f"  • {genre.name}: {genre.popularity_score:.0f}%, "
              f"growth: {genre.growth_rate*100:+.0f}%")
    
    print("\n🔥 VIRAL POTENTIAL:")
    for genre in monitor.get_viral_genres(5):
        print(f"  • {genre.name}: {genre.viral_potential*100:.0f}%")
    
    print("\n🎵 TIKTOK RECOMMENDATIONS:")
    for rec in monitor.recommend_styles("tiktok-viral", limit=3):
        print(f"  • {rec.genre} ({rec.energy_level}) - "
              f"BPM: {rec.target_bpm_range[0]}-{rec.target_bpm_range[1]}")
    
    print("\n💡 PRODUCTION TIPS (General):")
    for tip in monitor.get_production_tips()["general"][:3]:
        print(f"  • {tip}")
