"""
AI DJ Project - Lightweight Caching Module

Simple and fast caching utilities for the music generation system.
Provides decorator-based caching and simple key-value storage.

Usage:
    from cache import cache, cached
    
    # Set/get values
    cache.set("bpm_128", {"tracks": [...]})
    data = cache.get("bpm_128")
    
    # Decorator for function results
    @cached(ttl=300)
    def analyze_audio(path):
        # expensive operation
        return result
"""

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

# Cache directory
CACHE_DIR = Path("/Users/johnpeter/ai-dj-project/src/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache for fast access
_memory_cache: dict[str, tuple[Any, float, Optional[int]]] = {}  # key -> (value, created_at, ttl)


def _get_file_path(key: str) -> Path:
    """Get persistent cache file path for key."""
    key_hash = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{key_hash}.json"


def get(key: str, default: Any = None) -> Any:
    """
    Get value from cache.
    
    Args:
        key: Cache key
        default: Default value if not found
        
    Returns:
        Cached value or default
    """
    # Check memory first
    if key in _memory_cache:
        value, created_at, ttl = _memory_cache[key]
        if ttl is None or (time.time() - created_at) <= ttl:
            return value
        else:
            del _memory_cache[key]
    
    # Check disk
    path = _get_file_path(key)
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            value = data["value"]
            created_at = data.get("created_at", time.time())
            ttl = data.get("ttl")
            
            # Check expiry
            if ttl is not None and (time.time() - created_at) > ttl:
                path.unlink()
                return default
            
            # Store in memory for faster access
            _memory_cache[key] = (value, created_at, ttl)
            return value
        except (json.JSONDecodeError, KeyError, IOError):
            pass
    
    return default


def set(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """
    Set value in cache.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds (None = never expires)
    """
    created_at = time.time()
    
    # Store in memory
    _memory_cache[key] = (value, created_at, ttl)
    
    # Store on disk
    path = _get_file_path(key)
    try:
        with open(path, 'w') as f:
            json.dump({
                "value": value,
                "created_at": created_at,
                "ttl": ttl
            }, f)
    except (TypeError, IOError):
        pass  # Skip if not JSON serializable


def delete(key: str) -> bool:
    """
    Delete key from cache.
    
    Returns:
        True if key was deleted
    """
    # Remove from memory
    if key in _memory_cache:
        del _memory_cache[key]
    
    # Remove from disk
    path = _get_file_path(key)
    if path.exists():
        path.unlink()
        return True
    return False


def clear() -> None:
    """Clear all cached data."""
    _memory_cache.clear()
    
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()


def cleanup() -> int:
    """
    Remove expired entries.
    
    Returns:
        Number of entries removed
    """
    removed = 0
    now = time.time()
    
    # Clean memory
    expired = [k for k, (_, created, ttl) in _memory_cache.items() 
               if ttl is not None and (now - created) > ttl]
    for k in expired:
        del _memory_cache[k]
    removed += len(expired)
    
    # Clean disk
    for f in CACHE_DIR.glob("*.json"):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
            created = data.get("created_at", 0)
            ttl = data.get("ttl")
            if ttl is not None and (now - created) > ttl:
                f.unlink()
                removed += 1
        except (json.JSONDecodeError, IOError):
            f.unlink()
            removed += 1
    
    return removed


def cached(ttl: Optional[int] = 3600, key_prefix: str = ""):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds (default 1 hour)
        key_prefix: Optional prefix for cache keys
        
    Usage:
        @cached(ttl=600)
        def expensive_operation(x, y):
            return x + y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Build cache key
            cache_key = f"{key_prefix}{func.__module__}.{func.__name__}"
            cache_key += f":{':'.join(str(a) for a in args)}"
            cache_key += f":{':'.join(f'{k}={v}' for k, v in sorted(kwargs.items()))}"
            
            # Try to get cached result
            result = get(cache_key)
            if result is not None:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache it
            set(cache_key, result, ttl)
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: delete(cache_key)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def cached_property(ttl: Optional[int] = 3600):
    """
    Decorator for caching properties.
    
    Usage:
        class MyClass:
            @cached_property(ttl=300)
            def expensive(self):
                return compute_value()
    """
    def decorator(func):
        attr_name = f"_cached_{func.__name__}"
        
        def wrapper(self):
            if not hasattr(self, attr_name):
                setattr(self, attr_name, func(self))
            return getattr(self, attr_name)
        
        return property(wrapper)
    return decorator


# Specialized caches for common use cases

class AudioCache:
    """Specialized cache for audio analysis results."""
    
    @staticmethod
    def get_analysis(audio_path: str) -> Optional[dict]:
        """Get cached audio analysis."""
        key = f"audio_analysis:{hashlib.md5(audio_path.encode()).hexdigest()}"
        return get(key)
    
    @staticmethod
    def set_analysis(audio_path: str, analysis: dict, ttl: int = 86400) -> None:
        """Cache audio analysis (default 24 hours)."""
        key = f"audio_analysis:{hashlib.md5(audio_path.encode()).hexdigest()}"
        set(key, analysis, ttl)
    
    @staticmethod
    def get_bpm(audio_path: str) -> Optional[float]:
        """Get cached BPM."""
        key = f"bpm:{audio_path}"
        return get(key)
    
    @staticmethod
    def set_bpm(audio_path: str, bpm: float, ttl: int = 86400) -> None:
        """Cache BPM detection result."""
        key = f"bpm:{audio_path}"
        set(key, bpm, ttl)
    
    @staticmethod
    def get_key(audio_path: str) -> Optional[str]:
        """Get cached key detection."""
        key = f"key:{audio_path}"
        return get(key)
    
    @staticmethod
    def set_key(audio_path: str, musical_key: str, ttl: int = 86400) -> None:
        """Cache key detection result."""
        key = f"key:{audio_path}"
        set(key, musical_key, ttl)


class ModelCache:
    """Specialized cache for ML model outputs."""
    
    @staticmethod
    def get_model_output(model_name: str, input_hash: str) -> Optional[Any]:
        """Get cached model output."""
        key = f"model:{model_name}:{input_hash}"
        return get(key)
    
    @staticmethod
    def set_model_output(model_name: str, input_hash: str, output: Any, ttl: int = 3600) -> None:
        """Cache model output (default 1 hour)."""
        key = f"model:{model_name}:{input_hash}"
        set(key, output, ttl)


class GenreCache:
    """Specialized cache for genre classification."""
    
    @staticmethod
    def get_genre(track_hash: str) -> Optional[dict]:
        """Get cached genre classification."""
        key = f"genre:{track_hash}"
        return get(key)
    
    @staticmethod
    def set_genre(track_hash: str, genre_data: dict, ttl: int = 604800) -> None:
        """Cache genre classification (default 1 week)."""
        key = f"genre:{track_hash}"
        set(key, genre_data, ttl)


# Convenience aliases for common operations
# These provide a simple functional API

def cache(key: str, default: Any = None) -> Any:
    """Quick cache get (alias for get())."""
    return get(key, default)


# Global cache instance for convenience
class _CacheProxy:
    """Proxy for easy cache access."""
    
    def get(self, key: str, default: Any = None) -> Any:
        return get(key, default)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        return set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        return delete(key)
    
    def clear(self) -> None:
        return clear()
    
    def cleanup(self) -> int:
        return cleanup()
    
    def cached(self, ttl: Optional[int] = 3600, key_prefix: str = ""):
        return cached(ttl, key_prefix)


cache = _CacheProxy()
def configure(cache_dir: str = None, default_ttl: int = 3600) -> None:
    """
    Configure cache settings.
    
    Args:
        cache_dir: Custom cache directory
        default_ttl: Default TTL in seconds
    """
    global CACHE_DIR
    if cache_dir:
        CACHE_DIR = Path(cache_dir)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
