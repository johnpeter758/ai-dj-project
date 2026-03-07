"""
AI DJ Project Cache Manager
Provides disk and memory caching for the music generation system.
"""

import hashlib
import json
import os
import pickle
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Union


class CacheEntry:
    """Single cache entry with metadata."""
    
    def __init__(self, value: Any, ttl: Optional[int] = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl  # seconds, None = never expires
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        entry = cls(data["value"], data.get("ttl"))
        entry.created_at = data.get("created_at", time.time())
        return entry


class MemoryCache:
    """Thread-safe in-memory LRU cache."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = 3600):
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._cache[key]
                return None
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()
            
            ttl = ttl if ttl is not None else self.default_ttl
            self._cache[key] = CacheEntry(value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entry."""
        if not self._cache:
            return
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
    
    def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed."""
        with self._lock:
            expired = [k for k, v in self._cache.items() if v.is_expired()]
            for k in expired:
                del self._cache[k]
            return len(expired)
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "expired_count": sum(1 for v in self._cache.values() if v.is_expired())
            }


class DiskCache:
    """Persistent disk-based cache with JSON storage."""
    
    def __init__(self, cache_dir: str = None, default_ttl: Optional[int] = 86400):
        self.cache_dir = Path(cache_dir or "/Users/johnpeter/ai-dj-project/src/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl  # 24 hours default for disk
        self._lock = threading.RLock()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash the key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            path = self._get_cache_path(key)
            if not path.exists():
                return None
            
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                entry = CacheEntry.from_dict(data)
                if entry.is_expired():
                    path.unlink()
                    return None
                return entry.value
            except (json.JSONDecodeError, KeyError, IOError):
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in disk cache."""
        with self._lock:
            path = self._get_cache_path(key)
            ttl = ttl if ttl is not None else self.default_ttl
            entry = CacheEntry(value, ttl)
            
            try:
                with open(path, 'w') as f:
                    json.dump(entry.to_dict(), f)
            except IOError:
                pass  # Fail silently on disk errors
    
    def delete(self, key: str) -> bool:
        """Delete key from disk cache."""
        with self._lock:
            path = self._get_cache_path(key)
            if path.exists():
                path.unlink()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all disk cache entries."""
        with self._lock:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed."""
        with self._lock:
            removed = 0
            for f in self.cache_dir.glob("*.json"):
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                    entry = CacheEntry.from_dict(data)
                    if entry.is_expired():
                        f.unlink()
                        removed += 1
                except (json.JSONDecodeError, IOError):
                    f.unlink()  # Remove corrupted files
                    removed += 1
            return removed
    
    def stats(self) -> dict:
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "cache_dir": str(self.cache_dir)
        }


class CacheManager:
    """
    Unified caching system combining memory and disk caches.
    
    Usage:
        cache = CacheManager()
        
        # Basic get/set
        cache.set("song_123", {"title": "Test", "bpm": 120})
        data = cache.get("song_123")
        
        # With TTL (in seconds)
        cache.set("temp_data", value, ttl=300)  # 5 minutes
        
        # Clear caches
        cache.clear_memory()
        cache.clear_disk()
    """
    
    def __init__(
        self,
        memory_max_size: int = 1000,
        memory_ttl: int = 3600,
        disk_ttl: int = 86400,
        cache_dir: str = None
    ):
        self.memory = MemoryCache(max_size=memory_max_size, default_ttl=memory_ttl)
        self.disk = DiskCache(cache_dir=cache_dir, default_ttl=disk_ttl)
    
    def get(self, key: str, use_disk: bool = True) -> Optional[Any]:
        """Get value from memory first, then disk."""
        # Try memory first
        value = self.memory.get(key)
        if value is not None:
            return value
        
        # Fall back to disk
        if use_disk:
            value = self.disk.get(key)
            if value is not None:
                # Populate memory cache
                self.memory.set(key, value)
                return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, use_disk: bool = True) -> None:
        """Set value in both memory and disk."""
        self.memory.set(key, value, ttl)
        if use_disk:
            self.disk.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete from both caches."""
        mem_deleted = self.memory.delete(key)
        disk_deleted = self.disk.delete(key)
        return mem_deleted or disk_deleted
    
    def clear_memory(self) -> None:
        """Clear memory cache."""
        self.memory.clear()
    
    def clear_disk(self) -> None:
        """Clear disk cache."""
        self.disk.clear()
    
    def clear_all(self) -> None:
        """Clear both caches."""
        self.clear_memory()
        self.clear_disk()
    
    def cleanup(self) -> dict:
        """Clean up expired entries in both caches."""
        mem_cleaned = self.memory.cleanup_expired()
        disk_cleaned = self.disk.cleanup_expired()
        return {"memory_removed": mem_cleaned, "disk_removed": disk_cleaned}
    
    def stats(self) -> dict:
        """Get statistics for both caches."""
        return {
            "memory": self.memory.stats(),
            "disk": self.disk.stats()
        }
    
    def cached(self, ttl: Optional[int] = None, use_disk: bool = True) -> Callable:
        """
        Decorator for caching function results.
        
        Usage:
            @cache.cached(ttl=600)
            def expensive_function(x, y):
                # expensive computation
                return result
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [func.__module__, func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
                
                # Try to get from cache
                result = self.get(cache_key, use_disk=use_disk)
                if result is not None:
                    return result
                
                # Compute and cache
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl, use_disk=use_disk)
                return result
            
            wrapper.cache_clear = lambda: self.delete(cache_key)  # type: ignore
            return wrapper
        return decorator


# Global cache instance
_default_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get the default global cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheManager()
    return _default_cache
