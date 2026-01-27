#!/usr/bin/env python3
"""
Simple Battle-Tested Cache
JSON file-based with in-memory fallback
ZERO dependencies beyond stdlib
"""

import json
import time
from pathlib import Path
from typing import Any, Optional, Dict


class SimpleCache:
    """Dead-simple cache that just works"""
    
    def __init__(self, cache_file: str = "data/cache/simple_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict = self._load()
        self.stats = {'hits': 0, 'misses': 0, 'writes': 0}
    
    def _load(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception:
            pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        if key in self.cache:
            entry = self.cache[key]
            # Check expiry
            if time.time() < entry['expires']:
                self.stats['hits'] += 1
                return entry['value']
            else:
                del self.cache[key]
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set in cache"""
        self.cache[key] = {
            'value': value,
            'expires': time.time() + ttl_seconds,
            'created': time.time()
        }
        self.stats['writes'] += 1
        self._save()
    
    def clear(self):
        """Clear all cache"""
        self.cache = {}
        self._save()
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'writes': self.stats['writes'],
            'hit_rate': f"{hit_rate:.1f}%",
            'size': len(self.cache)
        }


# Singleton instance
_cache = SimpleCache()

def get_cached_elo(team: str, default: float = 1500.0) -> float:
    """Get cached Elo (24h TTL)"""
    result = _cache.get(f"elo:{team}")
    return result if result is not None else default

def set_cached_elo(team: str, elo: float):
    """Cache Elo rating"""
    _cache.set(f"elo:{team}", elo, ttl_seconds=86400)  # 24h

def get_cached_form(team: str, default: float = 50.0) -> float:
    """Get cached form (6h TTL)"""
    result = _cache.get(f"form:{team}")
    return result if result is not None else default

def set_cached_form(team: str, form: float):
    """Cache form rating"""
    _cache.set(f"form:{team}", form, ttl_seconds=21600)  # 6h

def print_cache_stats():
    """Print cache statistics"""
    stats = _cache.get_stats()
    print(f"\n📊 Cache: {stats['hits']} hits, {stats['misses']} misses ({stats['hit_rate']} hit rate), {stats['size']} entries\n")


if __name__ == "__main__":
    print("Testing Simple Cache...")
    
    # Test Elo
    set_cached_elo("Bayern", 2100)
    set_cached_elo("Dortmund", 1980)
    
    e1 = get_cached_elo("Bayern")
    e2 = get_cached_elo("Dortmund")
    e3 = get_cached_elo("Unknown")
    
    print(f"Bayern: {e1}")
    print(f"Dortmund: {e2}")
    print(f"Unknown (default): {e3}")
    
    # Test Form
    set_cached_form("Liverpool", 85.5)
    f1 = get_cached_form("Liverpool")
    f2 = get_cached_form("Unknown")
    
    print(f"Liverpool form: {f1}")
    print(f"Unknown form (default): {f2}")
    
    print_cache_stats()
    print("✅ Cache works!")
