"""
💾 Model Cache - Analysis Caching for Cost Control
==================================================
Persistent caching of LLM analyses to avoid redundant inference.

Features:
- SQLite-based persistent storage
- TTL-based expiration
- Match-level and feature-level caching
- Cache statistics and cleanup
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelCache:
    """
    💾 Persistent Model Output Cache
    
    Caches LLM analysis results to:
    - Reduce inference costs
    - Speed up repeated queries
    - Enable offline operation
    
    Example:
        cache = ModelCache()
        
        # Check cache
        result = cache.get("bayern_dortmund_20260128", "match_analysis")
        
        if not result:
            result = llm.analyze(...)
            cache.set("bayern_dortmund_20260128", "match_analysis", result)
    """
    
    def __init__(
        self,
        db_path: str = None,
        default_ttl_hours: int = 24,
        max_entries: int = 10000
    ):
        self.db_path = db_path or str(Path(__file__).parent.parent.parent / "data" / "model_cache.db")
        self.default_ttl_hours = default_ttl_hours
        self.max_entries = max_entries
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    category TEXT,
                    value TEXT,
                    created_at TEXT,
                    expires_at TEXT,
                    hit_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_category 
                ON cache(category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires 
                ON cache(expires_at)
            """)
            
            conn.commit()
    
    def _generate_key(self, identifier: str, category: str) -> str:
        """Generate cache key from identifier and category."""
        key_data = f"{category}:{identifier}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def get(
        self,
        identifier: str,
        category: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached value.
        
        Args:
            identifier: Unique identifier (e.g., match_id)
            category: Cache category (e.g., "match_analysis", "features")
        
        Returns:
            Cached value or None
        """
        key = self._generate_key(identifier, category)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM cache WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                expires_at = datetime.fromisoformat(row['expires_at'])
                
                if expires_at > datetime.now():
                    # Update hit count
                    conn.execute(
                        "UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?",
                        (key,)
                    )
                    conn.commit()
                    
                    return json.loads(row['value'])
                else:
                    # Expired - delete
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
        
        return None
    
    def set(
        self,
        identifier: str,
        category: str,
        value: Dict[str, Any],
        ttl_hours: int = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Set cached value.
        
        Args:
            identifier: Unique identifier
            category: Cache category
            value: Value to cache (will be JSON serialized)
            ttl_hours: Time-to-live in hours (default from config)
            metadata: Additional metadata to store
        """
        key = self._generate_key(identifier, category)
        ttl = ttl_hours or self.default_ttl_hours
        
        now = datetime.now()
        expires = now + timedelta(hours=ttl)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache 
                (key, category, value, created_at, expires_at, hit_count, metadata)
                VALUES (?, ?, ?, ?, ?, 0, ?)
            """, (
                key,
                category,
                json.dumps(value),
                now.isoformat(),
                expires.isoformat(),
                json.dumps(metadata or {})
            ))
            conn.commit()
    
    def delete(self, identifier: str, category: str = "default"):
        """Delete cached value."""
        key = self._generate_key(identifier, category)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
    
    def clear_category(self, category: str):
        """Clear all entries in a category."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE category = ?", (category,))
            conn.commit()
    
    def clear_expired(self) -> int:
        """Clear all expired entries. Returns count deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM cache WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            )
            conn.commit()
            return cursor.rowcount
    
    def clear_all(self):
        """Clear entire cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Total entries
            total = conn.execute("SELECT COUNT(*) as count FROM cache").fetchone()['count']
            
            # By category
            categories = conn.execute("""
                SELECT category, COUNT(*) as count, SUM(hit_count) as hits
                FROM cache GROUP BY category
            """).fetchall()
            
            # Expired entries
            expired = conn.execute(
                "SELECT COUNT(*) as count FROM cache WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            ).fetchone()['count']
            
            # Total hits
            total_hits = conn.execute(
                "SELECT SUM(hit_count) as hits FROM cache"
            ).fetchone()['hits'] or 0
            
            # Database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'total_entries': total,
                'expired_entries': expired,
                'total_hits': total_hits,
                'db_size_bytes': db_size,
                'db_size_mb': round(db_size / (1024 * 1024), 2),
                'categories': {
                    row['category']: {
                        'count': row['count'],
                        'hits': row['hits'] or 0
                    }
                    for row in categories
                }
            }
    
    def prune(self, keep_recent: int = None) -> int:
        """
        Prune cache to stay within limits.
        
        Args:
            keep_recent: Number of recent entries to keep (default: max_entries)
        
        Returns:
            Number of entries deleted
        """
        keep = keep_recent or self.max_entries
        
        # First clear expired
        deleted = self.clear_expired()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get count
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            
            if total > keep:
                # Delete oldest entries beyond limit
                cursor = conn.execute("""
                    DELETE FROM cache WHERE key IN (
                        SELECT key FROM cache 
                        ORDER BY created_at ASC 
                        LIMIT ?
                    )
                """, (total - keep,))
                deleted += cursor.rowcount
                conn.commit()
        
        return deleted


class MatchAnalysisCache(ModelCache):
    """
    Specialized cache for match analysis results.
    
    Provides convenience methods for football-specific caching.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def get_match_analysis(
        self,
        home_team: str,
        away_team: str,
        match_date: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached match analysis."""
        identifier = f"{home_team}_vs_{away_team}_{match_date}"
        return self.get(identifier, "match_analysis")
    
    def set_match_analysis(
        self,
        home_team: str,
        away_team: str,
        match_date: str,
        analysis: Dict[str, Any],
        ttl_hours: int = 12
    ):
        """Cache match analysis (shorter TTL for match-day data)."""
        identifier = f"{home_team}_vs_{away_team}_{match_date}"
        metadata = {
            'home_team': home_team,
            'away_team': away_team,
            'match_date': match_date
        }
        self.set(identifier, "match_analysis", analysis, ttl_hours, metadata)
    
    def get_team_features(
        self,
        team: str,
        league: str,
        season: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached team features."""
        identifier = f"{team}_{league}_{season}"
        return self.get(identifier, "team_features")
    
    def set_team_features(
        self,
        team: str,
        league: str,
        season: str,
        features: Dict[str, Any],
        ttl_hours: int = 72  # Longer TTL for season-level data
    ):
        """Cache team features."""
        identifier = f"{team}_{league}_{season}"
        self.set(identifier, "team_features", features, ttl_hours)
    
    def get_league_stats(
        self,
        league: str,
        season: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached league statistics."""
        identifier = f"{league}_{season}"
        return self.get(identifier, "league_stats")
    
    def set_league_stats(
        self,
        league: str,
        season: str,
        stats: Dict[str, Any],
        ttl_hours: int = 168  # 1 week
    ):
        """Cache league statistics."""
        identifier = f"{league}_{season}"
        self.set(identifier, "league_stats", stats, ttl_hours)
