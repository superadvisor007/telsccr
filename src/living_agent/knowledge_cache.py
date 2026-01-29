"""
📚 KNOWLEDGE CACHE - Persistent Memory for Living Agent
=======================================================
Caches match analyses, reasoning chains, and historical insights.
Prevents redundant LLM calls and enables learning over time.

Features:
- SQLite for durability (survives restarts)
- Hash-based deduplication
- TTL-based expiration
- Memory retrieval for context

Battle-tested patterns from:
- LangChain memory systems
- RAG caching strategies
- Production AI systems
"""

import os
import json
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import pickle


@dataclass
class CachedAnalysis:
    """Cached match analysis with reasoning chain."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    match_date: str
    analysis_date: str
    reasoning_chain: Dict[str, Any]
    market_predictions: Dict[str, float]
    confidence_scores: Dict[str, float]
    scenarios: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    ttl_hours: int = 168  # 7 days default


@dataclass
class LeagueInsight:
    """Cached league-level insight."""
    league: str
    insight_type: str  # 'btts_tendency', 'over_1_5_rate', etc.
    value: float
    confidence: float
    sample_size: int
    last_updated: str
    

class KnowledgeCache:
    """
    🧠 Persistent Knowledge Cache for Living Agent
    
    Stores:
    - Match analyses (avoid redundant LLM calls)
    - League tendencies (soft priors)
    - Team patterns (historical behavior)
    - Reasoning chains (for learning)
    - Results & feedback (for improvement)
    
    Uses SQLite for persistence + in-memory index for speed.
    """
    
    def __init__(self, db_path: str = "data/knowledge_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._init_db()
        self._load_memory_index()
        
    def _init_db(self):
        """Initialize SQLite database with all required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Match analyses cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_analyses (
                match_id TEXT PRIMARY KEY,
                home_team TEXT,
                away_team TEXT,
                league TEXT,
                match_date TEXT,
                analysis_date TEXT,
                reasoning_chain BLOB,
                market_predictions BLOB,
                confidence_scores BLOB,
                scenarios BLOB,
                metadata BLOB,
                ttl_hours INTEGER DEFAULT 168,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # League insights (soft priors)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS league_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                league TEXT,
                insight_type TEXT,
                value REAL,
                confidence REAL,
                sample_size INTEGER,
                last_updated TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(league, insight_type)
            )
        """)
        
        # Team patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT,
                pattern_type TEXT,
                pattern_value BLOB,
                confidence REAL,
                sample_size INTEGER,
                last_updated TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team_name, pattern_type)
            )
        """)
        
        # Results tracking (for feedback loop)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT,
                market TEXT,
                predicted_prob REAL,
                predicted_confidence REAL,
                actual_outcome INTEGER,
                odds_at_prediction REAL,
                profit_loss REAL,
                reasoning_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Reasoning patterns (what works, what doesn't)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_description TEXT,
                success_rate REAL,
                sample_size INTEGER,
                last_updated TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern_type, pattern_description)
            )
        """)
        
        # Curiosity findings (hidden edges discovered)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS curiosity_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                finding_type TEXT,
                description TEXT,
                confidence REAL,
                validation_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_validated TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
    def _load_memory_index(self):
        """Load in-memory index for fast lookups."""
        self.memory_index = {
            'analyses': {},
            'league_insights': {},
            'team_patterns': {},
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load recent analyses
        cursor.execute("""
            SELECT match_id, home_team, away_team, league, match_date
            FROM match_analyses
            WHERE datetime(created_at) > datetime('now', '-7 days')
        """)
        for row in cursor.fetchall():
            self.memory_index['analyses'][row[0]] = {
                'home_team': row[1],
                'away_team': row[2],
                'league': row[3],
                'match_date': row[4]
            }
            
        # Load league insights
        cursor.execute("SELECT league, insight_type, value, confidence FROM league_insights")
        for row in cursor.fetchall():
            key = f"{row[0]}_{row[1]}"
            self.memory_index['league_insights'][key] = {
                'value': row[2],
                'confidence': row[3]
            }
            
        conn.close()
        
    def _generate_match_id(self, home_team: str, away_team: str, match_date: str) -> str:
        """Generate unique match ID from teams and date."""
        raw = f"{home_team}_{away_team}_{match_date}".lower().replace(" ", "_")
        return hashlib.md5(raw.encode()).hexdigest()[:16]
    
    # ==================== MATCH ANALYSIS CACHE ====================
    
    def get_cached_analysis(self, home_team: str, away_team: str, 
                            match_date: str) -> Optional[CachedAnalysis]:
        """
        Retrieve cached analysis if available and not expired.
        
        Returns None if:
        - No cached analysis exists
        - Cache has expired
        """
        match_id = self._generate_match_id(home_team, away_team, match_date)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM match_analyses
            WHERE match_id = ? 
            AND datetime(created_at, '+' || ttl_hours || ' hours') > datetime('now')
        """, (match_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return CachedAnalysis(
                match_id=row[0],
                home_team=row[1],
                away_team=row[2],
                league=row[3],
                match_date=row[4],
                analysis_date=row[5],
                reasoning_chain=pickle.loads(row[6]),
                market_predictions=pickle.loads(row[7]),
                confidence_scores=pickle.loads(row[8]),
                scenarios=pickle.loads(row[9]),
                metadata=pickle.loads(row[10]),
                ttl_hours=row[11]
            )
        return None
    
    def cache_analysis(self, analysis: CachedAnalysis) -> bool:
        """Store analysis in cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO match_analyses
                (match_id, home_team, away_team, league, match_date, analysis_date,
                 reasoning_chain, market_predictions, confidence_scores, scenarios,
                 metadata, ttl_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.match_id,
                analysis.home_team,
                analysis.away_team,
                analysis.league,
                analysis.match_date,
                analysis.analysis_date,
                pickle.dumps(analysis.reasoning_chain),
                pickle.dumps(analysis.market_predictions),
                pickle.dumps(analysis.confidence_scores),
                pickle.dumps(analysis.scenarios),
                pickle.dumps(analysis.metadata),
                analysis.ttl_hours
            ))
            conn.commit()
            
            # Update memory index
            self.memory_index['analyses'][analysis.match_id] = {
                'home_team': analysis.home_team,
                'away_team': analysis.away_team,
                'league': analysis.league,
                'match_date': analysis.match_date
            }
            
            return True
        except Exception as e:
            print(f"Cache error: {e}")
            return False
        finally:
            conn.close()
    
    # ==================== LEAGUE INSIGHTS ====================
    
    def get_league_insight(self, league: str, insight_type: str) -> Optional[LeagueInsight]:
        """Get cached league insight (soft prior)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT league, insight_type, value, confidence, sample_size, last_updated
            FROM league_insights
            WHERE league = ? AND insight_type = ?
        """, (league, insight_type))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return LeagueInsight(
                league=row[0],
                insight_type=row[1],
                value=row[2],
                confidence=row[3],
                sample_size=row[4],
                last_updated=row[5]
            )
        return None
    
    def update_league_insight(self, insight: LeagueInsight) -> bool:
        """Update or create league insight."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO league_insights
                (league, insight_type, value, confidence, sample_size, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                insight.league,
                insight.insight_type,
                insight.value,
                insight.confidence,
                insight.sample_size,
                insight.last_updated
            ))
            conn.commit()
            
            # Update memory index
            key = f"{insight.league}_{insight.insight_type}"
            self.memory_index['league_insights'][key] = {
                'value': insight.value,
                'confidence': insight.confidence
            }
            
            return True
        except Exception as e:
            print(f"Insight update error: {e}")
            return False
        finally:
            conn.close()
    
    def get_all_league_insights(self, league: str) -> Dict[str, float]:
        """Get all insights for a league as dict."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT insight_type, value FROM league_insights WHERE league = ?
        """, (league,))
        
        insights = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return insights
    
    # ==================== PREDICTION RESULTS ====================
    
    def record_prediction_result(self, match_id: str, market: str,
                                  predicted_prob: float, predicted_confidence: float,
                                  actual_outcome: int, odds: float,
                                  reasoning_summary: str) -> bool:
        """Record prediction result for feedback loop."""
        profit_loss = (odds - 1) if actual_outcome == 1 else -1
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO prediction_results
                (match_id, market, predicted_prob, predicted_confidence,
                 actual_outcome, odds_at_prediction, profit_loss, reasoning_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id, market, predicted_prob, predicted_confidence,
                actual_outcome, odds, profit_loss, reasoning_summary
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Result recording error: {e}")
            return False
        finally:
            conn.close()
    
    def get_performance_stats(self, market: str = None, 
                               days: int = 30) -> Dict[str, Any]:
        """Get performance statistics for feedback analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        where_clause = "WHERE datetime(created_at) > datetime('now', ? || ' days')"
        params = [f"-{days}"]
        
        if market:
            where_clause += " AND market = ?"
            params.append(market)
        
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN actual_outcome = 1 THEN 1 ELSE 0 END) as wins,
                SUM(profit_loss) as total_profit,
                AVG(predicted_prob) as avg_predicted_prob,
                AVG(predicted_confidence) as avg_confidence
            FROM prediction_results
            {where_clause}
        """, params)
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0] > 0:
            return {
                'total_bets': row[0],
                'wins': row[1],
                'win_rate': row[1] / row[0],
                'total_profit': row[2],
                'avg_predicted_prob': row[3],
                'avg_confidence': row[4],
                'roi': row[2] / row[0] if row[0] > 0 else 0
            }
        return {'total_bets': 0, 'wins': 0, 'win_rate': 0, 'total_profit': 0}
    
    # ==================== CURIOSITY FINDINGS ====================
    
    def add_curiosity_finding(self, finding_type: str, description: str, 
                               confidence: float) -> bool:
        """Store a new insight discovered through curiosity prompts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO curiosity_findings
                (finding_type, description, confidence)
                VALUES (?, ?, ?)
            """, (finding_type, description, confidence))
            conn.commit()
            return True
        except Exception as e:
            print(f"Finding storage error: {e}")
            return False
        finally:
            conn.close()
    
    def get_relevant_findings(self, finding_type: str = None, 
                               min_confidence: float = 0.6) -> List[Dict]:
        """Get validated curiosity findings for reasoning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT finding_type, description, confidence, validation_count, success_count
            FROM curiosity_findings
            WHERE confidence >= ?
        """
        params = [min_confidence]
        
        if finding_type:
            query += " AND finding_type = ?"
            params.append(finding_type)
        
        cursor.execute(query + " ORDER BY confidence DESC LIMIT 20", params)
        
        findings = []
        for row in cursor.fetchall():
            findings.append({
                'type': row[0],
                'description': row[1],
                'confidence': row[2],
                'validations': row[3],
                'successes': row[4]
            })
        
        conn.close()
        return findings
    
    # ==================== CLEANUP ====================
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries. Returns count of removed entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM match_analyses
            WHERE datetime(created_at, '+' || ttl_hours || ' hours') < datetime('now')
        """)
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        # Refresh memory index
        self._load_memory_index()
        
        return deleted
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        for table in ['match_analyses', 'league_insights', 'team_patterns', 
                      'prediction_results', 'curiosity_findings']:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        
        conn.close()
        return stats


# ==================== DEFAULT LEAGUE PRIORS ====================

DEFAULT_LEAGUE_INSIGHTS = {
    'Bundesliga': {
        'avg_goals': 3.15,
        'over_1_5_rate': 0.78,
        'over_2_5_rate': 0.55,
        'btts_rate': 0.53,
        'home_win_rate': 0.43,
        'away_win_rate': 0.32,
    },
    'Premier League': {
        'avg_goals': 2.85,
        'over_1_5_rate': 0.75,
        'over_2_5_rate': 0.50,
        'btts_rate': 0.51,
        'home_win_rate': 0.41,
        'away_win_rate': 0.31,
    },
    'La Liga': {
        'avg_goals': 2.65,
        'over_1_5_rate': 0.72,
        'over_2_5_rate': 0.46,
        'btts_rate': 0.46,
        'home_win_rate': 0.45,
        'away_win_rate': 0.28,
    },
    'Serie A': {
        'avg_goals': 2.75,
        'over_1_5_rate': 0.74,
        'over_2_5_rate': 0.48,
        'btts_rate': 0.50,
        'home_win_rate': 0.42,
        'away_win_rate': 0.30,
    },
    'Ligue 1': {
        'avg_goals': 2.80,
        'over_1_5_rate': 0.73,
        'over_2_5_rate': 0.48,
        'btts_rate': 0.48,
        'home_win_rate': 0.44,
        'away_win_rate': 0.29,
    },
    'Eredivisie': {
        'avg_goals': 3.35,
        'over_1_5_rate': 0.82,
        'over_2_5_rate': 0.58,
        'btts_rate': 0.56,
        'home_win_rate': 0.45,
        'away_win_rate': 0.30,
    },
    'Championship': {
        'avg_goals': 2.90,
        'over_1_5_rate': 0.76,
        'over_2_5_rate': 0.50,
        'btts_rate': 0.52,
        'home_win_rate': 0.40,
        'away_win_rate': 0.32,
    },
}


def initialize_default_insights(cache: KnowledgeCache):
    """Initialize cache with default league insights."""
    now = datetime.now().isoformat()
    
    for league, insights in DEFAULT_LEAGUE_INSIGHTS.items():
        for insight_type, value in insights.items():
            insight = LeagueInsight(
                league=league,
                insight_type=insight_type,
                value=value,
                confidence=0.8,  # Historical data confidence
                sample_size=1000,  # Approximation
                last_updated=now
            )
            cache.update_league_insight(insight)
    
    print(f"✅ Initialized {len(DEFAULT_LEAGUE_INSIGHTS)} leagues with default insights")
