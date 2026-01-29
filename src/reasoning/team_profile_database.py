"""
TEAM PROFILE DATABASE
=====================

Strukturierte Datenbank mit Spielstil, Transition-Metriken und
wiederkehrenden Mustern für alle Teams im System.

Basierend auf historischen Daten + Taktik-Analyse.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from enum import Enum


# Lokale Kopie von PlayingStyle (vermeidet circular import)
class PlayingStyle(Enum):
    """Spielstil-Kategorien"""
    HIGH_PRESSING = "high_pressing"
    POSSESSION_BASED = "possession_based"
    COUNTER_ATTACKING = "counter_attacking"
    TRANSITION_FAST = "transition_fast"
    BALANCED = "balanced"
    DEFENSIVE_COMPACT = "defensive_compact"


class TeamProfileDatabase:
    """
    Zentrale Datenbank für Team-Profile
    
    Speichert für jedes Team:
    - Spielstil-Kategorisierung
    - Transition-Metriken  
    - Zonen-Schwächen
    - Tor-Muster
    - Ligaspezifische Anpassungen
    """
    
    def __init__(self, db_path: str = "data/team_profiles.json"):
        self.db_path = Path(db_path)
        self.profiles = self._load_profiles()
        
    def _load_profiles(self) -> Dict:
        """Lade Team-Profile aus JSON"""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_profiles(self):
        """Speichere Team-Profile zu JSON"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.profiles, f, indent=2, ensure_ascii=False)
    
    def get_profile(self, team_name: str) -> Optional[Dict]:
        """Hole Team-Profil (None wenn unbekannt)"""
        return self.profiles.get(team_name)
    
    def infer_style_from_stats(
        self,
        possession_avg: float,
        ppda: float,  # Passes Per Defensive Action (Pressingintensität)
        counter_attack_pct: float,
        avg_goals_scored: float
    ) -> PlayingStyle:
        """
        Inferiere Spielstil aus Statistiken
        
        Args:
            possession_avg: Durchschnittlicher Ballbesitz (%)
            ppda: PPDA-Wert (niedriger = aggressiver Press)
            counter_attack_pct: % der Tore aus Kontern
            avg_goals_scored: Durchschnittliche Tore pro Spiel
        """
        # HIGH_PRESSING: Niedriger PPDA + hoher Ballbesitz
        if ppda < 10.0 and possession_avg > 55:
            return PlayingStyle.HIGH_PRESSING
        
        # COUNTER_ATTACKING: Niedriger Ballbesitz + hohe Konter-Quote
        if possession_avg < 45 and counter_attack_pct > 0.35:
            return PlayingStyle.COUNTER_ATTACKING
        
        # POSSESSION_BASED: Hoher Ballbesitz + geduldig (hoher PPDA)
        if possession_avg > 60 and ppda > 12.0:
            return PlayingStyle.POSSESSION_BASED
        
        # TRANSITION_FAST: Moderate Werte + hohe Konter-Quote
        if counter_attack_pct > 0.25 and avg_goals_scored > 1.5:
            return PlayingStyle.TRANSITION_FAST
        
        # DEFENSIVE_COMPACT: Niedriger Ballbesitz + wenige Gegentore
        if possession_avg < 40 and counter_attack_pct < 0.20:
            return PlayingStyle.DEFENSIVE_COMPACT
        
        # BALANCED: Fallback
        return PlayingStyle.BALANCED
    
    def create_profile_from_data(
        self,
        team_name: str,
        league: str,
        stats: Dict
    ) -> Dict:
        """
        Erstelle Team-Profil aus Statistiken
        
        Args:
            stats: Dictionary mit Statistiken (possession, ppda, shots, etc.)
        """
        # Inferiere Spielstil
        style = self.infer_style_from_stats(
            possession_avg=stats.get('possession_pct', 50.0),
            ppda=stats.get('ppda', 12.0),
            counter_attack_pct=stats.get('counter_pct', 0.20),
            avg_goals_scored=stats.get('avg_goals', 1.5)
        )
        
        # Transition Metriken (aus Stats oder Defaults)
        transition = {
            "time_to_shot_after_turnover": stats.get('transition_time', 10.0),
            "vertical_passes_per_transition": stats.get('vertical_passes', 3.0),
            "counter_attack_frequency": stats.get('counters_per_match', 4.0),
            "transition_xg": stats.get('transition_xg', 0.5)
        }
        
        # Zonen-Analyse (aus Stats oder Defaults)
        zones = {
            "left_wing_weakness": stats.get('left_wing_conceded_pct', 0.35),
            "right_wing_weakness": stats.get('right_wing_conceded_pct', 0.35),
            "central_penetration": stats.get('central_shots_pct', 0.45),
            "set_piece_strength": stats.get('set_piece_goals_pct', 0.25),
            "high_line_vulnerability": stats.get('counter_conceded_pct', 0.30)
        }
        
        # Tor-Muster
        goal_pattern = {
            "from_set_pieces": stats.get('goals_from_set_pieces', 0.25),
            "from_cutbacks": stats.get('goals_from_cutbacks', 0.20),
            "from_long_shots": stats.get('goals_from_long_shots', 0.15),
            "from_counters": stats.get('goals_from_counters', 0.25),
            "from_individual": stats.get('goals_from_individual', 0.15)
        }
        
        profile = {
            "team_name": team_name,
            "league": league,
            "playing_style": style.value,
            "transition_metrics": transition,
            "zone_analysis": zones,
            "goal_pattern": goal_pattern,
            "last_updated": "2026-01-29"
        }
        
        self.profiles[team_name] = profile
        self.save_profiles()
        
        return profile
    
    def initialize_top_teams(self):
        """
        Initialisiere Profile für Top-Teams (Bundesliga, Premier League, etc.)
        
        Basierend auf öffentlich verfügbaren Daten + Taktik-Wissen
        """
        top_teams_data = {
            # === BUNDESLIGA ===
            "Bayern München": {
                "league": "Bundesliga",
                "stats": {
                    "possession_pct": 65.0,
                    "ppda": 8.5,  # Sehr aggressiv
                    "counter_pct": 0.20,
                    "avg_goals": 2.8,
                    "transition_time": 8.0,
                    "vertical_passes": 4.2,
                    "counters_per_match": 5.5,
                    "transition_xg": 0.9,
                    "goals_from_set_pieces": 0.22,
                    "goals_from_counters": 0.18,
                    "goals_from_cutbacks": 0.30
                }
            },
            "Borussia Dortmund": {
                "league": "Bundesliga",
                "stats": {
                    "possession_pct": 58.0,
                    "ppda": 10.2,
                    "counter_pct": 0.35,  # Sehr konterorientiert
                    "avg_goals": 2.5,
                    "transition_time": 6.5,  # Sehr schnell
                    "vertical_passes": 4.8,
                    "counters_per_match": 6.8,
                    "transition_xg": 1.1,
                    "goals_from_counters": 0.40,
                    "goals_from_cutbacks": 0.25
                }
            },
            "RB Leipzig": {
                "league": "Bundesliga",
                "stats": {
                    "possession_pct": 56.0,
                    "ppda": 9.0,
                    "counter_pct": 0.38,
                    "avg_goals": 2.3,
                    "transition_time": 7.0,
                    "counters_per_match": 7.2,
                    "transition_xg": 1.0
                }
            },
            
            # === PREMIER LEAGUE ===
            "Manchester City": {
                "league": "Premier League",
                "stats": {
                    "possession_pct": 68.0,
                    "ppda": 7.8,  # Extrem aggressiv
                    "counter_pct": 0.15,
                    "avg_goals": 2.9,
                    "transition_time": 8.5,
                    "goals_from_cutbacks": 0.35,
                    "central_penetration": 0.55
                }
            },
            "Liverpool": {
                "league": "Premier League",
                "stats": {
                    "possession_pct": 62.0,
                    "ppda": 8.2,
                    "counter_pct": 0.28,
                    "avg_goals": 2.7,
                    "transition_time": 6.8,
                    "counters_per_match": 6.5,
                    "transition_xg": 1.0,
                    "goals_from_counters": 0.32
                }
            },
            
            # === LA LIGA ===
            "Real Madrid": {
                "league": "La Liga",
                "stats": {
                    "possession_pct": 60.0,
                    "ppda": 11.0,
                    "counter_pct": 0.30,
                    "avg_goals": 2.4,
                    "transition_time": 7.5,
                    "counters_per_match": 5.8,
                    "goals_from_counters": 0.35
                }
            },
            "Barcelona": {
                "league": "La Liga",
                "stats": {
                    "possession_pct": 70.0,
                    "ppda": 12.5,
                    "counter_pct": 0.12,
                    "avg_goals": 2.5,
                    "goals_from_cutbacks": 0.30,
                    "central_penetration": 0.50
                }
            },
            
            # === SERIE A ===
            "Inter Milan": {
                "league": "Serie A",
                "stats": {
                    "possession_pct": 55.0,
                    "ppda": 11.5,
                    "counter_pct": 0.32,
                    "avg_goals": 2.0,
                    "goals_from_counters": 0.35,
                    "high_line_vulnerability": 0.25
                }
            },
            "AC Milan": {
                "league": "Serie A",
                "stats": {
                    "possession_pct": 52.0,
                    "ppda": 12.0,
                    "counter_pct": 0.28,
                    "avg_goals": 1.9
                }
            },
            
            # === LIGUE 1 ===
            "Paris Saint-Germain": {
                "league": "Ligue 1",
                "stats": {
                    "possession_pct": 65.0,
                    "ppda": 9.5,
                    "counter_pct": 0.25,
                    "avg_goals": 2.6,
                    "goals_from_individual": 0.25,
                    "goals_from_cutbacks": 0.28
                }
            },
        }
        
        for team_name, data in top_teams_data.items():
            self.create_profile_from_data(team_name, data["league"], data["stats"])
        
        print(f"✅ Initialized {len(top_teams_data)} top team profiles")
        return len(top_teams_data)


if __name__ == "__main__":
    # Initialize database
    db = TeamProfileDatabase()
    
    # Create profiles for top teams
    count = db.initialize_top_teams()
    
    print(f"\n📊 Team Profile Database initialized with {count} teams")
    print(f"Database location: {db.db_path}")
    
    # Example: Get Bayern München profile
    bayern = db.get_profile("Bayern München")
    if bayern:
        print(f"\n🔵 Bayern München Profile:")
        print(f"  Playing Style: {bayern['playing_style']}")
        print(f"  Transition Time: {bayern['transition_metrics']['time_to_shot_after_turnover']:.1f}s")
        print(f"  Counter Attacks/Match: {bayern['transition_metrics']['counter_attack_frequency']:.1f}")
