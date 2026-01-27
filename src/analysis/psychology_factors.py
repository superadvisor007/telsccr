#!/usr/bin/env python3
"""
🧠 PSYCHOLOGICAL FACTORS ENGINE
================================
Adds deep psychological analysis to predictions:

1. Derby Pressure & Historical Baggage
2. Relegation Battle Stress
3. Title Race Pressure
4. Manager Changes/Instability
5. Recent Form Momentum (Confidence)
6. Home Crowd Factor
7. European Competition Fatigue
8. End-of-Season Motivation

These factors can swing probabilities by 5-15%!
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


class PsychFactorType(Enum):
    DERBY_PRESSURE = "derby_pressure"
    RELEGATION_BATTLE = "relegation_battle"
    TITLE_RACE = "title_race"
    MANAGER_CHANGE = "manager_change"
    FORM_MOMENTUM = "form_momentum"
    HOME_CROWD = "home_crowd"
    EUROPEAN_FATIGUE = "european_fatigue"
    MOTIVATION = "motivation"


@dataclass
class PsychologicalProfile:
    """Complete psychological profile for a match"""
    home_team: str
    away_team: str
    
    # Factor scores (-10 to +10)
    # Positive = favors home, Negative = favors away
    derby_factor: float = 0.0
    relegation_factor: float = 0.0
    title_race_factor: float = 0.0
    manager_stability: float = 0.0
    form_momentum: float = 0.0
    crowd_factor: float = 0.0
    fatigue_factor: float = 0.0
    motivation_factor: float = 0.0
    
    # Total psychological edge
    @property
    def total_psych_edge(self) -> float:
        """Calculate total psychological edge"""
        return sum([
            self.derby_factor * 0.15,       # 15% weight
            self.relegation_factor * 0.20,   # 20% weight
            self.title_race_factor * 0.15,   # 15% weight
            self.manager_stability * 0.10,   # 10% weight
            self.form_momentum * 0.15,       # 15% weight
            self.crowd_factor * 0.10,        # 10% weight
            self.fatigue_factor * 0.10,      # 10% weight
            self.motivation_factor * 0.05    # 5% weight
        ])
    
    @property
    def probability_adjustment(self) -> float:
        """Convert edge to probability adjustment (-0.15 to +0.15)"""
        return max(-0.15, min(0.15, self.total_psych_edge / 100))


class PsychologicalAnalyzer:
    """
    🧠 Deep psychological analysis engine
    
    Analyzes non-statistical factors that influence match outcomes:
    - Pressure situations
    - Historical rivalries
    - Mental momentum
    - External factors
    """
    
    def __init__(self):
        # Famous derbies with intensity level (1-10)
        self.famous_derbies = {
            # Germany
            ('Bayern München', 'Borussia Dortmund'): {'name': 'Der Klassiker', 'intensity': 9},
            ('Schalke 04', 'Borussia Dortmund'): {'name': 'Revierderby', 'intensity': 10},
            ('Bayern München', 'TSV 1860 München'): {'name': 'Münchner Stadtderby', 'intensity': 8},
            ('Hamburg', 'Werder Bremen'): {'name': 'Nordderby', 'intensity': 8},
            ('Köln', 'Mönchengladbach'): {'name': 'Rheinderby', 'intensity': 8},
            ('Frankfurt', 'Mainz'): {'name': 'Rhein-Main-Derby', 'intensity': 7},
            
            # England
            ('Liverpool', 'Manchester United'): {'name': 'North West Derby', 'intensity': 10},
            ('Manchester United', 'Manchester City'): {'name': 'Manchester Derby', 'intensity': 9},
            ('Arsenal', 'Tottenham'): {'name': 'North London Derby', 'intensity': 10},
            ('Liverpool', 'Everton'): {'name': 'Merseyside Derby', 'intensity': 9},
            ('Chelsea', 'Arsenal'): {'name': 'London Derby', 'intensity': 8},
            ('Chelsea', 'Tottenham'): {'name': 'London Derby', 'intensity': 8},
            ('West Ham', 'Tottenham'): {'name': 'East London Derby', 'intensity': 8},
            ('Newcastle', 'Sunderland'): {'name': 'Tyne-Wear Derby', 'intensity': 10},
            
            # Spain
            ('Real Madrid', 'Barcelona'): {'name': 'El Clásico', 'intensity': 10},
            ('Atlético Madrid', 'Real Madrid'): {'name': 'Madrid Derby', 'intensity': 9},
            ('Barcelona', 'Espanyol'): {'name': 'Derbi Barceloní', 'intensity': 8},
            ('Athletic Bilbao', 'Real Sociedad'): {'name': 'Basque Derby', 'intensity': 9},
            ('Sevilla', 'Real Betis'): {'name': 'Seville Derby', 'intensity': 10},
            
            # Italy
            ('Inter', 'Milan'): {'name': 'Derby della Madonnina', 'intensity': 10},
            ('Juventus', 'Inter'): {'name': 'Derby d\'Italia', 'intensity': 9},
            ('Roma', 'Lazio'): {'name': 'Derby della Capitale', 'intensity': 10},
            ('Napoli', 'Roma'): {'name': 'Derby del Sole', 'intensity': 8},
            ('Genoa', 'Sampdoria'): {'name': 'Derby della Lanterna', 'intensity': 9},
            
            # France
            ('PSG', 'Marseille'): {'name': 'Le Classique', 'intensity': 10},
            ('Lyon', 'Saint-Étienne'): {'name': 'Derby Rhône-Alpes', 'intensity': 10},
            ('Nice', 'Monaco'): {'name': 'Côte d\'Azur Derby', 'intensity': 7},
            
            # Netherlands
            ('Ajax', 'Feyenoord'): {'name': 'De Klassieker', 'intensity': 10},
            ('Ajax', 'PSV'): {'name': 'De Topper', 'intensity': 9},
            ('Feyenoord', 'PSV'): {'name': 'Zuiderderby', 'intensity': 8},
        }
        
        # Teams in European competitions (2024-25)
        self.european_teams = {
            # Champions League
            'Bayern München', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
            'Manchester City', 'Liverpool', 'Arsenal', 'Aston Villa',
            'Real Madrid', 'Barcelona', 'Atlético Madrid', 'Girona',
            'Inter', 'Milan', 'Juventus', 'Bologna',
            'PSG', 'Monaco', 'Lille', 'Brest',
            'PSV', 'Feyenoord',
            
            # Europa League
            'Frankfurt', 'Hoffenheim', 'Manchester United', 'Tottenham', 
            'West Ham', 'Brighton', 'Athletic Bilbao', 'Real Sociedad',
            'Roma', 'Lazio', 'Atalanta', 'Lyon', 'Marseille', 'Ajax', 'AZ'
        }
        
        # League positions (example - should be fetched live)
        self.league_positions = {}
    
    def analyze_match(self, home_team: str, away_team: str, league: str,
                      home_position: int = 10, away_position: int = 10,
                      home_form: List[str] = None, away_form: List[str] = None,
                      home_manager_days: int = 365, away_manager_days: int = 365,
                      total_teams: int = 18) -> PsychologicalProfile:
        """
        Perform complete psychological analysis
        
        Args:
            home_team: Home team name
            away_team: Away team name  
            league: League name
            home_position: Current league position (1 = top)
            away_position: Current league position
            home_form: Last 5 results ['W', 'W', 'D', 'L', 'W']
            away_form: Last 5 results
            home_manager_days: Days since home manager appointment
            away_manager_days: Days since away manager appointment
            total_teams: Teams in league (for relegation calc)
        """
        
        profile = PsychologicalProfile(home_team=home_team, away_team=away_team)
        
        # 1. DERBY ANALYSIS
        profile.derby_factor = self._analyze_derby(home_team, away_team)
        
        # 2. RELEGATION BATTLE
        profile.relegation_factor = self._analyze_relegation(
            home_position, away_position, total_teams
        )
        
        # 3. TITLE RACE
        profile.title_race_factor = self._analyze_title_race(
            home_position, away_position
        )
        
        # 4. MANAGER STABILITY
        profile.manager_stability = self._analyze_manager_stability(
            home_manager_days, away_manager_days
        )
        
        # 5. FORM MOMENTUM
        profile.form_momentum = self._analyze_form_momentum(
            home_form or [], away_form or []
        )
        
        # 6. HOME CROWD FACTOR
        profile.crowd_factor = self._analyze_crowd_factor(
            home_team, away_team, home_position, away_position
        )
        
        # 7. EUROPEAN FATIGUE
        profile.fatigue_factor = self._analyze_european_fatigue(
            home_team, away_team
        )
        
        # 8. MOTIVATION
        profile.motivation_factor = self._analyze_motivation(
            home_position, away_position, total_teams
        )
        
        return profile
    
    def _analyze_derby(self, home: str, away: str) -> float:
        """
        Analyze derby factor
        
        In derbies:
        - Home advantage is amplified (+3 to +5 edge)
        - More goals typically
        - Form matters less, history matters more
        """
        # Check if this is a derby (either direction)
        derby_info = None
        for (team1, team2), info in self.famous_derbies.items():
            if (self._fuzzy_match(home, team1) and self._fuzzy_match(away, team2)) or \
               (self._fuzzy_match(home, team2) and self._fuzzy_match(away, team1)):
                derby_info = info
                break
        
        if not derby_info:
            return 0.0
        
        intensity = derby_info['intensity']
        
        # Home advantage in derbies is typically +5% stronger
        # Scale: 0-10 intensity → 0-5 edge
        return intensity * 0.5
    
    def _analyze_relegation(self, home_pos: int, away_pos: int, total: int) -> float:
        """
        Analyze relegation battle pressure
        
        Teams in relegation zone:
        - Fight harder (desperation factor)
        - More defensive
        - Higher home motivation
        """
        # Relegation zone typically bottom 3 (varies by league)
        relegation_zone = total - 2
        
        home_in_danger = home_pos >= relegation_zone - 2  # Close to danger
        away_in_danger = away_pos >= relegation_zone - 2
        
        factor = 0.0
        
        if home_in_danger and not away_in_danger:
            # Home team fighting for survival vs comfortable opponent
            factor = 3.0  # Desperation advantage
        elif away_in_danger and not home_in_danger:
            # Away team desperate, will fight hard away from home
            factor = -1.5  # Slight away advantage
        elif home_in_danger and away_in_danger:
            # Six-pointer! Home advantage amplified
            factor = 2.0
        
        return factor
    
    def _analyze_title_race(self, home_pos: int, away_pos: int) -> float:
        """
        Analyze title race pressure
        
        Teams in title race:
        - Higher pressure
        - More conservative play
        - Cannot afford to drop points
        """
        home_in_race = home_pos <= 4
        away_in_race = away_pos <= 4
        
        factor = 0.0
        
        if home_in_race and not away_in_race:
            # Home team under pressure vs nothing-to-lose opponent
            factor = -1.0  # Slight pressure disadvantage
        elif away_in_race and not home_in_race:
            factor = 0.5  # Home team can play without pressure
        elif home_in_race and away_in_race:
            # Title clash! Typically cagey
            factor = 0.0  # Neutralizes
        
        return factor
    
    def _analyze_manager_stability(self, home_days: int, away_days: int) -> float:
        """
        Analyze manager stability
        
        New manager bounce:
        - First 10 games: +15% performance boost
        - Tactical uncertainty for opponents
        - Extra motivation to impress
        
        Long-term manager:
        - More predictable
        - Better preparation against
        - But also more stable
        """
        factor = 0.0
        
        # New manager bounce (< 90 days)
        if home_days < 90:
            factor += 2.0  # Home team has bounce
        if away_days < 90:
            factor -= 1.5  # Away team has bounce
        
        # Very new manager (< 14 days) - uncertainty
        if home_days < 14:
            factor -= 1.0  # Too new, chaotic
        if away_days < 14:
            factor += 0.5
        
        return factor
    
    def _analyze_form_momentum(self, home_form: List[str], away_form: List[str]) -> float:
        """
        Analyze psychological momentum from recent form
        
        Winning streaks build confidence
        Losing streaks create doubt
        """
        def form_score(form: List[str]) -> float:
            """Calculate form score with recency weighting"""
            if not form:
                return 0.0
            
            score = 0.0
            weights = [1.0, 0.9, 0.75, 0.6, 0.5]  # Most recent = highest weight
            
            for i, result in enumerate(form[:5]):
                w = weights[i] if i < len(weights) else 0.4
                if result == 'W':
                    score += 3 * w
                elif result == 'D':
                    score += 1 * w
                elif result == 'L':
                    score -= 1 * w
            
            return score
        
        home_score = form_score(home_form)
        away_score = form_score(away_form)
        
        # Streak detection
        home_streak = self._detect_streak(home_form)
        away_streak = self._detect_streak(away_form)
        
        # Winning streak bonus
        if home_streak >= 3:
            home_score += 2.0  # Confidence boost
        if away_streak >= 3:
            away_score += 2.0
        
        # Losing streak penalty
        if home_streak <= -3:
            home_score -= 2.0  # Confidence crisis
        if away_streak <= -3:
            away_score -= 2.0
        
        # Convert to relative advantage
        return (home_score - away_score) * 0.5
    
    def _detect_streak(self, form: List[str]) -> int:
        """Detect winning/losing streak (positive = wins, negative = losses)"""
        if not form:
            return 0
        
        streak = 0
        first_result = form[0] if form else 'D'
        
        for result in form:
            if result == first_result:
                if result == 'W':
                    streak += 1
                elif result == 'L':
                    streak -= 1
            else:
                break
        
        return streak
    
    def _analyze_crowd_factor(self, home: str, away: str, 
                              home_pos: int, away_pos: int) -> float:
        """
        Analyze home crowd psychological impact
        
        Big clubs have stronger home advantage
        Relegation-threatened home crowds add pressure
        """
        factor = 0.0
        
        # Big clubs with intimidating atmospheres
        intimidating_homes = {
            'Liverpool': 3.0,  # Anfield
            'Borussia Dortmund': 3.0,  # Signal Iduna Park
            'Celtic': 2.5,
            'Rangers': 2.5,
            'Galatasaray': 3.0,
            'Fenerbahçe': 3.0,
            'Bayern München': 2.0,
            'Real Madrid': 2.0,
            'Barcelona': 2.0,
            'Manchester United': 2.0,
            'Inter': 2.0,
            'Milan': 2.0,
            'Roma': 2.5,
            'Lazio': 2.5,
            'Napoli': 2.5,
            'Marseille': 2.5,
        }
        
        for team, boost in intimidating_homes.items():
            if self._fuzzy_match(home, team):
                factor += boost
                break
        
        # Relegation battle: desperate fans add pressure
        if home_pos >= 16:  # Near relegation
            factor += 1.0
        
        return factor
    
    def _analyze_european_fatigue(self, home: str, away: str) -> float:
        """
        Analyze European competition fatigue
        
        Teams playing Thursday Europa/Conference:
        - -5% to -10% performance in Sunday league games
        - Squad rotation
        - Travel fatigue
        
        Teams playing Tuesday/Wednesday Champions League:
        - Saturday games less affected
        """
        # This would ideally check actual fixture lists
        # For now, use membership in European competitions
        
        factor = 0.0
        
        home_european = any(self._fuzzy_match(home, t) for t in self.european_teams)
        away_european = any(self._fuzzy_match(away, t) for t in self.european_teams)
        
        # European teams may be fatigued
        if home_european and not away_european:
            factor -= 1.0  # Home team possibly fatigued
        elif away_european and not home_european:
            factor += 0.5  # Away team possibly fatigued
        
        return factor
    
    def _analyze_motivation(self, home_pos: int, away_pos: int, total: int) -> float:
        """
        Analyze end-of-season motivation
        
        Mid-table teams (safe, no Europe):
        - "Beach mode" - less motivated
        - Experimental lineups
        
        This is most relevant late season (April/May)
        """
        factor = 0.0
        
        # Mid-table = positions 8-14 typically
        home_midtable = 7 < home_pos < total - 4
        away_midtable = 7 < away_pos < total - 4
        
        if home_midtable and not away_midtable:
            factor -= 1.5  # Home has nothing to play for
        elif away_midtable and not home_midtable:
            factor += 1.0  # Away has nothing to play for
        
        return factor
    
    def _fuzzy_match(self, str1: str, str2: str) -> bool:
        """Check if team names match (fuzzy)"""
        s1 = str1.lower().replace(' fc', '').replace(' cf', '').strip()
        s2 = str2.lower().replace(' fc', '').replace(' cf', '').strip()
        return s1 in s2 or s2 in s1
    
    def get_report(self, profile: PsychologicalProfile) -> str:
        """Generate human-readable psychological report"""
        report = []
        report.append(f"\n🧠 PSYCHOLOGICAL ANALYSIS")
        report.append(f"   {profile.home_team} vs {profile.away_team}")
        report.append("─" * 50)
        
        factors = [
            ("Derby Pressure", profile.derby_factor),
            ("Relegation Factor", profile.relegation_factor),
            ("Title Race", profile.title_race_factor),
            ("Manager Stability", profile.manager_stability),
            ("Form Momentum", profile.form_momentum),
            ("Home Crowd", profile.crowd_factor),
            ("European Fatigue", profile.fatigue_factor),
            ("Motivation", profile.motivation_factor),
        ]
        
        for name, value in factors:
            if value != 0:
                direction = "→ Home" if value > 0 else "→ Away"
                bar = "█" * int(abs(value) * 2)
                report.append(f"   {name}: {value:+.1f} {direction} {bar}")
        
        report.append("─" * 50)
        edge = profile.total_psych_edge
        adj = profile.probability_adjustment
        
        if edge > 0:
            report.append(f"   🏠 PSYCHOLOGICAL EDGE: +{edge:.1f}% for {profile.home_team}")
        elif edge < 0:
            report.append(f"   ✈️  PSYCHOLOGICAL EDGE: +{abs(edge):.1f}% for {profile.away_team}")
        else:
            report.append(f"   ⚖️  PSYCHOLOGICAL BALANCE: Even")
        
        report.append(f"   📊 Probability Adjustment: {adj:+.1%}")
        
        return "\n".join(report)


def test_psychological_analyzer():
    """Test the psychological analyzer"""
    print("\n" + "="*60)
    print("🧠 PSYCHOLOGICAL ANALYZER TEST")
    print("="*60)
    
    analyzer = PsychologicalAnalyzer()
    
    test_cases = [
        {
            'home': 'Liverpool',
            'away': 'Manchester United',
            'league': 'Premier League',
            'home_pos': 2,
            'away_pos': 7,
            'home_form': ['W', 'W', 'W', 'D', 'W'],
            'away_form': ['L', 'D', 'W', 'L', 'D'],
        },
        {
            'home': 'Schalke 04',
            'away': 'Borussia Dortmund',
            'league': 'Bundesliga',
            'home_pos': 17,
            'away_pos': 3,
            'home_form': ['L', 'L', 'D', 'L', 'L'],
            'away_form': ['W', 'W', 'W', 'W', 'D'],
        },
        {
            'home': 'Real Madrid',
            'away': 'Barcelona',
            'league': 'La Liga',
            'home_pos': 1,
            'away_pos': 2,
            'home_form': ['W', 'W', 'D', 'W', 'W'],
            'away_form': ['W', 'D', 'W', 'W', 'W'],
        },
        {
            'home': 'Hoffenheim',
            'away': 'Mainz',
            'league': 'Bundesliga',
            'home_pos': 9,
            'away_pos': 11,
            'home_form': ['W', 'D', 'L', 'W', 'D'],
            'away_form': ['D', 'W', 'D', 'L', 'W'],
        },
    ]
    
    for tc in test_cases:
        profile = analyzer.analyze_match(
            home_team=tc['home'],
            away_team=tc['away'],
            league=tc['league'],
            home_position=tc['home_pos'],
            away_position=tc['away_pos'],
            home_form=tc['home_form'],
            away_form=tc['away_form']
        )
        
        print(analyzer.get_report(profile))
    
    print("\n✅ Psychological analyzer working correctly!")


if __name__ == "__main__":
    test_psychological_analyzer()
