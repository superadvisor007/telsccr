# 📊 Form & Momentum Analysis

## Für AI Model - Form-basierte Vorhersagen

---

## 1️⃣ FORM-INDIKATOREN

### Punkte-basierte Form (Standard)

```python
def calculate_form_points(last_5_results: list) -> dict:
    """
    Berechne Form basierend auf den letzten 5 Spielen
    
    results = ['W', 'W', 'D', 'L', 'W']
    
    Punkte: W=3, D=1, L=0
    Max: 15 Punkte
    """
    points = {'W': 3, 'D': 1, 'L': 0}
    total = sum(points.get(r, 0) for r in last_5_results)
    
    return {
        'form_points': total,
        'form_percentage': (total / 15) * 100,
        'form_rating': get_form_rating(total)
    }

def get_form_rating(points: int) -> str:
    if points >= 13:
        return 'EXCELLENT'  # 4-5 Siege
    elif points >= 10:
        return 'GOOD'       # 3 Siege + Remis
    elif points >= 7:
        return 'AVERAGE'    # 2 Siege oder viele Remis
    elif points >= 4:
        return 'POOR'       # 1 Sieg
    else:
        return 'TERRIBLE'   # Keine Siege

# Beispiele:
# WWWWW = 15 Punkte = EXCELLENT
# WWWDL = 10 Punkte = GOOD
# WDDLD = 6 Punkte = POOR
# LLLLL = 0 Punkte = TERRIBLE
```

### Gewichtete Form (Recent Games Matter More)

```python
def weighted_form(last_5_results: list) -> float:
    """
    Gewichtete Form - neuere Spiele zählen mehr
    
    Gewichte: [0.35, 0.25, 0.20, 0.12, 0.08] (neuestes zuerst)
    """
    weights = [0.35, 0.25, 0.20, 0.12, 0.08]
    points = {'W': 3, 'D': 1, 'L': 0}
    
    weighted_sum = sum(
        weights[i] * points.get(last_5_results[i], 0) 
        for i in range(min(5, len(last_5_results)))
    )
    
    # Normalisiert auf 0-100 Skala
    max_weighted = sum(w * 3 for w in weights)  # = 3.0
    return (weighted_sum / max_weighted) * 100

# Beispiele:
# WLWLW: Alte Methode = 9 Punkte, Gewichtet = ~58%
# WWWLL: Alte Methode = 9 Punkte, Gewichtet = ~72% (besser weil neuere Siege)
# LLWWW: Alte Methode = 9 Punkte, Gewichtet = ~44% (schlechter weil neuere Niederlagen)
```

### Goal-basierte Form

```python
def goal_form(last_5_matches: list) -> dict:
    """
    Form basierend auf geschossenen und kassierten Toren
    
    matches = [{'gf': 2, 'ga': 1}, {'gf': 3, 'ga': 0}, ...]
    gf = goals for, ga = goals against
    """
    total_gf = sum(m['gf'] for m in last_5_matches)
    total_ga = sum(m['ga'] for m in last_5_matches)
    
    return {
        'goals_scored': total_gf,
        'goals_conceded': total_ga,
        'goal_diff': total_gf - total_ga,
        'avg_goals_scored': total_gf / len(last_5_matches),
        'avg_goals_conceded': total_ga / len(last_5_matches),
        'attack_form': total_gf / 10 * 100,  # Max ~10 Tore = 100%
        'defense_form': max(0, (10 - total_ga) / 10 * 100)  # Weniger kassiert = besser
    }

# Für Over/Under Vorhersagen:
# Attack Form + Defense Form (gegnerisch) → Goal Expectation
```

---

## 2️⃣ HOME/AWAY FORM (KRITISCH!)

### Home Form Index

```python
def home_form(home_results: list) -> dict:
    """
    Form NUR für Heimspiele
    
    Viele Teams haben massive Heim/Auswärts Unterschiede!
    """
    points = {'W': 3, 'D': 1, 'L': 0}
    total = sum(points.get(r, 0) for r in home_results[:5])
    
    return {
        'home_form_points': total,
        'home_win_rate': home_results.count('W') / len(home_results),
        'home_form_rating': get_form_rating(total),
        'is_home_fortress': total >= 12  # 4+ Heimsiege
    }

# Teams mit starker Heim-Form:
# - Union Berlin (Bundesliga)
# - Athletic Bilbao (La Liga)
# - Napoli (Serie A)
```

### Away Form Index

```python
def away_form(away_results: list) -> dict:
    """
    Form NUR für Auswärtsspiele
    
    Auswärts-starke Teams sind selten aber wertvoll!
    """
    points = {'W': 3, 'D': 1, 'L': 0}
    total = sum(points.get(r, 0) for r in away_results[:5])
    
    return {
        'away_form_points': total,
        'away_win_rate': away_results.count('W') / len(away_results),
        'away_form_rating': get_form_rating(total),
        'is_road_warrior': total >= 10  # 3+ Auswärtssiege
    }

# Teams mit starker Auswärts-Form:
# - Manchester City
# - Bayern München
# - Real Madrid
```

### Home/Away Differential

```python
def home_away_differential(home_results: list, away_results: list) -> dict:
    """
    Berechne den Unterschied zwischen Heim- und Auswärtsform
    
    Großer Unterschied = wichtiger Faktor!
    """
    points = {'W': 3, 'D': 1, 'L': 0}
    
    home_points = sum(points.get(r, 0) for r in home_results[:5])
    away_points = sum(points.get(r, 0) for r in away_results[:5])
    
    diff = home_points - away_points
    
    return {
        'home_points': home_points,
        'away_points': away_points,
        'differential': diff,
        'home_advantage_factor': home_points / max(1, away_points),
        'is_home_dependent': diff >= 6,      # Stark Heim-abhängig
        'is_consistent_away': diff <= 2      # Konstant auch Auswärts
    }

# Interpretation:
# diff >= 8: Extrem Heim-abhängig (z.B. Union Berlin)
# diff >= 5: Deutlicher Heimvorteil
# diff 2-4: Normal
# diff <= 1: Konstant Heim/Auswärts (z.B. Manchester City)
```

---

## 3️⃣ MOMENTUM-INDIKATOREN

### Winning Streak

```python
def analyze_streak(results: list) -> dict:
    """
    Analysiere aktuelle Serien (Winning/Losing/Unbeaten)
    """
    current_streak = 0
    streak_type = results[0] if results else None
    
    for r in results:
        if r == streak_type:
            current_streak += 1
        else:
            break
    
    # Unbeaten Streak (W oder D)
    unbeaten = 0
    for r in results:
        if r in ['W', 'D']:
            unbeaten += 1
        else:
            break
    
    # Winless Streak (D oder L)
    winless = 0
    for r in results:
        if r in ['D', 'L']:
            winless += 1
        else:
            break
    
    return {
        'current_streak': current_streak,
        'streak_type': streak_type,
        'unbeaten_run': unbeaten,
        'winless_run': winless,
        'momentum_score': calculate_momentum(results)
    }

def calculate_momentum(results: list) -> float:
    """
    Momentum Score: -100 (sehr schlecht) bis +100 (sehr gut)
    """
    points = {'W': 10, 'D': 0, 'L': -10}
    weights = [0.35, 0.25, 0.20, 0.12, 0.08]
    
    score = sum(
        weights[i] * points.get(results[i], 0)
        for i in range(min(5, len(results)))
    )
    
    return score * 10  # Normalisiert

# Beispiele:
# WWWWW = +100 (Perfekter Momentum)
# WDWDW = +30 (Positiver Momentum)
# DDDDD = 0 (Neutral)
# LDLDL = -30 (Negativer Momentum)
# LLLLL = -100 (Krise)
```

### Trend Detection

```python
def detect_trend(results: list) -> dict:
    """
    Erkenne Formtrends (steigend, fallend, stabil)
    
    Vergleiche erste 3 vs letzte 3 Spiele
    """
    points = {'W': 3, 'D': 1, 'L': 0}
    
    # Letzte 3 (neueste)
    recent = sum(points.get(r, 0) for r in results[:3])
    
    # Vorherige 3
    older = sum(points.get(r, 0) for r in results[3:6]) if len(results) >= 6 else recent
    
    diff = recent - older
    
    trend = 'stable'
    if diff >= 4:
        trend = 'rising_strong'
    elif diff >= 2:
        trend = 'rising'
    elif diff <= -4:
        trend = 'falling_strong'
    elif diff <= -2:
        trend = 'falling'
    
    return {
        'recent_points': recent,
        'older_points': older,
        'trend': trend,
        'trend_value': diff,
        'confidence_adjustment': get_trend_adjustment(trend)
    }

def get_trend_adjustment(trend: str) -> float:
    """
    Passe Wahrscheinlichkeiten basierend auf Trend an
    """
    adjustments = {
        'rising_strong': 1.10,   # +10% Confidence
        'rising': 1.05,          # +5% Confidence
        'stable': 1.00,
        'falling': 0.95,         # -5% Confidence
        'falling_strong': 0.90   # -10% Confidence
    }
    return adjustments.get(trend, 1.0)
```

---

## 4️⃣ FORM vs SPECIFIC OPPONENT TYPE

### Form gegen Top/Mid/Bottom Teams

```python
def form_by_opponent_quality(matches: list) -> dict:
    """
    Analysiere Form gegen verschiedene Gegner-Klassen
    
    matches = [{'result': 'W', 'opponent_rank': 3}, ...]
    """
    top_6_results = [m['result'] for m in matches if m['opponent_rank'] <= 6]
    mid_table_results = [m['result'] for m in matches if 7 <= m['opponent_rank'] <= 14]
    bottom_results = [m['result'] for m in matches if m['opponent_rank'] >= 15]
    
    points = {'W': 3, 'D': 1, 'L': 0}
    
    return {
        'vs_top_6': {
            'matches': len(top_6_results),
            'points_per_game': sum(points.get(r, 0) for r in top_6_results) / max(1, len(top_6_results))
        },
        'vs_mid_table': {
            'matches': len(mid_table_results),
            'points_per_game': sum(points.get(r, 0) for r in mid_table_results) / max(1, len(mid_table_results))
        },
        'vs_bottom': {
            'matches': len(bottom_results),
            'points_per_game': sum(points.get(r, 0) for r in bottom_results) / max(1, len(bottom_results))
        }
    }

# Teams die gegen Kleine stark sind:
# Bayern, City, Real - ~2.8 PPG vs Bottom
#
# Teams die gegen Große überperformen:
# "Giant Killers" wie Brighton, Newcastle
```

---

## 5️⃣ GOAL SCORING PATTERNS

### Scoring Consistency

```python
def scoring_consistency(goals_per_match: list) -> dict:
    """
    Analysiere Tor-Konsistenz
    
    goals_per_match = [2, 0, 3, 1, 2, 0, 4, 1]
    """
    import numpy as np
    
    goals = np.array(goals_per_match)
    
    return {
        'avg_goals': np.mean(goals),
        'std_dev': np.std(goals),                    # Niedrig = Konsistent
        'scored_rate': np.sum(goals > 0) / len(goals),  # % der Spiele mit Tor
        'multi_goal_rate': np.sum(goals >= 2) / len(goals),  # % mit 2+ Toren
        'failed_to_score_rate': np.sum(goals == 0) / len(goals),
        'is_consistent_scorer': np.std(goals) < 1.2 and np.mean(goals) >= 1.5
    }

# Für Over 1.5 Vorhersagen:
# scored_rate > 80% UND avg_goals > 1.5 → Guter Indikator
```

### Clean Sheet Analysis

```python
def clean_sheet_analysis(goals_conceded: list) -> dict:
    """
    Analysiere defensive Konsistenz
    
    goals_conceded = [0, 2, 1, 0, 0, 3, 1, 0]
    """
    import numpy as np
    
    conceded = np.array(goals_conceded)
    
    return {
        'avg_conceded': np.mean(conceded),
        'clean_sheet_rate': np.sum(conceded == 0) / len(conceded),
        'multi_conceded_rate': np.sum(conceded >= 2) / len(conceded),
        'defense_rating': max(0, 100 - (np.mean(conceded) * 40)),
        'is_solid_defense': np.mean(conceded) < 1.2 and np.sum(conceded == 0) / len(conceded) > 0.30
    }

# Für BTTS No Vorhersagen:
# clean_sheet_rate > 35% → Guter Indikator
```

---

## 6️⃣ FORM IMPACT AUF MÄRKTE

### Form → Over/Under Correlation

```python
FORM_MARKET_CORRELATIONS = {
    'over_2_5': {
        'attack_form_weight': 0.40,
        'opponent_defense_form_weight': 0.30,
        'historical_scoring_weight': 0.20,
        'momentum_weight': 0.10
    },
    'btts_yes': {
        'attack_form_weight': 0.25,
        'defense_weakness_weight': 0.25,
        'opponent_attack_weight': 0.25,
        'opponent_defense_weakness_weight': 0.25
    },
    'home_win': {
        'home_form_weight': 0.35,
        'overall_form_weight': 0.25,
        'opponent_away_form_weight': 0.25,
        'head_to_head_weight': 0.15
    },
    'draw': {
        'form_similarity_weight': 0.30,
        'both_mid_form_weight': 0.30,
        'historical_draw_rate_weight': 0.40
    }
}

def predict_with_form(home_form: dict, away_form: dict, market: str) -> float:
    """
    Berechne Wahrscheinlichkeit unter Einbeziehung der Form
    """
    weights = FORM_MARKET_CORRELATIONS.get(market, {})
    
    # Beispiel für Over 2.5:
    if market == 'over_2_5':
        attack_score = home_form['attack_form'] * weights['attack_form_weight']
        defense_score = (100 - away_form['defense_form']) * weights['opponent_defense_form_weight']
        # ...weitere Faktoren
        
        base_prob = (attack_score + defense_score) / 100
        return min(0.90, max(0.20, base_prob))
    
    return 0.50  # Default
```

---

## 7️⃣ IMPLEMENTATION FÜR AI

```python
class FormAnalyzer:
    """Komplette Form-Analyse für das AI System"""
    
    def __init__(self, team_data: dict):
        self.results = team_data.get('results', [])
        self.home_results = team_data.get('home_results', [])
        self.away_results = team_data.get('away_results', [])
        self.goals_scored = team_data.get('goals_scored', [])
        self.goals_conceded = team_data.get('goals_conceded', [])
    
    def get_comprehensive_form(self) -> dict:
        """Liefere alle Form-Metriken"""
        return {
            'points_form': calculate_form_points(self.results[:5]),
            'weighted_form': weighted_form(self.results[:5]),
            'home_form': home_form(self.home_results[:5]),
            'away_form': away_form(self.away_results[:5]),
            'momentum': analyze_streak(self.results),
            'trend': detect_trend(self.results),
            'scoring': scoring_consistency(self.goals_scored),
            'defense': clean_sheet_analysis(self.goals_conceded)
        }
    
    def get_form_rating_for_market(self, market: str, is_home: bool = True) -> float:
        """
        Berechne Form-basierte Wahrscheinlichkeitsanpassung für einen Markt
        
        Returns: Multiplikator (0.8 bis 1.2)
        """
        form_data = self.get_comprehensive_form()
        
        base = 1.0
        
        # Trend-Anpassung
        base *= form_data['trend']['confidence_adjustment']
        
        # Heim/Auswärts-Anpassung
        if is_home:
            if form_data['home_form']['is_home_fortress']:
                base *= 1.10
        else:
            if form_data['away_form']['is_road_warrior']:
                base *= 1.08
        
        # Markt-spezifische Anpassungen
        if market in ['over_1_5', 'over_2_5']:
            if form_data['scoring']['is_consistent_scorer']:
                base *= 1.05
        elif market in ['btts_no', 'under_2_5']:
            if form_data['defense']['is_solid_defense']:
                base *= 1.05
        
        return min(1.20, max(0.80, base))

# Nutzung:
# analyzer = FormAnalyzer(team_data)
# form = analyzer.get_comprehensive_form()
# adjustment = analyzer.get_form_rating_for_market('over_2_5', is_home=True)
```

---

## 📊 Quick Reference: Form-Schwellenwerte

```
╔════════════════════════════════════════════════════════════════╗
║                    FORM THRESHOLDS                              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  EXCELLENT FORM:  13-15 Punkte (4-5 Siege aus 5)              ║
║  GOOD FORM:       10-12 Punkte (3 Siege + Remis)              ║
║  AVERAGE FORM:    7-9 Punkte                                   ║
║  POOR FORM:       4-6 Punkte                                   ║
║  TERRIBLE FORM:   0-3 Punkte                                   ║
║                                                                ║
║  HOME FORTRESS:   12+ Punkte aus Heimspielen                  ║
║  ROAD WARRIOR:    10+ Punkte aus Auswärtsspielen              ║
║                                                                ║
║  RISING TREND:    +4 Punkte (letzte 3 vs vorherige 3)         ║
║  FALLING TREND:   -4 Punkte                                   ║
║                                                                ║
║  CONSISTENT SCORER: Avg Goals >= 1.5, Std Dev < 1.2           ║
║  SOLID DEFENSE:    Avg Conceded < 1.2, Clean Sheets > 30%     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**📚 Dieses Knowledge File enthält alle Form- und Momentum-Analysen für präzise AI-Vorhersagen.**
