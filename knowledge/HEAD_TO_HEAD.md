# 🏆 Head-to-Head (H2H) Analysis

## Für AI Model - Historische Duelle auswerten

---

## 1️⃣ H2H GRUNDLAGEN

### Warum H2H wichtig ist

```yaml
key_insights:
  - Manche Teams haben "bogey teams" (psychologische Barriere)
  - Derby-Spiele folgen oft eigenen Mustern
  - Taktische Match-ups können sich über Jahre halten
  - Trainer-Duelle haben eigene Dynamik

caution:
  - H2H verliert an Bedeutung wenn:
    - Teams stark verändert (neue Spieler/Trainer)
    - Letzte Duelle >2 Jahre her
    - Liga-Wechsel (Auf-/Abstieg)
```

### H2H Data Collection

```python
def collect_h2h_data(team_a: str, team_b: str, matches: list) -> dict:
    """
    Sammle alle relevanten H2H-Daten
    
    matches = [
        {'date': '2024-02-15', 'home': 'Bayern', 'away': 'Dortmund',
         'home_goals': 3, 'away_goals': 1, 'venue': 'home'},
        ...
    ]
    """
    h2h = {
        'total_matches': len(matches),
        'team_a_wins': 0,
        'team_b_wins': 0,
        'draws': 0,
        'team_a_goals': 0,
        'team_b_goals': 0,
        'team_a_home_wins': 0,
        'team_a_away_wins': 0,
        'recent_matches': matches[:5],  # Letzte 5
        'btts_count': 0,
        'over_2_5_count': 0
    }
    
    for m in matches:
        if m['home'] == team_a:
            h2h['team_a_goals'] += m['home_goals']
            h2h['team_b_goals'] += m['away_goals']
            
            if m['home_goals'] > m['away_goals']:
                h2h['team_a_wins'] += 1
                h2h['team_a_home_wins'] += 1
            elif m['home_goals'] < m['away_goals']:
                h2h['team_b_wins'] += 1
            else:
                h2h['draws'] += 1
        else:  # team_b war Heim
            h2h['team_a_goals'] += m['away_goals']
            h2h['team_b_goals'] += m['home_goals']
            
            if m['away_goals'] > m['home_goals']:
                h2h['team_a_wins'] += 1
                h2h['team_a_away_wins'] += 1
            elif m['away_goals'] < m['home_goals']:
                h2h['team_b_wins'] += 1
            else:
                h2h['draws'] += 1
        
        # Market statistics
        total_goals = m['home_goals'] + m['away_goals']
        if total_goals > 2.5:
            h2h['over_2_5_count'] += 1
        if m['home_goals'] > 0 and m['away_goals'] > 0:
            h2h['btts_count'] += 1
    
    return h2h
```

---

## 2️⃣ H2H METRIKEN

### Win Rates

```python
def calculate_h2h_rates(h2h: dict) -> dict:
    """
    Berechne alle wichtigen H2H-Raten
    """
    total = h2h['total_matches']
    if total == 0:
        return None
    
    return {
        # Win Rates
        'team_a_win_rate': h2h['team_a_wins'] / total,
        'team_b_win_rate': h2h['team_b_wins'] / total,
        'draw_rate': h2h['draws'] / total,
        
        # Goal Averages
        'avg_total_goals': (h2h['team_a_goals'] + h2h['team_b_goals']) / total,
        'team_a_avg_goals': h2h['team_a_goals'] / total,
        'team_b_avg_goals': h2h['team_b_goals'] / total,
        
        # Market Rates
        'over_2_5_rate': h2h['over_2_5_count'] / total,
        'btts_rate': h2h['btts_count'] / total,
        
        # Dominance
        'dominance_score': (h2h['team_a_wins'] - h2h['team_b_wins']) / total,
        'is_one_sided': abs(h2h['team_a_wins'] - h2h['team_b_wins']) / total > 0.4
    }

# Interpretation:
# dominance_score > 0.3: Team A dominiert H2H
# dominance_score < -0.3: Team B dominiert H2H
# is_one_sided: Einer gewinnt >70% der Duelle
```

### Recent Form Bias

```python
def recent_h2h_analysis(recent_matches: list, team_a: str) -> dict:
    """
    Analysiere nur die letzten 5 H2H-Spiele (wichtiger!)
    
    Recent results matter more than 10-year history
    """
    if len(recent_matches) == 0:
        return None
    
    recent_a_wins = 0
    recent_b_wins = 0
    recent_draws = 0
    recent_goals = 0
    
    for m in recent_matches[:5]:
        total_goals = m['home_goals'] + m['away_goals']
        recent_goals += total_goals
        
        if m['home'] == team_a:
            if m['home_goals'] > m['away_goals']:
                recent_a_wins += 1
            elif m['home_goals'] < m['away_goals']:
                recent_b_wins += 1
            else:
                recent_draws += 1
        else:
            if m['away_goals'] > m['home_goals']:
                recent_a_wins += 1
            elif m['away_goals'] < m['home_goals']:
                recent_b_wins += 1
            else:
                recent_draws += 1
    
    n = len(recent_matches[:5])
    
    return {
        'recent_count': n,
        'recent_a_win_rate': recent_a_wins / n,
        'recent_b_win_rate': recent_b_wins / n,
        'recent_draw_rate': recent_draws / n,
        'recent_avg_goals': recent_goals / n,
        'recent_momentum': 'team_a' if recent_a_wins > recent_b_wins else (
            'team_b' if recent_b_wins > recent_a_wins else 'neutral'
        )
    }
```

---

## 3️⃣ DERBY & RIVALRY DETECTION

### Derby Identification

```python
KNOWN_DERBYS = {
    # Deutschland
    'bundesliga': [
        ('Bayern München', 'Borussia Dortmund', 'Der Klassiker'),
        ('Borussia Dortmund', 'Schalke 04', 'Revierderby'),
        ('Bayern München', 'TSV 1860 München', 'Münchner Derby'),
        ('Hamburger SV', 'Werder Bremen', 'Nordderby'),
        ('Eintracht Frankfurt', 'Mainz 05', 'Rhein-Main Derby'),
    ],
    # England
    'premier_league': [
        ('Liverpool', 'Manchester United', 'North-West Derby'),
        ('Manchester City', 'Manchester United', 'Manchester Derby'),
        ('Arsenal', 'Tottenham', 'North London Derby'),
        ('Liverpool', 'Everton', 'Merseyside Derby'),
        ('Chelsea', 'Tottenham', 'London Derby'),
    ],
    # Spanien
    'la_liga': [
        ('Real Madrid', 'Barcelona', 'El Clásico'),
        ('Real Madrid', 'Atlético Madrid', 'Derby Madrileño'),
        ('Barcelona', 'Espanyol', 'Derbi Barceloní'),
        ('Athletic Bilbao', 'Real Sociedad', 'Derbi Vasco'),
        ('Sevilla', 'Real Betis', 'Derbi Sevillano'),
    ],
    # Italien
    'serie_a': [
        ('AC Milan', 'Inter Milan', 'Derby della Madonnina'),
        ('Roma', 'Lazio', 'Derby della Capitale'),
        ('Juventus', 'Torino', 'Derby della Mole'),
        ('Napoli', 'Roma', 'Derby del Sole'),
        ('Genoa', 'Sampdoria', 'Derby della Lanterna'),
    ]
}

def is_derby(team_a: str, team_b: str, league: str) -> dict:
    """
    Prüfe ob das Spiel ein Derby ist
    
    Derbys haben spezielle Charakteristiken!
    """
    league_derbys = KNOWN_DERBYS.get(league.lower(), [])
    
    for home, away, name in league_derbys:
        if (team_a == home and team_b == away) or (team_a == away and team_b == home):
            return {
                'is_derby': True,
                'derby_name': name,
                'intensity': 'high'
            }
    
    return {'is_derby': False}
```

### Derby Statistics

```python
DERBY_CHARACTERISTICS = {
    'general': {
        'more_cards': True,              # Durchschnittlich +1.2 Karten
        'more_fouls': True,              # +15% Fouls
        'more_draws': True,              # +8% Draw Rate
        'lower_goals': True,             # -0.3 Tore im Schnitt
        'unpredictable': True            # Form weniger relevant
    },
    'betting_implications': {
        'draw_value': 'often_good',      # Draws häufiger
        'over_2_5': 'slightly_avoid',    # Weniger Tore
        'btts': 'neutral',               # Keine klare Tendenz
        'cards_over': 'recommended'      # Mehr Karten erwartet
    }
}

def adjust_for_derby(base_probabilities: dict, derby_info: dict) -> dict:
    """
    Passe Wahrscheinlichkeiten für Derbys an
    """
    if not derby_info.get('is_derby'):
        return base_probabilities
    
    adjusted = base_probabilities.copy()
    
    # Derbys sind unvorhersehbarer - ziehe Wahrscheinlichkeiten zur Mitte
    for market in ['home_win', 'away_win']:
        if market in adjusted:
            # Reduziere extreme Wahrscheinlichkeiten
            adjusted[market] = 0.33 + (adjusted[market] - 0.33) * 0.7
    
    # Erhöhe Draw-Wahrscheinlichkeit
    if 'draw' in adjusted:
        adjusted['draw'] = min(0.40, adjusted['draw'] * 1.15)
    
    # Senke Goal-Erwartung
    if 'over_2_5' in adjusted:
        adjusted['over_2_5'] *= 0.92
    
    return adjusted
```

---

## 4️⃣ H2H GEWICHTUNG

### Time Decay

```python
def apply_time_decay(h2h_matches: list, decay_factor: float = 0.85) -> list:
    """
    Gewichte ältere Spiele weniger
    
    decay_factor = 0.85 bedeutet: Jedes Jahr verliert 15% Gewicht
    """
    from datetime import datetime
    
    weighted_matches = []
    today = datetime.now()
    
    for match in h2h_matches:
        match_date = datetime.strptime(match['date'], '%Y-%m-%d')
        years_ago = (today - match_date).days / 365
        
        weight = decay_factor ** years_ago
        
        weighted_matches.append({
            **match,
            'weight': weight,
            'years_ago': round(years_ago, 1)
        })
    
    return weighted_matches

# Gewichte:
# 0 Jahre: 1.00 (voll)
# 1 Jahr: 0.85
# 2 Jahre: 0.72
# 3 Jahre: 0.61
# 4 Jahre: 0.52
# 5+ Jahre: <0.50 (wenig relevant)
```

### Relevance Score

```python
def h2h_relevance_score(h2h_data: dict, team_changes: dict = None) -> float:
    """
    Berechne wie relevant die H2H-Daten sind
    
    Hohe Relevanz (>0.7): H2H stark berücksichtigen
    Niedrige Relevanz (<0.3): H2H kaum berücksichtigen
    """
    score = 1.0
    
    # Wenige Spiele = weniger relevant
    total_matches = h2h_data['total_matches']
    if total_matches < 3:
        score *= 0.5
    elif total_matches < 5:
        score *= 0.7
    elif total_matches > 10:
        score *= 1.1  # Mehr Daten = besser
    
    # Alte Spiele = weniger relevant
    recent = h2h_data.get('recent_matches', [])
    if recent:
        from datetime import datetime
        latest = datetime.strptime(recent[0]['date'], '%Y-%m-%d')
        years_since = (datetime.now() - latest).days / 365
        if years_since > 2:
            score *= 0.6
        elif years_since > 1:
            score *= 0.8
    
    # Teamveränderungen = weniger relevant
    if team_changes:
        if team_changes.get('new_manager', False):
            score *= 0.7
        if team_changes.get('major_transfers', 0) > 3:
            score *= 0.8
    
    return min(1.0, score)
```

---

## 5️⃣ H2H FÜR MÄRKTE

### Over/Under Based on H2H

```python
def h2h_over_under_prediction(h2h_rates: dict, line: float = 2.5) -> dict:
    """
    Vorhersage für Over/Under basierend auf H2H
    """
    avg_goals = h2h_rates['avg_total_goals']
    over_rate = h2h_rates['over_2_5_rate']
    
    # Basis: Historische Rate
    over_prob = over_rate
    
    # Anpassung basierend auf durchschnittlichen Toren
    if avg_goals > 3.0:
        over_prob = min(0.80, over_prob * 1.10)
    elif avg_goals < 2.0:
        over_prob = max(0.20, over_prob * 0.85)
    
    return {
        f'over_{line}': over_prob,
        f'under_{line}': 1 - over_prob,
        'confidence': 'high' if h2h_rates.get('total_matches', 0) >= 5 else 'medium'
    }
```

### BTTS Based on H2H

```python
def h2h_btts_prediction(h2h_rates: dict) -> dict:
    """
    Vorhersage für BTTS basierend auf H2H
    """
    btts_rate = h2h_rates['btts_rate']
    team_a_avg = h2h_rates['team_a_avg_goals']
    team_b_avg = h2h_rates['team_b_avg_goals']
    
    # Beide Teams müssen regelmäßig treffen
    both_score_indicator = min(team_a_avg, team_b_avg)
    
    if both_score_indicator < 0.5:
        btts_prob = btts_rate * 0.85  # Einer trifft selten
    elif both_score_indicator > 1.5:
        btts_prob = min(0.80, btts_rate * 1.10)  # Beide treffen oft
    else:
        btts_prob = btts_rate
    
    return {
        'btts_yes': btts_prob,
        'btts_no': 1 - btts_prob
    }
```

### 1X2 Based on H2H

```python
def h2h_1x2_prediction(h2h_rates: dict, recent_h2h: dict) -> dict:
    """
    Vorhersage für 1X2 basierend auf H2H
    
    Kombiniere historische und aktuelle H2H-Daten
    """
    # Historische Raten (40% Gewicht)
    hist_a = h2h_rates['team_a_win_rate']
    hist_b = h2h_rates['team_b_win_rate']
    hist_d = h2h_rates['draw_rate']
    
    # Aktuelle Raten (60% Gewicht)
    recent_a = recent_h2h['recent_a_win_rate']
    recent_b = recent_h2h['recent_b_win_rate']
    recent_d = recent_h2h['recent_draw_rate']
    
    # Gewichtete Kombination
    prob_a = 0.4 * hist_a + 0.6 * recent_a
    prob_b = 0.4 * hist_b + 0.6 * recent_b
    prob_d = 0.4 * hist_d + 0.6 * recent_d
    
    # Normalisieren
    total = prob_a + prob_b + prob_d
    
    return {
        'team_a_win': prob_a / total,
        'draw': prob_d / total,
        'team_b_win': prob_b / total
    }
```

---

## 6️⃣ IMPLEMENTATION FÜR AI

```python
class H2HAnalyzer:
    """Komplette H2H-Analyse für das AI System"""
    
    def __init__(self, team_a: str, team_b: str, matches: list, league: str):
        self.team_a = team_a
        self.team_b = team_b
        self.matches = matches
        self.league = league
        
        self.h2h_data = collect_h2h_data(team_a, team_b, matches)
        self.h2h_rates = calculate_h2h_rates(self.h2h_data)
        self.recent_h2h = recent_h2h_analysis(matches[:5], team_a)
        self.derby_info = is_derby(team_a, team_b, league)
    
    def get_full_analysis(self) -> dict:
        """Liefere komplette H2H-Analyse"""
        return {
            'basic_stats': self.h2h_data,
            'rates': self.h2h_rates,
            'recent_form': self.recent_h2h,
            'derby': self.derby_info,
            'relevance': h2h_relevance_score(self.h2h_data)
        }
    
    def get_market_probabilities(self) -> dict:
        """Liefere H2H-basierte Wahrscheinlichkeiten für alle Märkte"""
        probs = {}
        
        # Over/Under
        ou = h2h_over_under_prediction(self.h2h_rates)
        probs['over_2_5'] = ou['over_2.5']
        probs['under_2_5'] = ou['under_2.5']
        
        # BTTS
        btts = h2h_btts_prediction(self.h2h_rates)
        probs['btts_yes'] = btts['btts_yes']
        probs['btts_no'] = btts['btts_no']
        
        # 1X2
        if self.recent_h2h:
            match_result = h2h_1x2_prediction(self.h2h_rates, self.recent_h2h)
            probs['home_win'] = match_result['team_a_win']
            probs['draw'] = match_result['draw']
            probs['away_win'] = match_result['team_b_win']
        
        # Derby-Anpassungen
        if self.derby_info['is_derby']:
            probs = adjust_for_derby(probs, self.derby_info)
        
        return probs
    
    def get_betting_insights(self) -> list:
        """Liefere Wett-Empfehlungen basierend auf H2H"""
        insights = []
        
        if self.h2h_rates['is_one_sided']:
            dominant = 'team_a' if self.h2h_rates['dominance_score'] > 0 else 'team_b'
            insights.append(f"H2H stark einseitig zugunsten {dominant}")
        
        if self.h2h_rates['over_2_5_rate'] > 0.65:
            insights.append("H2H zeigt häufig Over 2.5 Goals")
        elif self.h2h_rates['over_2_5_rate'] < 0.35:
            insights.append("H2H zeigt häufig Under 2.5 Goals")
        
        if self.h2h_rates['btts_rate'] > 0.70:
            insights.append("H2H: Beide Teams treffen regelmäßig")
        
        if self.derby_info['is_derby']:
            insights.append(f"DERBY: {self.derby_info['derby_name']} - Unvorhersehbar!")
        
        return insights

# Nutzung:
# analyzer = H2HAnalyzer('Bayern', 'Dortmund', h2h_matches, 'bundesliga')
# analysis = analyzer.get_full_analysis()
# probs = analyzer.get_market_probabilities()
# insights = analyzer.get_betting_insights()
```

---

## 📊 Quick Reference: H2H Schwellenwerte

```
╔════════════════════════════════════════════════════════════════╗
║                    H2H THRESHOLDS                              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  MINIMUM MATCHES: 5+ für verlässliche Statistiken             ║
║  RELEVANCE DECAY: -15% pro Jahr                               ║
║                                                                ║
║  ONE-SIDED H2H: Dominance Score > 0.4 (>70% Siege)           ║
║  BALANCED H2H:  Dominance Score < 0.1                        ║
║                                                                ║
║  HIGH SCORING H2H: Avg Goals > 3.0                           ║
║  LOW SCORING H2H:  Avg Goals < 2.0                           ║
║                                                                ║
║  DERBY ADJUSTMENTS:                                           ║
║  - Draw: +15%                                                 ║
║  - Over 2.5: -8%                                             ║
║  - Extreme Probs: Pull towards 33%                           ║
║                                                                ║
║  H2H WEIGHT IN FINAL MODEL: 15-25%                           ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**📚 Dieses Knowledge File enthält alle H2H-Analysen für historisch fundierte AI-Vorhersagen.**
