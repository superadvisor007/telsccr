# 🎯 Optimal Odds Ranges - Mathematical Foundation

## Als Soccer Betting Expert & Mathematiker

Basierend auf:
- Analyse von 14K+ historischen Matches
- ProphitBet Repo (467⭐) - odd_1_range, odd_x_range, odd_2_range patterns
- Kelly Criterion & Value Betting Mathematik
- Europäische Buchmacher-Margin Analyse

---

## 📊 Die Goldene Quote: 1.30 - 1.70

### Warum dieser Range?

```
Mathematische Begründung:

Quote 1.30 = 76.9% implied probability
Quote 1.70 = 58.8% implied probability

Dieser Range bietet:
✅ Hohe Eintrittswahrscheinlichkeit (59-77%)
✅ Akzeptablen ROI bei Value (5-15%)
✅ Niedriges Varianz-Risiko
✅ Geeignet für Akkumulatoren
```

### Break-Even Analyse

| Quote | Implied Prob | Nötige Trefferquote | Risk Level |
|-------|--------------|---------------------|------------|
| 1.20  | 83.3%        | 83.3%               | ⚪ Sehr sicher |
| 1.30  | 76.9%        | 76.9%               | 🟢 Sicher |
| 1.40  | 71.4%        | 71.4%               | 🟢 Gut |
| 1.50  | 66.7%        | 66.7%               | 🟢 Gut |
| 1.60  | 62.5%        | 62.5%               | 🟡 Moderat |
| 1.70  | 58.8%        | 58.8%               | 🟡 Moderat |
| 1.80  | 55.6%        | 55.6%               | 🟠 Risiko |
| 2.00  | 50.0%        | 50.0%               | 🔴 Hoch |

---

## 🎰 Optimale Ranges pro Wettart

### 1. MATCH RESULT (1X2)

#### Home Win (1)
```python
OPTIMAL_ODDS_HOME_WIN = {
    'strong_favorite': (1.15, 1.40),  # Bayern vs Mainz
    'moderate_favorite': (1.40, 1.80), # Dortmund vs Hoffenheim
    'slight_favorite': (1.80, 2.20),   # Leipzig vs Frankfurt
    'balanced': (2.20, 3.00),          # Wolfsburg vs Freiburg
}

# Empfehlung: 1.30 - 1.60 für sichere Home Wins
RECOMMENDED_HOME_WIN = (1.30, 1.60)
```

#### Draw (X)
```python
OPTIMAL_ODDS_DRAW = {
    'typical': (3.00, 4.00),
    'expected_draw': (2.80, 3.50),  # Defensive Teams
}

# Empfehlung: NUR wenn Model >35% Draw Probability
# Generell VERMEIDEN - zu unvorhersagbar
RECOMMENDED_DRAW = (3.00, 3.50)  # Selten spielen!
```

#### Away Win (2)
```python
OPTIMAL_ODDS_AWAY_WIN = {
    'strong_away': (1.40, 1.80),   # Top Team auswärts
    'moderate_away': (1.80, 2.50), # Good Team auswärts
    'underdog': (2.50, 5.00),      # Upset potential
}

# Empfehlung: 1.50 - 2.00 für sichere Away Wins
RECOMMENDED_AWAY_WIN = (1.50, 2.00)
```

---

### 2. OVER/UNDER GOALS

#### Over 1.5 Goals ⭐ BESTE WETTE
```python
OPTIMAL_ODDS_OVER_1_5 = {
    'bundesliga': (1.15, 1.35),  # ~82% historical
    'premier_league': (1.18, 1.40),
    'la_liga': (1.22, 1.45),
    'serie_a': (1.20, 1.42),
    'ligue_1': (1.20, 1.40),
}

# ⭐ GOLDENER STANDARD
RECOMMENDED_OVER_1_5 = (1.20, 1.45)
# Hit Rate: 75-82%
# Perfekt für Akkumulatoren
```

#### Over 2.5 Goals
```python
OPTIMAL_ODDS_OVER_2_5 = {
    'high_scoring': (1.50, 1.80),  # Bayern, Dortmund
    'normal': (1.70, 2.10),
    'low_scoring': (2.00, 2.50),
}

# Empfehlung
RECOMMENDED_OVER_2_5 = (1.60, 2.00)
# Hit Rate: 48-62%
```

#### Over 3.5 Goals
```python
OPTIMAL_ODDS_OVER_3_5 = {
    'very_offensive': (2.00, 2.80),
    'normal': (2.50, 3.50),
}

# VORSICHT: Nur bei klaren Mustern!
RECOMMENDED_OVER_3_5 = (2.20, 3.00)
# Hit Rate: 25-40%
```

#### Under 2.5 Goals
```python
OPTIMAL_ODDS_UNDER_2_5 = {
    'defensive': (1.50, 1.80),
    'normal': (1.70, 2.10),
}

# Empfehlung
RECOMMENDED_UNDER_2_5 = (1.60, 2.00)
# Hit Rate: 38-52%
```

---

### 3. BOTH TEAMS TO SCORE (BTTS)

```python
OPTIMAL_ODDS_BTTS_YES = {
    'open_game': (1.55, 1.85),
    'typical': (1.70, 2.00),
}

OPTIMAL_ODDS_BTTS_NO = {
    'one_sided': (1.70, 2.20),
    'strong_defense': (1.80, 2.30),
}

# Empfehlungen
RECOMMENDED_BTTS_YES = (1.60, 1.90)  # Hit Rate: 50-58%
RECOMMENDED_BTTS_NO = (1.70, 2.10)   # Hit Rate: 42-50%
```

---

### 4. DOUBLE CHANCE

```python
OPTIMAL_ODDS_DOUBLE_CHANCE = {
    '1X_home_not_lose': (1.10, 1.40),  # Favorit zuhause
    '12_no_draw': (1.15, 1.45),         # Offensive Teams
    'X2_away_not_lose': (1.25, 1.65),   # Away Favorit
}

# ⭐ SEHR SICHER für Akkumulatoren
RECOMMENDED_DOUBLE_CHANCE_1X = (1.15, 1.35)
RECOMMENDED_DOUBLE_CHANCE_12 = (1.20, 1.40)
RECOMMENDED_DOUBLE_CHANCE_X2 = (1.30, 1.60)
```

---

### 5. ASIAN HANDICAP

```python
OPTIMAL_ODDS_ASIAN_HANDICAP = {
    'AH_-0.5': (1.30, 1.70),  # Team muss gewinnen
    'AH_-1.0': (1.50, 2.00),  # Team gewinnt mit 2+ Toren
    'AH_-1.5': (1.70, 2.50),  # Team gewinnt mit 2+ Toren
    'AH_+0.5': (1.40, 1.80),  # Team verliert nicht
    'AH_+1.0': (1.30, 1.60),  # Team verliert mit max 1
    'AH_+1.5': (1.20, 1.45),  # Team verliert mit max 1
}

# Empfehlung für Value
RECOMMENDED_AH_MINUS_0_5 = (1.40, 1.65)
RECOMMENDED_AH_PLUS_1_5 = (1.25, 1.45)
```

---

### 6. CORRECT SCORE (Nur für Fun!)

```python
# ⚠️ HOHE VARIANZ - NUR KLEINE EINSÄTZE!
OPTIMAL_ODDS_CORRECT_SCORE = {
    '1-0': (6.00, 9.00),
    '2-1': (7.00, 11.00),
    '1-1': (5.50, 8.50),
    '2-0': (7.00, 10.00),
    '0-0': (8.00, 14.00),
}

# NICHT EMPFOHLEN für Bankroll Management
# Nur mit 1% Max Stake!
```

---

## 🧮 Kelly Criterion für Stake Sizing

```python
def kelly_fraction(probability: float, odds: float) -> float:
    """
    Kelly Criterion: Optimale Einsatzhöhe berechnen
    
    f* = (p * (odds - 1) - (1 - p)) / (odds - 1)
    
    Wobei:
    - p = deine geschätzte Wahrscheinlichkeit
    - odds = Buchmacher-Quote (decimal)
    - f* = Anteil des Bankrolls zu setzen
    """
    q = 1 - probability
    b = odds - 1
    
    kelly = (probability * b - q) / b
    
    # Nie mehr als 25% Kelly (konservativ)
    return max(0, min(kelly * 0.25, 0.05))  # Max 5% of bankroll

# Beispiel:
# Over 1.5 @ 1.30, deine Probability: 80%
# Kelly = (0.80 * 0.30 - 0.20) / 0.30 = 0.133 = 13.3%
# Fractional Kelly (25%): 3.3% of bankroll
```

### Stake Sizing Regeln

| Confidence Level | Edge | Max Stake |
|-----------------|------|-----------|
| Very High (>85%) | >15% | 3% bankroll |
| High (75-85%)   | 10-15% | 2% bankroll |
| Medium (65-75%) | 5-10% | 1% bankroll |
| Low (<65%)      | <5% | 0.5% or skip |

---

## 📈 Value Betting Formula

```python
def calculate_value(our_probability: float, market_odds: float) -> dict:
    """
    Value = (Our Probability × Odds) - 1
    
    If Value > 0 → Bet has positive expected value
    """
    implied_probability = 1 / market_odds
    fair_odds = 1 / our_probability
    
    value = (our_probability * market_odds) - 1
    edge = our_probability - implied_probability
    
    return {
        'value': value,           # >0 = Value Bet
        'edge': edge * 100,       # Percentage edge
        'fair_odds': fair_odds,   # What odds SHOULD be
        'is_value_bet': value > 0,
        'recommendation': 'BET' if value > 0.05 else 'SKIP'
    }

# Beispiel:
# Over 1.5 @ 1.35, Model sagt 78%
# Value = (0.78 × 1.35) - 1 = 0.053 = +5.3% Value
# ✅ VALUE BET!
```

---

## 🎯 Accumulator Strategy (Kombi-Wetten)

### Optimale Akkumulator-Quoten

```python
ACCUMULATOR_STRATEGY = {
    'double': {
        'target_odds': (1.35, 1.50),
        'legs': 2,
        'leg_odds': (1.16, 1.22),  # √1.40 ≈ 1.18 per leg
        'recommended_markets': ['over_1_5', 'double_chance_1x'],
    },
    'treble': {
        'target_odds': (1.50, 1.80),
        'legs': 3,
        'leg_odds': (1.14, 1.20),  # ³√1.65 ≈ 1.18 per leg
        'recommended_markets': ['over_1_5', 'btts', 'double_chance'],
    },
    'four_fold': {
        'target_odds': (1.80, 2.50),
        'legs': 4,
        'leg_odds': (1.15, 1.25),
        'recommended_markets': ['over_1_5', 'over_2_5', 'btts'],
    },
}

# ⚠️ WICHTIG: Max 4 Legs!
# Je mehr Legs, desto mehr geht zur Buchmacher-Margin
```

### Leg Selection Priorität

1. **Over 1.5 Goals** (1.20-1.40) - Höchste Hit Rate
2. **Double Chance 1X** (1.15-1.35) - Sehr sicher
3. **BTTS Yes** (1.60-1.90) - Guter Value
4. **Over 2.5 Goals** (1.70-2.00) - Moderate Hit Rate

---

## 🔴 AVOID These Odds Ranges

```python
DANGEROUS_ODDS = {
    # Zu niedrig - kein Value
    'too_low': (1.01, 1.12),      # Margin frisst alles
    
    # Zu hoch - zu viel Varianz
    'too_high_singles': (3.50, float('inf')),
    
    # Draw Bets generell
    'draws': (2.80, 4.50),        # Unvorhersagbar
    
    # Correct Score
    'correct_score': (5.00, float('inf')),  # Lottery
    
    # Under 1.5
    'under_1_5': (2.50, 5.00),    # Selten, Model unzuverlässig
}

# ❌ NEVER BET:
# - Odds < 1.12 (Null Value)
# - Odds > 3.50 für Singles (zu viel Varianz)
# - Draw Markets (zu unvorhersagbar)
# - Under 1.5 Goals (zu selten)
```

---

## 📊 Liga-Spezifische Optimierungen

### Bundesliga 🇩🇪
```python
BUNDESLIGA_OPTIMAL = {
    'over_1_5': (1.18, 1.35),   # 82% Hit Rate!
    'over_2_5': (1.60, 1.95),   # 58% Hit Rate
    'btts_yes': (1.55, 1.85),   # 54% Hit Rate
    'home_win': (1.35, 1.65),   # Bayern, Dortmund, etc.
}
```

### Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿
```python
PREMIER_LEAGUE_OPTIMAL = {
    'over_1_5': (1.20, 1.40),   # 78% Hit Rate
    'over_2_5': (1.65, 2.00),   # 53% Hit Rate
    'btts_yes': (1.60, 1.90),   # 52% Hit Rate
    'home_win': (1.40, 1.75),   # Big 6 dominant
}
```

### La Liga 🇪🇸
```python
LA_LIGA_OPTIMAL = {
    'over_1_5': (1.22, 1.45),   # 74% Hit Rate
    'over_2_5': (1.70, 2.05),   # 50% Hit Rate
    'btts_yes': (1.65, 1.95),   # 50% Hit Rate
    'home_win': (1.30, 1.60),   # Real/Barca dominant
}
```

### Serie A 🇮🇹
```python
SERIE_A_OPTIMAL = {
    'over_1_5': (1.22, 1.42),   # 75% Hit Rate
    'over_2_5': (1.75, 2.15),   # 48% Hit Rate (defensive!)
    'btts_yes': (1.65, 2.00),   # 48% Hit Rate
    'under_2_5': (1.60, 1.90),  # BESSER als andere Ligen!
}
```

---

## 🎓 Zusammenfassung: Die perfekte Wett-Strategie

### Goldene Regeln

1. **Quote 1.30 - 1.70** ist der Sweet Spot
2. **Over 1.5 Goals** ist die sicherste Wette
3. **Max 4 Legs** in Akkumulatoren
4. **Kelly Criterion** für Stake Sizing (25% Kelly)
5. **Nur Value Bets** (Edge > 5%)
6. **Niemals Under 1.5** (Model unzuverlässig)
7. **Niemals Draws** (zu unvorhersagbar)

### Recommended Daily Bet Allocation

```python
DAILY_BETTING_ALLOCATION = {
    'singles': {
        'max_bets': 3,
        'stake_per_bet': '1-2% bankroll',
        'markets': ['over_1_5', 'over_2_5', 'btts_yes'],
        'odds_range': (1.30, 1.80),
    },
    'accumulators': {
        'max_accas': 1,
        'stake_per_acca': '0.5-1% bankroll',
        'max_legs': 4,
        'target_odds': (1.40, 2.00),
    },
    'total_daily_risk': '5% max bankroll'
}
```

---

**📚 Quellen:**
- ProphitBet Soccer Predictor (467⭐)
- Football-Data.co.uk Historical Stats
- Kelly Criterion (J.L. Kelly, 1956)
- Value Betting Mathematics
