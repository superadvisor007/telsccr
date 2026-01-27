# 🧠 Soccer Betting Mathematics - Complete Reference

## Für AI Model Training & Value Detection

---

## 1️⃣ GRUNDLAGEN: Quoten verstehen

### Decimal Odds (Europäisch) - UNSER STANDARD

```python
# Decimal Odds = Total Return für €1 Einsatz
decimal_odds = 1.50

# Bei Gewinn:
return_amount = stake * decimal_odds
profit = return_amount - stake

# Beispiel: €10 @ 1.50
return_amount = 10 * 1.50  # = €15
profit = 15 - 10           # = €5
```

### Implied Probability (Implizite Wahrscheinlichkeit)

```python
def implied_probability(decimal_odds: float) -> float:
    """
    Berechne die implizite Wahrscheinlichkeit aus Decimal Odds
    
    Formula: Implied Prob = 1 / Decimal Odds
    """
    return 1 / decimal_odds

# Beispiele:
implied_probability(1.20)  # = 83.3%
implied_probability(1.50)  # = 66.7%
implied_probability(2.00)  # = 50.0%
implied_probability(3.00)  # = 33.3%
implied_probability(5.00)  # = 20.0%
```

### Conversion Table

| Decimal | Implied Prob | Fractional | American |
|---------|--------------|------------|----------|
| 1.10 | 90.9% | 1/10 | -1000 |
| 1.20 | 83.3% | 1/5 | -500 |
| 1.30 | 76.9% | 3/10 | -333 |
| 1.40 | 71.4% | 2/5 | -250 |
| 1.50 | 66.7% | 1/2 | -200 |
| 1.60 | 62.5% | 3/5 | -167 |
| 1.70 | 58.8% | 7/10 | -143 |
| 1.80 | 55.6% | 4/5 | -125 |
| 1.90 | 52.6% | 9/10 | -111 |
| 2.00 | 50.0% | 1/1 | +100 |
| 2.50 | 40.0% | 3/2 | +150 |
| 3.00 | 33.3% | 2/1 | +200 |
| 4.00 | 25.0% | 3/1 | +300 |
| 5.00 | 20.0% | 4/1 | +400 |

---

## 2️⃣ BUCHMACHER MARGIN (Vig/Juice)

### Was ist Margin?

```python
def calculate_margin(odds_home: float, odds_draw: float, odds_away: float) -> float:
    """
    Berechne die Buchmacher-Margin (Overround)
    
    Formula: Margin = (1/odds_1 + 1/odds_X + 1/odds_2) - 1
    
    Beispiel: Bayern vs Mainz
    - 1: 1.30 (76.9%)
    - X: 5.50 (18.2%)
    - 2: 9.00 (11.1%)
    
    Total: 76.9 + 18.2 + 11.1 = 106.2%
    Margin: 106.2 - 100 = 6.2%
    """
    total_implied = (1/odds_home) + (1/odds_draw) + (1/odds_away)
    margin = (total_implied - 1) * 100
    return margin

# Typische Margins:
# Top Buchmacher: 2-4%
# Durchschnitt: 5-8%
# Schlechte: 10%+
```

### True Probability (Wahre Wahrscheinlichkeit)

```python
def true_probability(implied_prob: float, total_implied: float) -> float:
    """
    Berechne die wahre Wahrscheinlichkeit ohne Margin
    
    Formula: True Prob = Implied Prob / Total Implied
    """
    return implied_prob / total_implied

# Beispiel:
# Odds: 1.30 → Implied: 76.9%
# Total Implied: 106.2%
# True Prob: 76.9 / 106.2 = 72.4%
```

---

## 3️⃣ VALUE BETTING - Kernkonzept

### Was ist Value?

```python
def calculate_value(our_probability: float, decimal_odds: float) -> float:
    """
    Value = (Our Probability × Odds) - 1
    
    Wenn Value > 0 → Positive Expected Value (Langfristig profitabel!)
    Wenn Value < 0 → Negative Expected Value (Langfristig Verlust!)
    """
    value = (our_probability * decimal_odds) - 1
    return value

# Beispiel 1: VALUE BET ✅
# Model sagt: 75% Wahrscheinlichkeit
# Odds: 1.40 (impliziert 71.4%)
# Value = (0.75 × 1.40) - 1 = 0.05 = +5%

# Beispiel 2: KEIN VALUE ❌
# Model sagt: 65% Wahrscheinlichkeit  
# Odds: 1.40 (impliziert 71.4%)
# Value = (0.65 × 1.40) - 1 = -0.09 = -9%
```

### Edge (Vorteil über Buchmacher)

```python
def calculate_edge(our_probability: float, implied_probability: float) -> float:
    """
    Edge = Our Probability - Implied Probability
    
    Positiver Edge = Wir haben einen Vorteil
    """
    edge = our_probability - implied_probability
    return edge * 100  # Als Prozent

# Beispiel:
# Our Prob: 78%
# Implied Prob: 71.4% (Odds 1.40)
# Edge: 78 - 71.4 = +6.6%
```

### Minimum Value Threshold

```python
# Für profitables Wetten brauchen wir:
MIN_VALUE_THRESHOLD = 0.05  # 5% minimum value
MIN_EDGE_THRESHOLD = 0.05   # 5% minimum edge

def is_value_bet(our_prob: float, odds: float) -> bool:
    value = (our_prob * odds) - 1
    return value >= MIN_VALUE_THRESHOLD
```

---

## 4️⃣ EXPECTED VALUE (EV)

### Single Bet EV

```python
def expected_value(probability: float, odds: float, stake: float) -> float:
    """
    Expected Value = (Probability × Profit) - (1-Probability × Stake)
    
    Oder vereinfacht:
    EV = Stake × [(Probability × Odds) - 1]
    """
    profit_if_win = stake * (odds - 1)
    loss_if_lose = stake
    
    ev = (probability * profit_if_win) - ((1 - probability) * loss_if_lose)
    return ev

# Beispiel:
# Stake: €10
# Odds: 1.50
# Our Probability: 72%
# 
# Profit if win: €10 × 0.50 = €5
# EV = (0.72 × €5) - (0.28 × €10)
# EV = €3.60 - €2.80 = +€0.80
#
# Langfristig: €0.80 Gewinn pro €10 Einsatz = +8% ROI
```

### Accumulator EV

```python
def accumulator_ev(legs: list, stake: float) -> dict:
    """
    Berechne EV für Akkumulator
    
    legs = [{'prob': 0.75, 'odds': 1.25}, {'prob': 0.70, 'odds': 1.40}]
    """
    combined_prob = 1.0
    combined_odds = 1.0
    
    for leg in legs:
        combined_prob *= leg['prob']
        combined_odds *= leg['odds']
    
    potential_return = stake * combined_odds
    ev = (combined_prob * potential_return) - stake
    
    return {
        'combined_prob': combined_prob,
        'combined_odds': combined_odds,
        'potential_return': potential_return,
        'ev': ev,
        'roi_percent': (ev / stake) * 100
    }

# Beispiel: 3-fach Akku
# Leg 1: 75% @ 1.25
# Leg 2: 70% @ 1.40  
# Leg 3: 68% @ 1.35
#
# Combined Prob: 0.75 × 0.70 × 0.68 = 35.7%
# Combined Odds: 1.25 × 1.40 × 1.35 = 2.36
# 
# €10 Stake:
# Return: €10 × 2.36 = €23.60
# EV = (0.357 × €23.60) - €10 = €8.43 - €10 = -€1.57
# ❌ Negative EV trotz Value in einzelnen Legs!
```

---

## 5️⃣ KELLY CRITERION

### Full Kelly

```python
def kelly_criterion(probability: float, odds: float) -> float:
    """
    Kelly Fraction = (p × b - q) / b
    
    Wobei:
    - p = Unsere Wahrscheinlichkeit
    - q = 1 - p (Gegenwahrscheinlichkeit)
    - b = Decimal Odds - 1 (Net Odds)
    
    Returns: Optimaler Anteil des Bankrolls
    """
    p = probability
    q = 1 - p
    b = odds - 1
    
    kelly = (p * b - q) / b
    
    return max(0, kelly)  # Niemals negative Einsätze

# Beispiel:
# Probability: 60%
# Odds: 2.00 (b = 1.0)
#
# Kelly = (0.60 × 1.0 - 0.40) / 1.0
# Kelly = (0.60 - 0.40) / 1.0
# Kelly = 0.20 = 20% des Bankrolls
```

### Fractional Kelly (EMPFOHLEN!)

```python
def fractional_kelly(probability: float, odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly = Kelly × Fraction
    
    Standard: 25% Kelly (konservativ)
    Aggressiv: 50% Kelly
    Ultra-konservativ: 10% Kelly
    """
    full_kelly = kelly_criterion(probability, odds)
    return full_kelly * fraction

# Beispiel mit 25% Kelly:
# Full Kelly: 20%
# Fractional: 20% × 0.25 = 5% des Bankrolls
```

### Kelly Table (25% Fractional)

| Prob | Odds 1.30 | Odds 1.50 | Odds 1.80 | Odds 2.00 |
|------|-----------|-----------|-----------|-----------|
| 55% | 0.0% | 0.0% | 0.3% | 1.3% |
| 60% | 0.4% | 1.5% | 2.3% | 2.5% |
| 65% | 1.5% | 3.0% | 3.8% | 3.8% |
| 70% | 2.7% | 4.5% | 5.1% | 5.0% |
| 75% | 4.0% | 5.8% | 6.3% | 6.3% |
| 80% | 5.4% | 7.0% | 7.4% | 7.5% |

---

## 6️⃣ POISSON DISTRIBUTION FÜR TORE

### Grundlagen

```python
import math

def poisson_probability(k: int, lambda_: float) -> float:
    """
    Poisson Verteilung für Torvorhersage
    
    P(X = k) = (λ^k × e^(-λ)) / k!
    
    Wobei:
    - k = Anzahl Tore
    - λ (lambda) = Erwartete Tore
    """
    return (math.pow(lambda_, k) * math.exp(-lambda_)) / math.factorial(k)

# Beispiel: Bayern erwartet 2.5 Tore
expected_goals = 2.5

print("Torwahrscheinlichkeiten:")
for goals in range(6):
    prob = poisson_probability(goals, expected_goals)
    print(f"{goals} Tore: {prob*100:.1f}%")

# Output:
# 0 Tore: 8.2%
# 1 Tor:  20.5%
# 2 Tore: 25.7%
# 3 Tore: 21.4%
# 4 Tore: 13.4%
# 5 Tore: 6.7%
```

### Match Outcome Probabilities

```python
def match_probabilities(home_xg: float, away_xg: float, max_goals: int = 7) -> dict:
    """
    Berechne Wahrscheinlichkeiten für alle Ergebnisse
    """
    results = {}
    
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0
    
    for home_goals in range(max_goals):
        for away_goals in range(max_goals):
            home_prob = poisson_probability(home_goals, home_xg)
            away_prob = poisson_probability(away_goals, away_xg)
            
            score_prob = home_prob * away_prob
            results[f"{home_goals}-{away_goals}"] = score_prob
            
            if home_goals > away_goals:
                home_win_prob += score_prob
            elif home_goals == away_goals:
                draw_prob += score_prob
            else:
                away_win_prob += score_prob
    
    return {
        'home_win': home_win_prob,
        'draw': draw_prob,
        'away_win': away_win_prob,
        'scores': results
    }

# Beispiel: Bayern (xG: 2.3) vs Mainz (xG: 0.9)
probs = match_probabilities(2.3, 0.9)
print(f"Bayern Win: {probs['home_win']*100:.1f}%")  # ~71%
print(f"Draw: {probs['draw']*100:.1f}%")            # ~16%
print(f"Mainz Win: {probs['away_win']*100:.1f}%")   # ~13%
```

### Over/Under Probabilities

```python
def over_under_probability(home_xg: float, away_xg: float, line: float) -> dict:
    """
    Berechne Over/Under Wahrscheinlichkeiten
    """
    total_xg = home_xg + away_xg
    
    under_prob = 0
    for total_goals in range(int(line) + 1):
        for home in range(total_goals + 1):
            away = total_goals - home
            if away >= 0:
                prob = poisson_probability(home, home_xg) * poisson_probability(away, away_xg)
                under_prob += prob
    
    over_prob = 1 - under_prob
    
    return {
        f'over_{line}': over_prob,
        f'under_{line}': under_prob
    }

# Beispiel: Expected Total Goals = 3.2
probs = over_under_probability(2.3, 0.9, 2.5)
print(f"Over 2.5: {probs['over_2.5']*100:.1f}%")   # ~72%
print(f"Under 2.5: {probs['under_2.5']*100:.1f}%") # ~28%
```

---

## 7️⃣ ROI & YIELD BERECHNUNG

### Return on Investment (ROI)

```python
def calculate_roi(profit: float, total_staked: float) -> float:
    """
    ROI = (Profit / Total Staked) × 100
    
    Positiver ROI = Profitabel
    Negativer ROI = Verlust
    """
    return (profit / total_staked) * 100

# Beispiel:
# Total Staked: €1000
# Total Returns: €1080
# Profit: €80
# ROI = (80 / 1000) × 100 = 8%
```

### Yield (Pro Wette)

```python
def calculate_yield(profit: float, num_bets: int, avg_stake: float) -> float:
    """
    Yield = Profit / (Number of Bets × Average Stake) × 100
    
    Yield ist aussagekräftiger als ROI bei unterschiedlichen Stakes
    """
    total_staked = num_bets * avg_stake
    return (profit / total_staked) * 100

# Gute Yields:
# Professional Tipster: 5-10%
# Sehr gut: 3-5%
# Gut: 1-3%
# Break-even: 0%
```

### Bankroll Growth

```python
def bankroll_growth(initial_bankroll: float, roi_percent: float, 
                    bets_per_month: int, months: int) -> float:
    """
    Berechne Bankroll-Wachstum über Zeit
    
    Assumes consistent ROI (idealized)
    """
    roi = roi_percent / 100
    avg_stake_pct = 0.02  # 2% average stake
    
    bankroll = initial_bankroll
    
    for month in range(months):
        monthly_bets = bets_per_month
        for bet in range(monthly_bets):
            stake = bankroll * avg_stake_pct
            expected_profit = stake * roi
            bankroll += expected_profit
    
    return bankroll

# Beispiel:
# Start: €1000
# ROI: 5%
# 60 Bets/Monat
# Nach 12 Monaten:
growth = bankroll_growth(1000, 5, 60, 12)
print(f"Final Bankroll: €{growth:.2f}")  # ~€1430
```

---

## 8️⃣ VARIANZ & DRAWDOWNS

### Standard Deviation

```python
import numpy as np

def calculate_variance(results: list) -> dict:
    """
    Berechne Varianz-Metriken für Wettergebnisse
    
    results = [+10, -10, +15, -10, +20, -10, ...]
    """
    results = np.array(results)
    
    return {
        'mean': np.mean(results),
        'std_dev': np.std(results),
        'variance': np.var(results),
        'max_win': np.max(results),
        'max_loss': np.min(results),
        'win_rate': np.sum(results > 0) / len(results)
    }
```

### Maximum Drawdown

```python
def max_drawdown(equity_curve: list) -> dict:
    """
    Berechne den maximalen Drawdown
    
    equity_curve = [1000, 1050, 1020, 980, 1100, 1080, ...]
    """
    peak = equity_curve[0]
    max_dd = 0
    max_dd_pct = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        
        dd = peak - value
        dd_pct = dd / peak
        
        if dd_pct > max_dd_pct:
            max_dd = dd
            max_dd_pct = dd_pct
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct * 100
    }

# Gesunde Drawdowns:
# Professional: Max 15-20%
# Acceptable: Max 25-30%
# Danger Zone: >40%
```

### Sharpe Ratio (Risk-Adjusted Return)

```python
def sharpe_ratio(returns: list, risk_free_rate: float = 0.02) -> float:
    """
    Sharpe Ratio = (Mean Return - Risk Free Rate) / Standard Deviation
    
    Higher is better (>1 is good, >2 is excellent)
    """
    returns = np.array(returns)
    excess_return = np.mean(returns) - risk_free_rate
    std_dev = np.std(returns)
    
    if std_dev == 0:
        return 0
    
    return excess_return / std_dev

# Interpretation:
# < 0: Schlecht
# 0-1: Durchschnitt
# 1-2: Gut
# > 2: Exzellent
```

---

## 9️⃣ PRACTICAL FORMULAS FOR AI

### Complete Prediction Pipeline

```python
class BettingMath:
    """Alle mathematischen Funktionen für das AI System"""
    
    @staticmethod
    def implied_probability(odds: float) -> float:
        return 1 / odds
    
    @staticmethod
    def fair_odds(probability: float) -> float:
        return 1 / probability if probability > 0 else float('inf')
    
    @staticmethod
    def value(our_prob: float, odds: float) -> float:
        return (our_prob * odds) - 1
    
    @staticmethod
    def edge(our_prob: float, odds: float) -> float:
        implied = 1 / odds
        return (our_prob - implied) * 100
    
    @staticmethod
    def kelly(prob: float, odds: float, fraction: float = 0.25) -> float:
        b = odds - 1
        q = 1 - prob
        full_kelly = (prob * b - q) / b if b > 0 else 0
        return max(0, min(full_kelly * fraction, 0.10))  # Max 10%
    
    @staticmethod
    def ev(prob: float, odds: float, stake: float) -> float:
        return stake * ((prob * odds) - 1)
    
    @staticmethod
    def is_value_bet(our_prob: float, odds: float, min_edge: float = 0.05) -> bool:
        edge = our_prob - (1 / odds)
        return edge >= min_edge

# Nutzung im System:
math = BettingMath()

# Prüfe ob Value Bet:
our_probability = 0.75
market_odds = 1.40

print(f"Implied Prob: {math.implied_probability(market_odds):.1%}")
print(f"Fair Odds: {math.fair_odds(our_probability):.2f}")
print(f"Value: {math.value(our_probability, market_odds):.1%}")
print(f"Edge: {math.edge(our_probability, market_odds):.1f}%")
print(f"Kelly Stake: {math.kelly(our_probability, market_odds):.1%}")
print(f"Is Value: {math.is_value_bet(our_probability, market_odds)}")
```

---

## 📊 Quick Reference Card

```
╔═══════════════════════════════════════════════════════════════╗
║              BETTING MATHEMATICS CHEAT SHEET                   ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  IMPLIED PROBABILITY = 1 / Decimal Odds                       ║
║  FAIR ODDS = 1 / Our Probability                             ║
║  VALUE = (Our Prob × Odds) - 1                               ║
║  EDGE = Our Prob - Implied Prob                              ║
║  KELLY = (p × b - q) / b  where b = odds-1, q = 1-p         ║
║  EV = Stake × [(Prob × Odds) - 1]                           ║
║  ROI = (Profit / Total Staked) × 100                        ║
║                                                               ║
║  VALUE BET: Value > 0.05 (5%)                                ║
║  STAKE: 25% Kelly, max 5% bankroll                           ║
║  OPTIMAL ODDS: 1.30 - 1.70                                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

**📚 Dieses Knowledge File enthält alle mathematischen Grundlagen für profitable Soccer Betting.**
