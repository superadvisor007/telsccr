# 💰 Bankroll Management & Staking Strategies

## Für AI System - Professionelles Geldmanagement

---

## 1️⃣ BANKROLL BASICS

### Was ist Bankroll Management?

```yaml
definition: >
  Systematische Verwaltung des Wettbudgets um langfristig profitabel
  zu sein und Totalverlust zu vermeiden.

key_principle: >
  Auch mit 60% Win Rate kann man bankrott gehen ohne
  korrektes Bankroll Management!

golden_rules:
  - Nur Geld verwenden, dessen Verlust man verkraften kann
  - Bankroll getrennt vom Alltags-Budget halten
  - Staking-Regeln IMMER befolgen (keine Emotionen!)
  - Niemals "Verlusten nachjagen"
```

### Bankroll Sizing

```python
def recommended_starting_bankroll(monthly_bets: int, avg_odds: float = 1.50) -> dict:
    """
    Empfohlene Startbankroll basierend auf Wettfrequenz
    """
    # Faustregel: Bankroll sollte 50-100 Einsätze überleben können
    
    unit_size_pct = 0.02  # 2% pro Wette
    units_needed = 50     # Minimum 50 Units
    
    if monthly_bets > 100:
        units_needed = 100  # Häufige Wetter brauchen mehr Buffer
    
    return {
        'minimum_units': units_needed,
        'recommended_units': units_needed * 1.5,
        'example': f"Bei €10 pro Unit: €{units_needed * 10} - €{int(units_needed * 1.5 * 10)} Bankroll"
    }

# Empfehlungen:
# Anfänger: €200-500 Bankroll (€2-5 Units)
# Intermediate: €500-2000 Bankroll (€5-20 Units)
# Advanced: €2000+ Bankroll (€20+ Units)
```

---

## 2️⃣ STAKING SYSTEME

### Flat Staking (EMPFOHLEN für Anfänger)

```python
class FlatStaking:
    """
    Einfachstes System: Immer gleicher Einsatz
    
    Vorteile:
    - Einfach zu befolgen
    - Minimiert Varianz
    - Psychologisch einfach
    
    Nachteile:
    - Nutzt Confidence nicht
    - Langsames Wachstum
    """
    
    def __init__(self, bankroll: float, stake_percent: float = 0.02):
        self.bankroll = bankroll
        self.stake_percent = stake_percent
    
    def get_stake(self, confidence: float = None) -> float:
        """Immer gleicher Prozentsatz"""
        return self.bankroll * self.stake_percent
    
    def update_bankroll(self, profit_loss: float):
        self.bankroll += profit_loss

# Nutzung:
# staking = FlatStaking(bankroll=1000, stake_percent=0.02)
# stake = staking.get_stake()  # = €20 (immer)
```

### Percentage Staking (EMPFOHLEN für Fortgeschrittene)

```python
class PercentageStaking:
    """
    Einsatz als Prozent der AKTUELLEN Bankroll
    
    Vorteile:
    - Automatische Anpassung bei Gewinnen/Verlusten
    - Bankrott theoretisch unmöglich
    
    Nachteile:
    - Einsätze schrumpfen bei Pechsträhnen
    """
    
    def __init__(self, bankroll: float, base_percent: float = 0.02):
        self.bankroll = bankroll
        self.base_percent = base_percent
    
    def get_stake(self, confidence: float = None) -> float:
        """Prozent der aktuellen Bankroll"""
        return self.bankroll * self.base_percent
    
    def update_bankroll(self, profit_loss: float):
        self.bankroll += profit_loss
        self.bankroll = max(self.bankroll, 0)

# Beispiel-Verlauf:
# Start: €1000, 2% = €20
# Nach Verlust: €980, 2% = €19.60
# Nach Gewinn: €1050, 2% = €21
```

### Kelly Criterion Staking (EMPFOHLEN für Experten)

```python
class KellyStaking:
    """
    Mathematisch optimale Staking-Strategie
    
    Vorteile:
    - Maximiert langfristiges Wachstum
    - Berücksichtigt Confidence und Odds
    
    Nachteile:
    - Erfordert präzise Wahrscheinlichkeitsschätzungen
    - Full Kelly ist zu volatil (Fractional nutzen!)
    """
    
    def __init__(self, bankroll: float, kelly_fraction: float = 0.25, max_stake_pct: float = 0.05):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction  # 25% Kelly = konservativ
        self.max_stake_pct = max_stake_pct    # Max 5% pro Wette
    
    def kelly_formula(self, probability: float, odds: float) -> float:
        """
        Kelly = (p * b - q) / b
        
        p = Wahrscheinlichkeit
        q = 1 - p
        b = Odds - 1
        """
        p = probability
        q = 1 - p
        b = odds - 1
        
        if b <= 0:
            return 0
        
        kelly = (p * b - q) / b
        return max(0, kelly)
    
    def get_stake(self, probability: float, odds: float) -> float:
        """Berechne optimalen Einsatz"""
        full_kelly = self.kelly_formula(probability, odds)
        fractional_kelly = full_kelly * self.kelly_fraction
        
        # Cap at maximum stake percentage
        stake_pct = min(fractional_kelly, self.max_stake_pct)
        
        return self.bankroll * stake_pct
    
    def update_bankroll(self, profit_loss: float):
        self.bankroll += profit_loss
        self.bankroll = max(self.bankroll, 0)

# Beispiel:
# Bankroll: €1000
# Probability: 65%
# Odds: 1.70
#
# Full Kelly = (0.65 * 0.70 - 0.35) / 0.70 = 15%
# 25% Fractional = 3.75%
# Stake = €1000 * 0.0375 = €37.50
```

### Confidence-Based Staking

```python
class ConfidenceStaking:
    """
    Einsatz variiert nach Confidence Level
    
    Praktischer Mittelweg zwischen Flat und Kelly
    """
    
    def __init__(self, bankroll: float):
        self.bankroll = bankroll
        
        # Staking Table basierend auf Confidence
        self.stake_table = {
            'ultra_high': 0.04,   # >75% Confidence: 4%
            'high': 0.03,         # 70-75%: 3%
            'medium': 0.02,       # 60-70%: 2%
            'low': 0.01,          # 55-60%: 1%
            'skip': 0.00          # <55%: Nicht wetten
        }
    
    def get_confidence_level(self, probability: float, edge: float) -> str:
        """Bestimme Confidence Level"""
        if probability >= 0.75 and edge >= 0.10:
            return 'ultra_high'
        elif probability >= 0.70 and edge >= 0.07:
            return 'high'
        elif probability >= 0.60 and edge >= 0.05:
            return 'medium'
        elif probability >= 0.55 and edge >= 0.03:
            return 'low'
        else:
            return 'skip'
    
    def get_stake(self, probability: float, odds: float) -> float:
        """Berechne Einsatz basierend auf Confidence"""
        implied = 1 / odds
        edge = probability - implied
        
        level = self.get_confidence_level(probability, edge)
        stake_pct = self.stake_table[level]
        
        return self.bankroll * stake_pct

# Beispiel:
# Probability 72%, Odds 1.50 (implied 66.7%)
# Edge = 72% - 66.7% = 5.3%
# Level = 'medium' (60-70% prob, 5%+ edge)
# Stake = Bankroll * 2%
```

---

## 3️⃣ ACCUMULATOR STAKING

### Accumulator Risk Management

```python
class AccumulatorStaking:
    """
    Spezielle Staking-Regeln für Akkumulatoren
    
    WICHTIG: Akkus haben höheres Risiko!
    """
    
    def __init__(self, bankroll: float, max_acca_pct: float = 0.01):
        self.bankroll = bankroll
        self.max_acca_pct = max_acca_pct  # Max 1% für Akkus
    
    def get_stake(self, num_legs: int, combined_odds: float) -> float:
        """
        Stake für Akkumulator
        
        Je mehr Legs, desto niedriger der Einsatz!
        """
        # Reduzierung pro zusätzlichem Leg
        leg_penalty = {
            2: 1.00,   # Double: voller Acca-Stake
            3: 0.75,   # Treble: 75%
            4: 0.50,   # 4-fold: 50%
            5: 0.33,   # 5-fold: 33%
            6: 0.20    # 6-fold: 20%
        }
        
        penalty = leg_penalty.get(num_legs, 0.10)
        stake_pct = self.max_acca_pct * penalty
        
        return self.bankroll * stake_pct
    
    def calculate_target_odds(self, legs: list) -> dict:
        """
        Berechne Target-Odds für profitable Akkus
        
        Optimal: 1.40 - 2.00 Combined Odds
        """
        combined_prob = 1.0
        combined_odds = 1.0
        
        for leg in legs:
            combined_prob *= leg['probability']
            combined_odds *= leg['odds']
        
        break_even = 1 / combined_prob
        
        return {
            'combined_prob': combined_prob,
            'combined_odds': combined_odds,
            'break_even_odds': break_even,
            'value': combined_odds / break_even,
            'is_value': combined_odds > break_even * 1.05
        }

# Empfehlungen für Akkus:
# - Max 4 Legs (Risiko steigt exponentiell!)
# - Target Odds: 1.40 - 2.00
# - Nur Value Bets kombinieren
# - Max 1% Bankroll pro Akku
```

### System Bets (Alternative zu Akkus)

```python
def system_bet_options(legs: list, bankroll: float) -> list:
    """
    System-Wetten als Alternative zu Akkumulatoren
    
    Erlaubt 1+ Fehler bei geringerem Gewinn
    """
    n = len(legs)
    options = []
    
    if n >= 3:
        options.append({
            'name': f'Trixie ({n}-fold)',
            'description': '3 Doubles + 1 Treble',
            'bets': 4,
            'stake_per_bet': bankroll * 0.005,
            'allows_errors': 1
        })
    
    if n >= 4:
        options.append({
            'name': f'Patent ({n}-fold)',
            'description': '3 Singles + 3 Doubles + 1 Treble',
            'bets': 7,
            'stake_per_bet': bankroll * 0.003,
            'allows_errors': 2
        })
    
    return options
```

---

## 4️⃣ VERLUSTBEGRENZUNG

### Stop-Loss Regeln

```python
class StopLossManager:
    """
    Automatische Verlustbegrenzung
    
    KRITISCH für langfristigen Erfolg!
    """
    
    def __init__(self, bankroll: float):
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        
        # Stop-Loss Levels
        self.daily_loss_limit = 0.05      # 5% pro Tag
        self.weekly_loss_limit = 0.15     # 15% pro Woche
        self.monthly_loss_limit = 0.30    # 30% pro Monat
        
        # Tracking
        self.daily_loss = 0
        self.weekly_loss = 0
        self.monthly_loss = 0
    
    def record_result(self, profit_loss: float):
        """Erfasse Wett-Ergebnis"""
        self.current_bankroll += profit_loss
        
        if profit_loss < 0:
            self.daily_loss += abs(profit_loss)
            self.weekly_loss += abs(profit_loss)
            self.monthly_loss += abs(profit_loss)
    
    def can_bet(self) -> dict:
        """Prüfe ob weitere Wetten erlaubt sind"""
        daily_pct = self.daily_loss / self.initial_bankroll
        weekly_pct = self.weekly_loss / self.initial_bankroll
        monthly_pct = self.monthly_loss / self.initial_bankroll
        
        return {
            'allowed': (daily_pct < self.daily_loss_limit and 
                       weekly_pct < self.weekly_loss_limit and
                       monthly_pct < self.monthly_loss_limit),
            'daily_remaining': self.daily_loss_limit - daily_pct,
            'weekly_remaining': self.weekly_loss_limit - weekly_pct,
            'monthly_remaining': self.monthly_loss_limit - monthly_pct,
            'reason': self._get_stop_reason(daily_pct, weekly_pct, monthly_pct)
        }
    
    def _get_stop_reason(self, daily, weekly, monthly) -> str:
        if daily >= self.daily_loss_limit:
            return 'DAILY_LIMIT_REACHED'
        if weekly >= self.weekly_loss_limit:
            return 'WEEKLY_LIMIT_REACHED'
        if monthly >= self.monthly_loss_limit:
            return 'MONTHLY_LIMIT_REACHED'
        return 'OK'
    
    def reset_daily(self):
        self.daily_loss = 0
    
    def reset_weekly(self):
        self.weekly_loss = 0
        self.daily_loss = 0
    
    def reset_monthly(self):
        self.monthly_loss = 0
        self.weekly_loss = 0
        self.daily_loss = 0
```

### Win Target

```python
class WinTargetManager:
    """
    Gewinnziele setzen (optional aber hilfreich)
    """
    
    def __init__(self, bankroll: float):
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        
        # Take Profit Levels
        self.daily_target = 0.03    # 3% pro Tag
        self.weekly_target = 0.10   # 10% pro Woche
    
    def check_targets(self) -> dict:
        """Prüfe ob Gewinnziele erreicht"""
        profit = self.current_bankroll - self.initial_bankroll
        profit_pct = profit / self.initial_bankroll
        
        return {
            'profit': profit,
            'profit_pct': profit_pct,
            'daily_target_reached': profit_pct >= self.daily_target,
            'weekly_target_reached': profit_pct >= self.weekly_target,
            'recommendation': 'CONSIDER_STOPPING' if profit_pct >= self.daily_target else 'CONTINUE'
        }
```

---

## 5️⃣ DRAWDOWN MANAGEMENT

### Drawdown Tracking

```python
class DrawdownTracker:
    """
    Überwache Drawdowns (Verlustphasen)
    """
    
    def __init__(self, bankroll: float):
        self.peak = bankroll
        self.current = bankroll
        self.history = [bankroll]
    
    def update(self, new_balance: float):
        self.current = new_balance
        self.history.append(new_balance)
        
        if new_balance > self.peak:
            self.peak = new_balance
    
    def get_drawdown(self) -> dict:
        """Berechne aktuellen Drawdown"""
        if self.peak == 0:
            return {'drawdown': 0, 'drawdown_pct': 0}
        
        drawdown = self.peak - self.current
        drawdown_pct = drawdown / self.peak
        
        return {
            'drawdown': drawdown,
            'drawdown_pct': drawdown_pct,
            'peak': self.peak,
            'current': self.current,
            'status': self._get_status(drawdown_pct)
        }
    
    def _get_status(self, dd_pct: float) -> str:
        if dd_pct < 0.10:
            return 'HEALTHY'
        elif dd_pct < 0.20:
            return 'CAUTION'
        elif dd_pct < 0.30:
            return 'WARNING'
        else:
            return 'CRITICAL'
    
    def max_drawdown(self) -> float:
        """Historischer maximaler Drawdown"""
        peak = self.history[0]
        max_dd = 0
        
        for value in self.history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd

# Empfehlungen bei Drawdown:
# 10-15%: Normal, weiter machen
# 15-25%: Stake um 50% reduzieren
# 25-35%: Stake um 75% reduzieren, Strategie prüfen
# >35%: Pause, komplette Analyse
```

---

## 6️⃣ STAKE SIZING TABLE

### Quick Reference

```python
STAKE_SIZING_TABLE = {
    'bankroll_500': {
        'flat_stake': 10,          # €10 = 2%
        'min_stake': 5,            # €5 = 1%
        'max_stake': 25,           # €25 = 5%
        'acca_stake': 5,           # €5 = 1%
    },
    'bankroll_1000': {
        'flat_stake': 20,
        'min_stake': 10,
        'max_stake': 50,
        'acca_stake': 10,
    },
    'bankroll_2500': {
        'flat_stake': 50,
        'min_stake': 25,
        'max_stake': 125,
        'acca_stake': 25,
    },
    'bankroll_5000': {
        'flat_stake': 100,
        'min_stake': 50,
        'max_stake': 250,
        'acca_stake': 50,
    }
}

def get_stake_for_bankroll(bankroll: float, bet_type: str, confidence: str = 'medium') -> float:
    """
    Quick Stake Lookup
    """
    base_pct = {
        'single_low': 0.01,
        'single_medium': 0.02,
        'single_high': 0.03,
        'single_ultra_high': 0.04,
        'accumulator': 0.01,
        'system_bet': 0.005
    }
    
    key = f"{bet_type}_{confidence}" if bet_type == 'single' else bet_type
    pct = base_pct.get(key, 0.02)
    
    return bankroll * pct
```

---

## 7️⃣ IMPLEMENTATION FÜR AI

```python
class BankrollManager:
    """Komplettes Bankroll Management für das AI System"""
    
    def __init__(self, initial_bankroll: float, strategy: str = 'confidence'):
        self.bankroll = initial_bankroll
        self.initial = initial_bankroll
        self.strategy = strategy
        
        # Initialize components
        self.stop_loss = StopLossManager(initial_bankroll)
        self.drawdown = DrawdownTracker(initial_bankroll)
        
        # Choose staking strategy
        if strategy == 'flat':
            self.staker = FlatStaking(initial_bankroll)
        elif strategy == 'kelly':
            self.staker = KellyStaking(initial_bankroll, kelly_fraction=0.25)
        else:
            self.staker = ConfidenceStaking(initial_bankroll)
    
    def get_stake(self, probability: float, odds: float) -> dict:
        """Berechne empfohlenen Einsatz"""
        
        # Check if betting is allowed
        bet_check = self.stop_loss.can_bet()
        if not bet_check['allowed']:
            return {
                'stake': 0,
                'allowed': False,
                'reason': bet_check['reason']
            }
        
        # Calculate stake
        if self.strategy == 'kelly':
            stake = self.staker.get_stake(probability, odds)
        else:
            stake = self.staker.get_stake(probability, odds)
        
        # Adjust for drawdown
        dd = self.drawdown.get_drawdown()
        if dd['status'] == 'WARNING':
            stake *= 0.50
        elif dd['status'] == 'CRITICAL':
            stake *= 0.25
        
        return {
            'stake': round(stake, 2),
            'allowed': True,
            'stake_pct': stake / self.bankroll,
            'drawdown_status': dd['status']
        }
    
    def record_bet(self, stake: float, odds: float, won: bool):
        """Erfasse Wett-Ergebnis"""
        if won:
            profit = stake * (odds - 1)
        else:
            profit = -stake
        
        self.bankroll += profit
        self.staker.update_bankroll(profit)
        self.stop_loss.record_result(profit)
        self.drawdown.update(self.bankroll)
        
        return {
            'profit_loss': profit,
            'new_bankroll': self.bankroll,
            'total_profit': self.bankroll - self.initial,
            'roi': (self.bankroll - self.initial) / self.initial * 100
        }
    
    def get_status(self) -> dict:
        """Aktueller Bankroll-Status"""
        return {
            'bankroll': self.bankroll,
            'initial': self.initial,
            'profit': self.bankroll - self.initial,
            'roi_pct': (self.bankroll - self.initial) / self.initial * 100,
            'drawdown': self.drawdown.get_drawdown(),
            'can_bet': self.stop_loss.can_bet(),
            'max_drawdown': self.drawdown.max_drawdown()
        }

# Nutzung:
# manager = BankrollManager(initial_bankroll=1000, strategy='confidence')
# stake_info = manager.get_stake(probability=0.68, odds=1.55)
# result = manager.record_bet(stake=20, odds=1.55, won=True)
# status = manager.get_status()
```

---

## 📊 Quick Reference Card

```
╔════════════════════════════════════════════════════════════════╗
║              BANKROLL MANAGEMENT CHEAT SHEET                   ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  STAKE SIZING:                                                 ║
║  - Low Confidence:     1% Bankroll                            ║
║  - Medium Confidence:  2% Bankroll                            ║
║  - High Confidence:    3% Bankroll                            ║
║  - Max Single Bet:     5% Bankroll                            ║
║  - Accumulator:        1% Bankroll                            ║
║                                                                ║
║  STOP-LOSS LIMITS:                                            ║
║  - Daily:   5% max loss                                       ║
║  - Weekly:  15% max loss                                      ║
║  - Monthly: 30% max loss                                      ║
║                                                                ║
║  DRAWDOWN ACTIONS:                                            ║
║  - <15%:   Normal                                             ║
║  - 15-25%: Reduce stakes by 50%                              ║
║  - 25-35%: Reduce stakes by 75%                              ║
║  - >35%:   STOP - Full review                                ║
║                                                                ║
║  KELLY (25% Fractional):                                      ║
║  - Optimal for value betting                                  ║
║  - Max 5% per bet                                             ║
║  - Requires accurate probabilities                            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

**📚 Dieses Knowledge File enthält alle Bankroll- und Staking-Strategien für nachhaltiges Wetten.**
