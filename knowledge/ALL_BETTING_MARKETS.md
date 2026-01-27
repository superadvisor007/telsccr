# 📚 Complete Soccer Betting Markets Encyclopedia

## Alle Wettarten für AI Prediction System

Dieses Knowledge File dient als Referenz für alle Wettmärkte,
die das System analysieren und empfehlen kann.

---

## 1️⃣ MAIN MARKETS (Hauptmärkte)

### 1.1 Match Result (1X2) - Spielausgang

**Was ist es?**
Wette auf den Ausgang des Spiels nach 90 Minuten + Nachspielzeit.

**Optionen:**
| Symbol | Bedeutung | Beispiel |
|--------|-----------|----------|
| 1 | Heimsieg | Bayern gewinnt |
| X | Unentschieden | 1-1, 2-2, etc. |
| 2 | Auswärtssieg | Gast gewinnt |

**Typische Quoten:**
```
Favorit zuhause:     1 @ 1.30-1.60
Unentschieden:       X @ 3.00-4.00
Underdog auswärts:   2 @ 3.50-8.00
```

**Wann setzen:**
- 1 (Home): Starker Favorit zuhause, Form >75%
- X (Draw): SELTEN - beide Teams defensiv
- 2 (Away): Top-Team auswärts, gute Auswärtsform

**AI Feature Requirements:**
- Elo Rating Differenz
- Home/Away Form (letzte 5)
- Head-to-Head History
- Motivation (Tabellenposition)

---

### 1.2 Double Chance (Doppelte Chance)

**Was ist es?**
Absicherung durch 2 von 3 möglichen Ergebnissen.

**Optionen:**
| Symbol | Bedeutung | Abgedeckte Ergebnisse |
|--------|-----------|----------------------|
| 1X | Heim gewinnt nicht verliert | Heimsieg ODER Unentschieden |
| 12 | Kein Unentschieden | Heimsieg ODER Auswärtssieg |
| X2 | Gast verliert nicht | Unentschieden ODER Auswärtssieg |

**Typische Quoten:**
```
1X (Favorit zuhause):  1.10-1.35
12 (Kein Draw):        1.15-1.45
X2 (Gast + Draw):      1.30-1.70
```

**Wann setzen:**
- 1X: Starker Heimfavorit absichern
- 12: Offensive Teams, wenig Draws erwartet
- X2: Starker Auswärtsfavorit absichern

**Perfekt für Akkumulatoren!** ⭐

---

### 1.3 Draw No Bet (Unentschieden = Einsatz zurück)

**Was ist es?**
Wie 1X2, aber bei Unentschieden wird der Einsatz erstattet.

**Optionen:**
| Symbol | Bedeutung | Bei Draw |
|--------|-----------|----------|
| 1 DNB | Heim gewinnt | Geld zurück |
| 2 DNB | Gast gewinnt | Geld zurück |

**Typische Quoten:**
```
DNB Favorit:    1.15-1.40
DNB Underdog:   2.00-4.00
```

**Wann setzen:**
- Wenn du an einen Sieg glaubst, aber Draw möglich ist
- Weniger Risiko als reines 1X2

---

## 2️⃣ GOAL MARKETS (Tor-Märkte)

### 2.1 Over/Under Goals (Über/Unter Tore)

**Was ist es?**
Wette auf die Gesamtzahl der Tore im Spiel.

#### Over 0.5 Goals (Mind. 1 Tor)
```
Bedeutung: Es fällt mindestens 1 Tor
Win: 1-0, 0-1, 2-2, 5-3, etc.
Lose: 0-0
Typical Odds: 1.03-1.10
Hit Rate: ~95%
⚠️ Zu niedrige Quote für Value!
```

#### Over 1.5 Goals (Mind. 2 Tore) ⭐ BEST BET
```
Bedeutung: Es fallen mindestens 2 Tore
Win: 2-0, 1-1, 3-1, 4-2, etc.
Lose: 0-0, 1-0, 0-1
Typical Odds: 1.18-1.45
Hit Rate: 70-82%
✅ EMPFOHLEN - Beste Balance Quote/Chance
```

#### Over 2.5 Goals (Mind. 3 Tore)
```
Bedeutung: Es fallen mindestens 3 Tore
Win: 2-1, 3-0, 2-2, 4-1, etc.
Lose: 0-0, 1-0, 1-1, 2-0
Typical Odds: 1.60-2.20
Hit Rate: 45-60%
✅ Gut für offensive Teams
```

#### Over 3.5 Goals (Mind. 4 Tore)
```
Bedeutung: Es fallen mindestens 4 Tore
Win: 3-1, 2-2, 4-0, 3-2, etc.
Lose: 2-1, 1-1, 3-0, 2-0, etc.
Typical Odds: 2.20-3.50
Hit Rate: 25-40%
⚠️ Nur bei sehr offensiven Teams!
```

#### Under 2.5 Goals (Max. 2 Tore)
```
Bedeutung: Es fallen maximal 2 Tore
Win: 0-0, 1-0, 0-1, 1-1, 2-0
Lose: 2-1, 3-0, 3-1, etc.
Typical Odds: 1.55-2.10
Hit Rate: 40-55%
✅ Für defensive Ligen (Serie A)
```

#### Under 3.5 Goals (Max. 3 Tore)
```
Bedeutung: Es fallen maximal 3 Tore
Win: 0-0 bis 2-1, 3-0, 1-2
Lose: 3-1, 2-2, 4-0, etc.
Typical Odds: 1.25-1.55
Hit Rate: 60-75%
✅ Sichere Option
```

**AI Feature Requirements:**
- Team Goals Scored Average
- Team Goals Conceded Average
- League Goal Average
- H2H Total Goals
- Current Form (Goals)

---

### 2.2 Both Teams To Score (BTTS)

**Was ist es?**
Wette ob beide Teams mindestens je 1 Tor schießen.

#### BTTS Yes (Beide treffen)
```
Bedeutung: Beide Teams schießen mind. 1 Tor
Win: 1-1, 2-1, 1-2, 3-2, etc.
Lose: 0-0, 1-0, 2-0, 0-3, etc.
Typical Odds: 1.55-2.00
Hit Rate: 48-58%
✅ Guter Value bei offenen Spielen
```

#### BTTS No (Mind. ein Team ohne Tor)
```
Bedeutung: Mind. ein Team schießt kein Tor
Win: 0-0, 1-0, 2-0, 0-1, 3-0, etc.
Lose: 1-1, 2-1, 3-2, etc.
Typical Odds: 1.65-2.20
Hit Rate: 42-52%
✅ Für defensive Teams / One-sided Matches
```

**AI Feature Requirements:**
- Both Teams Score Rate (Historical)
- Clean Sheet Rate (Home & Away)
- Attack vs Defense Strength
- Recent BTTS Patterns

---

### 2.3 Team Goals (Einzelne Team Tore)

#### Home Team Over 0.5/1.5/2.5 Goals
```
HT Over 0.5: Heim schießt mind. 1 Tor
HT Over 1.5: Heim schießt mind. 2 Tore
HT Over 2.5: Heim schießt mind. 3 Tore
```

#### Away Team Over 0.5/1.5/2.5 Goals
```
AT Over 0.5: Gast schießt mind. 1 Tor
AT Over 1.5: Gast schießt mind. 2 Tore
AT Over 2.5: Gast schießt mind. 3 Tore
```

**Wann setzen:**
- Top-Team gegen schwache Defensive
- Bayern/Real/City gegen Abstiegskandidaten

---

### 2.4 Exact Goals

```
Exactly 0 Goals: 0-0 only
Exactly 1 Goal: 1-0 or 0-1
Exactly 2 Goals: 2-0, 1-1, 0-2
Exactly 3 Goals: 3-0, 2-1, 1-2, 0-3
Exactly 4 Goals: etc.
```

**Typische Quoten:** 5.00-15.00
**⚠️ Hohes Risiko - Nur kleine Einsätze!**

---

## 3️⃣ HANDICAP MARKETS

### 3.1 European Handicap (EH)

**Was ist es?**
Virtuelle Tore werden vor Spielbeginn addiert/subtrahiert.

**Beispiel: Bayern (-1) vs Mainz (+1)**
```
Reales Ergebnis: 2-0 Bayern
Nach Handicap:   1-0 Bayern (-1 Tor)

Bayern -1: WIN (wenn Bayern mit 2+ Toren gewinnt)
Mainz +1:  LOSE (wenn Mainz mit 2+ Toren verliert)
Draw -1:   WIN (wenn Bayern mit genau 1 Tor gewinnt)
```

---

### 3.2 Asian Handicap (AH) ⭐

**Was ist es?**
Fortgeschrittenes Handicap-System ohne Draw-Option.

#### Full Ball Handicaps

```
AH -0.5: Team muss gewinnen (= DNB)
AH -1.0: Team muss mit 2+ gewinnen
AH -1.5: Team muss mit 2+ gewinnen
AH -2.0: Team muss mit 3+ gewinnen

AH +0.5: Team darf nicht verlieren (= DNB)
AH +1.0: Team verliert mit max 1 Tor (oder gewinnt/draw)
AH +1.5: Team verliert mit max 1 Tor
AH +2.0: Team verliert mit max 2 Toren
```

#### Quarter Ball Handicaps (Split Bet)

```
AH -0.25: Halbe Stake auf -0, Halbe auf -0.5
AH -0.75: Halbe Stake auf -0.5, Halbe auf -1.0
AH +0.25: Halbe Stake auf 0, Halbe auf +0.5
AH +0.75: Halbe Stake auf +0.5, Halbe auf +1.0
```

**Beispiel AH -0.75:**
- Team gewinnt: WIN
- Team gewinnt mit 1 Tor: Half Win
- Draw: Half Lose
- Team verliert: LOSE

**Typische Quoten:**
```
AH -0.5 Favorit:   1.70-1.95
AH +0.5 Underdog:  1.85-2.05
AH -1.0 Favorit:   2.00-2.30
AH +1.0 Underdog:  1.65-1.85
```

**AI Feature Requirements:**
- Goal Difference Expected
- Clean Sheet Probability
- Blowout Game Probability (3+ Goal Wins)

---

## 4️⃣ HALF-TIME MARKETS

### 4.1 Half-Time Result (HT 1X2)

```
Wette auf den Stand zur Halbzeit
1: Heim führt
X: Unentschieden
2: Gast führt
```

**Typische Quoten:**
```
HT 1 (Favorit):  1.80-2.50
HT X (Draw):     2.00-2.50
HT 2 (Underdog): 3.00-6.00
```

---

### 4.2 Half-Time / Full-Time (HT/FT)

**Was ist es?**
Kombination aus Halbzeit-Ergebnis und Endstand.

**9 Möglichkeiten:**
| HT/FT | Bedeutung |
|-------|-----------|
| 1/1 | Heim führt HT, Heim gewinnt |
| 1/X | Heim führt HT, Unentschieden |
| 1/2 | Heim führt HT, Gast gewinnt (Comeback!) |
| X/1 | Draw HT, Heim gewinnt |
| X/X | Draw HT, Draw |
| X/2 | Draw HT, Gast gewinnt |
| 2/1 | Gast führt HT, Heim gewinnt (Comeback!) |
| 2/X | Gast führt HT, Unentschieden |
| 2/2 | Gast führt HT, Gast gewinnt |

**Typische Quoten:**
```
1/1 (Favorit dominiert):    2.50-4.00
X/1 (Spätes Heim-Tor):      4.00-6.00
2/1 (Heim Comeback):        15.00-30.00
```

---

### 4.3 Half-Time Goals

```
HT Over 0.5: Mind. 1 Tor in 1. Hälfte
HT Over 1.5: Mind. 2 Tore in 1. Hälfte
HT Under 0.5: Keine Tore in 1. Hälfte
```

**Typische Quoten:**
```
HT Over 0.5:  1.30-1.50
HT Over 1.5:  2.20-3.00
HT Under 0.5: 2.50-3.50
```

---

## 5️⃣ PLAYER MARKETS

### 5.1 Anytime Goalscorer

```
Spieler schießt mindestens 1 Tor
Typical Odds: 2.00-6.00 (je nach Spieler)
```

### 5.2 First Goalscorer

```
Spieler schießt das 1. Tor
Typical Odds: 5.00-15.00
⚠️ Hohe Varianz!
```

### 5.3 Player Shots/Cards/Assists

```
Player Over 0.5 Shots on Target
Player To Be Booked
Player To Score & Assist
```

**⚠️ Diese Märkte sind NICHT für unser System empfohlen**
- Zu viel Varianz
- Schwer zu modellieren
- Verletzungs-/Aufstellungsrisiko

---

## 6️⃣ SPECIAL MARKETS

### 6.1 Correct Score

```
Exaktes Endergebnis vorhersagen
Win: Nur bei exaktem Ergebnis
Typical Odds: 6.00-50.00
```

**Beliebteste Ergebnisse:**
| Ergebnis | Wahrscheinlichkeit | Typical Odds |
|----------|-------------------|--------------|
| 1-1 | ~10% | 6.00-8.00 |
| 2-1 | ~8% | 7.00-9.00 |
| 1-0 | ~9% | 6.50-8.50 |
| 2-0 | ~7% | 8.00-10.00 |
| 0-0 | ~8% | 9.00-12.00 |

**❌ NICHT EMPFOHLEN - Zu viel Varianz!**

---

### 6.2 Clean Sheet

```
Team hält die Null (0 Gegentore)
Home Clean Sheet: Heim kassiert kein Tor
Away Clean Sheet: Gast kassiert kein Tor
```

**Typical Odds:**
```
Home CS (Favorit): 2.20-3.00
Away CS: 2.80-4.50
```

---

### 6.3 Win To Nil

```
Team gewinnt und kassiert kein Gegentor
Win: 1-0, 2-0, 3-0, etc.
Lose: 1-1, 2-1, 0-1, etc.
```

**Typical Odds:** 2.80-5.00

---

### 6.4 First/Last Goal

```
Welches Team schießt erstes/letztes Tor?
Home First Goal
Away First Goal
No Goal
```

---

## 7️⃣ LONG-TERM MARKETS (Futures)

### 7.1 League Winner

```
Wer wird Meister?
Bayern Meister: @1.30
Dortmund Meister: @5.00
```

### 7.2 Top 4 Finish

```
Team beendet in Top 4
Team Top 4: @1.80
```

### 7.3 Top Goalscorer

```
Wer wird Torschützenkönig?
Typical Odds: 5.00-20.00
```

**⚠️ NICHT für tägliches System - Zu langfristig**

---

## 🎯 EMPFOHLENE MÄRKTE FÜR AI SYSTEM

### Tier 1: Höchste Zuverlässigkeit ⭐⭐⭐
| Markt | Quote Range | Hit Rate | Empfehlung |
|-------|-------------|----------|------------|
| Over 1.5 Goals | 1.20-1.45 | 75-82% | ⭐ BESTE WAHL |
| Double Chance 1X | 1.15-1.35 | 70-80% | ⭐ Sehr sicher |
| Under 3.5 Goals | 1.30-1.55 | 65-75% | Sicher |

### Tier 2: Gute Zuverlässigkeit ⭐⭐
| Markt | Quote Range | Hit Rate | Empfehlung |
|-------|-------------|----------|------------|
| Over 2.5 Goals | 1.65-2.00 | 50-60% | Offensiv |
| BTTS Yes | 1.60-1.90 | 50-58% | Open Games |
| Home Win (1) | 1.35-1.70 | 45-55% | Starke Teams |

### Tier 3: Moderat ⭐
| Markt | Quote Range | Hit Rate | Empfehlung |
|-------|-------------|----------|------------|
| Away Win (2) | 1.60-2.20 | 35-45% | Auswärts-Favorit |
| Under 2.5 Goals | 1.65-2.00 | 40-55% | Defensiv |
| BTTS No | 1.70-2.10 | 42-50% | Clean Sheets |

### ❌ NICHT EMPFOHLEN
| Markt | Grund |
|-------|-------|
| Draw (X) | Zu unvorhersagbar (~28% Hit Rate) |
| Under 1.5 Goals | Zu selten (~18% Hit Rate) |
| Correct Score | Zu viel Varianz |
| Player Markets | Aufstellungs-Risiko |
| First Goalscorer | Extrem variabel |

---

## 📊 Feature Requirements Matrix

| Market | Elo | Form | H2H | Goals | Defense | xG |
|--------|-----|------|-----|-------|---------|-----|
| 1X2 | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| Over 1.5 | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Over 2.5 | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ |
| BTTS | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Double Chance | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ |
| Asian HC | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ |

---

## 🔧 Implementation Checklist

```python
SUPPORTED_MARKETS = {
    # Tier 1
    'over_1_5': True,      # ✅ Implemented
    'double_chance_1x': True,
    'under_3_5': True,
    
    # Tier 2  
    'over_2_5': True,      # ✅ Implemented
    'btts_yes': True,      # ✅ Implemented
    'home_win': True,
    
    # Tier 3
    'away_win': True,
    'under_2_5': True,
    'btts_no': True,
    
    # Future
    'asian_handicap': False,  # TODO
    'draw_no_bet': False,     # TODO
    'over_3_5': False,        # TODO
}
```

---

**📚 Dieses Knowledge File dient dem AI System als vollständige Referenz für alle Soccer Betting Märkte.**
