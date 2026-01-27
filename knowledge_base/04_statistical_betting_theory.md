# Statistical Betting Theory - The Mathematics of Winning

## Expected Value (EV) - The Only Metric That Matters

### Definition
```
EV = (Probability of Winning × Amount Won) - (Probability of Losing × Amount Lost)
```

### Example
**Bet**: Bayern München Over 1.5 Goals @ 1.30 odds
- **Your Model**: 87% probability
- **Market Implied**: 76.9% (1/1.30)
- **Stake**: $10

```
EV = (0.87 × $3.00) - (0.13 × $10.00)
EV = $2.61 - $1.30
EV = +$1.31
```

**Interpretation**: On average, you win $1.31 per $10 bet (13.1% ROI).

### EV Thresholds for Betting
| EV | Decision |
|----|----------|
| <0% | NEVER BET (negative EV) |
| 0-3% | MARGINAL (not worth risk) |
| 3-5% | CONSIDER (if confident) |
| 5-10% | STRONG BET (professional tier) |
| >10% | EXCELLENT (rare, top 1%) |

---

## Implied Probability vs True Probability

### Market Odds → Implied Probability
```
Implied Probability = 1 / Decimal Odds
```

### Examples
| Decimal Odds | Implied Probability | Interpretation |
|--------------|---------------------|----------------|
| 1.20 | 83.3% | Heavy favorite |
| 1.50 | 66.7% | Moderate favorite |
| 2.00 | 50.0% | Even money |
| 3.00 | 33.3% | Underdog |
| 5.00 | 20.0% | Strong underdog |

### The Vig (Bookmaker's Edge)
Bookmakers build in profit margin (2-10% depending on market):

**Example: Man City vs Arsenal**
- Man City Win: 2.10 (47.6%)
- Draw: 3.50 (28.6%)
- Arsenal Win: 3.40 (29.4%)
- **Total**: 105.6% (5.6% vig)

**Fair Odds (without vig)**:
- Man City: 2.21 (45.2%)
- Draw: 3.69 (27.1%)
- Arsenal: 3.58 (27.9%)

### Value Betting Formula
```
Value = (Your Probability × Odds) - 1
```

**Example**:
- Your model: 52% chance Over 2.5 Goals
- Bookmaker odds: 2.20
- Value = (0.52 × 2.20) - 1 = 0.144 (14.4% edge)

**Rule**: Only bet if Value > 0.05 (5% minimum edge).

---

## Kelly Criterion - Optimal Stake Sizing

### Full Kelly Formula
```
f* = (bp - q) / b
```
Where:
- f* = Fraction of bankroll to bet
- b = Odds - 1 (net odds)
- p = Probability of winning
- q = Probability of losing (1 - p)

### Example
**Bet**: Liverpool Over 1.5 @ 1.60 odds
- Your probability: 70%
- Market probability: 62.5%
- Odds: 1.60 (b = 0.60)

```
f* = (0.60 × 0.70 - 0.30) / 0.60
f* = (0.42 - 0.30) / 0.60
f* = 0.12 / 0.60
f* = 0.20 (20% of bankroll)
```

**DANGER**: Full Kelly is **aggressive** (20% on single bet = high risk).

### Fractional Kelly (Recommended)
```
Fractional Kelly = f* × Fraction
```

**Common Fractions**:
- **Half Kelly**: 50% of f* (balance growth + safety)
- **Quarter Kelly**: 25% of f* (conservative, recommended for beginners)

**Example Above**:
- Full Kelly: 20%
- Half Kelly: 10%
- Quarter Kelly: 5%

**Professional Standard**: Use **25-50% fractional Kelly** (reduces variance).

---

## Standard Deviation & Variance

### Why Variance Matters
Even with +EV bets, you **will lose** 40-50% of the time. Variance is natural.

### Bankroll Fluctuations
**Example**: 100 bets at 60% win rate, $10 stake each
- **Expected Wins**: 60
- **Expected Losses**: 40
- **Standard Deviation**: ±10 wins (normal variance)

**Possible Outcomes**:
- **Lucky**: 70 wins (15% above expected)
- **Expected**: 60 wins (on target)
- **Unlucky**: 50 wins (10 below expected)

**Lesson**: Need **100+ bets** to reduce variance noise and prove statistical edge.

### Drawdowns (Losing Streaks)
**Normal Drawdowns** (even with edge):
- **10% drawdown**: 60% probability over 100 bets
- **20% drawdown**: 30% probability over 100 bets
- **30% drawdown**: 10% probability over 100 bets

**Professional Rule**: If drawdown >25%, **stop betting** and review system.

---

## ROI vs Win Rate - Understanding the Relationship

### ROI Formula
```
ROI = (Total Profit / Total Staked) × 100%
```

### Win Rate Requirements by Odds
| Average Odds | Break-Even Win Rate | Good Win Rate | Excellent Win Rate |
|--------------|---------------------|---------------|-------------------|
| 1.40 | 71.4% | 75% | 80% |
| 1.60 | 62.5% | 67% | 72% |
| 1.80 | 55.6% | 60% | 65% |
| 2.00 | 50.0% | 55% | 60% |
| 2.50 | 40.0% | 45% | 50% |

### Professional ROI Benchmarks
| Tier | ROI | Win Rate @ 1.80 Odds |
|------|-----|----------------------|
| Amateur | <2% | <57% |
| Semi-Pro | 2-5% | 57-60% |
| Professional | 5-10% | 60-65% |
| Top 10% | 8-12% | 63-68% |
| Top 1% | 10-15%+ | 65-70%+ |

**Reality Check**: Professional tipsters with **5-10% ROI** are in top 10%. **Don't expect 20% ROI** (unrealistic).

---

## Closing Line Value (CLV) - The Holy Grail

### What is CLV?
The difference between your bet odds and the odds at kickoff (closing line).

### Why CLV Matters
**Market at kickoff = sharpest** (most information, sharp bettors moved line).

**Positive CLV** = You beat sharp money (good predictor of long-term profit).
**Negative CLV** = You bet worse than closing (poor long-term predictor).

### Example
**Your Bet** (Monday): Liverpool Over 1.5 @ 1.60
**Closing Line** (Saturday kickoff): Liverpool Over 1.5 @ 1.50

**CLV** = 1.60 / 1.50 - 1 = **+6.7%**

**Interpretation**: Your bet was 6.7% better than market consensus (positive CLV).

### CLV Targets
| CLV | Quality |
|-----|---------|
| <0% | Bad (losing long-term) |
| 0-2% | Marginal |
| 2-5% | Good |
| 5-10% | Excellent (professional tier) |
| >10% | Exceptional (top 1%) |

**Professional Standard**: 60-70% of bets should have positive CLV.

---

## Sample Size - When Can You Trust Your Edge?

### Statistical Significance Formula
```
z = (Observed Win Rate - Expected Win Rate) / Standard Error
```

**Rule**: Need **z > 1.96** (95% confidence) to prove edge.

### Sample Sizes Required
| Confidence Level | Bets Required |
|------------------|---------------|
| 80% | 50 bets |
| 90% | 100 bets |
| 95% | 200 bets |
| 99% | 500 bets |

**Lesson**: Don't trust system with <100 bets. Variance too high.

---

## Poisson Distribution - Modeling Goals

### What is Poisson?
Probability distribution for rare events (like goals scored).

### Formula
```
P(x goals) = (λ^x × e^-λ) / x!
```
Where λ = expected goals (xG).

### Example: Team Expected to Score 1.8 Goals
| Goals | Probability |
|-------|-------------|
| 0 | 16.5% |
| 1 | 29.7% |
| 2 | 26.7% |
| 3 | 16.1% |
| 4+ | 11.0% |

**Betting Application**:
- Over 0.5: 83.5% (1 - 16.5%)
- Over 1.5: 53.8% (26.7% + 16.1% + 11.0%)
- Over 2.5: 27.1% (16.1% + 11.0%)

### Both Teams Combined (Independent Events)
**Team A xG**: 1.8
**Team B xG**: 1.4
**Total xG**: 3.2

**Over 2.5 Probability** (Poisson):
- Simulate all combinations (0-0, 1-0, 2-0, etc.)
- Sum probabilities where total >2.5
- **Result**: ~65% (compared to 62% market = 3% edge)

---

## Regression to the Mean - Avoid Recency Bias

### Concept
Extreme performances (very good or very bad) tend to **revert to average** over time.

### Example: Team on 5-Game Win Streak
- **Current Form**: 100% (5 wins)
- **True Quality**: 60% win rate (league average)
- **Next Match Expectation**: ~65% (weighted between current + true)

**Betting Mistake**: Betting on team because "they're on fire" (ignoring regression).

### Regression Formula (Weighted Average)
```
Predicted Win Rate = (α × Recent Form) + ((1 - α) × True Quality)
```
Where α = weight (0.3-0.5 typical).

**Example**:
- Recent form: 100% (5 wins)
- True quality: 60%
- α = 0.4 (40% weight on recent form)

```
Predicted = (0.4 × 1.00) + (0.6 × 0.60) = 0.76 (76%)
```

**Lesson**: Don't overreact to streaks. Weight long-term quality heavily.

---

## Correlation & Independence

### Correlated Bets (AVOID in Accumulators)
**Example**: Bayern Win + Over 2.5 Goals
- If Bayern wins 3-0, Over 2.5 also hits
- If Bayern loses 0-1, both lose
- **Correlation**: 70% (not independent)

**Accumulator Impact**: True odds ≠ multiplied odds (bookmaker wins).

### Independent Bets (OK in Accumulators)
**Example**: Bayern Win + Liverpool Win (different matches)
- Outcomes unrelated
- **Correlation**: ~0% (independent)

**Accumulator Impact**: True odds = multiplied odds (fair).

**Professional Rule**: Only combine **independent** bets in accumulators.

---

## Bankroll Management - Survival is Everything

### Fixed Staking (Conservative)
- **Method**: Bet same % on every bet (1-3%)
- **Pros**: Simple, low risk
- **Cons**: Doesn't maximize growth

### Kelly Criterion (Aggressive)
- **Method**: Bet proportional to edge (5-20%)
- **Pros**: Maximizes growth
- **Cons**: High variance, requires accurate probabilities

### Hybrid (Recommended)
- **Method**: Fractional Kelly (25-50%) with 5% max stake
- **Pros**: Balanced growth + safety
- **Cons**: Requires calculation

### Bankroll Rules
1. **Never bet >5%** on single bet (risk of ruin)
2. **Reserve 50%** of bankroll for drawdowns
3. **Stop betting** if bankroll drops >25%
4. **Withdraw profits** periodically (secure gains)

---

## Monte Carlo Simulation - Testing Your System

### What is Monte Carlo?
Simulating 10,000+ betting scenarios to test long-term outcomes.

### Example: Testing 100-Bet System
**Assumptions**:
- Win rate: 60%
- Average odds: 1.80
- Stake: 2% per bet
- Bankroll: $1000

**Simulation Runs**: 10,000
**Results**:
- Median Profit: +$180 (18% ROI)
- Best Case (95th percentile): +$450
- Worst Case (5th percentile): -$50
- Probability of Profit: 92%

**Lesson**: Even with edge, 8% chance of loss over 100 bets (variance).

---

## Statistical Betting Checklist

Before placing bet:
- [ ] EV > +5% (positive expected value)
- [ ] Kelly stake calculated (25-50% fractional)
- [ ] Stake <5% of bankroll
- [ ] Bet independent (if accumulator)
- [ ] Sample size sufficient (100+ bets tracked)
- [ ] CLV tracked (aim for positive CLV 60%+ of time)
- [ ] Regression to mean considered (don't chase streaks)
- [ ] Poisson probability calculated (for goal markets)

**Bottom Line**: Betting is a **marathon, not a sprint**. Edge compounds over 1000+ bets. Variance is natural. Survive long enough, and math wins.
