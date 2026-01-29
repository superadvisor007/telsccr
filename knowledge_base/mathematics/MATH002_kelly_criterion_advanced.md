# Advanced Kelly Criterion

## Standard Kelly
f* = (bp - q) / b

Where:
- f* = fraction of bankroll
- b = decimal odds - 1
- p = probability of winning
- q = 1 - p

## Fractional Kelly (Recommended)
```python
def fractional_kelly(probability, odds, fraction=0.25):
    b = odds - 1
    p = probability
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly * fraction)
```

## Kelly Fraction Guidelines
| Risk Tolerance | Fraction |
|----------------|----------|
| Conservative | 0.10 (10%) |
| Moderate | 0.25 (25%) |
| Aggressive | 0.50 (50%) |
| Full Kelly | 1.00 (100%) - NOT RECOMMENDED |

## Multi-Bet Kelly
For accumulators, multiply individual Kellys and reduce by additional 50%
