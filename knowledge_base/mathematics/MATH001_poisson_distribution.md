# Poisson Distribution in Soccer Betting

## Core Formula
P(X=k) = (λ^k * e^-λ) / k!

Where:
- λ = expected goals (from xG or historical average)
- k = actual goals
- e = Euler's number (2.71828)

## Application
```python
def poisson_prob(expected_goals, actual_goals):
    from math import exp, factorial
    return (expected_goals ** actual_goals * exp(-expected_goals)) / factorial(actual_goals)

# Example: Expected 2.5 goals, probability of exactly 3
prob = poisson_prob(2.5, 3)  # = 0.2138 (21.38%)
```

## Goal Market Calculations
| Expected | Over 2.5 | Under 2.5 | Over 1.5 |
|----------|----------|-----------|----------|
| 2.0 | 32.3% | 67.7% | 59.4% |
| 2.5 | 45.6% | 54.4% | 71.3% |
| 3.0 | 57.7% | 42.3% | 80.1% |
| 3.5 | 67.9% | 32.1% | 86.4% |
