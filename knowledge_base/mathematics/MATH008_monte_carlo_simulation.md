# Monte Carlo Simulations for Betting

## Concept
Run thousands of simulations using probability distributions

## Application: Season Simulation
```python
import random

def simulate_season(team_xg, opponent_xg, games=38):
    wins = draws = losses = 0
    for _ in range(games):
        team_goals = poisson_random(team_xg)
        opp_goals = poisson_random(opponent_xg)
        if team_goals > opp_goals:
            wins += 1
        elif team_goals == opp_goals:
            draws += 1
        else:
            losses += 1
    return wins * 3 + draws

# Run 10000 simulations
results = [simulate_season(1.8, 1.2) for _ in range(10000)]
avg_points = sum(results) / len(results)
```

## Use Cases
- Title probability
- Relegation odds
- Top 4 chances
