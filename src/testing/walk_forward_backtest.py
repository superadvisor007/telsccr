"""
Walk-Forward Backtesting Framework
Prevents look-ahead bias through rolling window validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import sys
import os
import joblib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.features.advanced_features import EloRatingSystem, ValueBettingCalculator

class WalkForwardBacktester:
    """
    Rolling window backtest that simulates real trading:
    - Train on past N matches
    - Test on next M matches
    - Roll forward and repeat
    """
    
    def __init__(
        self,
        train_window: int = 500,
        test_window: int = 50,
        step_size: int = 50,
        min_edge: float = 0.08,
        kelly_fraction: float = 0.25
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.value_calculator = ValueBettingCalculator()
        
    def run_backtest(
        self,
        data: pd.DataFrame,
        model_class,
        markets: List[str],
        initial_bankroll: float = 1000.0
    ) -> Dict:
        """
        Execute walk-forward backtest
        
        Returns comprehensive results with per-window breakdown
        """
        print("\n" + "="*80)
        print("🔄 WALK-FORWARD BACKTESTING")
        print("="*80)
        print(f"📊 Total Data: {len(data)} matches")
        print(f"🪟 Train Window: {self.train_window} matches")
        print(f"🎯 Test Window: {self.test_window} matches")
        print(f"👣 Step Size: {self.step_size} matches")
        print(f"💰 Initial Bankroll: ${initial_bankroll:.2f}")
        print(f"🎲 Kelly Fraction: {self.kelly_fraction}")
        print(f"📈 Min Edge: {self.min_edge*100:.0f}%")
        print("="*80)
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Initialize tracking
        bankroll = initial_bankroll
        all_bets = []
        window_results = []
        
        # Calculate number of windows
        max_start = len(data) - self.train_window - self.test_window
        num_windows = (max_start // self.step_size) + 1
        
        print(f"\n🔢 Will test {num_windows} windows\n")
        
        for window_idx in range(num_windows):
            train_start = window_idx * self.step_size
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = test_start + self.test_window
            
            # Check bounds
            if test_end > len(data):
                break
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            print(f"📍 Window {window_idx + 1}/{num_windows}")
            print(f"   Train: {train_data['date'].min()} → {train_data['date'].max()} ({len(train_data)} matches)")
            print(f"   Test:  {test_data['date'].min()} → {test_data['date'].max()} ({len(test_data)} matches)")
            
            # Train models for this window
            models = {}
            for market in markets:
                model = model_class()
                try:
                    model.train_model(train_data, market, verbose=False)
                    models[market] = model
                except Exception as e:
                    print(f"   ⚠️  Failed to train {market}: {e}")
                    continue
            
            # Test on out-of-sample data
            window_bets = []
            window_start_bankroll = bankroll
            
            for idx, match in test_data.iterrows():
                # Generate predictions for each market
                for market in markets:
                    if market not in models:
                        continue
                    
                    try:
                        # Predict probability
                        features = self._extract_features(match, market)
                        prob = models[market].predict_proba(features)
                        
                        # Simulate market odds (inverse of fair probability + margin)
                        fair_odds = 1.0 / prob if prob > 0.01 else 100.0
                        market_odds = fair_odds * 1.05  # 5% bookmaker margin
                        
                        # Calculate edge
                        market_prob = 1.0 / market_odds
                        edge = prob - market_prob
                        
                        # Value bet filter
                        if edge >= self.min_edge and prob >= 0.55:
                            # Kelly stake
                            kelly_stake = self.value_calculator.kelly_criterion(
                                prob, market_odds, bankroll, self.kelly_fraction
                            )
                            
                            # Max 10% bankroll per bet
                            stake = min(kelly_stake, bankroll * 0.10)
                            
                            # Evaluate outcome
                            actual_outcome = self._evaluate_outcome(match, market)
                            won = actual_outcome == 1
                            
                            profit = (stake * market_odds - stake) if won else -stake
                            bankroll += profit
                            
                            bet = {
                                'window': window_idx + 1,
                                'date': match['date'],
                                'home_team': match['home_team'],
                                'away_team': match['away_team'],
                                'market': market,
                                'probability': prob,
                                'odds': market_odds,
                                'edge': edge,
                                'stake': stake,
                                'won': won,
                                'profit': profit,
                                'bankroll': bankroll
                            }
                            
                            window_bets.append(bet)
                            all_bets.append(bet)
                    
                    except Exception as e:
                        continue
            
            # Window statistics
            if window_bets:
                window_profit = bankroll - window_start_bankroll
                window_roi = (window_profit / window_start_bankroll) * 100
                window_win_rate = sum(1 for b in window_bets if b['won']) / len(window_bets)
                
                window_results.append({
                    'window': window_idx + 1,
                    'num_bets': len(window_bets),
                    'win_rate': window_win_rate,
                    'profit': window_profit,
                    'roi': window_roi,
                    'bankroll': bankroll
                })
                
                print(f"   💰 Bets: {len(window_bets)} | Win Rate: {window_win_rate*100:.1f}% | P/L: ${window_profit:+.2f} ({window_roi:+.1f}%)")
                print(f"   💼 Bankroll: ${bankroll:.2f}\n")
            else:
                print(f"   ⏭️  No value bets in this window\n")
        
        # Final statistics
        return self._calculate_final_stats(all_bets, window_results, initial_bankroll, bankroll)
    
    def _extract_features(self, match: pd.Series, market: str) -> np.ndarray:
        """Extract features for prediction with full feature engineering"""
        
        # Base features from data
        home_elo = match.get('home_elo', 1500)
        away_elo = match.get('away_elo', 1500)
        elo_diff = match.get('elo_diff', 0)
        predicted_home_goals = match.get('predicted_home_goals', 1.5)
        predicted_away_goals = match.get('predicted_away_goals', 1.3)
        predicted_total_goals = match.get('predicted_total_goals', 2.8)
        home_form = match.get('home_form', 50.0)
        away_form = match.get('away_form', 50.0)
        league = match.get('league', 'Mixed')
        
        # Engineer additional features (matching train_knowledge_enhanced_ml.py)
        elo_home_strength = home_elo / 1500  # Relative to baseline
        elo_away_strength = away_elo / 1500
        form_advantage = home_form - away_form
        
        # League-specific adjustments
        league_map = {
            'Bundesliga': {'avg_goals': 3.1, 'over_2_5_rate': 0.58, 'btts_rate': 0.52},
            'Premier League': {'avg_goals': 2.8, 'over_2_5_rate': 0.53, 'btts_rate': 0.48},
            'La Liga': {'avg_goals': 2.6, 'over_2_5_rate': 0.48, 'btts_rate': 0.45},
            'Serie A': {'avg_goals': 2.5, 'over_2_5_rate': 0.43, 'btts_rate': 0.42},
            'Ligue 1': {'avg_goals': 2.7, 'over_2_5_rate': 0.50, 'btts_rate': 0.46},
            'Eredivisie': {'avg_goals': 3.0, 'over_2_5_rate': 0.57, 'btts_rate': 0.54},
            'Championship': {'avg_goals': 2.6, 'over_2_5_rate': 0.48, 'btts_rate': 0.47},
            'Mixed': {'avg_goals': 2.8, 'over_2_5_rate': 0.52, 'btts_rate': 0.50}
        }
        
        league_stats = league_map.get(league, league_map['Mixed'])
        league_avg_goals = league_stats['avg_goals']
        league_over_2_5_rate = league_stats['over_2_5_rate']
        league_btts_rate = league_stats['btts_rate']
        
        # Interaction features
        elo_advantage = elo_diff / 400  # Normalized
        elo_total_strength = elo_home_strength + elo_away_strength
        elo_gap = abs(elo_home_strength - elo_away_strength)
        predicted_goals_diff = predicted_home_goals - predicted_away_goals
        elo_x_form = elo_advantage * form_advantage
        goals_x_league = predicted_total_goals * (league_avg_goals / 2.8)
        
        # Assemble feature vector (17 features - matching training)
        features = [
            home_elo,
            away_elo,
            elo_diff,
            predicted_home_goals,
            predicted_away_goals,
            predicted_total_goals,
            home_form,
            away_form,
            elo_home_strength,
            elo_away_strength,
            form_advantage,
            league_avg_goals,
            league_over_2_5_rate,
            league_btts_rate,
            elo_total_strength,
            elo_gap,
            predicted_goals_diff
        ]
        
        return np.array(features).reshape(1, -1)
    
    def _evaluate_outcome(self, match: pd.Series, market: str) -> int:
        """Check if prediction was correct"""
        home_goals = match['home_goals']
        away_goals = match['away_goals']
        total_goals = home_goals + away_goals
        
        if market == 'over_1_5':
            return 1 if total_goals > 1.5 else 0
        elif market == 'over_2_5':
            return 1 if total_goals > 2.5 else 0
        elif market == 'btts':
            return 1 if (home_goals > 0 and away_goals > 0) else 0
        elif market == 'under_1_5':
            return 1 if total_goals < 1.5 else 0
        elif market == 'under_2_5':
            return 1 if total_goals < 2.5 else 0
        else:
            return 0
    
    def _calculate_final_stats(
        self,
        all_bets: List[Dict],
        window_results: List[Dict],
        initial_bankroll: float,
        final_bankroll: float
    ) -> Dict:
        """Calculate comprehensive statistics"""
        
        if not all_bets:
            return {'error': 'No bets placed'}
        
        df_bets = pd.DataFrame(all_bets)
        df_windows = pd.DataFrame(window_results)
        
        # Overall metrics
        total_bets = len(all_bets)
        wins = sum(1 for b in all_bets if b['won'])
        losses = total_bets - wins
        win_rate = wins / total_bets
        
        total_staked = sum(b['stake'] for b in all_bets)
        total_profit = final_bankroll - initial_bankroll
        roi = (total_profit / initial_bankroll) * 100
        
        # Bet size stats
        avg_stake = df_bets['stake'].mean()
        max_stake = df_bets['stake'].max()
        
        # Profit stats
        avg_profit_per_bet = total_profit / total_bets
        best_bet = df_bets.loc[df_bets['profit'].idxmax()]
        worst_bet = df_bets.loc[df_bets['profit'].idxmin()]
        
        # Drawdown calculation
        bankroll_series = df_bets['bankroll']
        running_max = bankroll_series.expanding().max()
        drawdown = (bankroll_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Window consistency
        profitable_windows = sum(1 for w in window_results if w['profit'] > 0)
        window_consistency = profitable_windows / len(window_results)
        
        # Market breakdown
        market_stats = df_bets.groupby('market').agg({
            'won': ['count', 'sum', 'mean'],
            'profit': 'sum',
            'edge': 'mean'
        }).round(3)
        
        results = {
            'summary': {
                'total_bets': total_bets,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_staked': total_staked,
                'total_profit': total_profit,
                'roi': roi,
                'initial_bankroll': initial_bankroll,
                'final_bankroll': final_bankroll,
                'bankroll_change_pct': ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
            },
            'betting': {
                'avg_stake': avg_stake,
                'max_stake': max_stake,
                'avg_profit_per_bet': avg_profit_per_bet,
                'best_bet': {
                    'match': f"{best_bet['home_team']} vs {best_bet['away_team']}",
                    'market': best_bet['market'],
                    'profit': best_bet['profit']
                },
                'worst_bet': {
                    'match': f"{worst_bet['home_team']} vs {worst_bet['away_team']}",
                    'market': worst_bet['market'],
                    'profit': worst_bet['profit']
                }
            },
            'risk': {
                'max_drawdown_pct': max_drawdown,
                'sharpe_ratio': self._calculate_sharpe(df_bets['profit']),
                'sortino_ratio': self._calculate_sortino(df_bets['profit'])
            },
            'windows': {
                'total_windows': len(window_results),
                'profitable_windows': profitable_windows,
                'consistency_rate': window_consistency,
                'avg_window_roi': df_windows['roi'].mean(),
                'best_window_roi': df_windows['roi'].max(),
                'worst_window_roi': df_windows['roi'].min()
            },
            'markets': market_stats.to_dict(),
            'raw_bets': all_bets,
            'window_results': window_results
        }
        
        return results
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino Ratio (penalizes only downside volatility)"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)
    
    def print_results(self, results: Dict):
        """Pretty print backtest results"""
        if 'error' in results:
            print(f"\n❌ {results['error']}")
            return
        
        print("\n" + "="*80)
        print("📊 WALK-FORWARD BACKTEST RESULTS")
        print("="*80)
        
        s = results['summary']
        print(f"\n💼 BANKROLL:")
        print(f"   Initial:  ${s['initial_bankroll']:,.2f}")
        print(f"   Final:    ${s['final_bankroll']:,.2f}")
        print(f"   Change:   ${s['total_profit']:+,.2f} ({s['bankroll_change_pct']:+.2f}%)")
        
        print(f"\n🎯 BETTING PERFORMANCE:")
        print(f"   Total Bets:    {s['total_bets']}")
        print(f"   Wins:          {s['wins']} ({s['win_rate']*100:.1f}%)")
        print(f"   Losses:        {s['losses']}")
        print(f"   Total Staked:  ${s['total_staked']:,.2f}")
        print(f"   ROI:           {s['roi']:+.2f}%")
        
        b = results['betting']
        print(f"\n💰 STAKE STATISTICS:")
        print(f"   Avg Stake:     ${b['avg_stake']:.2f}")
        print(f"   Max Stake:     ${b['max_stake']:.2f}")
        print(f"   Avg P/L:       ${b['avg_profit_per_bet']:+.2f} per bet")
        
        r = results['risk']
        print(f"\n⚠️  RISK METRICS:")
        print(f"   Max Drawdown:  {r['max_drawdown_pct']:.2f}%")
        print(f"   Sharpe Ratio:  {r['sharpe_ratio']:.3f}")
        print(f"   Sortino Ratio: {r['sortino_ratio']:.3f}")
        
        w = results['windows']
        print(f"\n🪟 WINDOW CONSISTENCY:")
        print(f"   Total Windows:      {w['total_windows']}")
        print(f"   Profitable Windows: {w['profitable_windows']} ({w['consistency_rate']*100:.1f}%)")
        print(f"   Avg Window ROI:     {w['avg_window_roi']:+.2f}%")
        print(f"   Best Window:        {w['best_window_roi']:+.2f}%")
        print(f"   Worst Window:       {w['worst_window_roi']:+.2f}%")
        
        print("\n" + "="*80)
        
        # Verdict
        if s['roi'] > 10 and w['consistency_rate'] > 0.6:
            print("✅ EXCELLENT: Strong profitable system with consistency")
        elif s['roi'] > 5 and w['consistency_rate'] > 0.5:
            print("✅ GOOD: Profitable with acceptable consistency")
        elif s['roi'] > 0:
            print("⚠️  MARGINAL: Profitable but needs improvement")
        else:
            print("❌ UNPROFITABLE: System needs rework")
        
        print("="*80)


# Lightweight model wrapper for backtesting
class SimpleGradientBoostModel:
    """Simplified model for walk-forward testing"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def train_model(self, data: pd.DataFrame, market: str, verbose: bool = False):
        """Train on subset of data"""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Features
        feature_cols = [
            'home_elo', 'away_elo', 'elo_diff', 'elo_home_strength', 'elo_away_strength',
            'predicted_home_goals', 'predicted_away_goals', 'predicted_total_goals',
            'home_form', 'away_form', 'form_advantage', 'league_avg_goals',
            'league_over_2_5_rate', 'league_btts_rate', 'elo_total_strength',
            'elo_gap', 'predicted_goals_diff'
        ]
        
        X = data[feature_cols].fillna(0).values
        y = data[market].values
        
        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, features: np.ndarray) -> float:
        """Predict probability"""
        if self.model is None:
            return 0.5
        
        X_scaled = self.scaler.transform(features)
        prob = self.model.predict_proba(X_scaled)[0, 1]
        return prob


if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data/historical/massive_training_data.csv')
    print(f"📊 Loaded {len(data)} matches")
    
    # Run walk-forward backtest
    backtester = WalkForwardBacktester(
        train_window=500,
        test_window=50,
        step_size=50,
        min_edge=0.08,
        kelly_fraction=0.25
    )
    
    results = backtester.run_backtest(
        data=data,
        model_class=SimpleGradientBoostModel,
        markets=['over_1_5', 'over_2_5', 'btts'],
        initial_bankroll=1000.0
    )
    
    backtester.print_results(results)
    
    # Save results
    import json
    output_path = 'data/backtests/walk_forward_results.json'
    os.makedirs('data/backtests', exist_ok=True)
    
    # Remove raw bets for smaller file size
    results_clean = {k: v for k, v in results.items() if k != 'raw_bets'}
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_path}")
