#!/usr/bin/env python3
"""
🎯 BATTLE-TESTED CENTRAL PIPELINE ORCHESTRATOR
==============================================
Production-grade orchestration using battle-tested patterns from:
- datarootsio/your-best-bet: MLOps best practices
- FinQuant: Portfolio optimization patterns
- scikit-learn: TimeSeriesSplit for walk-forward

Design Principles:
1. NO LOOK-AHEAD BIAS: Strict temporal ordering
2. MODULAR: Each component is independently testable
3. FAIL-SAFE: Graceful degradation on errors
4. LOGGING: Full audit trail of all decisions
5. REPRODUCIBLE: Seed-controlled randomness
"""

import os
import sys
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
    precision_score, recall_score, f1_score
)

# Suppress warnings in production
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/orchestrator.log', mode='a')
    ]
)
logger = logging.getLogger('CentralOrchestrator')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


# ==============================================================================
# DATA CONTRACTS - Type-safe data structures
# ==============================================================================

@dataclass
class MatchData:
    """Standardized match representation."""
    match_id: str
    date: str
    home_team: str
    away_team: str
    league: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    home_form: float = 50.0
    away_form: float = 50.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    """Standardized prediction output."""
    match_id: str
    market: str
    probability: float
    confidence: float
    recommended_odds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = "v1.0"
    
    @property
    def fair_odds(self) -> float:
        return 1.0 / self.probability if self.probability > 0.01 else 100.0


@dataclass
class BettingDecision:
    """Betting decision with full audit trail."""
    match_id: str
    market: str
    decision: str  # BET, SKIP, AVOID
    stake: float
    odds: float
    edge: float
    expected_value: float
    confidence: float
    reasoning: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BacktestResult:
    """Walk-forward backtest results."""
    total_bets: int
    winning_bets: int
    losing_bets: int
    total_staked: float
    total_profit: float
    roi: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profitable_windows: int
    total_windows: int
    window_consistency: float


# ==============================================================================
# FEATURE ENGINEERING - Battle-tested features
# ==============================================================================

class FeatureEngineer:
    """
    Feature engineering following datarootsio/your-best-bet patterns.
    
    Features are designed to:
    1. Capture team strength (Elo)
    2. Capture recent form
    3. Capture league characteristics
    4. Avoid look-ahead bias
    """
    
    # League statistics (historical averages - no look-ahead)
    LEAGUE_STATS = {
        'Bundesliga': {'avg_goals': 3.1, 'home_adv': 0.12, 'volatility': 1.4},
        'Premier League': {'avg_goals': 2.8, 'home_adv': 0.10, 'volatility': 1.3},
        'La Liga': {'avg_goals': 2.6, 'home_adv': 0.15, 'volatility': 1.2},
        'Serie A': {'avg_goals': 2.5, 'home_adv': 0.13, 'volatility': 1.2},
        'Ligue 1': {'avg_goals': 2.7, 'home_adv': 0.11, 'volatility': 1.3},
        'Eredivisie': {'avg_goals': 3.0, 'home_adv': 0.08, 'volatility': 1.5},
        'Championship': {'avg_goals': 2.6, 'home_adv': 0.14, 'volatility': 1.4},
    }
    
    FEATURE_NAMES = [
        'home_elo', 'away_elo', 'elo_diff', 'elo_sum',
        'home_form', 'away_form', 'form_diff',
        'league_avg_goals', 'league_home_adv', 'league_volatility',
        'predicted_home_goals', 'predicted_away_goals', 'predicted_total',
        'elo_x_form', 'home_strength', 'away_strength',
        'match_competitiveness'
    ]
    
    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit scaler on training data only."""
        X = self._compute_features(df)
        self.scaler.fit(X)
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self._fitted:
            raise ValueError("FeatureEngineer must be fitted first!")
        X = self._compute_features(df)
        return self.scaler.transform(X)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def _compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute raw features before scaling."""
        features = []
        
        for _, row in df.iterrows():
            # Base Elo features
            home_elo = row.get('home_elo', 1500)
            away_elo = row.get('away_elo', 1500)
            elo_diff = home_elo - away_elo
            elo_sum = home_elo + away_elo
            
            # Form features
            home_form = row.get('home_form', 50.0)
            away_form = row.get('away_form', 50.0)
            form_diff = home_form - away_form
            
            # League features
            league = row.get('league', 'Mixed')
            stats = self.LEAGUE_STATS.get(league, {
                'avg_goals': 2.7, 'home_adv': 0.12, 'volatility': 1.3
            })
            
            # Predicted goals (from Elo-based model)
            home_strength = home_elo / 1500.0
            away_strength = away_elo / 1500.0
            pred_home = row.get('predicted_home_goals', 1.4 * home_strength)
            pred_away = row.get('predicted_away_goals', 1.3 * away_strength)
            pred_total = pred_home + pred_away
            
            # Interaction features
            elo_x_form = (elo_diff / 400) * (form_diff / 100)
            
            # Competitiveness (closer Elo = more competitive)
            competitiveness = 1.0 - (abs(elo_diff) / 800)
            
            features.append([
                home_elo, away_elo, elo_diff, elo_sum,
                home_form, away_form, form_diff,
                stats['avg_goals'], stats['home_adv'], stats['volatility'],
                pred_home, pred_away, pred_total,
                elo_x_form, home_strength, away_strength,
                competitiveness
            ])
        
        return np.array(features)


# ==============================================================================
# MODEL REGISTRY - Battle-tested models
# ==============================================================================

class ModelRegistry:
    """
    Model registry with battle-tested configurations.
    
    Models are selected based on:
    1. Performance on historical data
    2. Calibration quality
    3. Computational efficiency
    """
    
    @staticmethod
    def get_model(model_type: str = 'gradient_boosting') -> Pipeline:
        """Get a fresh model instance."""
        if model_type == 'gradient_boosting':
            return Pipeline([
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                ))
            ])
        elif model_type == 'random_forest':
            return Pipeline([
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ))
            ])
        elif model_type == 'logistic':
            return Pipeline([
                ('classifier', LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=42
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# ==============================================================================
# WALK-FORWARD BACKTESTER - No cheating!
# ==============================================================================

class WalkForwardEngine:
    """
    Walk-Forward Backtesting Engine.
    
    Implements strict temporal ordering to prevent look-ahead bias:
    1. Train on window [t-N, t]
    2. Predict on window [t+1, t+M]
    3. Roll forward by step_size
    4. Repeat
    
    This is the GOLD STANDARD for backtesting trading strategies.
    """
    
    def __init__(
        self,
        train_window: int = 500,
        test_window: int = 50,
        step_size: int = 50,
        min_edge: float = 0.02,  # Reduced: 2% minimum edge
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.10
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        
        # Tracking
        self.all_bets = []
        self.window_results = []
        self.equity_curve = []
    
    def run(
        self,
        df: pd.DataFrame,
        markets: List[str] = ['over_1_5', 'over_2_5', 'btts'],
        initial_bankroll: float = 1000.0,
        model_type: str = 'gradient_boosting',
        verbose: bool = True,
        min_prob: float = 0.55  # Minimum model probability to bet
    ) -> BacktestResult:
        """
        Execute walk-forward backtest.
        
        CRITICAL: Data must be sorted by date BEFORE calling this method!
        """
        # CRITICAL: Sort by date to ensure temporal ordering
        df = df.sort_values('date').reset_index(drop=True)
        
        if verbose:
            logger.info("="*80)
            logger.info("🔄 WALK-FORWARD BACKTESTING (NO LOOK-AHEAD BIAS)")
            logger.info("="*80)
            logger.info(f"📊 Total Matches: {len(df)}")
            logger.info(f"📅 Date Range: {df['date'].min()} → {df['date'].max()}")
            logger.info(f"🪟 Train Window: {self.train_window} matches")
            logger.info(f"🎯 Test Window: {self.test_window} matches")
            logger.info(f"👣 Step Size: {self.step_size} matches")
            logger.info(f"📈 Min Edge: {self.min_edge*100:.1f}%")
            logger.info(f"🎲 Kelly Fraction: {self.kelly_fraction}")
        
        # Initialize
        bankroll = initial_bankroll
        self.all_bets = []
        self.window_results = []
        self.equity_curve = [initial_bankroll]
        
        feature_engineer = FeatureEngineer()
        
        # Calculate windows
        max_start = len(df) - self.train_window - self.test_window
        num_windows = max(1, (max_start // self.step_size) + 1)
        
        if verbose:
            logger.info(f"🔢 Testing {num_windows} windows")
        
        for window_idx in range(num_windows):
            train_start = window_idx * self.step_size
            train_end = train_start + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, len(df))
            
            if test_end <= test_start:
                break
            
            # Split data - STRICT TEMPORAL ORDERING
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            if verbose:
                logger.info(f"\n📍 Window {window_idx + 1}/{num_windows}")
                logger.info(f"   Train: {train_df['date'].iloc[0]} → {train_df['date'].iloc[-1]} ({len(train_df)} matches)")
                logger.info(f"   Test:  {test_df['date'].iloc[0]} → {test_df['date'].iloc[-1]} ({len(test_df)} matches)")
            
            # Fit feature engineer on training data ONLY
            feature_engineer.fit(train_df)
            X_train = feature_engineer.transform(train_df)
            X_test = feature_engineer.transform(test_df)
            
            # Train models for each market
            window_start_bankroll = bankroll
            window_bets = []
            
            for market in markets:
                # Get target
                if market not in train_df.columns:
                    continue
                
                y_train = train_df[market].values
                y_test = test_df[market].values
                
                # Skip if no positive examples
                if y_train.sum() == 0 or (1 - y_train).sum() == 0:
                    continue
                
                # Train model
                model = ModelRegistry.get_model(model_type)
                try:
                    model.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(f"Failed to train {market}: {e}")
                    continue
                
                # Predict probabilities
                try:
                    probs = model.predict_proba(X_test)[:, 1]
                except Exception as e:
                    logger.warning(f"Failed to predict {market}: {e}")
                    continue
                
                # Get market probabilities from data
                market_prob_col = f"{market}_prob"
                if market_prob_col in test_df.columns:
                    market_probs = test_df[market_prob_col].values
                else:
                    # Default: assume 50% market probability
                    market_probs = np.full(len(test_df), 0.5)
                
                # Evaluate bets
                for i, (prob, actual, market_prob) in enumerate(zip(probs, y_test, market_probs)):
                    # Calculate market odds from market probability (with ~5% margin)
                    market_odds = 1.0 / market_prob if market_prob > 0.05 else 20.0
                    market_odds = market_odds * 0.95  # Bookmaker margin (vigorish)
                    implied_prob = 1.0 / market_odds
                    
                    # Calculate edge: our probability vs market implied probability
                    edge = prob - implied_prob
                    
                    # Value bet filter - only bet when we have edge AND high confidence
                    if edge >= self.min_edge and prob >= min_prob:
                        # Flat staking - more stable than Kelly for low-odds bets
                        # Use 2% of INITIAL bankroll (not current)
                        stake = initial_bankroll * 0.02  # Fixed $20 on $1000 bankroll
                        
                        # Don't bet more than 5% of current bankroll
                        stake = min(stake, bankroll * 0.05)
                        
                        # Minimum stake
                        if stake < 1.0 or bankroll < stake:
                            continue
                        
                        # Determine outcome
                        won = actual == 1
                        profit = (stake * market_odds - stake) if won else -stake
                        bankroll += profit
                        
                        # Record bet
                        bet = {
                            'window': window_idx + 1,
                            'date': test_df['date'].iloc[i],
                            'home_team': test_df['home_team'].iloc[i],
                            'away_team': test_df['away_team'].iloc[i],
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
                        self.all_bets.append(bet)
                        self.equity_curve.append(bankroll)
            
            # Window statistics
            if window_bets:
                window_profit = bankroll - window_start_bankroll
                window_roi = (window_profit / window_start_bankroll) * 100 if window_start_bankroll > 0 else 0
                window_wins = sum(1 for b in window_bets if b['won'])
                window_win_rate = window_wins / len(window_bets)
                
                self.window_results.append({
                    'window': window_idx + 1,
                    'bets': len(window_bets),
                    'wins': window_wins,
                    'win_rate': window_win_rate,
                    'profit': window_profit,
                    'roi': window_roi,
                    'bankroll': bankroll
                })
                
                if verbose:
                    logger.info(f"   💰 Bets: {len(window_bets)} | Win: {window_wins} ({window_win_rate*100:.1f}%) | P/L: ${window_profit:+.2f} ({window_roi:+.1f}%)")
        
        # Calculate final statistics
        return self._calculate_results(initial_bankroll, bankroll)
    
    def _calculate_results(
        self,
        initial_bankroll: float,
        final_bankroll: float
    ) -> BacktestResult:
        """Calculate comprehensive backtest statistics."""
        if not self.all_bets:
            return BacktestResult(
                total_bets=0, winning_bets=0, losing_bets=0,
                total_staked=0, total_profit=0, roi=0,
                win_rate=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, profitable_windows=0,
                total_windows=0, window_consistency=0
            )
        
        # Basic stats
        total_bets = len(self.all_bets)
        winning_bets = sum(1 for b in self.all_bets if b['won'])
        losing_bets = total_bets - winning_bets
        total_staked = sum(b['stake'] for b in self.all_bets)
        total_profit = final_bankroll - initial_bankroll
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        # Drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max() * 100
        
        # Returns for Sharpe/Sortino
        returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([0])
        
        # Sharpe Ratio (annualized, assuming 365 bets/year)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
        else:
            sharpe = 0
        
        # Sortino Ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1:
            downside_std = negative_returns.std()
            sortino = (returns.mean() / downside_std) * np.sqrt(365) if downside_std > 0 else 0
        else:
            sortino = sharpe
        
        # Window consistency
        profitable_windows = sum(1 for w in self.window_results if w['profit'] > 0)
        total_windows = len(self.window_results)
        window_consistency = profitable_windows / total_windows if total_windows > 0 else 0
        
        return BacktestResult(
            total_bets=total_bets,
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            total_staked=total_staked,
            total_profit=total_profit,
            roi=roi,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profitable_windows=profitable_windows,
            total_windows=total_windows,
            window_consistency=window_consistency
        )


# ==============================================================================
# CENTRAL ORCHESTRATOR - Brings it all together
# ==============================================================================

class CentralOrchestrator:
    """
    🎯 Central Pipeline Orchestrator
    
    Connects all components in a production-ready pipeline:
    
    Data → Features → Models → Predictions → Decisions → Telegram
                                    ↓
                            Walk-Forward Backtest
                                    ↓
                            Performance Reports
    
    Usage:
        orchestrator = CentralOrchestrator()
        
        # Run backtest
        result = orchestrator.run_backtest('data/historical/massive_training_data.csv')
        
        # Generate predictions
        predictions = orchestrator.predict_today()
        
        # Send to Telegram
        orchestrator.send_predictions(predictions)
    """
    
    def __init__(
        self,
        data_path: str = None,
        config: Dict[str, Any] = None
    ):
        self.data_path = data_path or str(PROJECT_ROOT / 'data/historical/massive_training_data.csv')
        self.config = config or self._default_config()
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.backtest_engine = WalkForwardEngine(
            train_window=self.config['train_window'],
            test_window=self.config['test_window'],
            step_size=self.config['step_size'],
            min_edge=self.config['min_edge'],
            kelly_fraction=self.config['kelly_fraction']
        )
        
        # State
        self._data = None
        self._models = {}
        self._last_backtest = None
        
        logger.info("🎯 Central Orchestrator initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'train_window': 500,
            'test_window': 50,
            'step_size': 50,
            'min_edge': 0.05,  # 5% minimum edge for quality
            'kelly_fraction': 0.25,
            'max_bet_fraction': 0.10,
            'markets': ['over_1_5', 'over_2_5'],  # BTTS removed (negative edge)
            'model_type': 'gradient_boosting',
            'initial_bankroll': 1000.0,
            'telegram_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
        }
    
    def load_data(self, path: str = None) -> pd.DataFrame:
        """Load and prepare data."""
        path = path or self.data_path
        
        logger.info(f"📊 Loading data from {path}")
        
        df = pd.read_csv(path)
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"   Loaded {len(df)} matches")
        logger.info(f"   Date range: {df['date'].min()} → {df['date'].max()}")
        logger.info(f"   Leagues: {df['league'].nunique()}")
        
        self._data = df
        return df
    
    def run_backtest(
        self,
        data_path: str = None,
        verbose: bool = True,
        min_prob: float = 0.55  # Minimum probability to place bet
    ) -> BacktestResult:
        """
        Run walk-forward backtest on historical data.
        
        This is the GOLD STANDARD test - no look-ahead bias!
        """
        # Load data
        df = self.load_data(data_path)
        
        # Run backtest
        result = self.backtest_engine.run(
            df=df,
            markets=self.config['markets'],
            initial_bankroll=self.config['initial_bankroll'],
            model_type=self.config['model_type'],
            verbose=verbose,
            min_prob=min_prob
        )
        
        self._last_backtest = result
        
        # Print results
        if verbose:
            self._print_backtest_results(result)
        
        return result
    
    def _print_backtest_results(self, result: BacktestResult):
        """Print formatted backtest results."""
        print("\n" + "="*80)
        print("📊 WALK-FORWARD BACKTEST RESULTS")
        print("="*80)
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                        BACKTEST SUMMARY                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Total Bets:         {result.total_bets:>10}                                    ║
║  Winning Bets:       {result.winning_bets:>10}                                    ║
║  Win Rate:           {result.win_rate*100:>9.1f}%                                    ║
║  Total Staked:       ${result.total_staked:>9.2f}                                    ║
║  Total Profit:       ${result.total_profit:>+9.2f}                                    ║
║  ROI:                {result.roi:>+9.2f}%                                    ║
║  Max Drawdown:       {result.max_drawdown:>9.1f}%                                    ║
║  Sharpe Ratio:       {result.sharpe_ratio:>9.2f}                                    ║
║  Sortino Ratio:      {result.sortino_ratio:>9.2f}                                    ║
║  Profitable Windows: {result.profitable_windows:>3}/{result.total_windows:<3} ({result.window_consistency*100:.1f}%)                          ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        
        # Verdict
        if result.roi > 5 and result.win_rate > 0.53 and result.window_consistency > 0.55:
            verdict = "✅ PROFITABLE - Ready for paper trading"
        elif result.roi > 0 and result.win_rate > 0.50:
            verdict = "⚠️ MARGINAL - Needs optimization"
        else:
            verdict = "❌ NOT PROFITABLE - Do not deploy"
        
        print(f"\n🎯 VERDICT: {verdict}")
        print("="*80)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'data_loaded': self._data is not None,
            'data_size': len(self._data) if self._data is not None else 0,
            'last_backtest': asdict(self._last_backtest) if self._last_backtest else None,
            'models_trained': len(self._models),
        }


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Run the central orchestrator."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🎯 BATTLE-TESTED CENTRAL PIPELINE ORCHESTRATOR                     ║
║                                                                      ║
║   Walk-Forward Backtesting • No Look-Ahead Bias • Production Ready   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Create orchestrator
    orchestrator = CentralOrchestrator()
    
    # Run backtest
    result = orchestrator.run_backtest()
    
    # Save results
    results_dir = PROJECT_ROOT / 'data' / 'backtest_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = results_dir / f'backtest_{timestamp}.json'
    
    with open(result_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {result_file}")
    
    return result


if __name__ == "__main__":
    main()
