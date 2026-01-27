"""Reinforcement Learning agent for optimal staking and bet selection."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Any, Dict, List, Optional, Tuple


class BettingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for soccer betting.
    
    State: [bankroll_normalized, recent_roi, avg_odds, prediction_confidence, 
            league_category, market_type]
    Action: [bet_or_not (0/1), stake_percentage (0-5%)]
    Reward: Profit/loss from bet, adjusted for risk (Sharpe ratio)
    """
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        max_stake_pct: float = 5.0,
        episode_length: int = 100,
    ):
        super().__init__()
        
        self.initial_bankroll = initial_bankroll
        self.max_stake_pct = max_stake_pct
        self.episode_length = episode_length
        
        # State space: [bankroll_norm, recent_roi, avg_odds, confidence, league_cat, market_type]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
            high=np.array([10.0, 1.0, 5.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action space: [bet_decision (0/1), stake_percentage]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, max_stake_pct]),
            dtype=np.float32
        )
        
        # Episode state
        self.bankroll = initial_bankroll
        self.step_count = 0
        self.bet_history: List[Dict] = []
        self.episode_returns: List[float] = []
        
        # Current bet opportunity
        self.current_bet: Optional[Dict] = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.bankroll = self.initial_bankroll
        self.step_count = 0
        self.bet_history = []
        self.episode_returns = []
        
        # Generate first bet opportunity
        self.current_bet = self._generate_bet_opportunity()
        
        state = self._get_state()
        return state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return next state, reward, terminated, truncated, info.
        
        Args:
            action: [bet_decision (0-1), stake_percentage (0-5)]
        """
        bet_decision = action[0]
        stake_pct = action[1]
        
        # Threshold for binary bet decision
        should_bet = bet_decision > 0.5
        
        reward = 0.0
        info = {}
        
        if should_bet and self.current_bet:
            # Calculate stake
            stake = self.bankroll * (stake_pct / 100.0)
            stake = min(stake, self.bankroll)  # Can't bet more than bankroll
            
            # Simulate bet outcome based on true probability
            bet_won = np.random.random() < self.current_bet['true_probability']
            
            if bet_won:
                profit = stake * (self.current_bet['odds'] - 1)
                self.bankroll += profit
                reward = profit / self.initial_bankroll  # Normalized reward
            else:
                self.bankroll -= stake
                reward = -stake / self.initial_bankroll
            
            # Risk-adjusted reward (penalize high variance)
            risk_penalty = stake_pct / 100.0  # Higher stakes = more risk
            reward = reward - (0.1 * risk_penalty)
            
            # Track bet
            self.bet_history.append({
                'step': self.step_count,
                'stake': stake,
                'odds': self.current_bet['odds'],
                'won': bet_won,
                'profit': profit if bet_won else -stake,
                'bankroll': self.bankroll,
            })
            
            info['bet_placed'] = True
            info['bet_won'] = bet_won
        else:
            info['bet_placed'] = False
        
        self.episode_returns.append(reward)
        self.step_count += 1
        
        # Generate next bet opportunity
        self.current_bet = self._generate_bet_opportunity()
        
        # Episode termination conditions
        terminated = self.bankroll <= 0  # Bust
        truncated = self.step_count >= self.episode_length
        
        # Bonus reward for maintaining/growing bankroll
        if truncated and self.bankroll > self.initial_bankroll:
            profit_bonus = (self.bankroll - self.initial_bankroll) / self.initial_bankroll
            reward += profit_bonus
        
        next_state = self._get_state()
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Bankroll normalized to initial
        bankroll_norm = self.bankroll / self.initial_bankroll
        
        # Recent ROI (last 10 bets)
        recent_bets = self.bet_history[-10:] if len(self.bet_history) >= 10 else self.bet_history
        if recent_bets:
            recent_roi = sum(b['profit'] for b in recent_bets) / (sum(b['stake'] for b in recent_bets) or 1)
        else:
            recent_roi = 0.0
        
        # Current bet features
        avg_odds = self.current_bet['odds'] if self.current_bet else 1.5
        confidence = self.current_bet['confidence'] if self.current_bet else 0.5
        league_cat = self.current_bet['league_category'] if self.current_bet else 0.5
        market_type = self.current_bet['market_type'] if self.current_bet else 0.5
        
        state = np.array([
            bankroll_norm,
            np.clip(recent_roi, -1.0, 1.0),
            avg_odds,
            confidence,
            league_cat,
            market_type,
        ], dtype=np.float32)
        
        return state
    
    def _generate_bet_opportunity(self) -> Dict[str, Any]:
        """Generate a random betting opportunity for simulation."""
        # Simulate different quality bets
        quality = np.random.choice(['good', 'medium', 'bad'], p=[0.3, 0.5, 0.2])
        
        if quality == 'good':
            odds = np.random.uniform(1.15, 1.30)
            true_prob = np.random.uniform(0.80, 0.90)
            confidence = np.random.uniform(0.75, 0.95)
        elif quality == 'medium':
            odds = np.random.uniform(1.25, 1.50)
            true_prob = np.random.uniform(0.65, 0.80)
            confidence = np.random.uniform(0.60, 0.75)
        else:  # bad
            odds = np.random.uniform(1.40, 1.80)
            true_prob = np.random.uniform(0.50, 0.65)
            confidence = np.random.uniform(0.40, 0.60)
        
        return {
            'odds': odds,
            'true_probability': true_prob,
            'confidence': confidence,
            'league_category': np.random.random(),  # 0=low scoring, 1=high scoring
            'market_type': np.random.random(),  # 0=over_1_5, 1=btts
        }
    
    def render(self) -> None:
        """Render environment state (optional)."""
        logger.info(f"Step {self.step_count}: Bankroll €{self.bankroll:.2f}, Bets: {len(self.bet_history)}")


class RewardLoggingCallback(BaseCallback):
    """Callback for logging training metrics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Log episode rewards
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                if self.verbose > 0:
                    logger.info(
                        f"Episode {len(self.episode_rewards)}: "
                        f"Reward={info['episode']['r']:.3f}, "
                        f"Length={info['episode']['l']}"
                    )
        
        return True


class RLStakingAgent:
    """Reinforcement Learning agent for optimal staking decisions."""
    
    def __init__(
        self,
        model_path: str = "models/rl_agent",
        initial_bankroll: float = 1000.0,
    ):
        self.model_path = model_path
        self.initial_bankroll = initial_bankroll
        self.model: Optional[PPO] = None
        self.env: Optional[BettingEnvironment] = None
    
    def train(
        self,
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
    ) -> None:
        """Train the RL agent."""
        logger.info("Training RL staking agent...")
        
        # Create environment
        self.env = BettingEnvironment(initial_bankroll=self.initial_bankroll)
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Initialize PPO agent
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="logs/rl_agent",
        )
        
        # Train with callback
        callback = RewardLoggingCallback(verbose=1)
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
        
        # Save model
        self.model.save(self.model_path)
        logger.info(f"RL agent trained and saved to {self.model_path}")
    
    def load(self) -> None:
        """Load trained model."""
        try:
            self.model = PPO.load(self.model_path)
            logger.info(f"RL agent loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load RL agent: {e}")
    
    def decide_stake(
        self,
        bankroll: float,
        recent_roi: float,
        odds: float,
        confidence: float,
        league_category: float = 0.5,
        market_type: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Use RL agent to decide whether to bet and how much.
        
        Returns:
            (should_bet, stake_percentage)
        """
        if self.model is None:
            logger.warning("RL model not loaded, using default staking")
            return True, 2.0  # Default 2%
        
        # Construct state
        state = np.array([
            bankroll / self.initial_bankroll,
            np.clip(recent_roi, -1.0, 1.0),
            odds,
            confidence,
            league_category,
            market_type,
        ], dtype=np.float32)
        
        # Get action from model
        action, _ = self.model.predict(state, deterministic=True)
        
        bet_decision = action[0] > 0.5
        stake_pct = float(action[1])
        
        logger.debug(f"RL decision: Bet={bet_decision}, Stake={stake_pct:.2f}%")
        
        return bet_decision, stake_pct
    
    def backtest(self, historical_bets: List[Dict]) -> Dict[str, float]:
        """
        Backtest RL agent on historical bet data.
        
        Args:
            historical_bets: List of dicts with 'odds', 'won', 'confidence', etc.
        
        Returns:
            Performance metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        bankroll = self.initial_bankroll
        total_staked = 0
        total_profit = 0
        bets_placed = 0
        bets_won = 0
        
        for bet in historical_bets:
            # Get RL decision
            should_bet, stake_pct = self.decide_stake(
                bankroll=bankroll,
                recent_roi=0.0,  # Simplified
                odds=bet['odds'],
                confidence=bet.get('confidence', 0.7),
            )
            
            if should_bet:
                stake = bankroll * (stake_pct / 100.0)
                total_staked += stake
                bets_placed += 1
                
                if bet['won']:
                    profit = stake * (bet['odds'] - 1)
                    bankroll += profit
                    total_profit += profit
                    bets_won += 1
                else:
                    bankroll -= stake
                    total_profit -= stake
        
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        win_rate = (bets_won / bets_placed * 100) if bets_placed > 0 else 0
        
        return {
            'final_bankroll': bankroll,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': roi,
            'bets_placed': bets_placed,
            'win_rate': win_rate,
        }
