"""Betting engine for accumulator construction and bankroll management."""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.core.config import settings


class BettingEngine:
    """Engine for constructing accumulators and managing bankroll."""
    
    def __init__(
        self,
        initial_bankroll: float,
        target_quote: float = 1.40,
        min_probability: float = 0.72,
        max_stake_percentage: float = 2.0,
        stop_loss_percentage: float = 15.0,
    ):
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.target_quote = target_quote
        self.min_probability = min_probability
        self.max_stake_percentage = max_stake_percentage
        self.stop_loss_percentage = stop_loss_percentage
        
        self.bet_history: List[Dict] = []
    
    def find_value_bets(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify value bets from predictions.
        
        A bet has value if: researched_probability > implied_probability
        
        Args:
            predictions: List of prediction dictionaries with probabilities and odds
        
        Returns:
            List of value bet opportunities
        """
        value_bets = []
        
        for pred in predictions:
            # Check Over 1.5 market
            if "over_1_5_odds" in pred and "over_1_5_probability" in pred:
                over_1_5_value = self._check_value(
                    researched_prob=pred["over_1_5_probability"],
                    odds=pred["over_1_5_odds"],
                    min_prob=self.min_probability,
                )
                
                if over_1_5_value["has_value"]:
                    value_bets.append({
                        "match_id": pred.get("match_id"),
                        "match_info": f"{pred.get('home_team')} vs {pred.get('away_team')}",
                        "market": "over_1_5",
                        "odds": pred["over_1_5_odds"],
                        "researched_probability": pred["over_1_5_probability"],
                        "implied_probability": over_1_5_value["implied_probability"],
                        "expected_value": over_1_5_value["expected_value"],
                        "confidence": pred.get("confidence_score", 0.5),
                        "key_factors": pred.get("key_factors", []),
                    })
            
            # Check BTTS market
            if "btts_odds" in pred and "btts_probability" in pred:
                btts_value = self._check_value(
                    researched_prob=pred["btts_probability"],
                    odds=pred["btts_odds"],
                    min_prob=0.70,  # Lower threshold for BTTS
                )
                
                if btts_value["has_value"]:
                    value_bets.append({
                        "match_id": pred.get("match_id"),
                        "match_info": f"{pred.get('home_team')} vs {pred.get('away_team')}",
                        "market": "btts",
                        "odds": pred["btts_odds"],
                        "researched_probability": pred["btts_probability"],
                        "implied_probability": btts_value["implied_probability"],
                        "expected_value": btts_value["expected_value"],
                        "confidence": pred.get("confidence_score", 0.5),
                        "key_factors": pred.get("key_factors", []),
                    })
        
        logger.info(f"Found {len(value_bets)} value betting opportunities")
        return value_bets
    
    def build_accumulator(
        self,
        value_bets: List[Dict[str, Any]],
        num_selections: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """
        Build an accumulator targeting the target quote.
        
        Args:
            value_bets: List of value bet opportunities
            num_selections: Number of selections for accumulator (2-4 recommended)
        
        Returns:
            Accumulator details or None if no suitable combination found
        """
        if len(value_bets) < num_selections:
            logger.warning(f"Not enough value bets ({len(value_bets)}) for {num_selections}-leg accumulator")
            return None
        
        # Sort by expected value descending
        sorted_bets = sorted(value_bets, key=lambda x: x["expected_value"], reverse=True)
        
        best_accumulator = None
        best_score = 0
        
        # Try different combinations
        from itertools import combinations
        
        for combo in combinations(sorted_bets[:min(10, len(sorted_bets))], num_selections):
            total_odds = np.prod([bet["odds"] for bet in combo])
            combined_probability = np.prod([bet["researched_probability"] for bet in combo])
            
            # Check if close to target quote
            odds_diff = abs(total_odds - self.target_quote)
            
            # Score: prioritize combinations close to target with high probability
            score = combined_probability * 100 - odds_diff * 10
            
            if total_odds >= self.target_quote * 0.95 and total_odds <= self.target_quote * 1.10:
                if score > best_score:
                    best_score = score
                    best_accumulator = {
                        "selections": list(combo),
                        "total_odds": total_odds,
                        "combined_probability": combined_probability,
                        "num_selections": num_selections,
                        "expected_value": self._calculate_accumulator_ev(combo, total_odds),
                        "score": score,
                    }
        
        if best_accumulator:
            logger.info(
                f"Built {num_selections}-leg accumulator: "
                f"Odds {best_accumulator['total_odds']:.2f}, "
                f"Probability {best_accumulator['combined_probability']:.1%}"
            )
        
        return best_accumulator
    
    def calculate_stake(
        self,
        accumulator: Dict[str, Any],
        use_kelly: bool = False,
        kelly_fraction: float = 0.25,
    ) -> float:
        """
        Calculate optimal stake for accumulator.
        
        Args:
            accumulator: Accumulator details
            use_kelly: Whether to use Kelly Criterion
            kelly_fraction: Fraction of Kelly to use (conservative)
        
        Returns:
            Stake amount
        """
        if use_kelly:
            # Kelly Criterion: f = (bp - q) / b
            # where b = odds - 1, p = probability, q = 1 - p
            b = accumulator["total_odds"] - 1
            p = accumulator["combined_probability"]
            q = 1 - p
            
            kelly_stake = (b * p - q) / b
            kelly_stake = max(0, min(kelly_stake, 0.10))  # Cap at 10%
            
            stake_percentage = kelly_stake * kelly_fraction * 100
        else:
            # Fixed percentage
            stake_percentage = self.max_stake_percentage
        
        stake = self.bankroll * (stake_percentage / 100)
        
        logger.info(f"Calculated stake: €{stake:.2f} ({stake_percentage:.2f}% of bankroll)")
        return stake
    
    def place_bet(
        self,
        accumulator: Dict[str, Any],
        stake: float,
    ) -> Dict[str, Any]:
        """
        Record bet placement (simulation).
        
        Returns:
            Bet record
        """
        bet = {
            "bet_id": len(self.bet_history) + 1,
            "timestamp": datetime.utcnow(),
            "accumulator": accumulator,
            "stake": stake,
            "potential_return": stake * accumulator["total_odds"],
            "potential_profit": stake * (accumulator["total_odds"] - 1),
            "status": "pending",
        }
        
        self.bet_history.append(bet)
        logger.info(
            f"Bet placed: €{stake:.2f} @ {accumulator['total_odds']:.2f} "
            f"(potential return: €{bet['potential_return']:.2f})"
        )
        
        return bet
    
    def check_stop_loss(self) -> bool:
        """Check if stop-loss threshold is reached."""
        loss_percentage = (1 - self.bankroll / self.initial_bankroll) * 100
        
        if loss_percentage >= self.stop_loss_percentage:
            logger.warning(
                f"STOP LOSS TRIGGERED: {loss_percentage:.1f}% loss "
                f"(Bankroll: €{self.bankroll:.2f} from €{self.initial_bankroll:.2f})"
            )
            return True
        
        return False
    
    def update_bankroll(self, amount: float) -> None:
        """Update bankroll after bet result."""
        old_bankroll = self.bankroll
        self.bankroll += amount
        
        logger.info(f"Bankroll updated: €{old_bankroll:.2f} → €{self.bankroll:.2f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get betting statistics."""
        if not self.bet_history:
            return {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "pending": 0,
                "win_rate": 0,
                "profit_loss": 0,
                "roi": 0,
            }
        
        total = len(self.bet_history)
        wins = sum(1 for bet in self.bet_history if bet["status"] == "won")
        losses = sum(1 for bet in self.bet_history if bet["status"] == "lost")
        pending = sum(1 for bet in self.bet_history if bet["status"] == "pending")
        
        total_staked = sum(bet["stake"] for bet in self.bet_history)
        total_returns = sum(
            bet["potential_return"] for bet in self.bet_history 
            if bet["status"] == "won"
        )
        profit_loss = total_returns - total_staked
        roi = (profit_loss / total_staked * 100) if total_staked > 0 else 0
        
        return {
            "total_bets": total,
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": wins / (wins + losses) * 100 if (wins + losses) > 0 else 0,
            "profit_loss": profit_loss,
            "roi": roi,
            "current_bankroll": self.bankroll,
            "bankroll_change": ((self.bankroll / self.initial_bankroll) - 1) * 100,
        }
    
    @staticmethod
    def _check_value(
        researched_prob: float,
        odds: float,
        min_prob: float,
    ) -> Dict[str, Any]:
        """Check if bet has value."""
        implied_prob = 1 / odds
        has_value = researched_prob > implied_prob and researched_prob >= min_prob
        
        # Expected value = (probability × profit) - (1 - probability) × stake
        # Normalized to stake = 1
        expected_value = (researched_prob * (odds - 1)) - (1 - researched_prob)
        
        return {
            "has_value": has_value,
            "implied_probability": implied_prob,
            "probability_edge": researched_prob - implied_prob,
            "expected_value": expected_value,
        }
    
    @staticmethod
    def _calculate_accumulator_ev(selections: List[Dict], total_odds: float) -> float:
        """Calculate expected value for accumulator."""
        combined_prob = np.prod([s["researched_probability"] for s in selections])
        return (combined_prob * (total_odds - 1)) - (1 - combined_prob)
