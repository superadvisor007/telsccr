"""
🎟️ Multi-Bet Ticket Builder
============================
Integrates goal-directed reasoning with ML predictions and Telegram output.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# TICKET CONFIGURATION
# =============================================================================

@dataclass
class TicketConfig:
    """Configuration for ticket building."""
    # Odds constraints
    min_single_odds: float = 1.40
    max_single_odds: float = 1.70
    target_total_odds: float = 10.0
    
    # Selection constraints
    min_legs: int = 3
    max_legs: int = 6
    min_confidence: float = 0.65
    min_edge: float = 0.05
    
    # Staking
    base_stake: float = 50.0
    max_stake: float = 100.0
    kelly_fraction: float = 0.25
    
    # Markets (priority order)
    enabled_markets: List[str] = field(
        default_factory=lambda: ['over_1_5', 'over_2_5']  # BTTS excluded
    )
    
    # Diversification
    max_legs_per_league: int = 2
    max_legs_per_market: int = 3


@dataclass
class EnhancedBetLeg:
    """Enhanced bet leg with reasoning and ML support."""
    # Match info
    match_id: str
    home_team: str
    away_team: str
    league: str
    kickoff: str
    
    # Bet info
    market: str
    tip: str
    odds: float
    
    # Probabilities
    ml_probability: float
    reasoning_probability: float
    blended_probability: float
    
    # Quality metrics
    confidence: float
    edge: float
    value_score: float
    
    # Reasoning
    tactical_reasoning: str
    scenario_reasoning: str
    structural_justification: str
    
    # Source tracking
    source: str = 'combined'  # ml, reasoning, combined
    
    @property
    def is_value(self) -> bool:
        return self.edge > 0.05 and self.confidence > 0.65
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def format_for_telegram(self) -> str:
        """Format for Telegram display."""
        emoji = '⚽' if 'btts' in self.market else '🥅'
        market_display = {
            'over_1_5': 'Over 1.5 Goals',
            'over_2_5': 'Over 2.5 Goals',
            'btts': 'BTTS',
            'btts_yes': 'BTTS Yes',
            'btts_no': 'BTTS No'
        }.get(self.market, self.market.upper())
        
        return (
            f"{emoji} {self.home_team} vs {self.away_team}\n"
            f"   📊 {market_display}: {self.tip}\n"
            f"   💰 Odds: {self.odds:.2f} | Conf: {self.confidence:.0%}\n"
            f"   📈 Edge: {self.edge:+.1%} | Value: {self.value_score:.2f}"
        )


@dataclass
class EnhancedTicket:
    """Enhanced multi-bet ticket with full reasoning."""
    # Core
    ticket_id: str
    legs: List[EnhancedBetLeg]
    
    # Aggregates
    total_odds: float
    combined_probability: float
    expected_value: float
    
    # Staking
    stake: float
    potential_win: float
    kelly_stake: float
    
    # Quality
    overall_confidence: float
    overall_edge: float
    diversification_score: float
    
    # Meta
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = 'goal_directed_system'
    
    @classmethod
    def from_legs(
        cls,
        legs: List[EnhancedBetLeg],
        base_stake: float = 50.0
    ) -> 'EnhancedTicket':
        """Build ticket from legs."""
        if not legs:
            raise ValueError("Cannot create ticket with no legs")
        
        # Calculate aggregates
        total_odds = 1.0
        combined_prob = 1.0
        total_conf = 0.0
        total_edge = 0.0
        
        leagues = set()
        markets = {}
        
        for leg in legs:
            total_odds *= leg.odds
            combined_prob *= leg.blended_probability
            total_conf += leg.confidence
            total_edge += leg.edge
            
            leagues.add(leg.league)
            markets[leg.market] = markets.get(leg.market, 0) + 1
        
        avg_conf = total_conf / len(legs)
        avg_edge = total_edge / len(legs)
        
        # Calculate diversification
        league_diversity = len(leagues) / len(legs)
        market_diversity = len(markets) / len(legs)
        diversification = (league_diversity + market_diversity) / 2
        
        # Calculate stakes
        potential_win = base_stake * total_odds
        ev = combined_prob * potential_win - base_stake
        
        # Kelly stake
        if avg_edge > 0:
            kelly_fraction = min(0.25, avg_edge / (total_odds - 1))
            kelly_stake = base_stake * kelly_fraction * 10  # Scale
        else:
            kelly_stake = base_stake * 0.5
        
        ticket_id = f"TS-{datetime.now().strftime('%Y%m%d%H%M')}-{len(legs)}L-{int(total_odds*100)}"
        
        return cls(
            ticket_id=ticket_id,
            legs=legs,
            total_odds=round(total_odds, 2),
            combined_probability=round(combined_prob, 4),
            expected_value=round(ev, 2),
            stake=base_stake,
            potential_win=round(potential_win, 2),
            kelly_stake=round(kelly_stake, 2),
            overall_confidence=round(avg_conf, 3),
            overall_edge=round(avg_edge, 3),
            diversification_score=round(diversification, 3)
        )
    
    def to_dict(self) -> Dict:
        return {
            'ticket_id': self.ticket_id,
            'legs': [l.to_dict() for l in self.legs],
            'total_odds': self.total_odds,
            'combined_probability': self.combined_probability,
            'expected_value': self.expected_value,
            'stake': self.stake,
            'potential_win': self.potential_win,
            'kelly_stake': self.kelly_stake,
            'overall_confidence': self.overall_confidence,
            'overall_edge': self.overall_edge,
            'diversification_score': self.diversification_score,
            'created_at': self.created_at,
            'source': self.source
        }
    
    def format_for_telegram(self) -> str:
        """Format complete ticket for Telegram."""
        lines = [
            "🎟️ TELEGRAMSOCCER BETTING TICKET",
            f"━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📋 ID: {self.ticket_id}",
            f"📅 Created: {self.created_at[:16].replace('T', ' ')}",
            "",
            "🎯 SELECTIONS:",
            "─────────────────────"
        ]
        
        for i, leg in enumerate(self.legs, 1):
            lines.append(f"\n{i}. {leg.format_for_telegram()}")
        
        lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            "📊 TICKET SUMMARY:",
            f"   🎰 Total Odds: {self.total_odds:.2f}",
            f"   💰 Stake: €{self.stake:.2f}",
            f"   🏆 Potential Win: €{self.potential_win:.2f}",
            f"   📈 EV: {self.expected_value:+.2f}",
            f"   🎯 Confidence: {self.overall_confidence:.0%}",
            f"   ⚡ Edge: {self.overall_edge:+.1%}",
            f"   🌐 Diversification: {self.diversification_score:.0%}",
            "",
            "⚠️ Gamble responsibly | 18+ only",
            "━━━━━━━━━━━━━━━━━━━━━━━━"
        ])
        
        return "\n".join(lines)


# =============================================================================
# MULTI-BET TICKET BUILDER
# =============================================================================

class MultiBetTicketBuilder:
    """
    🎟️ Multi-Bet Ticket Builder
    
    Integrates:
    - Goal-directed reasoning analysis
    - ML model predictions
    - Walk-forward validated parameters
    - Odds constraints
    - Diversification rules
    """
    
    def __init__(self, config: TicketConfig = None):
        self.config = config or TicketConfig()
        self.reasoning_engine = None
        self.ml_model = None
        
        logger.info("🎟️ MultiBetTicketBuilder initialized")
    
    def set_reasoning_engine(self, engine):
        """Set goal-directed reasoning engine."""
        self.reasoning_engine = engine
    
    def set_ml_model(self, model):
        """Set ML model for predictions."""
        self.ml_model = model
    
    def _calculate_implied_probability(self, odds: float) -> float:
        """Calculate implied probability from odds."""
        return 1 / odds if odds > 0 else 0
    
    def _calculate_odds_from_probability(self, prob: float, margin: float = 0.05) -> float:
        """Calculate market odds from probability with margin."""
        if prob <= 0 or prob >= 1:
            return 0
        fair_odds = 1 / prob
        return fair_odds * (1 - margin)
    
    def _blend_probabilities(
        self,
        ml_prob: float,
        reasoning_prob: float,
        ml_weight: float = 0.6
    ) -> float:
        """Blend ML and reasoning probabilities."""
        return ml_prob * ml_weight + reasoning_prob * (1 - ml_weight)
    
    def _score_leg_value(self, leg: EnhancedBetLeg) -> float:
        """Calculate value score for leg prioritization."""
        edge_score = max(0, leg.edge) * 10
        conf_score = leg.confidence
        prob_score = leg.blended_probability
        
        return edge_score * conf_score * prob_score
    
    def create_leg_from_analysis(
        self,
        match_data: Dict,
        analysis: Dict,
        market: str,
        ml_prediction: Dict = None
    ) -> Optional[EnhancedBetLeg]:
        """Create enhanced leg from analysis."""
        market_data = analysis.get('market_recommendations', {}).get(market)
        if not market_data:
            return None
        
        reasoning_prob = market_data.get('probability', 0.5)
        ml_prob = ml_prediction.get('probability', reasoning_prob) if ml_prediction else reasoning_prob
        
        blended_prob = self._blend_probabilities(ml_prob, reasoning_prob)
        
        # Calculate odds
        odds = self._calculate_odds_from_probability(blended_prob)
        
        # Check constraints
        if odds < self.config.min_single_odds or odds > self.config.max_single_odds:
            return None
        
        implied_prob = self._calculate_implied_probability(odds)
        edge = blended_prob - implied_prob
        
        if edge < self.config.min_edge:
            return None
        
        confidence = market_data.get('confidence', 0.5)
        if confidence < self.config.min_confidence:
            return None
        
        value_score = edge * confidence * blended_prob * 10
        
        return EnhancedBetLeg(
            match_id=match_data.get('match_id', ''),
            home_team=match_data.get('home_team', ''),
            away_team=match_data.get('away_team', ''),
            league=match_data.get('league', ''),
            kickoff=match_data.get('kickoff', match_data.get('date', '')),
            market=market,
            tip=market_data.get('tip', 'Yes'),
            odds=round(odds, 2),
            ml_probability=round(ml_prob, 4),
            reasoning_probability=round(reasoning_prob, 4),
            blended_probability=round(blended_prob, 4),
            confidence=round(confidence, 3),
            edge=round(edge, 4),
            value_score=round(value_score, 3),
            tactical_reasoning=analysis.get('home_analysis', {}).get('tactical_notes', ''),
            scenario_reasoning=str(analysis.get('scenarios', [])),
            structural_justification=market_data.get('structural_justification', ''),
            source='combined'
        )
    
    def build_ticket(
        self,
        matches: List[Dict],
        analyses: List[Dict],
        ml_predictions: List[Dict] = None,
        stake: float = None
    ) -> Optional[EnhancedTicket]:
        """
        Build optimal multi-bet ticket.
        
        Applies:
        - Odds constraints
        - Confidence/edge thresholds
        - Diversification rules
        - Target odds optimization
        """
        stake = stake or self.config.base_stake
        ml_predictions = ml_predictions or [{} for _ in matches]
        
        # Collect all candidate legs
        candidates = []
        
        for i, (match, analysis) in enumerate(zip(matches, analyses)):
            ml_pred = ml_predictions[i] if i < len(ml_predictions) else {}
            
            for market in self.config.enabled_markets:
                leg = self.create_leg_from_analysis(match, analysis, market, ml_pred)
                if leg and leg.is_value:
                    candidates.append(leg)
        
        if len(candidates) < self.config.min_legs:
            logger.warning(f"Not enough candidates: {len(candidates)} < {self.config.min_legs}")
            return None
        
        # Sort by value score
        candidates.sort(key=lambda x: x.value_score, reverse=True)
        
        # Greedy selection with diversification
        selected = []
        league_counts = {}
        market_counts = {}
        current_odds = 1.0
        
        for leg in candidates:
            if len(selected) >= self.config.max_legs:
                break
            
            # Check diversification
            if league_counts.get(leg.league, 0) >= self.config.max_legs_per_league:
                continue
            if market_counts.get(leg.market, 0) >= self.config.max_legs_per_market:
                continue
            
            # Check total odds
            potential_odds = current_odds * leg.odds
            if potential_odds > self.config.target_total_odds * 1.5:
                continue
            
            selected.append(leg)
            league_counts[leg.league] = league_counts.get(leg.league, 0) + 1
            market_counts[leg.market] = market_counts.get(leg.market, 0) + 1
            current_odds = potential_odds
        
        if len(selected) < self.config.min_legs:
            logger.warning(f"Selection too small: {len(selected)}")
            return None
        
        return EnhancedTicket.from_legs(selected, stake)
    
    def build_multiple_tickets(
        self,
        matches: List[Dict],
        analyses: List[Dict],
        ml_predictions: List[Dict] = None,
        num_tickets: int = 3,
        stake: float = None
    ) -> List[EnhancedTicket]:
        """Build multiple diverse tickets."""
        tickets = []
        used_matches = set()
        
        stake = stake or self.config.base_stake
        ml_predictions = ml_predictions or [{} for _ in matches]
        
        for _ in range(num_tickets):
            # Filter unused matches
            available = [
                (m, a, ml_predictions[i] if i < len(ml_predictions) else {})
                for i, (m, a) in enumerate(zip(matches, analyses))
                if m.get('match_id', f"{m.get('home_team')}_{m.get('away_team')}") not in used_matches
            ]
            
            if not available:
                break
            
            avail_matches = [x[0] for x in available]
            avail_analyses = [x[1] for x in available]
            avail_preds = [x[2] for x in available]
            
            ticket = self.build_ticket(avail_matches, avail_analyses, avail_preds, stake)
            
            if ticket:
                tickets.append(ticket)
                for leg in ticket.legs:
                    used_matches.add(leg.match_id)
        
        return tickets


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TicketConfig',
    'EnhancedBetLeg',
    'EnhancedTicket',
    'MultiBetTicketBuilder'
]
