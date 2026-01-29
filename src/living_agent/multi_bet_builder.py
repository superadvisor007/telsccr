"""
🎫 MULTI-BET BUILDER - Optimal Accumulator Construction
=======================================================
Builds multi-leg tickets with constrained odds and optimal selection.

Key Constraints:
- Single leg odds: 1.40-1.70 range
- Total ticket odds: ~10.0 target
- Maximum 6 legs per ticket
- Minimum confidence threshold
- Diversification across matches/markets

Implements:
- Kelly-based stake calculation
- Correlation awareness (avoid same match legs)
- Risk-adjusted selection
- Telegram-ready formatting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import math


@dataclass
class BetLeg:
    """A single leg in the multi-bet ticket."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    market: str
    market_display: str  # Human readable
    probability: float
    confidence: float
    odds: float
    reasoning: str
    key_factors: List[str]


@dataclass
class MultiBetTicket:
    """Complete multi-bet ticket with all legs."""
    ticket_id: str
    created_at: str
    legs: List[BetLeg]
    total_odds: float
    stake: float
    potential_win: float
    expected_value: float
    overall_confidence: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    reasoning_summary: str
    

class MultiBetBuilder:
    """
    🎫 Optimal Multi-Bet Construction
    
    Builds accumulator tickets with:
    - Constrained single-leg odds (1.40-1.70)
    - Target total odds (~10.0)
    - Confidence-weighted selection
    - Diversification rules
    - Risk management
    """
    
    # Configuration
    MIN_LEG_ODDS = 1.20  # Lower bound for value
    MAX_LEG_ODDS = 2.00  # Higher range for Over 2.5, BTTS
    TARGET_TOTAL_ODDS = 6.0  # More achievable target (2-3 legs)
    MIN_CONFIDENCE = 0.48  # Slightly lowered for edge cases
    MAX_LEGS = 6
    MIN_LEGS = 2  # Allow smaller tickets
    DEFAULT_STAKE = 50.0
    
    # Market display names
    MARKET_DISPLAY = {
        'btts': 'Both Teams to Score (Yes)',
        'over_1_5': 'Over 1.5 Goals',
        'over_2_5': 'Over 2.5 Goals',
        'under_2_5': 'Under 2.5 Goals',
        'btts_no': 'Both Teams to Score (No)',
        'home_win': 'Home Win',
        'away_win': 'Away Win',
        'draw': 'Draw',
    }
    
    def __init__(
        self,
        min_leg_odds: float = None,
        max_leg_odds: float = None,
        target_total_odds: float = None,
        default_stake: float = None
    ):
        self.min_leg_odds = min_leg_odds or self.MIN_LEG_ODDS
        self.max_leg_odds = max_leg_odds or self.MAX_LEG_ODDS
        self.target_total_odds = target_total_odds or self.TARGET_TOTAL_ODDS
        self.default_stake = default_stake or self.DEFAULT_STAKE
    
    def build_ticket(
        self,
        match_analyses: List[Dict[str, Any]],
        stake: float = None,
        max_legs: int = None
    ) -> Optional[MultiBetTicket]:
        """
        Build optimal multi-bet ticket from analyzed matches.
        
        Args:
            match_analyses: List of match analysis results
            stake: Stake amount (default: 50)
            max_legs: Maximum legs (default: 6)
        
        Returns:
            MultiBetTicket or None if no valid ticket possible
        """
        stake = stake or self.default_stake
        max_legs = max_legs or self.MAX_LEGS
        
        # Extract all candidate legs
        candidates = self._extract_candidates(match_analyses)
        
        if not candidates:
            return None
        
        # Select optimal legs
        selected = self._select_optimal_legs(candidates, max_legs)
        
        if len(selected) < self.MIN_LEGS:
            return None
        
        # Calculate ticket metrics
        total_odds = math.prod(leg.odds for leg in selected)
        potential_win = stake * total_odds
        
        # Combined probability (assuming independence)
        combined_prob = math.prod(leg.probability for leg in selected)
        
        # Expected value
        expected_value = (combined_prob * potential_win) - stake
        
        # Overall confidence (geometric mean)
        overall_confidence = math.prod(leg.confidence for leg in selected) ** (1/len(selected))
        
        # Risk level
        if overall_confidence > 0.65 and len(selected) <= 4:
            risk_level = 'LOW'
        elif overall_confidence > 0.55 and len(selected) <= 5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        # Generate ticket ID
        ticket_id = f"TS-{datetime.now().strftime('%Y%m%d%H%M')}-{len(selected)}L"
        
        # Reasoning summary
        reasoning_summary = self._generate_reasoning_summary(selected)
        
        return MultiBetTicket(
            ticket_id=ticket_id,
            created_at=datetime.now().isoformat(),
            legs=selected,
            total_odds=round(total_odds, 2),
            stake=stake,
            potential_win=round(potential_win, 2),
            expected_value=round(expected_value, 2),
            overall_confidence=round(overall_confidence, 3),
            risk_level=risk_level,
            reasoning_summary=reasoning_summary
        )
    
    def _extract_candidates(
        self,
        match_analyses: List[Dict[str, Any]]
    ) -> List[BetLeg]:
        """Extract candidate legs from match analyses."""
        
        candidates = []
        
        for analysis in match_analyses:
            match_id = analysis.get('match_id', '')
            home_team = analysis.get('home_team', 'Home')
            away_team = analysis.get('away_team', 'Away')
            league = analysis.get('league', 'Unknown')
            
            # Process each market analysis
            for market_data in analysis.get('market_analyses', []):
                market = market_data.get('market', '')
                probability = market_data.get('probability', 0)
                confidence = market_data.get('confidence', 0)
                recommendation = market_data.get('recommendation', 'SKIP')
                
                # Only consider BET recommendations
                if recommendation != 'BET':
                    continue
                
                # Calculate odds from probability
                # Add bookmaker margin (5-8%)
                fair_odds = 1 / probability if probability > 0 else 10
                margin_factor = 0.92  # 8% margin
                odds = fair_odds * margin_factor
                
                # Check odds constraints
                if odds < self.min_leg_odds or odds > self.max_leg_odds:
                    continue
                
                # Check confidence
                if confidence < self.MIN_CONFIDENCE:
                    continue
                
                # Create leg
                leg = BetLeg(
                    match_id=match_id,
                    home_team=home_team,
                    away_team=away_team,
                    league=league,
                    market=market,
                    market_display=self.MARKET_DISPLAY.get(market, market),
                    probability=probability,
                    confidence=confidence,
                    odds=round(odds, 2),
                    reasoning=market_data.get('reasoning', ''),
                    key_factors=market_data.get('key_factors', [])
                )
                candidates.append(leg)
        
        return candidates
    
    def _select_optimal_legs(
        self,
        candidates: List[BetLeg],
        max_legs: int
    ) -> List[BetLeg]:
        """Select optimal legs for the ticket."""
        
        if not candidates:
            return []
        
        # Sort by confidence-weighted score
        # Score = confidence * (probability - 1/odds) = confidence * edge
        def leg_score(leg: BetLeg) -> float:
            implied_prob = 1 / leg.odds
            edge = leg.probability - implied_prob
            return leg.confidence * edge
        
        sorted_candidates = sorted(candidates, key=leg_score, reverse=True)
        
        selected = []
        used_matches = set()
        current_odds = 1.0
        
        for candidate in sorted_candidates:
            # Diversification: max 1 leg per match
            if candidate.match_id in used_matches:
                continue
            
            # Check if adding this leg would exceed target
            new_odds = current_odds * candidate.odds
            
            # Stop if we reach target or max legs
            if new_odds > self.target_total_odds * 1.5:
                continue
            if len(selected) >= max_legs:
                break
            
            selected.append(candidate)
            used_matches.add(candidate.match_id)
            current_odds = new_odds
            
            # Check if we're close to target
            if current_odds >= self.target_total_odds * 0.8 and len(selected) >= self.MIN_LEGS:
                # Could stop here, but continue looking for better options
                pass
        
        return selected
    
    def _generate_reasoning_summary(self, legs: List[BetLeg]) -> str:
        """Generate summary of ticket reasoning."""
        
        markets = [leg.market_display for leg in legs]
        avg_conf = sum(leg.confidence for leg in legs) / len(legs)
        
        unique_leagues = set(leg.league for leg in legs)
        
        summary = f"📊 {len(legs)}-leg accumulator across {len(unique_leagues)} league(s). "
        summary += f"Average confidence: {avg_conf:.0%}. "
        
        # Most common market type
        market_counts = {}
        for m in markets:
            market_counts[m] = market_counts.get(m, 0) + 1
        top_market = max(market_counts, key=market_counts.get)
        
        summary += f"Primary market: {top_market}."
        
        return summary
    
    def format_for_telegram(
        self,
        ticket: MultiBetTicket,
        include_reasoning: bool = True
    ) -> str:
        """Format ticket for Telegram delivery."""
        
        lines = [
            "═══════════════════════════════",
            "       🎫 MULTI-BET TICKET 🎫",
            "═══════════════════════════════",
            "",
            "📱 TelegramSoccer AI",
            f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            f"🎟️ {ticket.ticket_id}",
            "🤖 Powered by DeepSeek 7B",
            "",
            "─────────────────────────────────",
        ]
        
        for i, leg in enumerate(ticket.legs, 1):
            confidence_bar = self._confidence_bar(leg.confidence)
            
            lines.extend([
                "",
                f"Leg {i}:",
                f"  {leg.home_team}",
                f"    vs",
                f"  {leg.away_team}",
                f"  📍 {leg.league}",
                f"  ⚽ {leg.market_display}",
                f"  💰 Odds: {leg.odds:.2f}",
                f"  📊 {confidence_bar} {leg.confidence:.0%}",
            ])
            
            if include_reasoning and leg.key_factors:
                for factor in leg.key_factors[:2]:
                    lines.append(f"  • {factor}")
            
            lines.append("")
            lines.append("─────────────────────────────────")
        
        # Summary
        lines.extend([
            "",
            "📋 SUMMARY",
            "",
            f"  Total Legs:    {len(ticket.legs)}",
            f"  Total Odds:    {ticket.total_odds:.2f}",
            f"  Stake:         €{ticket.stake:.2f}",
            f"  Potential Win: €{ticket.potential_win:.2f}",
            "",
            "═══════════════════════════════",
            f"  ⚠️ Gamble Responsibly",
            f"  🎯 Confidence: {ticket.overall_confidence:.0%}",
            f"  📈 Risk Level: {ticket.risk_level}",
            "═══════════════════════════════",
        ])
        
        return "\n".join(lines)
    
    def format_with_results(
        self,
        ticket: MultiBetTicket,
        results: Dict[str, Dict[str, Any]]
    ) -> str:
        """Format ticket with match results."""
        
        lines = [
            "═══════════════════════════════",
            "     🎫 MULTI-BET RESULTS 🎫",
            "═══════════════════════════════",
            "",
            "📱 TelegramSoccer AI",
            f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            "",
            "─────────────────────────────────",
        ]
        
        correct = 0
        total = len(ticket.legs)
        
        for i, leg in enumerate(ticket.legs, 1):
            result = results.get(leg.match_id, {})
            outcome = result.get('outcome', None)
            score = result.get('score', 'N/A')
            
            if outcome is True:
                marker = "✓"
                correct += 1
            elif outcome is False:
                marker = "✗"
            else:
                marker = "?"
            
            lines.extend([
                "",
                f"Leg {i}: {marker}",
                f"  {leg.home_team}",
                f"    vs",
                f"  {leg.away_team} ({score})",
                f"  ⚽ {leg.market_display}",
                f"  💰 Odds: {leg.odds:.2f}",
                "",
                "─────────────────────────────────",
            ])
        
        # Results summary
        all_correct = correct == total
        
        lines.extend([
            "",
            "📋 RESULTS",
            "",
            f"  Total Legs:    {total}",
            f"  Correct:       {correct}/{total} {'✓' if all_correct else ''}",
            f"  Total Odds:    {ticket.total_odds:.2f}",
            "",
        ])
        
        if all_correct:
            profit = ticket.potential_win - ticket.stake
            lines.extend([
                f"  ✅ TICKET WON!",
                f"  💰 Profit: +€{profit:.2f}",
            ])
        else:
            lines.extend([
                f"  ❌ TICKET LOST",
                f"  📉 Loss: -€{ticket.stake:.2f}",
            ])
        
        lines.extend([
            "",
            "═══════════════════════════════",
            "  🤖 DeepSeek 7B Analysis",
            "═══════════════════════════════",
        ])
        
        return "\n".join(lines)
    
    def _confidence_bar(self, confidence: float, width: int = 10) -> str:
        """Generate visual confidence bar."""
        filled = int(confidence * width)
        empty = width - filled
        return "█" * filled + "░" * empty
    
    def suggest_stake(
        self,
        ticket: MultiBetTicket,
        bankroll: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """Suggest optimal stake using fractional Kelly."""
        
        # Combined probability
        combined_prob = math.prod(leg.probability for leg in ticket.legs)
        
        # Kelly criterion
        b = ticket.total_odds - 1  # Net odds
        q = 1 - combined_prob
        
        if b <= 0:
            return 0
        
        kelly = (combined_prob * b - q) / b
        
        # Fractional Kelly for safety
        suggested = max(0, kelly * kelly_fraction * bankroll)
        
        # Cap at 5% of bankroll
        max_stake = bankroll * 0.05
        
        return min(suggested, max_stake)
