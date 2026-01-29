"""
Professional Multi-Bet Ticket Generator for Telegram.

Creates betting tickets that look like real-world betting slips,
with support for result verification (✓/X markers).

Integrates with DeepSeek LLM for intelligent tip generation.
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class MarketType(Enum):
    """Supported betting markets."""
    BTTS = "Both Teams to Score (BTTS)"
    OVER_1_5 = "Over 1.5 Goals"
    OVER_2_5 = "Over 2.5 Goals"
    UNDER_2_5 = "Under 2.5 Goals"
    HOME_WIN = "Home Win (1)"
    AWAY_WIN = "Away Win (2)"
    DRAW = "Draw (X)"
    DOUBLE_CHANCE_1X = "Double Chance (1X)"
    DOUBLE_CHANCE_X2 = "Double Chance (X2)"
    DOUBLE_CHANCE_12 = "Double Chance (12)"


MARKET_EMOJIS = {
    MarketType.BTTS: "⚽",
    MarketType.OVER_1_5: "🔥",
    MarketType.OVER_2_5: "🔥🔥",
    MarketType.UNDER_2_5: "🛡️",
    MarketType.HOME_WIN: "🏠",
    MarketType.AWAY_WIN: "✈️",
    MarketType.DRAW: "🤝",
    MarketType.DOUBLE_CHANCE_1X: "🏠🤝",
    MarketType.DOUBLE_CHANCE_X2: "🤝✈️",
    MarketType.DOUBLE_CHANCE_12: "🏠✈️",
}


@dataclass
class BetLeg:
    """Single leg of a multi-bet ticket."""
    home_team: str
    away_team: str
    market: MarketType
    odds: float
    league: str = ""
    match_date: str = ""
    match_time: str = ""
    
    # Result fields (filled after match)
    result: Optional[str] = None  # "WIN", "LOSS", "PENDING"
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    
    # DeepSeek analysis
    probability: Optional[float] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None


@dataclass
class MultiBetTicket:
    """Complete multi-bet ticket."""
    legs: List[BetLeg] = field(default_factory=list)
    bookmaker: str = "TelegramSoccer AI"
    ticket_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    stake: float = 10.0
    
    # DeepSeek metadata
    llm_model: str = "deepseek-llm:7b"
    analysis_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.ticket_id:
            self.ticket_id = f"TS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    @property
    def total_odds(self) -> float:
        """Calculate combined odds."""
        if not self.legs:
            return 1.0
        odds = 1.0
        for leg in self.legs:
            odds *= leg.odds
        return round(odds, 2)
    
    @property
    def potential_win(self) -> float:
        """Calculate potential winnings."""
        return round(self.stake * self.total_odds, 2)
    
    @property
    def total_legs(self) -> int:
        """Number of legs in ticket."""
        return len(self.legs)
    
    @property
    def wins(self) -> int:
        """Count winning legs."""
        return sum(1 for leg in self.legs if leg.result == "WIN")
    
    @property
    def losses(self) -> int:
        """Count losing legs."""
        return sum(1 for leg in self.legs if leg.result == "LOSS")
    
    @property
    def is_winner(self) -> bool:
        """Check if ticket is a winner (all legs won)."""
        return all(leg.result == "WIN" for leg in self.legs)
    
    @property
    def is_settled(self) -> bool:
        """Check if all legs have results."""
        return all(leg.result in ["WIN", "LOSS"] for leg in self.legs)


class TicketGenerator:
    """
    Professional Multi-Bet Ticket Generator.
    
    Creates Telegram-formatted betting tickets that look like
    real-world betting slips.
    """
    
    # Ticket visual constants
    SEPARATOR = "─" * 35
    HEADER_LINE = "═" * 35
    
    def __init__(self, bookmaker_name: str = "TelegramSoccer AI"):
        self.bookmaker = bookmaker_name
        
        # Try to import DeepSeek for analysis
        try:
            from src.llm.deepseek_client import get_deepseek_llm
            self.llm = get_deepseek_llm()
            self.llm_available = True
            logger.info("DeepSeek LLM connected for ticket analysis")
        except Exception as e:
            self.llm = None
            self.llm_available = False
            logger.warning(f"DeepSeek not available: {e}")
    
    def create_ticket(
        self,
        predictions: List[Dict[str, Any]],
        stake: float = 10.0,
        use_llm_analysis: bool = True,
    ) -> MultiBetTicket:
        """
        Create a multi-bet ticket from predictions.
        
        Args:
            predictions: List of prediction dicts with match data
            stake: Stake amount
            use_llm_analysis: Whether to use DeepSeek for analysis
        
        Returns:
            MultiBetTicket object
        """
        ticket = MultiBetTicket(
            bookmaker=self.bookmaker,
            stake=stake,
            llm_model="deepseek-llm:7b" if self.llm_available else "statistical",
        )
        
        for pred in predictions:
            # Determine market type
            market_str = pred.get("market", "over_1_5").lower()
            market = self._parse_market(market_str)
            
            leg = BetLeg(
                home_team=pred.get("home_team", "Home Team"),
                away_team=pred.get("away_team", "Away Team"),
                market=market,
                odds=float(pred.get("odds", 1.50)),
                league=pred.get("league", ""),
                match_date=pred.get("date", ""),
                match_time=pred.get("time", ""),
                probability=pred.get("probability"),
                confidence=pred.get("confidence"),
                reasoning=pred.get("reasoning"),
            )
            
            # Get DeepSeek analysis if available
            if use_llm_analysis and self.llm_available and not leg.reasoning:
                analysis = self._get_llm_analysis(pred, market_str)
                leg.probability = analysis.get("probability", leg.probability)
                leg.confidence = analysis.get("confidence", leg.confidence)
                leg.reasoning = analysis.get("reasoning", leg.reasoning)
            
            ticket.legs.append(leg)
        
        ticket.analysis_timestamp = datetime.now()
        return ticket
    
    def _parse_market(self, market_str: str) -> MarketType:
        """Parse market string to MarketType enum."""
        market_map = {
            "btts": MarketType.BTTS,
            "both_teams_to_score": MarketType.BTTS,
            "over_1_5": MarketType.OVER_1_5,
            "over_2_5": MarketType.OVER_2_5,
            "under_2_5": MarketType.UNDER_2_5,
            "home": MarketType.HOME_WIN,
            "away": MarketType.AWAY_WIN,
            "draw": MarketType.DRAW,
            "1x": MarketType.DOUBLE_CHANCE_1X,
            "x2": MarketType.DOUBLE_CHANCE_X2,
            "12": MarketType.DOUBLE_CHANCE_12,
        }
        return market_map.get(market_str.lower(), MarketType.OVER_1_5)
    
    def _get_llm_analysis(self, match_data: Dict, market: str) -> Dict[str, Any]:
        """Get DeepSeek analysis for a match."""
        if not self.llm_available:
            return {}
        
        try:
            return self.llm.analyze_match(match_data, market)
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return {}
    
    def format_ticket(
        self,
        ticket: MultiBetTicket,
        show_results: bool = False,
        show_confidence: bool = True,
        show_scores: bool = True,
    ) -> str:
        """
        Format ticket as Telegram message (monospace).
        
        Args:
            ticket: MultiBetTicket to format
            show_results: Show ✓/X result markers
            show_confidence: Show AI confidence levels
            show_scores: Show final scores if available
        
        Returns:
            Formatted string for Telegram
        """
        lines = []
        
        # Header
        lines.append(f"```")
        lines.append(f"{'═' * 35}")
        lines.append(f"    🎫 MULTI-BET TICKET 🎫")
        lines.append(f"{'═' * 35}")
        lines.append(f"")
        lines.append(f"📱 {ticket.bookmaker}")
        lines.append(f"📅 {ticket.created_at.strftime('%d/%m/%Y %H:%M')}")
        lines.append(f"🎟️ ID: {ticket.ticket_id}")
        if self.llm_available:
            lines.append(f"🤖 AI: DeepSeek 7B")
        lines.append(f"")
        lines.append(self.SEPARATOR)
        
        # Legs
        for i, leg in enumerate(ticket.legs, 1):
            # Result marker
            result_marker = ""
            if show_results and leg.result:
                if leg.result == "WIN":
                    result_marker = " ✓"
                elif leg.result == "LOSS":
                    result_marker = " ✗"
                else:
                    result_marker = " ⏳"
            
            # Score display
            score_str = ""
            if show_scores and leg.home_score is not None and leg.away_score is not None:
                score_str = f" ({leg.home_score}-{leg.away_score})"
            
            # Market emoji
            emoji = MARKET_EMOJIS.get(leg.market, "⚽")
            
            lines.append(f"")
            lines.append(f"Leg {i}:{result_marker}")
            lines.append(f"  {leg.home_team}")
            lines.append(f"    vs")
            lines.append(f"  {leg.away_team}{score_str}")
            
            if leg.league:
                lines.append(f"  📍 {leg.league}")
            
            lines.append(f"  {emoji} {leg.market.value}")
            lines.append(f"  💰 Odds: {leg.odds:.2f}")
            
            # Confidence indicator
            if show_confidence and leg.probability:
                conf_bar = self._confidence_bar(leg.probability)
                lines.append(f"  📊 {conf_bar} {leg.probability:.0%}")
            
            lines.append(f"")
            lines.append(self.SEPARATOR)
        
        # Summary
        lines.append(f"")
        lines.append(f"📋 SUMMARY")
        lines.append(f"")
        lines.append(f"  Total Legs:    {ticket.total_legs}")
        lines.append(f"  Total Odds:    {ticket.total_odds:.2f}")
        lines.append(f"  Stake:         €{ticket.stake:.2f}")
        lines.append(f"  Potential Win: €{ticket.potential_win:.2f}")
        
        # Results summary if available
        if show_results and ticket.is_settled:
            lines.append(f"")
            lines.append(self.SEPARATOR)
            lines.append(f"")
            if ticket.is_winner:
                lines.append(f"  🎉 TICKET WON! 🎉")
                lines.append(f"  💵 Payout: €{ticket.potential_win:.2f}")
            else:
                lines.append(f"  ❌ TICKET LOST")
                lines.append(f"  📊 Correct: {ticket.wins}/{ticket.total_legs}")
        
        # Footer
        lines.append(f"")
        lines.append(f"{'═' * 35}")
        lines.append(f"  ⚠️ Gamble Responsibly")
        lines.append(f"  🤖 Powered by DeepSeek AI")
        lines.append(f"{'═' * 35}")
        lines.append(f"```")
        
        return "\n".join(lines)
    
    def _confidence_bar(self, probability: float, length: int = 10) -> str:
        """Create visual confidence bar."""
        filled = int(probability * length)
        empty = length - filled
        
        if probability >= 0.75:
            fill_char = "🟢"
        elif probability >= 0.60:
            fill_char = "🟡"
        else:
            fill_char = "🔴"
        
        return "█" * filled + "░" * empty
    
    def format_ticket_html(
        self,
        ticket: MultiBetTicket,
        show_results: bool = False,
    ) -> str:
        """
        Format ticket with HTML tags for Telegram (colored).
        
        Use parse_mode='HTML' when sending.
        """
        lines = []
        
        lines.append("<b>🎫 MULTI-BET TICKET</b>")
        lines.append("")
        lines.append(f"📱 <code>{ticket.bookmaker}</code>")
        lines.append(f"📅 <code>{ticket.created_at.strftime('%d/%m/%Y %H:%M')}</code>")
        lines.append(f"🎟️ <code>{ticket.ticket_id}</code>")
        lines.append(f"🤖 <i>DeepSeek 7B Analysis</i>")
        lines.append("")
        lines.append("─" * 30)
        
        for i, leg in enumerate(ticket.legs, 1):
            result_html = ""
            if show_results and leg.result:
                if leg.result == "WIN":
                    result_html = " <b>✓</b>"
                elif leg.result == "LOSS":
                    result_html = " <b>✗</b>"
            
            emoji = MARKET_EMOJIS.get(leg.market, "⚽")
            
            lines.append("")
            lines.append(f"<b>Leg {i}:</b>{result_html}")
            lines.append(f"  {leg.home_team} vs {leg.away_team}")
            lines.append(f"  {emoji} <i>{leg.market.value}</i>")
            lines.append(f"  💰 <code>{leg.odds:.2f}</code>")
            
            if leg.probability:
                conf_color = "green" if leg.probability >= 0.70 else "orange"
                lines.append(f"  📊 Confidence: <b>{leg.probability:.0%}</b>")
        
        lines.append("")
        lines.append("─" * 30)
        lines.append("")
        lines.append(f"📋 <b>Total Legs:</b> {ticket.total_legs}")
        lines.append(f"💰 <b>Total Odds:</b> <code>{ticket.total_odds:.2f}</code>")
        lines.append(f"💵 <b>Potential Win:</b> <code>€{ticket.potential_win:.2f}</code>")
        
        if show_results and ticket.is_settled:
            lines.append("")
            if ticket.is_winner:
                lines.append("🎉 <b>TICKET WON!</b> 🎉")
            else:
                lines.append(f"❌ Ticket lost ({ticket.wins}/{ticket.total_legs} correct)")
        
        lines.append("")
        lines.append("<i>⚠️ Gamble Responsibly | 🤖 DeepSeek AI</i>")
        
        return "\n".join(lines)
    
    def update_results(
        self,
        ticket: MultiBetTicket,
        results: List[Dict[str, Any]],
    ) -> MultiBetTicket:
        """
        Update ticket with match results.
        
        Args:
            ticket: Original ticket
            results: List of result dicts with home_score, away_score
        
        Returns:
            Updated ticket
        """
        for i, (leg, result) in enumerate(zip(ticket.legs, results)):
            home_score = result.get("home_score")
            away_score = result.get("away_score")
            
            if home_score is None or away_score is None:
                leg.result = "PENDING"
                continue
            
            leg.home_score = home_score
            leg.away_score = away_score
            
            # Determine if leg won
            total_goals = home_score + away_score
            both_scored = home_score > 0 and away_score > 0
            
            won = False
            if leg.market == MarketType.BTTS:
                won = both_scored
            elif leg.market == MarketType.OVER_1_5:
                won = total_goals >= 2
            elif leg.market == MarketType.OVER_2_5:
                won = total_goals >= 3
            elif leg.market == MarketType.UNDER_2_5:
                won = total_goals < 3
            elif leg.market == MarketType.HOME_WIN:
                won = home_score > away_score
            elif leg.market == MarketType.AWAY_WIN:
                won = away_score > home_score
            elif leg.market == MarketType.DRAW:
                won = home_score == away_score
            elif leg.market == MarketType.DOUBLE_CHANCE_1X:
                won = home_score >= away_score
            elif leg.market == MarketType.DOUBLE_CHANCE_X2:
                won = away_score >= home_score
            elif leg.market == MarketType.DOUBLE_CHANCE_12:
                won = home_score != away_score
            
            leg.result = "WIN" if won else "LOSS"
        
        return ticket


class DailyTicketService:
    """
    Service for generating daily multi-bet tickets.
    
    Uses DeepSeek LLM for intelligent match analysis and
    creates professional betting tickets.
    """
    
    def __init__(self):
        self.generator = TicketGenerator()
        self.tickets: List[MultiBetTicket] = []
    
    def generate_daily_ticket(
        self,
        predictions: List[Dict[str, Any]],
        target_odds: float = 1.40,
        max_legs: int = 4,
        min_confidence: float = 0.70,
    ) -> MultiBetTicket:
        """
        Generate optimized daily multi-bet ticket.
        
        Args:
            predictions: Available predictions for today
            target_odds: Target combined odds (~1.40)
            max_legs: Maximum number of legs
            min_confidence: Minimum confidence threshold
        
        Returns:
            Optimized MultiBetTicket
        """
        # Filter by confidence
        valid_predictions = [
            p for p in predictions
            if p.get("probability", 0) >= min_confidence
        ]
        
        # Sort by value (probability / implied_probability)
        for p in valid_predictions:
            odds = p.get("odds", 1.5)
            implied_prob = 1 / odds
            actual_prob = p.get("probability", 0.5)
            p["value"] = actual_prob / implied_prob
        
        valid_predictions.sort(key=lambda x: x.get("value", 0), reverse=True)
        
        # Select best predictions up to max_legs
        selected = valid_predictions[:max_legs]
        
        # Create ticket
        ticket = self.generator.create_ticket(selected)
        self.tickets.append(ticket)
        
        return ticket
    
    def get_telegram_message(
        self,
        ticket: MultiBetTicket,
        use_html: bool = False,
    ) -> str:
        """Get formatted Telegram message for ticket."""
        if use_html:
            return self.generator.format_ticket_html(ticket)
        return self.generator.format_ticket(ticket)
    
    def get_results_message(
        self,
        ticket: MultiBetTicket,
        results: List[Dict[str, Any]],
        use_html: bool = False,
    ) -> str:
        """Get formatted results message with ✓/X markers."""
        updated_ticket = self.generator.update_results(ticket, results)
        
        if use_html:
            return self.generator.format_ticket_html(updated_ticket, show_results=True)
        return self.generator.format_ticket(updated_ticket, show_results=True, show_scores=True)


# Example usage and test
if __name__ == "__main__":
    # Test predictions
    test_predictions = [
        {
            "home_team": "AS Monaco FC",
            "away_team": "Juventus",
            "market": "btts",
            "odds": 1.60,
            "league": "Champions League",
            "probability": 0.72,
            "confidence": 0.75,
        },
        {
            "home_team": "Napoli",
            "away_team": "Chelsea",
            "market": "btts",
            "odds": 1.65,
            "league": "Champions League",
            "probability": 0.68,
            "confidence": 0.70,
        },
        {
            "home_team": "Real Betis",
            "away_team": "Feyenoord",
            "market": "btts",
            "odds": 1.60,
            "league": "Europa League",
            "probability": 0.70,
            "confidence": 0.72,
        },
        {
            "home_team": "1. FC Magdeburg",
            "away_team": "Hannover 96",
            "market": "btts",
            "odds": 1.44,
            "league": "2. Bundesliga",
            "probability": 0.78,
            "confidence": 0.80,
        },
    ]
    
    # Create ticket
    generator = TicketGenerator()
    ticket = generator.create_ticket(test_predictions, stake=10.0)
    
    # Print initial ticket
    print("=" * 50)
    print("INITIAL TICKET (Before Results)")
    print("=" * 50)
    print(generator.format_ticket(ticket))
    
    # Simulate results
    test_results = [
        {"home_score": 2, "away_score": 1},  # WIN (BTTS)
        {"home_score": 1, "away_score": 0},  # LOSS (no BTTS)
        {"home_score": 2, "away_score": 2},  # WIN (BTTS)
        {"home_score": 1, "away_score": 1},  # WIN (BTTS)
    ]
    
    # Update with results
    ticket = generator.update_results(ticket, test_results)
    
    print("\n" + "=" * 50)
    print("TICKET WITH RESULTS (After Matches)")
    print("=" * 50)
    print(generator.format_ticket(ticket, show_results=True, show_scores=True))
