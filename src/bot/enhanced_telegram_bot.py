"""
🎯 Enhanced Telegram Bot with Unified Pipeline Integration
==========================================================
Connects the complete goal-directed reasoning system to Telegram delivery.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Import unified pipeline
from src.unified_pipeline import UnifiedPipeline, UnifiedConfig
from src.betting.multibet_ticket_builder import EnhancedTicket, EnhancedBetLeg

logger = logging.getLogger(__name__)


class EnhancedTelegramBot:
    """
    🎯 Enhanced Telegram Bot
    
    Features:
    - Unified pipeline integration
    - Goal-directed reasoning
    - Multi-bet ticket generation
    - Result tracking
    - Statistics dashboard
    """
    
    def __init__(
        self,
        token: str = None,
        chat_id: str = None,
        admin_ids: List[int] = None
    ):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.admin_ids = admin_ids or []
        
        # Initialize pipeline
        config = UnifiedConfig(
            telegram_token=self.token,
            telegram_chat_id=self.chat_id
        )
        self.pipeline = UnifiedPipeline(config)
        
        # State
        self.current_ticket: Optional[EnhancedTicket] = None
        self.ticket_history: List[Dict] = []
        self.stats = {
            'total_tickets': 0,
            'wins': 0,
            'losses': 0,
            'pending': 0,
            'profit_loss': 0.0,
            'roi': 0.0
        }
        
        # Application
        self.application = None
        
        logger.info("🎯 EnhancedTelegramBot initialized")
    
    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram MarkdownV2."""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    def format_ticket_message(
        self,
        ticket: EnhancedTicket,
        show_results: bool = False
    ) -> str:
        """Format ticket for Telegram display."""
        lines = [
            "🎟️ *TELEGRAMSOCCER BETTING TICKET*",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📋 ID: `{ticket.ticket_id}`",
            f"📅 Created: {ticket.created_at[:16].replace('T', ' ')}",
            "",
            "🎯 *SELECTIONS:*",
            "─────────────────────"
        ]
        
        for i, leg in enumerate(ticket.legs, 1):
            emoji = '⚽' if 'btts' in leg.market.lower() else '🥅'
            market_display = {
                'over_1_5': 'Over 1.5 Goals',
                'over_2_5': 'Over 2.5 Goals',
                'btts': 'BTTS Yes',
                'btts_yes': 'BTTS Yes',
                'btts_no': 'BTTS No'
            }.get(leg.market, leg.market.upper())
            
            lines.append(f"\n{i}. {emoji} *{leg.home_team}* vs *{leg.away_team}*")
            lines.append(f"   📊 {market_display}: {leg.tip}")
            lines.append(f"   💰 Odds: {leg.odds:.2f} | Conf: {leg.confidence:.0%}")
            lines.append(f"   📈 Edge: {leg.edge:+.1%}")
            
            if leg.tactical_reasoning:
                reason = leg.tactical_reasoning[:50] + "..." if len(leg.tactical_reasoning) > 50 else leg.tactical_reasoning
                lines.append(f"   💡 {reason}")
        
        lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            "📊 *TICKET SUMMARY:*",
            f"   🎰 Total Odds: *{ticket.total_odds:.2f}*",
            f"   💰 Stake: €{ticket.stake:.2f}",
            f"   🏆 Potential Win: *€{ticket.potential_win:.2f}*",
            f"   📈 Expected Value: {ticket.expected_value:+.2f}",
            f"   🎯 Confidence: {ticket.overall_confidence:.0%}",
            f"   ⚡ Avg Edge: {ticket.overall_edge:+.1%}",
            "",
            "⚠️ Gamble responsibly | 18+ only",
            "━━━━━━━━━━━━━━━━━━━━━━━━"
        ])
        
        return "\n".join(lines)
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        welcome = """
🎯 *Welcome to TelegramSoccer Bot!*

AI-powered soccer betting assistant with:
• Goal-directed LLM reasoning 🧠
• Walk-forward validated ML models 📊
• Battle-tested parameters ✅
• Target: ~10x multi-bet odds

*Performance (Backtested):*
• 77% Win Rate
• +5.38% ROI
• 1.47 Sharpe Ratio

*Commands:*
/start - This message
/today - Get today's ticket
/ticket - View current ticket
/analyze - Detailed match analysis
/backtest - View backtest stats
/stats - Your betting stats
/help - Help & strategy guide

⚠️ Bet responsibly. Gambling involves risk.
        """
        await update.message.reply_text(welcome, parse_mode="Markdown")
        logger.info(f"User {update.effective_user.id} started bot")
    
    async def today_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /today command - generate today's ticket."""
        await update.message.reply_text("🔄 Generating today's ticket... Please wait.")
        
        try:
            # Build ticket through unified pipeline
            ticket = self.pipeline.build_daily_ticket()
            
            if ticket:
                self.current_ticket = ticket
                context.bot_data['current_ticket'] = ticket
                self.ticket_history.append({
                    'ticket': ticket.to_dict(),
                    'created_at': datetime.now().isoformat(),
                    'status': 'pending'
                })
                self.stats['total_tickets'] += 1
                self.stats['pending'] += 1
                
                message = self.format_ticket_message(ticket)
                await update.message.reply_text(message, parse_mode="Markdown")
            else:
                await update.message.reply_text(
                    "❌ Could not generate a valid ticket for today.\n"
                    "Not enough matches meet our criteria."
                )
            
        except Exception as e:
            logger.error(f"Error generating ticket: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    async def ticket_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ticket command - show current ticket."""
        ticket = context.bot_data.get('current_ticket', self.current_ticket)
        
        if not ticket:
            await update.message.reply_text(
                "🎫 No active ticket.\nUse /today to generate today's tips!"
            )
            return
        
        message = self.format_ticket_message(ticket)
        await update.message.reply_text(message, parse_mode="Markdown")
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze command - show detailed analysis."""
        analyses = self.pipeline._today_analyses
        
        if not analyses:
            await update.message.reply_text(
                "📊 No analysis available.\nUse /today first to generate analysis."
            )
            return
        
        lines = ["📊 *MATCH ANALYSIS*", "━━━━━━━━━━━━━━━━━━━━━━━━", ""]
        
        for a in analyses[:5]:  # Limit to 5
            lines.append(f"⚽ *{a.home_team}* vs *{a.away_team}*")
            lines.append(f"🏆 League: {a.league}")
            lines.append("")
            
            # Tactical profiles
            lines.append(f"🏠 Home Profile:")
            lines.append(f"   Attack: {a.home_analysis.attacking_strength:.0%}")
            lines.append(f"   Def Risk: {a.home_analysis.defensive_risk:.0%}")
            lines.append(f"   Pressing: {a.home_analysis.pressing_intensity:.0%}")
            
            lines.append(f"✈️ Away Profile:")
            lines.append(f"   Attack: {a.away_analysis.attacking_strength:.0%}")
            lines.append(f"   Def Risk: {a.away_analysis.defensive_risk:.0%}")
            lines.append(f"   Pressing: {a.away_analysis.pressing_intensity:.0%}")
            
            # Scenarios
            lines.append("")
            lines.append("🎲 Scenarios:")
            for s in a.scenarios:
                lines.append(f"   • {s.home_goals}-{s.away_goals} ({s.probability:.0%})")
            
            # Market recommendations
            lines.append("")
            lines.append("📈 Markets:")
            for name, rec in a.market_recommendations.items():
                status = "✅" if rec.is_actionable else "❌"
                lines.append(f"   {status} {name}: {rec.probability:.0%} (edge: {rec.edge:+.1%})")
            
            lines.append("")
            lines.append("─────────────────────")
            lines.append("")
        
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    
    async def backtest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /backtest command - show backtest results."""
        message = """
📊 *BATTLE-TESTED BACKTEST RESULTS*
━━━━━━━━━━━━━━━━━━━━━━━━

*Dataset:* 14,349 matches (7 leagues, 5 seasons)

*Overall Performance:*
• Total Bets: 6,869
• Win Rate: 77.0%
• ROI: +5.38%
• Max Drawdown: 4.9%
• Sharpe Ratio: 1.47

*Market Breakdown:*
• Over 1.5: 6,503 bets, 77.6% WR, +4.87% ROI
• Over 2.5: 366 bets, 66.1% WR, +14.40% ROI

*Parameters (Battle-Tested):*
• Min Edge: 8%
• Min Confidence: 62%
• Odds Range: 1.25 - 1.80

*Verdict:* ✅ PRODUCTION READY

━━━━━━━━━━━━━━━━━━━━━━━━
        """
        await update.message.reply_text(message, parse_mode="Markdown")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stats command - show user statistics."""
        message = f"""
📊 *YOUR BETTING STATISTICS*
━━━━━━━━━━━━━━━━━━━━━━━━

*Tickets:*
• Total: {self.stats['total_tickets']}
• Wins: {self.stats['wins']} ✅
• Losses: {self.stats['losses']} ❌
• Pending: {self.stats['pending']} ⏳

*Financial:*
• Profit/Loss: €{self.stats['profit_loss']:.2f}
• ROI: {self.stats['roi']:.1%}

━━━━━━━━━━━━━━━━━━━━━━━━
        """
        await update.message.reply_text(message, parse_mode="Markdown")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = """
ℹ️ *TELEGRAMSOCCER HELP*
━━━━━━━━━━━━━━━━━━━━━━━━

*How It Works:*
1. 🧠 Goal-Directed Reasoning
   - LLM analyzes team tactics
   - Simulates match scenarios
   - Evaluates betting markets

2. 📊 ML Predictions
   - Walk-forward validated models
   - 14K match training data
   - No lookahead bias

3. 🎯 Value Detection
   - Edge-based selection
   - Confidence filtering
   - Diversification rules

4. 🎟️ Ticket Building
   - Target: ~10x total odds
   - 3-6 legs per ticket
   - Optimized for value

*Markets:*
• Over 1.5 Goals (primary)
• Over 2.5 Goals (secondary)

*Strategy:*
• Flat staking (1-2% per bet)
• Only bet with edge > 5%
• Confidence > 62%

*Commands:*
/today - Daily ticket
/ticket - Current ticket
/analyze - Match analysis
/backtest - System stats
/stats - Your stats

⚠️ Always gamble responsibly
━━━━━━━━━━━━━━━━━━━━━━━━
        """
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors."""
        logger.error(f"Update {update} caused error {context.error}")
    
    def build_application(self) -> Application:
        """Build Telegram application."""
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")
        
        self.application = Application.builder().token(self.token).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("today", self.today_command))
        self.application.add_handler(CommandHandler("ticket", self.ticket_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("backtest", self.backtest_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
        
        return self.application
    
    def run(self):
        """Run the bot."""
        app = self.build_application()
        logger.info("🚀 Starting TelegramSoccer Bot...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def send_ticket_direct(
        self,
        ticket: EnhancedTicket = None,
        chat_id: str = None
    ) -> bool:
        """Send ticket directly without running polling."""
        ticket = ticket or self.current_ticket
        chat_id = chat_id or self.chat_id
        
        if not ticket:
            logger.error("No ticket to send")
            return False
        
        if not self.token or not chat_id:
            logger.error("Telegram credentials not configured")
            return False
        
        bot = Bot(token=self.token)
        message = self.format_ticket_message(ticket)
        
        try:
            await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
            logger.info("✅ Ticket sent to Telegram")
            return True
        except Exception as e:
            logger.error(f"Failed to send: {e}")
            return False


# =============================================================================
# QUICK SEND FUNCTION
# =============================================================================

async def send_daily_ticket():
    """Quick function to generate and send daily ticket."""
    bot = EnhancedTelegramBot()
    
    # Generate ticket
    ticket = bot.pipeline.build_daily_ticket()
    
    if ticket:
        bot.current_ticket = ticket
        success = await bot.send_ticket_direct(ticket)
        return success
    else:
        logger.warning("No valid ticket generated")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="🎯 TelegramSoccer Bot")
    parser.add_argument('--send-only', action='store_true',
                        help='Just send daily ticket without running bot')
    
    args = parser.parse_args()
    
    if args.send_only:
        success = asyncio.run(send_daily_ticket())
        return 0 if success else 1
    else:
        bot = EnhancedTelegramBot()
        bot.run()
        return 0


if __name__ == '__main__':
    sys.exit(main())
