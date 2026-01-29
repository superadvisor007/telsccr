"""
TELEGRAM BOT V2 - LIVE TRACKING & REAL-TIME PREDICTIONS
======================================================

Features:
1. Daily predictions with live bet tracking
2. Real-time performance stats (/stats command)
3. Live bet updates (/bets command)
4. Weekly performance reports
5. Value bet notifications

Commands:
- /start - Initialize bot
- /predict - Get today's predictions
- /stats - View performance statistics
- /bets - View active bets
- /report - Generate weekly report
- /help - Show available commands
"""

import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import asyncio

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.prediction_with_tracking import IntegratedPredictionPipeline
from src.tracking.live_bet_tracker import LiveBetTracker


class TelegramBotV2:
    """
    Telegram Bot V2 with Live Tracking Integration
    """
    
    def __init__(
        self,
        token: str,
        chat_id: Optional[str] = None,
        model_dir: str = "models/knowledge_enhanced"
    ):
        self.token = token
        self.chat_id = chat_id
        self.model_dir = model_dir
        
        # Initialize pipeline and tracker
        self.pipeline = IntegratedPredictionPipeline(model_dir=model_dir)
        self.tracker = LiveBetTracker()
        
        # Build application
        self.app = Application.builder().token(token).build()
        
        # Register command handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all command handlers"""
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("predict", self.cmd_predict))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("bets", self.cmd_bets))
        self.app.add_handler(CommandHandler("report", self.cmd_report))
        
        # Echo handler (for unknown messages)
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.echo))
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_text = """
🎯 **TELEGRAMSOCCER BOT V2**
━━━━━━━━━━━━━━━━━━━━━━

Welcome to the advanced soccer betting assistant!

**Available Commands:**
/predict - Get today's value bets
/stats - View performance stats
/bets - View active bets
/report - Weekly performance report
/help - Show this message

**Features:**
✅ Calibrated ML predictions (ROC-AUC 0.56)
✅ Live bet tracking & ROI monitoring
✅ Expected value calculations
✅ Kelly Criterion staking

Let's get started! Use /predict to see today's tips.
        """
        await update.message.reply_text(welcome_text)
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📚 **COMMAND REFERENCE**
━━━━━━━━━━━━━━━━━━━━━━

**/predict** - Daily Value Bets
Get today's high-value betting opportunities based on advanced ML models.

**/stats [days]** - Performance Statistics
View win rate, ROI, and profit over the last N days (default: 30).
Example: /stats 7

**/bets** - Active Bets
See all pending bets that haven't been settled yet.

**/report [days]** - Performance Report
Generate detailed report including market breakdown and daily performance.
Example: /report 14

**/help** - Show this help message

**Need Support?**
Check the documentation or contact support.
        """
        await update.message.reply_text(help_text)
    
    async def cmd_predict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /predict command - Generate today's predictions"""
        await update.message.reply_text("🔄 Generating predictions... This may take a moment.")
        
        try:
            # For demo, use sample matches
            # In production, fetch from API
            demo_matches = [
                {
                    'match_id': 'bundesliga_001',
                    'home_team': 'Bayern München',
                    'away_team': 'Borussia Dortmund',
                    'league': 'Bundesliga',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'home_elo': 1850,
                    'away_elo': 1720,
                    'home_form': 2.5,
                    'away_form': 1.8,
                    'predicted_home_goals': 2.1,
                    'predicted_away_goals': 1.4,
                },
                {
                    'match_id': 'premier_001',
                    'home_team': 'Manchester City',
                    'away_team': 'Liverpool',
                    'league': 'Premier League',
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'home_elo': 1920,
                    'away_elo': 1880,
                    'home_form': 2.8,
                    'away_form': 2.6,
                    'predicted_home_goals': 2.3,
                    'predicted_away_goals': 1.9,
                },
            ]
            
            # Demo odds
            odds_dict = {
                'bundesliga_001': {
                    'over_1_5': 1.25,
                    'over_2_5': 1.65,
                    'btts': 1.70
                },
                'premier_001': {
                    'over_1_5': 1.20,
                    'over_2_5': 1.50,
                    'btts': 1.60
                }
            }
            
            # Generate predictions
            results = self.pipeline.generate_daily_predictions(
                matches=demo_matches,
                odds_dict=odds_dict,
                bankroll=1000.0
            )
            
            # Format message
            if results['value_bets_found'] > 0:
                message = f"""
🎯 **TODAY'S VALUE BETS** ({datetime.now().strftime('%Y-%m-%d')})
━━━━━━━━━━━━━━━━━━━━━━

**Matches Analyzed:** {results['total_matches']}
**Value Bets Found:** {results['value_bets_found']}
**Total Expected Value:** {results['total_ev']:+.1f}%

"""
                for i, bet in enumerate(results['bets'], 1):
                    message += f"""
**{i}. {bet['home_team']} vs {bet['away_team']}**
   League: {bet['league']}
   Market: {bet['market'].upper()}
   Odds: {bet['odds']:.2f}
   Predicted Probability: {bet['predicted_prob']:.1%}
   Expected Value: {bet['ev']:+.1f}%
   Confidence: {bet['confidence'].upper()}
   Recommended Stake: {bet['kelly_stake']:.1%}

"""
                
                message += "\n✅ All bets logged to tracker. Use /bets to view active bets."
            else:
                message = f"""
📊 **NO VALUE BETS TODAY** ({datetime.now().strftime('%Y-%m-%d')})
━━━━━━━━━━━━━━━━━━━━━━

Analyzed {results['total_matches']} matches, but no bets met our strict value criteria:
• Minimum Expected Value: 5%
• Minimum Confidence: 60%
• Maximum Odds: 2.0

**Quality over quantity!** We only bet when there's clear value.

Check back tomorrow for new opportunities.
                """
            
            await update.message.reply_text(message)
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error generating predictions: {str(e)}")
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - Show performance statistics"""
        # Parse days parameter
        days = 30
        if context.args and len(context.args) > 0:
            try:
                days = int(context.args[0])
            except ValueError:
                days = 30
        
        try:
            stats = self.tracker.get_stats(days=days)
            
            message = f"""
📊 **PERFORMANCE STATISTICS** (Last {days} Days)
━━━━━━━━━━━━━━━━━━━━━━

**Overall Performance**
• Total Bets: {stats['total_bets']:.0f}
• Settled: {stats['settled_bets']:.0f}
• Pending: {stats['pending_bets']:.0f}

**Results**
• Wins: {stats['wins']:.0f}
• Losses: {stats['losses']:.0f}
• Win Rate: {stats['win_rate']:.1%}

**Financial**
• Total Stake: {stats['total_stake']:.2f} €
• Total Profit: {stats['total_profit']:+.2f} €
• ROI: {stats['roi']:+.1f}%
• Average Odds: {stats['avg_odds']:.2f}

"""
            
            # Add status indicator
            if stats['settled_bets'] >= 100:
                if stats['win_rate'] >= 0.56 and stats['roi'] >= 8.0:
                    message += "✅ **TOP 1% PERFORMANCE ACHIEVED!**\n"
                    message += f"Target: >56% WR, >8% ROI\n"
                else:
                    gap_wr = max(0, 56 - stats['win_rate'] * 100)
                    gap_roi = max(0, 8 - stats['roi'])
                    message += f"⚠️ **Gap to Top 1%:**\n"
                    message += f"Win Rate: {gap_wr:.1f} percentage points\n"
                    message += f"ROI: {gap_roi:.1f} percentage points\n"
            else:
                needed = 100 - stats['settled_bets']
                message += f"⏳ Need {needed:.0f} more settled bets for evaluation\n"
            
            await update.message.reply_text(message)
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error retrieving stats: {str(e)}")
    
    async def cmd_bets(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /bets command - Show active bets"""
        try:
            # Get stats to count pending bets
            stats = self.tracker.get_stats(days=7)
            
            if stats['pending_bets'] == 0:
                message = """
📋 **ACTIVE BETS**
━━━━━━━━━━━━━━━━━━━━━━

No pending bets at the moment.

Use /predict to get today's value bets!
                """
            else:
                message = f"""
📋 **ACTIVE BETS** ({stats['pending_bets']:.0f} Pending)
━━━━━━━━━━━━━━━━━━━━━━

These bets are awaiting results:

"""
                # In production, query tracker database for pending bets
                # For now, show summary
                message += f"\nTotal stake at risk: {stats['total_stake']:.2f} €\n"
                message += "\nResults will be updated automatically when matches finish.\n"
                message += "Use /stats to view overall performance."
            
            await update.message.reply_text(message)
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error retrieving bets: {str(e)}")
    
    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /report command - Generate detailed report"""
        # Parse days parameter
        days = 30
        if context.args and len(context.args) > 0:
            try:
                days = int(context.args[0])
            except ValueError:
                days = 30
        
        await update.message.reply_text(f"📊 Generating {days}-day performance report...")
        
        try:
            report = self.tracker.generate_report(days=days)
            
            # Telegram has 4096 character limit, split if needed
            if len(report) <= 4000:
                await update.message.reply_text(f"```\n{report}\n```", parse_mode='Markdown')
            else:
                # Split into chunks
                chunks = [report[i:i+4000] for i in range(0, len(report), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(f"```\n{chunk}\n```", parse_mode='Markdown')
        
        except Exception as e:
            await update.message.reply_text(f"❌ Error generating report: {str(e)}")
    
    async def echo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle unknown messages"""
        await update.message.reply_text(
            "I don't understand that command. Use /help to see available commands."
        )
    
    def run(self):
        """Start the bot"""
        print(f"🤖 Starting Telegram Bot V2...")
        print(f"   Models: {self.model_dir}")
        print(f"   Tracker: data/tracking/live_bets.db")
        print(f"\n✅ Bot is running. Press Ctrl+C to stop.\n")
        
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    # Load token from environment
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set in environment")
        print("   Export it: export TELEGRAM_BOT_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Create and run bot
    bot = TelegramBotV2(
        token=token,
        chat_id=chat_id,
        model_dir="models/knowledge_enhanced"
    )
    
    bot.run()
