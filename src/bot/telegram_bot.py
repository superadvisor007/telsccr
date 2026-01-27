"""Telegram bot interface for daily tips."""
from datetime import datetime
from typing import Any, Dict, List

from loguru import logger
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.core.config import settings


class TelegramBot:
    """Telegram bot for delivering betting tips."""
    
    def __init__(self):
        self.token = settings.telegram.bot_token
        self.admin_ids = settings.telegram.admin_ids
        self.application = None
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        welcome_message = """
🎯 *Welcome to TelegramSoccer Bot!*

I provide daily low-odds accumulator tips for soccer betting, focusing on:
• Over 1.5 Goals
• Both Teams to Score (BTTS)
• Target quote: ~1.40

*Commands:*
/start - Show this message
/today - Get today's tips
/stats - View betting statistics
/bankroll - Check current bankroll
/help - Get help

Powered by AI analysis (LLM + Statistical Models)
⚠️ Bet responsibly. Gambling involves risk.
        """
        await update.message.reply_text(welcome_message, parse_mode="Markdown")
        logger.info(f"User {update.effective_user.id} started bot")
    
    async def today_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /today command - show today's tips."""
        # This will be populated by the pipeline
        tips = context.bot_data.get("today_tips", [])
        
        if not tips:
            await update.message.reply_text(
                "📭 No tips available for today yet. Check back after 9:00 AM UTC!"
            )
            return
        
        message = self._format_tips_message(tips)
        await update.message.reply_text(message, parse_mode="Markdown")
        logger.info(f"User {update.effective_user.id} requested today's tips")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /stats command."""
        stats = context.bot_data.get("betting_stats", {})
        
        if not stats:
            await update.message.reply_text("📊 No statistics available yet.")
            return
        
        message = f"""
📊 *Betting Statistics*

*Overall Performance:*
• Total Bets: {stats.get('total_bets', 0)}
• Wins: {stats.get('wins', 0)} ✅
• Losses: {stats.get('losses', 0)} ❌
• Pending: {stats.get('pending', 0)} ⏳
• Win Rate: {stats.get('win_rate', 0):.1f}%

*Financial:*
• Profit/Loss: €{stats.get('profit_loss', 0):.2f}
• ROI: {stats.get('roi', 0):.2f}%
• Bankroll Change: {stats.get('bankroll_change', 0):+.1f}%
        """
        
        await update.message.reply_text(message, parse_mode="Markdown")
    
    async def bankroll_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /bankroll command."""
        stats = context.bot_data.get("betting_stats", {})
        bankroll = stats.get("current_bankroll", 0)
        change = stats.get("bankroll_change", 0)
        
        emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        
        message = f"""
💰 *Current Bankroll*

Balance: €{bankroll:.2f}
Change: {change:+.1f}% {emoji}
        """
        
        await update.message.reply_text(message, parse_mode="Markdown")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = """
ℹ️ *Help - TelegramSoccer Bot*

*What is this bot?*
An AI-powered betting assistant that analyzes soccer matches and provides daily accumulator tips with odds around 1.40.

*How it works:*
1. Data Collection: Weather, team stats, injuries, odds
2. AI Analysis: LLM + XGBoost models assess each match
3. Value Detection: Find bets where probability > implied odds
4. Accumulator Building: Combine 2-3 selections for ~1.40 quote

*Market Focus:*
• Over 1.5 Goals (1.20-1.50 odds)
• Both Teams to Score (1.30-1.60 odds)

*Betting Strategy:*
• Fixed staking: 1-2% per bet
• Stop-loss: 15% drawdown
• Only bet when EV > 0

*Commands:*
/today - Daily tips (available after 9 AM UTC)
/stats - Performance tracking
/bankroll - Current balance

⚠️ *Disclaimer:* This is an analytical tool. Always bet responsibly and within your means.
        """
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def broadcast_tips(self, tips: List[Dict[str, Any]], subscribers: List[int]) -> None:
        """Broadcast tips to all subscribers."""
        message = self._format_tips_message(tips)
        
        for user_id in subscribers:
            try:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=message,
                    parse_mode="Markdown"
                )
                logger.info(f"Tips sent to user {user_id}")
            except Exception as e:
                logger.error(f"Failed to send tips to {user_id}: {e}")
    
    def _format_tips_message(self, tips: List[Dict[str, Any]]) -> str:
        """Format tips into readable message."""
        if not tips:
            return "📭 No tips available."
        
        message = f"🎯 *Daily Tips - {datetime.utcnow().strftime('%Y-%m-%d')}*\n\n"
        
        for i, tip in enumerate(tips, 1):
            acc = tip.get("accumulator", {})
            selections = acc.get("selections", [])
            
            message += f"*Tip #{i}:* {len(selections)}-Leg Accumulator\n"
            message += f"💰 Total Odds: {acc.get('total_odds', 0):.2f}\n"
            message += f"📊 Combined Probability: {acc.get('combined_probability', 0):.1%}\n"
            message += f"💵 Suggested Stake: €{tip.get('stake', 0):.2f}\n\n"
            
            message += "*Selections:*\n"
            for j, sel in enumerate(selections, 1):
                message += f"{j}. {sel['match_info']}\n"
                message += f"   • Market: {sel['market'].upper()}\n"
                message += f"   • Odds: {sel['odds']:.2f}\n"
                message += f"   • Probability: {sel['researched_probability']:.1%}\n"
                if sel.get('key_factors'):
                    message += f"   • Key: {sel['key_factors'][0][:60]}...\n"
                message += "\n"
            
            message += f"⚡ EV: {acc.get('expected_value', 0):.2%}\n"
            message += "─" * 30 + "\n\n"
        
        message += "⚠️ *Risk Management:*\n"
        message += "• Only bet what you can afford to lose\n"
        message += "• Follow 1-2% staking rule\n"
        message += "• Review factors before placing\n\n"
        message += "_Analysis by AI (LLM + XGBoost)_"
        
        return message
    
    def run(self) -> None:
        """Run the bot."""
        self.application = Application.builder().token(self.token).build()
        
        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("today", self.today_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("bankroll", self.bankroll_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        
        # Run bot
        logger.info("Telegram bot started")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def update_bot_data(self, tips: List[Dict], stats: Dict) -> None:
        """Update bot data with latest tips and stats."""
        if self.application:
            self.application.bot_data["today_tips"] = tips
            self.application.bot_data["betting_stats"] = stats
            logger.info("Bot data updated")
