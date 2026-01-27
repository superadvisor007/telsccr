#!/usr/bin/env python3
"""
System Health Monitoring
Tracks: Win Rate, Model Accuracy, API Health, Telegram Delivery
Sends Alerts wenn Performance drops
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import requests


@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: str
    total_predictions: int = 0
    correct_predictions: int = 0
    win_rate: float = 0.0
    api_success_rate: float = 100.0
    telegram_delivery_rate: float = 100.0
    last_prediction_time: Optional[str] = None
    status: str = "HEALTHY"
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SystemHealthMonitor:
    """Monitor system health and send alerts"""
    
    def __init__(self):
        self.metrics_file = Path("data/monitoring/health_metrics.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Alert thresholds
        self.min_win_rate = 0.52  # Below 52% = problem
        self.min_api_success = 0.80  # Below 80% = API issues
        self.max_hours_no_prediction = 48  # 48h no prediction = problem
    
    def record_prediction(self, correct: bool, api_success: bool, telegram_sent: bool):
        """Record a new prediction and update metrics"""
        metrics = self._load_metrics()
        
        metrics.total_predictions += 1
        if correct:
            metrics.correct_predictions += 1
        
        metrics.win_rate = (metrics.correct_predictions / metrics.total_predictions 
                           if metrics.total_predictions > 0 else 0.0)
        
        # Update rates (simple running average)
        metrics.api_success_rate = ((metrics.api_success_rate * 0.9) + 
                                   (100 if api_success else 0) * 0.1)
        metrics.telegram_delivery_rate = ((metrics.telegram_delivery_rate * 0.9) + 
                                         (100 if telegram_sent else 0) * 0.1)
        
        metrics.last_prediction_time = datetime.now().isoformat()
        metrics.timestamp = datetime.now().isoformat()
        
        # Check health status
        metrics.status = self._assess_health(metrics)
        
        # Save
        self._save_metrics(metrics)
        
        # Send alert if unhealthy
        if metrics.status != "HEALTHY":
            self._send_alert(metrics)
        
        return metrics
    
    def _assess_health(self, metrics: HealthMetrics) -> str:
        """Assess overall system health"""
        issues = []
        
        # Check win rate (only if we have enough data)
        if metrics.total_predictions >= 20 and metrics.win_rate < self.min_win_rate:
            issues.append(f"Low win rate: {metrics.win_rate:.1%}")
        
        # Check API health
        if metrics.api_success_rate < self.min_api_success * 100:
            issues.append(f"API issues: {metrics.api_success_rate:.0f}%")
        
        # Check telegram delivery
        if metrics.telegram_delivery_rate < 90:
            issues.append(f"Telegram failures: {metrics.telegram_delivery_rate:.0f}%")
        
        # Check last prediction time
        if metrics.last_prediction_time:
            last_pred = datetime.fromisoformat(metrics.last_prediction_time)
            hours_since = (datetime.now() - last_pred).total_seconds() / 3600
            if hours_since > self.max_hours_no_prediction:
                issues.append(f"No predictions for {hours_since:.0f}h")
        
        if not issues:
            return "HEALTHY"
        elif len(issues) == 1:
            return "WARNING"
        else:
            return "CRITICAL"
    
    def _load_metrics(self) -> HealthMetrics:
        """Load metrics from file"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                return HealthMetrics(**data)
            except Exception:
                pass
        
        return HealthMetrics(timestamp=datetime.now().isoformat())
    
    def _save_metrics(self, metrics: HealthMetrics):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def _send_alert(self, metrics: HealthMetrics):
        """Send Telegram alert for unhealthy system"""
        from config.telegram_config import get_bot_token, get_chat_id
        
        status_emoji = {
            "HEALTHY": "✅",
            "WARNING": "⚠️",
            "CRITICAL": "🚨"
        }
        
        message = f"""
{status_emoji[metrics.status]} **SYSTEM HEALTH ALERT**

**Status:** {metrics.status}

**Metrics:**
• Win Rate: {metrics.win_rate:.1%} (target: ≥52%)
• Total Predictions: {metrics.total_predictions}
• Correct: {metrics.correct_predictions}
• API Success: {metrics.api_success_rate:.0f}%
• Telegram Delivery: {metrics.telegram_delivery_rate:.0f}%

**Last Prediction:** {metrics.last_prediction_time or 'Never'}

**Action Required:** Check logs and investigate issues!
"""
        
        try:
            url = f"https://api.telegram.org/bot{get_bot_token()}/sendMessage"
            requests.post(
                url,
                json={'chat_id': get_chat_id(), 'text': message, 'parse_mode': 'Markdown'},
                timeout=10
            )
            print(f"🚨 Alert sent: System {metrics.status}")
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
    
    def get_current_health(self) -> Dict:
        """Get current health status"""
        metrics = self._load_metrics()
        return metrics.to_dict()
    
    def print_health_report(self):
        """Print health report to console"""
        metrics = self._load_metrics()
        
        print("\n" + "="*60)
        print("📊 SYSTEM HEALTH REPORT")
        print("="*60)
        print(f"Status: {metrics.status}")
        print(f"Win Rate: {metrics.win_rate:.1%} ({metrics.correct_predictions}/{metrics.total_predictions})")
        print(f"API Success: {metrics.api_success_rate:.0f}%")
        print(f"Telegram Delivery: {metrics.telegram_delivery_rate:.0f}%")
        print(f"Last Prediction: {metrics.last_prediction_time or 'Never'}")
        print("="*60 + "\n")


# Test
if __name__ == "__main__":
    monitor = SystemHealthMonitor()
    
    # Simulate some predictions
    print("📝 Simulating predictions...\n")
    monitor.record_prediction(correct=True, api_success=True, telegram_sent=True)
    monitor.record_prediction(correct=True, api_success=True, telegram_sent=True)
    monitor.record_prediction(correct=False, api_success=True, telegram_sent=True)
    
    # Show report
    monitor.print_health_report()
