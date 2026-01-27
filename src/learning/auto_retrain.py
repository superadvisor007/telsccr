#!/usr/bin/env python3
"""Automated Retraining System - Battle Tested"""
import json
from pathlib import Path
from datetime import datetime, timedelta

class AutoRetrainer:
    """Triggers retraining when performance drops"""
    
    def __init__(self):
        self.metrics_file = Path("data/monitoring/health_metrics.json")
        self.last_retrain_file = Path("data/monitoring/last_retrain.json")
        self.min_win_rate = 0.52
        self.min_predictions_before_retrain = 20
        self.min_days_between_retrains = 7
    
    def should_retrain(self) -> tuple[bool, str]:
        """Check if retraining is needed"""
        # Load current metrics
        if not self.metrics_file.exists():
            return False, "No metrics available yet"
        
        with open(self.metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Check last retrain time
        if self.last_retrain_file.exists():
            with open(self.last_retrain_file, 'r') as f:
                last_retrain = json.load(f)
            
            last_date = datetime.fromisoformat(last_retrain['timestamp'])
            days_since = (datetime.now() - last_date).days
            
            if days_since < self.min_days_between_retrains:
                return False, f"Last retrain was {days_since} days ago (min: {self.min_days_between_retrains})"
        
        # Check if enough predictions
        if metrics['total_predictions'] < self.min_predictions_before_retrain:
            return False, f"Only {metrics['total_predictions']} predictions (need {self.min_predictions_before_retrain})"
        
        # Check win rate
        if metrics['win_rate'] < self.min_win_rate:
            return True, f"Win rate {metrics['win_rate']:.1%} below threshold {self.min_win_rate:.1%}"
        
        return False, "Performance is good, no retrain needed"
    
    def trigger_retrain(self):
        """Execute retraining"""
        print("🔄 TRIGGERING AUTOMATED RETRAINING...")
        
        # Record retrain
        self.last_retrain_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.last_retrain_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'reason': 'Low win rate detected'
            }, f)
        
        # Execute training script
        import subprocess
        try:
            result = subprocess.run(
                ["python3", "src/finetuning/train_knowledge_models.py"],
                capture_output=True,
                timeout=300
            )
            if result.returncode == 0:
                print("✅ Retraining completed successfully")
                return True
            else:
                print(f"❌ Retraining failed: {result.stderr.decode()}")
                return False
        except Exception as e:
            print(f"❌ Retraining error: {e}")
            return False

# Test
if __name__ == "__main__":
    retrainer = AutoRetrainer()
    should, reason = retrainer.should_retrain()
    print(f"Should retrain: {should}")
    print(f"Reason: {reason}")
    print("✅ Auto Retrainer works!")
