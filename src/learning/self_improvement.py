"""
Self-Learning & Improvement Pipeline
Analyzes mistakes, updates models, improves predictions over time
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import joblib
from datetime import datetime
import json
import os

class SelfImprovementSystem:
    """
    Continuous learning system that:
    1. Analyzes prediction errors
    2. Identifies patterns in mistakes
    3. Updates feature weights
    4. Retrains models with new data
    """
    
    def __init__(self):
        self.error_log = []
        self.improvement_metrics = {
            'weekly_accuracy': [],
            'monthly_roi': [],
            'model_versions': []
        }
    
    def analyze_errors(self, verified_predictions: List[Dict]) -> Dict:
        """
        Analyze where and why predictions failed
        
        Returns insights about error patterns
        """
        print("\n🔍 Analyzing Prediction Errors...")
        
        errors = [p for p in verified_predictions if p['outcome'] == False]
        
        if not errors:
            print("  ✅ No errors to analyze!")
            return {'error_count': 0}
        
        df = pd.DataFrame(errors)
        
        # Error patterns
        error_by_market = df.groupby('prediction').apply(lambda x: len(x)).to_dict() if 'prediction' in df.columns else {}
        error_by_league = df.groupby('result').apply(lambda x: x['league'].iloc[0] if 'league' in x['result'].iloc[0] else 'Unknown').value_counts().to_dict() if 'result' in df.columns else {}
        
        # Analyze predicted vs actual probabilities
        high_confidence_errors = [e for e in errors if e.get('prediction', {}).get('probability', 0) > 0.70]
        low_confidence_errors = [e for e in errors if e.get('prediction', {}).get('probability', 0) < 0.60]
        
        # Overconfidence analysis
        overconfident = len(high_confidence_errors)
        underconfident = len([p for p in verified_predictions if p['outcome'] == True and p.get('prediction', {}).get('probability', 0) < 0.60])
        
        insights = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(verified_predictions),
            'error_by_market': error_by_market,
            'error_by_league': error_by_league,
            'high_confidence_errors': overconfident,
            'low_confidence_errors': len(low_confidence_errors),
            'overconfidence_ratio': overconfident / len(errors) if errors else 0,
            'calibration_needed': overconfident > len(errors) * 0.3,  # >30% overconfident
            'raw_errors': errors
        }
        
        print(f"\n📊 ERROR ANALYSIS:")
        print(f"   Total Errors:     {insights['total_errors']}")
        print(f"   Error Rate:       {insights['error_rate']*100:.1f}%")
        print(f"   Overconfident:    {overconfident} ({insights['overconfidence_ratio']*100:.1f}%)")
        print(f"   Calibration Fix:  {'⚠️  NEEDED' if insights['calibration_needed'] else '✅ OK'}")
        
        # Save error log
        self.error_log.extend(errors)
        self._save_error_log()
        
        return insights
    
    def identify_feature_importance_shifts(self, old_model_path: str, new_data: pd.DataFrame) -> Dict:
        """
        Compare feature importance between old model and new data
        Identifies if feature relationships have changed (concept drift)
        """
        print("\n📈 Analyzing Feature Importance Shifts...")
        
        try:
            # Load old feature importance
            old_importance = pd.read_csv(f"{old_model_path}_feature_importance.csv")
            
            # Train temp model on new data to compare
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            
            feature_cols = [
                'home_elo', 'away_elo', 'elo_diff', 'elo_home_strength', 'elo_away_strength',
                'predicted_home_goals', 'predicted_away_goals', 'predicted_total_goals',
                'home_form', 'away_form', 'form_advantage', 'league_avg_goals',
                'league_over_2_5_rate', 'league_btts_rate', 'elo_total_strength',
                'elo_gap', 'predicted_goals_diff'
            ]
            
            X = new_data[feature_cols].fillna(0).values
            y = new_data['over_2_5'].values  # Use one market as example
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            new_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Compare top features
            old_top_5 = set(old_importance.head(5)['feature'].values)
            new_top_5 = set(new_importance.head(5)['feature'].values)
            
            drift_detected = len(old_top_5.intersection(new_top_5)) < 3  # Less than 3 common features
            
            insights = {
                'drift_detected': drift_detected,
                'old_top_features': old_top_5,
                'new_top_features': new_top_5,
                'common_features': old_top_5.intersection(new_top_5),
                'new_importance': new_importance.to_dict('records')
            }
            
            print(f"\n🔄 CONCEPT DRIFT ANALYSIS:")
            print(f"   Drift Detected:    {'⚠️  YES' if drift_detected else '✅ NO'}")
            print(f"   Old Top Features:  {', '.join(old_top_5)}")
            print(f"   New Top Features:  {', '.join(new_top_5)}")
            
            if drift_detected:
                print(f"   ⚠️  RECOMMENDATION: Retrain models with new data!")
            
            return insights
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'error': str(e)}
    
    def suggest_model_improvements(self, error_insights: Dict, verification_stats: Dict) -> List[str]:
        """
        Generate actionable improvement suggestions based on analysis
        """
        print("\n💡 Generating Improvement Suggestions...")
        
        suggestions = []
        
        # Check win rate
        win_rate = verification_stats.get('win_rate', 0)
        if win_rate < 0.55:
            suggestions.append("⚠️  Win rate below 55% - Consider increasing minimum edge threshold")
            suggestions.append("📊 Analyze market-specific performance - some markets may be unprofitable")
        
        # Check calibration
        if error_insights.get('calibration_needed'):
            suggestions.append("🎯 Calibration needed - Apply Platt scaling or isotonic regression")
            suggestions.append("📉 Reduce confidence multiplier to prevent overconfidence")
        
        # Check overconfidence
        overconf_ratio = error_insights.get('overconfidence_ratio', 0)
        if overconf_ratio > 0.4:
            suggestions.append(f"⚠️  {overconf_ratio*100:.0f}% of errors were high-confidence - Review probability adjustments")
        
        # Check error concentration
        error_by_market = error_insights.get('error_by_market', {})
        if error_by_market:
            worst_market = max(error_by_market.items(), key=lambda x: x[1])
            suggestions.append(f"🎲 Market '{worst_market[0]}' has most errors ({worst_market[1]}) - May need separate model")
        
        # General suggestions
        if len(suggestions) == 0:
            suggestions.append("✅ System performing well - Continue monitoring")
            suggestions.append("📈 Consider A/B testing with alternative models")
        
        print(f"\n📋 IMPROVEMENT SUGGESTIONS ({len(suggestions)}):")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        return suggestions
    
    def trigger_retraining(self, new_data: pd.DataFrame, model_dir: str = 'models/knowledge_enhanced') -> bool:
        """
        Trigger model retraining with new data
        
        Returns True if retraining successful
        """
        print("\n🔄 Triggering Model Retraining...")
        
        try:
            # Check if enough new data
            if len(new_data) < 50:
                print(f"   ⚠️  Only {len(new_data)} new matches - need at least 50 for retraining")
                return False
            
            print(f"   📊 Retraining with {len(new_data)} new matches")
            
            # Load existing training data
            existing_data_path = 'data/historical/massive_training_data.csv'
            if os.path.exists(existing_data_path):
                existing_data = pd.read_csv(existing_data_path)
                print(f"   📂 Loaded {len(existing_data)} existing matches")
                
                # Combine with new data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                
                # Remove duplicates
                combined_data = combined_data.drop_duplicates(subset=['date', 'home_team', 'away_team'])
                
                print(f"   ✅ Combined dataset: {len(combined_data)} matches")
            else:
                combined_data = new_data
            
            # Save updated training data
            combined_data.to_csv(existing_data_path, index=False)
            print(f"   💾 Saved updated training data")
            
            # Retrain models (call external script)
            import subprocess
            result = subprocess.run(
                ['python3', 'train_knowledge_enhanced_ml.py'],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                print(f"   ✅ Retraining successful!")
                
                # Update version tracking
                version_info = {
                    'timestamp': datetime.now().isoformat(),
                    'total_matches': len(combined_data),
                    'new_matches': len(new_data),
                    'model_dir': model_dir
                }
                self.improvement_metrics['model_versions'].append(version_info)
                self._save_improvement_metrics()
                
                return True
            else:
                print(f"   ❌ Retraining failed: {result.stderr}")
                return False
        
        except Exception as e:
            print(f"   ❌ Error during retraining: {e}")
            return False
    
    def _save_error_log(self):
        """Save error log to disk"""
        os.makedirs('data/learning', exist_ok=True)
        with open('data/learning/error_log.json', 'w') as f:
            json.dump(self.error_log, f, indent=2, default=str)
    
    def _save_improvement_metrics(self):
        """Save improvement metrics"""
        os.makedirs('data/learning', exist_ok=True)
        with open('data/learning/improvement_metrics.json', 'w') as f:
            json.dump(self.improvement_metrics, f, indent=2, default=str)
    
    def weekly_improvement_cycle(self, start_date: str, end_date: str) -> Dict:
        """
        Complete weekly improvement cycle:
        1. Collect results for past week
        2. Verify predictions
        3. Analyze errors
        4. Retrain if needed
        """
        print("\n" + "="*80)
        print("🔄 WEEKLY SELF-IMPROVEMENT CYCLE")
        print("="*80)
        print(f"📅 Period: {start_date} to {end_date}")
        
        # Step 1: Collect results
        from src.ingestion.result_collector import ResultCollector
        collector = ResultCollector()
        
        # Mock predictions for demonstration (in production, load from database)
        predictions = []  # Would load actual predictions here
        
        if not predictions:
            print("\n⏭️  No predictions to verify this week")
            return {'status': 'no_predictions'}
        
        # Step 2: Verify predictions
        verification_stats = collector.bulk_verify_predictions(predictions, start_date, end_date)
        
        # Step 3: Analyze errors
        error_insights = self.analyze_errors(verification_stats['verified_predictions'])
        
        # Step 4: Generate suggestions
        suggestions = self.suggest_model_improvements(error_insights, verification_stats)
        
        # Step 5: Decide if retraining needed
        should_retrain = (
            verification_stats['win_rate'] < 0.53 or  # Win rate dropped
            error_insights['calibration_needed'] or  # Calibration off
            len(verification_stats['verified_predictions']) >= 100  # Enough new data
        )
        
        if should_retrain:
            print("\n🔄 Retraining triggered...")
            # Convert verified predictions to training data
            new_training_data = self._convert_to_training_data(verification_stats['verified_predictions'])
            success = self.trigger_retraining(new_training_data)
        else:
            print("\n✅ Models performing well - no retraining needed")
            success = True
        
        # Track metrics
        self.improvement_metrics['weekly_accuracy'].append({
            'week': end_date,
            'accuracy': 1 - error_insights['error_rate'],
            'win_rate': verification_stats['win_rate']
        })
        self._save_improvement_metrics()
        
        return {
            'verification_stats': verification_stats,
            'error_insights': error_insights,
            'suggestions': suggestions,
            'retrained': should_retrain and success
        }
    
    def _convert_to_training_data(self, verified_predictions: List[Dict]) -> pd.DataFrame:
        """Convert verified predictions back to training format"""
        rows = []
        
        for vp in verified_predictions:
            pred = vp['prediction']
            result = vp['result']
            
            row = {
                'date': result['date'],
                'home_team': result['home_team'],
                'away_team': result['away_team'],
                'home_goals': result['home_score'],
                'away_goals': result['away_score'],
                'league': result['league'],
                # Add features if available in prediction
                **pred  # Merge prediction features
            }
            rows.append(row)
        
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test self-improvement system
    system = SelfImprovementSystem()
    
    print("🧪 Testing Self-Improvement System...")
    
    # Mock error analysis
    mock_errors = [
        {
            'prediction': {'market': 'over_2_5', 'probability': 0.75},
            'result': {'home_team': 'Team A', 'away_team': 'Team B', 'home_score': 1, 'away_score': 0, 'league': 'Premier League'},
            'outcome': False
        },
        {
            'prediction': {'market': 'btts', 'probability': 0.68},
            'result': {'home_team': 'Team C', 'away_team': 'Team D', 'home_score': 2, 'away_score': 0, 'league': 'Bundesliga'},
            'outcome': False
        }
    ]
    
    error_insights = system.analyze_errors(mock_errors)
    
    mock_stats = {'win_rate': 0.52, 'total_verified': 10}
    suggestions = system.suggest_model_improvements(error_insights, mock_stats)
    
    print("\n✅ Self-Improvement System Test Complete")
