#!/usr/bin/env python3
"""
Professional ML Training Pipeline
Complete workflow: Data → Features → Training → Validation → Deployment
"""
import sys
sys.path.append('/workspaces/telegramsoccer')

from pathlib import Path
import pandas as pd
from src.ingestion.historical_data_collector import HistoricalDataCollector
from src.models.professional_model import ProfessionalSoccerModel


def main():
    """Complete training pipeline"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   PROFESSIONAL SOCCER ML TRAINING PIPELINE                               ║
║   Replacing amateur LLM prompting with statistical edge                  ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Collect historical data
    print("\n" + "="*80)
    print("STEP 1: HISTORICAL DATA COLLECTION")
    print("="*80 + "\n")
    
    training_data_path = Path('data/training_data.csv')
    
    if not training_data_path.exists():
        print("📥 Collecting historical Bundesliga data (3 seasons)...")
        collector = HistoricalDataCollector()
        
        historical_matches = collector.collect_bundesliga_history(
            seasons=['2023', '2022', '2021'],
            league='bl1'
        )
        
        if len(historical_matches) > 0:
            enriched_data = collector.enrich_with_features(historical_matches)
            collector.save_training_data(enriched_data, training_data_path)
        else:
            print("❌ Failed to collect data")
            return
    else:
        print(f"✅ Training data already exists: {training_data_path}")
    
    # Step 2: Load training data
    print("\n" + "="*80)
    print("STEP 2: LOADING TRAINING DATA")
    print("="*80 + "\n")
    
    df = pd.read_csv(training_data_path)
    print(f"✅ Loaded {len(df)} matches")
    print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"   Features: {len(df.columns) - 7}")
    print()
    
    # Data split: Train (70%), Validation (15%), Test (15%)
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    print(f"📊 Data split:")
    print(f"   Training: {len(train_df)} matches ({len(train_df)/len(df):.0%})")
    print(f"   Validation: {len(val_df)} matches ({len(val_df)/len(df):.0%})")
    print(f"   Test: {len(test_df)} matches ({len(test_df)/len(df):.0%})")
    print()
    
    # Step 3: Train models for each market
    print("\n" + "="*80)
    print("STEP 3: TRAINING ML MODELS")
    print("="*80 + "\n")
    
    model = ProfessionalSoccerModel(model_type='xgboost')
    
    markets = ['over_1_5', 'over_2_5', 'btts', 'under_1_5']
    
    for market in markets:
        print(f"\n{'─'*80}")
        print(f"Training {market.upper()} model...")
        print(f"{'─'*80}\n")
        
        X_train, y_train = model.prepare_training_data(train_df, market)
        X_val, y_val = model.prepare_training_data(val_df, market)
        
        # Combine for training with validation split
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])
        
        metrics = model.train_model(
            X_combined, y_combined,
            market=market,
            validation_split=0.2,
            use_time_series_split=True
        )
    
    # Step 4: Calibrate models
    print("\n" + "="*80)
    print("STEP 4: MODEL CALIBRATION")
    print("="*80 + "\n")
    
    print("🔧 Calibrating probabilities for betting accuracy...")
    for market in markets:
        X_val, y_val = model.prepare_training_data(val_df, market)
        try:
            model.calibrate_model(X_val, y_val, market=market, method='isotonic')
        except Exception as e:
            print(f"   ⚠️  Calibration failed for {market}: {e}")
    
    # Step 5: Backtest on test set
    print("\n" + "="*80)
    print("STEP 5: COMPREHENSIVE BACKTESTING")
    print("="*80 + "\n")
    
    backtest_results = model.backtest(
        test_df,
        markets=markets,
        initial_bankroll=1000.0,
        kelly_fraction=0.25,
        min_edge=0.05
    )
    
    # Step 6: Save models
    print("\n" + "="*80)
    print("STEP 6: SAVING TRAINED MODELS")
    print("="*80 + "\n")
    
    models_dir = Path('models/trained')
    model.save_models(models_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("🎯 TRAINING PIPELINE COMPLETE")
    print("="*80 + "\n")
    
    if backtest_results:
        print("📊 PROFESSIONAL SYSTEM PERFORMANCE:")
        print(f"   ROI: {backtest_results['roi']:+.2f}%")
        print(f"   Win Rate: {backtest_results['win_rate']:.2%}")
        print(f"   Bankroll Growth: {backtest_results['bankroll_change_pct']:+.2f}%")
        print()
        
        if backtest_results['roi'] > 5:
            print("✅ VERDICT: SYSTEM HAS GENUINE EDGE")
            print("   Ready for production deployment")
        elif backtest_results['roi'] > 0:
            print("⚠️  VERDICT: MARGINAL EDGE")
            print("   Consider feature engineering improvements")
        else:
            print("❌ VERDICT: NO STATISTICAL EDGE")
            print("   System needs major revision")
    
    print("\n🚀 Next steps:")
    print("   1. Run ./run_daily_predictions.sh for live predictions")
    print("   2. Monitor CLV (Closing Line Value) to validate edge")
    print("   3. Continuously retrain with new match data")
    print()


if __name__ == '__main__':
    main()
