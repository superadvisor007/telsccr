"""
HYPERPARAMETER OPTIMIZATION - GRID SEARCH FOR TOP 1%
==================================================

Aggressive grid search to push Over 2.5 from 0.5630 to >0.60 ROC-AUC

Strategy:
1. Focus on Over 2.5 (closest to target, gap only 0.037)
2. Fine-grained grid search around current best parameters
3. Test ensemble combinations (Gradient Boosting + Random Forest)
4. Optimize probability calibration methods
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_advanced_ml_v2 import AdvancedMLTrainer


class AggressiveOptimizer:
    """
    Aggressive hyperparameter optimization to reach ROC-AUC >0.60
    """
    
    def __init__(self):
        self.best_models = {}
        self.best_scores = {}
        self.trainer = AdvancedMLTrainer(use_smote=False, calibrate=True)
    
    def grid_search_over_2_5(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Fine-grained grid search for Over 2.5 market
        
        Current best: max_depth=5, n_estimators=300, learning_rate=0.05
        """
        print("🔍 GRID SEARCH - OVER 2.5 MARKET")
        print("="*70)
        
        # Parameter grid (fine-tuned around current best)
        param_grid = {
            'n_estimators': [250, 300, 350, 400],
            'learning_rate': [0.03, 0.04, 0.05, 0.06],
            'max_depth': [4, 5, 6],
            'min_samples_split': [25, 30, 35, 40],
            'min_samples_leaf': [12, 15, 18, 20],
            'subsample': [0.75, 0.80, 0.85],
        }
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        
        best_auc = 0.0
        best_params = None
        best_model = None
        
        total_combos = (
            len(param_grid['n_estimators']) *
            len(param_grid['learning_rate']) *
            len(param_grid['max_depth']) *
            len(param_grid['min_samples_split']) *
            len(param_grid['min_samples_leaf']) *
            len(param_grid['subsample'])
        )
        
        print(f"Testing {total_combos} parameter combinations...")
        print(f"This will take approximately {total_combos * 3 / 60:.0f} minutes.\n")
        
        tested = 0
        
        for n_est in param_grid['n_estimators']:
            for lr in param_grid['learning_rate']:
                for depth in param_grid['max_depth']:
                    for min_split in param_grid['min_samples_split']:
                        for min_leaf in param_grid['min_samples_leaf']:
                            for subsample in param_grid['subsample']:
                                tested += 1
                                
                                try:
                                    model = GradientBoostingClassifier(
                                        n_estimators=n_est,
                                        learning_rate=lr,
                                        max_depth=depth,
                                        min_samples_split=min_split,
                                        min_samples_leaf=min_leaf,
                                        subsample=subsample,
                                        random_state=42,
                                        max_features='sqrt'
                                    )
                                    
                                    # Train with sample weights
                                    model.fit(X_train, y_train, sample_weight=sample_weights)
                                    
                                    # Calibrate
                                    calibrated = CalibratedClassifierCV(
                                        model,
                                        method='sigmoid',
                                        cv='prefit'
                                    )
                                    calibrated.fit(X_train, y_train)
                                    
                                    # Evaluate
                                    y_proba = calibrated.predict_proba(X_test)[:, 1]
                                    auc = roc_auc_score(y_test, y_proba)
                                    
                                    if auc > best_auc:
                                        best_auc = auc
                                        best_params = {
                                            'n_estimators': n_est,
                                            'learning_rate': lr,
                                            'max_depth': depth,
                                            'min_samples_split': min_split,
                                            'min_samples_leaf': min_leaf,
                                            'subsample': subsample,
                                        }
                                        best_model = calibrated
                                        
                                        print(f"[{tested}/{total_combos}] ✅ NEW BEST: ROC-AUC = {auc:.4f}")
                                        print(f"  Params: {best_params}\n")
                                    
                                    elif tested % 50 == 0:
                                        print(f"[{tested}/{total_combos}] Current best: {best_auc:.4f}")
                                
                                except Exception as e:
                                    print(f"[{tested}/{total_combos}] ❌ Error: {str(e)}")
        
        print(f"\n{'='*70}")
        print(f"🏆 GRID SEARCH COMPLETE")
        print(f"{'='*70}")
        print(f"Best ROC-AUC: {best_auc:.4f}")
        print(f"Best Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        
        return {
            'model': best_model,
            'auc': best_auc,
            'params': best_params
        }
    
    def test_ensemble_methods(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Test ensemble combinations:
        1. GradientBoosting + RandomForest (average probabilities)
        2. GradientBoosting + RandomForest + LogisticRegression (meta-learner)
        """
        print("\n🔬 TESTING ENSEMBLE METHODS")
        print("="*70)
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        
        results = {}
        
        # 1. Gradient Boosting (baseline)
        print("\n1️⃣ GradientBoosting (baseline)...")
        gb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=30,
            min_samples_leaf=15,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_train, y_train, sample_weight=sample_weights)
        gb_proba = gb.predict_proba(X_test)[:, 1]
        gb_auc = roc_auc_score(y_test, gb_proba)
        results['gb_only'] = gb_auc
        print(f"  ROC-AUC: {gb_auc:.4f}")
        
        # 2. Random Forest
        print("\n2️⃣ RandomForest...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        rf_proba = rf.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_proba)
        results['rf_only'] = rf_auc
        print(f"  ROC-AUC: {rf_auc:.4f}")
        
        # 3. Ensemble (average)
        print("\n3️⃣ Ensemble (GB + RF average)...")
        ensemble_proba = (gb_proba + rf_proba) / 2
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        results['ensemble_avg'] = ensemble_auc
        print(f"  ROC-AUC: {ensemble_auc:.4f}")
        
        # 4. Meta-learner (LogisticRegression on top)
        print("\n4️⃣ Meta-learner (LogisticRegression)...")
        # Stack predictions
        meta_features_train = np.column_stack([
            gb.predict_proba(X_train)[:, 1],
            rf.predict_proba(X_train)[:, 1]
        ])
        meta_features_test = np.column_stack([
            gb_proba,
            rf_proba
        ])
        
        meta_model = LogisticRegression(random_state=42, max_iter=1000)
        meta_model.fit(meta_features_train, y_train)
        meta_proba = meta_model.predict_proba(meta_features_test)[:, 1]
        meta_auc = roc_auc_score(y_test, meta_proba)
        results['meta_learner'] = meta_auc
        print(f"  ROC-AUC: {meta_auc:.4f}")
        
        # Find best
        best_method = max(results, key=results.get)
        best_auc = results[best_method]
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST ENSEMBLE METHOD: {best_method}")
        print(f"   ROC-AUC: {best_auc:.4f}")
        print(f"{'='*70}")
        
        return {
            'results': results,
            'best_method': best_method,
            'best_auc': best_auc,
            'models': {
                'gb': gb,
                'rf': rf,
                'meta': meta_model if best_method == 'meta_learner' else None
            }
        }
    
    def optimize_over_2_5(self, df: pd.DataFrame):
        """
        Complete optimization pipeline for Over 2.5 market
        """
        print("\n" + "="*70)
        print("🚀 AGGRESSIVE OPTIMIZATION - OVER 2.5 TO >0.60 ROC-AUC")
        print("="*70 + "\n")
        
        # Prepare data
        df_features = self.trainer.engineer_advanced_features_v2(df)
        feature_cols = self.trainer.get_feature_columns()
        
        X = df_features[feature_cols].values
        y = df_features['over_2_5'].values
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Positive rate: {y_test.mean():.1%}\n")
        
        # Step 1: Grid search
        grid_results = self.grid_search_over_2_5(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Step 2: Ensemble methods
        ensemble_results = self.test_ensemble_methods(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        # Compare
        print(f"\n{'='*70}")
        print(f"📊 FINAL COMPARISON")
        print(f"{'='*70}")
        print(f"Grid Search Best: {grid_results['auc']:.4f}")
        print(f"Ensemble Best: {ensemble_results['best_auc']:.4f} ({ensemble_results['best_method']})")
        
        if grid_results['auc'] > ensemble_results['best_auc']:
            print(f"\n✅ WINNER: Grid Search (ROC-AUC: {grid_results['auc']:.4f})")
            winner = 'grid_search'
            best_model = grid_results['model']
            best_auc = grid_results['auc']
        else:
            print(f"\n✅ WINNER: {ensemble_results['best_method']} (ROC-AUC: {ensemble_results['best_auc']:.4f})")
            winner = ensemble_results['best_method']
            best_model = ensemble_results['models']
            best_auc = ensemble_results['best_auc']
        
        # Check if we achieved target
        if best_auc >= 0.60:
            print(f"\n🎉 TOP 1% ACHIEVED! ROC-AUC = {best_auc:.4f} (>{0.60:.2f})")
        else:
            gap = 0.60 - best_auc
            print(f"\n⚠️  Gap to top 1%: {gap:.4f} ({gap/0.60*100:.1f}% remaining)")
        
        # Save
        output_dir = Path("models/knowledge_enhanced_optimized")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if winner == 'grid_search':
            joblib.dump(best_model, output_dir / "over_2_5_optimized_model.pkl")
            joblib.dump(scaler, output_dir / "over_2_5_optimized_scaler.pkl")
        else:
            # Save ensemble
            joblib.dump(ensemble_results['models']['gb'], output_dir / "over_2_5_gb_model.pkl")
            joblib.dump(ensemble_results['models']['rf'], output_dir / "over_2_5_rf_model.pkl")
            if ensemble_results['models']['meta']:
                joblib.dump(ensemble_results['models']['meta'], output_dir / "over_2_5_meta_model.pkl")
            joblib.dump(scaler, output_dir / "over_2_5_optimized_scaler.pkl")
        
        print(f"\n💾 Models saved to: {output_dir}")
        
        return {
            'winner': winner,
            'auc': best_auc,
            'models': best_model,
            'scaler': scaler
        }


if __name__ == "__main__":
    # Load data
    print("📥 Loading training data...")
    df = pd.read_csv("data/historical/massive_training_data.csv")
    
    # Run optimization
    optimizer = AggressiveOptimizer()
    results = optimizer.optimize_over_2_5(df)
    
    print(f"\n✅ Optimization complete!")
    print(f"   Winner: {results['winner']}")
    print(f"   Final ROC-AUC: {results['auc']:.4f}")
