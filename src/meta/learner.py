"""Meta-learning ensemble that learns to weight multiple models."""
import joblib
import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Tuple

from src.core.database import DatabaseManager


class MetaLearner:
    """
    Meta-learning system that learns optimal weighting of:
    - LLM contextual predictions
    - XGBoost statistical models
    - RL agent recommendations
    - Historical performance in different contexts (league, market, odds range)
    """
    
    def __init__(
        self,
        model_path: str = "models/meta_learner.pkl",
    ):
        self.model_path = model_path
        self.model: Optional[LogisticRegression] = None
        self.feature_names = [
            'llm_probability',
            'llm_confidence',
            'xgboost_probability',
            'rl_stake_recommendation',
            'odds',
            'league_category',
            'market_type',
            'llm_accuracy_league',
            'xgboost_accuracy_league',
            'llm_accuracy_market',
            'xgboost_accuracy_market',
            'recent_llm_performance',
            'recent_xgboost_performance',
        ]
    
    def prepare_training_data(
        self,
        db_manager: DatabaseManager,
        min_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from historical predictions.
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (bet won = 1, bet lost = 0)
        """
        logger.info("Preparing meta-learner training data...")
        
        # Fetch historical predictions with outcomes
        session = db_manager.get_session()
        
        # This is a simplified version - in production, you'd join
        # Tip, Prediction, Match tables and compute all features
        # For now, we'll create synthetic training data structure
        
        X_list = []
        y_list = []
        
        # Placeholder: In real implementation, fetch from database
        # query = session.query(Tip, Prediction, Match).filter(...)
        
        # Example feature vector construction:
        # for tip, pred, match in results:
        #     features = self._extract_features(tip, pred, match)
        #     X_list.append(features)
        #     y_list.append(1 if tip.result == "won" else 0)
        
        session.close()
        
        if len(X_list) < min_samples:
            logger.warning(
                f"Insufficient data for meta-learner training "
                f"({len(X_list)} samples, need {min_samples})"
            )
            return np.array([]), np.array([])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        return X, y
    
    def _extract_features(
        self,
        tip: Dict,
        prediction: Dict,
        match: Dict,
        llm_stats: Dict,
        xgboost_stats: Dict,
    ) -> np.ndarray:
        """
        Extract meta-features for a single prediction.
        
        Args:
            tip: Tip details (market, odds, stake)
            prediction: Model predictions (LLM, XGBoost)
            match: Match context (league, teams)
            llm_stats: Recent LLM performance by league/market
            xgboost_stats: Recent XGBoost performance by league/market
        
        Returns:
            Feature vector matching self.feature_names
        """
        features = [
            prediction.get('llm_probability', 0.5),
            prediction.get('llm_confidence', 0.5),
            prediction.get('xgboost_probability', 0.5),
            prediction.get('rl_stake_pct', 2.0) / 100.0,  # Normalize to 0-1
            tip.get('odds', 1.5),
            self._encode_league_category(match.get('league', '')),
            self._encode_market_type(tip.get('market', 'over_1_5')),
            llm_stats.get('accuracy_in_league', 0.6),
            xgboost_stats.get('accuracy_in_league', 0.6),
            llm_stats.get('accuracy_in_market', 0.6),
            xgboost_stats.get('accuracy_in_market', 0.6),
            llm_stats.get('recent_accuracy_7d', 0.6),
            xgboost_stats.get('recent_accuracy_7d', 0.6),
        ]
        
        return np.array(features)
    
    def _encode_league_category(self, league: str) -> float:
        """Encode league as high-scoring (1) vs low-scoring (0) category."""
        high_scoring_leagues = [
            'Bundesliga', 'Eredivisie', 'Belgian First Division',
        ]
        return 1.0 if league in high_scoring_leagues else 0.0
    
    def _encode_market_type(self, market: str) -> float:
        """Encode market type (Over 1.5 = 0, BTTS = 1)."""
        return 1.0 if 'btts' in market.lower() else 0.0
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Train meta-learner on historical data.
        
        Returns:
            Training metrics (cross-validated accuracy, precision, recall)
        """
        if len(X) == 0:
            raise ValueError("No training data available")
        
        logger.info(f"Training meta-learner on {len(X)} samples...")
        
        # Initialize logistic regression meta-learner
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',  # Handle imbalanced win/loss
            random_state=42,
        )
        
        # Cross-validation before final training
        cv_scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv_folds,
            scoring='accuracy',
        )
        
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Train on full dataset
        self.model.fit(X, y)
        
        # Save model
        joblib.dump(self.model, self.model_path)
        logger.info(f"Meta-learner saved to {self.model_path}")
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_names,
            np.abs(self.model.coef_[0])
        ))
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        logger.info("Top 5 most important features:")
        for feat, importance in sorted_features[:5]:
            logger.info(f"  {feat}: {importance:.3f}")
        
        return {
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(X),
        }
    
    def load(self) -> None:
        """Load trained meta-learner."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Meta-learner loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load meta-learner: {e}")
    
    def predict_ensemble(
        self,
        llm_probability: float,
        llm_confidence: float,
        xgboost_probability: float,
        rl_stake_pct: float,
        odds: float,
        league: str,
        market: str,
        llm_stats: Dict,
        xgboost_stats: Dict,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Meta-learned ensemble prediction.
        
        Returns:
            (final_probability, weights_dict)
        """
        if self.model is None:
            logger.warning("Meta-learner not loaded, using fixed weights")
            return self._fallback_ensemble(
                llm_probability,
                llm_confidence,
                xgboost_probability,
            )
        
        # Construct feature vector
        features = self._extract_features(
            tip={'odds': odds, 'market': market},
            prediction={
                'llm_probability': llm_probability,
                'llm_confidence': llm_confidence,
                'xgboost_probability': xgboost_probability,
                'rl_stake_pct': rl_stake_pct,
            },
            match={'league': league},
            llm_stats=llm_stats,
            xgboost_stats=xgboost_stats,
        )
        
        # Get meta-prediction (probability of bet winning)
        meta_probability = self.model.predict_proba(features.reshape(1, -1))[0][1]
        
        # Derive effective weights from logistic regression coefficients
        coeffs = self.model.coef_[0]
        llm_weight = np.abs(coeffs[0]) / (np.abs(coeffs[0]) + np.abs(coeffs[2]))
        xgboost_weight = 1.0 - llm_weight
        
        weights = {
            'llm': llm_weight,
            'xgboost': xgboost_weight,
            'meta_override': 0.2,  # Meta-learner gets 20% say
        }
        
        # Blend: 80% weighted average, 20% meta-learner
        weighted_avg = (llm_weight * llm_probability + xgboost_weight * xgboost_probability)
        final_probability = 0.8 * weighted_avg + 0.2 * meta_probability
        
        logger.debug(
            f"Meta-ensemble: LLM={llm_probability:.3f} ({llm_weight:.2f}), "
            f"XGB={xgboost_probability:.3f} ({xgboost_weight:.2f}), "
            f"Meta={meta_probability:.3f} → Final={final_probability:.3f}"
        )
        
        return final_probability, weights
    
    def _fallback_ensemble(
        self,
        llm_probability: float,
        llm_confidence: float,
        xgboost_probability: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Fallback ensemble when meta-learner unavailable."""
        # Confidence-weighted blend (max 70% LLM weight)
        llm_weight = min(llm_confidence, 0.7)
        xgboost_weight = 1.0 - llm_weight
        
        final_probability = (
            llm_weight * llm_probability +
            xgboost_weight * xgboost_probability
        )
        
        return final_probability, {
            'llm': llm_weight,
            'xgboost': xgboost_weight,
            'meta_override': 0.0,
        }


class ContextualPerformanceTracker:
    """Tracks model performance by context (league, market, odds range)."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def get_model_stats(
        self,
        model_name: str,
        league: Optional[str] = None,
        market: Optional[str] = None,
        days_back: int = 30,
    ) -> Dict[str, float]:
        """
        Get recent performance stats for a model in specific context.
        
        Returns:
            {'accuracy_in_league': 0.65, 'accuracy_in_market': 0.62, 'recent_accuracy_7d': 0.68}
        """
        # Placeholder implementation
        # In production, query database for:
        # - Predictions where source = model_name
        # - Filter by league, market, date range
        # - Calculate win rate
        
        return {
            'accuracy_in_league': 0.65,
            'accuracy_in_market': 0.62,
            'recent_accuracy_7d': 0.68,
        }
