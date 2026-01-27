"""Statistical models for match outcome prediction."""
import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xgboost as xgb
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


class PredictionModel:
    """Ensemble prediction model for Over 1.5 and BTTS markets."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.over_1_5_model: Optional[xgb.XGBClassifier] = None
        self.btts_model: Optional[xgb.XGBClassifier] = None
        
        # Try to load existing models
        self._load_models()
    
    def _initialize_model(self) -> xgb.XGBClassifier:
        """Initialize XGBoost classifier with optimized parameters."""
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1,
        )
    
    def train_over_1_5(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train Over 1.5 goals prediction model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (1 if >1.5 goals, 0 otherwise)
            validation_split: Fraction of data for validation
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training Over 1.5 goals model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Initialize and train
        self.over_1_5_model = self._initialize_model()
        self.over_1_5_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.over_1_5_model.predict(X_val)
        y_pred_proba = self.over_1_5_model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_pred_proba),
            "precision": self._calculate_precision(y_val, y_pred),
            "recall": self._calculate_recall(y_val, y_pred),
        }
        
        logger.info(f"Over 1.5 model trained - Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        
        # Save model
        self._save_model(self.over_1_5_model, "over_1_5_model.joblib")
        
        return metrics
    
    def train_btts(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """Train BTTS prediction model."""
        logger.info("Training BTTS model...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        self.btts_model = self._initialize_model()
        self.btts_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred = self.btts_model.predict(X_val)
        y_pred_proba = self.btts_model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_pred_proba),
            "precision": self._calculate_precision(y_val, y_pred),
            "recall": self._calculate_recall(y_val, y_pred),
        }
        
        logger.info(f"BTTS model trained - Accuracy: {metrics['accuracy']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        self._save_model(self.btts_model, "btts_model.joblib")
        
        return metrics
    
    def predict_over_1_5(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict Over 1.5 goals probability.
        
        Returns:
            (predictions, probabilities) - Both as numpy arrays
        """
        if self.over_1_5_model is None:
            raise ValueError("Over 1.5 model not trained or loaded")
        
        predictions = self.over_1_5_model.predict(X)
        probabilities = self.over_1_5_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def predict_btts(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict BTTS probability."""
        if self.btts_model is None:
            raise ValueError("BTTS model not trained or loaded")
        
        predictions = self.btts_model.predict(X)
        probabilities = self.btts_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def ensemble_predict(
        self,
        X: np.ndarray,
        llm_over_1_5_prob: float,
        llm_btts_prob: float,
        llm_confidence: float = 0.7,
    ) -> Dict[str, float]:
        """
        Ensemble prediction combining statistical model and LLM.
        
        Args:
            X: Feature vector
            llm_over_1_5_prob: LLM's Over 1.5 probability
            llm_btts_prob: LLM's BTTS probability
            llm_confidence: Confidence in LLM (0-1)
        
        Returns:
            Dictionary with ensemble probabilities
        """
        # Get statistical model predictions
        _, stat_over_1_5_prob = self.predict_over_1_5(X.reshape(1, -1))
        _, stat_btts_prob = self.predict_btts(X.reshape(1, -1))
        
        stat_over_1_5_prob = stat_over_1_5_prob[0]
        stat_btts_prob = stat_btts_prob[0]
        
        # Weight by LLM confidence
        stat_weight = 1 - (llm_confidence * 0.5)  # LLM gets at most 50% weight
        llm_weight = llm_confidence * 0.5
        
        # Normalize weights
        total_weight = stat_weight + llm_weight
        stat_weight /= total_weight
        llm_weight /= total_weight
        
        ensemble_over_1_5 = (
            stat_over_1_5_prob * stat_weight +
            llm_over_1_5_prob * llm_weight
        )
        
        ensemble_btts = (
            stat_btts_prob * stat_weight +
            llm_btts_prob * llm_weight
        )
        
        return {
            "over_1_5_probability": float(ensemble_over_1_5),
            "btts_probability": float(ensemble_btts),
            "statistical_over_1_5": float(stat_over_1_5_prob),
            "statistical_btts": float(stat_btts_prob),
            "llm_over_1_5": llm_over_1_5_prob,
            "llm_btts": llm_btts_prob,
            "stat_weight": stat_weight,
            "llm_weight": llm_weight,
        }
    
    def _save_model(self, model: Any, filename: str) -> None:
        """Save model to disk."""
        filepath = self.models_dir / filename
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def _load_models(self) -> None:
        """Load existing models from disk."""
        over_1_5_path = self.models_dir / "over_1_5_model.joblib"
        btts_path = self.models_dir / "btts_model.joblib"
        
        if over_1_5_path.exists():
            self.over_1_5_model = joblib.load(over_1_5_path)
            logger.info("Loaded Over 1.5 model")
        
        if btts_path.exists():
            self.btts_model = joblib.load(btts_path)
            logger.info("Loaded BTTS model")
    
    @staticmethod
    def _calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    @staticmethod
    def _calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
