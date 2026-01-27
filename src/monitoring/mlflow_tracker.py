"""MLflow experiment tracking and model monitoring."""
import mlflow
import mlflow.sklearn
from datetime import datetime
from loguru import logger
from typing import Dict, List, Optional


class MLflowTracker:
    """
    MLflow integration for experiment tracking and model versioning.
    
    Tracks:
    - Model training runs (hyperparameters, metrics)
    - Daily predictions and outcomes
    - Performance degradation alerts
    - Model versions and rollbacks
    """
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "telegramsoccer",
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow tracking initialized: {tracking_uri}/{experiment_name}")
    
    def log_model_training(
        self,
        model_name: str,
        model,
        params: Dict,
        metrics: Dict,
        artifacts: Optional[List[str]] = None,
    ) -> str:
        """
        Log model training run.
        
        Args:
            model_name: Name of model (e.g., "xgboost_over_1_5")
            model: Trained model object
            params: Hyperparameters
            metrics: Training metrics (accuracy, precision, etc.)
            artifacts: Paths to additional files to log
        
        Returns:
            Run ID
        """
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path=model_name,
                registered_model_name=model_name,
            )
            
            # Log additional artifacts
            if artifacts:
                for artifact_path in artifacts:
                    mlflow.log_artifact(artifact_path)
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Logged {model_name} training run: {run_id}")
            
            return run_id
    
    def log_daily_predictions(
        self,
        predictions: List[Dict],
        date: datetime,
    ) -> None:
        """
        Log daily predictions for tracking.
        
        Args:
            predictions: List of prediction dicts with match, market, probability, odds
            date: Prediction date
        """
        with mlflow.start_run(run_name=f"daily_predictions_{date.strftime('%Y%m%d')}"):
            # Aggregate metrics
            avg_probability = sum(p['probability'] for p in predictions) / len(predictions)
            avg_odds = sum(p['odds'] for p in predictions) / len(predictions)
            avg_value_score = sum(
                (p['probability'] / (1 / p['odds'])) - 1
                for p in predictions
            ) / len(predictions)
            
            mlflow.log_metrics({
                'num_predictions': len(predictions),
                'avg_probability': avg_probability,
                'avg_odds': avg_odds,
                'avg_value_score': avg_value_score,
            })
            
            # Log prediction details as artifact
            import json
            predictions_file = f"data/predictions_{date.strftime('%Y%m%d')}.json"
            with open(predictions_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            mlflow.log_artifact(predictions_file)
            
            logger.info(f"Logged {len(predictions)} daily predictions")
    
    def log_daily_outcomes(
        self,
        outcomes: List[Dict],
        date: datetime,
    ) -> None:
        """
        Log outcomes of settled tips.
        
        Args:
            outcomes: List of outcome dicts with tip_id, result (won/lost), profit
            date: Settlement date
        """
        with mlflow.start_run(run_name=f"daily_outcomes_{date.strftime('%Y%m%d')}"):
            # Calculate metrics
            wins = sum(1 for o in outcomes if o['result'] == 'won')
            losses = len(outcomes) - wins
            win_rate = wins / len(outcomes) if outcomes else 0
            
            total_profit = sum(o['profit'] for o in outcomes)
            total_staked = sum(abs(o['profit']) for o in outcomes)  # Simplified
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            
            mlflow.log_metrics({
                'num_settled_tips': len(outcomes),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'roi': roi,
            })
            
            logger.info(f"Logged {len(outcomes)} outcomes: {win_rate:.1%} win rate, {roi:.1f}% ROI")
    
    def log_rl_training(
        self,
        episode_rewards: List[float],
        episode_lengths: List[int],
        total_timesteps: int,
    ) -> None:
        """Log RL agent training metrics."""
        with mlflow.start_run(run_name=f"rl_training_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_params({
                'total_timesteps': total_timesteps,
                'num_episodes': len(episode_rewards),
            })
            
            mlflow.log_metrics({
                'mean_episode_reward': sum(episode_rewards) / len(episode_rewards),
                'max_episode_reward': max(episode_rewards),
                'mean_episode_length': sum(episode_lengths) / len(episode_lengths),
            })
            
            logger.info("Logged RL training run")
    
    def log_finetuning(
        self,
        base_model: str,
        num_epochs: int,
        training_samples: int,
        loss_history: List[float],
    ) -> None:
        """Log LLM fine-tuning run."""
        with mlflow.start_run(run_name=f"llm_finetuning_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_params({
                'base_model': base_model,
                'num_epochs': num_epochs,
                'training_samples': training_samples,
            })
            
            mlflow.log_metrics({
                'final_loss': loss_history[-1] if loss_history else 0,
                'avg_loss': sum(loss_history) / len(loss_history) if loss_history else 0,
            })
            
            logger.info("Logged fine-tuning run")
    
    def check_performance_degradation(
        self,
        model_name: str,
        current_metric: float,
        metric_name: str = "accuracy",
        threshold: float = 0.05,
    ) -> bool:
        """
        Check if model performance has degraded compared to best run.
        
        Args:
            model_name: Name of model to check
            current_metric: Current metric value (e.g., accuracy)
            metric_name: Name of metric to compare
            threshold: Degradation threshold (e.g., 0.05 = 5% drop triggers alert)
        
        Returns:
            True if degradation detected
        """
        # Get best run for this model
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.model_name = '{model_name}'",
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1,
        )
        
        if runs.empty:
            logger.warning(f"No previous runs found for {model_name}")
            return False
        
        best_metric = runs.iloc[0][f"metrics.{metric_name}"]
        degradation = best_metric - current_metric
        
        if degradation > threshold:
            logger.warning(
                f"Performance degradation detected for {model_name}: "
                f"{metric_name} dropped from {best_metric:.3f} to {current_metric:.3f} "
                f"(-{degradation:.3f})"
            )
            return True
        
        return False
    
    def get_best_model_version(self, model_name: str) -> Optional[str]:
        """Get the best registered model version by accuracy."""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Get all versions of the model
            versions = client.search_model_versions(f"name='{model_name}'")
            
            if not versions:
                logger.warning(f"No versions found for model {model_name}")
                return None
            
            # Find version with highest accuracy
            best_version = max(
                versions,
                key=lambda v: float(v.run_data.metrics.get('accuracy', 0))
            )
            
            return best_version.version
        
        except Exception as e:
            logger.error(f"Error fetching best model version: {e}")
            return None


class PerformanceAlerts:
    """Send alerts when performance degrades."""
    
    def __init__(self, mlflow_tracker: MLflowTracker):
        self.tracker = mlflow_tracker
    
    def check_and_alert(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
    ) -> None:
        """
        Check metrics and send alerts if degradation detected.
        
        Args:
            model_name: Model to check
            current_metrics: Current performance metrics
        """
        degradation_detected = False
        
        for metric_name, current_value in current_metrics.items():
            if self.tracker.check_performance_degradation(
                model_name=model_name,
                current_metric=current_value,
                metric_name=metric_name,
                threshold=0.05,
            ):
                degradation_detected = True
        
        if degradation_detected:
            self._send_alert(model_name, current_metrics)
    
    def _send_alert(self, model_name: str, metrics: Dict) -> None:
        """Send alert (Slack, email, etc.)."""
        logger.error(
            f"🚨 PERFORMANCE ALERT: {model_name} degradation detected!\n"
            f"Current metrics: {metrics}\n"
            f"Retraining recommended."
        )
        
        # In production: Send Slack notification, email, etc.
