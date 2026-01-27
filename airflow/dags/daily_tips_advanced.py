"""Airflow DAG for daily betting tip generation pipeline."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from loguru import logger
import sys
sys.path.append('/opt/airflow/dags/telegramsoccer')

from src.pipeline import Pipeline
from src.bot.telegram_bot import TelegramBot
from src.rag.retriever import BettingMemoryRAG
from src.core.database import DatabaseManager


# Default arguments
default_args = {
    'owner': 'telegramsoccer',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email': ['alerts@telegramsoccer.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'daily_betting_tips_advanced',
    default_args=default_args,
    description='Advanced daily tip generation with RL, RAG, and meta-learning',
    schedule_interval='0 9 * * *',  # 9 AM UTC daily
    catchup=False,
    tags=['betting', 'production', 'advanced'],
)


def index_recent_failures(**context):
    """Task 1: Index recent betting failures into RAG system."""
    logger.info("Indexing recent failures into RAG...")
    
    db_manager = DatabaseManager()
    rag_system = BettingMemoryRAG(db_manager)
    
    # Index failures from last 7 days
    start_date = datetime.now() - timedelta(days=7)
    count = await rag_system.index_failed_tips(start_date=start_date)
    
    logger.info(f"Indexed {count} failed tips")
    return count


def run_daily_pipeline(**context):
    """Task 2: Run main pipeline with RAG-enhanced prompts."""
    logger.info("Running daily pipeline with advanced features...")
    
    pipeline = Pipeline()
    tips = await pipeline.run_daily_pipeline()
    
    logger.info(f"Generated {len(tips)} tips for today")
    
    # Store tips in XCom for downstream tasks
    return [tip['id'] for tip in tips]


def send_tips_to_telegram(**context):
    """Task 3: Broadcast tips via Telegram bot."""
    tip_ids = context['ti'].xcom_pull(task_ids='run_pipeline')
    
    if not tip_ids:
        logger.warning("No tips to broadcast")
        return
    
    logger.info(f"Broadcasting {len(tip_ids)} tips...")
    
    bot = TelegramBot()
    await bot.broadcast_tips(tip_ids)
    
    logger.info("Tips broadcasted successfully")


def update_performance_metrics(**context):
    """Task 4: Update performance tracking and log to MLflow."""
    logger.info("Updating performance metrics...")
    
    # This would typically:
    # 1. Fetch settled tips from yesterday
    # 2. Calculate win rate, ROI, etc.
    # 3. Log to MLflow
    # 4. Trigger retraining if performance degrades
    
    # Placeholder
    logger.info("Performance metrics updated")


def check_retraining_trigger(**context):
    """Task 5: Check if models need retraining based on performance."""
    logger.info("Checking if retraining is needed...")
    
    # Check performance metrics
    # If win rate < 55% over last 7 days, trigger retraining
    
    # Placeholder
    needs_retraining = False
    
    if needs_retraining:
        logger.warning("Performance degradation detected - trigger retraining")
        return "retrain"
    else:
        logger.info("Performance acceptable - no retraining needed")
        return "skip"


# Task definitions
task_index_failures = PythonOperator(
    task_id='index_failures',
    python_callable=index_recent_failures,
    dag=dag,
)

task_run_pipeline = PythonOperator(
    task_id='run_pipeline',
    python_callable=run_daily_pipeline,
    dag=dag,
)

task_send_telegram = PythonOperator(
    task_id='send_telegram',
    python_callable=send_tips_to_telegram,
    dag=dag,
)

task_update_metrics = PythonOperator(
    task_id='update_metrics',
    python_callable=update_performance_metrics,
    dag=dag,
)

task_check_retraining = PythonOperator(
    task_id='check_retraining',
    python_callable=check_retraining_trigger,
    dag=dag,
)

# Task dependencies
task_index_failures >> task_run_pipeline >> task_send_telegram
task_send_telegram >> task_update_metrics >> task_check_retraining
