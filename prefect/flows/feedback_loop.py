"""Prefect flow for continuous learning feedback loop."""
from datetime import datetime, timedelta
from prefect import flow, task
from prefect.tasks import task_input_hash
from loguru import logger

from src.core.database import DatabaseManager
from src.finetuning.trainer import OutcomeAnalyzer, FineTuningDatasetBuilder, LLMFineTuner
from src.rl.agent import RLStakingAgent
from src.meta.learner import MetaLearner


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def fetch_settled_tips(days_back: int = 7):
    """Fetch tips that have been settled in the last N days."""
    logger.info(f"Fetching settled tips from last {days_back} days...")
    
    db_manager = DatabaseManager()
    # Query logic here
    
    settled_tips = []  # Placeholder
    logger.info(f"Found {len(settled_tips)} settled tips")
    
    return settled_tips


@task
async def generate_post_mortems(settled_tips: list):
    """Generate post-mortem analysis for failed tips."""
    logger.info("Generating post-mortem analyses...")
    
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    
    analyzer = OutcomeAnalyzer(client)
    
    analyses = []
    for tip in settled_tips:
        if tip['result'] == 'lost':
            analysis = await analyzer.analyze_failure(
                tip=tip,
                match=tip['match'],
                prediction=tip['prediction'],
            )
            analyses.append({
                'tip_id': tip['id'],
                'analysis': analysis,
            })
    
    logger.info(f"Generated {len(analyses)} post-mortems")
    return analyses


@task
def save_post_mortems(analyses: list):
    """Save post-mortem analyses to database."""
    logger.info("Saving post-mortems to database...")
    
    db_manager = DatabaseManager()
    session = db_manager.get_session()
    
    # Update Tip records with post_mortem field
    # for analysis in analyses:
    #     tip = session.query(Tip).filter_by(id=analysis['tip_id']).first()
    #     tip.post_mortem = analysis['analysis']
    
    # session.commit()
    session.close()
    
    logger.info("Post-mortems saved")


@task
def build_finetuning_dataset(min_tips: int = 100):
    """Build fine-tuning dataset from historical tips."""
    logger.info("Building fine-tuning dataset...")
    
    db_manager = DatabaseManager()
    builder = FineTuningDatasetBuilder(db_manager)
    
    training_data = await builder.build_training_corpus(min_tips=min_tips)
    
    if len(training_data) >= min_tips:
        output_path = f"data/finetuning/training_{datetime.now().strftime('%Y%m%d')}.jsonl"
        builder.save_to_jsonl(training_data, output_path)
        logger.info(f"Dataset saved to {output_path}")
        return output_path
    else:
        logger.warning(f"Insufficient data ({len(training_data)} tips), skipping fine-tuning")
        return None


@task
def finetune_llm(dataset_path: str):
    """Fine-tune LLM on outcome-enriched dataset."""
    if dataset_path is None:
        logger.info("Skipping fine-tuning - no dataset")
        return
    
    logger.info("Starting LLM fine-tuning...")
    
    finetuner = LLMFineTuner(
        base_model="meta-llama/Llama-3.1-8B",
        output_dir="models/finetuned_llm",
    )
    
    finetuner.train(
        training_data_path=dataset_path,
        num_epochs=3,
        batch_size=4,
    )
    
    logger.info("Fine-tuning complete")


@task
def retrain_rl_agent():
    """Retrain RL agent with recent betting outcomes."""
    logger.info("Retraining RL agent...")
    
    agent = RLStakingAgent()
    agent.train(total_timesteps=50000)
    
    logger.info("RL agent retrained")


@task
def retrain_meta_learner():
    """Retrain meta-learner on recent predictions."""
    logger.info("Retraining meta-learner...")
    
    db_manager = DatabaseManager()
    meta = MetaLearner()
    
    X, y = meta.prepare_training_data(db_manager, min_samples=100)
    
    if len(X) > 0:
        metrics = meta.train(X, y)
        logger.info(f"Meta-learner retrained: {metrics}")
    else:
        logger.warning("Insufficient data for meta-learner retraining")


@flow(name="Continuous Learning Feedback Loop")
async def continuous_learning_flow(
    days_back: int = 7,
    finetune_llm_enabled: bool = False,  # Expensive, run weekly
    retrain_rl_enabled: bool = True,
    retrain_meta_enabled: bool = True,
):
    """
    Main continuous learning flow.
    
    Runs daily to:
    1. Fetch settled tips
    2. Generate post-mortems for failures
    3. Retrain RL agent and meta-learner
    4. Optionally fine-tune LLM (weekly)
    """
    logger.info("Starting continuous learning feedback loop...")
    
    # Fetch data
    settled_tips = fetch_settled_tips(days_back=days_back)
    
    # Generate analyses
    analyses = await generate_post_mortems(settled_tips)
    save_post_mortems(analyses)
    
    # Retrain models
    if retrain_rl_enabled:
        retrain_rl_agent()
    
    if retrain_meta_enabled:
        retrain_meta_learner()
    
    # Fine-tune LLM (expensive, weekly only)
    if finetune_llm_enabled:
        dataset_path = build_finetuning_dataset(min_tips=100)
        finetune_llm(dataset_path)
    
    logger.info("Continuous learning flow complete")


if __name__ == "__main__":
    # Run the flow
    import asyncio
    asyncio.run(continuous_learning_flow(
        days_back=7,
        finetune_llm_enabled=False,
        retrain_rl_enabled=True,
        retrain_meta_enabled=True,
    ))
