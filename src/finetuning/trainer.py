"""LLM fine-tuning pipeline for learning from bet outcomes."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from trl import SFTTrainer

from src.core.database import DatabaseManager, Match, Prediction, Tip


class OutcomeAnalyzer:
    """Generates retrospective analysis of why tips succeeded or failed."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def analyze_failure(
        self,
        tip: Dict,
        match: Dict,
        prediction: Dict,
    ) -> str:
        """
        Use LLM to analyze why a tip failed.
        
        Returns:
            Detailed failure analysis string
        """
        prompt = f"""Analyze why this betting tip FAILED:

**Match**: {match['home_team']} vs {match['away_team']}
**Market**: {tip['market']} @ {tip['odds']}
**Predicted Probability**: {tip['probability']:.1%}
**Actual Outcome**: {tip['result']} (Match ended {match['home_score']}-{match['away_score']})

**Pre-Match Analysis**:
{prediction.get('reasoning', 'N/A')}

**Key Factors Considered**:
{', '.join(prediction.get('key_factors', []))}

**Post-Match Data**:
- xG: {match.get('home_xg', 'N/A')} - {match.get('away_xg', 'N/A')}
- Possession: {match.get('home_possession', 'N/A')}% - {match.get('away_possession', 'N/A')}%
- Shots: {match.get('home_shots', 'N/A')} - {match.get('away_shots', 'N/A')}

Provide a concise (2-3 sentences) root cause analysis of what was overlooked or misjudged in the pre-match assessment."""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert soccer analyst specializing in post-match analysis for betting."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7,
            )
            
            analysis = response.choices[0].message.content.strip()
            logger.debug(f"Generated failure analysis: {analysis[:100]}...")
            return analysis
        
        except Exception as e:
            logger.error(f"Failed to generate outcome analysis: {e}")
            return "Analysis unavailable due to API error."
    
    async def analyze_success(
        self,
        tip: Dict,
        match: Dict,
        prediction: Dict,
    ) -> str:
        """Analyze why a tip succeeded (for reinforcement)."""
        prompt = f"""Analyze why this betting tip SUCCEEDED:

**Match**: {match['home_team']} vs {match['away_team']}
**Market**: {tip['market']} @ {tip['odds']}
**Predicted Probability**: {tip['probability']:.1%}
**Actual Outcome**: {tip['result']} (Match ended {match['home_score']}-{match['away_score']})

**Pre-Match Analysis**:
{prediction.get('reasoning', 'N/A')}

Identify 1-2 key factors that were correctly weighted in the prediction."""
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert soccer analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Failed to generate success analysis: {e}")
            return "Correct assessment of match conditions."


class FineTuningDatasetBuilder:
    """Builds training datasets from historical tips with outcomes."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    async def build_training_corpus(
        self,
        start_date: Optional[datetime] = None,
        min_tips: int = 50,
    ) -> List[Dict]:
        """
        Create training examples from completed tips.
        
        Format: {"input": pre_match_context, "output": corrected_prediction, "outcome": success/failure}
        """
        logger.info("Building fine-tuning dataset from historical tips...")
        
        # Fetch completed tips
        session = self.db.get_session()
        query = session.query(Tip, Match, Prediction).join(
            Match, Tip.match_id == Match.id
        ).join(
            Prediction, Tip.prediction_id == Prediction.id
        ).filter(
            Tip.result.isnot(None)  # Only settled tips
        )
        
        if start_date:
            query = query.filter(Match.date >= start_date)
        
        results = query.all()
        session.close()
        
        if len(results) < min_tips:
            logger.warning(f"Only {len(results)} tips available, need {min_tips} for fine-tuning")
            return []
        
        training_examples = []
        
        for tip, match, prediction in results:
            # Construct input (pre-match context)
            input_text = self._format_input(match, prediction)
            
            # Construct output (corrected prediction with outcome knowledge)
            output_text = self._format_output(tip, match, prediction)
            
            training_examples.append({
                "input": input_text,
                "output": output_text,
                "outcome": "success" if tip.result == "won" else "failure",
                "market": tip.market,
                "odds": tip.odds,
                "probability": tip.probability,
            })
        
        logger.info(f"Built dataset with {len(training_examples)} examples")
        return training_examples
    
    def _format_input(self, match: Match, prediction: Prediction) -> str:
        """Format pre-match context as LLM input."""
        return f"""Analyze this match for {prediction.market} betting:

**Teams**: {match.home_team} vs {match.away_team}
**League**: {match.league}
**Date**: {match.date.strftime('%Y-%m-%d')}

**Home Team Stats**:
- Goals/Game: {match.home_goals_per_game:.2f}
- Clean Sheets: {match.home_clean_sheets}%
- Form (PPG): {match.home_form_ppg:.2f}

**Away Team Stats**:
- Goals/Game: {match.away_goals_per_game:.2f}
- Clean Sheets: {match.away_clean_sheets}%
- Form (PPG): {match.away_form_ppg:.2f}

**Context**:
- Weather: {match.weather_description or 'Clear'} ({match.weather_temp or 15}°C)
- Odds: Over 1.5 @ {match.over_1_5_odds}, BTTS @ {match.btts_odds}

Provide probability estimate for {prediction.market}:"""
    
    def _format_output(self, tip: Tip, match: Match, prediction: Prediction) -> str:
        """Format corrected prediction with outcome knowledge."""
        actual_outcome = "SUCCESS" if tip.result == "won" else "FAILURE"
        score = f"{match.home_score}-{match.away_score}" if match.home_score is not None else "N/A"
        
        output = f"""**Prediction**: {prediction.probability:.1%} for {prediction.market}
**Actual Result**: {actual_outcome} (Score: {score})

**Analysis**:
{prediction.reasoning}

"""
        
        # Add corrective insights based on outcome
        if tip.result == "lost":
            output += f"""**What Was Missed**:
{tip.post_mortem or 'Underestimated defensive form or overestimated attacking threat.'}
"""
        else:
            output += f"""**What Was Correctly Identified**:
{', '.join(prediction.key_factors or ['Strong attacking form', 'Weak defensive records'])}
"""
        
        return output
    
    def save_to_jsonl(self, training_examples: List[Dict], output_path: str) -> None:
        """Save training data in JSONL format for Hugging Face datasets."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(training_examples)} examples to {output_path}")


class LLMFineTuner:
    """Fine-tune LLM on historical betting outcomes using LoRA/PEFT."""
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B",
        output_dir: str = "models/finetuned_llm",
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
    
    def load_base_model(self) -> None:
        """Load base model and tokenizer."""
        logger.info(f"Loading base model: {self.base_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        logger.info("Base model loaded successfully")
    
    def prepare_lora_config(self) -> LoraConfig:
        """Configure LoRA for parameter-efficient fine-tuning."""
        return LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    
    def train(
        self,
        training_data_path: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
    ) -> None:
        """Fine-tune model on betting outcomes dataset."""
        if self.model is None:
            self.load_base_model()
        
        # Load dataset
        logger.info(f"Loading training data from {training_data_path}")
        with open(training_data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        # Format for SFT (supervised fine-tuning)
        formatted_data = []
        for example in data:
            formatted_data.append({
                "text": f"### Input:\n{example['input']}\n\n### Output:\n{example['output']}"
            })
        
        dataset = Dataset.from_list(formatted_data)
        
        # Apply LoRA
        lora_config = self.prepare_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            warmup_steps=50,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
        )
        
        # Initialize SFT trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            max_seq_length=2048,
            dataset_text_field="text",
        )
        
        # Train
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save fine-tuned model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Fine-tuned model saved to {self.output_dir}")
    
    def merge_and_save(self, merged_output_dir: str) -> None:
        """Merge LoRA weights with base model for inference."""
        if self.model is None:
            raise ValueError("No model loaded")
        
        logger.info("Merging LoRA weights with base model...")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(merged_output_dir)
        self.tokenizer.save_pretrained(merged_output_dir)
        
        logger.info(f"Merged model saved to {merged_output_dir}")
