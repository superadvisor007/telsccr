"""Ollama integration for local LLM inference (100% FREE)."""
import json
from typing import Any, Dict, List, Optional

import ollama
from loguru import logger


class OllamaLLM:
    """
    Local LLM inference using Ollama.
    
    Supported models (all free):
    - llama3.2:3b (3B parameters, fast, good for analysis)
    - phi4:latest (Microsoft Phi-4, excellent reasoning)
    - mistral:7b (7B parameters, balanced)
    - qwen2.5:7b (Alibaba, strong on structured output)
    
    No API keys needed, runs completely offline.
    """
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        
        logger.info(f"Ollama LLM initialized with model: {model}")
    
    def ensure_model_exists(self) -> bool:
        """Check if model is pulled, download if not."""
        try:
            models = self.client.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if self.model not in model_names:
                logger.info(f"Pulling model {self.model} (this may take a few minutes)...")
                self.client.pull(self.model)
                logger.info(f"Model {self.model} pulled successfully")
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking/pulling model: {e}")
            return False
    
    async def analyze_match(
        self,
        match_data: Dict[str, Any],
        market: str = "over_1_5",
    ) -> Dict[str, Any]:
        """
        Analyze match and return prediction.
        
        Args:
            match_data: Dict with home_team, away_team, stats, odds, etc.
            market: Target market ('over_1_5' or 'btts')
        
        Returns:
            {
                'probability': float,
                'confidence': float,
                'key_factors': List[str],
                'reasoning': str,
                'recommendation': 'BET' | 'AVOID' | 'MONITOR'
            }
        """
        # Construct prompt
        prompt = self._build_analysis_prompt(match_data, market)
        
        try:
            # Call Ollama (completely free, local inference)
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative soccer analyst. Respond ONLY with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.7,
                    "num_predict": 512,  # Max tokens
                }
            )
            
            # Parse response
            content = response['message']['content']
            
            # Extract JSON (handle potential markdown code blocks)
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            
            # Validate and normalize
            return {
                'probability': float(result.get('probability', 0.5)),
                'confidence': float(result.get('confidence', 0.5)),
                'key_factors': result.get('key_factors', []),
                'reasoning': result.get('reasoning', ''),
                'recommendation': result.get('recommendation', 'MONITOR'),
                'model': self.model,
            }
        
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            
            # Fallback to simple statistical estimate
            return self._fallback_analysis(match_data, market)
    
    def _build_analysis_prompt(self, match_data: Dict, market: str) -> str:
        """Build analysis prompt for Ollama."""
        market_description = {
            'over_1_5': 'Over 1.5 Goals (match ends with 2+ total goals)',
            'btts': 'Both Teams To Score',
        }
        
        prompt = f"""Analyze this soccer match for **{market_description.get(market, market)}** betting:

**Match**: {match_data.get('home_team', 'Home')} vs {match_data.get('away_team', 'Away')}
**League**: {match_data.get('league', 'Unknown')}
**Date**: {match_data.get('date', 'Today')}

**Home Team Statistics**:
- Goals/Game: {match_data.get('home_goals_per_game', 0):.2f}
- Goals Conceded/Game: {match_data.get('home_goals_conceded_per_game', 0):.2f}
- Clean Sheets: {match_data.get('home_clean_sheets', 0)}%
- Form (PPG): {match_data.get('home_form_ppg', 0):.2f}
- BTTS Rate: {match_data.get('home_btts_rate', 0):.1f}%

**Away Team Statistics**:
- Goals/Game: {match_data.get('away_goals_per_game', 0):.2f}
- Goals Conceded/Game: {match_data.get('away_goals_conceded_per_game', 0):.2f}
- Clean Sheets: {match_data.get('away_clean_sheets', 0)}%
- Form (PPG): {match_data.get('away_form_ppg', 0):.2f}
- BTTS Rate: {match_data.get('away_btts_rate', 0):.1f}%

**Head-to-Head (Last 5)**:
- Average Goals: {match_data.get('h2h_avg_goals', 0):.1f}
- {market_description.get(market)} Rate: {match_data.get(f'h2h_{market}_rate', 0):.1f}%

**Market Odds**:
- Over 1.5: {match_data.get('over_1_5_odds', 1.5):.2f} (implied prob: {(1/match_data.get('over_1_5_odds', 1.5)*100):.1f}%)
- BTTS: {match_data.get('btts_odds', 1.8):.2f} (implied prob: {(1/match_data.get('btts_odds', 1.8)*100):.1f}%)

**Weather**: {match_data.get('weather_description', 'Clear')}, {match_data.get('weather_temp', 15)}°C

**Analysis Instructions**:
1. Consider attacking strength, defensive vulnerability, recent form
2. Evaluate H2H patterns and league scoring trends
3. Compare your probability estimate to market odds (value = your_prob / implied_prob)
4. Respond with ONLY valid JSON in this exact format:

{{
  "probability": 0.XX,
  "confidence": 0.XX,
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "reasoning": "2-3 sentence explanation",
  "recommendation": "BET" | "AVOID" | "MONITOR"
}}

Recommendation guide:
- BET: probability > 75% AND confidence > 70% AND value > 1.05
- AVOID: probability < 60% OR value < 1.0
- MONITOR: Otherwise (marginal value)
"""
        return prompt
    
    def _fallback_analysis(self, match_data: Dict, market: str) -> Dict:
        """Fallback statistical estimate if LLM fails."""
        # Simple heuristic
        home_gpg = match_data.get('home_goals_per_game', 1.5)
        away_gpg = match_data.get('away_goals_per_game', 1.5)
        avg_goals = (home_gpg + away_gpg) / 2
        
        if market == 'over_1_5':
            probability = min(0.95, max(0.50, avg_goals / 3.0))
        else:  # btts
            home_btts = match_data.get('home_btts_rate', 50) / 100
            away_btts = match_data.get('away_btts_rate', 50) / 100
            probability = (home_btts + away_btts) / 2
        
        return {
            'probability': probability,
            'confidence': 0.5,
            'key_factors': ['Statistical fallback - LLM unavailable'],
            'reasoning': 'Analysis based on simple goal averages.',
            'recommendation': 'MONITOR',
            'model': 'fallback',
        }
    
    def generate_tip_report(self, tips: List[Dict]) -> str:
        """Generate formatted tip report."""
        try:
            prompt = f"""Generate a professional betting tips report for {len(tips)} tips.

Tips data (JSON):
{json.dumps(tips, indent=2)}

Create a clear, concise Markdown report with:
1. Executive summary (2-3 lines)
2. Each tip with match, market, odds, probability, key reason
3. Risk warnings

Keep it professional and factual."""
            
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional betting analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response['message']['content']
        
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return "# Daily Tips\n\n" + "\n\n".join([
                f"**{tip['match']}** - {tip['market']} @ {tip['odds']}"
                for tip in tips
            ])
