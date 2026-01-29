"""LLM-based match analysis using DeepSeek 7B (100% FREE via Ollama)."""
import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger

from src.llm.deepseek_client import DeepSeekLLM, get_deepseek_llm


class LLMAnalyzer:
    """LLM-powered contextual match analysis using DeepSeek 7B.
    
    DeepSeek 7B runs locally via Ollama - completely FREE!
    No API keys needed, no external costs.
    """
    
    def __init__(self):
        # DeepSeek 7B - FREE local LLM via Ollama
        model = os.environ.get("LLM_MODEL", "deepseek-llm:7b")
        self.deepseek = get_deepseek_llm(model=model)
        self.model = model
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "2000"))
    
    def analyze_match(
        self,
        match_data: Dict[str, Any],
        features: Dict[str, Any],
        home_stats: Dict[str, Any],
        away_stats: Dict[str, Any],
        h2h_stats: Dict[str, Any],
        weather: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive LLM analysis of a match using DeepSeek 7B.
        
        Returns:
            Dictionary with probabilities, reasoning, key factors, and confidence
        """
        prompt = self._build_analysis_prompt(
            match_data, features, home_stats, away_stats, h2h_stats, weather
        )
        
        try:
            # Use DeepSeek 7B (FREE via Ollama)
            response = self._analyze_with_deepseek(prompt)
            
            logger.info(f"DeepSeek analysis completed for {match_data['home_team']} vs {match_data['away_team']}")
            return response
            
        except Exception as e:
            logger.error(f"DeepSeek analysis failed: {e}")
            # Return fallback analysis
            return self._fallback_analysis(features)
    
    def _build_analysis_prompt(
        self,
        match_data: Dict,
        features: Dict,
        home_stats: Dict,
        away_stats: Dict,
        h2h_stats: Dict,
        weather: Optional[Dict],
    ) -> str:
        """Build comprehensive analysis prompt for LLM."""
        
        prompt = f"""You are an expert soccer betting analyst specializing in low-odds accumulator strategies. Analyze this match for Over 1.5 Goals and Both Teams to Score (BTTS) markets.

MATCH DETAILS:
- {match_data['home_team']} (Home) vs {match_data['away_team']} (Away)
- League: {match_data['league']}
- Date: {match_data['match_date']}

HOME TEAM STATISTICS (Last {home_stats.get('matches_played', 5)} matches):
- Form: {home_stats.get('form', 'N/A')}
- Goals per game: {home_stats.get('goals_per_game', 0):.2f}
- Goals conceded per game: {home_stats.get('goals_conceded_per_game', 0):.2f}
- Over 1.5 rate: {home_stats.get('over_1_5_percentage', 0):.1f}%
- BTTS rate: {home_stats.get('btts_percentage', 0):.1f}%
- Clean sheets: {home_stats.get('clean_sheet_percentage', 0):.1f}%
- Points per game: {home_stats.get('ppg', 0):.2f}

AWAY TEAM STATISTICS (Last {away_stats.get('matches_played', 5)} matches):
- Form: {away_stats.get('form', 'N/A')}
- Goals per game: {away_stats.get('goals_per_game', 0):.2f}
- Goals conceded per game: {away_stats.get('goals_conceded_per_game', 0):.2f}
- Over 1.5 rate: {away_stats.get('over_1_5_percentage', 0):.1f}%
- BTTS rate: {away_stats.get('btts_percentage', 0):.1f}%
- Clean sheets: {away_stats.get('clean_sheet_percentage', 0):.1f}%
- Points per game: {away_stats.get('ppg', 0):.2f}

HEAD-TO-HEAD:
- Recent meetings: {h2h_stats.get('matches', 0)}
- Average goals: {h2h_stats.get('avg_goals', 0):.2f}
- BTTS rate: {h2h_stats.get('btts_rate', 0):.1f}%

ENGINEERED FEATURES:
- Total expected goals: {features.get('total_expected_goals', 0):.2f}
- Home attack strength: {features.get('home_attack_strength', 0):.2f}
- Away attack strength: {features.get('away_attack_strength', 0):.2f}
- Combined attack strength: {features.get('combined_attack_strength', 0):.2f}
- Defensive weakness (combined): {features.get('defensive_weakness', 0):.2f}
"""

        if weather:
            prompt += f"""
WEATHER CONDITIONS:
- Condition: {weather.get('condition', 'Unknown')} - {weather.get('description', '')}
- Temperature: {weather.get('temperature', 0):.1f}°C
- Wind speed: {weather.get('wind_speed', 0):.1f} m/s
- Rain: {weather.get('rain_volume', 0):.1f}mm
- Impact score: {weather.get('impact_score', 0)}/10
- Favorable for goals: {'Yes' if weather.get('favorable_for_goals', True) else 'No'}
- Reasoning: {weather.get('reasoning', 'N/A')}
"""

        if "odds" in match_data:
            odds = match_data["odds"]
            prompt += f"""
MARKET ODDS:
- Over 1.5 goals: {odds.get('over_1_5', 'N/A')} (implied prob: {1/odds.get('over_1_5', 1.5):.1%})
- BTTS Yes: {odds.get('btts_yes', 'N/A')} (implied prob: {1/odds.get('btts_yes', 1.5):.1%})
"""

        prompt += """
ANALYSIS TASK:
Consider ALL contextual factors:
1. Environmental: Weather impact, pitch conditions, travel distance
2. Team/Player: Current form, fatigue from schedule congestion, key absences
3. Psychological: Rivalry intensity, momentum, pressure (league position)
4. Tactical: Playing styles, set-piece strength, defensive vulnerabilities
5. Historical: H2H patterns, venue-specific performance

Provide your analysis as a JSON object with this EXACT structure:
{
    "over_1_5_probability": 0.XX,
    "btts_probability": 0.XX,
    "confidence_score": 0.XX,
    "key_factors": [
        "Factor 1 explanation",
        "Factor 2 explanation",
        "Factor 3 explanation"
    ],
    "reasoning": "Comprehensive 2-3 sentence analysis of why these probabilities make sense",
    "risks": [
        "Risk 1",
        "Risk 2"
    ],
    "recommendation": "BET/AVOID/MONITOR"
}

CRITICAL REQUIREMENTS:
- over_1_5_probability: Your estimated probability (0.00-1.00) that match will have >1.5 goals
- btts_probability: Your estimated probability (0.00-1.00) that both teams will score
- confidence_score: How confident you are (0.00-1.00) in this analysis
- Only recommend BET if probability >0.80 for Over 1.5 OR >0.70 for BTTS
- Be conservative - we're building low-odds accumulators that require high accuracy
- Return ONLY valid JSON, no additional text
"""
        
        return prompt
    
    def _analyze_with_deepseek(self, prompt: str) -> Dict[str, Any]:
        """Analyze using DeepSeek 7B (FREE via Ollama)."""
        system_prompt = """You are DeepSeek, an expert soccer betting analyst specializing in low-odds accumulator strategies.
Analyze matches conservatively - accuracy over confidence.
Return ONLY valid JSON with no additional text."""

        result = self.deepseek.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=True
        )
        
        if "error" in result:
            raise Exception(f"DeepSeek error: {result['error']}")
        
        # Ensure required fields exist
        return {
            "over_1_5_probability": float(result.get("over_1_5_probability", 0.5)),
            "btts_probability": float(result.get("btts_probability", 0.5)),
            "confidence_score": float(result.get("confidence_score", 0.5)),
            "key_factors": result.get("key_factors", ["DeepSeek analysis completed"]),
            "reasoning": result.get("reasoning", "Analysis by DeepSeek 7B"),
            "risks": result.get("risks", ["Standard market risks"]),
            "recommendation": result.get("recommendation", "MONITOR"),
            "model": self.model,
            "provider": "deepseek-ollama",
        }
    
    def _fallback_analysis(self, features: Dict) -> Dict[str, Any]:
        """Provide fallback analysis if DeepSeek fails."""
        return {
            "over_1_5_probability": features.get("over_1_5_baseline_prob", 0.5),
            "btts_probability": features.get("btts_baseline_prob", 0.5),
            "confidence_score": 0.3,  # Low confidence for fallback
            "key_factors": [
                "DeepSeek unavailable - using statistical baseline",
                f"Feature-based Over 1.5 estimate: {features.get('over_1_5_baseline_prob', 0.5):.2%}",
                f"Feature-based BTTS estimate: {features.get('btts_baseline_prob', 0.5):.2%}",
            ],
            "reasoning": "Analysis based purely on statistical features without DeepSeek reasoning.",
            "risks": ["No LLM analysis available", "Limited contextual awareness"],
            "recommendation": "MONITOR",
            "model": "fallback-statistical",
            "provider": "fallback",
        }
    
    def batch_analyze(
        self,
        matches_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple matches in batch.
        
        Args:
            matches_data: List of match data dicts (each with match_data, features, stats, etc.)
        
        Returns:
            List of analysis results
        """
        results = []
        
        for data in matches_data:
            try:
                analysis = self.analyze_match(
                    match_data=data["match_data"],
                    features=data["features"],
                    home_stats=data["home_stats"],
                    away_stats=data["away_stats"],
                    h2h_stats=data["h2h_stats"],
                    weather=data.get("weather"),
                )
                results.append(analysis)
            except Exception as e:
                logger.error(f"Batch analysis failed for match: {e}")
                results.append(self._fallback_analysis(data.get("features", {})))
        
        return results
