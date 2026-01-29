"""
DeepSeek LLM 7B Client - 100% FREE via Ollama.

DeepSeek 7B is a powerful open-source LLM that runs locally.
No API keys needed, no external costs - runs entirely on GitHub Codespaces.

Models available via Ollama:
- deepseek-llm:7b - General purpose (best for analysis)
- deepseek-coder:7b - Code-focused variant
"""
import json
import os
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


class DeepSeekLLM:
    """
    DeepSeek 7B LLM client using Ollama for local inference.
    
    100% FREE - No API costs, runs on GitHub Codespaces compute.
    Perfect for soccer betting analysis with strong reasoning capabilities.
    """
    
    # Available DeepSeek models via Ollama
    AVAILABLE_MODELS = [
        "deepseek-llm:7b",      # Best for general analysis
        "deepseek-coder:7b",    # Code-focused variant
        "deepseek-llm:latest",  # Latest version
    ]
    
    def __init__(
        self,
        model: str = "deepseek-llm:7b",
        base_url: str = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 120,
    ):
        """
        Initialize DeepSeek LLM client.
        
        Args:
            model: DeepSeek model name (default: deepseek-llm:7b)
            base_url: Ollama API URL (default: http://localhost:11434)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        logger.info(f"DeepSeek LLM initialized: {model} @ {self.base_url}")
    
    def is_available(self) -> bool:
        """Check if Ollama server is running and DeepSeek model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if our model is available
                if any(self.model in name for name in model_names):
                    return True
                
                logger.warning(f"Model {self.model} not found. Available: {model_names}")
                return False
            return False
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    def pull_model(self) -> bool:
        """Pull/download the DeepSeek model if not available."""
        try:
            logger.info(f"Pulling DeepSeek model: {self.model} (this may take 5-10 minutes)...")
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=600,  # 10 minutes for download
                stream=True
            )
            
            # Stream progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status or "downloading" in status:
                        logger.info(f"Download progress: {status}")
            
            logger.info(f"Model {self.model} pulled successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        json_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate response from DeepSeek LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            json_mode: If True, expect JSON response
        
        Returns:
            Parsed response dict or raw text
        """
        if system_prompt is None:
            system_prompt = "You are an expert soccer betting analyst. Respond with valid JSON only."
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                    "stream": False,
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}"}
            
            result = response.json()
            content = result.get("message", {}).get("content", "")
            
            if json_mode:
                return self._parse_json_response(content)
            
            return {"text": content}
            
        except requests.exceptions.Timeout:
            logger.error("DeepSeek request timed out")
            return {"error": "timeout"}
        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            return {"error": str(e)}
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        content = content.strip()
        
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}")
            # Try to extract JSON object
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return {"raw_text": content, "parse_error": str(e)}
    
    def analyze_match(
        self,
        match_data: Dict[str, Any],
        market: str = "over_1_5",
    ) -> Dict[str, Any]:
        """
        Analyze a soccer match for betting prediction.
        
        Args:
            match_data: Match statistics and context
            market: Target market ('over_1_5', 'btts', 'over_2_5')
        
        Returns:
            Analysis with probability, confidence, reasoning
        """
        prompt = self._build_match_analysis_prompt(match_data, market)
        
        system_prompt = """You are DeepSeek, an expert quantitative soccer analyst specializing in low-odds accumulator betting strategies.

Your task is to analyze matches and provide probability estimates for specific betting markets.
You must respond ONLY with valid JSON in the exact format requested.
Be conservative and analytical - accuracy is more important than confidence."""

        result = self.generate(prompt, system_prompt=system_prompt, json_mode=True)
        
        if "error" in result:
            return self._fallback_analysis(match_data, market)
        
        # Normalize and validate response
        return {
            "probability": float(result.get("probability", 0.5)),
            "confidence": float(result.get("confidence", 0.5)),
            "key_factors": result.get("key_factors", ["Analysis completed"]),
            "reasoning": result.get("reasoning", "DeepSeek analysis"),
            "recommendation": result.get("recommendation", "MONITOR"),
            "model": self.model,
            "provider": "deepseek-ollama",
        }
    
    def _build_match_analysis_prompt(self, match_data: Dict, market: str) -> str:
        """Build comprehensive analysis prompt for DeepSeek."""
        market_descriptions = {
            "over_1_5": "Over 1.5 Goals (match ends with 2+ total goals)",
            "over_2_5": "Over 2.5 Goals (match ends with 3+ total goals)",
            "btts": "Both Teams To Score (each team scores at least 1 goal)",
        }
        
        prompt = f"""Analyze this soccer match for **{market_descriptions.get(market, market)}** betting market:

## MATCH INFORMATION
- **Home Team**: {match_data.get('home_team', 'Unknown')}
- **Away Team**: {match_data.get('away_team', 'Unknown')}
- **League**: {match_data.get('league', 'Unknown')}
- **Date**: {match_data.get('date', 'Today')}

## HOME TEAM STATISTICS
- Goals per game: {match_data.get('home_goals_per_game', 1.5):.2f}
- Goals conceded per game: {match_data.get('home_goals_conceded_per_game', 1.2):.2f}
- Form (points per game): {match_data.get('home_form_ppg', 1.5):.2f}
- Over 1.5 rate: {match_data.get('home_over_1_5_rate', 65):.1f}%
- BTTS rate: {match_data.get('home_btts_rate', 50):.1f}%
- Clean sheet rate: {match_data.get('home_clean_sheet_rate', 25):.1f}%

## AWAY TEAM STATISTICS
- Goals per game: {match_data.get('away_goals_per_game', 1.3):.2f}
- Goals conceded per game: {match_data.get('away_goals_conceded_per_game', 1.5):.2f}
- Form (points per game): {match_data.get('away_form_ppg', 1.2):.2f}
- Over 1.5 rate: {match_data.get('away_over_1_5_rate', 60):.1f}%
- BTTS rate: {match_data.get('away_btts_rate', 50):.1f}%
- Clean sheet rate: {match_data.get('away_clean_sheet_rate', 20):.1f}%

## HEAD-TO-HEAD (Last 5 meetings)
- Average goals: {match_data.get('h2h_avg_goals', 2.5):.2f}
- Over 1.5 rate: {match_data.get('h2h_over_1_5_rate', 70):.1f}%
- BTTS rate: {match_data.get('h2h_btts_rate', 55):.1f}%

## MARKET ODDS
- Over 1.5 odds: {match_data.get('over_1_5_odds', 1.30):.2f}
- BTTS odds: {match_data.get('btts_odds', 1.70):.2f}

## YOUR TASK
Analyze all factors and provide your assessment as JSON:

```json
{{
    "probability": 0.XX,
    "confidence": 0.XX,
    "key_factors": [
        "Factor 1: explanation",
        "Factor 2: explanation",
        "Factor 3: explanation"
    ],
    "reasoning": "2-3 sentence comprehensive analysis",
    "recommendation": "BET|AVOID|MONITOR"
}}
```

RECOMMENDATION CRITERIA:
- BET: probability >= 0.75 AND confidence >= 0.70 AND value exists
- AVOID: probability < 0.60 OR major risks identified
- MONITOR: Marginal cases requiring more data

Respond with ONLY the JSON object, no additional text."""

        return prompt
    
    def _fallback_analysis(self, match_data: Dict, market: str) -> Dict[str, Any]:
        """Fallback statistical analysis when LLM fails."""
        home_gpg = match_data.get('home_goals_per_game', 1.5)
        away_gpg = match_data.get('away_goals_per_game', 1.3)
        combined_gpg = home_gpg + away_gpg
        
        if market == "over_1_5":
            # Poisson approximation for P(goals >= 2)
            probability = min(0.90, max(0.50, 1 - (1 + combined_gpg) * (2.718 ** -combined_gpg)))
        elif market == "over_2_5":
            probability = min(0.85, max(0.35, combined_gpg / 4.0))
        else:  # btts
            home_btts = match_data.get('home_btts_rate', 50) / 100
            away_btts = match_data.get('away_btts_rate', 50) / 100
            probability = (home_btts + away_btts) / 2
        
        return {
            "probability": round(probability, 3),
            "confidence": 0.4,
            "key_factors": ["Statistical fallback - DeepSeek unavailable"],
            "reasoning": f"Analysis based on statistical averages. Combined GPG: {combined_gpg:.2f}",
            "recommendation": "MONITOR",
            "model": "fallback-statistical",
            "provider": "fallback",
        }
    
    def batch_analyze(
        self,
        matches: List[Dict[str, Any]],
        market: str = "over_1_5",
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple matches.
        
        Args:
            matches: List of match data dicts
            market: Target betting market
        
        Returns:
            List of analysis results
        """
        results = []
        
        for match in matches:
            logger.info(f"Analyzing: {match.get('home_team')} vs {match.get('away_team')}")
            analysis = self.analyze_match(match, market)
            results.append(analysis)
        
        return results
    
    def generate_tip_report(self, tips: List[Dict]) -> str:
        """Generate formatted betting tips report."""
        prompt = f"""Generate a professional betting tips report based on the following {len(tips)} tips:

{json.dumps(tips, indent=2)}

Create a Markdown report with:
1. **Executive Summary** (2-3 lines overview)
2. **Tips Table** with match, market, odds, probability, recommendation
3. **Risk Assessment** (brief warnings)

Keep it professional, concise, and factual."""

        result = self.generate(
            prompt,
            system_prompt="You are a professional betting analyst. Generate clear Markdown reports.",
            json_mode=False
        )
        
        if "error" in result:
            # Fallback simple report
            report = "# 🎯 Daily Soccer Tips\n\n"
            for tip in tips:
                report += f"**{tip.get('match', 'Unknown')}** - {tip.get('market', '')} @ {tip.get('odds', '')}\n\n"
            return report
        
        return result.get("text", "Report generation failed")


# Singleton instance for easy import
_deepseek_instance = None


def get_deepseek_llm(
    model: str = "deepseek-llm:7b",
    **kwargs
) -> DeepSeekLLM:
    """Get or create DeepSeek LLM instance."""
    global _deepseek_instance
    
    if _deepseek_instance is None or _deepseek_instance.model != model:
        _deepseek_instance = DeepSeekLLM(model=model, **kwargs)
    
    return _deepseek_instance
