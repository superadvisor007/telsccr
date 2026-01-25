"""AI prediction engine using Groq/Mistral API."""
import json
import logging
from typing import Dict, List, Optional
from groq import AsyncGroq

logger = logging.getLogger(__name__)


class PredictionEngine:
    """AI-powered prediction engine for soccer matches."""

    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        """Initialize prediction engine.
        
        Args:
            api_key: Groq API key
            model: Model to use for predictions
        """
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def generate_prediction(
        self,
        match_data: Dict,
        team_stats: Optional[Dict] = None,
        h2h_data: Optional[List[Dict]] = None,
        learning_stats: Optional[Dict] = None
    ) -> Dict:
        """Generate prediction for a match.
        
        Args:
            match_data: Match information
            team_stats: Team statistics (optional)
            h2h_data: Head-to-head history (optional)
            learning_stats: Historical learning data (optional)
            
        Returns:
            Prediction with confidence and reasoning
        """
        # Build context for the AI
        context = self._build_context(match_data, team_stats, h2h_data, learning_stats)
        
        # Create prompt for prediction
        prompt = self._create_prediction_prompt(context)
        
        try:
            # Call Groq API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert Swiss soccer analyst specializing in 
                        predictions for Swiss Super League matches. Provide detailed, data-driven 
                        predictions with confidence levels and clear reasoning. Always respond in 
                        valid JSON format."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse response
            result = self._parse_prediction_response(response.choices[0].message.content)
            logger.info(f"Generated prediction for {match_data['home_team']} vs {match_data['away_team']}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            # Return fallback prediction
            return self._fallback_prediction(match_data)

    def _build_context(
        self,
        match_data: Dict,
        team_stats: Optional[Dict],
        h2h_data: Optional[List[Dict]],
        learning_stats: Optional[Dict]
    ) -> Dict:
        """Build context for prediction."""
        context = {
            "match": match_data,
            "team_stats": team_stats or {},
            "h2h": h2h_data or [],
            "learning": learning_stats or {}
        }
        return context

    def _create_prediction_prompt(self, context: Dict) -> str:
        """Create prediction prompt."""
        match = context["match"]
        
        prompt = f"""Analyze the following Swiss Super League match and provide a prediction:

**Match Details:**
- Home Team: {match['home_team']}
- Away Team: {match['away_team']}
- Date: {match['match_date']}
- Venue: {match.get('venue', 'Unknown')}

"""
        
        # Add team statistics if available
        if context["team_stats"]:
            prompt += f"\n**Team Statistics:**\n{json.dumps(context['team_stats'], indent=2)}\n"
        
        # Add H2H data if available
        if context["h2h"]:
            prompt += f"\n**Head-to-Head History (Last {len(context['h2h'])} matches):**\n"
            for match in context["h2h"][:5]:  # Limit to 5 most recent
                prompt += f"- {match.get('teams', {}).get('home', {}).get('name', 'N/A')} vs "
                prompt += f"{match.get('teams', {}).get('away', {}).get('name', 'N/A')}: "
                prompt += f"{match.get('goals', {}).get('home', 'N/A')}-{match.get('goals', {}).get('away', 'N/A')}\n"
        
        # Add learning stats if available
        if context["learning"].get("total_predictions"):
            accuracy = (context["learning"].get("correct_predictions", 0) / 
                       context["learning"]["total_predictions"] * 100)
            prompt += f"\n**Model Performance:**\n- Accuracy: {accuracy:.1f}%\n"
            prompt += f"- Average Confidence: {context['learning'].get('avg_confidence', 0):.2f}\n"

        prompt += """
**Required Response Format (JSON):**
{
    "prediction": "home_win|away_win|draw",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of the prediction",
    "key_factors": ["factor1", "factor2", "factor3"],
    "suggested_bet": "1|X|2 or specific bet type",
    "value_assessment": "high|medium|low"
}

Provide a thorough analysis considering:
1. Recent form and statistics
2. Head-to-head history
3. Home advantage
4. Tactical considerations
5. Key player availability (if known)
6. League position and motivation

Be realistic and data-driven. If confidence is low, explain why."""
        
        return prompt

    def _parse_prediction_response(self, response: str) -> Dict:
        """Parse AI response into structured format."""
        try:
            # Try to extract JSON from response
            # Sometimes the model includes extra text
            start = response.find("{")
            end = response.rfind("}") + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                prediction = json.loads(json_str)
                
                # Validate required fields
                required = ["prediction", "confidence", "reasoning"]
                if all(field in prediction for field in required):
                    return prediction
            
            # If parsing fails, try to extract basic info
            logger.warning("Could not parse JSON response, using fallback")
            return self._extract_basic_prediction(response)
            
        except Exception as e:
            logger.error(f"Failed to parse prediction response: {e}")
            return self._extract_basic_prediction(response)

    def _extract_basic_prediction(self, response: str) -> Dict:
        """Extract basic prediction from unstructured response."""
        # Simple heuristic-based extraction
        response_lower = response.lower()
        
        if "home" in response_lower and "win" in response_lower:
            prediction = "home_win"
        elif "away" in response_lower and "win" in response_lower:
            prediction = "away_win"
        else:
            prediction = "draw"
        
        return {
            "prediction": prediction,
            "confidence": 0.6,
            "reasoning": response[:500],  # First 500 chars
            "key_factors": [],
            "suggested_bet": prediction.replace("_", " ").title(),
            "value_assessment": "medium"
        }

    def _fallback_prediction(self, match_data: Dict) -> Dict:
        """Generate fallback prediction when API fails."""
        logger.warning("Using fallback prediction")
        return {
            "prediction": "draw",
            "confidence": 0.5,
            "reasoning": "Unable to generate AI prediction. This is a fallback conservative prediction.",
            "key_factors": ["No data available"],
            "suggested_bet": "X",
            "value_assessment": "low"
        }

    async def analyze_past_predictions(
        self,
        predictions: List[Dict],
        results: List[Dict]
    ) -> Dict:
        """Analyze past predictions to improve future ones.
        
        Args:
            predictions: List of past predictions
            results: List of actual results
            
        Returns:
            Analysis and improvement suggestions
        """
        if not predictions or not results:
            return {"status": "insufficient_data"}
        
        prompt = f"""Analyze these past predictions and their outcomes:

**Predictions and Results:**
{json.dumps(list(zip(predictions[:10], results[:10])), indent=2)}

Provide analysis in JSON format:
{{
    "accuracy_rate": 0.0-1.0,
    "common_errors": ["error1", "error2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "confidence_calibration": "overconfident|underconfident|well_calibrated"
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst specialized in evaluating sports prediction models."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Past predictions analysis failed: {e}")
            return {"status": "error", "message": str(e)}
