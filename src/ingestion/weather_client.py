"""Weather API client for match conditions."""
from datetime import datetime
from typing import Dict, Optional, Tuple

from loguru import logger

from src.core.config import settings
from src.ingestion.base_client import BaseAPIClient


class WeatherAPIClient(BaseAPIClient):
    """Client for OpenWeatherMap API."""
    
    # Stadium coordinates (examples - expand this)
    STADIUM_COORDS = {
        "Manchester": (53.4631, -2.2914),
        "London": (51.5074, -0.1278),
        "Munich": (48.1351, 11.5820),
        "Madrid": (40.4168, -3.7038),
        "Barcelona": (41.3809, 2.1228),
        "Milan": (45.4642, 9.1900),
        "Paris": (48.8566, 2.3522),
        "Amsterdam": (52.3676, 4.9041),
        "Lisbon": (38.7223, -9.1393),
        "Vienna": (48.2082, 16.3738),
    }
    
    def __init__(self):
        super().__init__(
            base_url="https://api.openweathermap.org/data/2.5",
            api_key=settings.api.openweather_api_key,
        )
    
    def _find_stadium_coords(self, city_name: str) -> Optional[Tuple[float, float]]:
        """Find coordinates for a city/stadium."""
        # Try exact match
        if city_name in self.STADIUM_COORDS:
            return self.STADIUM_COORDS[city_name]
        
        # Try partial match
        for stadium, coords in self.STADIUM_COORDS.items():
            if city_name.lower() in stadium.lower() or stadium.lower() in city_name.lower():
                return coords
        
        logger.warning(f"No coordinates found for {city_name}")
        return None
    
    async def get_forecast(
        self,
        city_name: str,
        match_time: datetime
    ) -> Optional[Dict[str, any]]:
        """
        Get weather forecast for a match.
        
        Args:
            city_name: City or stadium name
            match_time: Match kickoff time
        
        Returns:
            Weather conditions dict with temperature, precipitation, wind, etc.
        """
        coords = self._find_stadium_coords(city_name)
        if not coords:
            return None
        
        lat, lon = coords
        
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",  # Celsius
            }
            
            data = await self._get("forecast", params=params)
            
            # Find forecast closest to match time
            forecasts = data.get("list", [])
            closest_forecast = None
            min_time_diff = float('inf')
            
            for forecast in forecasts:
                forecast_time = datetime.fromtimestamp(forecast["dt"])
                time_diff = abs((forecast_time - match_time).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_forecast = forecast
            
            if not closest_forecast:
                return None
            
            # Extract relevant weather data
            weather_data = {
                "temperature": closest_forecast["main"]["temp"],
                "feels_like": closest_forecast["main"]["feels_like"],
                "humidity": closest_forecast["main"]["humidity"],
                "pressure": closest_forecast["main"]["pressure"],
                "wind_speed": closest_forecast["wind"]["speed"],
                "wind_direction": closest_forecast["wind"].get("deg"),
                "rain_probability": closest_forecast.get("pop", 0) * 100,  # Probability of precipitation
                "rain_volume": closest_forecast.get("rain", {}).get("3h", 0),  # mm in 3 hours
                "snow_volume": closest_forecast.get("snow", {}).get("3h", 0),
                "clouds": closest_forecast["clouds"]["all"],  # Cloud coverage %
                "condition": closest_forecast["weather"][0]["main"],
                "description": closest_forecast["weather"][0]["description"],
                "forecast_time": datetime.fromtimestamp(closest_forecast["dt"]),
                "hours_before_match": min_time_diff / 3600,
            }
            
            logger.info(f"Weather for {city_name}: {weather_data['condition']}, {weather_data['temperature']}°C")
            return weather_data
            
        except Exception as e:
            logger.error(f"Failed to fetch weather for {city_name}: {e}")
            return None
    
    def assess_weather_impact(self, weather: Dict[str, any]) -> Dict[str, any]:
        """
        Assess how weather might impact the match.
        
        Returns:
            Impact assessment with scores and reasoning
        """
        if not weather:
            return {
                "impact_score": 0,  # 0-10 scale
                "favorable_for_goals": True,
                "reasoning": "No weather data available"
            }
        
        impact_score = 0
        factors = []
        
        # Temperature
        temp = weather["temperature"]
        if temp < 5:
            impact_score += 2
            factors.append(f"Cold weather ({temp}°C) may slow play")
        elif temp > 30:
            impact_score += 2
            factors.append(f"Hot weather ({temp}°C) may cause fatigue")
        
        # Rain
        if weather["rain_volume"] > 5:
            impact_score += 3
            factors.append(f"Heavy rain ({weather['rain_volume']}mm) affects ball control")
        elif weather["rain_volume"] > 1:
            impact_score += 1
            factors.append("Light rain may make pitch slippery")
        
        # Wind
        if weather["wind_speed"] > 15:
            impact_score += 2
            factors.append(f"Strong wind ({weather['wind_speed']} m/s) affects passing")
        
        # Snow
        if weather["snow_volume"] > 0:
            impact_score += 4
            factors.append("Snow significantly impacts play quality")
        
        # Determine if favorable for goals
        favorable_for_goals = impact_score < 4  # Less than moderate impact
        
        return {
            "impact_score": min(impact_score, 10),
            "favorable_for_goals": favorable_for_goals,
            "reasoning": "; ".join(factors) if factors else "Good conditions for football",
            "temperature": temp,
            "precipitation": weather["rain_volume"] + weather["snow_volume"],
            "wind_speed": weather["wind_speed"],
        }
