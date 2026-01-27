"""Base API client with retry logic and error handling."""
from typing import Any, Dict, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class BaseAPIClient:
    """Base class for API clients with common functionality."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request with retry logic."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = await self.client.get(url, headers=self._get_headers(), params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request with retry logic."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = await self.client.post(url, headers=self._get_headers(), json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            raise
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
