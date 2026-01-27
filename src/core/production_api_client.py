#!/usr/bin/env python3
"""
Production-Grade API Client
Implements: Retry Logic, Rate Limiting, Circuit Breaker, Error Recovery
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import httpx
from loguru import logger
from functools import wraps


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class APIMetrics:
    """Track API performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by failing fast
    """
    failure_threshold: int = 5  # Open circuit after N failures
    timeout_seconds: int = 60   # Stay open for N seconds
    half_open_max_requests: int = 3  # Test with N requests in half-open
    
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_requests: int = 0
    
    def record_success(self) -> None:
        """Record successful request"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_requests += 1
            if self.half_open_requests >= self.half_open_max_requests:
                self.state = CircuitState.CLOSED
                self.half_open_requests = 0
                logger.info("Circuit breaker: CLOSED (recovered)")
    
    def record_failure(self) -> None:
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker: OPEN (failed during recovery)")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker: OPEN (threshold reached: {self.failure_count})")
    
    def can_attempt_request(self) -> bool:
        """Check if request should be attempted"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_requests = 0
                    logger.info("Circuit breaker: HALF_OPEN (testing recovery)")
                    return True
            return False
        
        # HALF_OPEN state
        return True


class RateLimiter:
    """
    Token bucket rate limiter
    Prevents API rate limit violations
    """
    
    def __init__(self, max_requests: int, time_window_seconds: int):
        self.max_requests = max_requests
        self.time_window = time_window_seconds
        self.requests: List[datetime] = []
    
    async def acquire(self) -> None:
        """Wait until request can be made"""
        now = datetime.now()
        
        # Remove old requests outside time window
        cutoff = now - timedelta(seconds=self.time_window)
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]
        
        # If at limit, wait
        if len(self.requests) >= self.max_requests:
            oldest_request = self.requests[0]
            wait_until = oldest_request + timedelta(seconds=self.time_window)
            wait_seconds = (wait_until - now).total_seconds()
            
            if wait_seconds > 0:
                logger.warning(f"Rate limit reached, waiting {wait_seconds:.1f}s")
                await asyncio.sleep(wait_seconds)
                # Retry acquire after waiting
                return await self.acquire()
        
        # Add current request
        self.requests.append(now)


class ProductionAPIClient:
    """
    Production-grade API client with:
    - Exponential backoff retry
    - Circuit breaker pattern
    - Rate limiting
    - Request/response logging
    - Performance metrics
    """
    
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60
    ):
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        
        # Initialize components
        self.metrics = APIMetrics()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            timeout_seconds=circuit_breaker_timeout
        )
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        
        # HTTP client (reuse connection pool)
        self.client = httpx.AsyncClient(timeout=timeout_seconds)
        
        logger.info(f"Initialized {name} API client: {base_url}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "User-Agent": "TelegramSoccer/1.0",
            "Accept": "application/json"
        }
        
        if self.api_key:
            # Different APIs use different auth header formats
            if "football-data.org" in self.base_url:
                headers["X-Auth-Token"] = self.api_key
            elif "api-football" in self.base_url:
                headers["x-rapidapi-key"] = self.api_key
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make GET request with full production features
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            custom_headers: Additional headers
        
        Returns:
            JSON response or None on failure
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt_request():
            logger.error(f"{self.name}: Circuit breaker OPEN, skipping request")
            return None
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        # Build request
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        if custom_headers:
            headers.update(custom_headers)
        
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                self.metrics.total_requests += 1
                self.metrics.last_request_time = datetime.now()
                
                logger.debug(f"{self.name}: GET {endpoint} (attempt {attempt + 1}/{self.max_retries})")
                
                response = await self.client.get(url, headers=headers, params=params)
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.total_latency_ms += latency_ms
                
                # Handle response
                if response.status_code == 200:
                    self.metrics.successful_requests += 1
                    self.circuit_breaker.record_success()
                    
                    logger.debug(f"{self.name}: SUCCESS ({latency_ms:.0f}ms)")
                    return response.json()
                
                elif response.status_code == 429:
                    # Rate limit hit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"{self.name}: Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                
                elif response.status_code >= 500:
                    # Server error - retry
                    logger.warning(f"{self.name}: Server error {response.status_code}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                else:
                    # Client error - don't retry
                    logger.error(f"{self.name}: Client error {response.status_code}: {response.text[:200]}")
                    self.metrics.failed_requests += 1
                    self.metrics.last_error = f"HTTP {response.status_code}"
                    return None
            
            except httpx.TimeoutException:
                logger.warning(f"{self.name}: Timeout on attempt {attempt + 1}")
                last_exception = "Timeout"
                await asyncio.sleep(2 ** attempt)
            
            except httpx.HTTPError as e:
                logger.warning(f"{self.name}: HTTP error on attempt {attempt + 1}: {e}")
                last_exception = str(e)
                await asyncio.sleep(2 ** attempt)
            
            except Exception as e:
                logger.error(f"{self.name}: Unexpected error: {e}")
                last_exception = str(e)
                break
        
        # All retries exhausted
        self.metrics.failed_requests += 1
        self.metrics.last_error = last_exception
        self.metrics.consecutive_failures += 1
        self.circuit_breaker.record_failure()
        
        logger.error(f"{self.name}: Request failed after {self.max_retries} attempts")
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "name": self.name,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": f"{self.metrics.success_rate * 100:.1f}%",
            "average_latency_ms": f"{self.metrics.average_latency_ms:.0f}",
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "consecutive_failures": self.metrics.consecutive_failures,
            "last_error": self.metrics.last_error or "None"
        }
    
    async def close(self) -> None:
        """Close HTTP client"""
        await self.client.aclose()
        logger.info(f"{self.name} client closed")


# Decorator for automatic retry with production features
def production_api_call(max_retries: int = 3, backoff_factor: float = 2.0):
    """
    Decorator for production API calls
    
    Usage:
        @production_api_call(max_retries=3)
        async def fetch_matches(self) -> List[Dict]:
            # Your API call here
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    result = await func(*args, **kwargs)
                    return result
                
                except Exception as e:
                    last_exception = e
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        await asyncio.sleep(wait_time)
            
            logger.error(f"{func.__name__} failed after {max_retries} attempts: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    async def test():
        # Initialize client
        client = ProductionAPIClient(
            name="OpenLigaDB",
            base_url="https://api.openligadb.de",
            rate_limit_requests=100,
            rate_limit_window=60
        )
        
        # Make request
        data = await client.get("getmatchdata/bl1")
        
        if data:
            print(f"✅ Got {len(data)} matches")
        else:
            print("❌ Request failed")
        
        # Show metrics
        metrics = client.get_metrics()
        print(f"\n📊 Metrics: {metrics}")
        
        await client.close()
    
    asyncio.run(test())
