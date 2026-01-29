"""
🧠 Foundation Models Package
============================
Core LLM stack for structural reasoning using DeepSeek-LLM 7B.

Components:
- deepseek_engine.py: DeepSeek LLM inference client
- vllm_server.py: vLLM high-throughput inference server
- model_cache.py: Analysis caching for cost control
- prompt_templates.py: Structured prompts for football analysis
"""

from .deepseek_engine import DeepSeekEngine, DeepSeekConfig
from .model_cache import ModelCache

__all__ = [
    'DeepSeekEngine',
    'DeepSeekConfig', 
    'ModelCache',
]
