"""
🎯 Feature Engineering Package
==============================
Football-specific feature extraction and SPADL conversion.

Components:
- spadl_converter.py: Convert events to SPADL format
- action_valuation.py: VAEP and xT calculations
- structural_features.py: Advanced tactical features
"""

from .spadl_converter import SPADLConverter
from .structural_features import StructuralFeatureEngine

__all__ = [
    'SPADLConverter',
    'StructuralFeatureEngine',
]
