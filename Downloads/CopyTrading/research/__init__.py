"""
Research Module

Tools for extracting comprehensive trading data for hedge fund research.

This module contains:
- MasterDataExtractor: Comprehensive data extraction for all trader activity
- AlphaExtractor: Alpha signal extraction for actionable insights
"""

from .master_extractor import MasterDataExtractor
from .alpha_extractor import AlphaExtractor

__all__ = ['MasterDataExtractor', 'AlphaExtractor']
