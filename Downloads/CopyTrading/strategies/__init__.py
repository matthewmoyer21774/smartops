"""
Strategy Detection Module

Analyzes trader transaction history to identify and categorize
trading strategies used by smart money traders.

Strategy Types:
- Arbitrage: Bots exploiting price inefficiencies
- Insider: Traders with apparent information advantage
- Lazy Positions: Buying already-resolved markets
- Momentum: Following trends and other smart money
"""

from .base import BaseStrategyDetector
from .arbitrage import ArbitrageDetector
from .insider import InsiderDetector
from .lazy_positions import LazyPositionDetector
from .momentum import MomentumDetector
from .classifier import StrategyClassifier, StrategyProfile

__all__ = [
    'BaseStrategyDetector',
    'ArbitrageDetector',
    'InsiderDetector',
    'LazyPositionDetector',
    'MomentumDetector',
    'StrategyClassifier',
    'StrategyProfile'
]
