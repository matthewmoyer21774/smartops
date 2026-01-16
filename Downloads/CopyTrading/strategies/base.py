"""
Base Strategy Detector

Abstract base class for all strategy detection implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trader import Trader


@dataclass
class StrategySignals:
    """Base class for strategy detection signals."""
    confidence_score: float = 0.0
    indicators: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'confidence_score': self.confidence_score,
            'indicators': self.indicators,
            'warnings': self.warnings
        }


class BaseStrategyDetector(ABC):
    """
    Abstract base class for strategy detection.

    Each strategy detector analyzes a trader's transaction history
    to detect specific trading patterns and behaviors.
    """

    # Strategy type identifier
    STRATEGY_TYPE: str = "base"

    # Human-readable name
    STRATEGY_NAME: str = "Base Strategy"

    # Description of what this strategy detects
    DESCRIPTION: str = "Base strategy detector"

    # Whether this strategy is copyable by humans
    IS_COPYABLE: bool = True

    @abstractmethod
    def analyze(self, trader: Trader) -> StrategySignals:
        """
        Analyze a trader and return strategy signals.

        Args:
            trader: Trader object with full transaction history

        Returns:
            StrategySignals with confidence score and indicators
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_confidence(self, signals: Dict[str, Any]) -> float:
        """
        Calculate confidence score from raw signals.

        Args:
            signals: Dictionary of detected signal values

        Returns:
            Confidence score between 0.0 and 1.0
        """
        raise NotImplementedError

    def get_strategy_info(self) -> dict:
        """Return strategy metadata."""
        return {
            'type': self.STRATEGY_TYPE,
            'name': self.STRATEGY_NAME,
            'description': self.DESCRIPTION,
            'is_copyable': self.IS_COPYABLE
        }

    @staticmethod
    def normalize_score(value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to 0-1 range.

        Args:
            value: Raw value to normalize
            min_val: Minimum expected value (maps to 0)
            max_val: Maximum expected value (maps to 1)

        Returns:
            Normalized score between 0.0 and 1.0
        """
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def weighted_average(scores: List[tuple]) -> float:
        """
        Calculate weighted average of scores.

        Args:
            scores: List of (score, weight) tuples

        Returns:
            Weighted average score
        """
        total_weight = sum(w for _, w in scores)
        if total_weight == 0:
            return 0.0
        return sum(s * w for s, w in scores) / total_weight
