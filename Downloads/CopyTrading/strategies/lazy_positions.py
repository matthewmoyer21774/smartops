"""
Lazy Position Buying Detector

Detects traders buying already-resolved or near-resolved markets:
- Buying at 0.99+ or 0.01- on markets about to resolve
- Positions in markets where outcome is already known
- Low risk, guaranteed small profit strategy
- Exploiting markets others abandoned
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trader import Trader
from strategies.base import BaseStrategyDetector, StrategySignals
from analysis.transaction_analyzer import TransactionAnalyzer


@dataclass
class LazyPositionSignals(StrategySignals):
    """Signals specific to lazy position detection."""

    # Key metrics
    near_resolution_buys: int = 0  # Buys at 0.99+ or 0.01-
    guaranteed_profit_positions: int = 0  # Near-certain wins
    avg_certainty_level: float = 0.0  # How close to 0/1
    resolved_market_ratio: float = 0.0  # % in resolved markets
    tiny_roi_wins: int = 0  # Wins with <5% ROI
    perfect_win_rate: float = 0.0  # Win rate on near-certain positions

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'near_resolution_buys': self.near_resolution_buys,
            'guaranteed_profit_positions': self.guaranteed_profit_positions,
            'avg_certainty_level': round(self.avg_certainty_level, 4),
            'resolved_market_ratio': round(self.resolved_market_ratio, 4),
            'tiny_roi_wins': self.tiny_roi_wins,
            'perfect_win_rate': round(self.perfect_win_rate, 4)
        })
        return base


class LazyPositionDetector(BaseStrategyDetector):
    """
    Detects lazy position buying strategies.

    These traders buy into markets that are essentially already decided,
    capturing small guaranteed profits from abandoned liquidity.

    Characteristics:
    1. Extreme Entry Prices: 0.99+ for Yes, 0.01- for No
    2. Near-Certain Outcomes: Buying when result is almost known
    3. Small ROI: 1-5% returns per position
    4. High Win Rate: 95%+ on these positions
    5. Low Risk: Essentially guaranteed profits
    """

    STRATEGY_TYPE = "lazy_positions"
    STRATEGY_NAME = "Lazy Position Buyer"
    DESCRIPTION = "Buys nearly-resolved markets for small guaranteed profits"
    IS_COPYABLE = True  # Can be replicated manually

    # Thresholds
    NEAR_CERTAIN_THRESHOLD = 0.95  # 0.95+ or 0.05- considered near-certain
    TINY_ROI_THRESHOLD = 10  # % ROI considered tiny
    HIGH_WIN_RATE_THRESHOLD = 90  # % win rate on near-certain

    def analyze(self, trader: Trader) -> LazyPositionSignals:
        """
        Analyze trader for lazy position buying patterns.

        Args:
            trader: Trader with full transaction history

        Returns:
            LazyPositionSignals with confidence and indicators
        """
        signals = LazyPositionSignals()

        if not trader.positions:
            return signals

        near_certain_positions = []
        near_certain_wins = 0
        tiny_roi_wins = 0
        certainty_levels = []
        resolved_count = 0

        for position in trader.positions.values():
            entry = position.avg_entry_price
            if entry <= 0:
                continue

            # Determine certainty level (distance from 0.5)
            certainty = abs(entry - 0.5) * 2  # 0 = fair value, 1 = certain

            # Check for near-certain entries
            is_near_certain = False
            if position.outcome == 'Yes' and entry >= self.NEAR_CERTAIN_THRESHOLD:
                is_near_certain = True
                certainty_levels.append(entry)
            elif position.outcome == 'No' and entry <= (1 - self.NEAR_CERTAIN_THRESHOLD):
                is_near_certain = True
                certainty_levels.append(1 - entry)

            if is_near_certain:
                signals.near_resolution_buys += 1
                near_certain_positions.append(position)

                if position.is_resolved and position.realized_pnl > 0:
                    near_certain_wins += 1
                    signals.guaranteed_profit_positions += 1

                    # Check for tiny ROI wins
                    if 0 < position.roi_pct < self.TINY_ROI_THRESHOLD:
                        tiny_roi_wins += 1

            if position.is_resolved:
                resolved_count += 1

        # Calculate metrics
        total_positions = len(trader.positions)

        if certainty_levels:
            signals.avg_certainty_level = sum(certainty_levels) / len(certainty_levels)

        if near_certain_positions:
            near_certain_resolved = sum(1 for p in near_certain_positions if p.is_resolved)
            if near_certain_resolved > 0:
                signals.perfect_win_rate = (near_certain_wins / near_certain_resolved) * 100

        signals.tiny_roi_wins = tiny_roi_wins

        if total_positions > 0:
            signals.resolved_market_ratio = resolved_count / total_positions

        # Build indicators
        signals.indicators = {
            'many_near_certain': signals.near_resolution_buys > 10,
            'high_certainty': signals.avg_certainty_level > 0.95,
            'perfect_win_rate': signals.perfect_win_rate > self.HIGH_WIN_RATE_THRESHOLD,
            'tiny_profits': tiny_roi_wins > 5,
            'mostly_resolved': signals.resolved_market_ratio > 0.7,
            'guaranteed_profits': signals.guaranteed_profit_positions > 5
        }

        # Calculate confidence
        signals.confidence_score = self.calculate_confidence(signals.indicators)

        # Add warnings
        if signals.confidence_score > 0.5:
            signals.warnings.append("Primarily trades near-certain outcomes")
        if signals.indicators['tiny_profits']:
            signals.warnings.append(f"{tiny_roi_wins} positions with tiny (<{self.TINY_ROI_THRESHOLD}%) ROI")
        if signals.indicators['high_certainty']:
            signals.warnings.append(f"Average entry certainty: {signals.avg_certainty_level:.1%}")
        if signals.indicators['guaranteed_profits']:
            signals.warnings.append("Low risk strategy - profits nearly guaranteed")

        return signals

    def calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate lazy position confidence from indicators.

        Weights:
        - Many near-certain: 30%
        - High certainty level: 25%
        - Perfect win rate: 20%
        - Tiny profits: 15%
        - Guaranteed profits: 10%
        """
        scores = [
            (1.0 if indicators.get('many_near_certain') else 0.0, 0.30),
            (1.0 if indicators.get('high_certainty') else 0.0, 0.25),
            (1.0 if indicators.get('perfect_win_rate') else 0.0, 0.20),
            (1.0 if indicators.get('tiny_profits') else 0.0, 0.15),
            (1.0 if indicators.get('guaranteed_profits') else 0.0, 0.10),
        ]

        base_score = self.weighted_average(scores)

        # Strong signal combo
        if indicators.get('many_near_certain') and indicators.get('perfect_win_rate'):
            base_score = min(1.0, base_score * 1.3)

        return round(base_score, 3)
