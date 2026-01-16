"""
Momentum/Trend Following Strategy Detector

Detects traders following trends and other smart money:
- Entries when price already moving in direction
- Following other profitable traders into positions
- Entry prices between 0.6-0.8 (following established trend)
- Taking profits before full resolution
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
class MomentumSignals(StrategySignals):
    """Signals specific to momentum strategy detection."""

    # Key metrics
    trend_entry_ratio: float = 0.0  # % entries in trend direction
    avg_entry_price: float = 0.0  # Typically 0.6-0.8
    moderate_entry_ratio: float = 0.0  # % entries at 0.55-0.85
    diversified_markets: bool = False  # Spread across categories
    consistent_direction: float = 0.0  # % on same side (Yes/No)
    win_rate_at_trend: float = 0.0  # Win rate on trend entries

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'trend_entry_ratio': round(self.trend_entry_ratio, 4),
            'avg_entry_price': round(self.avg_entry_price, 4),
            'moderate_entry_ratio': round(self.moderate_entry_ratio, 4),
            'diversified_markets': self.diversified_markets,
            'consistent_direction': round(self.consistent_direction, 4),
            'win_rate_at_trend': round(self.win_rate_at_trend, 2)
        })
        return base


class MomentumDetector(BaseStrategyDetector):
    """
    Detects momentum/trend following strategies.

    These traders follow established trends and other smart money,
    entering positions after initial moves but before resolution.

    Characteristics:
    1. Trend Entries: Entry prices 0.55-0.85 (not extremes, not fair value)
    2. Consistent Direction: Mostly Yes or mostly No positions
    3. Diversified: Spread across multiple market categories
    4. Moderate Win Rate: 55-75% (better than random, not perfect)
    5. Reasonable ROI: 10-40% (not tiny, not extreme)
    """

    STRATEGY_TYPE = "momentum"
    STRATEGY_NAME = "Momentum Trader"
    DESCRIPTION = "Follows trends and established market direction"
    IS_COPYABLE = True  # Can follow similar strategies

    # Thresholds
    TREND_PRICE_MIN = 0.55
    TREND_PRICE_MAX = 0.85
    CONSISTENT_DIRECTION_THRESHOLD = 0.65  # 65% on same side
    MODERATE_WIN_RATE_MIN = 55
    MODERATE_WIN_RATE_MAX = 80

    def analyze(self, trader: Trader) -> MomentumSignals:
        """
        Analyze trader for momentum/trend following patterns.

        Args:
            trader: Trader with full transaction history

        Returns:
            MomentumSignals with confidence and indicators
        """
        signals = MomentumSignals()

        if not trader.positions:
            return signals

        # Get transaction metrics
        tx_analyzer = TransactionAnalyzer()
        metrics = tx_analyzer.analyze(trader)
        categories = tx_analyzer.get_market_categories(trader)

        entry_prices = []
        moderate_entries = 0
        yes_positions = 0
        no_positions = 0
        trend_wins = 0
        trend_total = 0

        for position in trader.positions.values():
            entry = position.avg_entry_price
            if entry <= 0:
                continue

            entry_prices.append(entry)

            # Count direction
            if position.outcome == 'Yes':
                yes_positions += 1
            else:
                no_positions += 1

            # Check for trend-level entries (not extreme, not fair value)
            # For Yes: 0.55-0.85 is trend following
            # For No: 0.15-0.45 is trend following (same as Yes 0.55-0.85)
            is_trend_entry = False
            if position.outcome == 'Yes':
                if self.TREND_PRICE_MIN <= entry <= self.TREND_PRICE_MAX:
                    is_trend_entry = True
                    moderate_entries += 1
            else:  # No
                effective_price = 1 - entry  # Convert to equivalent Yes price
                if self.TREND_PRICE_MIN <= effective_price <= self.TREND_PRICE_MAX:
                    is_trend_entry = True
                    moderate_entries += 1

            # Track trend entry performance
            if is_trend_entry and position.is_resolved:
                trend_total += 1
                if position.realized_pnl > 0:
                    trend_wins += 1

        # Calculate metrics
        total_positions = len(trader.positions)

        if entry_prices:
            signals.avg_entry_price = sum(entry_prices) / len(entry_prices)

        if total_positions > 0:
            signals.moderate_entry_ratio = moderate_entries / total_positions
            signals.trend_entry_ratio = signals.moderate_entry_ratio

            # Consistent direction (mostly one side)
            dominant_side = max(yes_positions, no_positions)
            signals.consistent_direction = dominant_side / total_positions

        if trend_total > 0:
            signals.win_rate_at_trend = (trend_wins / trend_total) * 100

        # Check diversification
        signals.diversified_markets = len(categories) >= 3

        # Build indicators
        win_rate = trader.stats.win_rate
        signals.indicators = {
            'trend_entries': signals.trend_entry_ratio > 0.4,
            'moderate_prices': 0.55 <= signals.avg_entry_price <= 0.75,
            'consistent_direction': signals.consistent_direction > self.CONSISTENT_DIRECTION_THRESHOLD,
            'diversified': signals.diversified_markets,
            'moderate_win_rate': self.MODERATE_WIN_RATE_MIN <= win_rate <= self.MODERATE_WIN_RATE_MAX,
            'good_trend_performance': signals.win_rate_at_trend > 55,
            'reasonable_volume': trader.stats.total_trades >= 20
        }

        # Calculate confidence
        signals.confidence_score = self.calculate_confidence(signals.indicators)

        # Add warnings/notes
        if signals.confidence_score > 0.5:
            signals.warnings.append("Follows established market trends")
        if signals.indicators['consistent_direction']:
            side = "Yes" if yes_positions > no_positions else "No"
            signals.warnings.append(f"Primarily bets {side} ({signals.consistent_direction:.0%})")
        if signals.indicators['diversified']:
            signals.warnings.append(f"Diversified across {len(categories)} categories")
        if signals.indicators['good_trend_performance']:
            signals.warnings.append(f"Good trend entry performance ({signals.win_rate_at_trend:.1f}% win rate)")

        return signals

    def calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate momentum strategy confidence from indicators.

        Weights:
        - Trend entries: 30%
        - Moderate win rate: 25%
        - Consistent direction: 20%
        - Good trend performance: 15%
        - Diversified: 10%
        """
        scores = [
            (1.0 if indicators.get('trend_entries') else 0.0, 0.30),
            (1.0 if indicators.get('moderate_win_rate') else 0.0, 0.25),
            (1.0 if indicators.get('consistent_direction') else 0.0, 0.20),
            (1.0 if indicators.get('good_trend_performance') else 0.0, 0.15),
            (1.0 if indicators.get('diversified') else 0.0, 0.10),
        ]

        base_score = self.weighted_average(scores)

        # Boost for classic momentum profile
        if (indicators.get('trend_entries') and
            indicators.get('moderate_win_rate') and
            indicators.get('consistent_direction')):
            base_score = min(1.0, base_score * 1.2)

        return round(base_score, 3)
