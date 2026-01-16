"""
Arbitrage Strategy Detector

Detects traders using arbitrage/bot strategies characterized by:
- Very high trade counts per position
- Entry prices near extremes (>0.95 or <0.05)
- Small profit margins (0.1-2% ROI)
- Rapid trade execution
- Positions on both sides of same market
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
class ArbitrageSignals(StrategySignals):
    """Signals specific to arbitrage detection."""

    # Key metrics
    trades_per_position: float = 0.0
    avg_entry_distance_from_50: float = 0.0
    avg_profit_margin: float = 0.0
    trade_velocity: float = 0.0  # trades per hour
    hedge_ratio: float = 0.0  # % markets with both sides
    extreme_entry_ratio: float = 0.0  # % entries near 0/1

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'trades_per_position': round(self.trades_per_position, 2),
            'avg_entry_distance_from_50': round(self.avg_entry_distance_from_50, 4),
            'avg_profit_margin': round(self.avg_profit_margin, 2),
            'trade_velocity': round(self.trade_velocity, 2),
            'hedge_ratio': round(self.hedge_ratio, 4),
            'extreme_entry_ratio': round(self.extreme_entry_ratio, 4)
        })
        return base


class ArbitrageDetector(BaseStrategyDetector):
    """
    Detects arbitrage/bot trading strategies.

    Arbitrage traders exploit price inefficiencies between markets
    or capture tiny margins at extreme prices. Key characteristics:

    1. High Trade Volume: 100+ trades per position (automated)
    2. Extreme Prices: Buy at 0.95+ or 0.05- (near certainty)
    3. Tiny Margins: 0.1-2% profit per position
    4. Fast Execution: Multiple trades per minute
    5. Hedging: Positions on both Yes and No in same market
    """

    STRATEGY_TYPE = "arbitrage"
    STRATEGY_NAME = "Arbitrage Bot"
    DESCRIPTION = "Automated trading exploiting price inefficiencies with high frequency and small margins"
    IS_COPYABLE = False  # Requires bot infrastructure

    # Thresholds
    HIGH_TRADE_THRESHOLD = 50  # trades per position
    EXTREME_PRICE_THRESHOLD = 0.05  # distance from 0 or 1
    TINY_ROI_THRESHOLD = 5  # % ROI considered tiny
    HIGH_VELOCITY_THRESHOLD = 10  # trades per hour

    def analyze(self, trader: Trader) -> ArbitrageSignals:
        """
        Analyze trader for arbitrage patterns.

        Args:
            trader: Trader with full transaction history

        Returns:
            ArbitrageSignals with confidence and indicators
        """
        signals = ArbitrageSignals()

        if not trader.positions:
            return signals

        # Get transaction metrics
        tx_analyzer = TransactionAnalyzer()
        metrics = tx_analyzer.analyze(trader)

        # Calculate key indicators
        signals.trades_per_position = metrics.avg_trades_per_position

        # Entry price distance from 0.5 (fair value)
        signals.avg_entry_distance_from_50 = abs(metrics.avg_entry_price - 0.5)

        # Average profit margin
        signals.avg_profit_margin = abs(metrics.avg_roi_per_position)

        # Trade velocity
        signals.trade_velocity = metrics.trades_per_hour

        # Hedge ratio (markets with both sides)
        if metrics.unique_markets > 0:
            signals.hedge_ratio = metrics.markets_with_both_sides / metrics.unique_markets

        # Extreme entry ratio
        total_positions = len(trader.positions)
        if total_positions > 0:
            signals.extreme_entry_ratio = metrics.entries_near_extreme / total_positions

        # Build indicator dict
        signals.indicators = {
            'high_trade_count': metrics.avg_trades_per_position > self.HIGH_TRADE_THRESHOLD,
            'extreme_prices': signals.extreme_entry_ratio > 0.3,
            'tiny_margins': 0 < signals.avg_profit_margin < self.TINY_ROI_THRESHOLD,
            'high_velocity': signals.trade_velocity > self.HIGH_VELOCITY_THRESHOLD,
            'hedging_detected': signals.hedge_ratio > 0.1,
            'high_win_rate': trader.stats.win_rate > 90,
            'many_positions': total_positions > 100
        }

        # Calculate confidence score
        signals.confidence_score = self.calculate_confidence(signals.indicators)

        # Add warnings
        if signals.confidence_score > 0.7:
            signals.warnings.append("Likely automated bot - not manually copyable")
            signals.warnings.append("Requires sub-second execution speed")
        if signals.indicators['hedging_detected']:
            signals.warnings.append("Uses market hedging strategies")
        if signals.indicators['high_win_rate'] and signals.indicators['tiny_margins']:
            signals.warnings.append("High win rate with tiny margins = arbitrage signature")

        return signals

    def calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate arbitrage confidence from indicators.

        Weights:
        - High trade count: 30% (strong signal)
        - Extreme prices: 25%
        - Tiny margins: 20%
        - High velocity: 15%
        - Hedging: 10%
        """
        scores = [
            (1.0 if indicators.get('high_trade_count') else 0.0, 0.30),
            (1.0 if indicators.get('extreme_prices') else 0.0, 0.25),
            (1.0 if indicators.get('tiny_margins') else 0.0, 0.20),
            (1.0 if indicators.get('high_velocity') else 0.0, 0.15),
            (1.0 if indicators.get('hedging_detected') else 0.0, 0.10),
        ]

        base_score = self.weighted_average(scores)

        # Boost if multiple strong signals
        strong_signals = sum(1 for s, _ in scores if s > 0)
        if strong_signals >= 4:
            base_score = min(1.0, base_score * 1.2)

        # High win rate with many trades is very suspicious
        if indicators.get('high_win_rate') and indicators.get('many_positions'):
            base_score = min(1.0, base_score * 1.3)

        return round(base_score, 3)
