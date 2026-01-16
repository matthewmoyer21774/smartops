"""
Insider/Informed Trading Detector

Detects traders with apparent information advantage characterized by:
- Large positions before major price moves
- Entry prices far from 0.5 that resolve correctly
- Sports/event betting with perfect timing
- High ROI (>50%) on time-sensitive markets
- Concentrated positions (few markets, big bets)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trader import Trader
from strategies.base import BaseStrategyDetector, StrategySignals
from analysis.transaction_analyzer import TransactionAnalyzer


@dataclass
class InsiderSignals(StrategySignals):
    """Signals specific to insider/informed trading detection."""

    # Key metrics
    avg_entry_vs_resolution: float = 0.0  # How far entry was from 50/50
    perfect_call_rate: float = 0.0  # Wins where entry < 0.3 or > 0.7
    position_concentration: float = 0.0  # % volume in top 3 positions
    event_market_ratio: float = 0.0  # % in sports/time-sensitive
    avg_winning_roi: float = 0.0  # ROI on winning positions
    large_position_wins: int = 0  # Big bets that won

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'avg_entry_vs_resolution': round(self.avg_entry_vs_resolution, 4),
            'perfect_call_rate': round(self.perfect_call_rate, 4),
            'position_concentration': round(self.position_concentration, 4),
            'event_market_ratio': round(self.event_market_ratio, 4),
            'avg_winning_roi': round(self.avg_winning_roi, 2),
            'large_position_wins': self.large_position_wins
        })
        return base


class InsiderDetector(BaseStrategyDetector):
    """
    Detects insider/informed trading patterns.

    These traders appear to have advance knowledge of outcomes,
    especially on time-sensitive events like sports. Characteristics:

    1. Entry Far From Fair Value: Buying at 0.2-0.3 on outcomes that win
    2. Time-Sensitive Markets: Focus on sports/events with fixed end times
    3. High ROI: 50%+ returns on resolved positions
    4. Concentrated Bets: Few markets, large position sizes
    5. Perfect Timing: Positions taken shortly before resolution
    """

    STRATEGY_TYPE = "insider"
    STRATEGY_NAME = "Informed Trader"
    DESCRIPTION = "Trader with apparent information advantage on time-sensitive events"
    IS_COPYABLE = False  # Requires information edge

    # Thresholds
    PERFECT_CALL_PRICE = 0.35  # Entry below this on Yes that wins = perfect call
    HIGH_ROI_THRESHOLD = 50  # % ROI considered exceptional
    LARGE_POSITION_THRESHOLD = 10000  # $ size for "large" position
    CONCENTRATION_THRESHOLD = 0.5  # % volume in top 3

    # Keywords for time-sensitive markets
    EVENT_KEYWORDS = [
        'nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football', 'basketball',
        'game', 'match', 'vs', 'super-bowl', 'playoffs', 'finals',
        'fight', 'ufc', 'boxing', 'race', 'olympics'
    ]

    def analyze(self, trader: Trader) -> InsiderSignals:
        """
        Analyze trader for insider/informed trading patterns.

        Args:
            trader: Trader with full transaction history

        Returns:
            InsiderSignals with confidence and indicators
        """
        signals = InsiderSignals()

        if not trader.positions:
            return signals

        # Analyze positions
        perfect_calls = 0
        resolved_positions = 0
        winning_rois = []
        position_volumes = []
        event_positions = 0
        large_wins = 0
        entry_distances = []

        for position in trader.positions.values():
            cost = position.total_cost
            position_volumes.append(cost)

            # Check if time-sensitive market
            slug = (position.market_slug or '').lower()
            title = (position.market_title or '').lower()
            text = f"{slug} {title}"
            is_event = any(kw in text for kw in self.EVENT_KEYWORDS)
            if is_event:
                event_positions += 1

            if position.is_resolved:
                resolved_positions += 1
                entry = position.avg_entry_price
                won = position.realized_pnl > 0
                roi = position.roi_pct

                # Track entry distance from 0.5
                if entry > 0:
                    entry_distances.append(abs(entry - 0.5))

                # Perfect call detection
                if won:
                    winning_rois.append(roi)

                    # Entry below 0.35 on Yes that wins = perfect call
                    if position.outcome == 'Yes' and entry < self.PERFECT_CALL_PRICE:
                        perfect_calls += 1
                    # Entry above 0.65 on No that wins = perfect call
                    elif position.outcome == 'No' and entry > (1 - self.PERFECT_CALL_PRICE):
                        perfect_calls += 1

                    # Large position wins
                    if cost > self.LARGE_POSITION_THRESHOLD:
                        large_wins += 1

        # Calculate metrics
        total_positions = len(trader.positions)

        if resolved_positions > 0:
            signals.perfect_call_rate = perfect_calls / resolved_positions

        if entry_distances:
            signals.avg_entry_vs_resolution = sum(entry_distances) / len(entry_distances)

        if winning_rois:
            signals.avg_winning_roi = sum(winning_rois) / len(winning_rois)

        signals.large_position_wins = large_wins

        if total_positions > 0:
            signals.event_market_ratio = event_positions / total_positions

        # Position concentration (top 3 as % of total)
        if position_volumes:
            position_volumes.sort(reverse=True)
            total_volume = sum(position_volumes)
            if total_volume > 0:
                top3_volume = sum(position_volumes[:3])
                signals.position_concentration = top3_volume / total_volume

        # Build indicators
        signals.indicators = {
            'perfect_calls': signals.perfect_call_rate > 0.2,
            'high_roi_wins': signals.avg_winning_roi > self.HIGH_ROI_THRESHOLD,
            'concentrated_bets': signals.position_concentration > self.CONCENTRATION_THRESHOLD,
            'event_focus': signals.event_market_ratio > 0.3,
            'large_position_wins': large_wins > 2,
            'entry_far_from_fair': signals.avg_entry_vs_resolution > 0.2,
            'few_positions': total_positions < 50
        }

        # Calculate confidence
        signals.confidence_score = self.calculate_confidence(signals.indicators)

        # Add warnings
        if signals.confidence_score > 0.5:
            signals.warnings.append("Shows patterns of informed trading")
        if signals.indicators['perfect_calls']:
            signals.warnings.append(f"High rate of 'perfect calls' ({signals.perfect_call_rate:.0%})")
        if signals.indicators['event_focus']:
            signals.warnings.append("Focuses on time-sensitive event markets")
        if signals.indicators['large_position_wins']:
            signals.warnings.append(f"{large_wins} large position wins (>${self.LARGE_POSITION_THRESHOLD:,})")

        return signals

    def calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate insider trading confidence from indicators.

        Weights:
        - Perfect calls: 30% (strongest signal)
        - High ROI wins: 25%
        - Event focus: 20%
        - Large position wins: 15%
        - Concentrated bets: 10%
        """
        scores = [
            (1.0 if indicators.get('perfect_calls') else 0.0, 0.30),
            (1.0 if indicators.get('high_roi_wins') else 0.0, 0.25),
            (1.0 if indicators.get('event_focus') else 0.0, 0.20),
            (1.0 if indicators.get('large_position_wins') else 0.0, 0.15),
            (1.0 if indicators.get('concentrated_bets') else 0.0, 0.10),
        ]

        base_score = self.weighted_average(scores)

        # Boost if multiple strong signals together
        if indicators.get('perfect_calls') and indicators.get('event_focus'):
            base_score = min(1.0, base_score * 1.3)

        if indicators.get('high_roi_wins') and indicators.get('large_position_wins'):
            base_score = min(1.0, base_score * 1.2)

        return round(base_score, 3)
