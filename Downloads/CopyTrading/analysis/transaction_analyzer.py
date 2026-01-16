"""
Transaction Analyzer

Deep analysis of trader transaction patterns including timing,
velocity, and behavioral patterns.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.trader import Trader
from models.position import Position


@dataclass
class TransactionMetrics:
    """Computed metrics from transaction analysis."""

    # Trade velocity
    total_trades: int = 0
    trades_per_day: float = 0.0
    trades_per_hour: float = 0.0
    max_trades_in_hour: int = 0

    # Position metrics
    avg_trades_per_position: float = 0.0
    max_trades_per_position: int = 0
    positions_with_high_trade_count: int = 0  # >50 trades

    # Price patterns
    avg_entry_price: float = 0.0
    entries_near_extreme: int = 0  # >0.95 or <0.05
    entries_at_fair_value: int = 0  # 0.4-0.6

    # Timing patterns
    active_hours: int = 0
    active_days: int = 0
    trading_span_days: float = 0.0

    # Market diversity
    unique_markets: int = 0
    markets_with_both_sides: int = 0  # Both Yes and No

    # Win/loss patterns
    win_rate: float = 0.0
    avg_win_size: float = 0.0
    avg_loss_size: float = 0.0
    profit_factor: float = 0.0

    # ROI patterns
    avg_roi_per_position: float = 0.0
    positions_with_tiny_roi: int = 0  # <2%
    positions_with_huge_roi: int = 0  # >50%

    def to_dict(self) -> dict:
        return {
            'total_trades': self.total_trades,
            'trades_per_day': round(self.trades_per_day, 2),
            'trades_per_hour': round(self.trades_per_hour, 2),
            'max_trades_in_hour': self.max_trades_in_hour,
            'avg_trades_per_position': round(self.avg_trades_per_position, 2),
            'max_trades_per_position': self.max_trades_per_position,
            'positions_with_high_trade_count': self.positions_with_high_trade_count,
            'avg_entry_price': round(self.avg_entry_price, 4),
            'entries_near_extreme': self.entries_near_extreme,
            'entries_at_fair_value': self.entries_at_fair_value,
            'active_hours': self.active_hours,
            'active_days': self.active_days,
            'trading_span_days': round(self.trading_span_days, 1),
            'unique_markets': self.unique_markets,
            'markets_with_both_sides': self.markets_with_both_sides,
            'win_rate': round(self.win_rate, 2),
            'avg_win_size': round(self.avg_win_size, 2),
            'avg_loss_size': round(self.avg_loss_size, 2),
            'profit_factor': round(self.profit_factor, 2),
            'avg_roi_per_position': round(self.avg_roi_per_position, 2),
            'positions_with_tiny_roi': self.positions_with_tiny_roi,
            'positions_with_huge_roi': self.positions_with_huge_roi
        }


class TransactionAnalyzer:
    """
    Analyzes trader transaction patterns for strategy detection.
    """

    def analyze(self, trader: Trader) -> TransactionMetrics:
        """
        Perform deep analysis of trader transactions.

        Args:
            trader: Trader with full transaction history

        Returns:
            TransactionMetrics with computed patterns
        """
        metrics = TransactionMetrics()

        if not trader.positions:
            return metrics

        # Basic counts
        metrics.total_trades = trader.stats.total_trades
        metrics.unique_markets = trader.stats.unique_markets

        # Analyze positions
        self._analyze_positions(trader, metrics)

        # Analyze timing (if we have trade data)
        if trader.trades:
            self._analyze_timing(trader, metrics)

        # Analyze win/loss patterns
        self._analyze_performance(trader, metrics)

        return metrics

    def _analyze_positions(self, trader: Trader, metrics: TransactionMetrics):
        """Analyze position-level patterns."""

        trade_counts = []
        entry_prices = []
        roi_values = []
        condition_ids = defaultdict(set)

        for key, position in trader.positions.items():
            # Trade count per position
            trade_count = len(position.trades) if position.trades else 1
            trade_counts.append(trade_count)

            if trade_count > 50:
                metrics.positions_with_high_trade_count += 1

            # Entry price patterns
            entry_price = position.avg_entry_price
            if entry_price > 0:
                entry_prices.append(entry_price)

                if entry_price > 0.95 or entry_price < 0.05:
                    metrics.entries_near_extreme += 1
                elif 0.4 <= entry_price <= 0.6:
                    metrics.entries_at_fair_value += 1

            # ROI patterns
            roi = position.roi_pct
            if position.is_resolved and position.total_cost > 0:
                roi_values.append(roi)

                if abs(roi) < 2:
                    metrics.positions_with_tiny_roi += 1
                elif abs(roi) > 50:
                    metrics.positions_with_huge_roi += 1

            # Track both-sides positions
            condition_ids[position.condition_id].add(position.outcome)

        # Calculate averages
        if trade_counts:
            metrics.avg_trades_per_position = sum(trade_counts) / len(trade_counts)
            metrics.max_trades_per_position = max(trade_counts)

        if entry_prices:
            metrics.avg_entry_price = sum(entry_prices) / len(entry_prices)

        if roi_values:
            metrics.avg_roi_per_position = sum(roi_values) / len(roi_values)

        # Count markets with both Yes and No positions
        for outcomes in condition_ids.values():
            if 'Yes' in outcomes and 'No' in outcomes:
                metrics.markets_with_both_sides += 1

    def _analyze_timing(self, trader: Trader, metrics: TransactionMetrics):
        """Analyze timing patterns from trade history."""

        if not trader.trades:
            return

        # Parse timestamps
        timestamps = []
        for trade in trader.trades:
            if hasattr(trade, 'timestamp') and trade.timestamp:
                try:
                    if isinstance(trade.timestamp, datetime):
                        timestamps.append(trade.timestamp)
                    elif isinstance(trade.timestamp, str):
                        ts = datetime.fromisoformat(trade.timestamp.replace('Z', '+00:00'))
                        timestamps.append(ts)
                except:
                    pass

        if not timestamps:
            return

        # Sort timestamps
        timestamps.sort()

        # Trading span
        first_trade = timestamps[0]
        last_trade = timestamps[-1]
        span = last_trade - first_trade
        metrics.trading_span_days = span.total_seconds() / (24 * 3600)

        # Trades per day/hour
        if metrics.trading_span_days > 0:
            metrics.trades_per_day = len(timestamps) / max(metrics.trading_span_days, 1)
            metrics.trades_per_hour = metrics.trades_per_day / 24

        # Count active days and hours
        active_days = set()
        active_hours = set()
        hourly_counts = defaultdict(int)

        for ts in timestamps:
            active_days.add(ts.date())
            active_hours.add((ts.date(), ts.hour))
            hour_key = ts.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1

        metrics.active_days = len(active_days)
        metrics.active_hours = len(active_hours)

        if hourly_counts:
            metrics.max_trades_in_hour = max(hourly_counts.values())

    def _analyze_performance(self, trader: Trader, metrics: TransactionMetrics):
        """Analyze win/loss performance patterns."""

        wins = []
        losses = []

        for position in trader.positions.values():
            if position.is_resolved:
                pnl = position.realized_pnl
                if pnl > 0:
                    wins.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))

        total_resolved = len(wins) + len(losses)

        if total_resolved > 0:
            metrics.win_rate = (len(wins) / total_resolved) * 100

        if wins:
            metrics.avg_win_size = sum(wins) / len(wins)

        if losses:
            metrics.avg_loss_size = sum(losses) / len(losses)

        # Profit factor
        total_wins = sum(wins)
        total_losses = sum(losses)
        if total_losses > 0:
            metrics.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            metrics.profit_factor = float('inf')

    def get_market_categories(self, trader: Trader) -> Dict[str, int]:
        """
        Categorize trader's markets by type.

        Returns dict of category -> position count
        """
        categories = defaultdict(int)

        keywords = {
            'sports': ['nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football', 'basketball',
                      'baseball', 'hockey', 'game', 'match', 'vs', 'super-bowl', 'playoffs'],
            'politics': ['trump', 'biden', 'election', 'president', 'senate', 'congress',
                        'democrat', 'republican', 'vote', 'poll', 'governor'],
            'crypto': ['bitcoin', 'btc', 'eth', 'ethereum', 'crypto', 'price', 'token'],
            'events': ['elon', 'musk', 'twitter', 'x-', 'tweet', 'post'],
            'finance': ['fed', 'rate', 'stock', 'market', 'gdp', 'inflation']
        }

        for position in trader.positions.values():
            slug = position.market_slug.lower() if position.market_slug else ''
            title = position.market_title.lower() if position.market_title else ''
            text = f"{slug} {title}"

            categorized = False
            for category, words in keywords.items():
                if any(word in text for word in words):
                    categories[category] += 1
                    categorized = True
                    break

            if not categorized:
                categories['other'] += 1

        return dict(categories)
