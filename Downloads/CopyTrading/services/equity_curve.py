"""
Equity curve builder for tracking P&L over time.
"""

from typing import List, Dict
from datetime import datetime, timedelta
from collections import defaultdict

import sys
sys.path.insert(0, '..')
from models.trader import Trader, EquityCurvePoint
from models.trade import Trade


class EquityCurveBuilder:
    """Build equity curve (P&L over time) for a trader."""

    def build(
        self,
        trader: Trader,
        granularity: str = 'daily'
    ) -> List[EquityCurvePoint]:
        """
        Build equity curve by replaying trades chronologically.

        Args:
            trader: Trader object with trades and positions
            granularity: 'hourly', 'daily', or 'weekly'

        Returns:
            List of EquityCurvePoint objects
        """
        if not trader.trades:
            return []

        # Group trades by time bucket
        buckets = self._bucket_trades(trader.trades, granularity)

        equity_curve = []
        cumulative_volume = 0.0

        # Track open positions: {condition_id:outcome -> (shares, cost_basis)}
        open_positions: Dict[str, tuple] = {}

        # Track realized P&L
        cumulative_realized = 0.0

        for timestamp, trades in sorted(buckets.items()):
            # Process trades in this bucket
            for trade in trades:
                key = f"{trade.condition_id}:{trade.outcome}"
                cumulative_volume += trade.size

                current_shares, current_cost = open_positions.get(key, (0.0, 0.0))

                if trade.side == 'BUY':
                    new_shares = current_shares + trade.shares
                    new_cost = current_cost + trade.size
                else:  # SELL
                    new_shares = current_shares - trade.shares
                    # Reduce cost basis proportionally
                    if current_shares > 0:
                        sell_ratio = trade.shares / current_shares
                        new_cost = current_cost * (1 - sell_ratio)
                    else:
                        new_cost = 0

                open_positions[key] = (new_shares, new_cost)

            # Check for any positions that resolved
            # We use the trader's position data to determine resolutions
            for key, (shares, cost) in list(open_positions.items()):
                if shares <= 0:
                    continue

                condition_id, outcome = key.split(':')
                pos_key = trader.get_position_key(condition_id, outcome)

                if pos_key in trader.positions:
                    pos = trader.positions[pos_key]
                    if pos.is_resolved:
                        # Calculate realized P&L for this position
                        pnl = shares * pos.resolved_price - cost
                        cumulative_realized += pnl
                        open_positions[key] = (0.0, 0.0)

            # Calculate unrealized P&L based on current prices
            unrealized = 0.0
            open_count = 0

            for key, (shares, cost) in open_positions.items():
                if shares <= 0:
                    continue

                condition_id, outcome = key.split(':')
                pos_key = trader.get_position_key(condition_id, outcome)

                if pos_key in trader.positions:
                    pos = trader.positions[pos_key]
                    if not pos.is_resolved:
                        current_value = shares * pos.current_price
                        unrealized += current_value - cost
                        open_count += 1

            equity_curve.append(EquityCurvePoint(
                timestamp=timestamp,
                cumulative_pnl=cumulative_realized + unrealized,
                realized_pnl=cumulative_realized,
                unrealized_pnl=unrealized,
                open_positions=open_count,
                total_volume=cumulative_volume
            ))

        return equity_curve

    def _bucket_trades(
        self,
        trades: List[Trade],
        granularity: str
    ) -> Dict[datetime, List[Trade]]:
        """Group trades into time buckets."""
        buckets = defaultdict(list)

        for trade in trades:
            if granularity == 'hourly':
                bucket = trade.timestamp.replace(minute=0, second=0, microsecond=0)
            elif granularity == 'daily':
                bucket = trade.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif granularity == 'weekly':
                # Start of week (Monday)
                days_since_monday = trade.timestamp.weekday()
                bucket = trade.timestamp - timedelta(days=days_since_monday)
                bucket = bucket.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                bucket = trade.timestamp

            buckets[bucket].append(trade)

        return dict(buckets)

    def calculate_max_drawdown(self, equity_curve: List[EquityCurvePoint]) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Returns:
            Max drawdown as positive number (peak to trough)
        """
        if not equity_curve:
            return 0.0

        peak = equity_curve[0].cumulative_pnl
        max_drawdown = 0.0

        for point in equity_curve:
            if point.cumulative_pnl > peak:
                peak = point.cumulative_pnl

            drawdown = peak - point.cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def calculate_sharpe_like_ratio(
        self,
        equity_curve: List[EquityCurvePoint],
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate a Sharpe-like ratio from equity curve.

        Note: This is approximate since we're using P&L changes, not returns.

        Args:
            equity_curve: List of equity points
            risk_free_rate: Daily risk-free rate (default 0)

        Returns:
            Sharpe-like ratio
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate daily P&L changes
        pnl_changes = []
        for i in range(1, len(equity_curve)):
            change = equity_curve[i].cumulative_pnl - equity_curve[i-1].cumulative_pnl
            pnl_changes.append(change)

        if not pnl_changes:
            return 0.0

        import statistics
        mean_return = statistics.mean(pnl_changes)
        if len(pnl_changes) > 1:
            std_return = statistics.stdev(pnl_changes)
        else:
            std_return = 0

        if std_return == 0:
            return 0.0

        return (mean_return - risk_free_rate) / std_return
