"""
Position model representing aggregated trades in a single market outcome.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .trade import Trade


@dataclass
class Position:
    """Aggregated position in a single market outcome."""

    condition_id: str
    outcome: str  # 'Yes' or 'No'
    market_slug: str
    market_title: str

    # Aggregated from trades
    total_shares: float = 0.0
    total_cost: float = 0.0  # Total spent to acquire shares
    avg_entry_price: float = 0.0

    # Current state
    current_price: float = 0.0
    current_value: float = 0.0

    # Resolution
    is_resolved: bool = False
    resolved_price: Optional[float] = None

    # Constituent trades
    trades: List[Trade] = field(default_factory=list)

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L based on current price."""
        if self.is_resolved:
            return 0.0
        return self.current_value - self.total_cost

    @property
    def realized_pnl(self) -> float:
        """Realized P&L if position is resolved."""
        if not self.is_resolved or self.resolved_price is None:
            return 0.0
        final_value = self.total_shares * self.resolved_price
        return final_value - self.total_cost

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized or unrealized)."""
        if self.is_resolved:
            return self.realized_pnl
        return self.unrealized_pnl

    @property
    def roi_pct(self) -> float:
        """Return on investment percentage."""
        if self.total_cost <= 0:
            return 0.0
        return (self.total_pnl / self.total_cost) * 100

    @property
    def is_winning(self) -> Optional[bool]:
        """Whether this position is/was profitable. None if unresolved."""
        if not self.is_resolved:
            return None
        return self.realized_pnl > 0

    def add_trade(self, trade: Trade):
        """Add a trade to this position."""
        self.trades.append(trade)

        if trade.side == 'BUY':
            self.total_shares += trade.shares
            self.total_cost += trade.size
        else:  # SELL
            self.total_shares -= trade.shares
            # When selling, we reduce cost basis proportionally
            if self.total_shares > 0:
                sell_ratio = trade.shares / (self.total_shares + trade.shares)
                self.total_cost *= (1 - sell_ratio)
            else:
                self.total_cost = 0

        # Recalculate average entry
        if self.total_shares > 0 and self.total_cost > 0:
            self.avg_entry_price = self.total_cost / self.total_shares
        else:
            self.avg_entry_price = 0

        # Update current value
        self.current_value = self.total_shares * self.current_price

    def update_price(self, price: float):
        """Update current market price."""
        self.current_price = price
        self.current_value = self.total_shares * price

    def resolve(self, won: bool):
        """Mark position as resolved."""
        self.is_resolved = True
        self.resolved_price = 1.0 if won else 0.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'condition_id': self.condition_id,
            'outcome': self.outcome,
            'market_slug': self.market_slug,
            'market_title': self.market_title,
            'total_shares': self.total_shares,
            'total_cost': self.total_cost,
            'avg_entry_price': self.avg_entry_price,
            'current_price': self.current_price,
            'current_value': self.current_value,
            'is_resolved': self.is_resolved,
            'resolved_price': self.resolved_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'roi_pct': self.roi_pct,
            'trade_count': len(self.trades)
        }
