"""
Strategy analysis report model for deep trader analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CategoryStats:
    """Statistics for a market category."""
    name: str
    position_count: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_volume: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0


@dataclass
class StrategyReport:
    """Comprehensive strategy analysis report."""

    address: str
    name: Optional[str] = None
    pseudonym: Optional[str] = None

    # Strategy Classification
    strategy_type: str = "Unknown"
    strategy_confidence: str = "LOW"

    # Win Pattern Analysis
    total_positions: int = 0
    total_resolved: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win_size: float = 0.0
    avg_loss_size: float = 0.0
    profit_factor: float = 0.0  # total_wins / total_losses

    # P&L Summary
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Market Category Breakdown
    categories: Dict[str, CategoryStats] = field(default_factory=dict)

    # Position Sizing Analysis
    total_trades: int = 0
    total_volume: float = 0.0
    avg_position: float = 0.0
    median_position: float = 0.0
    max_position: float = 0.0
    min_position: float = 0.0
    position_stddev: float = 0.0

    # Entry Quality Analysis
    avg_entry_price: float = 0.0  # average price when they enter
    avg_winning_entry: float = 0.0  # avg entry price on winners
    avg_losing_entry: float = 0.0  # avg entry price on losers

    # Price Behavior
    buys_below_50: int = 0  # buys when price < 0.5
    buys_above_50: int = 0  # buys when price >= 0.5
    sells_below_50: int = 0
    sells_above_50: int = 0
    total_buys: int = 0
    total_sells: int = 0

    # Trading Activity
    unique_markets: int = 0
    active_days: int = 0
    first_trade_date: Optional[str] = None
    last_trade_date: Optional[str] = None

    # Top Positions
    top_winners: List[dict] = field(default_factory=list)
    top_losers: List[dict] = field(default_factory=list)

    # Red Flags / Verification
    suspicious_patterns: List[str] = field(default_factory=list)
    positive_signals: List[str] = field(default_factory=list)
    data_quality_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'address': self.address,
            'name': self.name,
            'pseudonym': self.pseudonym,
            'strategy_type': self.strategy_type,
            'strategy_confidence': self.strategy_confidence,
            'total_positions': self.total_positions,
            'total_resolved': self.total_resolved,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate,
            'avg_win_size': self.avg_win_size,
            'avg_loss_size': self.avg_loss_size,
            'profit_factor': self.profit_factor,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'categories': {k: vars(v) for k, v in self.categories.items()},
            'total_trades': self.total_trades,
            'total_volume': self.total_volume,
            'avg_position': self.avg_position,
            'median_position': self.median_position,
            'max_position': self.max_position,
            'position_stddev': self.position_stddev,
            'avg_entry_price': self.avg_entry_price,
            'buys_below_50_pct': (self.buys_below_50 / self.total_buys * 100) if self.total_buys > 0 else 0,
            'unique_markets': self.unique_markets,
            'active_days': self.active_days,
            'top_winners': self.top_winners,
            'top_losers': self.top_losers,
            'suspicious_patterns': self.suspicious_patterns,
            'positive_signals': self.positive_signals,
            'data_quality_issues': self.data_quality_issues
        }
