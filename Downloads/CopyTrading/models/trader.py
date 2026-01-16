"""
Trader model with complete profile, stats, and equity curve.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from .position import Position
from .trade import Trade


@dataclass
class TraderStats:
    """Computed statistics for a trader."""

    # Core metrics
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Volume metrics
    total_volume: float = 0.0
    total_trades: int = 0
    unique_markets: int = 0

    # Performance metrics
    roi_pct: float = 0.0
    win_rate: float = 0.0
    winning_positions: int = 0
    losing_positions: int = 0
    total_resolved: int = 0

    # Activity metrics
    first_trade_date: Optional[datetime] = None
    last_trade_date: Optional[datetime] = None
    active_days: int = 0

    # Risk metrics
    avg_position_size: float = 0.0
    max_position_size: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_volume': self.total_volume,
            'total_trades': self.total_trades,
            'unique_markets': self.unique_markets,
            'roi_pct': self.roi_pct,
            'win_rate': self.win_rate,
            'winning_positions': self.winning_positions,
            'losing_positions': self.losing_positions,
            'total_resolved': self.total_resolved,
            'first_trade_date': self.first_trade_date.isoformat() if self.first_trade_date else None,
            'last_trade_date': self.last_trade_date.isoformat() if self.last_trade_date else None,
            'active_days': self.active_days,
            'avg_position_size': self.avg_position_size,
            'max_position_size': self.max_position_size,
            'max_drawdown': self.max_drawdown
        }


@dataclass
class EquityCurvePoint:
    """Single point in equity curve."""

    timestamp: datetime
    cumulative_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    open_positions: int
    total_volume: float

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cumulative_pnl': self.cumulative_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'open_positions': self.open_positions,
            'total_volume': self.total_volume
        }


@dataclass
class Trader:
    """Complete trader profile with all data."""

    # Identity
    address: str
    name: Optional[str] = None
    pseudonym: Optional[str] = None

    # Raw data
    trades: List[Trade] = field(default_factory=list)
    positions: Dict[str, Position] = field(default_factory=dict)  # key: condition_id:outcome

    # Computed
    stats: TraderStats = field(default_factory=TraderStats)
    equity_curve: List[EquityCurvePoint] = field(default_factory=list)

    # Discovery metadata
    discovered_from_markets: List[str] = field(default_factory=list)
    discovery_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    @staticmethod
    def get_position_key(condition_id: str, outcome: str) -> str:
        """Generate unique key for position."""
        return f"{condition_id}:{outcome}"

    def get_or_create_position(self, condition_id: str, outcome: str, market_slug: str = '', market_title: str = '') -> Position:
        """Get existing position or create new one."""
        key = self.get_position_key(condition_id, outcome)
        if key not in self.positions:
            self.positions[key] = Position(
                condition_id=condition_id,
                outcome=outcome,
                market_slug=market_slug,
                market_title=market_title
            )
        return self.positions[key]

    def to_dict(self) -> dict:
        """Serialize trader for JSON export."""
        return {
            'address': self.address,
            'name': self.name,
            'pseudonym': self.pseudonym,
            'stats': self.stats.to_dict(),
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'equity_curve': [p.to_dict() for p in self.equity_curve],
            'discovered_from_markets': self.discovered_from_markets,
            'discovery_date': self.discovery_date.isoformat() if self.discovery_date else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'trade_count': len(self.trades)
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Trader {self.address[:10]}...\n"
            f"  P&L: ${self.stats.total_pnl:,.2f} (${self.stats.realized_pnl:,.2f} realized)\n"
            f"  ROI: {self.stats.roi_pct:.1f}%\n"
            f"  Win Rate: {self.stats.win_rate:.1f}% ({self.stats.winning_positions}W / {self.stats.losing_positions}L)\n"
            f"  Trades: {self.stats.total_trades} across {self.stats.unique_markets} markets\n"
            f"  Volume: ${self.stats.total_volume:,.2f}"
        )
