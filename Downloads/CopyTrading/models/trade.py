"""
Trade model representing a single trade on Polymarket.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """Individual trade record from Polymarket."""

    # Identifiers
    transaction_hash: str
    condition_id: str  # Market ID
    asset: str  # Token ID

    # Trade details
    timestamp: datetime
    side: str  # 'BUY' or 'SELL'
    outcome: str  # 'Yes' or 'No'
    outcome_index: int  # 0 or 1
    price: float  # 0.0 to 1.0
    size: float  # USD amount
    shares: float  # Calculated: size / price for BUY

    # Market context
    market_slug: str
    market_title: str

    # Resolution (if market closed)
    resolved_price: Optional[float] = None  # 1.0 for win, 0.0 for loss

    @property
    def is_resolved(self) -> bool:
        return self.resolved_price is not None

    def calculate_pnl(self) -> float:
        """Calculate P&L for this trade if resolved."""
        if not self.is_resolved:
            return 0.0

        if self.side == 'BUY':
            # Bought shares, now worth resolved_price each
            return self.shares * self.resolved_price - self.size
        else:
            # Sold shares (received size), owe resolved_price * shares
            return self.size - self.shares * self.resolved_price

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'transaction_hash': self.transaction_hash,
            'condition_id': self.condition_id,
            'asset': self.asset,
            'timestamp': self.timestamp.isoformat(),
            'side': self.side,
            'outcome': self.outcome,
            'outcome_index': self.outcome_index,
            'price': self.price,
            'size': self.size,
            'shares': self.shares,
            'market_slug': self.market_slug,
            'market_title': self.market_title,
            'resolved_price': self.resolved_price
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Trade':
        """Deserialize from dictionary."""
        timestamp = data['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        return cls(
            transaction_hash=data['transaction_hash'],
            condition_id=data['condition_id'],
            asset=data.get('asset', ''),
            timestamp=timestamp,
            side=data['side'],
            outcome=data['outcome'],
            outcome_index=data.get('outcome_index', 0),
            price=float(data['price']),
            size=float(data['size']),
            shares=float(data['shares']),
            market_slug=data.get('market_slug', ''),
            market_title=data.get('market_title', ''),
            resolved_price=data.get('resolved_price')
        )
