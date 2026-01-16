"""
Watchlist and Alert models for tracking traders.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import uuid


@dataclass
class AlertConfig:
    """Configuration for a specific alert type."""
    alert_type: str  # pnl_change_percent, new_position, position_closed, drawdown_threshold
    threshold: float = 0
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            'alert_type': self.alert_type,
            'threshold': self.threshold,
            'enabled': self.enabled
        }

    @staticmethod
    def from_dict(data: dict) -> 'AlertConfig':
        return AlertConfig(
            alert_type=data.get('alert_type', ''),
            threshold=data.get('threshold', 0),
            enabled=data.get('enabled', True)
        )


@dataclass
class AlertEvent:
    """A triggered alert event."""
    id: str
    watchlist_id: str
    trader_address: str
    alert_type: str
    message: str
    value: float  # The value that triggered the alert
    threshold: float  # The threshold that was exceeded
    triggered_at: datetime
    read: bool = False

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'watchlist_id': self.watchlist_id,
            'trader_address': self.trader_address,
            'alert_type': self.alert_type,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'triggered_at': self.triggered_at.isoformat(),
            'read': self.read
        }

    @staticmethod
    def from_dict(data: dict) -> 'AlertEvent':
        return AlertEvent(
            id=data.get('id', str(uuid.uuid4())),
            watchlist_id=data.get('watchlist_id', ''),
            trader_address=data.get('trader_address', ''),
            alert_type=data.get('alert_type', ''),
            message=data.get('message', ''),
            value=data.get('value', 0),
            threshold=data.get('threshold', 0),
            triggered_at=datetime.fromisoformat(data['triggered_at']) if data.get('triggered_at') else datetime.now(),
            read=data.get('read', False)
        )


@dataclass
class Watchlist:
    """A watchlist of traders to monitor."""
    id: str
    name: str
    traders: List[str] = field(default_factory=list)  # addresses
    alerts: List[AlertConfig] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_checked: Optional[datetime] = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'traders': self.traders,
            'alerts': [a.to_dict() for a in self.alerts],
            'created_at': self.created_at.isoformat(),
            'last_checked': self.last_checked.isoformat() if self.last_checked else None,
            'description': self.description
        }

    @staticmethod
    def from_dict(data: dict) -> 'Watchlist':
        return Watchlist(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', 'Unnamed'),
            traders=data.get('traders', []),
            alerts=[AlertConfig.from_dict(a) for a in data.get('alerts', [])],
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            last_checked=datetime.fromisoformat(data['last_checked']) if data.get('last_checked') else None,
            description=data.get('description', '')
        )

    @staticmethod
    def create_new(name: str, traders: List[str] = None, description: str = "") -> 'Watchlist':
        """Factory method to create a new watchlist with default alerts."""
        return Watchlist(
            id=str(uuid.uuid4()),
            name=name,
            traders=traders or [],
            alerts=[
                AlertConfig('pnl_change_percent', threshold=10, enabled=True),
                AlertConfig('new_position', threshold=0, enabled=True),
                AlertConfig('position_closed', threshold=0, enabled=True),
                AlertConfig('drawdown_threshold', threshold=1000, enabled=False)
            ],
            created_at=datetime.now(),
            description=description
        )


# Alert type definitions
ALERT_TYPES = {
    'pnl_change_percent': {
        'name': 'P&L Change',
        'description': 'Alert when P&L changes by X%',
        'default_threshold': 10
    },
    'new_position': {
        'name': 'New Position',
        'description': 'Alert when trader opens a new position',
        'default_threshold': 0
    },
    'position_closed': {
        'name': 'Position Closed',
        'description': 'Alert when trader closes a position',
        'default_threshold': 0
    },
    'drawdown_threshold': {
        'name': 'Drawdown Alert',
        'description': 'Alert when drawdown exceeds threshold in USD',
        'default_threshold': 1000
    },
    'win_rate_change': {
        'name': 'Win Rate Change',
        'description': 'Alert when win rate drops below threshold %',
        'default_threshold': 50
    }
}
