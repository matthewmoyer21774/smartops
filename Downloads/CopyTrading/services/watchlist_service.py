"""
Service for managing watchlists and alerts.
"""

from typing import List, Dict, Optional
from datetime import datetime
import os
import json
import uuid

import sys
sys.path.insert(0, '..')
from models.watchlist import Watchlist, AlertConfig, AlertEvent, ALERT_TYPES
from cache.file_cache import TraderCache
from config import DATA_DIR


class WatchlistService:
    """Manage watchlists, track traders, and trigger alerts."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.watchlists_dir = os.path.join(data_dir, "watchlists")
        self.alerts_file = os.path.join(data_dir, "alerts_history.json")
        self.snapshots_dir = os.path.join(data_dir, "snapshots")
        self.cache = TraderCache()

        # Create directories
        for d in [self.watchlists_dir, self.snapshots_dir]:
            os.makedirs(d, exist_ok=True)

    # ==========================================================================
    # WATCHLIST CRUD
    # ==========================================================================

    def create_watchlist(
        self,
        name: str,
        traders: List[str] = None,
        description: str = ""
    ) -> Watchlist:
        """
        Create a new watchlist.

        Args:
            name: Watchlist name
            traders: List of trader addresses
            description: Optional description

        Returns:
            Created Watchlist object
        """
        watchlist = Watchlist.create_new(name, traders, description)
        self._save_watchlist(watchlist)

        # Take initial snapshot of traders
        self._take_snapshot(watchlist.id, traders or [])

        return watchlist

    def get_watchlist(self, watchlist_id: str) -> Optional[Watchlist]:
        """Get watchlist by ID."""
        filepath = os.path.join(self.watchlists_dir, f"{watchlist_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Watchlist.from_dict(data)
        return None

    def list_watchlists(self) -> List[Watchlist]:
        """List all watchlists."""
        watchlists = []
        if not os.path.exists(self.watchlists_dir):
            return watchlists

        for filename in os.listdir(self.watchlists_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.watchlists_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    watchlists.append(Watchlist.from_dict(data))

        return watchlists

    def update_watchlist(
        self,
        watchlist_id: str,
        name: str = None,
        traders: List[str] = None,
        description: str = None
    ) -> Optional[Watchlist]:
        """
        Update an existing watchlist.

        Args:
            watchlist_id: Watchlist ID
            name: New name (optional)
            traders: New trader list (optional)
            description: New description (optional)

        Returns:
            Updated Watchlist or None if not found
        """
        watchlist = self.get_watchlist(watchlist_id)
        if not watchlist:
            return None

        if name is not None:
            watchlist.name = name
        if traders is not None:
            watchlist.traders = traders
            self._take_snapshot(watchlist_id, traders)
        if description is not None:
            watchlist.description = description

        self._save_watchlist(watchlist)
        return watchlist

    def delete_watchlist(self, watchlist_id: str) -> bool:
        """Delete a watchlist."""
        filepath = os.path.join(self.watchlists_dir, f"{watchlist_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            # Also remove snapshots
            snapshot_file = os.path.join(self.snapshots_dir, f"{watchlist_id}.json")
            if os.path.exists(snapshot_file):
                os.remove(snapshot_file)
            return True
        return False

    def _save_watchlist(self, watchlist: Watchlist):
        """Save watchlist to file."""
        filepath = os.path.join(self.watchlists_dir, f"{watchlist.id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(watchlist.to_dict(), f, indent=2)

    # ==========================================================================
    # ALERTS CONFIGURATION
    # ==========================================================================

    def configure_alerts(
        self,
        watchlist_id: str,
        alerts: List[Dict]
    ) -> Optional[Watchlist]:
        """
        Configure alerts for a watchlist.

        Args:
            watchlist_id: Watchlist ID
            alerts: List of alert configs [{alert_type, threshold, enabled}]

        Returns:
            Updated Watchlist or None
        """
        watchlist = self.get_watchlist(watchlist_id)
        if not watchlist:
            return None

        watchlist.alerts = [AlertConfig.from_dict(a) for a in alerts]
        self._save_watchlist(watchlist)
        return watchlist

    def get_alert_types(self) -> Dict:
        """Get available alert types and their defaults."""
        return ALERT_TYPES

    # ==========================================================================
    # STATUS & MONITORING
    # ==========================================================================

    def get_watchlist_status(self, watchlist_id: str) -> Dict:
        """
        Get current status of all traders in a watchlist.

        Args:
            watchlist_id: Watchlist ID

        Returns:
            Dict with current status of each trader
        """
        watchlist = self.get_watchlist(watchlist_id)
        if not watchlist:
            return {'error': 'Watchlist not found'}

        # Load previous snapshot
        prev_snapshot = self._load_snapshot(watchlist_id)

        trader_statuses = []
        for addr in watchlist.traders:
            trader = self.cache.load_trader(addr)
            if trader:
                stats = trader.get('stats', {})
                prev_stats = prev_snapshot.get(addr, {})

                # Calculate changes
                pnl_change = stats.get('total_pnl', 0) - prev_stats.get('total_pnl', 0)
                pnl_change_pct = (
                    (pnl_change / abs(prev_stats.get('total_pnl', 1))) * 100
                    if prev_stats.get('total_pnl') else 0
                )

                # Count new/closed positions
                current_positions = set(trader.get('positions', {}).keys())
                prev_positions = set(prev_stats.get('position_keys', []))
                new_positions = current_positions - prev_positions
                closed_positions = prev_positions - current_positions

                trader_statuses.append({
                    'address': addr,
                    'name': trader.get('name') or trader.get('pseudonym'),
                    'current_pnl': stats.get('total_pnl', 0),
                    'pnl_change': round(pnl_change, 2),
                    'pnl_change_pct': round(pnl_change_pct, 2),
                    'win_rate': stats.get('win_rate', 0),
                    'open_positions': len([
                        p for p in trader.get('positions', {}).values()
                        if not p.get('is_resolved', True)
                    ]),
                    'new_positions_count': len(new_positions),
                    'closed_positions_count': len(closed_positions),
                    'last_updated': trader.get('last_updated')
                })
            else:
                trader_statuses.append({
                    'address': addr,
                    'name': None,
                    'error': 'Trader not in cache'
                })

        # Update last checked
        watchlist.last_checked = datetime.now()
        self._save_watchlist(watchlist)

        return {
            'watchlist_id': watchlist_id,
            'watchlist_name': watchlist.name,
            'checked_at': datetime.now().isoformat(),
            'traders': trader_statuses,
            'trader_count': len(watchlist.traders)
        }

    def check_alerts(self, watchlist_id: str) -> List[AlertEvent]:
        """
        Check for triggered alerts on a watchlist.

        Args:
            watchlist_id: Watchlist ID

        Returns:
            List of triggered AlertEvents
        """
        watchlist = self.get_watchlist(watchlist_id)
        if not watchlist:
            return []

        prev_snapshot = self._load_snapshot(watchlist_id)
        triggered = []

        for addr in watchlist.traders:
            trader = self.cache.load_trader(addr)
            if not trader:
                continue

            stats = trader.get('stats', {})
            prev_stats = prev_snapshot.get(addr, {})

            for alert_config in watchlist.alerts:
                if not alert_config.enabled:
                    continue

                event = self._check_single_alert(
                    watchlist_id, addr, alert_config, stats, prev_stats, trader
                )
                if event:
                    triggered.append(event)

        # Save triggered alerts
        if triggered:
            self._save_alerts(triggered)

        # Update snapshot
        self._take_snapshot(watchlist_id, watchlist.traders)

        return triggered

    def _check_single_alert(
        self,
        watchlist_id: str,
        trader_address: str,
        alert_config: AlertConfig,
        current_stats: Dict,
        prev_stats: Dict,
        trader: Dict
    ) -> Optional[AlertEvent]:
        """Check if a single alert condition is triggered."""

        if alert_config.alert_type == 'pnl_change_percent':
            prev_pnl = prev_stats.get('total_pnl', 0)
            curr_pnl = current_stats.get('total_pnl', 0)
            if prev_pnl != 0:
                change_pct = abs((curr_pnl - prev_pnl) / prev_pnl * 100)
                if change_pct >= alert_config.threshold:
                    return AlertEvent(
                        id=str(uuid.uuid4()),
                        watchlist_id=watchlist_id,
                        trader_address=trader_address,
                        alert_type='pnl_change_percent',
                        message=f"P&L changed by {change_pct:.1f}% (${prev_pnl:,.0f} -> ${curr_pnl:,.0f})",
                        value=change_pct,
                        threshold=alert_config.threshold,
                        triggered_at=datetime.now()
                    )

        elif alert_config.alert_type == 'new_position':
            current_positions = set(trader.get('positions', {}).keys())
            prev_positions = set(prev_stats.get('position_keys', []))
            new_positions = current_positions - prev_positions
            if new_positions:
                return AlertEvent(
                    id=str(uuid.uuid4()),
                    watchlist_id=watchlist_id,
                    trader_address=trader_address,
                    alert_type='new_position',
                    message=f"Opened {len(new_positions)} new position(s)",
                    value=len(new_positions),
                    threshold=0,
                    triggered_at=datetime.now()
                )

        elif alert_config.alert_type == 'position_closed':
            current_positions = set(trader.get('positions', {}).keys())
            prev_positions = set(prev_stats.get('position_keys', []))
            closed_positions = prev_positions - current_positions
            if closed_positions:
                return AlertEvent(
                    id=str(uuid.uuid4()),
                    watchlist_id=watchlist_id,
                    trader_address=trader_address,
                    alert_type='position_closed',
                    message=f"Closed {len(closed_positions)} position(s)",
                    value=len(closed_positions),
                    threshold=0,
                    triggered_at=datetime.now()
                )

        elif alert_config.alert_type == 'drawdown_threshold':
            max_dd = current_stats.get('max_drawdown', 0)
            if max_dd >= alert_config.threshold:
                return AlertEvent(
                    id=str(uuid.uuid4()),
                    watchlist_id=watchlist_id,
                    trader_address=trader_address,
                    alert_type='drawdown_threshold',
                    message=f"Drawdown ${max_dd:,.0f} exceeds threshold ${alert_config.threshold:,.0f}",
                    value=max_dd,
                    threshold=alert_config.threshold,
                    triggered_at=datetime.now()
                )

        elif alert_config.alert_type == 'win_rate_change':
            win_rate = current_stats.get('win_rate', 0)
            if win_rate < alert_config.threshold:
                return AlertEvent(
                    id=str(uuid.uuid4()),
                    watchlist_id=watchlist_id,
                    trader_address=trader_address,
                    alert_type='win_rate_change',
                    message=f"Win rate {win_rate:.1f}% dropped below {alert_config.threshold}%",
                    value=win_rate,
                    threshold=alert_config.threshold,
                    triggered_at=datetime.now()
                )

        return None

    # ==========================================================================
    # ALERTS HISTORY
    # ==========================================================================

    def get_alerts_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history."""
        if not os.path.exists(self.alerts_file):
            return []

        with open(self.alerts_file, 'r', encoding='utf-8') as f:
            alerts = json.load(f)

        # Sort by triggered_at descending
        alerts.sort(key=lambda a: a.get('triggered_at', ''), reverse=True)
        return alerts[:limit]

    def _save_alerts(self, events: List[AlertEvent]):
        """Append alerts to history file."""
        existing = []
        if os.path.exists(self.alerts_file):
            with open(self.alerts_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)

        for event in events:
            existing.append(event.to_dict())

        # Keep last 500 alerts
        existing = existing[-500:]

        with open(self.alerts_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2)

    # ==========================================================================
    # SNAPSHOTS
    # ==========================================================================

    def _take_snapshot(self, watchlist_id: str, traders: List[str]):
        """Save current state of traders for comparison."""
        snapshot = {}

        for addr in traders:
            trader = self.cache.load_trader(addr)
            if trader:
                stats = trader.get('stats', {})
                snapshot[addr] = {
                    'total_pnl': stats.get('total_pnl', 0),
                    'win_rate': stats.get('win_rate', 0),
                    'max_drawdown': stats.get('max_drawdown', 0),
                    'position_keys': list(trader.get('positions', {}).keys()),
                    'snapshot_at': datetime.now().isoformat()
                }

        filepath = os.path.join(self.snapshots_dir, f"{watchlist_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2)

    def _load_snapshot(self, watchlist_id: str) -> Dict:
        """Load previous snapshot for comparison."""
        filepath = os.path.join(self.snapshots_dir, f"{watchlist_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
