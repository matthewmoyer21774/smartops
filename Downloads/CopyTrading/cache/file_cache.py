"""
File-based caching for trader data and exports.
"""

import os
import json
import csv
from typing import List, Optional, Dict
from datetime import datetime

import sys
sys.path.insert(0, '..')
from config import DATA_DIR, TRADERS_DIR, EXPORTS_DIR
from models.trader import Trader


class TraderCache:
    """Persist trader data to disk for incremental processing."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.traders_dir = os.path.join(data_dir, "traders")
        self.exports_dir = os.path.join(data_dir, "exports")

        # Create directories
        for d in [self.data_dir, self.traders_dir, self.exports_dir]:
            os.makedirs(d, exist_ok=True)

    def save_trader(self, trader: Trader):
        """Save trader to JSON file."""
        filepath = os.path.join(self.traders_dir, f"{trader.address}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trader.to_dict(), f, indent=2, default=str)

    def load_trader(self, address: str) -> Optional[Dict]:
        """Load trader from cache."""
        filepath = os.path.join(self.traders_dir, f"{address.lower()}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def trader_exists(self, address: str) -> bool:
        """Check if trader is cached."""
        filepath = os.path.join(self.traders_dir, f"{address.lower()}.json")
        return os.path.exists(filepath)

    def get_cached_trader_addresses(self) -> List[str]:
        """Get list of cached trader addresses."""
        addresses = []
        if not os.path.exists(self.traders_dir):
            return addresses

        for filename in os.listdir(self.traders_dir):
            if filename.endswith('.json'):
                addresses.append(filename[:-5])  # Remove .json
        return addresses

    def load_all_traders(self) -> List[Dict]:
        """Load all cached traders."""
        traders = []
        for address in self.get_cached_trader_addresses():
            data = self.load_trader(address)
            if data:
                traders.append(data)
        return traders

    def save_discovered_traders(self, traders_info: List[Dict]):
        """Save discovered trader info."""
        filepath = os.path.join(self.data_dir, "discovered_traders.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'traders': traders_info,
                'discovered_at': datetime.now().isoformat(),
                'count': len(traders_info)
            }, f, indent=2)

    def load_discovered_traders(self) -> List[Dict]:
        """Load discovered trader info."""
        filepath = os.path.join(self.data_dir, "discovered_traders.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('traders', [])
        return []

    def export_traders_to_csv(
        self,
        traders: List[Trader] = None,
        filename: str = "traders_analysis.csv",
        use_cache: bool = False
    ):
        """
        Export trader analysis to CSV.

        Args:
            traders: List of Trader objects (or None to use cache)
            filename: Output filename
            use_cache: If True and traders is None, load from cache
        """
        if traders is None and use_cache:
            # Load from cache
            traders_data = self.load_all_traders()
        elif traders:
            traders_data = [t.to_dict() for t in traders]
        else:
            traders_data = []

        if not traders_data:
            print("No traders to export")
            return

        filepath = os.path.join(self.exports_dir, filename)

        headers = [
            'address', 'name', 'pseudonym',
            'total_pnl', 'realized_pnl', 'unrealized_pnl',
            'roi_pct', 'win_rate', 'total_trades',
            'unique_markets', 'winning_positions', 'losing_positions',
            'total_resolved', 'total_volume',
            'avg_position_size', 'max_position_size',
            'first_trade_date', 'last_trade_date', 'active_days',
            'last_updated'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for data in traders_data:
                stats = data.get('stats', {})
                row = [
                    data.get('address', ''),
                    data.get('name', ''),
                    data.get('pseudonym', ''),
                    stats.get('total_pnl', 0),
                    stats.get('realized_pnl', 0),
                    stats.get('unrealized_pnl', 0),
                    stats.get('roi_pct', 0),
                    stats.get('win_rate', 0),
                    stats.get('total_trades', 0),
                    stats.get('unique_markets', 0),
                    stats.get('winning_positions', 0),
                    stats.get('losing_positions', 0),
                    stats.get('total_resolved', 0),
                    stats.get('total_volume', 0),
                    stats.get('avg_position_size', 0),
                    stats.get('max_position_size', 0),
                    stats.get('first_trade_date', ''),
                    stats.get('last_trade_date', ''),
                    stats.get('active_days', 0),
                    data.get('last_updated', '')
                ]
                writer.writerow(row)

        print(f"Exported {len(traders_data)} traders to {filepath}")

    def export_filtered_traders(
        self,
        min_trades: int = None,
        min_pnl: float = None,
        min_roi: float = None,
        min_win_rate: float = None,
        sort_by: str = 'total_pnl',
        descending: bool = True,
        top_n: int = None,
        filename: str = "filtered_traders.csv"
    ):
        """
        Export filtered and sorted traders to CSV.

        Args:
            min_trades: Minimum number of trades
            min_pnl: Minimum total P&L
            min_roi: Minimum ROI percentage
            min_win_rate: Minimum win rate percentage
            sort_by: Field to sort by
            descending: Sort descending
            top_n: Limit to top N results
            filename: Output filename
        """
        # Load all cached traders
        traders_data = self.load_all_traders()

        # Apply filters
        filtered = traders_data

        if min_trades:
            filtered = [t for t in filtered if t['stats'].get('total_trades', 0) >= min_trades]

        if min_pnl:
            filtered = [t for t in filtered if t['stats'].get('total_pnl', 0) >= min_pnl]

        if min_roi:
            filtered = [t for t in filtered if t['stats'].get('roi_pct', 0) >= min_roi]

        if min_win_rate:
            filtered = [t for t in filtered if t['stats'].get('win_rate', 0) >= min_win_rate]

        # Sort
        filtered.sort(
            key=lambda t: t['stats'].get(sort_by, 0) or 0,
            reverse=descending
        )

        # Limit
        if top_n:
            filtered = filtered[:top_n]

        # Export
        if not filtered:
            print("No traders match the filters")
            return

        filepath = os.path.join(self.exports_dir, filename)

        headers = [
            'rank', 'address', 'name',
            'total_pnl', 'roi_pct', 'win_rate',
            'total_trades', 'unique_markets',
            'winning_positions', 'losing_positions'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for i, data in enumerate(filtered, 1):
                stats = data.get('stats', {})
                row = [
                    i,
                    data.get('address', ''),
                    data.get('name', '') or data.get('pseudonym', ''),
                    f"${stats.get('total_pnl', 0):,.2f}",
                    f"{stats.get('roi_pct', 0):.1f}%",
                    f"{stats.get('win_rate', 0):.1f}%",
                    stats.get('total_trades', 0),
                    stats.get('unique_markets', 0),
                    stats.get('winning_positions', 0),
                    stats.get('losing_positions', 0)
                ]
                writer.writerow(row)

        print(f"Exported {len(filtered)} filtered traders to {filepath}")
        return filtered

    def get_filtered_traders(
        self,
        min_trades: int = None,
        min_pnl: float = None,
        min_roi: float = None,
        min_win_rate: float = None,
        sort_by: str = 'total_pnl',
        descending: bool = True,
        limit: int = None
    ) -> List[Dict]:
        """
        Filter and return traders as list (for API responses).

        Args:
            min_trades: Minimum number of trades
            min_pnl: Minimum total P&L
            min_roi: Minimum ROI percentage
            min_win_rate: Minimum win rate percentage
            sort_by: Field to sort by
            descending: Sort descending
            limit: Limit to N results

        Returns:
            List of trader dicts matching criteria
        """
        traders_data = self.load_all_traders()

        filtered = traders_data

        if min_trades:
            filtered = [t for t in filtered if t.get('stats', {}).get('total_trades', 0) >= min_trades]

        if min_pnl:
            filtered = [t for t in filtered if t.get('stats', {}).get('total_pnl', 0) >= min_pnl]

        if min_roi:
            filtered = [t for t in filtered if t.get('stats', {}).get('roi_pct', 0) >= min_roi]

        if min_win_rate:
            filtered = [t for t in filtered if t.get('stats', {}).get('win_rate', 0) >= min_win_rate]

        # Sort
        filtered.sort(
            key=lambda t: t.get('stats', {}).get(sort_by, 0) or 0,
            reverse=descending
        )

        # Limit
        if limit:
            filtered = filtered[:limit]

        return filtered

    def get_stats(self) -> Dict:
        """
        Return aggregate statistics about cached data.

        Returns:
            Dict with total_traders, total_pnl, avg_roi, etc.
        """
        traders = self.load_all_traders()

        if not traders:
            return {
                'total_traders': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_roi': 0,
                'avg_win_rate': 0,
                'top_trader_pnl': 0,
                'last_updated': None
            }

        total_pnl = sum(t.get('stats', {}).get('total_pnl', 0) for t in traders)
        roi_values = [t.get('stats', {}).get('roi_pct', 0) for t in traders if t.get('stats', {}).get('roi_pct')]
        win_rate_values = [t.get('stats', {}).get('win_rate', 0) for t in traders if t.get('stats', {}).get('win_rate')]

        # Find most recent update
        last_updated_times = [
            t.get('last_updated') for t in traders
            if t.get('last_updated')
        ]
        last_updated = max(last_updated_times) if last_updated_times else None

        # Top trader by P&L
        top_trader = max(traders, key=lambda t: t.get('stats', {}).get('total_pnl', 0))

        return {
            'total_traders': len(traders),
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(traders) if traders else 0,
            'avg_roi': sum(roi_values) / len(roi_values) if roi_values else 0,
            'avg_win_rate': sum(win_rate_values) / len(win_rate_values) if win_rate_values else 0,
            'top_trader_pnl': top_trader.get('stats', {}).get('total_pnl', 0),
            'top_trader_address': top_trader.get('address'),
            'last_updated': last_updated
        }

    def delete_trader(self, address: str) -> bool:
        """
        Delete trader from cache.

        Args:
            address: Trader wallet address

        Returns:
            True if deleted, False if not found
        """
        filepath = os.path.join(self.traders_dir, f"{address.lower()}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    # ==========================================================================
    # SAVED FILTERS
    # ==========================================================================

    def save_filter(self, filter_data: Dict) -> str:
        """
        Save a custom filter configuration.

        Args:
            filter_data: Dict with name, description, criteria

        Returns:
            Filter ID
        """
        import uuid

        filters_dir = os.path.join(self.data_dir, "filters")
        os.makedirs(filters_dir, exist_ok=True)

        filter_id = filter_data.get('id') or str(uuid.uuid4())
        filter_data['id'] = filter_id
        filter_data['created_at'] = filter_data.get('created_at') or datetime.now().isoformat()

        filepath = os.path.join(filters_dir, f"{filter_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(filter_data, f, indent=2)

        return filter_id

    def load_filters(self) -> List[Dict]:
        """Load all saved custom filters."""
        filters_dir = os.path.join(self.data_dir, "filters")
        filters = []

        if not os.path.exists(filters_dir):
            return filters

        for filename in os.listdir(filters_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(filters_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    filters.append(json.load(f))

        return filters

    def get_filter(self, filter_id: str) -> Optional[Dict]:
        """Load a specific filter by ID."""
        filters_dir = os.path.join(self.data_dir, "filters")
        filepath = os.path.join(filters_dir, f"{filter_id}.json")

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def delete_filter(self, filter_id: str) -> bool:
        """Delete a saved filter."""
        filters_dir = os.path.join(self.data_dir, "filters")
        filepath = os.path.join(filters_dir, f"{filter_id}.json")

        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
