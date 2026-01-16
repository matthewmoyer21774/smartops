"""
Client for Polymarket Data API (trades, positions, activity, leaderboard).
"""

from typing import List, Dict, Optional, Any
from .base_client import RateLimitedClient

import sys
sys.path.insert(0, '..')
from config import DATA_API_BASE


class DataAPIClient(RateLimitedClient):
    """Client for Polymarket Data API."""

    def __init__(self):
        super().__init__(base_url=DATA_API_BASE, requests_per_minute=30)

    def get_leaderboard(
        self,
        time_period: str = "all",
        limit: int = 100
    ) -> List[Dict]:
        """
        Get trader leaderboard rankings.

        Args:
            time_period: 'all', 'daily', 'weekly', 'monthly'
            limit: Number of traders to fetch

        Returns:
            List of leaderboard entries with address, pnl, volume, etc.
        """
        params = {
            'window': time_period,
            'limit': limit
        }
        # Use /v1/leaderboard endpoint
        result = self.get('/v1/leaderboard', params=params)

        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return result.get('leaderboard', result.get('data', []))
        return []

    def get_trader_trades(
        self,
        address: str,
        limit: int = 50000
    ) -> List[Dict]:
        """
        Get all trades for a specific trader.

        Args:
            address: Wallet address
            limit: Max trades to fetch (default 50000)

        Returns:
            List of trade records
        """
        all_trades = []
        offset = 0
        page_size = 500  # API returns max 500 per request

        while len(all_trades) < limit:
            params = {
                'user': address.lower(),
                'limit': page_size,
                'offset': offset
            }

            result = self.get('/trades', params=params)

            if not result:
                break

            trades = result if isinstance(result, list) else result.get('data', [])

            if not trades:
                break

            all_trades.extend(trades)

            if len(trades) < page_size:
                break

            offset += page_size

        return all_trades[:limit]

    def get_trader_positions(self, address: str) -> List[Dict]:
        """
        Get current open positions for a trader.

        Args:
            address: Wallet address

        Returns:
            List of position records
        """
        result = self.get('/positions', params={'user': address.lower()})

        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return result.get('positions', result.get('data', []))
        return []

    def get_market_trades(
        self,
        condition_id: str,
        limit: int = 500
    ) -> List[Dict]:
        """
        Get trades for a specific market.

        Args:
            condition_id: Market condition ID
            limit: Max trades to fetch

        Returns:
            List of trades in this market
        """
        params = {
            'market': condition_id,
            'limit': limit
        }
        result = self.get('/trades', params=params)

        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return result.get('data', [])
        return []

    def get_trader_activity(
        self,
        address: str,
        limit: int = 500
    ) -> List[Dict]:
        """
        Get activity history for a trader.

        Args:
            address: Wallet address
            limit: Max activities to fetch

        Returns:
            List of activity records (trades, redemptions, etc.)
        """
        params = {
            'user': address.lower(),
            'limit': limit
        }
        result = self.get('/activity', params=params)

        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return result.get('data', [])
        return []
