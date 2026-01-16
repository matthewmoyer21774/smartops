"""
Client for Polymarket Gamma API (markets, events).
"""

from typing import List, Dict, Optional
from .base_client import RateLimitedClient

import sys
sys.path.insert(0, '..')
from config import GAMMA_API_BASE


class GammaAPIClient(RateLimitedClient):
    """Client for Polymarket Gamma API."""

    def __init__(self):
        super().__init__(base_url=GAMMA_API_BASE, requests_per_minute=60)

    def get_markets(
        self,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        tag_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get markets with optional filtering.

        Args:
            closed: Include closed markets
            limit: Number of markets per request
            offset: Pagination offset
            tag_id: Filter by tag (category)

        Returns:
            List of market dictionaries
        """
        params = {
            'closed': str(closed).lower(),
            'limit': limit,
            'offset': offset
        }
        if tag_id:
            params['tag_id'] = tag_id

        result = self.get('/markets', params=params)
        return result if isinstance(result, list) else []

    def get_all_open_markets(self, max_markets: int = 500, include_closed: bool = False) -> List[Dict]:
        """
        Fetch all markets with pagination.

        Args:
            max_markets: Maximum markets to fetch
            include_closed: Include closed/resolved markets

        Returns:
            List of markets sorted by volume
        """
        all_markets = []
        offset = 0
        limit = 100

        while len(all_markets) < max_markets:
            markets = self.get_markets(closed=include_closed, limit=limit, offset=offset)
            if not markets:
                break
            all_markets.extend(markets)
            if len(markets) < limit:
                break
            offset += limit

        # Sort by volume descending
        all_markets.sort(key=lambda m: float(m.get('volume', 0) or 0), reverse=True)
        return all_markets[:max_markets]

    def get_market_by_slug(self, slug: str) -> Optional[Dict]:
        """
        Get single market by slug.

        Args:
            slug: Market slug identifier

        Returns:
            Market dict or None
        """
        return self.get(f'/markets/{slug}')

    def get_market_by_condition_id(self, condition_id: str) -> Optional[Dict]:
        """
        Get market by condition ID.

        Args:
            condition_id: Market condition ID

        Returns:
            Market dict or None
        """
        result = self.get('/markets', params={'condition_id': condition_id})
        if isinstance(result, list) and result:
            return result[0]
        return None

    def get_events(
        self,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get events (which contain markets).

        Args:
            closed: Include closed events
            limit: Number per request
            offset: Pagination offset

        Returns:
            List of event dictionaries
        """
        params = {
            'closed': str(closed).lower(),
            'limit': limit,
            'offset': offset
        }
        result = self.get('/events', params=params)
        return result if isinstance(result, list) else []
