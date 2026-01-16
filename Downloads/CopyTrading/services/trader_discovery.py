"""
Service for discovering profitable traders from Polymarket.
"""

from typing import List, Set, Dict
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, '..')
from api.gamma_api import GammaAPIClient
from api.data_api import DataAPIClient
from config import EXCLUDED_WALLETS, DEFAULT_MARKETS_TO_SCAN, DEFAULT_HOLDERS_PER_MARKET, MARKET_CATEGORIES


class TraderDiscoveryService:
    """Discover profitable traders from markets and leaderboard."""

    def __init__(self):
        self.gamma_client = GammaAPIClient()
        self.data_client = DataAPIClient()

    def discover_from_leaderboard(
        self,
        time_period: str = "all",
        limit: int = 100
    ) -> List[Dict]:
        """
        Get top traders from official leaderboard.

        Args:
            time_period: 'all', 'daily', 'weekly', 'monthly'
            limit: Number of traders to fetch

        Returns:
            List of trader info dicts with address, pnl, etc.
        """
        print(f"Fetching leaderboard ({time_period}, limit={limit})...")
        leaderboard = self.data_client.get_leaderboard(
            time_period=time_period,
            limit=limit
        )

        traders = []
        for entry in leaderboard:
            address = entry.get('proxyWallet') or entry.get('address') or entry.get('user')
            if not address:
                continue

            address = address.lower()
            if address in EXCLUDED_WALLETS:
                continue

            traders.append({
                'address': address,
                'name': entry.get('userName'),
                'pseudonym': entry.get('xUsername') or entry.get('pseudonym'),
                'pnl': float(entry.get('pnl', 0) or entry.get('profit', 0) or 0),
                'volume': float(entry.get('vol', 0) or entry.get('volume', 0) or 0),
                'rank': entry.get('rank'),
                'source': 'leaderboard'
            })

        print(f"Found {len(traders)} traders from leaderboard")
        return traders

    def discover_from_markets(
        self,
        num_markets: int = DEFAULT_MARKETS_TO_SCAN,
        holders_per_market: int = DEFAULT_HOLDERS_PER_MARKET
    ) -> List[Dict]:
        """
        Discover traders by scanning top holders in open markets.

        Strategy: Get trades for each market, find traders with highest volume.

        Args:
            num_markets: Number of markets to scan
            holders_per_market: Top N traders per market

        Returns:
            List of trader info dicts with address and source markets
        """
        print(f"Fetching top {num_markets} open markets...")
        markets = self.gamma_client.get_all_open_markets(max_markets=num_markets)
        print(f"Got {len(markets)} markets")

        # Track traders across markets
        trader_data: Dict[str, Dict] = {}

        for i, market in enumerate(markets):
            slug = market.get('slug', '')
            volume = float(market.get('volume', 0) or 0)
            condition_id = market.get('conditionId')

            if not condition_id:
                continue

            print(f"[{i+1}/{len(markets)}] {slug[:50]}... (${volume:,.0f})")

            # Get trades to find active traders
            trades = self.data_client.get_market_trades(condition_id, limit=500)

            if not trades:
                continue

            # Aggregate volume by trader
            trader_volumes: Dict[str, float] = defaultdict(float)
            trader_info: Dict[str, Dict] = {}

            for trade in trades:
                address = (trade.get('proxyWallet') or trade.get('user') or '').lower()
                if not address or address in EXCLUDED_WALLETS:
                    continue

                trade_volume = float(trade.get('size', 0) or 0)
                trader_volumes[address] += trade_volume

                if address not in trader_info:
                    trader_info[address] = {
                        'name': trade.get('name'),
                        'pseudonym': trade.get('pseudonym')
                    }

            # Get top traders by volume in this market
            top_traders = sorted(
                trader_volumes.items(),
                key=lambda x: x[1],
                reverse=True
            )[:holders_per_market]

            for address, vol in top_traders:
                if address not in trader_data:
                    trader_data[address] = {
                        'address': address,
                        'name': trader_info[address].get('name'),
                        'pseudonym': trader_info[address].get('pseudonym'),
                        'total_volume': 0.0,
                        'markets_found_in': [],
                        'source': 'market_scan'
                    }

                trader_data[address]['total_volume'] += vol
                trader_data[address]['markets_found_in'].append(slug)

        traders = list(trader_data.values())
        # Sort by number of markets found in (active across more markets = more interesting)
        traders.sort(key=lambda t: len(t['markets_found_in']), reverse=True)

        print(f"\nDiscovered {len(traders)} unique traders from {len(markets)} markets")
        return traders

    def discover_specific_addresses(self, addresses: List[str]) -> List[Dict]:
        """
        Create trader entries for specific known addresses.

        Args:
            addresses: List of wallet addresses

        Returns:
            List of trader info dicts
        """
        traders = []
        for addr in addresses:
            address = addr.lower().strip()
            if address and address not in EXCLUDED_WALLETS:
                traders.append({
                    'address': address,
                    'name': None,
                    'pseudonym': None,
                    'source': 'manual'
                })
        return traders

    def run_discovery(
        self,
        use_leaderboard: bool = True,
        use_market_scan: bool = True,
        num_markets: int = DEFAULT_MARKETS_TO_SCAN,
        holders_per_market: int = DEFAULT_HOLDERS_PER_MARKET,
        additional_addresses: List[str] = None
    ) -> List[Dict]:
        """
        Run full discovery process.

        Args:
            use_leaderboard: Include leaderboard traders
            use_market_scan: Scan markets for top holders
            num_markets: Markets to scan
            holders_per_market: Traders per market
            additional_addresses: Manual addresses to include

        Returns:
            Deduplicated list of trader info dicts
        """
        all_traders: Dict[str, Dict] = {}

        if use_leaderboard:
            leaderboard_traders = self.discover_from_leaderboard()
            for t in leaderboard_traders:
                addr = t['address']
                if addr not in all_traders:
                    all_traders[addr] = t
                else:
                    # Merge info
                    all_traders[addr]['pnl'] = t.get('pnl', 0)
                    all_traders[addr]['rank'] = t.get('rank')

        if use_market_scan:
            market_traders = self.discover_from_markets(
                num_markets=num_markets,
                holders_per_market=holders_per_market
            )
            for t in market_traders:
                addr = t['address']
                if addr not in all_traders:
                    all_traders[addr] = t
                else:
                    # Merge market info
                    existing = all_traders[addr]
                    existing['markets_found_in'] = t.get('markets_found_in', [])
                    existing['total_volume'] = existing.get('total_volume', 0) + t.get('total_volume', 0)

        if additional_addresses:
            manual_traders = self.discover_specific_addresses(additional_addresses)
            for t in manual_traders:
                addr = t['address']
                if addr not in all_traders:
                    all_traders[addr] = t

        traders = list(all_traders.values())
        print(f"\n{'='*60}")
        print(f"DISCOVERY COMPLETE: {len(traders)} unique traders")
        print(f"{'='*60}")

        return traders

    def _matches_category(self, market: Dict, category: str, search_pattern: str = None) -> bool:
        """
        Check if a market matches the given category or search pattern.

        Args:
            market: Market dict from API
            category: Category key from MARKET_CATEGORIES
            search_pattern: Optional additional keyword filter

        Returns:
            True if market matches
        """
        slug = (market.get('slug') or '').lower()
        title = (market.get('title') or market.get('question') or '').lower()
        text = f"{slug} {title}"

        # Check category keywords
        if category and category in MARKET_CATEGORIES:
            keywords = MARKET_CATEGORIES[category]
            if not any(kw in text for kw in keywords):
                return False

        # Check additional search pattern
        if search_pattern:
            if search_pattern.lower() not in text:
                return False

        return True

    def discover_from_category(
        self,
        category: str = None,
        search_pattern: str = None,
        num_markets: int = 50,
        traders_per_market: int = 100,
        include_closed: bool = False,
        min_volume: float = 0
    ) -> Dict:
        """
        Discover traders from markets matching a category.

        Scans markets for the category and extracts all traders with their
        positions (which side they're betting on).

        Args:
            category: Category key (nba, nfl, crypto, etc.) - see config.MARKET_CATEGORIES
            search_pattern: Additional keyword to filter markets
            num_markets: Max markets to scan
            traders_per_market: Max traders per market
            include_closed: Include closed/resolved markets
            min_volume: Minimum market volume to include

        Returns:
            Dict with:
                - markets: List of scanned markets with trader positions
                - traders: Aggregated trader data
                - category: Category scanned
        """
        print(f"{'='*60}")
        print(f"CATEGORY SCAN: {category or 'all'}")
        if search_pattern:
            print(f"Search filter: '{search_pattern}'")
        print(f"{'='*60}")

        # Fetch markets
        print(f"\nFetching markets (max {num_markets * 3} to filter)...")
        all_markets = self.gamma_client.get_all_open_markets(
            max_markets=num_markets * 3,  # Fetch more to filter
            include_closed=include_closed
        )

        # Filter by category
        matching_markets = []
        for market in all_markets:
            if self._matches_category(market, category, search_pattern):
                vol = float(market.get('volume', 0) or 0)
                if vol >= min_volume:
                    matching_markets.append(market)

            if len(matching_markets) >= num_markets:
                break

        print(f"Found {len(matching_markets)} matching markets")

        if not matching_markets:
            return {
                'markets': [],
                'traders': {},
                'traders_list': [],
                'category': category,
                'search_pattern': search_pattern
            }

        # Scan each market for traders and their positions
        market_results = []
        trader_data: Dict[str, Dict] = {}

        for i, market in enumerate(matching_markets):
            slug = market.get('slug', '')
            title = market.get('title', market.get('question', ''))
            condition_id = market.get('conditionId')
            volume = float(market.get('volume', 0) or 0)
            is_closed = market.get('closed', False)

            if not condition_id:
                continue

            print(f"[{i+1}/{len(matching_markets)}] {slug[:50]}... (${volume:,.0f})")

            # Get trades for this market
            trades = self.data_client.get_market_trades(condition_id, limit=1000)

            if not trades:
                continue

            # Track traders and their positions in this market
            market_traders: Dict[str, Dict] = {}

            for trade in trades:
                address = (trade.get('proxyWallet') or trade.get('user') or '').lower()
                if not address or address in EXCLUDED_WALLETS:
                    continue

                # Trade details
                side = trade.get('side', 'BUY').upper()
                outcome = trade.get('outcome', 'Yes')
                size = float(trade.get('size', 0) or 0)
                price = float(trade.get('price', 0) or 0)

                if address not in market_traders:
                    market_traders[address] = {
                        'address': address,
                        'name': trade.get('name'),
                        'pseudonym': trade.get('pseudonym'),
                        'total_volume': 0.0,
                        'positions': {},  # outcome -> net_shares
                        'trades': []
                    }

                mt = market_traders[address]
                mt['total_volume'] += size

                # Track position by outcome
                shares = size / price if price > 0 else 0
                if outcome not in mt['positions']:
                    mt['positions'][outcome] = {'shares': 0, 'cost': 0}

                if side == 'BUY':
                    mt['positions'][outcome]['shares'] += shares
                    mt['positions'][outcome]['cost'] += size
                else:
                    mt['positions'][outcome]['shares'] -= shares
                    mt['positions'][outcome]['cost'] -= size

                mt['trades'].append({
                    'side': side,
                    'outcome': outcome,
                    'price': price,
                    'size': size
                })

            # Get top traders by volume in this market
            sorted_traders = sorted(
                market_traders.values(),
                key=lambda x: x['total_volume'],
                reverse=True
            )[:traders_per_market]

            # Determine dominant position for each trader
            for t in sorted_traders:
                dominant_outcome = None
                max_shares = 0
                for outcome, pos in t['positions'].items():
                    if pos['shares'] > max_shares:
                        max_shares = pos['shares']
                        dominant_outcome = outcome
                t['dominant_position'] = dominant_outcome
                t['dominant_shares'] = max_shares

            market_result = {
                'slug': slug,
                'title': title,
                'condition_id': condition_id,
                'volume': volume,
                'is_closed': is_closed,
                'traders': sorted_traders,
                'trader_count': len(sorted_traders)
            }
            market_results.append(market_result)

            # Aggregate trader data across markets
            for t in sorted_traders:
                addr = t['address']
                if addr not in trader_data:
                    trader_data[addr] = {
                        'address': addr,
                        'name': t.get('name'),
                        'pseudonym': t.get('pseudonym'),
                        'total_volume': 0.0,
                        'markets': [],
                        'positions_by_market': {},
                        'source': f'category_scan:{category}'
                    }

                trader_data[addr]['total_volume'] += t['total_volume']
                trader_data[addr]['markets'].append(slug)
                trader_data[addr]['positions_by_market'][slug] = {
                    'dominant_position': t['dominant_position'],
                    'positions': t['positions']
                }

        # Sort traders by number of markets
        traders_list = list(trader_data.values())
        traders_list.sort(key=lambda x: len(x['markets']), reverse=True)

        print(f"\n{'='*60}")
        print(f"SCAN COMPLETE")
        print(f"{'='*60}")
        print(f"Markets scanned: {len(market_results)}")
        print(f"Unique traders:  {len(traders_list)}")

        return {
            'markets': market_results,
            'traders': {t['address']: t for t in traders_list},
            'traders_list': traders_list,
            'category': category,
            'search_pattern': search_pattern
        }
