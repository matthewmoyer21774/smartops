"""
Service for analyzing trader correlations and detecting smart money clusters.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

import sys
sys.path.insert(0, '..')
from services.trader_discovery import TraderDiscoveryService
from services.trader_analyzer import TraderAnalyzerService
from cache.file_cache import TraderCache


@dataclass
class TraderCluster:
    """Group of correlated traders."""
    traders: List[str]
    trader_names: Dict[str, str]  # address -> name/pseudonym
    shared_markets: List[str]
    avg_position_alignment: float  # 0-1 (0 = opposite sides, 1 = same side)
    total_volume: float

    def __repr__(self):
        return f"TraderCluster({len(self.traders)} traders, {len(self.shared_markets)} shared markets)"

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON export."""
        return {
            'traders': self.traders,
            'trader_names': self.trader_names,
            'shared_markets': self.shared_markets,
            'avg_position_alignment': self.avg_position_alignment,
            'total_volume': self.total_volume,
            'trader_count': len(self.traders),
            'shared_market_count': len(self.shared_markets)
        }


@dataclass
class MarketConsensus:
    """Smart money consensus on a market."""
    market_slug: str
    market_title: str
    traders: List[str]
    trader_names: Dict[str, str]
    positions: Dict[str, List[str]]  # outcome -> list of trader addresses
    consensus_side: str  # 'Yes', 'No', or 'Split'
    consensus_strength: float  # 0-1 (what % of traders agree)
    total_volume: float

    def __repr__(self):
        return f"MarketConsensus({self.market_slug}: {self.consensus_side} @ {self.consensus_strength:.0%})"

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON export."""
        return {
            'market_slug': self.market_slug,
            'market_title': self.market_title,
            'traders': self.traders,
            'trader_names': self.trader_names,
            'positions': self.positions,
            'consensus_side': self.consensus_side,
            'consensus_strength': self.consensus_strength,
            'total_volume': self.total_volume,
            'trader_count': len(self.traders)
        }


class CorrelationAnalyzer:
    """Analyze trader correlations and clustering."""

    def __init__(self):
        self.discovery_service = TraderDiscoveryService()
        self.analyzer_service = TraderAnalyzerService()
        self.cache = TraderCache()

    def analyze_from_scan(
        self,
        category: str = None,
        search_pattern: str = None,
        num_markets: int = 20,
        traders_per_market: int = 100,
        min_shared_markets: int = 2,
        include_closed: bool = False,
        # Smart money filters
        min_pnl: float = 0,
        min_roi: float = 0,
        min_win_rate: float = 0,
        min_trades: int = 0,
        analyze_limit: int = 50
    ) -> Dict:
        """
        Run category scan and analyze correlations.

        Args:
            category: Market category (nba, nfl, crypto, etc.)
            search_pattern: Additional keyword filter
            num_markets: Max markets to scan
            traders_per_market: Max traders per market
            min_shared_markets: Min markets a trader must appear in to be included
            include_closed: Include closed/resolved markets
            min_pnl: Minimum P&L to include trader (smart money filter)
            min_roi: Minimum ROI % to include trader
            min_win_rate: Minimum win rate % to include trader
            min_trades: Minimum resolved trades to include trader
            analyze_limit: Max traders to analyze for performance (default 50)

        Returns:
            Dict with scan results, correlations, and consensus data
        """
        # Run the category scan
        scan_results = self.discovery_service.discover_from_category(
            category=category,
            search_pattern=search_pattern,
            num_markets=num_markets,
            traders_per_market=traders_per_market,
            include_closed=include_closed
        )

        if not scan_results['markets']:
            return {
                'scan': scan_results,
                'clusters': [],
                'market_consensus': [],
                'correlation_matrix': {},
                'smart_money_traders': [],
                'filters_applied': {}
            }

        # Check if we need to filter by performance
        needs_performance_filter = any([min_pnl > 0, min_roi > 0, min_win_rate > 0, min_trades > 0])

        smart_money_traders = []
        all_analyzed_traders = []

        if needs_performance_filter:
            # Analyze top traders by volume to get their stats
            print(f"\n{'='*60}")
            print("ANALYZING TRADER PERFORMANCE")
            print("=" * 60)
            print(f"Analyzing top {analyze_limit} traders by volume...")

            analyzed = self.analyze_top_traders(
                scan_results['traders_list'][:analyze_limit]
            )
            all_analyzed_traders = analyzed

            # Filter by performance criteria
            for t in analyzed:
                stats = t.get('stats', {})
                pnl = stats.get('total_pnl', 0)
                roi = stats.get('roi_pct', 0)
                win_rate = stats.get('win_rate', 0)
                trades = stats.get('total_resolved', 0)

                if pnl >= min_pnl and roi >= min_roi and win_rate >= min_win_rate and trades >= min_trades:
                    smart_money_traders.append(t)

            print(f"\nTraders analyzed: {len(analyzed)}")
            print(f"Matching criteria: {len(smart_money_traders)}")

            # Update scan_results to only include smart money traders in market data
            if smart_money_traders:
                smart_addresses = {t['address'] for t in smart_money_traders}
                scan_results = self._filter_scan_to_traders(scan_results, smart_addresses)

        # Analyze correlations
        clusters = self.find_overlapping_traders(
            scan_results,
            min_shared_markets=min_shared_markets
        )

        # Find market consensus (now only among filtered traders if smart money mode)
        consensus = self.find_market_consensus(scan_results)

        # Build correlation matrix for top traders
        if smart_money_traders:
            top_traders = [t['address'] for t in smart_money_traders[:30]]
        else:
            top_traders = [t['address'] for t in scan_results['traders_list'][:30]]
        correlation_matrix = self.build_correlation_matrix(scan_results, top_traders)

        return {
            'scan': scan_results,
            'clusters': clusters,
            'market_consensus': consensus,
            'correlation_matrix': correlation_matrix,
            'smart_money_traders': smart_money_traders,
            'all_analyzed_traders': all_analyzed_traders,
            'filters_applied': {
                'min_pnl': min_pnl,
                'min_roi': min_roi,
                'min_win_rate': min_win_rate,
                'min_trades': min_trades
            } if needs_performance_filter else {}
        }

    def analyze_top_traders(self, traders_list: List[Dict]) -> List[Dict]:
        """
        Analyze performance for a list of traders.

        Checks cache first, then fetches from API if needed.

        Args:
            traders_list: List of trader dicts from scan

        Returns:
            List of trader dicts with stats attached
        """
        analyzed = []

        for i, t in enumerate(traders_list):
            addr = t['address']
            print(f"  [{i+1}/{len(traders_list)}] {addr[:16]}...", end=' ')

            # Check cache first
            cached = self.cache.load_trader(addr)
            if cached and 'stats' in cached:
                stats = cached['stats']
                print(f"[CACHED] P&L: ${stats.get('total_pnl', 0):,.0f} | "
                      f"ROI: {stats.get('roi_pct', 0):.1f}% | "
                      f"Win: {stats.get('win_rate', 0):.1f}%")
                analyzed.append({
                    'address': addr,
                    'name': t.get('name'),
                    'pseudonym': t.get('pseudonym'),
                    'stats': stats,
                    'markets': t.get('markets', []),
                    'positions_by_market': t.get('positions_by_market', {})
                })
                continue

            # Analyze fresh
            try:
                trader = self.analyzer_service.analyze_trader(
                    address=addr,
                    name=t.get('name'),
                    pseudonym=t.get('pseudonym')
                )
                stats = {
                    'total_pnl': trader.stats.total_pnl,
                    'realized_pnl': trader.stats.realized_pnl,
                    'unrealized_pnl': trader.stats.unrealized_pnl,
                    'roi_pct': trader.stats.roi_pct,
                    'win_rate': trader.stats.win_rate,
                    'total_trades': trader.stats.total_trades,
                    'total_resolved': trader.stats.total_resolved,
                    'total_volume': trader.stats.total_volume
                }
                print(f"P&L: ${stats['total_pnl']:,.0f} | "
                      f"ROI: {stats['roi_pct']:.1f}% | "
                      f"Win: {stats['win_rate']:.1f}%")

                # Cache for future
                self.cache.save_trader(trader)

                analyzed.append({
                    'address': addr,
                    'name': t.get('name'),
                    'pseudonym': t.get('pseudonym'),
                    'stats': stats,
                    'markets': t.get('markets', []),
                    'positions_by_market': t.get('positions_by_market', {})
                })
            except Exception as e:
                print(f"Error: {e}")

        return analyzed

    def _filter_scan_to_traders(self, scan_results: Dict, addresses: set) -> Dict:
        """
        Filter scan results to only include specific trader addresses.

        Args:
            scan_results: Original scan results
            addresses: Set of addresses to keep

        Returns:
            Filtered scan results
        """
        # Filter traders dict
        filtered_traders = {
            addr: data for addr, data in scan_results['traders'].items()
            if addr in addresses
        }

        # Filter traders_list
        filtered_traders_list = [
            t for t in scan_results['traders_list']
            if t['address'] in addresses
        ]

        # Filter traders in each market
        filtered_markets = []
        for market in scan_results['markets']:
            filtered_market = market.copy()
            filtered_market['traders'] = [
                t for t in market.get('traders', [])
                if t['address'] in addresses
            ]
            filtered_market['trader_count'] = len(filtered_market['traders'])
            filtered_markets.append(filtered_market)

        return {
            'markets': filtered_markets,
            'traders': filtered_traders,
            'traders_list': filtered_traders_list,
            'category': scan_results.get('category'),
            'search_pattern': scan_results.get('search_pattern')
        }

    def find_overlapping_traders(
        self,
        scan_results: Dict,
        min_shared_markets: int = 2
    ) -> List[TraderCluster]:
        """
        Find groups of traders who trade in the same markets.

        Args:
            scan_results: Results from discover_from_category()
            min_shared_markets: Minimum shared markets to form a cluster

        Returns:
            List of TraderCluster objects
        """
        traders_data = scan_results['traders']

        # Build trader -> markets mapping
        trader_markets = {}
        trader_names = {}
        for addr, data in traders_data.items():
            trader_markets[addr] = set(data.get('markets', []))
            trader_names[addr] = data.get('name') or data.get('pseudonym') or ''

        # Find overlapping pairs
        addresses = list(trader_markets.keys())
        overlap_graph = defaultdict(set)  # trader -> set of overlapping traders

        for i, addr1 in enumerate(addresses):
            markets1 = trader_markets[addr1]
            for addr2 in addresses[i+1:]:
                markets2 = trader_markets[addr2]
                shared = markets1 & markets2
                if len(shared) >= min_shared_markets:
                    overlap_graph[addr1].add(addr2)
                    overlap_graph[addr2].add(addr1)

        # Find connected components (clusters)
        visited = set()
        clusters = []

        for addr in addresses:
            if addr in visited or addr not in overlap_graph:
                continue

            # BFS to find cluster
            cluster_traders = set()
            queue = [addr]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster_traders.add(current)
                for neighbor in overlap_graph[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(cluster_traders) >= 2:
                # Find markets that multiple traders in cluster are in
                # Count how many cluster members are in each market
                market_counts = defaultdict(int)
                for trader in cluster_traders:
                    for market in trader_markets[trader]:
                        market_counts[market] += 1

                # Get markets with at least 2 cluster members
                shared_markets = [
                    m for m, count in market_counts.items()
                    if count >= min(len(cluster_traders), 3)
                ]
                shared_markets.sort(key=lambda m: market_counts[m], reverse=True)

                # Calculate position alignment
                alignment = self._calculate_position_alignment(
                    cluster_traders,
                    list(shared_markets) if shared_markets else [],
                    scan_results
                )

                # Calculate total volume
                total_vol = sum(
                    traders_data[t]['total_volume']
                    for t in cluster_traders if t in traders_data
                )

                clusters.append(TraderCluster(
                    traders=list(cluster_traders),
                    trader_names={t: trader_names[t] for t in cluster_traders},
                    shared_markets=shared_markets,
                    avg_position_alignment=alignment,
                    total_volume=total_vol
                ))

        # Sort clusters by size
        clusters.sort(key=lambda c: len(c.traders), reverse=True)
        return clusters

    def _calculate_position_alignment(
        self,
        traders: set,
        shared_markets: List[str],
        scan_results: Dict
    ) -> float:
        """
        Calculate how aligned traders are in their positions.

        Returns 0-1 where 1 means all traders on same side.
        """
        if not shared_markets:
            return 0.0

        traders_data = scan_results['traders']
        alignments = []

        for market_slug in shared_markets:
            positions = defaultdict(list)  # outcome -> traders

            for trader in traders:
                if trader not in traders_data:
                    continue
                pos_data = traders_data[trader].get('positions_by_market', {})
                if market_slug not in pos_data:
                    continue

                dominant = pos_data[market_slug].get('dominant_position')
                if dominant:
                    positions[dominant].append(trader)

            if positions:
                # Find the most common position
                max_count = max(len(v) for v in positions.values())
                total_traders = sum(len(v) for v in positions.values())
                if total_traders > 0:
                    alignments.append(max_count / total_traders)

        return sum(alignments) / len(alignments) if alignments else 0.0

    def find_market_consensus(self, scan_results: Dict) -> List[MarketConsensus]:
        """
        Find market consensus (what side are most traders betting).

        Args:
            scan_results: Results from discover_from_category()

        Returns:
            List of MarketConsensus objects sorted by number of traders
        """
        markets = scan_results['markets']
        consensus_list = []

        for market in markets:
            slug = market['slug']
            title = market.get('title', slug)
            traders_in_market = market.get('traders', [])

            if not traders_in_market:
                continue

            # Group by position
            positions = defaultdict(list)
            trader_names = {}
            total_vol = 0

            for t in traders_in_market:
                addr = t['address']
                pos = t.get('dominant_position')
                if pos:
                    positions[pos].append(addr)
                trader_names[addr] = t.get('name') or t.get('pseudonym') or ''
                total_vol += t.get('total_volume', 0)

            if not positions:
                continue

            # Determine consensus
            max_outcome = max(positions.keys(), key=lambda k: len(positions[k]))
            max_count = len(positions[max_outcome])
            total_traders = sum(len(v) for v in positions.values())

            # Check if split (no clear majority)
            if total_traders > 1 and max_count / total_traders < 0.6:
                consensus_side = 'Split'
                strength = max_count / total_traders
            else:
                consensus_side = max_outcome
                strength = max_count / total_traders if total_traders > 0 else 0

            consensus_list.append(MarketConsensus(
                market_slug=slug,
                market_title=title,
                traders=list(trader_names.keys()),
                trader_names=trader_names,
                positions=dict(positions),
                consensus_side=consensus_side,
                consensus_strength=strength,
                total_volume=total_vol
            ))

        # Sort by number of traders (more traders = more signal)
        consensus_list.sort(key=lambda c: len(c.traders), reverse=True)
        return consensus_list

    def build_correlation_matrix(
        self,
        scan_results: Dict,
        traders: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build trader-to-trader correlation matrix based on shared markets.

        Args:
            scan_results: Results from discover_from_category()
            traders: List of trader addresses to include

        Returns:
            Dict of {trader1: {trader2: correlation, ...}, ...}
        """
        traders_data = scan_results['traders']

        # Get markets for each trader
        trader_markets = {}
        for addr in traders:
            if addr in traders_data:
                trader_markets[addr] = set(traders_data[addr].get('markets', []))
            else:
                trader_markets[addr] = set()

        # Calculate pairwise correlation (Jaccard similarity)
        matrix = {}
        for addr1 in traders:
            matrix[addr1] = {}
            markets1 = trader_markets.get(addr1, set())

            for addr2 in traders:
                if addr1 == addr2:
                    matrix[addr1][addr2] = 1.0
                else:
                    markets2 = trader_markets.get(addr2, set())
                    intersection = len(markets1 & markets2)
                    union = len(markets1 | markets2)
                    similarity = intersection / union if union > 0 else 0.0
                    matrix[addr1][addr2] = round(similarity, 3)

        return matrix

    def print_report(
        self,
        results: Dict,
        show_matrix: bool = False,
        top_markets: int = 10,
        top_clusters: int = 5
    ):
        """Print formatted correlation analysis report."""
        scan = results['scan']
        clusters = results['clusters']
        consensus = results['market_consensus']
        matrix = results['correlation_matrix']
        smart_money = results.get('smart_money_traders', [])
        filters = results.get('filters_applied', {})

        category = scan.get('category', 'all')
        search = scan.get('search_pattern')

        # Check if smart money mode
        is_smart_money = len(filters) > 0

        print("\n" + "=" * 70)
        if is_smart_money:
            print(f"SMART MONEY CONSENSUS: {category.upper() if category else 'ALL'} Markets")
        else:
            print(f"TRADER CORRELATION ANALYSIS: {category.upper() if category else 'ALL'} Markets")
        if search:
            print(f"Search filter: '{search}'")
        print("=" * 70)

        # Show filters if applied
        if filters:
            print("\nFilters applied:")
            filter_parts = []
            if filters.get('min_pnl', 0) > 0:
                filter_parts.append(f"Min P&L: ${filters['min_pnl']:,.0f}")
            if filters.get('min_roi', 0) > 0:
                filter_parts.append(f"Min ROI: {filters['min_roi']}%")
            if filters.get('min_win_rate', 0) > 0:
                filter_parts.append(f"Min Win Rate: {filters['min_win_rate']}%")
            if filters.get('min_trades', 0) > 0:
                filter_parts.append(f"Min Trades: {filters['min_trades']}")
            print("  " + " | ".join(filter_parts))

        print(f"\nMarkets scanned: {len(scan['markets'])}")
        print(f"Unique traders: {len(scan['traders'])}")
        if smart_money:
            print(f"Profitable traders matching criteria: {len(smart_money)}")
        print(f"Clusters found: {len(clusters)}")

        # Show top profitable traders if in smart money mode
        if smart_money:
            print(f"\n{'='*70}")
            print("TOP PROFITABLE TRADERS IN CATEGORY")
            print("=" * 70)

            # Sort by P&L
            sorted_traders = sorted(smart_money, key=lambda t: t['stats'].get('total_pnl', 0), reverse=True)

            for i, t in enumerate(sorted_traders[:15], 1):
                addr = t['address'][:16]
                name = t.get('name') or t.get('pseudonym') or ''
                stats = t['stats']
                pnl = stats.get('total_pnl', 0)
                roi = stats.get('roi_pct', 0)
                win = stats.get('win_rate', 0)
                trades = stats.get('total_resolved', 0)

                if name:
                    display = f"{addr}... {name[:12]}"
                else:
                    display = f"{addr}..."

                print(f"{i:2}. {display:<30} | P&L: ${pnl:>10,.0f} | ROI: {roi:>5.1f}% | Win: {win:>5.1f}% | {trades} resolved")

        # Show clusters
        if clusters:
            print(f"\n{'='*70}")
            print("TRADER CLUSTERS (groups trading same markets)")
            print("=" * 70)

            for i, cluster in enumerate(clusters[:top_clusters], 1):
                print(f"\n--- Cluster {i}: {len(cluster.traders)} traders ---")
                print(f"Shared markets: {len(cluster.shared_markets)}")
                print(f"Position alignment: {cluster.avg_position_alignment:.0%}")
                print(f"Combined volume: ${cluster.total_volume:,.0f}")

                print("\nTraders:")
                for addr in cluster.traders[:10]:
                    name = cluster.trader_names.get(addr, '')
                    display = f"{name} ({addr[:12]}...)" if name else f"{addr[:16]}..."
                    print(f"  - {display}")

                if len(cluster.traders) > 10:
                    print(f"  ... and {len(cluster.traders) - 10} more")

                print("\nShared markets:")
                for slug in cluster.shared_markets[:5]:
                    print(f"  - {slug[:50]}")
                if len(cluster.shared_markets) > 5:
                    print(f"  ... and {len(cluster.shared_markets) - 5} more")

        # Show market consensus
        if consensus:
            print(f"\n{'='*70}")
            print("MARKET CONSENSUS (what side are traders betting)")
            print("=" * 70)

            print(f"\n{'Market':<45} | {'Traders':>7} | {'Consensus':>10} | {'Strength':>8}")
            print("-" * 70)

            for mc in consensus[:top_markets]:
                slug = mc.market_slug[:44]
                traders_count = len(mc.traders)
                side = mc.consensus_side[:10]
                strength = f"{mc.consensus_strength:.0%}"
                print(f"{slug:<45} | {traders_count:>7} | {side:>10} | {strength:>8}")

            # Show detailed breakdown for top 3 markets
            print(f"\n--- POSITION BREAKDOWN (Top 3 Markets) ---")
            for mc in consensus[:3]:
                print(f"\n{mc.market_slug}")
                print(f"  Volume: ${mc.total_volume:,.0f}")
                for outcome, addrs in mc.positions.items():
                    pct = len(addrs) / len(mc.traders) * 100 if mc.traders else 0
                    print(f"  {outcome}: {len(addrs)} traders ({pct:.0f}%)")
                    for addr in addrs[:3]:
                        name = mc.trader_names.get(addr, '')
                        display = f"{name}" if name else f"{addr[:12]}..."
                        print(f"    - {display}")
                    if len(addrs) > 3:
                        print(f"    ... and {len(addrs) - 3} more")

        # Show correlation matrix
        if show_matrix and matrix:
            print(f"\n{'='*70}")
            print("CORRELATION MATRIX (market overlap)")
            print("=" * 70)

            traders = list(matrix.keys())[:10]
            traders_data = scan['traders']

            # Header
            header = "Address        "
            for t in traders:
                name = traders_data.get(t, {}).get('pseudonym') or t[:8]
                header += f" {name[:8]:>8}"
            print(header)

            # Rows
            for t1 in traders:
                name1 = traders_data.get(t1, {}).get('pseudonym') or t1[:12]
                row = f"{name1[:14]:<14}"
                for t2 in traders:
                    corr = matrix.get(t1, {}).get(t2, 0)
                    if t1 == t2:
                        row += "     1.00"
                    elif corr > 0:
                        row += f"     {corr:.2f}"
                    else:
                        row += "        -"
                print(row)

    def export_results(self, results: Dict, filepath: str):
        """Export correlation results to JSON."""
        export_data = {
            'category': results['scan'].get('category'),
            'search_pattern': results['scan'].get('search_pattern'),
            'markets_scanned': len(results['scan']['markets']),
            'traders_found': len(results['scan']['traders']),
            'clusters': [
                {
                    'traders': c.traders,
                    'trader_names': c.trader_names,
                    'shared_markets': c.shared_markets,
                    'position_alignment': c.avg_position_alignment,
                    'total_volume': c.total_volume
                }
                for c in results['clusters']
            ],
            'market_consensus': [
                {
                    'market_slug': mc.market_slug,
                    'market_title': mc.market_title,
                    'traders_count': len(mc.traders),
                    'positions': mc.positions,
                    'consensus_side': mc.consensus_side,
                    'consensus_strength': mc.consensus_strength,
                    'total_volume': mc.total_volume
                }
                for mc in results['market_consensus']
            ],
            'correlation_matrix': results['correlation_matrix']
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"\nExported correlation analysis to {filepath}")
