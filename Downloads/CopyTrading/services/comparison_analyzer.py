"""
Service for comparing multiple traders side-by-side.
"""

from typing import List, Dict, Optional
from collections import defaultdict
import statistics
import math

import sys
sys.path.insert(0, '..')
from cache.file_cache import TraderCache


class ComparisonAnalyzer:
    """Compare multiple traders with normalized metrics and correlation analysis."""

    def __init__(self):
        self.cache = TraderCache()

    def compare_traders(self, addresses: List[str]) -> Dict:
        """
        Compare multiple traders side-by-side.

        Args:
            addresses: List of trader wallet addresses

        Returns:
            Dict with comparison data, normalized metrics, and correlations
        """
        traders_data = []

        for addr in addresses:
            trader = self.cache.load_trader(addr)
            if trader:
                traders_data.append(trader)

        if len(traders_data) < 2:
            return {
                'traders': [],
                'correlation_matrix': {},
                'shared_markets': [],
                'divergent_positions': [],
                'error': 'Need at least 2 traders to compare'
            }

        # Extract metrics for each trader
        metrics_list = []
        for t in traders_data:
            stats = t.get('stats', {})
            metrics_list.append({
                'address': t.get('address'),
                'name': t.get('name') or t.get('pseudonym'),
                'metrics': {
                    'total_pnl': stats.get('total_pnl', 0),
                    'roi_pct': stats.get('roi_pct', 0),
                    'win_rate': stats.get('win_rate', 0),
                    'max_drawdown': stats.get('max_drawdown', 0),
                    'total_trades': stats.get('total_trades', 0),
                    'total_volume': stats.get('total_volume', 0),
                    'unique_markets': stats.get('unique_markets', 0),
                    'sharpe_estimate': self._estimate_sharpe(t)
                }
            })

        # Calculate normalized (z-score) metrics
        all_pnls = [m['metrics']['total_pnl'] for m in metrics_list]
        all_rois = [m['metrics']['roi_pct'] for m in metrics_list]
        all_win_rates = [m['metrics']['win_rate'] for m in metrics_list]

        for m in metrics_list:
            m['normalized'] = {
                'pnl_zscore': self._calculate_zscore(m['metrics']['total_pnl'], all_pnls),
                'roi_zscore': self._calculate_zscore(m['metrics']['roi_pct'], all_rois),
                'win_rate_zscore': self._calculate_zscore(m['metrics']['win_rate'], all_win_rates)
            }

        # Find shared markets
        shared_markets = self._find_shared_markets(traders_data)

        # Find divergent positions (where traders disagree)
        divergent = self._find_divergent_positions(traders_data)

        # Build correlation matrix
        correlation_matrix = self._build_correlation_matrix(traders_data)

        return {
            'traders': metrics_list,
            'correlation_matrix': correlation_matrix,
            'shared_markets': shared_markets[:20],  # Top 20
            'divergent_positions': divergent[:10],  # Top 10
            'summary': {
                'total_compared': len(traders_data),
                'shared_market_count': len(shared_markets),
                'divergent_count': len(divergent),
                'avg_correlation': self._avg_correlation(correlation_matrix)
            }
        }

    def find_divergence(self, addresses: List[str]) -> Dict:
        """
        Find markets where traders have opposing positions.

        Args:
            addresses: List of trader wallet addresses

        Returns:
            Dict with markets where traders disagree
        """
        traders_data = []
        for addr in addresses:
            trader = self.cache.load_trader(addr)
            if trader:
                traders_data.append(trader)

        if len(traders_data) < 2:
            return {'divergent_positions': [], 'error': 'Need at least 2 traders'}

        divergent = self._find_divergent_positions(traders_data)

        return {
            'divergent_positions': divergent,
            'total_divergent': len(divergent)
        }

    def _estimate_sharpe(self, trader: Dict) -> float:
        """
        Estimate Sharpe-like ratio from equity curve.

        Args:
            trader: Trader data dict

        Returns:
            Estimated Sharpe ratio
        """
        equity_curve = trader.get('equity_curve', [])

        if len(equity_curve) < 2:
            return 0

        # Calculate daily returns
        returns = []
        for i in range(1, len(equity_curve)):
            prev_pnl = equity_curve[i-1].get('cumulative_pnl', 0)
            curr_pnl = equity_curve[i].get('cumulative_pnl', 0)
            if prev_pnl != 0:
                ret = (curr_pnl - prev_pnl) / abs(prev_pnl) if prev_pnl else 0
                returns.append(ret)

        if not returns or len(returns) < 2:
            return 0

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 1

        if std_return == 0:
            return 0

        # Annualize (assume daily data, 365 trading days)
        sharpe = (mean_return / std_return) * math.sqrt(365)
        return round(sharpe, 2)

    def _calculate_zscore(self, value: float, all_values: List[float]) -> float:
        """Calculate z-score for a value within a list."""
        if len(all_values) < 2:
            return 0

        mean = statistics.mean(all_values)
        std = statistics.stdev(all_values) if len(all_values) > 1 else 1

        if std == 0:
            return 0

        return round((value - mean) / std, 2)

    def _find_shared_markets(self, traders: List[Dict]) -> List[str]:
        """Find markets where multiple traders have positions."""
        market_counts = defaultdict(int)

        for trader in traders:
            positions = trader.get('positions', {})
            seen_markets = set()
            for key, pos in positions.items():
                slug = pos.get('market_slug', '')
                if slug and slug not in seen_markets:
                    market_counts[slug] += 1
                    seen_markets.add(slug)

        # Return markets with 2+ traders
        shared = [m for m, count in market_counts.items() if count >= 2]
        shared.sort(key=lambda m: market_counts[m], reverse=True)
        return shared

    def _find_divergent_positions(self, traders: List[Dict]) -> List[Dict]:
        """Find markets where traders have opposing positions."""
        # Build market -> {outcome -> [traders]} mapping
        market_positions = defaultdict(lambda: defaultdict(list))

        for trader in traders:
            addr = trader.get('address')
            positions = trader.get('positions', {})

            for key, pos in positions.items():
                slug = pos.get('market_slug', '')
                outcome = pos.get('outcome', '')

                if slug and outcome:
                    market_positions[slug][outcome].append({
                        'address': addr,
                        'name': trader.get('name') or trader.get('pseudonym'),
                        'shares': pos.get('total_shares', 0),
                        'pnl': pos.get('realized_pnl', 0) or pos.get('unrealized_pnl', 0)
                    })

        # Find markets with traders on both sides
        divergent = []
        for slug, outcomes in market_positions.items():
            if len(outcomes) >= 2:  # Has at least 2 different outcomes
                # Check for Yes/No split
                yes_traders = outcomes.get('Yes', [])
                no_traders = outcomes.get('No', [])

                if yes_traders and no_traders:
                    divergent.append({
                        'market_slug': slug,
                        'traders_yes': yes_traders,
                        'traders_no': no_traders,
                        'yes_count': len(yes_traders),
                        'no_count': len(no_traders)
                    })

        # Sort by total trader involvement
        divergent.sort(key=lambda d: d['yes_count'] + d['no_count'], reverse=True)
        return divergent

    def _build_correlation_matrix(self, traders: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Build correlation matrix based on market overlap.

        Args:
            traders: List of trader data dicts

        Returns:
            Dict of {addr: {addr: correlation}}
        """
        # Get markets for each trader
        trader_markets = {}
        for t in traders:
            addr = t.get('address')
            positions = t.get('positions', {})
            markets = set()
            for key, pos in positions.items():
                slug = pos.get('market_slug', '')
                if slug:
                    markets.add(slug)
            trader_markets[addr] = markets

        # Calculate pairwise Jaccard similarity
        matrix = {}
        addresses = list(trader_markets.keys())

        for addr1 in addresses:
            matrix[addr1] = {}
            markets1 = trader_markets.get(addr1, set())

            for addr2 in addresses:
                if addr1 == addr2:
                    matrix[addr1][addr2] = 1.0
                else:
                    markets2 = trader_markets.get(addr2, set())
                    intersection = len(markets1 & markets2)
                    union = len(markets1 | markets2)
                    similarity = intersection / union if union > 0 else 0
                    matrix[addr1][addr2] = round(similarity, 3)

        return matrix

    def _avg_correlation(self, matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate average off-diagonal correlation."""
        correlations = []
        addresses = list(matrix.keys())

        for i, addr1 in enumerate(addresses):
            for addr2 in addresses[i+1:]:
                correlations.append(matrix.get(addr1, {}).get(addr2, 0))

        return round(statistics.mean(correlations), 3) if correlations else 0
