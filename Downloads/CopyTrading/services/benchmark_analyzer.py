"""
Service for analyzing trader performance by category and generating benchmarks.
"""

from typing import List, Dict, Optional
from collections import defaultdict
import statistics

import sys
sys.path.insert(0, '..')
from cache.file_cache import TraderCache
from config import MARKET_CATEGORIES


class BenchmarkAnalyzer:
    """Analyze and rank traders within specific market categories."""

    def __init__(self):
        self.cache = TraderCache()

    def get_category_leaderboard(
        self,
        category: str,
        limit: int = 50
    ) -> Dict:
        """
        Get leaderboard rankings for a specific category.

        Args:
            category: Market category (nba, nfl, crypto, etc.)
            limit: Max traders to return

        Returns:
            Dict with rankings, median stats, and percentiles
        """
        traders = self.cache.load_all_traders()

        if not traders:
            return {
                'category': category,
                'rankings': [],
                'median_roi': 0,
                'median_win_rate': 0,
                'total_traders': 0
            }

        keywords = MARKET_CATEGORIES.get(category, [category])

        # Calculate category-specific stats for each trader
        category_stats = []
        for trader in traders:
            stats = self._calculate_category_stats(trader, keywords)
            if stats and stats['positions'] > 0:
                category_stats.append({
                    'address': trader.get('address'),
                    'name': trader.get('name'),
                    'pseudonym': trader.get('pseudonym'),
                    'category_stats': stats,
                    'overall_stats': trader.get('stats', {})
                })

        if not category_stats:
            return {
                'category': category,
                'rankings': [],
                'median_roi': 0,
                'median_win_rate': 0,
                'total_traders': 0
            }

        # Sort by category P&L
        category_stats.sort(
            key=lambda t: t['category_stats'].get('pnl', 0),
            reverse=True
        )

        # Calculate percentiles
        all_rois = [t['category_stats']['roi'] for t in category_stats if t['category_stats']['roi']]
        all_win_rates = [t['category_stats']['win_rate'] for t in category_stats if t['category_stats']['win_rate']]

        median_roi = statistics.median(all_rois) if all_rois else 0
        median_win_rate = statistics.median(all_win_rates) if all_win_rates else 0

        # Add rankings and percentiles
        rankings = []
        for i, t in enumerate(category_stats[:limit], 1):
            roi = t['category_stats']['roi']
            percentile = self._calculate_percentile(roi, all_rois) if all_rois else 0

            rankings.append({
                'rank': i,
                'address': t['address'],
                'name': t.get('name') or t.get('pseudonym'),
                'category_stats': t['category_stats'],
                'percentile': round(percentile, 1)
            })

        return {
            'category': category,
            'rankings': rankings,
            'median_roi': round(median_roi, 2),
            'median_win_rate': round(median_win_rate, 2),
            'total_traders': len(category_stats)
        }

    def get_trader_category_breakdown(self, address: str) -> Dict:
        """
        Get per-category performance breakdown for a trader.

        Args:
            address: Trader wallet address

        Returns:
            Dict with stats for each category the trader has traded in
        """
        trader = self.cache.load_trader(address)

        if not trader:
            return {
                'address': address,
                'categories': {},
                'primary_category': None,
                'category_count': 0
            }

        positions = trader.get('positions', {})
        category_breakdown = {}

        for category, keywords in MARKET_CATEGORIES.items():
            stats = self._calculate_category_stats(trader, keywords)
            if stats and stats['positions'] > 0:
                category_breakdown[category] = stats

        # Find primary category (most positions or highest P&L)
        primary = None
        max_positions = 0
        for cat, stats in category_breakdown.items():
            if stats['positions'] > max_positions:
                max_positions = stats['positions']
                primary = cat

        return {
            'address': address,
            'categories': category_breakdown,
            'primary_category': primary,
            'category_count': len(category_breakdown)
        }

    def _calculate_category_stats(
        self,
        trader: Dict,
        keywords: List[str]
    ) -> Optional[Dict]:
        """
        Calculate stats for positions matching category keywords.

        Args:
            trader: Trader data dict
            keywords: List of keywords to match

        Returns:
            Dict with pnl, roi, win_rate, positions for the category
        """
        positions = trader.get('positions', {})

        if not positions:
            return None

        category_pnl = 0
        category_volume = 0
        category_wins = 0
        category_losses = 0
        category_positions = 0

        for key, pos in positions.items():
            market_slug = pos.get('market_slug', '').lower()
            market_title = pos.get('market_title', '').lower()

            # Check if position matches category
            matches = any(
                kw.lower() in market_slug or kw.lower() in market_title
                for kw in keywords
            )

            if matches:
                category_positions += 1
                realized = pos.get('realized_pnl', 0) or 0
                unrealized = pos.get('unrealized_pnl', 0) or 0
                category_pnl += realized + unrealized
                category_volume += pos.get('total_cost', 0) or 0

                if pos.get('is_resolved'):
                    if realized > 0:
                        category_wins += 1
                    elif realized < 0:
                        category_losses += 1

        if category_positions == 0:
            return None

        total_resolved = category_wins + category_losses
        win_rate = (category_wins / total_resolved * 100) if total_resolved > 0 else 0
        roi = (category_pnl / category_volume * 100) if category_volume > 0 else 0

        return {
            'pnl': round(category_pnl, 2),
            'roi': round(roi, 2),
            'win_rate': round(win_rate, 2),
            'positions': category_positions,
            'wins': category_wins,
            'losses': category_losses,
            'volume': round(category_volume, 2)
        }

    def _calculate_percentile(self, value: float, all_values: List[float]) -> float:
        """Calculate percentile of value within all_values."""
        if not all_values:
            return 0
        count_below = sum(1 for v in all_values if v < value)
        return (count_below / len(all_values)) * 100
