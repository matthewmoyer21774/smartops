"""
Service for analyzing trade timing and detecting momentum patterns.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

import sys
sys.path.insert(0, '..')
from cache.file_cache import TraderCache
from config import MARKET_CATEGORIES


class TimingAnalyzer:
    """Analyze entry timing quality and detect smart money momentum."""

    def __init__(self):
        self.cache = TraderCache()

    def analyze_timing(self, address: str) -> Dict:
        """
        Analyze entry timing quality for a trader.

        Args:
            address: Trader wallet address

        Returns:
            Dict with timing analysis metrics
        """
        trader = self.cache.load_trader(address)

        if not trader:
            return {'error': 'Trader not found'}

        positions = trader.get('positions', {})

        if not positions:
            return {
                'address': address,
                'timing_score': 0,
                'avg_entry_vs_resolution': 0,
                'entry_patterns': {},
                'positions_analyzed': 0
            }

        timing_data = []
        category_timing = defaultdict(list)

        for key, pos in positions.items():
            if not pos.get('is_resolved'):
                continue

            entry_price = pos.get('avg_entry_price', 0)
            outcome = pos.get('outcome', '')

            # Resolution price is 1 for winning side, 0 for losing
            pnl = pos.get('realized_pnl', 0) or 0
            won = pnl > 0

            if outcome == 'Yes':
                resolution_price = 1.0 if won else 0.0
            else:
                resolution_price = 0.0 if won else 1.0

            # Calculate edge captured
            if won:
                edge = abs(resolution_price - entry_price)
            else:
                edge = -abs(resolution_price - entry_price)

            timing_data.append({
                'entry_price': entry_price,
                'resolution_price': resolution_price,
                'edge': edge,
                'won': won,
                'market_slug': pos.get('market_slug', '')
            })

            # Track by category
            slug = pos.get('market_slug', '').lower()
            for cat, keywords in MARKET_CATEGORIES.items():
                if any(kw.lower() in slug for kw in keywords):
                    category_timing[cat].append(edge)
                    break

        if not timing_data:
            return {
                'address': address,
                'timing_score': 0,
                'avg_entry_vs_resolution': 0,
                'entry_patterns': {},
                'positions_analyzed': 0
            }

        # Calculate timing score (0-1, how good are entries)
        positive_edges = [t['edge'] for t in timing_data if t['edge'] > 0]
        timing_score = len(positive_edges) / len(timing_data) if timing_data else 0

        # Average edge captured
        avg_edge = statistics.mean([t['edge'] for t in timing_data])

        # Entry price distribution
        entry_prices = [t['entry_price'] for t in timing_data]
        below_50 = len([p for p in entry_prices if p < 0.5])
        above_50 = len([p for p in entry_prices if p >= 0.5])

        # Best category for timing
        best_cat = None
        best_cat_edge = -1
        for cat, edges in category_timing.items():
            avg = statistics.mean(edges) if edges else 0
            if avg > best_cat_edge:
                best_cat_edge = avg
                best_cat = cat

        return {
            'address': address,
            'timing_score': round(timing_score, 3),
            'avg_entry_vs_resolution': round(avg_edge, 3),
            'entry_patterns': {
                'prefers_underdogs': below_50 > above_50,
                'entries_below_50': below_50,
                'entries_above_50': above_50,
                'avg_entry_price': round(statistics.mean(entry_prices), 3),
                'best_category_timing': best_cat,
                'best_category_edge': round(best_cat_edge, 3) if best_cat else 0
            },
            'positions_analyzed': len(timing_data),
            'win_count': len(positive_edges)
        }

    def detect_momentum(
        self,
        category: str = None,
        time_window_hours: int = 24,
        min_traders: int = 3,
        min_win_rate: float = 55
    ) -> Dict:
        """
        Detect markets where smart money is entering recently.

        Args:
            category: Filter by category (optional)
            time_window_hours: Look for entries within this window
            min_traders: Minimum traders entering
            min_win_rate: Minimum win rate for "smart money"

        Returns:
            Dict with emerging position data
        """
        traders = self.cache.load_all_traders()

        # Filter smart money traders
        smart_traders = [
            t for t in traders
            if t.get('stats', {}).get('win_rate', 0) >= min_win_rate
        ]

        if not smart_traders:
            return {
                'emerging_positions': [],
                'message': 'No smart money traders found'
            }

        keywords = MARKET_CATEGORIES.get(category, []) if category else []

        # Aggregate recent positions
        market_entries = defaultdict(lambda: {'yes': [], 'no': [], 'title': ''})

        cutoff = datetime.now() - timedelta(hours=time_window_hours)

        for trader in smart_traders:
            addr = trader.get('address')
            stats = trader.get('stats', {})
            positions = trader.get('positions', {})

            for key, pos in positions.items():
                # Check if it's a recent open position
                if pos.get('is_resolved'):
                    continue

                slug = pos.get('market_slug', '')
                title = pos.get('market_title', '')

                # Filter by category if specified
                if keywords:
                    matches = any(
                        kw.lower() in slug.lower() or kw.lower() in title.lower()
                        for kw in keywords
                    )
                    if not matches:
                        continue

                if not market_entries[slug]['title']:
                    market_entries[slug]['title'] = title

                outcome = pos.get('outcome', '')
                trader_info = {
                    'address': addr,
                    'name': trader.get('name') or trader.get('pseudonym'),
                    'win_rate': stats.get('win_rate', 0),
                    'total_pnl': stats.get('total_pnl', 0),
                    'shares': pos.get('total_shares', 0),
                    'entry_price': pos.get('avg_entry_price', 0)
                }

                if outcome == 'Yes':
                    market_entries[slug]['yes'].append(trader_info)
                elif outcome == 'No':
                    market_entries[slug]['no'].append(trader_info)

        # Build emerging positions list
        emerging = []
        for slug, data in market_entries.items():
            yes_traders = data['yes']
            no_traders = data['no']

            # Check Yes side
            if len(yes_traders) >= min_traders:
                combined_win_rate = statistics.mean([t['win_rate'] for t in yes_traders])
                combined_volume = sum(t['shares'] * t['entry_price'] for t in yes_traders)

                emerging.append({
                    'market_slug': slug,
                    'market_title': data['title'],
                    'side': 'Yes',
                    'traders_entered': len(yes_traders),
                    'combined_win_rate': round(combined_win_rate, 1),
                    'combined_volume': round(combined_volume, 2),
                    'avg_entry_price': round(
                        statistics.mean([t['entry_price'] for t in yes_traders]), 3
                    ),
                    'traders': yes_traders[:5],  # Top 5
                    'time_window': f"{time_window_hours}h"
                })

            # Check No side
            if len(no_traders) >= min_traders:
                combined_win_rate = statistics.mean([t['win_rate'] for t in no_traders])
                combined_volume = sum(t['shares'] * t['entry_price'] for t in no_traders)

                emerging.append({
                    'market_slug': slug,
                    'market_title': data['title'],
                    'side': 'No',
                    'traders_entered': len(no_traders),
                    'combined_win_rate': round(combined_win_rate, 1),
                    'combined_volume': round(combined_volume, 2),
                    'avg_entry_price': round(
                        statistics.mean([t['entry_price'] for t in no_traders]), 3
                    ),
                    'traders': no_traders[:5],
                    'time_window': f"{time_window_hours}h"
                })

        # Sort by number of traders
        emerging.sort(key=lambda e: e['traders_entered'], reverse=True)

        return {
            'emerging_positions': emerging,
            'filters': {
                'category': category,
                'time_window_hours': time_window_hours,
                'min_traders': min_traders,
                'min_win_rate': min_win_rate
            },
            'smart_traders_count': len(smart_traders),
            'markets_with_momentum': len(emerging)
        }

    def get_entry_quality_leaderboard(self, limit: int = 20) -> Dict:
        """
        Rank traders by entry quality (timing score).

        Args:
            limit: Max traders to return

        Returns:
            Dict with ranked traders by timing
        """
        traders = self.cache.load_all_traders()

        timing_scores = []
        for trader in traders:
            analysis = self.analyze_timing(trader.get('address'))
            if analysis.get('positions_analyzed', 0) >= 5:  # Min 5 resolved positions
                timing_scores.append({
                    'address': trader.get('address'),
                    'name': trader.get('name') or trader.get('pseudonym'),
                    'timing_score': analysis['timing_score'],
                    'avg_edge': analysis['avg_entry_vs_resolution'],
                    'positions_analyzed': analysis['positions_analyzed'],
                    'overall_win_rate': trader.get('stats', {}).get('win_rate', 0),
                    'overall_pnl': trader.get('stats', {}).get('total_pnl', 0)
                })

        # Sort by timing score
        timing_scores.sort(key=lambda t: t['timing_score'], reverse=True)

        return {
            'rankings': timing_scores[:limit],
            'total_analyzed': len(timing_scores)
        }
