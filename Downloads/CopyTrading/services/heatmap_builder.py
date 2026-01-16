"""
Service for building position heat maps and consensus visualization data.
"""

from typing import List, Dict, Optional
from collections import defaultdict

import sys
sys.path.insert(0, '..')
from cache.file_cache import TraderCache
from config import MARKET_CATEGORIES


class HeatmapBuilder:
    """Build position density and consensus visualizations."""

    def __init__(self):
        self.cache = TraderCache()

    def get_category_heatmap(
        self,
        category: str,
        min_traders: int = 2,
        smart_money_only: bool = False,
        min_pnl: float = 0,
        min_win_rate: float = 0
    ) -> Dict:
        """
        Get position heat map for a market category.

        Args:
            category: Market category (nba, nfl, crypto, etc.)
            min_traders: Minimum traders to include a market
            smart_money_only: Filter to profitable traders only
            min_pnl: Minimum P&L for smart money filter
            min_win_rate: Minimum win rate for smart money filter

        Returns:
            Dict with market consensus data
        """
        traders = self.cache.load_all_traders()
        keywords = MARKET_CATEGORIES.get(category, [category])

        # Filter traders if smart money mode
        if smart_money_only or min_pnl > 0 or min_win_rate > 0:
            traders = [
                t for t in traders
                if t.get('stats', {}).get('total_pnl', 0) >= min_pnl
                and t.get('stats', {}).get('win_rate', 0) >= min_win_rate
            ]

        # Aggregate positions by market
        market_data = defaultdict(lambda: {
            'yes_traders': [],
            'no_traders': [],
            'title': ''
        })

        for trader in traders:
            addr = trader.get('address')
            stats = trader.get('stats', {})
            positions = trader.get('positions', {})

            for key, pos in positions.items():
                market_slug = pos.get('market_slug', '').lower()
                market_title = pos.get('market_title', '')

                # Check if matches category
                matches = any(
                    kw.lower() in market_slug or kw.lower() in market_title.lower()
                    for kw in keywords
                )

                if not matches:
                    continue

                outcome = pos.get('outcome', '')
                slug = pos.get('market_slug', '')

                if not market_data[slug]['title']:
                    market_data[slug]['title'] = market_title

                trader_info = {
                    'address': addr,
                    'name': trader.get('name') or trader.get('pseudonym'),
                    'shares': pos.get('total_shares', 0),
                    'entry_price': pos.get('avg_entry_price', 0),
                    'pnl': pos.get('realized_pnl', 0) or pos.get('unrealized_pnl', 0),
                    'trader_win_rate': stats.get('win_rate', 0),
                    'trader_pnl': stats.get('total_pnl', 0)
                }

                if outcome == 'Yes':
                    market_data[slug]['yes_traders'].append(trader_info)
                elif outcome == 'No':
                    market_data[slug]['no_traders'].append(trader_info)

        # Build heat map data
        markets = []
        for slug, data in market_data.items():
            yes_traders = data['yes_traders']
            no_traders = data['no_traders']
            total_traders = len(yes_traders) + len(no_traders)

            if total_traders < min_traders:
                continue

            # Calculate aggregates
            yes_volume = sum(t['shares'] * t['entry_price'] for t in yes_traders)
            no_volume = sum(t['shares'] * t['entry_price'] for t in no_traders)
            yes_avg_price = (
                sum(t['entry_price'] for t in yes_traders) / len(yes_traders)
                if yes_traders else 0
            )
            no_avg_price = (
                sum(t['entry_price'] for t in no_traders) / len(no_traders)
                if no_traders else 0
            )
            yes_avg_win_rate = (
                sum(t['trader_win_rate'] for t in yes_traders) / len(yes_traders)
                if yes_traders else 0
            )
            no_avg_win_rate = (
                sum(t['trader_win_rate'] for t in no_traders) / len(no_traders)
                if no_traders else 0
            )

            # Determine consensus
            yes_count = len(yes_traders)
            no_count = len(no_traders)

            if yes_count > no_count:
                consensus = 'Yes'
                consensus_strength = yes_count / total_traders
            elif no_count > yes_count:
                consensus = 'No'
                consensus_strength = no_count / total_traders
            else:
                consensus = 'Split'
                consensus_strength = 0.5

            # Smart money consensus (based on traders with higher win rates)
            smart_yes = [t for t in yes_traders if t['trader_win_rate'] > 55]
            smart_no = [t for t in no_traders if t['trader_win_rate'] > 55]

            if len(smart_yes) > len(smart_no):
                smart_money_side = 'Yes'
            elif len(smart_no) > len(smart_yes):
                smart_money_side = 'No'
            else:
                smart_money_side = consensus

            markets.append({
                'slug': slug,
                'title': data['title'],
                'yes_data': {
                    'trader_count': yes_count,
                    'total_volume': round(yes_volume, 2),
                    'avg_entry_price': round(yes_avg_price, 3),
                    'avg_win_rate': round(yes_avg_win_rate, 1)
                },
                'no_data': {
                    'trader_count': no_count,
                    'total_volume': round(no_volume, 2),
                    'avg_entry_price': round(no_avg_price, 3),
                    'avg_win_rate': round(no_avg_win_rate, 1)
                },
                'consensus': consensus,
                'consensus_strength': round(consensus_strength, 2),
                'smart_money_side': smart_money_side,
                'total_traders': total_traders,
                'total_volume': round(yes_volume + no_volume, 2)
            })

        # Sort by total traders
        markets.sort(key=lambda m: m['total_traders'], reverse=True)

        return {
            'category': category,
            'markets': markets,
            'total_markets': len(markets),
            'filters': {
                'min_traders': min_traders,
                'smart_money_only': smart_money_only,
                'min_pnl': min_pnl,
                'min_win_rate': min_win_rate
            }
        }

    def get_market_heatmap(
        self,
        market_slugs: List[str] = None,
        min_traders: int = 1
    ) -> Dict:
        """
        Get position heat map for specific markets.

        Args:
            market_slugs: List of market slugs (or None for all)
            min_traders: Minimum traders per market

        Returns:
            Dict with market position data
        """
        traders = self.cache.load_all_traders()

        # Aggregate positions
        market_data = defaultdict(lambda: {
            'yes_traders': [],
            'no_traders': [],
            'title': ''
        })

        for trader in traders:
            addr = trader.get('address')
            stats = trader.get('stats', {})
            positions = trader.get('positions', {})

            for key, pos in positions.items():
                slug = pos.get('market_slug', '')

                # Filter by slugs if provided
                if market_slugs and slug not in market_slugs:
                    continue

                if not market_data[slug]['title']:
                    market_data[slug]['title'] = pos.get('market_title', '')

                outcome = pos.get('outcome', '')
                trader_info = {
                    'address': addr,
                    'name': trader.get('name') or trader.get('pseudonym'),
                    'shares': pos.get('total_shares', 0),
                    'entry_price': pos.get('avg_entry_price', 0),
                    'pnl': pos.get('realized_pnl', 0) or pos.get('unrealized_pnl', 0),
                    'trader_win_rate': stats.get('win_rate', 0)
                }

                if outcome == 'Yes':
                    market_data[slug]['yes_traders'].append(trader_info)
                elif outcome == 'No':
                    market_data[slug]['no_traders'].append(trader_info)

        # Build response
        markets = []
        for slug, data in market_data.items():
            total = len(data['yes_traders']) + len(data['no_traders'])
            if total < min_traders:
                continue

            yes_count = len(data['yes_traders'])
            no_count = len(data['no_traders'])

            markets.append({
                'slug': slug,
                'title': data['title'],
                'yes_count': yes_count,
                'no_count': no_count,
                'total_traders': total,
                'consensus': 'Yes' if yes_count > no_count else ('No' if no_count > yes_count else 'Split'),
                'consensus_strength': max(yes_count, no_count) / total if total > 0 else 0,
                'yes_traders': data['yes_traders'][:10],  # Top 10
                'no_traders': data['no_traders'][:10]
            })

        markets.sort(key=lambda m: m['total_traders'], reverse=True)

        return {
            'markets': markets,
            'total_markets': len(markets)
        }

    def get_consensus_summary(self) -> Dict:
        """
        Get overall consensus summary across all categories.

        Returns:
            Dict with consensus counts per category
        """
        summary = {}

        for category in MARKET_CATEGORIES.keys():
            heatmap = self.get_category_heatmap(category, min_traders=2)

            yes_consensus = len([m for m in heatmap['markets'] if m['consensus'] == 'Yes'])
            no_consensus = len([m for m in heatmap['markets'] if m['consensus'] == 'No'])
            split = len([m for m in heatmap['markets'] if m['consensus'] == 'Split'])

            summary[category] = {
                'total_markets': heatmap['total_markets'],
                'yes_consensus': yes_consensus,
                'no_consensus': no_consensus,
                'split': split,
                'avg_strength': (
                    sum(m['consensus_strength'] for m in heatmap['markets']) / len(heatmap['markets'])
                    if heatmap['markets'] else 0
                )
            }

        return {
            'categories': summary,
            'total_categories': len(summary)
        }
