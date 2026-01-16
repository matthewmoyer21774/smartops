"""
Alpha Signal Extractor

Extracts actionable alpha signals from smart money trading patterns.
This is the core intelligence for the prediction market hedge fund.

Key signals extracted:
1. Entry timing edge - when do winners enter relative to market movement
2. Position sizing patterns - how do winners size their bets
3. Market selection alpha - which markets do winners choose
4. Exit timing - when do winners take profits vs hold to resolution
5. Category specialization - what domains do winners focus on
"""

import os
import sys
import csv
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.file_cache import TraderCache
from models.trader import Trader, TraderStats
from models.position import Position


@dataclass
class AlphaSignal:
    """Individual alpha signal extracted from trader behavior."""
    signal_type: str = ""  # entry_timing, position_sizing, market_selection, etc.
    market_category: str = ""
    signal_value: float = 0.0
    confidence: float = 0.0
    sample_size: int = 0
    description: str = ""
    actionable_insight: str = ""


@dataclass
class TraderAlphaProfile:
    """Complete alpha profile for a trader."""
    address: str = ""
    name: str = ""

    # Performance baseline
    total_pnl: float = 0.0
    roi_pct: float = 0.0
    win_rate: float = 0.0
    total_positions: int = 0

    # Entry timing alpha
    avg_entry_price: float = 0.0
    entry_price_std: float = 0.0
    pct_entries_below_40: float = 0.0  # Contrarian bets
    pct_entries_40_60: float = 0.0     # Fair value bets
    pct_entries_60_80: float = 0.0     # Trend following
    pct_entries_above_80: float = 0.0  # High conviction bets

    # Position sizing alpha
    avg_position_usd: float = 0.0
    max_position_usd: float = 0.0
    position_size_vs_confidence: float = 0.0  # Correlation

    # Win rate by entry zone
    win_rate_below_40: float = 0.0
    win_rate_40_60: float = 0.0
    win_rate_60_80: float = 0.0
    win_rate_above_80: float = 0.0

    # ROI by entry zone
    avg_roi_below_40: float = 0.0
    avg_roi_40_60: float = 0.0
    avg_roi_60_80: float = 0.0
    avg_roi_above_80: float = 0.0

    # Category specialization
    top_category: str = ""
    category_win_rate: Dict[str, float] = field(default_factory=dict)
    category_roi: Dict[str, float] = field(default_factory=dict)

    # Alpha signals
    signals: List[AlphaSignal] = field(default_factory=list)


class AlphaExtractor:
    """
    Extracts alpha signals from smart money traders.

    The goal is to find patterns that can be replicated:
    - What entry prices lead to wins?
    - What position sizes correlate with conviction?
    - Which categories have the most edge?
    - What's the optimal entry zone for different market types?
    """

    MARKET_CATEGORIES = {
        'politics': ['trump', 'biden', 'election', 'president', 'senate', 'congress', 'vote', 'poll', 'democrat', 'republican', 'governor'],
        'crypto': ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol', 'price', 'token', 'coin'],
        'sports': ['nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football', 'basketball', 'game', 'match', 'team', 'player', 'win', 'score'],
        'finance': ['stock', 'fed', 'rate', 'inflation', 'market', 'dow', 'nasdaq', 'sp500', 'gdp', 'unemployment'],
        'entertainment': ['oscar', 'grammy', 'movie', 'show', 'award', 'celebrity', 'music', 'tv'],
        'tech': ['ai', 'openai', 'google', 'apple', 'microsoft', 'meta', 'launch', 'release', 'product'],
        'world': ['war', 'conflict', 'country', 'international', 'treaty', 'sanction'],
    }

    def __init__(self):
        self.cache = TraderCache()
        self.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'exports'
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def categorize_market(self, title: str, slug: str = "") -> str:
        """Determine market category from title and slug."""
        text = (title + " " + slug).lower()
        for category, keywords in self.MARKET_CATEGORIES.items():
            if any(kw in text for kw in keywords):
                return category
        return "other"

    def extract_alpha_profile(self, trader: Trader) -> TraderAlphaProfile:
        """Extract alpha profile from a single trader."""
        profile = TraderAlphaProfile()
        profile.address = trader.address
        profile.name = trader.name or trader.pseudonym or ""

        # Baseline stats
        profile.total_pnl = trader.stats.total_pnl
        profile.roi_pct = trader.stats.roi_pct
        profile.win_rate = trader.stats.win_rate
        profile.total_positions = len(trader.positions)

        if not trader.positions:
            return profile

        # Analyze positions by entry zone
        entry_zones = {
            'below_40': [],
            '40_60': [],
            '60_80': [],
            'above_80': []
        }

        category_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'total_roi': 0, 'count': 0, 'volume': 0
        })

        entry_prices = []
        position_sizes = []

        for pos in trader.positions.values():
            if pos.avg_entry_price <= 0:
                continue

            entry = pos.avg_entry_price
            entry_prices.append(entry)
            position_sizes.append(pos.total_cost)

            # Categorize by entry zone (normalize for Yes/No)
            effective_entry = entry if pos.outcome == 'Yes' else (1 - entry)

            zone_data = {
                'entry': effective_entry,
                'size': pos.total_cost,
                'resolved': pos.is_resolved,
                'won': pos.is_resolved and pos.realized_pnl > 0,
                'roi': pos.roi_pct if pos.is_resolved else 0,
                'pnl': pos.realized_pnl if pos.is_resolved else pos.unrealized_pnl
            }

            if effective_entry < 0.4:
                entry_zones['below_40'].append(zone_data)
            elif effective_entry < 0.6:
                entry_zones['40_60'].append(zone_data)
            elif effective_entry < 0.8:
                entry_zones['60_80'].append(zone_data)
            else:
                entry_zones['above_80'].append(zone_data)

            # Category stats
            cat = self.categorize_market(pos.market_title or "", pos.market_slug or "")
            category_stats[cat]['count'] += 1
            category_stats[cat]['volume'] += pos.total_cost
            if pos.is_resolved:
                if pos.realized_pnl > 0:
                    category_stats[cat]['wins'] += 1
                else:
                    category_stats[cat]['losses'] += 1
                category_stats[cat]['total_roi'] += pos.roi_pct

        # Calculate entry zone metrics
        total = len(entry_prices)
        if total > 0:
            profile.avg_entry_price = statistics.mean(entry_prices)
            if len(entry_prices) > 1:
                profile.entry_price_std = statistics.stdev(entry_prices)

            profile.pct_entries_below_40 = len(entry_zones['below_40']) / total
            profile.pct_entries_40_60 = len(entry_zones['40_60']) / total
            profile.pct_entries_60_80 = len(entry_zones['60_80']) / total
            profile.pct_entries_above_80 = len(entry_zones['above_80']) / total

            profile.avg_position_usd = statistics.mean(position_sizes)
            profile.max_position_usd = max(position_sizes)

        # Win rate and ROI by zone
        for zone_name, zone_data in entry_zones.items():
            resolved = [d for d in zone_data if d['resolved']]
            if resolved:
                wins = sum(1 for d in resolved if d['won'])
                win_rate = wins / len(resolved) * 100
                avg_roi = statistics.mean([d['roi'] for d in resolved])

                setattr(profile, f'win_rate_{zone_name}', win_rate)
                setattr(profile, f'avg_roi_{zone_name}', avg_roi)

        # Category win rates and ROI
        for cat, stats in category_stats.items():
            total_resolved = stats['wins'] + stats['losses']
            if total_resolved > 0:
                profile.category_win_rate[cat] = stats['wins'] / total_resolved * 100
                profile.category_roi[cat] = stats['total_roi'] / total_resolved

        # Top category by volume
        if category_stats:
            profile.top_category = max(category_stats.keys(),
                                       key=lambda k: category_stats[k]['volume'])

        # Generate alpha signals
        profile.signals = self._generate_signals(profile, entry_zones, category_stats)

        return profile

    def _generate_signals(self, profile: TraderAlphaProfile,
                         entry_zones: Dict,
                         category_stats: Dict) -> List[AlphaSignal]:
        """Generate actionable alpha signals."""
        signals = []

        # Entry timing alpha
        best_zone = None
        best_zone_roi = -float('inf')

        for zone in ['below_40', '40_60', '60_80', 'above_80']:
            roi = getattr(profile, f'avg_roi_{zone}', 0)
            if roi > best_zone_roi and len(entry_zones[zone]) >= 5:
                best_zone_roi = roi
                best_zone = zone

        if best_zone:
            signals.append(AlphaSignal(
                signal_type='entry_timing',
                signal_value=best_zone_roi,
                confidence=min(1.0, len(entry_zones[best_zone]) / 20),
                sample_size=len(entry_zones[best_zone]),
                description=f"Best entry zone: {best_zone} ({best_zone_roi:.1f}% avg ROI)",
                actionable_insight=f"Enter positions when price is in {best_zone.replace('_', '-')} range"
            ))

        # Category specialization alpha
        if profile.category_win_rate:
            best_cat = max(profile.category_win_rate.keys(),
                          key=lambda k: profile.category_win_rate.get(k, 0))
            best_cat_wr = profile.category_win_rate[best_cat]

            if best_cat_wr > 60:
                signals.append(AlphaSignal(
                    signal_type='category_specialization',
                    market_category=best_cat,
                    signal_value=best_cat_wr,
                    confidence=min(1.0, category_stats[best_cat]['wins'] + category_stats[best_cat]['losses']) / 20,
                    sample_size=category_stats[best_cat]['count'],
                    description=f"Specializes in {best_cat} ({best_cat_wr:.1f}% win rate)",
                    actionable_insight=f"Follow this trader primarily in {best_cat} markets"
                ))

        # High conviction signal
        if profile.pct_entries_above_80 > 0.3 and profile.win_rate_above_80 > 70:
            signals.append(AlphaSignal(
                signal_type='high_conviction',
                signal_value=profile.win_rate_above_80,
                confidence=0.8,
                sample_size=len(entry_zones['above_80']),
                description=f"High conviction bettor ({profile.pct_entries_above_80:.0%} entries at 80%+)",
                actionable_insight="When this trader enters at high prices, they usually win"
            ))

        # Contrarian signal
        if profile.pct_entries_below_40 > 0.3 and profile.win_rate_below_40 > 50:
            signals.append(AlphaSignal(
                signal_type='contrarian',
                signal_value=profile.win_rate_below_40,
                confidence=0.7,
                sample_size=len(entry_zones['below_40']),
                description=f"Successful contrarian ({profile.win_rate_below_40:.1f}% win rate on low entries)",
                actionable_insight="This trader finds value in underpriced outcomes"
            ))

        return signals

    def extract_all(self, min_pnl: float = 10000,
                   min_positions: int = 20,
                   verbose: bool = True) -> Tuple[List[TraderAlphaProfile], str]:
        """
        Extract alpha profiles for all qualifying traders.

        Returns profiles and filepath to exported CSV.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load all traders
        all_cached = self.cache.load_all_traders()

        profiles = []

        for i, trader_data in enumerate(all_cached):
            if verbose and (i + 1) % 10 == 0:
                print(f"Processing {i + 1}/{len(all_cached)}...")

            trader = self._dict_to_trader(trader_data) if isinstance(trader_data, dict) else trader_data

            # Filter
            if trader.stats.total_pnl < min_pnl:
                continue
            if len(trader.positions) < min_positions:
                continue

            profile = self.extract_alpha_profile(trader)
            profiles.append(profile)

        if verbose:
            print(f"\nExtracted alpha profiles for {len(profiles)} traders")

        # Export to CSV
        filepath = os.path.join(self.output_dir, f"alpha_signals_{timestamp}.csv")
        self._export_profiles(profiles, filepath)

        # Export summary
        summary = self._create_summary(profiles)
        summary_file = os.path.join(self.output_dir, f"alpha_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

        if verbose:
            print(f"Exported to {filepath}")
            print(f"Summary at {summary_file}")

        return profiles, filepath

    def _export_profiles(self, profiles: List[TraderAlphaProfile], filepath: str):
        """Export alpha profiles to CSV."""
        fieldnames = [
            'address', 'name', 'total_pnl', 'roi_pct', 'win_rate', 'total_positions',
            'avg_entry_price', 'entry_price_std',
            'pct_entries_below_40', 'pct_entries_40_60', 'pct_entries_60_80', 'pct_entries_above_80',
            'avg_position_usd', 'max_position_usd',
            'win_rate_below_40', 'win_rate_40_60', 'win_rate_60_80', 'win_rate_above_80',
            'avg_roi_below_40', 'avg_roi_40_60', 'avg_roi_60_80', 'avg_roi_above_80',
            'top_category', 'primary_signal', 'signal_insight'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for p in profiles:
                row = {
                    'address': p.address,
                    'name': p.name,
                    'total_pnl': p.total_pnl,
                    'roi_pct': p.roi_pct,
                    'win_rate': p.win_rate,
                    'total_positions': p.total_positions,
                    'avg_entry_price': p.avg_entry_price,
                    'entry_price_std': p.entry_price_std,
                    'pct_entries_below_40': p.pct_entries_below_40,
                    'pct_entries_40_60': p.pct_entries_40_60,
                    'pct_entries_60_80': p.pct_entries_60_80,
                    'pct_entries_above_80': p.pct_entries_above_80,
                    'avg_position_usd': p.avg_position_usd,
                    'max_position_usd': p.max_position_usd,
                    'win_rate_below_40': p.win_rate_below_40,
                    'win_rate_40_60': p.win_rate_40_60,
                    'win_rate_60_80': p.win_rate_60_80,
                    'win_rate_above_80': p.win_rate_above_80,
                    'avg_roi_below_40': p.avg_roi_below_40,
                    'avg_roi_40_60': p.avg_roi_40_60,
                    'avg_roi_60_80': p.avg_roi_60_80,
                    'avg_roi_above_80': p.avg_roi_above_80,
                    'top_category': p.top_category,
                    'primary_signal': p.signals[0].signal_type if p.signals else '',
                    'signal_insight': p.signals[0].actionable_insight if p.signals else ''
                }
                writer.writerow(row)

    def _create_summary(self, profiles: List[TraderAlphaProfile]) -> dict:
        """Create aggregate summary of alpha signals."""
        if not profiles:
            return {}

        summary = {
            'extraction_timestamp': datetime.now().isoformat(),
            'total_traders': len(profiles),
            'aggregate_stats': {},
            'entry_zone_analysis': {},
            'category_analysis': {},
            'top_insights': []
        }

        # Aggregate entry zone stats
        for zone in ['below_40', '40_60', '60_80', 'above_80']:
            win_rates = [getattr(p, f'win_rate_{zone}', 0) for p in profiles if getattr(p, f'win_rate_{zone}', 0) > 0]
            rois = [getattr(p, f'avg_roi_{zone}', 0) for p in profiles if getattr(p, f'avg_roi_{zone}', 0) != 0]

            if win_rates:
                summary['entry_zone_analysis'][zone] = {
                    'avg_win_rate': statistics.mean(win_rates),
                    'avg_roi': statistics.mean(rois) if rois else 0,
                    'sample_traders': len(win_rates)
                }

        # Category analysis
        all_categories = set()
        for p in profiles:
            all_categories.update(p.category_win_rate.keys())

        for cat in all_categories:
            win_rates = [p.category_win_rate.get(cat, 0) for p in profiles if cat in p.category_win_rate]
            rois = [p.category_roi.get(cat, 0) for p in profiles if cat in p.category_roi]

            if win_rates:
                summary['category_analysis'][cat] = {
                    'avg_win_rate': statistics.mean(win_rates),
                    'avg_roi': statistics.mean(rois) if rois else 0,
                    'traders_with_data': len(win_rates)
                }

        # Top insights
        all_signals = []
        for p in profiles:
            for s in p.signals:
                all_signals.append({
                    'trader': p.name or p.address[:12],
                    'pnl': p.total_pnl,
                    'signal_type': s.signal_type,
                    'insight': s.actionable_insight
                })

        summary['top_insights'] = sorted(all_signals, key=lambda x: x['pnl'], reverse=True)[:20]

        return summary

    def _dict_to_trader(self, trader_data: dict) -> Trader:
        """Convert dict to Trader object."""
        trader = Trader(address=trader_data.get('address', ''))
        trader.name = trader_data.get('name')
        trader.pseudonym = trader_data.get('pseudonym')

        stats_data = trader_data.get('stats', {})
        trader.stats = TraderStats(
            total_pnl=stats_data.get('total_pnl', 0),
            realized_pnl=stats_data.get('realized_pnl', 0),
            unrealized_pnl=stats_data.get('unrealized_pnl', 0),
            total_volume=stats_data.get('total_volume', 0),
            total_trades=stats_data.get('total_trades', 0),
            unique_markets=stats_data.get('unique_markets', 0),
            roi_pct=stats_data.get('roi_pct', 0),
            win_rate=stats_data.get('win_rate', 0),
            winning_positions=stats_data.get('winning_positions', 0),
            losing_positions=stats_data.get('losing_positions', 0),
            total_resolved=stats_data.get('total_resolved', 0),
            active_days=stats_data.get('active_days', 0),
            avg_position_size=stats_data.get('avg_position_size', 0),
            max_position_size=stats_data.get('max_position_size', 0),
            max_drawdown=stats_data.get('max_drawdown', 0)
        )

        positions_data = trader_data.get('positions', {})
        for key, pos_data in positions_data.items():
            pos = Position(
                condition_id=pos_data.get('condition_id', ''),
                outcome=pos_data.get('outcome', ''),
                market_slug=pos_data.get('market_slug', ''),
                market_title=pos_data.get('market_title', ''),
                total_shares=pos_data.get('total_shares', 0),
                total_cost=pos_data.get('total_cost', 0),
                avg_entry_price=pos_data.get('avg_entry_price', 0),
                current_price=pos_data.get('current_price', 0),
                current_value=pos_data.get('current_value', 0),
                is_resolved=pos_data.get('is_resolved', False),
                resolved_price=pos_data.get('resolved_price')
            )
            trader.positions[key] = pos

        return trader


def main():
    """CLI for alpha extraction."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract alpha signals from smart money')
    parser.add_argument('--min-pnl', type=float, default=10000,
                       help='Minimum P&L to include (default: 10000)')
    parser.add_argument('--min-positions', type=int, default=20,
                       help='Minimum positions to include (default: 20)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show progress')

    args = parser.parse_args()

    extractor = AlphaExtractor()
    profiles, filepath = extractor.extract_all(
        min_pnl=args.min_pnl,
        min_positions=args.min_positions,
        verbose=args.verbose
    )

    print(f"\nExtracted {len(profiles)} alpha profiles")
    print(f"Output: {filepath}")


if __name__ == '__main__':
    main()
