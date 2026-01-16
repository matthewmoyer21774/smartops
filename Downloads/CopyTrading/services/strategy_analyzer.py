"""
Deep strategy analysis service for understanding trader behavior.
"""

import re
import statistics
from typing import Dict, List, Optional
from collections import defaultdict

import sys
sys.path.insert(0, '..')
from cache.file_cache import TraderCache
from services.trader_analyzer import TraderAnalyzerService
from models.strategy_report import StrategyReport, CategoryStats


# Market category keywords for auto-detection
CATEGORY_KEYWORDS = {
    'crypto': ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol', 'xrp', 'doge', 'memecoin', 'token', 'coin'],
    'politics': ['trump', 'biden', 'election', 'president', 'senate', 'congress', 'vote', 'democrat', 'republican', 'governor', 'political', 'poll'],
    'sports': ['nfl', 'nba', 'mlb', 'nhl', 'ufc', 'football', 'basketball', 'baseball', 'hockey', 'soccer', 'tennis', 'golf', 'superbowl', 'championship', 'playoffs'],
    'entertainment': ['oscar', 'grammy', 'emmy', 'movie', 'film', 'album', 'celebrity', 'kardashian', 'music', 'tv show'],
    'economy': ['fed', 'interest rate', 'gdp', 'inflation', 'unemployment', 'stock', 'market', 'recession', 'treasury'],
    'tech': ['apple', 'google', 'microsoft', 'ai', 'openai', 'chatgpt', 'tesla', 'spacex', 'meta', 'amazon'],
    'world': ['ukraine', 'russia', 'china', 'war', 'nato', 'israel', 'gaza', 'iran', 'north korea'],
}


class StrategyAnalyzer:
    """Analyze trader strategy in depth."""

    def __init__(self):
        self.cache = TraderCache()
        self.trader_analyzer = TraderAnalyzerService()

    def analyze_strategy(self, address: str, refresh: bool = False) -> StrategyReport:
        """
        Perform deep strategy analysis on a trader.

        Args:
            address: Wallet address
            refresh: Force refresh from API instead of using cache

        Returns:
            StrategyReport with comprehensive analysis
        """
        address = address.lower()

        # Load or fetch trader data
        trader_data = self.cache.load_trader(address)

        if not trader_data or refresh:
            print(f"Fetching fresh data for {address[:16]}...")
            trader = self.trader_analyzer.analyze_trader(address)
            self.cache.save_trader(trader)
            trader_data = trader.to_dict()

        # Build report
        report = StrategyReport(
            address=address,
            name=trader_data.get('name'),
            pseudonym=trader_data.get('pseudonym')
        )

        positions = trader_data.get('positions', {})
        stats = trader_data.get('stats', {})

        # Copy basic stats
        report.total_pnl = stats.get('total_pnl', 0)
        report.realized_pnl = stats.get('realized_pnl', 0)
        report.unrealized_pnl = stats.get('unrealized_pnl', 0)
        report.total_trades = stats.get('total_trades', 0)
        report.total_volume = stats.get('total_volume', 0)
        report.unique_markets = stats.get('unique_markets', 0)
        report.active_days = stats.get('active_days', 0)
        report.first_trade_date = str(stats.get('first_trade_date', ''))
        report.last_trade_date = str(stats.get('last_trade_date', ''))

        # Analyze components
        self._analyze_win_patterns(positions, report)
        self._analyze_market_categories(positions, report)
        self._analyze_position_sizing(positions, report)
        self._analyze_price_behavior(positions, report)
        self._find_top_positions(positions, report)
        self._detect_suspicious_patterns(report)
        self._detect_strategy_type(report)

        return report

    def _analyze_win_patterns(self, positions: Dict, report: StrategyReport):
        """Analyze win/loss patterns."""
        report.total_positions = len(positions)

        win_pnls = []
        loss_pnls = []

        for pos in positions.values():
            if pos.get('is_resolved'):
                report.total_resolved += 1
                pnl = pos.get('realized_pnl', 0)
                if pnl > 0:
                    report.wins += 1
                    win_pnls.append(pnl)
                elif pnl < 0:
                    report.losses += 1
                    loss_pnls.append(abs(pnl))

        # Calculate averages
        if win_pnls:
            report.avg_win_size = statistics.mean(win_pnls)
        if loss_pnls:
            report.avg_loss_size = statistics.mean(loss_pnls)

        # Win rate
        if report.total_resolved > 0:
            report.win_rate = (report.wins / report.total_resolved) * 100

        # Profit factor
        total_wins = sum(win_pnls)
        total_losses = sum(loss_pnls)
        if total_losses > 0:
            report.profit_factor = total_wins / total_losses
        elif total_wins > 0:
            report.profit_factor = float('inf')

    def _detect_category(self, market_slug: str, market_title: str) -> str:
        """Detect market category from slug/title."""
        text = f"{market_slug} {market_title}".lower()

        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return category

        return 'other'

    def _analyze_market_categories(self, positions: Dict, report: StrategyReport):
        """Break down performance by market category."""
        categories: Dict[str, CategoryStats] = {}

        for pos in positions.values():
            slug = pos.get('market_slug', '')
            title = pos.get('market_title', '')
            category = self._detect_category(slug, title)

            if category not in categories:
                categories[category] = CategoryStats(name=category)

            cat_stats = categories[category]
            cat_stats.position_count += 1
            cat_stats.total_volume += pos.get('total_cost', 0)

            if pos.get('is_resolved'):
                pnl = pos.get('realized_pnl', 0)
                cat_stats.total_pnl += pnl
                if pnl > 0:
                    cat_stats.wins += 1
                elif pnl < 0:
                    cat_stats.losses += 1

        report.categories = categories

    def _analyze_position_sizing(self, positions: Dict, report: StrategyReport):
        """Analyze position sizing patterns."""
        sizes = []

        for pos in positions.values():
            cost = pos.get('total_cost', 0)
            if cost > 0:
                sizes.append(cost)

        if not sizes:
            return

        report.avg_position = statistics.mean(sizes)
        report.median_position = statistics.median(sizes)
        report.max_position = max(sizes)
        report.min_position = min(sizes)

        if len(sizes) > 1:
            report.position_stddev = statistics.stdev(sizes)

    def _analyze_price_behavior(self, positions: Dict, report: StrategyReport):
        """Analyze price behavior and entry quality."""
        entry_prices = []
        winning_entries = []
        losing_entries = []

        for pos in positions.values():
            avg_entry = pos.get('avg_entry_price', 0)
            if avg_entry > 0:
                entry_prices.append(avg_entry)

                # Track trades by price level
                trade_count = pos.get('trade_count', 1)
                if avg_entry < 0.5:
                    report.buys_below_50 += trade_count
                else:
                    report.buys_above_50 += trade_count
                report.total_buys += trade_count

                # Winning vs losing entries
                if pos.get('is_resolved'):
                    pnl = pos.get('realized_pnl', 0)
                    if pnl > 0:
                        winning_entries.append(avg_entry)
                    elif pnl < 0:
                        losing_entries.append(avg_entry)

        if entry_prices:
            report.avg_entry_price = statistics.mean(entry_prices)
        if winning_entries:
            report.avg_winning_entry = statistics.mean(winning_entries)
        if losing_entries:
            report.avg_losing_entry = statistics.mean(losing_entries)

    def _find_top_positions(self, positions: Dict, report: StrategyReport, top_n: int = 5):
        """Find top winning and losing positions."""
        sorted_by_pnl = sorted(
            positions.values(),
            key=lambda p: p.get('realized_pnl', 0) or p.get('unrealized_pnl', 0),
            reverse=True
        )

        # Top winners
        for pos in sorted_by_pnl[:top_n]:
            pnl = pos.get('realized_pnl', 0) or pos.get('unrealized_pnl', 0)
            if pnl > 0:
                report.top_winners.append({
                    'market': pos.get('market_slug', '')[:50] or pos.get('market_title', '')[:50],
                    'outcome': pos.get('outcome'),
                    'entry_price': pos.get('avg_entry_price', 0),
                    'cost': pos.get('total_cost', 0),
                    'pnl': pnl,
                    'roi_pct': pos.get('roi_pct', 0),
                    'is_resolved': pos.get('is_resolved', False)
                })

        # Top losers
        for pos in reversed(sorted_by_pnl[-top_n:]):
            pnl = pos.get('realized_pnl', 0) or pos.get('unrealized_pnl', 0)
            if pnl < 0:
                report.top_losers.append({
                    'market': pos.get('market_slug', '')[:50] or pos.get('market_title', '')[:50],
                    'outcome': pos.get('outcome'),
                    'entry_price': pos.get('avg_entry_price', 0),
                    'cost': pos.get('total_cost', 0),
                    'pnl': pnl,
                    'roi_pct': pos.get('roi_pct', 0),
                    'is_resolved': pos.get('is_resolved', False)
                })

    def _detect_suspicious_patterns(self, report: StrategyReport):
        """Detect suspicious patterns and data quality issues."""
        # Suspicious patterns
        if report.win_rate >= 100 and report.total_resolved >= 10:
            report.suspicious_patterns.append(
                f"100% win rate with {report.total_resolved} resolved positions is statistically unusual"
            )

        if report.win_rate >= 95 and report.total_resolved >= 20:
            report.suspicious_patterns.append(
                f"{report.win_rate:.1f}% win rate over {report.total_resolved} positions is exceptional"
            )

        if report.profit_factor == float('inf'):
            report.suspicious_patterns.append(
                "No losing positions - either perfect trader or data issue"
            )

        if report.avg_position > 1_000_000:
            report.suspicious_patterns.append(
                f"Very large avg position size (${report.avg_position:,.0f}) - whale trader"
            )

        # Check for single-market concentration
        if report.categories:
            top_category = max(report.categories.values(), key=lambda c: c.position_count)
            if top_category.position_count / report.total_positions > 0.8:
                report.suspicious_patterns.append(
                    f"High concentration in {top_category.name} ({top_category.position_count}/{report.total_positions} positions)"
                )

        # Positive signals
        if report.unique_markets >= 10:
            report.positive_signals.append(
                f"Diversified across {report.unique_markets} markets"
            )

        if report.active_days >= 30:
            report.positive_signals.append(
                f"Consistent activity over {report.active_days} days"
            )

        if 0.4 <= report.avg_entry_price <= 0.6:
            report.positive_signals.append(
                "Enters at fair prices (near 50/50)"
            )

        if report.position_stddev > 0 and report.position_stddev / report.avg_position < 0.5:
            report.positive_signals.append(
                "Consistent position sizing"
            )

        if len(report.categories) >= 3:
            report.positive_signals.append(
                f"Trades multiple categories ({len(report.categories)})"
            )

        # Data quality issues
        if report.total_trades == 0:
            report.data_quality_issues.append("No trade data available")

        if report.total_resolved == 0 and report.total_positions > 0:
            report.data_quality_issues.append("No resolved positions - all trades are open")

        if report.total_volume == 0 and report.total_trades > 0:
            report.data_quality_issues.append("Volume data missing")

    def _detect_strategy_type(self, report: StrategyReport):
        """Detect likely trading strategy based on patterns."""
        indicators = {
            'market_maker': 0,
            'momentum': 0,
            'contrarian': 0,
            'whale': 0,
            'diversified': 0,
            'specialist': 0,
            'perfect_trader': 0,
        }

        # Market maker: high volume, many trades, both buys and sells
        if report.total_trades > 500 and report.total_buys > 0 and report.total_sells > 0:
            sell_ratio = report.total_sells / report.total_buys if report.total_buys > 0 else 0
            if 0.3 <= sell_ratio <= 3.0:
                indicators['market_maker'] += 3

        # Momentum: buys when price already high
        if report.buys_above_50 > report.buys_below_50 * 2:
            indicators['momentum'] += 2

        # Contrarian: buys when price low
        if report.buys_below_50 > report.buys_above_50 * 2:
            indicators['contrarian'] += 2

        # Whale: very large positions
        if report.avg_position > 500_000:
            indicators['whale'] += 3
        elif report.avg_position > 100_000:
            indicators['whale'] += 1

        # Diversified: many markets, multiple categories
        if report.unique_markets >= 20 and len(report.categories) >= 4:
            indicators['diversified'] += 2

        # Specialist: focused on one category
        if report.categories:
            top_cat = max(report.categories.values(), key=lambda c: c.position_count)
            if top_cat.position_count / max(report.total_positions, 1) > 0.7:
                indicators['specialist'] += 2

        # Perfect trader: unrealistic win rate
        if report.win_rate >= 95 and report.total_resolved >= 10:
            indicators['perfect_trader'] += 3

        # Find top indicator
        top_type = max(indicators.items(), key=lambda x: x[1])

        if top_type[1] >= 2:
            report.strategy_type = top_type[0].replace('_', ' ').title()
            report.strategy_confidence = 'HIGH' if top_type[1] >= 3 else 'MEDIUM'
        else:
            report.strategy_type = 'Mixed / Unclear'
            report.strategy_confidence = 'LOW'

    def print_report(self, report: StrategyReport):
        """Print formatted strategy report."""
        print("=" * 70)
        print(f"DEEP STRATEGY ANALYSIS")
        print("=" * 70)

        # Identity
        print(f"\nAddress: {report.address}")
        if report.name:
            print(f"Name:    {report.name}")
        if report.pseudonym:
            print(f"Handle:  @{report.pseudonym}")

        # Strategy Classification
        print(f"\n--- STRATEGY TYPE ---")
        print(f"Detected:   {report.strategy_type}")
        print(f"Confidence: {report.strategy_confidence}")

        # Win/Loss Breakdown
        print(f"\n--- WIN/LOSS BREAKDOWN ---")
        print(f"Total Positions: {report.total_positions}")
        print(f"  Resolved:      {report.total_resolved}")
        print(f"  Wins:          {report.wins} ({report.win_rate:.1f}%)")
        print(f"  Losses:        {report.losses}")
        print(f"\nAvg Win Size:  ${report.avg_win_size:,.2f}")
        print(f"Avg Loss Size: ${report.avg_loss_size:,.2f}")
        if report.profit_factor == float('inf'):
            print(f"Profit Factor: INF (no losses)")
        else:
            print(f"Profit Factor: {report.profit_factor:.2f}")

        # P&L Summary
        print(f"\n--- P&L SUMMARY ---")
        print(f"Total P&L:    ${report.total_pnl:,.2f}")
        print(f"  Realized:   ${report.realized_pnl:,.2f}")
        print(f"  Unrealized: ${report.unrealized_pnl:,.2f}")
        print(f"Total Volume: ${report.total_volume:,.2f}")

        # Market Categories
        print(f"\n--- MARKET CATEGORIES ---")
        if report.categories:
            print(f"{'Category':<15} | {'Positions':>9} | {'Win Rate':>8} | {'P&L':>15}")
            print("-" * 55)
            sorted_cats = sorted(
                report.categories.values(),
                key=lambda c: c.total_pnl,
                reverse=True
            )
            for cat in sorted_cats:
                print(f"{cat.name:<15} | {cat.position_count:>9} | {cat.win_rate:>7.1f}% | ${cat.total_pnl:>13,.0f}")
        else:
            print("  No category data available")

        # Position Sizing
        print(f"\n--- POSITION SIZING ---")
        print(f"Total Trades: {report.total_trades}")
        print(f"Avg Position: ${report.avg_position:,.2f}")
        print(f"Median:       ${report.median_position:,.2f}")
        print(f"Max:          ${report.max_position:,.2f}")
        print(f"Min:          ${report.min_position:,.2f}")
        print(f"Std Dev:      ${report.position_stddev:,.2f}")

        # Entry Quality
        print(f"\n--- ENTRY QUALITY ---")
        print(f"Avg Entry Price: {report.avg_entry_price:.3f}")
        if report.avg_winning_entry > 0:
            print(f"Avg Winner Entry: {report.avg_winning_entry:.3f}")
        if report.avg_losing_entry > 0:
            print(f"Avg Loser Entry:  {report.avg_losing_entry:.3f}")

        # Price Behavior
        print(f"\n--- PRICE BEHAVIOR ---")
        if report.total_buys > 0:
            below_pct = report.buys_below_50 / report.total_buys * 100
            above_pct = report.buys_above_50 / report.total_buys * 100
            print(f"Buys when price < 0.5: {report.buys_below_50} ({below_pct:.1f}%)")
            print(f"Buys when price >= 0.5: {report.buys_above_50} ({above_pct:.1f}%)")

        # Top Positions
        if report.top_winners:
            print(f"\n--- TOP WINNING POSITIONS ---")
            for i, pos in enumerate(report.top_winners[:5], 1):
                status = "CLOSED" if pos['is_resolved'] else "OPEN"
                print(f"{i}. {pos['market']}")
                print(f"   {pos['outcome']} @ {pos['entry_price']:.2f} | "
                      f"Cost: ${pos['cost']:,.0f} | "
                      f"P&L: ${pos['pnl']:,.0f} ({pos['roi_pct']:.1f}%) [{status}]")

        if report.top_losers:
            print(f"\n--- TOP LOSING POSITIONS ---")
            for i, pos in enumerate(report.top_losers[:5], 1):
                status = "CLOSED" if pos['is_resolved'] else "OPEN"
                print(f"{i}. {pos['market']}")
                print(f"   {pos['outcome']} @ {pos['entry_price']:.2f} | "
                      f"Cost: ${pos['cost']:,.0f} | "
                      f"P&L: ${pos['pnl']:,.0f} ({pos['roi_pct']:.1f}%) [{status}]")

        # Suspicious Patterns
        if report.suspicious_patterns:
            print(f"\n--- SUSPICIOUS PATTERNS ---")
            for pattern in report.suspicious_patterns:
                print(f"  ! {pattern}")

        # Positive Signals
        if report.positive_signals:
            print(f"\n--- POSITIVE SIGNALS ---")
            for signal in report.positive_signals:
                print(f"  + {signal}")

        # Data Quality
        if report.data_quality_issues:
            print(f"\n--- DATA QUALITY ISSUES ---")
            for issue in report.data_quality_issues:
                print(f"  ? {issue}")

        # Activity
        print(f"\n--- ACTIVITY ---")
        print(f"Unique Markets: {report.unique_markets}")
        print(f"Active Days:    {report.active_days}")
        if report.first_trade_date:
            print(f"First Trade:    {report.first_trade_date[:19]}")
        if report.last_trade_date:
            print(f"Last Trade:     {report.last_trade_date[:19]}")

        print("\n" + "=" * 70)
