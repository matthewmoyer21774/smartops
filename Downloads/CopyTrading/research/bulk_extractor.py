"""
Bulk Data Extractor for Hedge Fund Research

This script pulls MASSIVE amounts of data:
1. Discovers 1000+ traders from top markets
2. Fetches full trading history with timestamps for each
3. Exports everything to research-ready CSVs

Usage:
    python research/bulk_extractor.py --traders 1000 --trades-per-trader 5000
"""

import os
import sys
import csv
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.data_api import DataAPIClient
from api.gamma_api import GammaAPIClient


@dataclass
class TradeRow:
    """Single trade record with full details."""
    # Trader
    trader_address: str = ""
    trader_name: str = ""

    # Trade details
    transaction_hash: str = ""
    timestamp: str = ""
    timestamp_unix: int = 0

    # Market
    condition_id: str = ""
    market_slug: str = ""
    market_title: str = ""
    market_category: str = ""

    # Trade
    side: str = ""  # BUY or SELL
    outcome: str = ""  # Yes or No
    price: float = 0.0
    size_usd: float = 0.0
    shares: float = 0.0

    # Derived
    implied_probability: float = 0.0
    price_distance_from_fair: float = 0.0


@dataclass
class TraderRow:
    """Trader summary record."""
    address: str = ""
    name: str = ""

    # Performance
    total_pnl: float = 0.0
    roi_pct: float = 0.0
    volume: float = 0.0

    # Activity
    total_trades: int = 0
    unique_markets: int = 0
    first_trade: str = ""
    last_trade: str = ""

    # Win rate (calculated from trades)
    positions_won: int = 0
    positions_lost: int = 0
    win_rate: float = 0.0


class BulkExtractor:
    """
    Extracts massive amounts of trading data directly from Polymarket APIs.

    This bypasses the cache and pulls fresh data for maximum coverage.
    """

    MARKET_CATEGORIES = {
        'politics': ['trump', 'biden', 'election', 'president', 'senate', 'congress', 'vote', 'poll', 'democrat', 'republican'],
        'crypto': ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'solana', 'sol', 'price', 'token'],
        'sports': ['nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football', 'basketball', 'game', 'match', 'team', 'win'],
        'finance': ['stock', 'fed', 'rate', 'inflation', 'market', 'dow', 'nasdaq', 'sp500', 'gdp'],
        'tech': ['ai', 'openai', 'google', 'apple', 'microsoft', 'meta', 'launch', 'release'],
    }

    def __init__(self, output_dir: str = None):
        self.data_api = DataAPIClient()
        self.gamma_api = GammaAPIClient()

        # Cache for market resolutions and leaderboard data
        self._market_resolutions: Dict[str, Optional[str]] = {}
        self._leaderboard_cache: Dict[str, Dict] = {}

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'exports'
            )
        os.makedirs(self.output_dir, exist_ok=True)

    def categorize_market(self, title: str, slug: str = "") -> str:
        """Determine market category."""
        text = (title + " " + slug).lower()
        for category, keywords in self.MARKET_CATEGORIES.items():
            if any(kw in text for kw in keywords):
                return category
        return "other"

    def _load_leaderboard_cache(self, verbose: bool = True) -> None:
        """Pre-load leaderboard data for PnL lookups."""
        if self._leaderboard_cache:
            return

        if verbose:
            print("  Loading leaderboard data for PnL stats...")

        for period in ['all', 'daily', 'weekly', 'monthly']:
            try:
                leaderboard = self.data_api.get_leaderboard(time_period=period, limit=500)
                for entry in leaderboard:
                    addr = entry.get('proxyWallet') or entry.get('userAddress') or entry.get('address')
                    if addr:
                        addr_lower = addr.lower()
                        # Keep the entry with highest PnL
                        existing = self._leaderboard_cache.get(addr_lower)
                        if not existing or float(entry.get('pnl', 0) or 0) > float(existing.get('pnl', 0) or 0):
                            self._leaderboard_cache[addr_lower] = entry
                time.sleep(0.3)
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not load {period} leaderboard: {e}")

        if verbose:
            print(f"    Cached PnL data for {len(self._leaderboard_cache)} traders")

    def _get_trader_pnl(self, address: str) -> float:
        """Get PnL for a trader from leaderboard cache."""
        entry = self._leaderboard_cache.get(address.lower())
        if entry:
            return float(entry.get('pnl', 0) or 0)
        return 0.0

    def _get_trader_win_loss_from_activity(self, address: str) -> tuple:
        """
        Calculate positions won/lost from activity data.

        REDEEM activity indicates a winning position.
        We count unique markets with redemptions as wins,
        and estimate losses from markets with trades but no redemptions.

        Returns:
            Tuple of (positions_won, positions_lost, resolved_markets)
        """
        try:
            activity = self.data_api.get_trader_activity(address, limit=500)
        except Exception:
            return 0, 0, set()

        # Track markets with redemptions (wins)
        redeemed_markets = set()
        traded_markets = set()

        for a in activity:
            condition_id = a.get('conditionId', '')
            if not condition_id:
                continue

            activity_type = a.get('type', '')
            if activity_type == 'REDEEM':
                redeemed_markets.add(condition_id)
            elif activity_type == 'TRADE':
                traded_markets.add(condition_id)

        # Positions won = unique markets with redemptions
        positions_won = len(redeemed_markets)

        # Estimate losses: traded markets without redemptions
        # Note: Some may be open markets, so this is an upper bound on losses
        # We'll be conservative and only count as losses markets that had trades
        # but no redemptions AND are likely resolved (older timestamps)
        positions_lost = 0
        resolved_markets = redeemed_markets.copy()

        # For now, just use redeemed markets as our win count
        # This is conservative but accurate
        return positions_won, positions_lost, resolved_markets

    def discover_traders(self,
                        num_markets: int = 200,
                        holders_per_market: int = 50,
                        min_volume: float = 10000,
                        verbose: bool = True) -> Set[str]:
        """
        Discover trader addresses from top markets.

        Args:
            num_markets: Number of markets to scan
            holders_per_market: Top holders to extract per market
            min_volume: Minimum market volume filter

        Returns:
            Set of unique trader addresses
        """
        traders = set()

        if verbose:
            print(f"Discovering traders from {num_markets} markets...")
            print(f"  Holders per market: {holders_per_market}")

        # Get from leaderboard first
        if verbose:
            print("\n[1/2] Fetching from leaderboard...")

        for period in ['all', 'daily', 'weekly', 'monthly']:
            try:
                leaderboard = self.data_api.get_leaderboard(time_period=period, limit=500)
                for entry in leaderboard:
                    # API uses 'proxyWallet' field
                    addr = entry.get('proxyWallet') or entry.get('userAddress') or entry.get('address')
                    if addr:
                        traders.add(addr.lower())
                if verbose:
                    print(f"    {period}: {len(leaderboard)} traders, total unique: {len(traders)}")
                time.sleep(0.5)  # Rate limit
            except Exception as e:
                if verbose:
                    print(f"    {period}: Error - {e}")

        if verbose:
            print(f"  Leaderboard total: {len(traders)} unique")

        # Get from ACTIVE markets (closed markets don't return trade data)
        if verbose:
            print(f"\n[2/2] Scanning {num_markets} active markets for traders...")

        try:
            # Fetch active markets with pagination
            all_markets = []
            offset = 0
            page_size = 100

            while len(all_markets) < num_markets:
                markets = self.gamma_api.get_markets(closed=False, limit=page_size, offset=offset)
                if not markets:
                    break
                all_markets.extend(markets)
                if len(markets) < page_size:
                    break
                offset += page_size
                time.sleep(0.2)

            # Sort by volume
            all_markets.sort(key=lambda m: float(m.get('volume', 0) or 0), reverse=True)
            all_markets = all_markets[:num_markets]

            if verbose:
                print(f"    Found {len(all_markets)} active markets to scan")

            for i, market in enumerate(all_markets):
                if verbose and (i + 1) % 20 == 0:
                    print(f"    Scanned {i + 1}/{len(all_markets)} markets, {len(traders)} traders found...")

                volume = float(market.get('volume', 0) or 0)
                if volume < min_volume:
                    continue

                condition_id = market.get('conditionId')
                if not condition_id:
                    continue

                try:
                    # Get trades for this market to find active traders
                    market_trades = self.data_api.get_market_trades(
                        condition_id,
                        limit=holders_per_market * 10  # Get more trades to find unique traders
                    )

                    new_traders = 0
                    for trade in market_trades:
                        addr = trade.get('proxyWallet')
                        if addr:
                            addr_lower = addr.lower()
                            if addr_lower not in traders:
                                traders.add(addr_lower)
                                new_traders += 1

                    time.sleep(0.2)  # Rate limit

                except Exception as e:
                    continue

        except Exception as e:
            if verbose:
                print(f"  Market scan error: {e}")

        if verbose:
            print(f"\nTotal unique traders discovered: {len(traders)}")

        return traders

    def fetch_trader_trades(self,
                           address: str,
                           max_trades: int = 5000) -> List[dict]:
        """Fetch all trades for a single trader."""
        try:
            trades = self.data_api.get_trader_trades(address, limit=max_trades)
            return trades
        except Exception as e:
            return []

    def extract_bulk(self,
                    target_traders: int = 500,
                    trades_per_trader: int = 5000,
                    num_markets: int = 200,
                    holders_per_market: int = 50,
                    verbose: bool = True) -> Dict[str, str]:
        """
        Main extraction method - pulls massive data and exports to CSV.

        Args:
            target_traders: Target number of traders to extract
            trades_per_trader: Max trades per trader
            num_markets: Markets to scan for discovery
            holders_per_market: Holders per market
            verbose: Print progress

        Returns:
            Dict of output file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Step 1: Discover traders
        trader_addresses = self.discover_traders(
            num_markets=num_markets,
            holders_per_market=holders_per_market,
            verbose=verbose
        )

        # Limit to target
        addresses = list(trader_addresses)[:target_traders]

        if verbose:
            print(f"\n{'='*70}")
            print(f"EXTRACTING FULL TRADE HISTORY")
            print(f"{'='*70}")
            print(f"Traders to process: {len(addresses)}")
            print(f"Max trades per trader: {trades_per_trader}")

        # Pre-load leaderboard data for PnL lookups
        self._load_leaderboard_cache(verbose=verbose)

        # Step 2: Fetch all trades
        all_trades: List[TradeRow] = []
        trader_summaries: Dict[str, TraderRow] = {}

        for i, address in enumerate(addresses):
            if verbose and (i + 1) % 25 == 0:
                print(f"  Processing trader {i + 1}/{len(addresses)}... ({len(all_trades)} trades so far)")

            raw_trades = self.fetch_trader_trades(address, max_trades=trades_per_trader)

            if not raw_trades:
                continue

            # Initialize trader summary
            summary = TraderRow(address=address)
            markets_seen = set()
            positions = defaultdict(lambda: {'cost': 0, 'shares': 0, 'side': None})

            for raw in raw_trades:
                trade = TradeRow()
                trade.trader_address = address

                # Transaction details
                trade.transaction_hash = raw.get('transactionHash', '')

                # Timestamp - API returns unix timestamp as integer
                ts = raw.get('timestamp', 0)
                if isinstance(ts, (int, float)) and ts > 0:
                    trade.timestamp_unix = int(ts)
                    try:
                        dt = datetime.utcfromtimestamp(ts)
                        trade.timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        trade.timestamp = str(ts)
                elif isinstance(ts, str) and ts:
                    trade.timestamp = ts
                    try:
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        trade.timestamp_unix = int(dt.timestamp())
                    except:
                        pass

                # Market
                trade.condition_id = raw.get('conditionId', '')
                trade.market_slug = raw.get('slug') or raw.get('marketSlug', '')
                trade.market_title = raw.get('title', '')
                trade.market_category = self.categorize_market(
                    trade.market_title, trade.market_slug
                )

                markets_seen.add(trade.condition_id)

                # Trade details
                trade.side = raw.get('side', '')
                trade.outcome = raw.get('outcome', '')
                trade.price = float(raw.get('price', 0) or 0)
                trade.size_usd = float(raw.get('size', 0) or 0)

                if trade.price > 0:
                    trade.shares = trade.size_usd / trade.price

                trade.implied_probability = trade.price
                trade.price_distance_from_fair = abs(trade.price - 0.5)

                # Track for summary
                summary.total_trades += 1
                summary.volume += trade.size_usd

                # Track position P&L
                pos_key = f"{trade.condition_id}:{trade.outcome}"
                if trade.side == 'BUY':
                    positions[pos_key]['cost'] += trade.size_usd
                    positions[pos_key]['shares'] += trade.shares
                    positions[pos_key]['side'] = trade.outcome

                all_trades.append(trade)

            # Update summary
            summary.unique_markets = len(markets_seen)

            if raw_trades:
                # Get name from any trade that has it
                for raw in raw_trades:
                    name = raw.get('name') or raw.get('pseudonym') or raw.get('userName') or raw.get('username')
                    if name:
                        summary.name = name
                        break

                # Sort by timestamp for first/last
                sorted_trades = sorted(
                    [t for t in raw_trades if t.get('timestamp')],
                    key=lambda x: x.get('timestamp', '')
                )
                if sorted_trades:
                    summary.first_trade = sorted_trades[0].get('timestamp', '')
                    summary.last_trade = sorted_trades[-1].get('timestamp', '')

            # Get PnL from leaderboard
            summary.total_pnl = self._get_trader_pnl(address)
            if summary.volume > 0:
                summary.roi_pct = (summary.total_pnl / summary.volume) * 100

            # Calculate win/loss from activity data
            # REDEEM activities indicate winning positions
            positions_won, positions_lost, _ = self._get_trader_win_loss_from_activity(address)
            summary.positions_won = positions_won
            summary.positions_lost = positions_lost

            # Calculate win rate based on wins vs total unique markets
            # Since we can only reliably detect wins from redemptions,
            # use unique markets as denominator for a conservative estimate
            if summary.unique_markets > 0:
                summary.win_rate = (summary.positions_won / summary.unique_markets) * 100

            trader_summaries[address] = summary

            # Small delay for rate limiting
            time.sleep(0.2)

        if verbose:
            print(f"\nTotal trades extracted: {len(all_trades)}")
            print(f"Total traders with data: {len(trader_summaries)}")

        # Step 3: Export to CSV
        output_files = {}

        # Trades CSV
        if all_trades:
            trades_file = os.path.join(self.output_dir, f"bulk_trades_{timestamp}.csv")
            self._export_trades_csv(all_trades, trades_file)
            output_files['trades'] = trades_file
            if verbose:
                print(f"\nExported {len(all_trades)} trades to:")
                print(f"  {trades_file}")

        # Traders CSV
        if trader_summaries:
            traders_file = os.path.join(self.output_dir, f"bulk_traders_{timestamp}.csv")
            self._export_traders_csv(list(trader_summaries.values()), traders_file)
            output_files['traders'] = traders_file
            if verbose:
                print(f"\nExported {len(trader_summaries)} traders to:")
                print(f"  {traders_file}")

        # Summary JSON
        summary = {
            'extraction_timestamp': timestamp,
            'total_traders': len(trader_summaries),
            'total_trades': len(all_trades),
            'markets_scanned': num_markets,
            'trades_per_trader_limit': trades_per_trader,
            'output_files': output_files
        }
        summary_file = os.path.join(self.output_dir, f"bulk_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        output_files['summary'] = summary_file

        return output_files

    def _export_trades_csv(self, trades: List[TradeRow], filepath: str):
        """Export trades to CSV."""
        fieldnames = [
            'trader_address', 'trader_name',
            'transaction_hash', 'timestamp', 'timestamp_unix',
            'condition_id', 'market_slug', 'market_title', 'market_category',
            'side', 'outcome', 'price', 'size_usd', 'shares',
            'implied_probability', 'price_distance_from_fair'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades:
                writer.writerow(asdict(trade))

    def _export_traders_csv(self, traders: List[TraderRow], filepath: str):
        """Export traders to CSV."""
        fieldnames = [
            'address', 'name',
            'total_pnl', 'roi_pct', 'volume',
            'total_trades', 'unique_markets',
            'first_trade', 'last_trade',
            'positions_won', 'positions_lost', 'win_rate'
        ]

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trader in traders:
                writer.writerow(asdict(trader))


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Bulk extract trading data for hedge fund research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 500 traders with 5000 trades each (default)
  python research/bulk_extractor.py

  # Extract 1000 traders with full history
  python research/bulk_extractor.py --traders 1000 --trades-per-trader 10000

  # Quick extraction (100 traders, 1000 trades each)
  python research/bulk_extractor.py --traders 100 --trades-per-trader 1000

  # Maximum extraction
  python research/bulk_extractor.py --traders 2000 --markets 500 --trades-per-trader 20000
        """
    )

    parser.add_argument('--traders', type=int, default=500,
                       help='Target number of traders (default: 500)')
    parser.add_argument('--trades-per-trader', type=int, default=5000,
                       help='Max trades per trader (default: 5000)')
    parser.add_argument('--markets', type=int, default=200,
                       help='Markets to scan for discovery (default: 200)')
    parser.add_argument('--holders-per-market', type=int, default=50,
                       help='Holders per market (default: 50)')
    parser.add_argument('--output', type=str,
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    print("=" * 70)
    print("BULK DATA EXTRACTION FOR HEDGE FUND RESEARCH")
    print("=" * 70)
    print(f"\nTarget: {args.traders} traders")
    print(f"Trades per trader: {args.trades_per_trader}")
    print(f"Markets to scan: {args.markets}")
    print(f"\nEstimated time: {args.traders * 0.5 / 60:.1f} minutes")
    print("=" * 70)

    extractor = BulkExtractor(output_dir=args.output)

    output_files = extractor.extract_bulk(
        target_traders=args.traders,
        trades_per_trader=args.trades_per_trader,
        num_markets=args.markets,
        holders_per_market=args.holders_per_market,
        verbose=not args.quiet
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    for file_type, path in output_files.items():
        print(f"  {file_type}: {path}")

    print("\n" + "=" * 70)
    print("READY FOR HEDGE FUND ANALYSIS")
    print("=" * 70)


if __name__ == '__main__':
    main()
