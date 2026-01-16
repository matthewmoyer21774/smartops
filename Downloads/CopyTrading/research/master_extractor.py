"""
Master Data Extractor for Hedge Fund Research

Extracts comprehensive trading data from smart money traders
for analysis and strategy development.

Creates multiple CSV files:
1. trades_master.csv - Every single trade with full details
2. positions_master.csv - Position-level aggregates with timing
3. traders_master.csv - Trader profiles with strategy classification
4. market_performance.csv - Performance by market/category
"""

import os
import sys
import csv
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.data_api import DataAPIClient
from api.gamma_api import GammaAPIClient
from cache.file_cache import TraderCache
from models.trader import Trader, TraderStats
from models.position import Position
from models.trade import Trade
from strategies.classifier import StrategyClassifier


@dataclass
class TradeRecord:
    """Flattened trade record for CSV export."""
    # Trade identification
    trade_id: str = ""
    transaction_hash: str = ""

    # Trader info
    trader_address: str = ""
    trader_name: str = ""
    trader_pseudonym: str = ""

    # Market info
    condition_id: str = ""
    market_slug: str = ""
    market_title: str = ""
    market_category: str = ""

    # Trade details
    timestamp: str = ""
    timestamp_unix: int = 0
    side: str = ""  # BUY or SELL
    outcome: str = ""  # Yes or No
    price: float = 0.0
    size_usd: float = 0.0
    shares: float = 0.0

    # Derived metrics
    price_distance_from_fair: float = 0.0  # How far from 0.5
    implied_probability: float = 0.0  # Price as probability

    # Resolution info (if resolved)
    is_resolved: bool = False
    resolved_outcome: str = ""
    resolved_price: float = 0.0
    was_correct: bool = False

    # Position context
    position_key: str = ""
    trade_number_in_position: int = 0
    cumulative_shares: float = 0.0
    cumulative_cost: float = 0.0
    avg_price_at_trade: float = 0.0


@dataclass
class PositionRecord:
    """Flattened position record for CSV export."""
    # Identification
    position_key: str = ""
    trader_address: str = ""
    trader_name: str = ""

    # Market info
    condition_id: str = ""
    market_slug: str = ""
    market_title: str = ""
    market_category: str = ""
    outcome: str = ""

    # Entry metrics
    first_trade_timestamp: str = ""
    last_trade_timestamp: str = ""
    num_trades: int = 0
    total_shares: float = 0.0
    total_cost: float = 0.0
    avg_entry_price: float = 0.0

    # Entry quality
    entry_price_distance: float = 0.0  # Distance from 0.5
    entry_certainty: float = 0.0  # Distance from 0 or 1

    # Current/Resolution state
    is_resolved: bool = False
    resolved_price: float = 0.0
    current_price: float = 0.0
    current_value: float = 0.0

    # P&L metrics
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    roi_pct: float = 0.0

    # Timing metrics
    time_held_hours: float = 0.0
    time_held_days: float = 0.0

    # Outcome
    won: bool = False
    edge_captured: float = 0.0  # How much of potential profit captured


@dataclass
class TraderRecord:
    """Comprehensive trader record for CSV export."""
    # Identity
    address: str = ""
    name: str = ""
    pseudonym: str = ""

    # Strategy classification
    primary_strategy: str = ""
    strategy_confidence: float = 0.0
    secondary_strategy: str = ""
    risk_level: str = ""
    is_copyable: bool = False

    # Overall performance
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    roi_pct: float = 0.0

    # Volume metrics
    total_volume: float = 0.0
    total_trades: int = 0
    unique_markets: int = 0

    # Win/Loss
    win_rate: float = 0.0
    winning_positions: int = 0
    losing_positions: int = 0
    total_resolved: int = 0

    # Position sizing
    avg_position_size: float = 0.0
    max_position_size: float = 0.0
    max_drawdown: float = 0.0

    # Activity metrics
    first_trade_date: str = ""
    last_trade_date: str = ""
    active_days: int = 0
    trades_per_day: float = 0.0

    # Category preferences
    top_category: str = ""
    category_concentration: float = 0.0

    # Entry patterns
    avg_entry_price: float = 0.0
    pct_entries_above_fair: float = 0.0
    pct_entries_extreme: float = 0.0

    # Timing metrics
    avg_time_held_days: float = 0.0

    # Detailed signals (JSON string)
    strategy_signals_json: str = ""


class MasterDataExtractor:
    """
    Master extractor for comprehensive trading data.

    Pulls all available data and exports to research-ready CSV files.
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

    def __init__(self, output_dir: str = None):
        self.cache = TraderCache()
        self.data_api = DataAPIClient()
        self.gamma_api = GammaAPIClient()
        self.classifier = StrategyClassifier()

        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'research', 'exports'
            )
        os.makedirs(self.output_dir, exist_ok=True)

        # Market metadata cache
        self.market_cache: Dict[str, dict] = {}

    def categorize_market(self, title: str, slug: str = "") -> str:
        """Determine market category from title and slug."""
        text = (title + " " + slug).lower()

        for category, keywords in self.MARKET_CATEGORIES.items():
            if any(kw in text for kw in keywords):
                return category

        return "other"

    def extract_all(self,
                    trader_addresses: List[str] = None,
                    limit: int = None,
                    include_trades: bool = True,
                    verbose: bool = True) -> Dict[str, str]:
        """
        Extract all data and export to CSV files.

        Args:
            trader_addresses: Specific addresses to extract (None = all cached)
            limit: Max traders to process
            include_trades: Whether to extract individual trades (can be slow)
            verbose: Print progress

        Returns:
            Dict of output file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load traders
        if trader_addresses:
            traders_data = []
            for addr in trader_addresses:
                data = self.cache.load_trader(addr)
                if data:
                    traders_data.append(data)
        else:
            traders_data = self.cache.load_all_traders()

        if limit:
            traders_data = traders_data[:limit]

        if verbose:
            print(f"Extracting data for {len(traders_data)} traders...")

        # Process each trader
        all_trades: List[TradeRecord] = []
        all_positions: List[PositionRecord] = []
        all_traders: List[TraderRecord] = []

        for i, trader_data in enumerate(traders_data):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processing trader {i + 1}/{len(traders_data)}...")

            # Convert to Trader object
            trader = self._dict_to_trader(trader_data) if isinstance(trader_data, dict) else trader_data

            # Extract trader summary
            trader_record = self._extract_trader_record(trader)
            all_traders.append(trader_record)

            # Extract positions
            position_records = self._extract_position_records(trader)
            all_positions.extend(position_records)

            # Extract trades (if enabled)
            if include_trades:
                trade_records = self._extract_trade_records(trader)
                all_trades.append(trade_records)

        # Flatten trades list
        if include_trades:
            all_trades = [t for trades in all_trades for t in trades]

        # Export to CSV
        output_files = {}

        # Trades CSV
        if include_trades and all_trades:
            trades_file = os.path.join(self.output_dir, f"trades_master_{timestamp}.csv")
            self._export_to_csv(all_trades, trades_file)
            output_files['trades'] = trades_file
            if verbose:
                print(f"Exported {len(all_trades)} trades to {trades_file}")

        # Positions CSV
        if all_positions:
            positions_file = os.path.join(self.output_dir, f"positions_master_{timestamp}.csv")
            self._export_to_csv(all_positions, positions_file)
            output_files['positions'] = positions_file
            if verbose:
                print(f"Exported {len(all_positions)} positions to {positions_file}")

        # Traders CSV
        if all_traders:
            traders_file = os.path.join(self.output_dir, f"traders_master_{timestamp}.csv")
            self._export_to_csv(all_traders, traders_file)
            output_files['traders'] = traders_file
            if verbose:
                print(f"Exported {len(all_traders)} traders to {traders_file}")

        # Create summary report
        summary = self._create_summary_report(all_traders, all_positions, all_trades if include_trades else [])
        summary_file = os.path.join(self.output_dir, f"extraction_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        output_files['summary'] = summary_file

        if verbose:
            print(f"\nExtraction complete!")
            print(f"Summary saved to {summary_file}")

        return output_files

    def _extract_trader_record(self, trader: Trader) -> TraderRecord:
        """Extract comprehensive trader record."""
        record = TraderRecord()

        # Identity
        record.address = trader.address
        record.name = trader.name or ""
        record.pseudonym = trader.pseudonym or ""

        # Strategy classification
        try:
            profile = self.classifier.classify(trader)
            record.primary_strategy = profile.primary_strategy
            record.strategy_confidence = profile.primary_confidence
            record.secondary_strategy = profile.secondary_strategy or ""
            record.risk_level = profile.risk_level
            record.is_copyable = profile.copyable
            record.strategy_signals_json = json.dumps(profile.to_dict().get('signals', {}))
        except Exception as e:
            record.primary_strategy = "unknown"
            record.strategy_signals_json = "{}"

        # Performance metrics
        stats = trader.stats
        record.total_pnl = stats.total_pnl
        record.realized_pnl = stats.realized_pnl
        record.unrealized_pnl = stats.unrealized_pnl
        record.roi_pct = stats.roi_pct
        record.total_volume = stats.total_volume
        record.total_trades = stats.total_trades
        record.unique_markets = stats.unique_markets
        record.win_rate = stats.win_rate
        record.winning_positions = stats.winning_positions
        record.losing_positions = stats.losing_positions
        record.total_resolved = stats.total_resolved
        record.avg_position_size = stats.avg_position_size
        record.max_position_size = stats.max_position_size
        record.max_drawdown = stats.max_drawdown
        record.active_days = stats.active_days

        # Date fields
        if stats.first_trade_date:
            record.first_trade_date = stats.first_trade_date.isoformat() if hasattr(stats.first_trade_date, 'isoformat') else str(stats.first_trade_date)
        if stats.last_trade_date:
            record.last_trade_date = stats.last_trade_date.isoformat() if hasattr(stats.last_trade_date, 'isoformat') else str(stats.last_trade_date)

        # Calculate trades per day
        if record.active_days > 0:
            record.trades_per_day = record.total_trades / record.active_days

        # Analyze positions for patterns
        if trader.positions:
            entry_prices = []
            time_held_list = []
            categories = defaultdict(float)

            for pos in trader.positions.values():
                if pos.avg_entry_price > 0:
                    entry_prices.append(pos.avg_entry_price)

                # Category tracking
                cat = self.categorize_market(pos.market_title or "", pos.market_slug or "")
                categories[cat] += pos.total_cost

            if entry_prices:
                record.avg_entry_price = sum(entry_prices) / len(entry_prices)
                record.pct_entries_above_fair = sum(1 for p in entry_prices if p > 0.5) / len(entry_prices)
                record.pct_entries_extreme = sum(1 for p in entry_prices if p > 0.85 or p < 0.15) / len(entry_prices)

            # Top category
            if categories:
                sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
                record.top_category = sorted_cats[0][0]
                total_cost = sum(categories.values())
                if total_cost > 0:
                    record.category_concentration = sorted_cats[0][1] / total_cost

        return record

    def _extract_position_records(self, trader: Trader) -> List[PositionRecord]:
        """Extract all position records for a trader."""
        records = []

        for key, pos in trader.positions.items():
            record = PositionRecord()

            # Identification
            record.position_key = key
            record.trader_address = trader.address
            record.trader_name = trader.name or trader.pseudonym or ""

            # Market info
            record.condition_id = pos.condition_id
            record.market_slug = pos.market_slug or ""
            record.market_title = pos.market_title or ""
            record.market_category = self.categorize_market(pos.market_title or "", pos.market_slug or "")
            record.outcome = pos.outcome

            # Entry metrics
            record.num_trades = len(pos.trades) if pos.trades else 1
            record.total_shares = pos.total_shares
            record.total_cost = pos.total_cost
            record.avg_entry_price = pos.avg_entry_price

            # Calculate entry quality
            if pos.avg_entry_price > 0:
                record.entry_price_distance = abs(pos.avg_entry_price - 0.5)
                if pos.outcome == 'Yes':
                    record.entry_certainty = pos.avg_entry_price
                else:
                    record.entry_certainty = 1 - pos.avg_entry_price

            # Resolution state
            record.is_resolved = pos.is_resolved
            record.resolved_price = pos.resolved_price if pos.resolved_price is not None else 0.0
            record.current_price = pos.current_price
            record.current_value = pos.current_value

            # P&L
            record.realized_pnl = pos.realized_pnl
            record.unrealized_pnl = pos.unrealized_pnl
            record.roi_pct = pos.roi_pct

            # Outcome
            if pos.is_resolved:
                record.won = pos.realized_pnl > 0
                # Edge captured = actual profit / max possible profit
                max_profit = pos.total_shares - pos.total_cost
                if max_profit > 0 and pos.realized_pnl > 0:
                    record.edge_captured = min(1.0, pos.realized_pnl / max_profit)

            # Timing (from trades if available)
            if pos.trades:
                sorted_trades = sorted(pos.trades, key=lambda t: t.timestamp if t.timestamp else datetime.min)
                if sorted_trades:
                    first_ts = sorted_trades[0].timestamp
                    last_ts = sorted_trades[-1].timestamp
                    if first_ts:
                        record.first_trade_timestamp = first_ts.isoformat() if hasattr(first_ts, 'isoformat') else str(first_ts)
                    if last_ts:
                        record.last_trade_timestamp = last_ts.isoformat() if hasattr(last_ts, 'isoformat') else str(last_ts)
                    if first_ts and last_ts:
                        delta = last_ts - first_ts
                        record.time_held_hours = delta.total_seconds() / 3600
                        record.time_held_days = delta.total_seconds() / 86400

            records.append(record)

        return records

    def _extract_trade_records(self, trader: Trader) -> List[TradeRecord]:
        """Extract all individual trade records for a trader."""
        records = []

        # Group trades by position
        position_trades: Dict[str, List] = defaultdict(list)

        for pos in trader.positions.values():
            key = f"{pos.condition_id}:{pos.outcome}"
            if pos.trades:
                position_trades[key] = sorted(pos.trades, key=lambda t: t.timestamp if t.timestamp else datetime.min)

        # Process each position's trades
        for position_key, trades in position_trades.items():
            cumulative_shares = 0.0
            cumulative_cost = 0.0

            for i, trade in enumerate(trades):
                record = TradeRecord()

                # Trade identification
                record.trade_id = f"{trader.address}_{position_key}_{i}"
                record.transaction_hash = trade.transaction_hash or ""

                # Trader info
                record.trader_address = trader.address
                record.trader_name = trader.name or ""
                record.trader_pseudonym = trader.pseudonym or ""

                # Market info
                record.condition_id = trade.condition_id
                record.market_slug = trade.market_slug or ""
                record.market_title = trade.market_title or ""
                record.market_category = self.categorize_market(trade.market_title or "", trade.market_slug or "")

                # Trade details
                if trade.timestamp:
                    record.timestamp = trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp)
                    record.timestamp_unix = int(trade.timestamp.timestamp()) if hasattr(trade.timestamp, 'timestamp') else 0

                record.side = trade.side
                record.outcome = trade.outcome
                record.price = trade.price
                record.size_usd = trade.size
                record.shares = trade.shares

                # Derived metrics
                record.price_distance_from_fair = abs(trade.price - 0.5)
                record.implied_probability = trade.price

                # Resolution info
                record.resolved_price = trade.resolved_price if trade.resolved_price is not None else 0.0
                record.is_resolved = record.resolved_price in [0.0, 1.0]
                if record.is_resolved:
                    record.resolved_outcome = "Yes" if record.resolved_price == 1.0 else "No"
                    record.was_correct = (record.outcome == record.resolved_outcome)

                # Position context
                record.position_key = position_key
                record.trade_number_in_position = i + 1

                # Update cumulative values
                if trade.side == 'BUY':
                    cumulative_shares += trade.shares
                    cumulative_cost += trade.size
                else:
                    cumulative_shares -= trade.shares
                    cumulative_cost -= trade.size

                record.cumulative_shares = cumulative_shares
                record.cumulative_cost = cumulative_cost
                record.avg_price_at_trade = cumulative_cost / cumulative_shares if cumulative_shares > 0 else 0

                records.append(record)

        return records

    def _export_to_csv(self, records: List, filepath: str):
        """Export dataclass records to CSV."""
        if not records:
            return

        # Get field names from first record
        if hasattr(records[0], '__dataclass_fields__'):
            fieldnames = list(records[0].__dataclass_fields__.keys())
        else:
            fieldnames = list(asdict(records[0]).keys())

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                row = asdict(record) if hasattr(record, '__dataclass_fields__') else record
                writer.writerow(row)

    def _create_summary_report(self, traders: List[TraderRecord],
                               positions: List[PositionRecord],
                               trades: List[TradeRecord]) -> dict:
        """Create summary report of extraction."""
        summary = {
            'extraction_timestamp': datetime.now().isoformat(),
            'counts': {
                'traders': len(traders),
                'positions': len(positions),
                'trades': len(trades)
            },
            'trader_stats': {},
            'strategy_distribution': {},
            'category_distribution': {},
            'performance_summary': {}
        }

        if traders:
            # Strategy distribution
            strategies = defaultdict(int)
            for t in traders:
                strategies[t.primary_strategy] += 1
            summary['strategy_distribution'] = dict(strategies)

            # Aggregate stats
            total_pnl = sum(t.total_pnl for t in traders)
            avg_roi = sum(t.roi_pct for t in traders) / len(traders)
            avg_win_rate = sum(t.win_rate for t in traders) / len(traders)

            summary['trader_stats'] = {
                'total_pnl_all_traders': total_pnl,
                'avg_roi_pct': avg_roi,
                'avg_win_rate': avg_win_rate,
                'total_volume': sum(t.total_volume for t in traders),
                'total_trades': sum(t.total_trades for t in traders)
            }

            # Top performers
            by_pnl = sorted(traders, key=lambda t: t.total_pnl, reverse=True)[:10]
            summary['top_performers_by_pnl'] = [
                {'name': t.name or t.address[:16], 'pnl': t.total_pnl, 'roi': t.roi_pct}
                for t in by_pnl
            ]

            by_roi = sorted([t for t in traders if t.total_resolved >= 10],
                           key=lambda t: t.roi_pct, reverse=True)[:10]
            summary['top_performers_by_roi'] = [
                {'name': t.name or t.address[:16], 'pnl': t.total_pnl, 'roi': t.roi_pct}
                for t in by_roi
            ]

        if positions:
            # Category distribution
            categories = defaultdict(lambda: {'count': 0, 'volume': 0, 'pnl': 0})
            for p in positions:
                cat = p.market_category
                categories[cat]['count'] += 1
                categories[cat]['volume'] += p.total_cost
                categories[cat]['pnl'] += p.realized_pnl
            summary['category_distribution'] = dict(categories)

            # Position stats
            resolved = [p for p in positions if p.is_resolved]
            if resolved:
                wins = sum(1 for p in resolved if p.won)
                summary['performance_summary'] = {
                    'total_resolved_positions': len(resolved),
                    'winning_positions': wins,
                    'losing_positions': len(resolved) - wins,
                    'overall_win_rate': wins / len(resolved) * 100,
                    'avg_roi_on_resolved': sum(p.roi_pct for p in resolved) / len(resolved)
                }

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

        # Handle date fields
        if stats_data.get('first_trade_date'):
            try:
                trader.stats.first_trade_date = datetime.fromisoformat(stats_data['first_trade_date'].replace('Z', '+00:00'))
            except:
                pass
        if stats_data.get('last_trade_date'):
            try:
                trader.stats.last_trade_date = datetime.fromisoformat(stats_data['last_trade_date'].replace('Z', '+00:00'))
            except:
                pass

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

            # Load trades if available
            trades_data = pos_data.get('trades', [])
            for trade_data in trades_data:
                trade = Trade(
                    transaction_hash=trade_data.get('transaction_hash', ''),
                    condition_id=trade_data.get('condition_id', ''),
                    side=trade_data.get('side', ''),
                    outcome=trade_data.get('outcome', ''),
                    price=trade_data.get('price', 0),
                    size=trade_data.get('size', 0),
                    shares=trade_data.get('shares', 0),
                    market_slug=trade_data.get('market_slug', ''),
                    market_title=trade_data.get('market_title', ''),
                    resolved_price=trade_data.get('resolved_price')
                )
                # Parse timestamp
                ts = trade_data.get('timestamp')
                if ts:
                    try:
                        trade.timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    except:
                        pass
                pos.trades.append(trade)

            trader.positions[key] = pos

        return trader

    def extract_fresh_trades(self,
                            addresses: List[str],
                            max_trades_per_trader: int = 10000,
                            verbose: bool = True) -> str:
        """
        Fetch fresh trade data directly from API for specified addresses.

        This bypasses the cache and gets the most recent data.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_trades = []

        for i, address in enumerate(addresses):
            if verbose:
                print(f"Fetching trades for {address[:16]}... ({i+1}/{len(addresses)})")

            try:
                raw_trades = self.data_api.get_trader_trades(address, limit=max_trades_per_trader)

                for raw in raw_trades:
                    record = TradeRecord()
                    record.trader_address = address
                    record.transaction_hash = raw.get('transactionHash', '')
                    record.condition_id = raw.get('conditionId', '')
                    record.market_slug = raw.get('marketSlug', '')
                    record.market_title = raw.get('title', '')
                    record.market_category = self.categorize_market(raw.get('title', ''), raw.get('marketSlug', ''))

                    # Timestamp
                    ts = raw.get('timestamp')
                    if ts:
                        record.timestamp = ts
                        try:
                            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            record.timestamp_unix = int(dt.timestamp())
                        except:
                            pass

                    record.side = raw.get('side', '')
                    record.outcome = raw.get('outcome', '')
                    record.price = float(raw.get('price', 0))
                    record.size_usd = float(raw.get('size', 0))

                    # Calculate shares
                    if record.price > 0:
                        record.shares = record.size_usd / record.price

                    record.price_distance_from_fair = abs(record.price - 0.5)
                    record.implied_probability = record.price

                    all_trades.append(record)

            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")

        # Export
        filepath = os.path.join(self.output_dir, f"fresh_trades_{timestamp}.csv")
        self._export_to_csv(all_trades, filepath)

        if verbose:
            print(f"\nExported {len(all_trades)} trades to {filepath}")

        return filepath


def main():
    """CLI for master extraction."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract comprehensive trading data for research')
    parser.add_argument('--limit', type=int, help='Max traders to process')
    parser.add_argument('--no-trades', action='store_true', help='Skip individual trade extraction')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--fresh', action='store_true', help='Fetch fresh data from API')
    parser.add_argument('--addresses', nargs='+', help='Specific addresses to extract')

    args = parser.parse_args()

    extractor = MasterDataExtractor(output_dir=args.output)

    if args.fresh and args.addresses:
        extractor.extract_fresh_trades(args.addresses)
    else:
        extractor.extract_all(
            trader_addresses=args.addresses,
            limit=args.limit,
            include_trades=not args.no_trades
        )


if __name__ == '__main__':
    main()
