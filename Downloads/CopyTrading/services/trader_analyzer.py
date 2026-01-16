"""
Service for analyzing trader performance and calculating metrics.
"""

from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, '..')
from api.data_api import DataAPIClient
from api.gamma_api import GammaAPIClient
from models.trader import Trader, TraderStats
from models.trade import Trade
from models.position import Position


class TraderAnalyzerService:
    """Analyze trader performance and calculate metrics."""

    def __init__(self):
        self.data_client = DataAPIClient()
        self.gamma_client = GammaAPIClient()
        # Cache market resolution status
        self.market_cache: Dict[str, Dict] = {}

    def analyze_trader(self, address: str, name: str = None, pseudonym: str = None) -> Trader:
        """
        Full analysis of a single trader.

        Steps:
        1. Fetch all trades
        2. Build positions from trades
        3. Fetch market resolution status
        4. Calculate P&L metrics
        5. Compute statistics

        Args:
            address: Wallet address
            name: Optional display name
            pseudonym: Optional handle

        Returns:
            Trader object with complete analysis
        """
        print(f"Analyzing trader: {address[:16]}...")

        trader = Trader(
            address=address.lower(),
            name=name,
            pseudonym=pseudonym,
            discovery_date=datetime.now()
        )

        # 1. Fetch all trades
        raw_trades = self.data_client.get_trader_trades(address)
        print(f"  Found {len(raw_trades)} trades")

        if not raw_trades:
            trader.last_updated = datetime.now()
            return trader

        # 2. Parse trades into model objects
        for raw in raw_trades:
            trade = self._parse_trade(raw)
            if trade:
                trader.trades.append(trade)

        # Sort trades by timestamp (oldest first for proper position building)
        trader.trades.sort(key=lambda t: t.timestamp)

        # 3. Build positions from trades
        self._build_positions(trader)
        print(f"  Built {len(trader.positions)} positions")

        # 4. Fetch market resolution status and update positions
        self._update_position_status(trader)

        # 5. Calculate statistics
        trader.stats = self._calculate_stats(trader)

        trader.last_updated = datetime.now()
        return trader

    def _parse_trade(self, raw: Dict) -> Optional[Trade]:
        """Parse raw API response into Trade model."""
        try:
            # Handle timestamp
            timestamp = raw.get('timestamp')
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                # Try ISO format
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    dt = datetime.now()
            else:
                dt = datetime.now()

            price = float(raw.get('price', 0) or 0)
            size = float(raw.get('size', 0) or raw.get('usdcSize', 0) or 0)

            # Skip invalid trades
            if price <= 0 or size <= 0:
                return None

            # Calculate shares
            shares = size / price if price > 0 else 0

            return Trade(
                transaction_hash=raw.get('transactionHash', ''),
                condition_id=raw.get('conditionId', ''),
                asset=raw.get('asset', '') or raw.get('tokenId', ''),
                timestamp=dt,
                side=raw.get('side', 'BUY').upper(),
                outcome=raw.get('outcome', 'Yes'),
                outcome_index=int(raw.get('outcomeIndex', 0) or 0),
                price=price,
                size=size,
                shares=shares,
                market_slug=raw.get('slug', ''),
                market_title=raw.get('title', '')
            )
        except Exception as e:
            print(f"    Error parsing trade: {e}")
            return None

    def _build_positions(self, trader: Trader):
        """Aggregate trades into positions."""
        for trade in trader.trades:
            position = trader.get_or_create_position(
                condition_id=trade.condition_id,
                outcome=trade.outcome,
                market_slug=trade.market_slug,
                market_title=trade.market_title
            )
            position.add_trade(trade)

    def _parse_outcome_prices(self, market: Dict) -> list:
        """Parse outcome prices from market data (may be string or list)."""
        import json
        outcome_prices = market.get('outcomePrices', [])

        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, TypeError):
                outcome_prices = []

        if not isinstance(outcome_prices, list):
            outcome_prices = []

        return outcome_prices

    def _update_position_status(self, trader: Trader):
        """Update current prices and resolution status for all positions."""
        # Group positions by condition_id to minimize API calls
        condition_ids = set(p.condition_id for p in trader.positions.values())

        for condition_id in condition_ids:
            market = self._get_market(condition_id)
            if not market:
                continue

            is_closed = market.get('closed', False)
            outcome_prices = self._parse_outcome_prices(market)

            # Find the resolution if closed
            resolved_yes = None
            if is_closed:
                # Check various fields for resolution
                resolution = market.get('resolution')
                if resolution is not None:
                    resolved_yes = resolution.lower() == 'yes' if isinstance(resolution, str) else bool(resolution)
                elif outcome_prices and len(outcome_prices) >= 2:
                    # Check outcome prices
                    try:
                        yes_price = float(outcome_prices[0]) if outcome_prices[0] else 0
                        if yes_price >= 0.99:
                            resolved_yes = True
                        elif yes_price <= 0.01:
                            resolved_yes = False
                    except (ValueError, TypeError):
                        pass

            # Get current price for open markets
            current_yes_price = 0.5
            if not is_closed and outcome_prices:
                try:
                    current_yes_price = float(outcome_prices[0]) if outcome_prices[0] else 0.5
                except (ValueError, TypeError):
                    current_yes_price = 0.5

            # Update all positions for this market
            for key, position in trader.positions.items():
                if position.condition_id != condition_id:
                    continue

                if is_closed and resolved_yes is not None:
                    position.is_resolved = True
                    # Yes position wins if market resolved Yes
                    won = (position.outcome.lower() == 'yes') == resolved_yes
                    position.resolved_price = 1.0 if won else 0.0
                else:
                    # Open market - set current price
                    if position.outcome.lower() == 'yes':
                        position.update_price(current_yes_price)
                    else:
                        position.update_price(1.0 - current_yes_price)

    def _get_market(self, condition_id: str) -> Optional[Dict]:
        """Get market data with caching."""
        if condition_id in self.market_cache:
            return self.market_cache[condition_id]

        market = self.gamma_client.get_market_by_condition_id(condition_id)
        if market:
            self.market_cache[condition_id] = market

        return market

    def _calculate_stats(self, trader: Trader) -> TraderStats:
        """Calculate comprehensive statistics for trader."""
        stats = TraderStats()

        # Basic counts
        stats.total_trades = len(trader.trades)
        stats.unique_markets = len(set(t.condition_id for t in trader.trades))

        # Volume calculation
        stats.total_volume = sum(t.size for t in trader.trades)

        # P&L calculations from positions
        for position in trader.positions.values():
            if position.is_resolved:
                stats.realized_pnl += position.realized_pnl
                stats.total_resolved += 1
                if position.realized_pnl > 0:
                    stats.winning_positions += 1
                elif position.realized_pnl < 0:
                    stats.losing_positions += 1
            else:
                stats.unrealized_pnl += position.unrealized_pnl

        stats.total_pnl = stats.realized_pnl + stats.unrealized_pnl

        # Win rate (only for resolved positions)
        if stats.total_resolved > 0:
            stats.win_rate = (stats.winning_positions / stats.total_resolved) * 100

        # ROI calculation
        if stats.total_volume > 0:
            stats.roi_pct = (stats.total_pnl / stats.total_volume) * 100

        # Date range
        if trader.trades:
            stats.first_trade_date = trader.trades[0].timestamp
            stats.last_trade_date = trader.trades[-1].timestamp
            unique_days = set(t.timestamp.date() for t in trader.trades)
            stats.active_days = len(unique_days)

        # Position sizing
        position_sizes = [t.size for t in trader.trades]
        if position_sizes:
            stats.avg_position_size = sum(position_sizes) / len(position_sizes)
            stats.max_position_size = max(position_sizes)

        return stats

    def analyze_multiple(
        self,
        traders_info: List[Dict],
        limit: Optional[int] = None
    ) -> List[Trader]:
        """
        Analyze multiple traders.

        Args:
            traders_info: List of trader info dicts from discovery
            limit: Max traders to analyze

        Returns:
            List of analyzed Trader objects
        """
        if limit:
            traders_info = traders_info[:limit]

        analyzed = []
        total = len(traders_info)

        for i, info in enumerate(traders_info):
            print(f"\n[{i+1}/{total}] ", end='')
            try:
                trader = self.analyze_trader(
                    address=info['address'],
                    name=info.get('name'),
                    pseudonym=info.get('pseudonym')
                )
                analyzed.append(trader)
                print(f"  P&L: ${trader.stats.total_pnl:,.2f} | ROI: {trader.stats.roi_pct:.1f}%")
            except Exception as e:
                print(f"  Error: {e}")

        return analyzed
