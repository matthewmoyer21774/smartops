"""
Service for backtesting trader strategies and simulating portfolios.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
import math

import sys
sys.path.insert(0, '..')
from cache.file_cache import TraderCache


class BacktestSimulator:
    """Simulate modified trading strategies and portfolio combinations."""

    def __init__(self):
        self.cache = TraderCache()

    def backtest_trader(
        self,
        address: str,
        modifications: Dict = None
    ) -> Dict:
        """
        Backtest a trader with modified parameters.

        Args:
            address: Trader wallet address
            modifications: Dict with:
                - position_size_multiplier: Scale all positions by this factor
                - take_profit_percent: Exit when profit hits this %
                - stop_loss_percent: Exit when loss hits this %
                - max_positions: Max concurrent positions

        Returns:
            Dict comparing original vs simulated performance
        """
        trader = self.cache.load_trader(address)

        if not trader:
            return {'error': 'Trader not found'}

        positions = trader.get('positions', {})
        original_stats = trader.get('stats', {})
        mods = modifications or {}

        # Default modifications
        size_mult = mods.get('position_size_multiplier', 1.0)
        take_profit = mods.get('take_profit_percent', 0)  # 0 = no take profit
        stop_loss = mods.get('stop_loss_percent', 0)  # 0 = no stop loss
        max_positions = mods.get('max_positions', 0)  # 0 = unlimited

        # Simulate each position
        simulated_pnl = 0
        simulated_wins = 0
        simulated_losses = 0
        trades_modified = 0
        stopped_out = 0
        took_profit = 0

        position_list = list(positions.values())

        # Sort by some proxy for entry time if available
        # For now, just process in order

        active_positions = 0
        skipped_positions = 0

        for pos in position_list:
            if not pos.get('is_resolved'):
                continue

            entry_price = pos.get('avg_entry_price', 0)
            shares = pos.get('total_shares', 0)
            cost = pos.get('total_cost', 0)
            original_pnl = pos.get('realized_pnl', 0)
            outcome = pos.get('outcome', '')

            # Check max positions limit
            if max_positions > 0 and active_positions >= max_positions:
                skipped_positions += 1
                continue

            # Apply size multiplier
            adj_shares = shares * size_mult
            adj_cost = cost * size_mult

            # Determine if position would have hit stop loss or take profit
            # Since we don't have price history, we estimate based on final outcome

            won = original_pnl > 0
            resolution_price = 1.0 if (outcome == 'Yes' and won) or (outcome == 'No' and not won) else 0.0

            # Calculate what the P&L would be at different exit points
            if won:
                # Winner - check if take profit would have triggered earlier
                max_profit_pct = abs(resolution_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

                if take_profit > 0 and max_profit_pct >= take_profit:
                    # Would have taken profit earlier
                    exit_price = entry_price * (1 + take_profit / 100)
                    sim_pnl = adj_shares * (exit_price - entry_price) if outcome == 'Yes' else adj_shares * (entry_price - exit_price)
                    took_profit += 1
                    trades_modified += 1
                else:
                    # Let it ride to resolution
                    sim_pnl = (original_pnl / shares * adj_shares) if shares > 0 else 0

                simulated_wins += 1
            else:
                # Loser - check if stop loss would have limited damage
                max_loss_pct = abs(resolution_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

                if stop_loss > 0 and max_loss_pct >= stop_loss:
                    # Stop loss would have triggered
                    exit_price = entry_price * (1 - stop_loss / 100) if outcome == 'Yes' else entry_price * (1 + stop_loss / 100)
                    sim_pnl = adj_shares * (exit_price - entry_price) if outcome == 'Yes' else adj_shares * (entry_price - exit_price)
                    stopped_out += 1
                    trades_modified += 1
                else:
                    # Full loss
                    sim_pnl = (original_pnl / shares * adj_shares) if shares > 0 else 0

                simulated_losses += 1

            simulated_pnl += sim_pnl
            active_positions += 1

        # Calculate simulated stats
        total_simulated = simulated_wins + simulated_losses
        simulated_win_rate = (simulated_wins / total_simulated * 100) if total_simulated > 0 else 0

        # Estimate simulated volume
        simulated_volume = original_stats.get('total_volume', 0) * size_mult

        # Calculate ROI
        simulated_roi = (simulated_pnl / simulated_volume * 100) if simulated_volume > 0 else 0

        # Estimate max drawdown change (simplified)
        original_dd = original_stats.get('max_drawdown', 0)
        if stop_loss > 0:
            # Stop losses should reduce max drawdown
            dd_reduction = min(0.5, stop_loss / 100)  # Cap at 50% reduction
            simulated_dd = original_dd * (1 - dd_reduction)
        else:
            simulated_dd = original_dd * size_mult

        return {
            'address': address,
            'modifications': mods,
            'original': {
                'total_pnl': original_stats.get('total_pnl', 0),
                'roi': original_stats.get('roi_pct', 0),
                'win_rate': original_stats.get('win_rate', 0),
                'max_drawdown': original_dd,
                'total_trades': original_stats.get('total_resolved', 0)
            },
            'simulated': {
                'total_pnl': round(simulated_pnl, 2),
                'roi': round(simulated_roi, 2),
                'win_rate': round(simulated_win_rate, 2),
                'max_drawdown': round(simulated_dd, 2),
                'total_trades': total_simulated
            },
            'comparison': {
                'pnl_diff': round(simulated_pnl - original_stats.get('total_pnl', 0), 2),
                'roi_diff': round(simulated_roi - original_stats.get('roi_pct', 0), 2),
                'win_rate_diff': round(simulated_win_rate - original_stats.get('win_rate', 0), 2),
                'drawdown_change_pct': round(
                    ((simulated_dd - original_dd) / original_dd * 100) if original_dd else 0, 1
                )
            },
            'details': {
                'trades_modified': trades_modified,
                'stopped_out': stopped_out,
                'took_profit': took_profit,
                'skipped_positions': skipped_positions
            }
        }

    def simulate_portfolio(
        self,
        traders: List[Dict],
        initial_capital: float = 10000,
        max_position_pct: float = 0.1
    ) -> Dict:
        """
        Simulate a portfolio following multiple traders.

        Args:
            traders: List of {address, weight} dicts
            initial_capital: Starting capital in USD
            max_position_pct: Max single position as % of capital

        Returns:
            Dict with portfolio simulation results
        """
        if not traders:
            return {'error': 'No traders specified'}

        # Normalize weights
        total_weight = sum(t.get('weight', 1) for t in traders)
        normalized = [
            {
                'address': t['address'],
                'weight': t.get('weight', 1) / total_weight
            }
            for t in traders
        ]

        # Load trader data
        trader_data = []
        for t in normalized:
            data = self.cache.load_trader(t['address'])
            if data:
                trader_data.append({
                    'data': data,
                    'weight': t['weight'],
                    'address': t['address']
                })

        if not trader_data:
            return {'error': 'No trader data found'}

        # Simulate combined portfolio
        portfolio_pnl = 0
        portfolio_volume = 0
        individual_contributions = []
        all_positions_pnl = []

        for td in trader_data:
            stats = td['data'].get('stats', {})
            positions = td['data'].get('positions', {})

            # Allocate capital based on weight
            allocated = initial_capital * td['weight']

            # Scale trader's performance to allocated capital
            trader_volume = stats.get('total_volume', 0)
            if trader_volume > 0:
                scale = allocated / trader_volume
            else:
                scale = 0

            trader_pnl_contribution = stats.get('total_pnl', 0) * scale
            portfolio_pnl += trader_pnl_contribution
            portfolio_volume += allocated

            individual_contributions.append({
                'address': td['address'],
                'name': td['data'].get('name') or td['data'].get('pseudonym'),
                'weight': round(td['weight'], 3),
                'allocated_capital': round(allocated, 2),
                'pnl_contribution': round(trader_pnl_contribution, 2),
                'original_roi': stats.get('roi_pct', 0)
            })

            # Collect position P&Ls for correlation analysis
            for pos in positions.values():
                pnl = pos.get('realized_pnl', 0) or pos.get('unrealized_pnl', 0)
                if pnl:
                    all_positions_pnl.append(pnl * scale)

        # Calculate portfolio metrics
        portfolio_roi = (portfolio_pnl / initial_capital * 100) if initial_capital > 0 else 0

        # Estimate portfolio win rate (weighted average)
        weighted_win_rate = sum(
            td['data'].get('stats', {}).get('win_rate', 0) * td['weight']
            for td in trader_data
        )

        # Calculate diversification score (inverse of correlation)
        # Simplified: based on market overlap
        diversification = self._calculate_diversification(trader_data)

        # Estimate max drawdown (simplified)
        weighted_dd = sum(
            td['data'].get('stats', {}).get('max_drawdown', 0) * td['weight']
            for td in trader_data
        )
        # Diversification reduces max drawdown
        portfolio_dd = weighted_dd * (1 - diversification * 0.3)

        # Estimate Sharpe-like ratio
        if all_positions_pnl and len(all_positions_pnl) > 1:
            mean_pnl = statistics.mean(all_positions_pnl)
            std_pnl = statistics.stdev(all_positions_pnl)
            sharpe = (mean_pnl / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0
        else:
            sharpe = 0

        # Risk score (0-1, lower is safer)
        risk_score = self._calculate_risk_score(portfolio_roi, portfolio_dd, diversification)

        return {
            'portfolio_stats': {
                'initial_capital': initial_capital,
                'total_pnl': round(portfolio_pnl, 2),
                'roi': round(portfolio_roi, 2),
                'win_rate': round(weighted_win_rate, 2),
                'max_drawdown': round(portfolio_dd, 2),
                'sharpe_estimate': round(sharpe, 2)
            },
            'individual_contributions': individual_contributions,
            'diversification_score': round(diversification, 3),
            'risk_score': round(risk_score, 3),
            'traders_count': len(trader_data)
        }

    def optimize_weights(
        self,
        addresses: List[str],
        initial_capital: float = 10000,
        optimization_target: str = 'sharpe'  # or 'roi', 'risk_adjusted'
    ) -> Dict:
        """
        Find optimal weights for a portfolio of traders.

        Args:
            addresses: List of trader addresses
            initial_capital: Starting capital
            optimization_target: What to optimize for

        Returns:
            Dict with optimized weights
        """
        if len(addresses) < 2:
            return {'error': 'Need at least 2 traders to optimize'}

        # Load all trader data
        trader_stats = []
        for addr in addresses:
            data = self.cache.load_trader(addr)
            if data:
                stats = data.get('stats', {})
                trader_stats.append({
                    'address': addr,
                    'name': data.get('name') or data.get('pseudonym'),
                    'roi': stats.get('roi_pct', 0),
                    'win_rate': stats.get('win_rate', 0),
                    'volume': stats.get('total_volume', 0),
                    'max_drawdown': stats.get('max_drawdown', 0)
                })

        if len(trader_stats) < 2:
            return {'error': 'Not enough trader data found'}

        # Simple optimization: weight by risk-adjusted return
        total_score = 0
        for t in trader_stats:
            # Score = ROI / (1 + drawdown ratio)
            dd_ratio = t['max_drawdown'] / t['volume'] if t['volume'] > 0 else 1
            t['score'] = t['roi'] / (1 + dd_ratio * 10) if dd_ratio < 1 else t['roi'] * 0.5
            total_score += max(t['score'], 0.01)  # Minimum score

        # Normalize to weights
        optimized = []
        for t in trader_stats:
            weight = max(t['score'], 0.01) / total_score
            optimized.append({
                'address': t['address'],
                'name': t['name'],
                'weight': round(weight, 3),
                'original_roi': t['roi'],
                'original_win_rate': t['win_rate']
            })

        # Sort by weight
        optimized.sort(key=lambda x: x['weight'], reverse=True)

        # Simulate with optimized weights
        sim_input = [{'address': t['address'], 'weight': t['weight']} for t in optimized]
        simulation = self.simulate_portfolio(sim_input, initial_capital)

        return {
            'optimized_weights': optimized,
            'simulation': simulation.get('portfolio_stats', {}),
            'optimization_target': optimization_target
        }

    def _calculate_diversification(self, trader_data: List[Dict]) -> float:
        """
        Calculate diversification score based on market overlap.

        Returns 0-1 where 1 is perfectly diversified (no overlap).
        """
        if len(trader_data) < 2:
            return 0

        # Get markets for each trader
        all_markets = []
        for td in trader_data:
            positions = td['data'].get('positions', {})
            markets = set(pos.get('market_slug', '') for pos in positions.values())
            all_markets.append(markets)

        # Calculate pairwise overlap
        overlaps = []
        for i in range(len(all_markets)):
            for j in range(i + 1, len(all_markets)):
                m1, m2 = all_markets[i], all_markets[j]
                if m1 and m2:
                    intersection = len(m1 & m2)
                    union = len(m1 | m2)
                    overlap = intersection / union if union > 0 else 0
                    overlaps.append(overlap)

        if not overlaps:
            return 0.5

        # Diversification = 1 - average overlap
        avg_overlap = statistics.mean(overlaps)
        return 1 - avg_overlap

    def _calculate_risk_score(
        self,
        roi: float,
        max_drawdown: float,
        diversification: float
    ) -> float:
        """
        Calculate overall risk score (0-1, lower is safer).
        """
        # Higher drawdown = higher risk
        dd_risk = min(max_drawdown / 10000, 1)  # Cap at $10k drawdown

        # Lower diversification = higher risk
        div_risk = 1 - diversification

        # Negative ROI = higher risk
        roi_risk = 0 if roi > 0 else min(abs(roi) / 100, 1)

        # Weighted combination
        risk = (dd_risk * 0.4 + div_risk * 0.3 + roi_risk * 0.3)
        return min(risk, 1)
