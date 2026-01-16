#!/usr/bin/env python3
"""
Polymarket Trader Discovery & Analysis - Main Entry Point

Quick start for common workflows.
"""

from services.trader_discovery import TraderDiscoveryService
from services.trader_analyzer import TraderAnalyzerService
from services.equity_curve import EquityCurveBuilder
from cache.file_cache import TraderCache


def quick_analyze_addresses(addresses: list):
    """
    Quick analysis of specific wallet addresses.

    Args:
        addresses: List of wallet addresses to analyze
    """
    analyzer = TraderAnalyzerService()
    cache = TraderCache()
    curve_builder = EquityCurveBuilder()

    for addr in addresses:
        print(f"\n{'='*60}")
        trader = analyzer.analyze_trader(addr)
        trader.equity_curve = curve_builder.build(trader)
        trader.stats.max_drawdown = curve_builder.calculate_max_drawdown(trader.equity_curve)

        cache.save_trader(trader)
        print(trader.summary())

    cache.export_traders_to_csv(use_cache=True)


def full_discovery_pipeline(
    num_markets: int = 50,
    holders_per_market: int = 20,
    analyze_limit: int = 100
):
    """
    Full discovery and analysis pipeline.

    1. Discover traders from leaderboard and markets
    2. Analyze top traders
    3. Export to CSV
    """
    print("=" * 60)
    print("POLYMARKET TRADER DISCOVERY PIPELINE")
    print("=" * 60)

    # Discovery
    discovery = TraderDiscoveryService()
    cache = TraderCache()

    traders_info = discovery.run_discovery(
        num_markets=num_markets,
        holders_per_market=holders_per_market
    )
    cache.save_discovered_traders(traders_info)

    # Analysis
    analyzer = TraderAnalyzerService()
    curve_builder = EquityCurveBuilder()

    traders_info = traders_info[:analyze_limit]
    print(f"\nAnalyzing top {len(traders_info)} traders...")

    for i, info in enumerate(traders_info):
        print(f"\n[{i+1}/{len(traders_info)}] ", end='')
        try:
            trader = analyzer.analyze_trader(
                address=info['address'],
                name=info.get('name'),
                pseudonym=info.get('pseudonym')
            )
            trader.equity_curve = curve_builder.build(trader)
            trader.stats.max_drawdown = curve_builder.calculate_max_drawdown(trader.equity_curve)
            cache.save_trader(trader)
        except Exception as e:
            print(f"Error: {e}")

    # Export
    cache.export_traders_to_csv(use_cache=True)

    # Show top performers
    print("\n" + "=" * 60)
    print("TOP 10 PROFITABLE TRADERS")
    print("=" * 60)

    all_traders = cache.load_all_traders()
    all_traders.sort(key=lambda t: t['stats'].get('total_pnl', 0), reverse=True)

    for i, t in enumerate(all_traders[:10], 1):
        stats = t['stats']
        print(f"\n{i}. {t['address'][:20]}...")
        print(f"   P&L: ${stats.get('total_pnl', 0):,.2f} | "
              f"ROI: {stats.get('roi_pct', 0):.1f}% | "
              f"Win: {stats.get('win_rate', 0):.1f}%")


if __name__ == '__main__':
    # Example: Analyze specific known profitable traders
    KNOWN_PROFITABLE_TRADERS = [
        '0x60a92c8620846d81f5ea17b0564e0d4b7c545a71',
        '0xc8ab97a9089a9ff7e6ef0688e6e591a066946418',
        '0xb1a1985278c50a25758e2e25a217d084706add34',
        '0xbcc9608ff5481e8452173a2236ffd735fad726fb',
        '0x25257a6a89dba93dd0c536b6279365632a4eb919',  # @Trading4Fridge
        '0x90ed5bffbffbfc344aa1195572d89719a398b5bc',  # @failstober
        '0x689ae12e11aa489adb3605afd8f39040ff52779e',  # @Annica
    ]

    print("Polymarket Trader Discovery & Analysis")
    print("=" * 60)
    print("\nOptions:")
    print("1. Analyze known profitable traders")
    print("2. Run full discovery pipeline")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        quick_analyze_addresses(KNOWN_PROFITABLE_TRADERS)
    elif choice == '2':
        full_discovery_pipeline()
    else:
        print("Goodbye!")
