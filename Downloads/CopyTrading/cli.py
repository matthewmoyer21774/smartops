#!/usr/bin/env python3
"""
CLI for Polymarket Trader Discovery & Analysis.

Usage:
    python cli.py discover [--markets N] [--holders N] [--no-leaderboard] [--no-markets]
    python cli.py analyze [--address ADDR] [--limit N] [--refresh]
    python cli.py filter [--min-trades N] [--min-pnl N] [--min-roi N] [--sort FIELD] [--top N]
    python cli.py show ADDRESS
    python cli.py scan --category nba [--markets N] [--search KEYWORD]
    python cli.py correlate --category nfl [--min-overlap 3] [--show-matrix]
"""

import argparse
import sys
from datetime import datetime

from services.trader_discovery import TraderDiscoveryService
from services.trader_analyzer import TraderAnalyzerService
from services.equity_curve import EquityCurveBuilder
from services.strategy_analyzer import StrategyAnalyzer
from services.correlation_analyzer import CorrelationAnalyzer
from strategies.classifier import StrategyClassifier
from cache.file_cache import TraderCache
from config import MARKET_CATEGORIES


def cmd_discover(args):
    """Discover traders from markets and leaderboard."""
    print("=" * 60)
    print("POLYMARKET TRADER DISCOVERY")
    print("=" * 60)

    service = TraderDiscoveryService()
    cache = TraderCache()

    # Add any manual addresses
    additional = []
    if args.addresses:
        additional = [a.strip() for a in args.addresses.split(',')]

    traders = service.run_discovery(
        use_leaderboard=not args.no_leaderboard,
        use_market_scan=not args.no_markets,
        num_markets=args.markets,
        holders_per_market=args.holders,
        additional_addresses=additional
    )

    # Save discovered traders
    cache.save_discovered_traders(traders)

    print(f"\nSaved {len(traders)} traders to cache")
    print("Run 'python cli.py analyze' to analyze them")


def cmd_analyze(args):
    """Analyze discovered or specified traders."""
    print("=" * 60)
    print("POLYMARKET TRADER ANALYSIS")
    print("=" * 60)

    cache = TraderCache()
    analyzer = TraderAnalyzerService()
    curve_builder = EquityCurveBuilder()

    # Determine which traders to analyze
    if args.address:
        # Single address
        traders_info = [{'address': args.address}]
    else:
        # Load discovered traders
        traders_info = cache.load_discovered_traders()
        if not traders_info:
            print("No discovered traders found. Run 'discover' first.")
            return

    if args.limit:
        traders_info = traders_info[:args.limit]

    print(f"Analyzing {len(traders_info)} traders...")

    analyzed_traders = []
    for i, info in enumerate(traders_info):
        address = info.get('address')
        print(f"\n[{i+1}/{len(traders_info)}] {address[:20]}...")

        # Check cache unless force refresh
        if not args.refresh and cache.trader_exists(address):
            cached = cache.load_trader(address)
            if cached:
                stats = cached.get('stats', {})
                print(f"  [CACHED] P&L: ${stats.get('total_pnl', 0):,.2f} | "
                      f"ROI: {stats.get('roi_pct', 0):.1f}% | "
                      f"Trades: {stats.get('total_trades', 0)}")
                continue

        try:
            trader = analyzer.analyze_trader(
                address=address,
                name=info.get('name'),
                pseudonym=info.get('pseudonym')
            )

            # Build equity curve
            trader.equity_curve = curve_builder.build(trader)
            trader.stats.max_drawdown = curve_builder.calculate_max_drawdown(trader.equity_curve)

            # Save to cache
            cache.save_trader(trader)
            analyzed_traders.append(trader)

            print(f"  P&L: ${trader.stats.total_pnl:,.2f} | "
                  f"ROI: {trader.stats.roi_pct:.1f}% | "
                  f"Win: {trader.stats.win_rate:.1f}% | "
                  f"Trades: {trader.stats.total_trades}")

        except Exception as e:
            print(f"  Error: {e}")

    # Export to CSV
    if analyzed_traders or not args.refresh:
        cache.export_traders_to_csv(use_cache=True)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


def cmd_filter(args):
    """Filter and rank traders by criteria."""
    print("=" * 60)
    print("POLYMARKET TRADER FILTER")
    print("=" * 60)

    cache = TraderCache()

    # Load all cached traders
    traders_data = cache.load_all_traders()
    print(f"Loaded {len(traders_data)} cached traders")

    if not traders_data:
        print("No traders in cache. Run 'discover' and 'analyze' first.")
        return

    # Apply filters
    filtered = traders_data

    if args.min_trades:
        filtered = [t for t in filtered if t['stats'].get('total_trades', 0) >= args.min_trades]
        print(f"After min_trades ({args.min_trades}): {len(filtered)} traders")

    if args.min_pnl:
        filtered = [t for t in filtered if t['stats'].get('total_pnl', 0) >= args.min_pnl]
        print(f"After min_pnl (${args.min_pnl}): {len(filtered)} traders")

    if args.min_roi:
        filtered = [t for t in filtered if t['stats'].get('roi_pct', 0) >= args.min_roi]
        print(f"After min_roi ({args.min_roi}%): {len(filtered)} traders")

    if args.min_win_rate:
        filtered = [t for t in filtered if t['stats'].get('win_rate', 0) >= args.min_win_rate]
        print(f"After min_win_rate ({args.min_win_rate}%): {len(filtered)} traders")

    # Sort
    sort_key = args.sort or 'total_pnl'
    filtered.sort(key=lambda t: t['stats'].get(sort_key, 0) or 0, reverse=True)

    # Display top results
    top_n = min(args.top, len(filtered))
    print(f"\n{'='*60}")
    print(f"TOP {top_n} TRADERS (sorted by {sort_key})")
    print(f"{'='*60}")

    for i, t in enumerate(filtered[:top_n], 1):
        stats = t['stats']
        addr = t['address']
        name = t.get('name') or t.get('pseudonym') or ''

        print(f"\n{i}. {addr}")
        if name:
            print(f"   Name: {name}")
        print(f"   P&L: ${stats.get('total_pnl', 0):,.2f} "
              f"(${stats.get('realized_pnl', 0):,.2f} realized)")
        print(f"   ROI: {stats.get('roi_pct', 0):.1f}% | "
              f"Win Rate: {stats.get('win_rate', 0):.1f}%")
        print(f"   Trades: {stats.get('total_trades', 0)} | "
              f"Markets: {stats.get('unique_markets', 0)} | "
              f"Resolved: {stats.get('total_resolved', 0)}")

    # Export filtered results
    if args.export:
        cache.export_filtered_traders(
            min_trades=args.min_trades,
            min_pnl=args.min_pnl,
            min_roi=args.min_roi,
            min_win_rate=args.min_win_rate,
            sort_by=sort_key,
            top_n=args.top
        )


def cmd_show(args):
    """Show detailed info for a specific trader."""
    cache = TraderCache()
    data = cache.load_trader(args.address)

    if not data:
        print(f"Trader {args.address} not found in cache")
        print("Run 'python cli.py analyze --address {address}' first")
        return

    stats = data.get('stats', {})
    positions = data.get('positions', {})

    print("=" * 60)
    print(f"TRADER: {data['address']}")
    print("=" * 60)

    if data.get('name'):
        print(f"Name: {data['name']}")
    if data.get('pseudonym'):
        print(f"Handle: {data['pseudonym']}")

    print(f"\n--- PERFORMANCE ---")
    print(f"Total P&L:     ${stats.get('total_pnl', 0):,.2f}")
    print(f"  Realized:    ${stats.get('realized_pnl', 0):,.2f}")
    print(f"  Unrealized:  ${stats.get('unrealized_pnl', 0):,.2f}")
    print(f"ROI:           {stats.get('roi_pct', 0):.1f}%")
    print(f"Win Rate:      {stats.get('win_rate', 0):.1f}%")
    print(f"Max Drawdown:  ${stats.get('max_drawdown', 0):,.2f}")

    print(f"\n--- ACTIVITY ---")
    print(f"Total Trades:  {stats.get('total_trades', 0)}")
    print(f"Unique Markets:{stats.get('unique_markets', 0)}")
    print(f"Volume:        ${stats.get('total_volume', 0):,.2f}")
    print(f"Active Days:   {stats.get('active_days', 0)}")

    print(f"\n--- POSITIONS ---")
    print(f"Winning:       {stats.get('winning_positions', 0)}")
    print(f"Losing:        {stats.get('losing_positions', 0)}")
    print(f"Open:          {len([p for p in positions.values() if not p.get('is_resolved', True)])}")

    # Show top positions by P&L
    if positions:
        print(f"\n--- TOP POSITIONS ---")
        sorted_pos = sorted(
            positions.values(),
            key=lambda p: abs(p.get('realized_pnl', 0) or p.get('unrealized_pnl', 0)),
            reverse=True
        )[:5]

        for pos in sorted_pos:
            title = pos.get('market_title', '')[:40] or pos.get('market_slug', '')[:40]
            pnl = pos.get('realized_pnl', 0) or pos.get('unrealized_pnl', 0)
            status = "CLOSED" if pos.get('is_resolved') else "OPEN"
            print(f"  {title}...")
            print(f"    {pos.get('outcome')}: ${pnl:,.2f} [{status}]")


def cmd_deep(args):
    """Deep strategy analysis for a specific trader."""
    analyzer = StrategyAnalyzer()

    print(f"Running deep strategy analysis for {args.address}...")

    report = analyzer.analyze_strategy(
        address=args.address,
        refresh=args.refresh
    )

    analyzer.print_report(report)

    # Export to JSON if requested
    if args.export:
        import json
        cache = TraderCache()
        filepath = f"{cache.exports_dir}/strategy_{args.address[:16]}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"\nExported to {filepath}")


def cmd_scan(args):
    """Scan markets by category and discover traders with their positions."""
    import json

    service = TraderDiscoveryService()
    cache = TraderCache()

    results = service.discover_from_category(
        category=args.category,
        search_pattern=args.search,
        num_markets=args.markets,
        traders_per_market=args.traders,
        include_closed=args.include_closed,
        min_volume=args.min_volume
    )

    markets = results['markets']
    traders = results['traders_list']
    category = results['category']

    if not markets:
        print("No matching markets found.")
        return

    # Show summary
    print(f"\n--- TOP TRADERS BY MARKET COVERAGE ---")
    for i, t in enumerate(traders[:20], 1):
        addr = t['address'][:16]
        name = t.get('name') or t.get('pseudonym') or ''
        num_markets = len(t['markets'])
        vol = t['total_volume']
        print(f"{i:2}. {addr}... {name[:15]:<15} | {num_markets} markets | ${vol:,.0f} volume")

    # Show market positions (who's betting what)
    if args.show_positions:
        print(f"\n--- MARKET POSITIONS ---")
        for market in markets[:10]:
            print(f"\n{market['slug'][:60]}")
            print(f"  Volume: ${market['volume']:,.0f}")

            # Group traders by position
            position_groups = {}
            for t in market['traders'][:10]:
                pos = t.get('dominant_position', 'Unknown')
                if pos not in position_groups:
                    position_groups[pos] = []
                position_groups[pos].append(t)

            for outcome, traders_in_pos in position_groups.items():
                print(f"\n  {outcome}:")
                for t in traders_in_pos[:5]:
                    addr = t['address'][:12]
                    vol = t['total_volume']
                    print(f"    {addr}... ${vol:,.0f}")

    # Save results
    if args.export:
        # Save scan results to JSON
        filepath = f"{cache.exports_dir}/scan_{category or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_data = {
            'category': category,
            'search_pattern': args.search,
            'markets_scanned': len(markets),
            'traders_found': len(traders),
            'markets': [
                {
                    'slug': m['slug'],
                    'title': m['title'],
                    'volume': m['volume'],
                    'trader_count': m['trader_count'],
                    'traders': [
                        {
                            'address': t['address'],
                            'name': t.get('name'),
                            'dominant_position': t.get('dominant_position'),
                            'volume': t['total_volume']
                        }
                        for t in m['traders'][:20]
                    ]
                }
                for m in markets
            ],
            'top_traders': [
                {
                    'address': t['address'],
                    'name': t.get('name'),
                    'pseudonym': t.get('pseudonym'),
                    'markets_count': len(t['markets']),
                    'total_volume': t['total_volume'],
                    'positions_by_market': t['positions_by_market']
                }
                for t in traders[:50]
            ]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\nExported to {filepath}")

    # Save discovered traders for later analysis
    discovered = [
        {
            'address': t['address'],
            'name': t.get('name'),
            'pseudonym': t.get('pseudonym'),
            'source': f'scan:{category}',
            'markets_found_in': t['markets']
        }
        for t in traders
    ]
    cache.save_discovered_traders(discovered)
    print(f"\nSaved {len(discovered)} traders to cache. Run 'python cli.py analyze' to analyze them.")


def cmd_correlate(args):
    """Analyze trader correlations and find smart money clusters."""
    analyzer = CorrelationAnalyzer()
    cache = TraderCache()

    # Handle --smart-money preset
    min_pnl = args.min_pnl
    min_roi = args.min_roi
    min_win_rate = args.min_win_rate
    min_trades = args.min_trades

    if args.smart_money:
        # Apply smart money preset defaults (unless explicitly overridden)
        if min_pnl == 0:
            min_pnl = 1000
        if min_roi == 0:
            min_roi = 5
        if min_win_rate == 0:
            min_win_rate = 55
        if min_trades == 0:
            min_trades = 5

    is_smart_money = any([min_pnl > 0, min_roi > 0, min_win_rate > 0, min_trades > 0])

    print("=" * 70)
    if is_smart_money:
        print("SMART MONEY ANALYSIS")
    else:
        print("TRADER CORRELATION ANALYSIS")
    print("=" * 70)

    # Run correlation analysis
    results = analyzer.analyze_from_scan(
        category=args.category,
        search_pattern=args.search,
        num_markets=args.markets,
        traders_per_market=args.traders,
        min_shared_markets=args.min_overlap,
        include_closed=args.include_closed,
        min_pnl=min_pnl,
        min_roi=min_roi,
        min_win_rate=min_win_rate,
        min_trades=min_trades,
        analyze_limit=args.analyze_limit
    )

    if not results['scan']['markets']:
        print("No matching markets found.")
        return

    # Print the report
    analyzer.print_report(
        results,
        show_matrix=args.show_matrix,
        top_markets=args.top_markets,
        top_clusters=args.top_clusters
    )

    # Export if requested
    if args.export:
        category = args.category or 'all'
        filepath = f"{cache.exports_dir}/correlate_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.export_results(results, filepath)

    # Save traders to cache for later analysis
    traders = results['scan']['traders_list']
    if traders:
        discovered = [
            {
                'address': t['address'],
                'name': t.get('name'),
                'pseudonym': t.get('pseudonym'),
                'source': f'correlate:{args.category}',
                'markets_found_in': t['markets']
            }
            for t in traders
        ]
        cache.save_discovered_traders(discovered)
        print(f"\nSaved {len(discovered)} traders to cache.")


def cmd_strategy(args):
    """Classify trader strategies based on transaction patterns."""
    from models.trader import Trader, TraderStats
    from models.position import Position

    cache = TraderCache()
    classifier = StrategyClassifier()

    print("=" * 70)
    print("STRATEGY CLASSIFICATION")
    print("=" * 70)

    # Handle different modes
    if args.types:
        # Show strategy types
        types = classifier.get_strategy_types()
        print("\nAvailable Strategy Types:")
        print("-" * 40)
        for st in types:
            copyable = "Yes" if st['copyable'] else "No"
            print(f"\n  {st['type'].upper()}")
            print(f"    Name: {st['name']}")
            print(f"    Description: {st['description']}")
            print(f"    Risk Level: {st['risk_level']}")
            print(f"    Copyable: {copyable}")
        return

    if args.find:
        # Find traders by strategy
        strategy_type = args.find
        valid_types = ['arbitrage', 'insider', 'lazy_positions', 'momentum']
        if strategy_type not in valid_types:
            print(f"Invalid strategy type. Valid: {valid_types}")
            return

        print(f"\nFinding traders with '{strategy_type}' strategy...")
        print(f"Min confidence: {args.min_confidence}")

        # Load all traders
        all_cached = cache.load_all_traders()
        traders = []

        for trader_data in all_cached:
            if isinstance(trader_data, dict):
                trader = _dict_to_trader(trader_data)
                traders.append(trader)

        profiles = classifier.find_by_strategy(
            traders=traders,
            strategy=strategy_type,
            min_confidence=args.min_confidence
        )

        profiles.sort(key=lambda p: p.primary_confidence, reverse=True)
        profiles = profiles[:args.limit]

        if not profiles:
            print(f"\nNo traders found with {strategy_type} strategy above {args.min_confidence} confidence")
            return

        print(f"\nFound {len(profiles)} traders:")
        print("-" * 60)

        for i, profile in enumerate(profiles, 1):
            name = profile.trader_name or profile.trader_address[:16] + "..."
            copyable = "Yes" if profile.copyable else "No"
            print(f"\n{i}. {name}")
            print(f"   Address: {profile.trader_address}")
            print(f"   Confidence: {profile.primary_confidence:.0%}")
            print(f"   Risk: {profile.risk_level} | Copyable: {copyable}")
            if profile.warnings:
                print(f"   Warning: {profile.warnings[0]}")

        return

    if args.all:
        # Classify all cached traders
        print("\nClassifying all cached traders...")
        all_cached = cache.load_all_traders()

        if not all_cached:
            print("No traders in cache. Run 'discover' and 'analyze' first.")
            return

        # Group by strategy
        strategy_counts = {'arbitrage': [], 'insider': [], 'lazy_positions': [], 'momentum': [], 'unknown': []}

        for trader_data in all_cached:
            if isinstance(trader_data, dict):
                trader = _dict_to_trader(trader_data)
            else:
                trader = trader_data

            profile = classifier.classify(trader)
            strategy = profile.primary_strategy if profile.primary_confidence > args.min_confidence else 'unknown'
            strategy_counts[strategy].append(profile)

        print("\n" + "=" * 60)
        print("STRATEGY DISTRIBUTION")
        print("=" * 60)

        for strategy, profiles in strategy_counts.items():
            if profiles:
                print(f"\n{strategy.upper()}: {len(profiles)} traders")
                # Show top 3 by confidence
                profiles.sort(key=lambda p: p.primary_confidence, reverse=True)
                for p in profiles[:3]:
                    name = p.trader_name or p.trader_address[:12] + "..."
                    print(f"  - {name}: {p.primary_confidence:.0%}")

        return

    # Single trader analysis
    if not args.address:
        print("Please provide --address, --all, --find, or --types")
        return

    trader_data = cache.load_trader(args.address)
    if not trader_data:
        print(f"Trader {args.address} not found in cache")
        print("Run 'python cli.py analyze --address <addr>' first")
        return

    print(f"\nAnalyzing: {args.address}")

    if isinstance(trader_data, dict):
        trader = _dict_to_trader(trader_data)
    else:
        trader = trader_data

    profile = classifier.classify(trader)

    # Print results
    print("\n" + "=" * 60)
    print(profile.summary())
    print("=" * 60)

    if args.verbose:
        print("\n--- DETAILED SIGNALS ---")
        if profile.arbitrage_signals:
            print(f"\nArbitrage: {profile.arbitrage_signals.confidence_score:.0%}")
            for key, val in profile.arbitrage_signals.indicators.items():
                print(f"  {key}: {val}")

        if profile.insider_signals:
            print(f"\nInsider: {profile.insider_signals.confidence_score:.0%}")
            for key, val in profile.insider_signals.indicators.items():
                print(f"  {key}: {val}")

        if profile.lazy_signals:
            print(f"\nLazy Positions: {profile.lazy_signals.confidence_score:.0%}")
            for key, val in profile.lazy_signals.indicators.items():
                print(f"  {key}: {val}")

        if profile.momentum_signals:
            print(f"\nMomentum: {profile.momentum_signals.confidence_score:.0%}")
            for key, val in profile.momentum_signals.indicators.items():
                print(f"  {key}: {val}")

    if profile.recommendations:
        print("\n--- RECOMMENDATIONS ---")
        for rec in profile.recommendations:
            print(f"  * {rec}")

    # Export if requested
    if args.export:
        import json
        filepath = f"{cache.exports_dir}/strategy_{args.address[:16]}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, indent=2, default=str)
        print(f"\nExported to {filepath}")


def cmd_research(args):
    """Extract comprehensive research data for hedge fund analysis."""
    from research.master_extractor import MasterDataExtractor

    print("=" * 70)
    print("HEDGE FUND RESEARCH DATA EXTRACTION")
    print("=" * 70)

    extractor = MasterDataExtractor(output_dir=args.output)

    if args.fresh:
        if not args.addresses:
            print("Error: --fresh requires --addresses")
            return

        print(f"\nFetching fresh data for {len(args.addresses)} addresses...")
        filepath = extractor.extract_fresh_trades(
            addresses=args.addresses,
            verbose=args.verbose or True
        )
        print(f"\nOutput: {filepath}")

    else:
        # Extract from cache
        if not args.all and not args.addresses:
            print("Please specify --all or --addresses")
            return

        print("\nExtracting data from cached traders...")
        if args.limit:
            print(f"  Limit: {args.limit} traders")
        if args.no_trades:
            print("  Skipping individual trade extraction")

        output_files = extractor.extract_all(
            trader_addresses=args.addresses,
            limit=args.limit,
            include_trades=not args.no_trades,
            verbose=args.verbose or True
        )

        print("\n" + "=" * 70)
        print("OUTPUT FILES:")
        print("=" * 70)
        for file_type, filepath in output_files.items():
            print(f"  {file_type.upper()}: {filepath}")

        print("\n" + "=" * 70)
        print("READY FOR ANALYSIS")
        print("=" * 70)
        print("\nUse these CSV files to:")
        print("  - traders_master.csv: Identify top performers and strategies")
        print("  - positions_master.csv: Analyze entry/exit patterns")
        print("  - trades_master.csv: Build predictive models")
        print("  - extraction_summary.json: Quick overview and top performers")


def cmd_alpha(args):
    """Extract actionable alpha signals from smart money traders."""
    from research.alpha_extractor import AlphaExtractor

    print("=" * 70)
    print("ALPHA SIGNAL EXTRACTION")
    print("=" * 70)

    extractor = AlphaExtractor()

    print(f"\nFilters: min P&L=${args.min_pnl:,.0f}, min positions={args.min_positions}")
    print("Extracting alpha signals from cached traders...\n")

    profiles, filepath = extractor.extract_all(
        min_pnl=args.min_pnl,
        min_positions=args.min_positions,
        verbose=args.verbose or True
    )

    if not profiles:
        print("No traders matched the filters.")
        return

    # Print summary insights
    print("\n" + "=" * 70)
    print("KEY ALPHA INSIGHTS")
    print("=" * 70)

    # Entry zone analysis
    zones = {'below_40': [], '40_60': [], '60_80': [], 'above_80': []}
    for p in profiles:
        for zone in zones.keys():
            wr = getattr(p, f'win_rate_{zone}', 0)
            roi = getattr(p, f'avg_roi_{zone}', 0)
            if wr > 0:
                zones[zone].append({'wr': wr, 'roi': roi})

    print("\n[ENTRY ZONE PERFORMANCE]")
    print("-" * 50)
    for zone, data in zones.items():
        if data:
            avg_wr = sum(d['wr'] for d in data) / len(data)
            avg_roi = sum(d['roi'] for d in data) / len(data)
            print(f"  {zone.replace('_', '-'):12} | Win Rate: {avg_wr:5.1f}% | Avg ROI: {avg_roi:8.1f}%")

    # Top traders by category
    print("\n[TOP TRADER SIGNALS]")
    print("-" * 50)
    sorted_profiles = sorted(profiles, key=lambda p: p.total_pnl, reverse=True)[:5]
    for p in sorted_profiles:
        name = p.name or p.address[:12]
        print(f"\n  {name}")
        print(f"    P&L: ${p.total_pnl:,.0f} | ROI: {p.roi_pct:.1f}% | Win Rate: {p.win_rate:.1f}%")
        print(f"    Focus: {p.top_category}")
        if p.signals:
            print(f"    Signal: {p.signals[0].actionable_insight}")

    print("\n" + "=" * 70)
    print(f"Full data exported to: {filepath}")
    print("=" * 70)


def _dict_to_trader(trader_data: dict):
    """Convert dict to Trader object."""
    from models.trader import Trader, TraderStats
    from models.position import Position

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
    parser = argparse.ArgumentParser(
        description='Polymarket Trader Discovery & Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py discover --markets 50 --holders 20
  python cli.py analyze --limit 50
  python cli.py filter --min-trades 100 --min-pnl 1000 --sort roi_pct
  python cli.py show 0x60a92c8620846d81f5ea17b0564e0d4b7c545a71
  python cli.py deep 0xcc553b67cfa321f74c... --export
  python cli.py scan --category nba --markets 20 --show-positions
  python cli.py scan --category nfl --search spread --export
  python cli.py correlate --category nfl --min-overlap 3 --show-matrix
  python cli.py correlate --category crypto --markets 10 --export
  python cli.py strategy --address 0x... --verbose
  python cli.py strategy --all
  python cli.py strategy --find arbitrage --min-confidence 0.5
  python cli.py strategy --types
  python cli.py research --all --verbose
  python cli.py research --limit 50 --no-trades
  python cli.py research --fresh --addresses 0x... 0x...
  python cli.py alpha --min-pnl 10000 --min-positions 20
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Discover traders from markets')
    discover_parser.add_argument('--no-leaderboard', action='store_true',
                                 help='Skip leaderboard')
    discover_parser.add_argument('--no-markets', action='store_true',
                                 help='Skip market scanning')
    discover_parser.add_argument('--markets', type=int, default=50,
                                 help='Number of markets to scan (default: 50)')
    discover_parser.add_argument('--holders', type=int, default=20,
                                 help='Top holders per market (default: 20)')
    discover_parser.add_argument('--addresses', type=str,
                                 help='Additional addresses (comma-separated)')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze traders')
    analyze_parser.add_argument('--address', type=str,
                                help='Analyze single address')
    analyze_parser.add_argument('--limit', type=int,
                                help='Limit number of traders')
    analyze_parser.add_argument('--refresh', action='store_true',
                                help='Force refresh cached traders')

    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter and rank traders')
    filter_parser.add_argument('--min-trades', type=int,
                               help='Minimum number of trades')
    filter_parser.add_argument('--min-pnl', type=float,
                               help='Minimum total P&L (USD)')
    filter_parser.add_argument('--min-roi', type=float,
                               help='Minimum ROI percentage')
    filter_parser.add_argument('--min-win-rate', type=float,
                               help='Minimum win rate percentage')
    filter_parser.add_argument('--sort', type=str, default='total_pnl',
                               choices=['total_pnl', 'roi_pct', 'win_rate', 'total_trades'],
                               help='Sort field (default: total_pnl)')
    filter_parser.add_argument('--top', type=int, default=20,
                               help='Show top N traders (default: 20)')
    filter_parser.add_argument('--export', action='store_true',
                               help='Export filtered results to CSV')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show trader details')
    show_parser.add_argument('address', type=str, help='Trader wallet address')

    # Deep strategy analysis command
    deep_parser = subparsers.add_parser('deep', help='Deep strategy analysis')
    deep_parser.add_argument('address', type=str, help='Trader wallet address')
    deep_parser.add_argument('--refresh', action='store_true',
                             help='Force refresh from API')
    deep_parser.add_argument('--export', action='store_true',
                             help='Export analysis to JSON')

    # Scan command - category-specific market scanning
    scan_parser = subparsers.add_parser('scan', help='Scan markets by category')
    scan_parser.add_argument('--category', type=str,
                             choices=list(MARKET_CATEGORIES.keys()),
                             help='Market category (nba, nfl, crypto, politics, etc.)')
    scan_parser.add_argument('--search', type=str,
                             help='Additional keyword filter (e.g., "spread", "total")')
    scan_parser.add_argument('--markets', type=int, default=20,
                             help='Max markets to scan (default: 20)')
    scan_parser.add_argument('--traders', type=int, default=50,
                             help='Max traders per market (default: 50)')
    scan_parser.add_argument('--min-volume', type=float, default=0,
                             help='Minimum market volume (default: 0)')
    scan_parser.add_argument('--include-closed', action='store_true',
                             help='Include closed/resolved markets')
    scan_parser.add_argument('--show-positions', action='store_true',
                             help='Show trader positions by market')
    scan_parser.add_argument('--export', action='store_true',
                             help='Export scan results to JSON')

    # Correlate command - find trader correlations and smart money clusters
    correlate_parser = subparsers.add_parser('correlate', help='Analyze trader correlations')
    correlate_parser.add_argument('--category', type=str,
                                  choices=list(MARKET_CATEGORIES.keys()),
                                  help='Market category (nba, nfl, crypto, politics, etc.)')
    correlate_parser.add_argument('--search', type=str,
                                  help='Additional keyword filter')
    correlate_parser.add_argument('--markets', type=int, default=20,
                                  help='Max markets to scan (default: 20)')
    correlate_parser.add_argument('--traders', type=int, default=100,
                                  help='Max traders per market (default: 100)')
    correlate_parser.add_argument('--min-overlap', type=int, default=2,
                                  help='Min shared markets to form cluster (default: 2)')
    correlate_parser.add_argument('--include-closed', action='store_true',
                                  help='Include closed/resolved markets')
    correlate_parser.add_argument('--show-matrix', action='store_true',
                                  help='Show correlation matrix')
    correlate_parser.add_argument('--top-markets', type=int, default=10,
                                  help='Show top N markets in consensus (default: 10)')
    correlate_parser.add_argument('--top-clusters', type=int, default=5,
                                  help='Show top N clusters (default: 5)')
    correlate_parser.add_argument('--export', action='store_true',
                                  help='Export correlation results to JSON')
    # Smart money filters
    correlate_parser.add_argument('--min-pnl', type=float, default=0,
                                  help='Minimum P&L in USD (smart money filter)')
    correlate_parser.add_argument('--min-roi', type=float, default=0,
                                  help='Minimum ROI percentage (smart money filter)')
    correlate_parser.add_argument('--min-win-rate', type=float, default=0,
                                  help='Minimum win rate percentage (smart money filter)')
    correlate_parser.add_argument('--min-trades', type=int, default=0,
                                  help='Minimum resolved trades (smart money filter)')
    correlate_parser.add_argument('--smart-money', action='store_true',
                                  help='Preset: min-pnl=1000, min-roi=5, min-win-rate=55, min-trades=5')
    correlate_parser.add_argument('--analyze-limit', type=int, default=50,
                                  help='Max traders to analyze for performance (default: 50)')

    # Strategy command - classify trader strategies
    strategy_parser = subparsers.add_parser('strategy', help='Classify trader strategies')
    strategy_parser.add_argument('--address', type=str,
                                 help='Analyze single trader address')
    strategy_parser.add_argument('--all', action='store_true',
                                 help='Classify all cached traders')
    strategy_parser.add_argument('--find', type=str,
                                 choices=['arbitrage', 'insider', 'lazy_positions', 'momentum'],
                                 help='Find traders using specific strategy')
    strategy_parser.add_argument('--types', action='store_true',
                                 help='Show available strategy types')
    strategy_parser.add_argument('--min-confidence', type=float, default=0.3,
                                 help='Minimum confidence threshold (default: 0.3)')
    strategy_parser.add_argument('--limit', type=int, default=20,
                                 help='Max results when finding by strategy (default: 20)')
    strategy_parser.add_argument('--verbose', '-v', action='store_true',
                                 help='Show detailed signal breakdown')
    strategy_parser.add_argument('--export', action='store_true',
                                 help='Export strategy profile to JSON')

    # Research command - comprehensive data extraction for hedge fund research
    research_parser = subparsers.add_parser('research', help='Extract comprehensive research data')
    research_parser.add_argument('--all', action='store_true',
                                 help='Extract all cached traders')
    research_parser.add_argument('--limit', type=int,
                                 help='Max traders to process')
    research_parser.add_argument('--addresses', nargs='+',
                                 help='Specific addresses to extract')
    research_parser.add_argument('--no-trades', action='store_true',
                                 help='Skip individual trade extraction (faster)')
    research_parser.add_argument('--output', type=str,
                                 help='Output directory')
    research_parser.add_argument('--fresh', action='store_true',
                                 help='Fetch fresh data from API (bypasses cache)')
    research_parser.add_argument('--verbose', '-v', action='store_true',
                                 help='Show detailed progress')

    # Alpha command - extract actionable alpha signals
    alpha_parser = subparsers.add_parser('alpha', help='Extract alpha signals from smart money')
    alpha_parser.add_argument('--min-pnl', type=float, default=10000,
                              help='Minimum P&L to include (default: 10000)')
    alpha_parser.add_argument('--min-positions', type=int, default=20,
                              help='Minimum positions to include (default: 20)')
    alpha_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Show detailed progress')

    args = parser.parse_args()

    if args.command == 'discover':
        cmd_discover(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'filter':
        cmd_filter(args)
    elif args.command == 'show':
        cmd_show(args)
    elif args.command == 'deep':
        cmd_deep(args)
    elif args.command == 'scan':
        cmd_scan(args)
    elif args.command == 'correlate':
        cmd_correlate(args)
    elif args.command == 'strategy':
        cmd_strategy(args)
    elif args.command == 'research':
        cmd_research(args)
    elif args.command == 'alpha':
        cmd_alpha(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
