#!/usr/bin/env python3
"""
Flask REST API for Polymarket Trader Discovery & Analysis.

Provides endpoints for:
- Listing and filtering cached traders
- Analyzing individual traders
- Scanning markets by category
- Correlation analysis and smart money detection
- Copy trading signals

Usage:
    python app.py
    # Server runs on http://localhost:5000
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from functools import wraps
import json
from datetime import datetime
import traceback

from services.trader_discovery import TraderDiscoveryService
from services.trader_analyzer import TraderAnalyzerService
from services.correlation_analyzer import CorrelationAnalyzer
from services.equity_curve import EquityCurveBuilder
from services.strategy_analyzer import StrategyAnalyzer
from services.benchmark_analyzer import BenchmarkAnalyzer
from services.comparison_analyzer import ComparisonAnalyzer
from services.watchlist_service import WatchlistService
from services.heatmap_builder import HeatmapBuilder
from services.timing_analyzer import TimingAnalyzer
from services.backtest_simulator import BacktestSimulator
from cache.file_cache import TraderCache
from config import MARKET_CATEGORIES, FILTER_PRESETS
from strategies.classifier import StrategyClassifier

app = Flask(__name__)
CORS(app)

# Initialize services
cache = TraderCache()
discovery = TraderDiscoveryService()
analyzer = TraderAnalyzerService()
correlator = CorrelationAnalyzer()
curve_builder = EquityCurveBuilder()
strategy_analyzer = StrategyAnalyzer()
benchmark_analyzer = BenchmarkAnalyzer()
comparison_analyzer = ComparisonAnalyzer()
watchlist_service = WatchlistService()
heatmap_builder = HeatmapBuilder()
timing_analyzer = TimingAnalyzer()
backtest_simulator = BacktestSimulator()
strategy_classifier = StrategyClassifier()


def api_response(data=None, error=None, status_code=200):
    """Create consistent API response format."""
    response = {
        'success': error is None,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    if data is not None:
        response['data'] = data
    if error is not None:
        response['error'] = error
    return jsonify(response), status_code


def handle_errors(f):
    """Decorator for consistent error handling."""
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            return api_response(
                error={
                    'code': 'INTERNAL_ERROR',
                    'message': str(e)
                },
                status_code=500
            )
    return decorated


# =============================================================================
# TRADER ENDPOINTS
# =============================================================================

@app.route('/api/traders', methods=['GET'])
@handle_errors
def list_traders():
    """
    List all cached traders.

    Query params:
        limit: Max traders to return (default: 100)
        sort: Sort field (total_pnl, roi_pct, win_rate, total_trades)
    """
    limit = request.args.get('limit', 100, type=int)
    sort_by = request.args.get('sort', 'total_pnl')

    traders = cache.get_filtered_traders(
        sort_by=sort_by,
        limit=limit
    )

    return api_response(data={
        'traders': traders,
        'count': len(traders)
    })


@app.route('/api/traders/<address>', methods=['GET'])
@handle_errors
def get_trader(address):
    """Get single trader profile from cache."""
    trader = cache.load_trader(address)

    if not trader:
        return api_response(
            error={
                'code': 'NOT_FOUND',
                'message': f'Trader {address} not found in cache'
            },
            status_code=404
        )

    return api_response(data=trader)


@app.route('/api/traders/<address>/analyze', methods=['POST'])
@handle_errors
def analyze_trader_endpoint(address):
    """
    Analyze trader (fetches fresh data from API).

    Body params:
        refresh: Force refresh even if cached (default: false)
        stream: Use SSE streaming for progress (default: false)
    """
    data = request.get_json() or {}
    refresh = data.get('refresh', False)
    stream = data.get('stream', False)

    # Check cache first
    if not refresh:
        cached = cache.load_trader(address)
        if cached:
            return api_response(data={
                'trader': cached,
                'from_cache': True
            })

    if stream:
        # Stream progress updates via SSE
        def generate():
            yield f"data: {json.dumps({'status': 'starting', 'progress': 0, 'message': 'Fetching trades...'})}\n\n"

            try:
                # Analyze trader
                trader = analyzer.analyze_trader(address=address)
                yield f"data: {json.dumps({'status': 'progress', 'progress': 50, 'message': 'Building equity curve...'})}\n\n"

                # Build equity curve
                trader.equity_curve = curve_builder.build(trader)
                trader.stats.max_drawdown = curve_builder.calculate_max_drawdown(trader.equity_curve)
                yield f"data: {json.dumps({'status': 'progress', 'progress': 80, 'message': 'Saving to cache...'})}\n\n"

                # Save to cache
                cache.save_trader(trader)

                trader_dict = trader.to_dict()
                yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'data': trader_dict})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    else:
        # Synchronous analysis
        trader = analyzer.analyze_trader(address=address)
        trader.equity_curve = curve_builder.build(trader)
        trader.stats.max_drawdown = curve_builder.calculate_max_drawdown(trader.equity_curve)
        cache.save_trader(trader)

        return api_response(data={
            'trader': trader.to_dict(),
            'from_cache': False
        })


@app.route('/api/traders/<address>', methods=['DELETE'])
@handle_errors
def delete_trader(address):
    """Delete trader from cache."""
    deleted = cache.delete_trader(address)

    if not deleted:
        return api_response(
            error={
                'code': 'NOT_FOUND',
                'message': f'Trader {address} not found in cache'
            },
            status_code=404
        )

    return api_response(data={'deleted': True, 'address': address})


@app.route('/api/traders/<address>/strategy', methods=['GET'])
@handle_errors
def get_trader_strategy(address):
    """Run deep strategy analysis for a trader."""
    refresh = request.args.get('refresh', 'false').lower() == 'true'

    report = strategy_analyzer.analyze_strategy(
        address=address,
        refresh=refresh
    )

    return api_response(data=report.to_dict())


# =============================================================================
# STATS ENDPOINT
# =============================================================================

@app.route('/api/stats', methods=['GET'])
@handle_errors
def get_stats():
    """Get aggregate statistics about cached data."""
    stats = cache.get_stats()
    return api_response(data=stats)


# =============================================================================
# FILTER ENDPOINT
# =============================================================================

@app.route('/api/filter', methods=['POST'])
@handle_errors
def filter_traders():
    """
    Filter and rank traders by criteria.

    Body params:
        min_trades: Minimum number of trades
        min_pnl: Minimum P&L in USD
        min_roi: Minimum ROI percentage
        min_win_rate: Minimum win rate percentage
        sort: Sort field (total_pnl, roi_pct, win_rate, total_trades)
        limit: Max results (default: 50)
    """
    data = request.get_json() or {}

    filtered = cache.get_filtered_traders(
        min_trades=data.get('min_trades'),
        min_pnl=data.get('min_pnl'),
        min_roi=data.get('min_roi'),
        min_win_rate=data.get('min_win_rate'),
        sort_by=data.get('sort', 'total_pnl'),
        limit=data.get('limit', 50)
    )

    return api_response(data={
        'traders': filtered,
        'count': len(filtered),
        'filters_applied': {
            k: v for k, v in data.items() if v is not None
        }
    })


# =============================================================================
# SCAN ENDPOINT
# =============================================================================

@app.route('/api/scan', methods=['POST'])
@handle_errors
def scan_markets():
    """
    Scan markets by category and discover traders.

    Body params:
        category: Market category (nba, nfl, crypto, politics, etc.)
        search: Additional keyword filter
        markets: Max markets to scan (default: 20)
        traders_per_market: Max traders per market (default: 50)
        include_closed: Include closed markets (default: false)
        stream: Use SSE streaming for progress (default: false)
    """
    data = request.get_json() or {}

    category = data.get('category')
    search = data.get('search')
    num_markets = data.get('markets', 20)
    traders_per_market = data.get('traders_per_market', 50)
    include_closed = data.get('include_closed', False)
    stream = data.get('stream', False)

    if stream:
        def generate():
            yield f"data: {json.dumps({'status': 'starting', 'progress': 0, 'message': 'Scanning markets...'})}\n\n"

            try:
                results = discovery.discover_from_category(
                    category=category,
                    search_pattern=search,
                    num_markets=num_markets,
                    traders_per_market=traders_per_market,
                    include_closed=include_closed
                )

                yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'data': {'markets_count': len(results['markets']), 'traders_count': len(results['traders_list'])}})}\n\n"

                # Full data
                export_data = {
                    'category': results.get('category'),
                    'search_pattern': search,
                    'markets': [
                        {
                            'slug': m['slug'],
                            'title': m.get('title', ''),
                            'volume': m.get('volume', 0),
                            'trader_count': m.get('trader_count', 0)
                        }
                        for m in results['markets']
                    ],
                    'traders': [
                        {
                            'address': t['address'],
                            'name': t.get('name'),
                            'pseudonym': t.get('pseudonym'),
                            'markets_count': len(t.get('markets', [])),
                            'total_volume': t.get('total_volume', 0)
                        }
                        for t in results['traders_list'][:100]
                    ]
                }
                yield f"data: {json.dumps({'status': 'data', 'data': export_data})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    else:
        results = discovery.discover_from_category(
            category=category,
            search_pattern=search,
            num_markets=num_markets,
            traders_per_market=traders_per_market,
            include_closed=include_closed
        )

        return api_response(data={
            'category': results.get('category'),
            'search_pattern': search,
            'markets_count': len(results['markets']),
            'traders_count': len(results['traders_list']),
            'markets': [
                {
                    'slug': m['slug'],
                    'title': m.get('title', ''),
                    'volume': m.get('volume', 0),
                    'trader_count': m.get('trader_count', 0)
                }
                for m in results['markets']
            ],
            'traders': [
                {
                    'address': t['address'],
                    'name': t.get('name'),
                    'pseudonym': t.get('pseudonym'),
                    'markets_count': len(t.get('markets', [])),
                    'total_volume': t.get('total_volume', 0)
                }
                for t in results['traders_list'][:100]
            ]
        })


# =============================================================================
# CORRELATE ENDPOINT
# =============================================================================

@app.route('/api/correlate', methods=['POST'])
@handle_errors
def correlate_traders():
    """
    Run correlation analysis and find smart money clusters.

    Body params:
        category: Market category
        search: Additional keyword filter
        markets: Max markets to scan (default: 20)
        min_overlap: Min shared markets for cluster (default: 2)
        smart_money: Use smart money preset filters (default: false)
        min_pnl: Minimum P&L filter
        min_roi: Minimum ROI filter
        min_win_rate: Minimum win rate filter
        min_trades: Minimum trades filter
        analyze_limit: Max traders to analyze (default: 50)
    """
    data = request.get_json() or {}

    # Handle smart_money preset
    min_pnl = data.get('min_pnl', 0)
    min_roi = data.get('min_roi', 0)
    min_win_rate = data.get('min_win_rate', 0)
    min_trades = data.get('min_trades', 0)

    if data.get('smart_money'):
        if min_pnl == 0:
            min_pnl = 1000
        if min_roi == 0:
            min_roi = 5
        if min_win_rate == 0:
            min_win_rate = 55
        if min_trades == 0:
            min_trades = 5

    results = correlator.analyze_from_scan(
        category=data.get('category'),
        search_pattern=data.get('search'),
        num_markets=data.get('markets', 20),
        traders_per_market=data.get('traders_per_market', 100),
        min_shared_markets=data.get('min_overlap', 2),
        include_closed=data.get('include_closed', False),
        min_pnl=min_pnl,
        min_roi=min_roi,
        min_win_rate=min_win_rate,
        min_trades=min_trades,
        analyze_limit=data.get('analyze_limit', 50)
    )

    # Serialize clusters and consensus using to_dict()
    return api_response(data={
        'category': results['scan'].get('category'),
        'markets_scanned': len(results['scan']['markets']),
        'traders_found': len(results['scan']['traders']),
        'clusters': [c.to_dict() for c in results['clusters']],
        'market_consensus': [mc.to_dict() for mc in results['market_consensus']],
        'smart_money_traders': results.get('smart_money_traders', []),
        'filters_applied': results.get('filters_applied', {})
    })


# =============================================================================
# COPY TRADING ENDPOINT
# =============================================================================

@app.route('/api/copy/signals', methods=['GET'])
@handle_errors
def get_copy_signals():
    """
    Get current positions from specified traders (for copy trading).

    Query params:
        traders: Comma-separated list of trader addresses
    """
    traders_param = request.args.get('traders', '')

    if not traders_param:
        return api_response(
            error={
                'code': 'BAD_REQUEST',
                'message': 'traders parameter is required (comma-separated addresses)'
            },
            status_code=400
        )

    addresses = [a.strip() for a in traders_param.split(',') if a.strip()]

    signals = []
    for address in addresses:
        trader = cache.load_trader(address)
        if trader and 'positions' in trader:
            # Get open positions
            open_positions = [
                {
                    'trader': address,
                    'trader_name': trader.get('name') or trader.get('pseudonym'),
                    'market_slug': pos.get('market_slug'),
                    'market_title': pos.get('market_title'),
                    'outcome': pos.get('outcome'),
                    'shares': pos.get('total_shares'),
                    'avg_price': pos.get('avg_entry_price'),
                    'current_price': pos.get('current_price'),
                    'unrealized_pnl': pos.get('unrealized_pnl')
                }
                for pos in trader['positions'].values()
                if not pos.get('is_resolved', True)
            ]
            signals.extend(open_positions)

    # Group by market
    markets = {}
    for sig in signals:
        slug = sig['market_slug']
        if slug not in markets:
            markets[slug] = {
                'market_slug': slug,
                'market_title': sig['market_title'],
                'positions': []
            }
        markets[slug]['positions'].append(sig)

    return api_response(data={
        'signals': list(markets.values()),
        'total_positions': len(signals),
        'traders_checked': len(addresses)
    })


# =============================================================================
# CATEGORIES ENDPOINT
# =============================================================================

@app.route('/api/categories', methods=['GET'])
@handle_errors
def get_categories():
    """Get available market categories for scanning."""
    return api_response(data={
        'categories': list(MARKET_CATEGORIES.keys())
    })


# =============================================================================
# FILTER PRESETS ENDPOINTS
# =============================================================================

@app.route('/api/filters/presets', methods=['GET'])
@handle_errors
def get_filter_presets():
    """Get pre-built filter profiles."""
    return api_response(data={
        'presets': FILTER_PRESETS
    })


@app.route('/api/filters/save', methods=['POST'])
@handle_errors
def save_filter():
    """
    Save a custom filter configuration.

    Body params:
        name: Filter name
        description: Filter description
        criteria: Dict with min_pnl, min_roi, min_win_rate, min_trades, etc.
    """
    data = request.get_json() or {}

    if not data.get('name'):
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'Filter name is required'},
            status_code=400
        )

    filter_id = cache.save_filter(data)
    return api_response(data={'id': filter_id, 'saved': True})


@app.route('/api/filters/saved', methods=['GET'])
@handle_errors
def list_saved_filters():
    """List all saved custom filters."""
    filters = cache.load_filters()
    return api_response(data={
        'filters': filters,
        'count': len(filters)
    })


@app.route('/api/filters/<filter_id>', methods=['DELETE'])
@handle_errors
def delete_filter(filter_id):
    """Delete a saved filter."""
    deleted = cache.delete_filter(filter_id)

    if not deleted:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': f'Filter {filter_id} not found'},
            status_code=404
        )

    return api_response(data={'deleted': True, 'id': filter_id})


# =============================================================================
# BENCHMARK ENDPOINTS
# =============================================================================

@app.route('/api/benchmarks/<category>', methods=['GET'])
@handle_errors
def get_category_benchmark(category):
    """
    Get category leaderboard and benchmarks.

    Query params:
        limit: Max traders to return (default: 20)
    """
    limit = request.args.get('limit', 20, type=int)

    result = benchmark_analyzer.get_category_leaderboard(
        category=category,
        limit=limit
    )

    if 'error' in result:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': result['error']},
            status_code=400
        )

    return api_response(data=result)


@app.route('/api/traders/<address>/categories', methods=['GET'])
@handle_errors
def get_trader_categories(address):
    """Get trader's per-category breakdown."""
    result = benchmark_analyzer.get_trader_category_breakdown(address)

    if 'error' in result:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': result['error']},
            status_code=404
        )

    return api_response(data=result)


# =============================================================================
# COMPARISON ENDPOINTS
# =============================================================================

@app.route('/api/compare/traders', methods=['POST'])
@handle_errors
def compare_traders_endpoint():
    """
    Compare multiple traders side-by-side.

    Body params:
        addresses: List of trader addresses (2-10)
    """
    data = request.get_json() or {}
    addresses = data.get('addresses', [])

    if len(addresses) < 2:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'At least 2 addresses required'},
            status_code=400
        )

    if len(addresses) > 10:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'Maximum 10 addresses allowed'},
            status_code=400
        )

    result = comparison_analyzer.compare_traders(addresses)
    return api_response(data=result)


@app.route('/api/compare/divergence', methods=['POST'])
@handle_errors
def find_divergence_endpoint():
    """
    Find markets where traders disagree.

    Body params:
        addresses: List of trader addresses (2-10)
    """
    data = request.get_json() or {}
    addresses = data.get('addresses', [])

    if len(addresses) < 2:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'At least 2 addresses required'},
            status_code=400
        )

    result = comparison_analyzer.find_divergence(addresses)
    return api_response(data=result)


# =============================================================================
# WATCHLIST ENDPOINTS
# =============================================================================

@app.route('/api/watchlist', methods=['POST'])
@handle_errors
def create_watchlist():
    """
    Create a new watchlist.

    Body params:
        name: Watchlist name
        traders: List of trader addresses
        alerts: Optional dict of alert configurations
    """
    data = request.get_json() or {}

    if not data.get('name'):
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'Watchlist name is required'},
            status_code=400
        )

    watchlist = watchlist_service.create_watchlist(
        name=data['name'],
        traders=data.get('traders', []),
        alerts=data.get('alerts', {})
    )

    return api_response(data=watchlist.to_dict())


@app.route('/api/watchlist', methods=['GET'])
@handle_errors
def list_watchlists():
    """List all watchlists."""
    watchlists = watchlist_service.list_watchlists()
    return api_response(data={
        'watchlists': [w.to_dict() for w in watchlists],
        'count': len(watchlists)
    })


@app.route('/api/watchlist/<watchlist_id>', methods=['GET'])
@handle_errors
def get_watchlist(watchlist_id):
    """Get a specific watchlist."""
    watchlist = watchlist_service.get_watchlist(watchlist_id)

    if not watchlist:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': f'Watchlist {watchlist_id} not found'},
            status_code=404
        )

    return api_response(data=watchlist.to_dict())


@app.route('/api/watchlist/<watchlist_id>', methods=['PUT'])
@handle_errors
def update_watchlist(watchlist_id):
    """
    Update a watchlist.

    Body params:
        name: New name (optional)
        traders: New trader list (optional)
        alerts: New alert config (optional)
    """
    data = request.get_json() or {}

    watchlist = watchlist_service.update_watchlist(
        watchlist_id=watchlist_id,
        name=data.get('name'),
        traders=data.get('traders'),
        alerts=data.get('alerts')
    )

    if not watchlist:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': f'Watchlist {watchlist_id} not found'},
            status_code=404
        )

    return api_response(data=watchlist.to_dict())


@app.route('/api/watchlist/<watchlist_id>', methods=['DELETE'])
@handle_errors
def delete_watchlist(watchlist_id):
    """Delete a watchlist."""
    deleted = watchlist_service.delete_watchlist(watchlist_id)

    if not deleted:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': f'Watchlist {watchlist_id} not found'},
            status_code=404
        )

    return api_response(data={'deleted': True, 'id': watchlist_id})


@app.route('/api/watchlist/<watchlist_id>/status', methods=['GET'])
@handle_errors
def get_watchlist_status(watchlist_id):
    """Get current status of watched traders."""
    result = watchlist_service.get_watchlist_status(watchlist_id)

    if 'error' in result:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': result['error']},
            status_code=404
        )

    return api_response(data=result)


@app.route('/api/alerts/configure', methods=['POST'])
@handle_errors
def configure_alerts():
    """
    Configure alerts for a watchlist.

    Body params:
        watchlist_id: Watchlist ID
        alerts: Dict of alert type -> threshold
    """
    data = request.get_json() or {}

    if not data.get('watchlist_id'):
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'watchlist_id is required'},
            status_code=400
        )

    result = watchlist_service.configure_alerts(
        watchlist_id=data['watchlist_id'],
        alerts=data.get('alerts', {})
    )

    if 'error' in result:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': result['error']},
            status_code=404
        )

    return api_response(data=result)


@app.route('/api/alerts/history', methods=['GET'])
@handle_errors
def get_alert_history():
    """
    Get triggered alerts history.

    Query params:
        watchlist_id: Filter by watchlist (optional)
        limit: Max alerts to return (default: 50)
    """
    watchlist_id = request.args.get('watchlist_id')
    limit = request.args.get('limit', 50, type=int)

    alerts = watchlist_service.get_alert_history(
        watchlist_id=watchlist_id,
        limit=limit
    )

    return api_response(data={
        'alerts': [a.to_dict() for a in alerts],
        'count': len(alerts)
    })


# =============================================================================
# HEATMAP ENDPOINTS
# =============================================================================

@app.route('/api/heatmap/category/<category>', methods=['GET'])
@handle_errors
def get_category_heatmap(category):
    """
    Get position heat map for a market category.

    Query params:
        min_traders: Minimum traders per market (default: 2)
        smart_money: Filter to profitable traders only (default: false)
        min_pnl: Minimum P&L for smart money filter
        min_win_rate: Minimum win rate for smart money filter
    """
    min_traders = request.args.get('min_traders', 2, type=int)
    smart_money = request.args.get('smart_money', 'false').lower() == 'true'
    min_pnl = request.args.get('min_pnl', 0, type=float)
    min_win_rate = request.args.get('min_win_rate', 0, type=float)

    result = heatmap_builder.get_category_heatmap(
        category=category,
        min_traders=min_traders,
        smart_money_only=smart_money,
        min_pnl=min_pnl,
        min_win_rate=min_win_rate
    )

    return api_response(data=result)


@app.route('/api/heatmap/markets', methods=['POST'])
@handle_errors
def get_markets_heatmap():
    """
    Get position heat map for specific markets.

    Body params:
        market_slugs: List of market slugs (optional, None for all)
        min_traders: Minimum traders per market (default: 1)
    """
    data = request.get_json() or {}

    result = heatmap_builder.get_market_heatmap(
        market_slugs=data.get('market_slugs'),
        min_traders=data.get('min_traders', 1)
    )

    return api_response(data=result)


@app.route('/api/heatmap/consensus', methods=['GET'])
@handle_errors
def get_consensus_summary():
    """Get overall consensus summary across all categories."""
    result = heatmap_builder.get_consensus_summary()
    return api_response(data=result)


# =============================================================================
# TIMING & MOMENTUM ENDPOINTS
# =============================================================================

@app.route('/api/timing/analyze', methods=['POST'])
@handle_errors
def analyze_timing():
    """
    Analyze entry timing quality for a trader.

    Body params:
        address: Trader wallet address
    """
    data = request.get_json() or {}
    address = data.get('address')

    if not address:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'address is required'},
            status_code=400
        )

    result = timing_analyzer.analyze_timing(address)

    if 'error' in result:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': result['error']},
            status_code=404
        )

    return api_response(data=result)


@app.route('/api/momentum/detect', methods=['POST'])
@handle_errors
def detect_momentum():
    """
    Detect markets where smart money is entering recently.

    Body params:
        category: Market category (optional)
        time_window_hours: Look for entries within this window (default: 24)
        min_traders: Minimum traders entering (default: 3)
        min_win_rate: Minimum win rate for smart money (default: 55)
    """
    data = request.get_json() or {}

    result = timing_analyzer.detect_momentum(
        category=data.get('category'),
        time_window_hours=data.get('time_window_hours', 24),
        min_traders=data.get('min_traders', 3),
        min_win_rate=data.get('min_win_rate', 55)
    )

    return api_response(data=result)


@app.route('/api/timing/leaderboard', methods=['GET'])
@handle_errors
def get_timing_leaderboard():
    """
    Get traders ranked by entry timing quality.

    Query params:
        limit: Max traders to return (default: 20)
    """
    limit = request.args.get('limit', 20, type=int)
    result = timing_analyzer.get_entry_quality_leaderboard(limit=limit)
    return api_response(data=result)


# =============================================================================
# BACKTESTING & PORTFOLIO ENDPOINTS
# =============================================================================

@app.route('/api/backtest/trader', methods=['POST'])
@handle_errors
def backtest_trader():
    """
    Backtest a trader with modified parameters.

    Body params:
        address: Trader wallet address
        modifications: Dict with:
            - position_size_multiplier: Scale all positions
            - take_profit_percent: Exit when profit hits this %
            - stop_loss_percent: Exit when loss hits this %
            - max_positions: Max concurrent positions
    """
    data = request.get_json() or {}
    address = data.get('address')

    if not address:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'address is required'},
            status_code=400
        )

    result = backtest_simulator.backtest_trader(
        address=address,
        modifications=data.get('modifications', {})
    )

    if 'error' in result:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': result['error']},
            status_code=404
        )

    return api_response(data=result)


@app.route('/api/portfolio/simulate', methods=['POST'])
@handle_errors
def simulate_portfolio():
    """
    Simulate a portfolio following multiple traders.

    Body params:
        traders: List of {address, weight} dicts
        initial_capital: Starting capital in USD (default: 10000)
        max_position_pct: Max single position as % of capital (default: 0.1)
    """
    data = request.get_json() or {}
    traders = data.get('traders', [])

    if not traders:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'traders list is required'},
            status_code=400
        )

    result = backtest_simulator.simulate_portfolio(
        traders=traders,
        initial_capital=data.get('initial_capital', 10000),
        max_position_pct=data.get('max_position_pct', 0.1)
    )

    if 'error' in result:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': result['error']},
            status_code=400
        )

    return api_response(data=result)


@app.route('/api/portfolio/optimize', methods=['POST'])
@handle_errors
def optimize_portfolio():
    """
    Find optimal weights for a portfolio of traders.

    Body params:
        addresses: List of trader addresses
        initial_capital: Starting capital (default: 10000)
        optimization_target: What to optimize for (sharpe, roi, risk_adjusted)
    """
    data = request.get_json() or {}
    addresses = data.get('addresses', [])

    if len(addresses) < 2:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'At least 2 addresses required'},
            status_code=400
        )

    result = backtest_simulator.optimize_weights(
        addresses=addresses,
        initial_capital=data.get('initial_capital', 10000),
        optimization_target=data.get('optimization_target', 'sharpe')
    )

    if 'error' in result:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': result['error']},
            status_code=400
        )

    return api_response(data=result)


# =============================================================================
# STRATEGY CLASSIFICATION ENDPOINTS
# =============================================================================

@app.route('/api/strategies/types', methods=['GET'])
@handle_errors
def get_strategy_types():
    """Get all available strategy types with descriptions."""
    types = strategy_classifier.get_strategy_types()
    return api_response(data={
        'strategy_types': types,
        'count': len(types)
    })


@app.route('/api/strategies/trader/<address>', methods=['GET'])
@handle_errors
def classify_trader_strategy(address):
    """
    Classify a trader's strategy based on transaction patterns.

    Returns detailed strategy profile with confidence scores.
    """
    # Load trader from cache
    trader_data = cache.load_trader(address)

    if not trader_data:
        return api_response(
            error={'code': 'NOT_FOUND', 'message': f'Trader {address} not found in cache'},
            status_code=404
        )

    # Convert dict to Trader object if needed
    from models.trader import Trader, TraderStats
    from models.position import Position

    if isinstance(trader_data, dict):
        trader = Trader(address=trader_data.get('address', address))
        trader.name = trader_data.get('name')
        trader.pseudonym = trader_data.get('pseudonym')

        # Load stats
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

        # Load positions
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
    else:
        trader = trader_data

    # Classify
    profile = strategy_classifier.classify(trader)

    return api_response(data=profile.to_dict())


@app.route('/api/strategies/find/<strategy_type>', methods=['GET'])
@handle_errors
def find_traders_by_strategy(strategy_type):
    """
    Find traders using a specific strategy.

    Query params:
        min_confidence: Minimum confidence threshold (default: 0.3)
        limit: Max results (default: 20)
    """
    min_confidence = request.args.get('min_confidence', 0.3, type=float)
    limit = request.args.get('limit', 20, type=int)

    # Validate strategy type
    valid_types = ['arbitrage', 'insider', 'lazy_positions', 'momentum']
    if strategy_type not in valid_types:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': f'Invalid strategy type. Valid: {valid_types}'},
            status_code=400
        )

    # Load all traders and convert to Trader objects
    from models.trader import Trader, TraderStats
    from models.position import Position

    traders = []
    all_cached = cache.load_all_traders()

    for trader_data in all_cached:
        if isinstance(trader_data, dict):
            trader = Trader(address=trader_data.get('address', ''))
            trader.name = trader_data.get('name')
            trader.pseudonym = trader_data.get('pseudonym')

            stats_data = trader_data.get('stats', {})
            trader.stats = TraderStats(
                total_pnl=stats_data.get('total_pnl', 0),
                realized_pnl=stats_data.get('realized_pnl', 0),
                total_volume=stats_data.get('total_volume', 0),
                total_trades=stats_data.get('total_trades', 0),
                unique_markets=stats_data.get('unique_markets', 0),
                roi_pct=stats_data.get('roi_pct', 0),
                win_rate=stats_data.get('win_rate', 0),
                winning_positions=stats_data.get('winning_positions', 0),
                losing_positions=stats_data.get('losing_positions', 0),
                active_days=stats_data.get('active_days', 0),
                avg_position_size=stats_data.get('avg_position_size', 0),
                max_position_size=stats_data.get('max_position_size', 0)
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
                    is_resolved=pos_data.get('is_resolved', False),
                    resolved_price=pos_data.get('resolved_price')
                )
                trader.positions[key] = pos
            traders.append(trader)
        else:
            traders.append(trader_data)

    # Find by strategy
    profiles = strategy_classifier.find_by_strategy(
        traders=traders,
        strategy=strategy_type,
        min_confidence=min_confidence
    )

    # Sort by confidence and limit
    profiles.sort(key=lambda p: p.primary_confidence, reverse=True)
    profiles = profiles[:limit]

    return api_response(data={
        'strategy_type': strategy_type,
        'traders': [p.to_dict() for p in profiles],
        'count': len(profiles),
        'min_confidence': min_confidence
    })


@app.route('/api/strategies/analyze', methods=['POST'])
@handle_errors
def analyze_strategies():
    """
    Analyze strategies for specific traders.

    Body params:
        addresses: List of trader addresses
    """
    data = request.get_json() or {}
    addresses = data.get('addresses', [])

    if not addresses:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': 'addresses list is required'},
            status_code=400
        )

    from models.trader import Trader, TraderStats
    from models.position import Position

    results = []

    for address in addresses[:20]:  # Limit to 20
        trader_data = cache.load_trader(address)
        if not trader_data:
            continue

        if isinstance(trader_data, dict):
            trader = Trader(address=trader_data.get('address', address))
            trader.name = trader_data.get('name')
            trader.pseudonym = trader_data.get('pseudonym')

            stats_data = trader_data.get('stats', {})
            trader.stats = TraderStats(
                total_pnl=stats_data.get('total_pnl', 0),
                roi_pct=stats_data.get('roi_pct', 0),
                win_rate=stats_data.get('win_rate', 0),
                total_trades=stats_data.get('total_trades', 0),
                unique_markets=stats_data.get('unique_markets', 0),
                avg_position_size=stats_data.get('avg_position_size', 0)
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
                    is_resolved=pos_data.get('is_resolved', False),
                    resolved_price=pos_data.get('resolved_price')
                )
                trader.positions[key] = pos
        else:
            trader = trader_data

        profile = strategy_classifier.classify(trader)
        results.append(profile.to_dict())

    return api_response(data={
        'profiles': results,
        'count': len(results)
    })


@app.route('/api/strategies/leaderboard/<strategy_type>', methods=['GET'])
@handle_errors
def get_strategy_leaderboard(strategy_type):
    """
    Get top traders for a specific strategy.

    Query params:
        limit: Max results (default: 10)
    """
    limit = request.args.get('limit', 10, type=int)

    valid_types = ['arbitrage', 'insider', 'lazy_positions', 'momentum']
    if strategy_type not in valid_types:
        return api_response(
            error={'code': 'BAD_REQUEST', 'message': f'Invalid strategy type. Valid: {valid_types}'},
            status_code=400
        )

    from models.trader import Trader, TraderStats
    from models.position import Position

    traders = []
    all_cached = cache.load_all_traders()

    for trader_data in all_cached:
        if isinstance(trader_data, dict):
            trader = Trader(address=trader_data.get('address', ''))
            trader.name = trader_data.get('name')
            trader.pseudonym = trader_data.get('pseudonym')

            stats_data = trader_data.get('stats', {})
            trader.stats = TraderStats(
                total_pnl=stats_data.get('total_pnl', 0),
                roi_pct=stats_data.get('roi_pct', 0),
                win_rate=stats_data.get('win_rate', 0),
                total_trades=stats_data.get('total_trades', 0)
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
                    is_resolved=pos_data.get('is_resolved', False),
                    resolved_price=pos_data.get('resolved_price')
                )
                trader.positions[key] = pos
            traders.append(trader)
        else:
            traders.append(trader_data)

    leaderboard = strategy_classifier.get_leaderboard(
        traders=traders,
        strategy=strategy_type,
        limit=limit
    )

    return api_response(data={
        'strategy_type': strategy_type,
        'leaderboard': leaderboard,
        'count': len(leaderboard)
    })


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return api_response(data={'status': 'healthy'})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("POLYMARKET TRADER DISCOVERY API")
    print("=" * 70)
    print("Starting Flask server...")
    print("API endpoints available at http://localhost:5000/api/")
    print("")
    print("CORE ENDPOINTS:")
    print("  GET  /api/traders              - List cached traders")
    print("  GET  /api/traders/<addr>       - Get trader profile")
    print("  POST /api/traders/<addr>/analyze - Analyze trader")
    print("  DEL  /api/traders/<addr>       - Delete from cache")
    print("  GET  /api/traders/<addr>/strategy - Deep strategy analysis")
    print("  GET  /api/traders/<addr>/categories - Category breakdown")
    print("  GET  /api/stats                - Cache statistics")
    print("  POST /api/filter               - Filter traders")
    print("  POST /api/scan                 - Scan markets by category")
    print("  POST /api/correlate            - Correlation analysis")
    print("  GET  /api/copy/signals         - Copy trading signals")
    print("  GET  /api/categories           - Available categories")
    print("")
    print("FILTER PRESETS:")
    print("  GET  /api/filters/presets      - Get pre-built filters")
    print("  POST /api/filters/save         - Save custom filter")
    print("  GET  /api/filters/saved        - List saved filters")
    print("  DEL  /api/filters/<id>         - Delete saved filter")
    print("")
    print("BENCHMARKS & COMPARISON:")
    print("  GET  /api/benchmarks/<cat>     - Category leaderboard")
    print("  POST /api/compare/traders      - Compare traders")
    print("  POST /api/compare/divergence   - Find divergent positions")
    print("")
    print("WATCHLIST & ALERTS:")
    print("  POST /api/watchlist            - Create watchlist")
    print("  GET  /api/watchlist            - List watchlists")
    print("  GET  /api/watchlist/<id>       - Get watchlist")
    print("  PUT  /api/watchlist/<id>       - Update watchlist")
    print("  DEL  /api/watchlist/<id>       - Delete watchlist")
    print("  GET  /api/watchlist/<id>/status - Watchlist status")
    print("  POST /api/alerts/configure     - Configure alerts")
    print("  GET  /api/alerts/history       - Alert history")
    print("")
    print("HEATMAPS & CONSENSUS:")
    print("  GET  /api/heatmap/category/<cat> - Category heatmap")
    print("  POST /api/heatmap/markets      - Markets heatmap")
    print("  GET  /api/heatmap/consensus    - Consensus summary")
    print("")
    print("TIMING & MOMENTUM:")
    print("  POST /api/timing/analyze       - Entry timing analysis")
    print("  POST /api/momentum/detect      - Detect smart money momentum")
    print("  GET  /api/timing/leaderboard   - Timing quality rankings")
    print("")
    print("BACKTESTING & PORTFOLIO:")
    print("  POST /api/backtest/trader      - Backtest trader strategy")
    print("  POST /api/portfolio/simulate   - Simulate portfolio")
    print("  POST /api/portfolio/optimize   - Optimize portfolio weights")
    print("")
    print("STRATEGY CLASSIFICATION:")
    print("  GET  /api/strategies/types     - List strategy types")
    print("  GET  /api/strategies/trader/<addr> - Classify trader strategy")
    print("  GET  /api/strategies/find/<type>   - Find traders by strategy")
    print("  POST /api/strategies/analyze   - Analyze multiple traders")
    print("  GET  /api/strategies/leaderboard/<type> - Strategy leaderboard")
    print("")
    print("  GET  /api/health               - Health check")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5000)
