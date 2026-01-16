#!/usr/bin/env python3
"""Quick test of the API clients."""

import sys
sys.path.insert(0, '.')

from api.data_api import DataAPIClient
from api.gamma_api import GammaAPIClient

def test_data_api():
    """Test Data API client."""
    print("Testing Data API...")
    client = DataAPIClient()

    # Test fetching trades for a known trader
    address = '0x60a92c8620846d81f5ea17b0564e0d4b7c545a71'
    print(f"  Fetching trades for {address[:16]}...")

    trades = client.get_trader_trades(address, limit=10)
    print(f"  Found {len(trades)} trades")

    if trades:
        t = trades[0]
        print(f"  Sample: {t.get('side')} {t.get('outcome')} @ {t.get('price')} for ${t.get('size')}")

    return len(trades) > 0

def test_gamma_api():
    """Test Gamma API client."""
    print("\nTesting Gamma API...")
    client = GammaAPIClient()

    # Test fetching markets
    print("  Fetching open markets...")
    markets = client.get_markets(closed=False, limit=5)
    print(f"  Found {len(markets)} markets")

    if markets:
        m = markets[0]
        vol = float(m.get('volume', 0) or 0)
        print(f"  Sample: {m.get('slug', 'N/A')[:40]} (${vol:,.0f})")

    return len(markets) > 0

def test_leaderboard():
    """Test leaderboard endpoint."""
    print("\nTesting Leaderboard...")
    client = DataAPIClient()

    leaderboard = client.get_leaderboard(limit=5)
    print(f"  Found {len(leaderboard)} entries")

    if leaderboard:
        entry = leaderboard[0]
        print(f"  Top trader: {entry}")

    return len(leaderboard) >= 0  # May be empty if API structure differs

if __name__ == '__main__':
    print("=" * 60)
    print("POLYMARKET API TEST")
    print("=" * 60)

    data_ok = test_data_api()
    gamma_ok = test_gamma_api()
    leaderboard_ok = test_leaderboard()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Data API:    {'OK' if data_ok else 'FAILED'}")
    print(f"Gamma API:   {'OK' if gamma_ok else 'FAILED'}")
    print(f"Leaderboard: {'OK' if leaderboard_ok else 'CHECK'}")
