"""
Backtest for Policy V2: Cost-Optimal Critical Fractile Newsvendor.

Proper held-out backtest: uses the last 26 known rows where features
and demands correspond to the SAME dates.

Usage:
    python backtest_v2.py
"""
import pandas as pd
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory
from play_game_v2 import compute_recommended_order

BACKTEST_SIZE = 26


def main():
    print("=" * 60)
    print("  BACKTEST — V2: Cost-Optimal Critical Fractile (q0.776)")
    print("=" * 60)

    print("\nLoading data and training models...")
    df = load_and_prepare()
    forecaster = DemandForecaster()
    forecaster.fit(df)

    # Build held-out backtest set
    known = df.dropna(subset=["sales"]).copy()
    known = known.dropna(subset=FEATURES)
    n = len(known)
    bt_start = n - BACKTEST_SIZE

    backtest_data = known.iloc[bt_start:]
    pre_bt_data = known.iloc[:bt_start]

    backtest_rows = []
    for _, r in backtest_data.iterrows():
        backtest_rows.append({f: r[f] for f in FEATURES} | {"date": r["date"]})

    demands = backtest_data["sales"].values.astype(int).tolist()
    dates = backtest_data["date"].values
    training_tail = pre_bt_data["sales"].values[-7:].tolist()

    print(f"\nBacktest: {BACKTEST_SIZE} periods")
    print(f"Date range: {pd.Timestamp(dates[0]).strftime('%Y-%m-%d')} "
          f"to {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")
    print(f"Demands: {demands}")
    print(f"Total demand: {sum(demands)}, Mean: {np.mean(demands):.1f}, Max: {max(demands)}")

    # Run backtest
    forecaster.training_tail = list(training_tail)
    forecaster.revealed_demands = []
    inv = PerishableInventory()

    print(f"\n{'Period':<8} {'Date':<12} {'Order':>6} {'Demand':>7} {'Sold':>5} "
          f"{'Short':>6} {'Expired':>8} {'Cost':>6} {'Total':>7}")
    print("-" * 72)

    for period in range(len(demands)):
        order = compute_recommended_order(inv, forecaster, backtest_rows, period)
        result = inv.step(order, demands[period])
        forecaster.update_with_demand(backtest_rows, period, demands[period])

        date_str = pd.Timestamp(dates[period]).strftime("%Y-%m-%d")
        print(f"{period+1:<8} {date_str:<12} {order:>6} {demands[period]:>7} "
              f"{result['sold']:>5} {result['shortage']:>6} {result['expired']:>8} "
              f"{result['period_cost']:>6.0f} {result['total_cost']:>7.0f}")

    # Summary
    print("=" * 72)
    total_h = sum(r["holding_cost"] for r in inv.history)
    total_s = sum(r["shortage_cost"] for r in inv.history)
    total_e = sum(r["expiry_cost"] for r in inv.history)
    total_ordered = sum(r["order"] for r in inv.history)

    print(f"\n  Total Cost:     {inv.total_cost:.0f}")
    print(f"  Holding Cost:   {total_h:.0f}")
    print(f"  Shortage Cost:  {total_s:.0f}")
    print(f"  Expiry Cost:    {total_e:.0f}")
    print(f"  Total Ordered:  {total_ordered}")
    print(f"  Total Demand:   {sum(demands)}")


if __name__ == "__main__":
    main()
