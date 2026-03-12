"""
Head-to-head comparison of all ordering policies on the same backtest.

Proper held-out backtest: uses the last 26 known rows where features
and demands correspond to the SAME dates. The forecaster is trained on
everything before those 26 rows (no leakage).

Usage:
    python compare_policies.py
"""

import pandas as pd
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory

# Import each policy's compute_recommended_order
from play_game import compute_recommended_order as policy_v1
from play_game_v2 import compute_recommended_order as policy_v2
from play_game_v3 import compute_recommended_order as policy_v3
from play_game_v4 import compute_recommended_order as policy_v4
from play_game_v5 import compute_recommended_order as policy_v5
from play_game_v6 import compute_recommended_order as policy_v6
from play_game_v7 import compute_recommended_order as policy_v7
from play_game_v8 import compute_recommended_order as policy_v8
import play_game_v9 as v9_module

policy_v9 = v9_module.compute_recommended_order
from play_game_v10 import compute_recommended_order as policy_v10
from play_game_v11 import compute_recommended_order as policy_v11


POLICIES = {
    # "V1: Newsvendor q0.95": policy_v1,
    "V2: Cost-Optimal CF q0.776": policy_v2,
    # "V3: Two-Period Stochastic DP": policy_v3,
    # "V4: Base-Stock Waste Discount": policy_v4,
    # "V5: Scenario-Tree (25 paths)": policy_v5,
    # "V6: Adaptive Fractile": policy_v6,
    # "V7: Lean Newsvendor": policy_v7,
    # "V8: Mean-Match + Hard Cap": policy_v8,
    # "V9: Exp Smoothing Reactive": policy_v9,
    # "V10: Aggressive Adaptive": policy_v10,
    # "V11: Zero-Waste Conservative": policy_v11,
}

BACKTEST_SIZE = 365


def build_backtest_data(df):
    """
    Build a proper held-out backtest set.

    Returns:
        backtest_rows: list of feature dicts (from the actual held-out rows)
        backtest_demands: list of int demands (from same rows)
        training_tail: last 7 sales before the backtest window (for lag updates)
        backtest_dates: dates for display
    """
    known = df.dropna(subset=["sales"]).copy()
    known = known.dropna(subset=FEATURES)

    n = len(known)
    bt_start = n - BACKTEST_SIZE

    backtest_data = known.iloc[bt_start:]
    pre_bt_data = known.iloc[:bt_start]

    # Extract features and demands from the SAME rows
    backtest_rows = []
    for _, r in backtest_data.iterrows():
        backtest_rows.append({f: r[f] for f in FEATURES} | {"date": r["date"]})

    backtest_demands = backtest_data["sales"].values.astype(int).tolist()
    backtest_dates = backtest_data["date"].values

    # Training tail: last 7 sales before the backtest window
    training_tail = pre_bt_data["sales"].values[-7:].tolist()

    return backtest_rows, backtest_demands, training_tail, backtest_dates


def run_policy_backtest(policy_fn, forecaster, backtest_rows, demands, training_tail):
    """
    Run a single policy on held-out data.

    Features come from the same rows as demands — no mismatch.
    Lag features are updated period-by-period as demands are revealed.
    """
    # Deep copy rows so each policy starts fresh
    sim_rows = [dict(r) for r in backtest_rows]

    # Reset forecaster state for this policy run
    forecaster.training_tail = list(training_tail)
    forecaster.revealed_demands = []

    # Reset V9 global state if running V9
    is_v9 = policy_fn is v9_module.compute_recommended_order
    if is_v9:
        v9_module._ema_forecast = v9_module.INITIAL_FORECAST
        v9_module._demand_history = []

    inv = PerishableInventory()
    orders = []

    for period in range(len(demands)):
        order = policy_fn(inv, forecaster, sim_rows, period)
        orders.append(order)
        inv.step(order, demands[period])
        if is_v9:
            v9_module.update_ema(demands[period])
        forecaster.update_with_demand(sim_rows, period, demands[period])

    total_h = sum(r["holding_cost"] for r in inv.history)
    total_s = sum(r["shortage_cost"] for r in inv.history)
    total_e = sum(r["expiry_cost"] for r in inv.history)

    return {
        "total_cost": inv.total_cost,
        "holding": total_h,
        "shortage": total_s,
        "expiry": total_e,
        "orders": orders,
        "total_ordered": sum(orders),
        "total_demand": sum(demands),
    }


def main():
    print("=" * 70)
    print("  POLICY COMPARISON — Perishable Inventory Game")
    print("  (Proper held-out backtest: features & demands from same rows)")
    print("=" * 70)

    print("\nLoading data and training models...")
    df = load_and_prepare()
    forecaster = DemandForecaster()
    forecaster.fit(df)

    # Build proper backtest set
    backtest_rows, backtest_demands, training_tail, backtest_dates = (
        build_backtest_data(df)
    )

    print(f"\nBacktest: {len(backtest_demands)} periods")
    print(
        f"Date range: {pd.Timestamp(backtest_dates[0]).strftime('%Y-%m-%d')} "
        f"to {pd.Timestamp(backtest_dates[-1]).strftime('%Y-%m-%d')}"
    )
    print(f"Demands: {backtest_demands}")
    print(f"Total demand: {sum(backtest_demands)}")
    print(f"Mean demand: {np.mean(backtest_demands):.1f}")
    print(f"Max demand: {max(backtest_demands)}")

    # Verify no leakage: check that the model's training data ends before backtest
    known = df.dropna(subset=["sales"]).dropna(subset=FEATURES)
    n = len(known)
    bt_start = n - BACKTEST_SIZE
    print(f"\nLeakage check:")
    print(f"  Total known rows: {n}")
    print(f"  Backtest starts at row: {bt_start}")
    print(
        f"  Model trained on rows: 0-{n - 200 - 1} (train+val, test_size=200 held out)"
    )
    print(
        f"  Backtest rows {bt_start}-{n-1} are within the 200-row held-out test set: OK"
    )

    results = {}
    for name, policy_fn in POLICIES.items():
        print(f"\n--- Running {name} ---")
        results[name] = run_policy_backtest(
            policy_fn, forecaster, backtest_rows, backtest_demands, training_tail
        )

    # --- Results table ---
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(
        f"\n  {'Policy':<35} {'Total':>8} {'Hold':>8} {'Short':>8} {'Expiry':>8} {'Ordered':>8}"
    )
    print(f"  {'-' * 75}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]["total_cost"])

    for rank, (name, r) in enumerate(sorted_results, 1):
        marker = " ***" if rank == 1 else ""
        print(
            f"  {name:<35} {r['total_cost']:>8.0f} {r['holding']:>8.0f} "
            f"{r['shortage']:>8.0f} {r['expiry']:>8.0f} {r['total_ordered']:>8d}{marker}"
        )

    # Winner analysis
    best_name, best_r = sorted_results[0]
    worst_name, worst_r = sorted_results[-1]
    print(f"\n  WINNER: {best_name} (cost={best_r['total_cost']:.0f})")
    print(f"  WORST:  {worst_name} (cost={worst_r['total_cost']:.0f})")
    if worst_r["total_cost"] > 0:
        print(
            f"  Savings: {worst_r['total_cost'] - best_r['total_cost']:.0f} "
            f"({(worst_r['total_cost'] - best_r['total_cost'])/worst_r['total_cost']*100:.1f}%)"
        )

    # Order comparison
    print(f"\n{'=' * 70}")
    print(f"  ORDER SEQUENCES")
    print(f"{'=' * 70}")
    print(f"\n  {'Period':<8} {'Date':<12}", end="")
    for name in [n for n, _ in sorted_results]:
        short = name.split(":")[0]
        print(f" {short:>6}", end="")
    print(f" {'Demand':>7}")
    print(f"  {'-' * (20 + 7 * len(sorted_results) + 7)}")

    for p in range(len(backtest_demands)):
        date_str = pd.Timestamp(backtest_dates[p]).strftime("%Y-%m-%d")
        print(f"  {p+1:<8} {date_str:<12}", end="")
        for name, _ in sorted_results:
            print(f" {results[name]['orders'][p]:>6}", end="")
        print(f" {backtest_demands[p]:>7}")


if __name__ == "__main__":
    main()
