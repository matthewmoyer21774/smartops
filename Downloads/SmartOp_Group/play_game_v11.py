"""
Policy V11: Zero-Waste Conservative.

Primary objective: minimize expiry to near-zero. Accepts higher shortage
cost in exchange for near-zero waste.

Targets q0.50 (median), subtracts ALL existing inventory (no discounting),
and refuses to order when well-stocked. The hypothesis: with shortage=19
and expiry=9, a policy that wastes nothing but has ~15 stockout units
(cost=285) still beats policies that waste 40+ units (cost=360+).

Usage:
    python play_game_v11.py
"""
import sys
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory


def compute_recommended_order(inv, forecaster, test_rows, period):
    """
    Zero-waste: target median, subtract all inventory, refuse when stocked.
    """
    n_periods = len(test_rows)

    if period + 2 >= n_periods:
        return 0

    mean_t2 = forecaster.predict_mean(test_rows[period + 2])
    q_t2 = forecaster.predict_quantiles(test_rows[period + 2])
    mean_t = forecaster.predict_mean(test_rows[period]) if period < n_periods else 0
    mean_t1 = forecaster.predict_mean(test_rows[period + 1]) if period + 1 < n_periods else 0

    # Total supply: everything we have and everything coming
    total_supply = (inv.on_hand[0] + inv.on_hand[1] +
                    inv.pipeline[0] + inv.pipeline[1])

    # Expected demand before arrival
    demand_before = mean_t + mean_t1

    # Net position at arrival
    net_at_arrival = total_supply - demand_before

    # Target: median demand at t+2 (conservative)
    target = q_t2.get(0.5, mean_t2)

    # If we already have enough, don't order
    if net_at_arrival >= target:
        return 0

    # If supply ratio is high, don't order
    expected_3_period = mean_t + mean_t1 + mean_t2
    if expected_3_period > 0 and total_supply / expected_3_period > 1.3:
        return 0

    order = target - net_at_arrival

    # Hard cap: never order more than the median
    order = min(order, target)

    # End-of-game: very conservative
    remaining = n_periods - (period + 2)
    if remaining <= 1:
        order = min(order, max(0, mean_t2 * 0.3 - net_at_arrival))
    elif remaining <= 2:
        order = min(order, max(0, mean_t2 * 0.5 - net_at_arrival))
    elif remaining <= 3:
        order = min(order, max(0, target * 0.7 - net_at_arrival))

    return max(0, int(round(order)))


def run_backtest(forecaster, test_rows, demands):
    original_rows = [dict(r) for r in test_rows]
    original_tail = list(forecaster.training_tail) if forecaster.training_tail else []
    forecaster.revealed_demands = []
    sim_rows = [dict(r) for r in original_rows]
    inv = PerishableInventory()
    for period in range(len(demands)):
        order = compute_recommended_order(inv, forecaster, sim_rows, period)
        inv.step(order, demands[period])
        forecaster.update_with_demand(sim_rows, period, demands[period])
    inv.summary()
    forecaster.training_tail = list(original_tail)
    forecaster.revealed_demands = []
    return inv.total_cost


def main():
    print("=" * 60)
    print("  PERISHABLE INVENTORY GAME - V11: Zero-Waste Conservative")
    print("  Target: q0.50, minimize expiry at all costs")
    print("=" * 60)
    print("\nLoading data and training demand models...")
    df = load_and_prepare()
    forecaster = DemandForecaster()
    forecaster.fit(df)
    test_rows = forecaster.get_test_features(df)
    n_periods = len(test_rows)
    print(f"Ready! {n_periods} test periods loaded.\n")
    inv = PerishableInventory(on_hand=[4, 3], pipeline=[5, 0])
    for period in range(n_periods):
        row = test_rows[period]
        date_str = row["date"].strftime("%Y-%m-%d (%a)")
        state = inv.get_state()
        mean_d = forecaster.predict_mean(row)
        q = forecaster.predict_quantiles(row)
        print(f"\n{'-'*60}")
        print(f"  PERIOD {period+1}/{n_periods}  |  {date_str}")
        print(f"  On-hand: {state['on_hand']} Pipeline: {state['pipeline']} Cost: {state['total_cost']:.0f}")
        print(f"  Forecast: mean={mean_d:.1f}, q50={q[0.5]}")
        recommended = compute_recommended_order(inv, forecaster, test_rows, period)
        print(f"  >>> RECOMMENDED ORDER: {recommended}")
        try:
            order_input = input(f"\n  Enter order quantity [{recommended}]: ").strip()
            order = int(order_input) if order_input else recommended
        except (ValueError, EOFError):
            order = recommended
        try:
            demand_input = input("  Enter revealed demand: ").strip()
            demand = int(demand_input)
        except (ValueError, EOFError):
            demand = 0
        result = inv.step(order, demand)
        forecaster.update_with_demand(test_rows, period, demand)
        print(f"  Sold: {result['sold']} Short: {result['shortage']} Expired: {result['expired']} Cost: {result['period_cost']} Total: {result['total_cost']:.0f}")
    inv.summary()


if __name__ == "__main__":
    main()
