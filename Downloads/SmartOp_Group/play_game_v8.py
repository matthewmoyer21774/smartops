"""
Policy V8: Mean-Matching with Hard Cap.

The simplest possible data-driven policy: order the Poisson mean forecast
for t+2, minus net inventory position. Hard cap prevents over-ordering.

V2 ordered 100 units for 101 total demand — essentially mean-matching.
This tests whether quantile models add value or just add noise.

Usage:
    python play_game_v8.py
"""
import sys
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory


def compute_recommended_order(inv, forecaster, test_rows, period):
    """
    Mean-matching: order the mean forecast minus inventory position.
    Hard cap at q75 to prevent over-ordering on noisy forecasts.
    """
    n_periods = len(test_rows)

    if period + 2 >= n_periods:
        return 0

    mean_t2 = forecaster.predict_mean(test_rows[period + 2])
    q_t2 = forecaster.predict_quantiles(test_rows[period + 2])

    # Inventory position
    IP = inv.on_hand[0] + inv.on_hand[1] + inv.pipeline[0] + inv.pipeline[1]

    # Expected demand before order arrives (t and t+1)
    mean_t = forecaster.predict_mean(test_rows[period]) if period < n_periods else 0
    mean_t1 = forecaster.predict_mean(test_rows[period + 1]) if period + 1 < n_periods else 0
    demand_before = mean_t + mean_t1

    # Net position when order arrives
    net = IP - demand_before

    # Order = mean forecast minus what we expect to have
    order = mean_t2 - net

    # Hard cap: never order more than q75
    cap = q_t2.get(0.75, mean_t2 * 1.2)
    order = min(order, cap)

    # Taper near end
    remaining = n_periods - (period + 2)
    if remaining <= 1:
        order = min(order, q_t2.get(0.5, mean_t2) * 0.5)
    elif remaining <= 2:
        order = min(order, q_t2.get(0.5, mean_t2))

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
    print("  PERISHABLE INVENTORY GAME - V8: Mean-Matching + Hard Cap")
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
        promo = "PROMO" if row["PROMO_01"] else "     "
        state = inv.get_state()
        mean_d = forecaster.predict_mean(row)
        print(f"\n{'-'*60}")
        print(f"  PERIOD {period+1}/{n_periods}  |  {date_str}  |  {promo}")
        print(f"  On-hand: {state['on_hand']} Pipeline: {state['pipeline']} Cost: {state['total_cost']:.0f}")
        print(f"  Forecast: mean={mean_d:.1f}")
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
