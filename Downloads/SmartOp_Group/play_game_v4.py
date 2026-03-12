"""
Policy V4: Inventory-Position Base-Stock with Waste Discount.

Simple, robust policy:
1. Compute "effective inventory position" that discounts age_1 stock
   (about to expire) and pipeline arrivals.
2. Order up to a base-stock level S.

    IP_effective = age_0 + alpha * age_1 + pipeline[0] + beta * pipeline[1]
    order = max(0, S - IP_effective)

alpha < 1 because age_1 will expire next period (partial credit).
beta < 1 because pipeline[1] arrives in 2 periods (discounted).
S is the target base-stock level.

Parameters (alpha, beta, S) can be grid-searched via backtest.

Usage:
    python play_game_v4.py           # interactive play
    python play_game_v4.py backtest  # run backtest
"""
import sys
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory

# Tunable parameters
ALPHA = 0.3   # discount for age_1 (high expiry risk → low credit)
BETA = 0.7    # discount for pipeline[1] (arrives in 2 periods)
BASE_S = None  # will be set dynamically based on forecast


def compute_recommended_order(inv, forecaster, test_rows, period):
    """
    Base-stock policy with waste-discounted inventory position.

    The base-stock level S is set dynamically based on the demand forecast
    at the arrival period (t+2) plus a safety buffer.
    """
    n_periods = len(test_rows)

    if period + 2 >= n_periods:
        return 0

    def get_forecasts(p):
        if 0 <= p < n_periods:
            return forecaster.predict_mean(test_rows[p]), forecaster.predict_quantiles(test_rows[p])
        return 0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0}

    mean_t2, q_t2 = get_forecasts(period + 2)
    mean_t3, q_t3 = get_forecasts(period + 3) if period + 3 < n_periods else (0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0})

    # Dynamic base-stock level: cover demand over the order's lifetime
    # Use q75 for the arrival period + median for the second period
    can_serve = min(2, n_periods - (period + 2))

    if can_serve <= 0:
        return 0
    elif can_serve == 1:
        S = q_t2.get(0.75, mean_t2 * 1.2)
    else:
        # Cover both periods the order serves, but second period at lower quantile
        S = q_t2.get(0.75, mean_t2 * 1.2) + q_t3.get(0.5, mean_t3) * 0.5

    # Effective inventory position (waste-discounted)
    IP = (inv.on_hand[0] +
          ALPHA * inv.on_hand[1] +
          inv.pipeline[0] +
          BETA * inv.pipeline[1])

    # Subtract expected demand consumed before order arrives (t and t+1)
    mean_t, _ = get_forecasts(period)
    mean_t1, _ = get_forecasts(period + 1)
    demand_before_arrival = mean_t + mean_t1

    # Net position: what we expect to have when order arrives
    net_position = IP - demand_before_arrival

    order = max(0, S - net_position)
    return max(0, int(round(order)))


def run_backtest(forecaster, test_rows, demands):
    """Run backtest with known demands."""
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
    print("  PERISHABLE INVENTORY GAME - V4: Base-Stock + Waste Discount")
    print(f"  alpha={ALPHA}, beta={BETA}")
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
        price = row["PRC_2_norm"]

        state = inv.get_state()
        q = forecaster.predict_quantiles(row)
        mean_d = forecaster.predict_mean(row)

        print(f"\n{'-'*60}")
        print(f"  PERIOD {period+1}/{n_periods}  |  {date_str}  |  {promo}  |  price={price:.2f}")
        print(f"{'-'*60}")
        print(f"  On-hand: {state['on_hand']} (total={state['on_hand_total']})")
        print(f"  Pipeline: {state['pipeline']} (arrives next / in 2)")
        print(f"  Cumulative cost: {state['total_cost']:.0f}")
        print(f"\n  Demand forecast: mean={mean_d:.1f}, "
              f"q50={q[0.5]}, q75={q[0.75]}, q90={q[0.9]}, q95={q[0.95]}")

        recommended = compute_recommended_order(inv, forecaster, test_rows, period)
        print(f"  >>> RECOMMENDED ORDER: {recommended} (base-stock)")

        try:
            order_input = input(f"\n  Enter order quantity [{recommended}]: ").strip()
            order = int(order_input) if order_input else recommended
        except (ValueError, EOFError):
            order = recommended

        try:
            demand_input = input("  Enter revealed demand: ").strip()
            demand = int(demand_input)
        except (ValueError, EOFError):
            print("  Invalid demand, using 0")
            demand = 0

        result = inv.step(order, demand)
        forecaster.update_with_demand(test_rows, period, demand)

        print(f"\n  Results:")
        print(f"    Arrived: {result['arrived']} | Ordered: {result['order']} | Demand: {result['demand']}")
        print(f"    Sold: {result['sold']} (age1={result['sold_age1']}, age0={result['sold_age0']})")
        if result['shortage'] > 0:
            print(f"    *** SHORTAGE: {result['shortage']} units (cost: {result['shortage_cost']})")
        if result['expired'] > 0:
            print(f"    *** EXPIRED: {result['expired']} units (cost: {result['expiry_cost']})")
        print(f"    Period cost: {result['period_cost']} "
              f"(hold={result['holding_cost']}, short={result['shortage_cost']}, exp={result['expiry_cost']})")
        print(f"    Running total: {result['total_cost']:.0f}")

    inv.summary()


if __name__ == "__main__":
    main()
