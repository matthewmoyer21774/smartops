"""
Policy V3: Two-Period Stochastic DP.

For each candidate order q in [0, 20], simulate the full FIFO/expiry
mechanics over the 2 periods the order will serve (t+2 and t+3),
across multiple demand scenarios drawn from quantile forecasts.

Key insight: the order arrives at t+2 as age_0, becomes age_1 at t+3,
then expires. We must jointly optimize over BOTH periods' demand.

Usage:
    python play_game_v3.py           # interactive play
    python play_game_v3.py backtest  # run backtest
"""
import sys
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory

SHORTAGE_COST = 19
HOLDING_COST = 1
EXPIRY_COST = 9


def _simulate_two_period_cost(existing_age1, order_qty, d_t2, d_t3):
    """
    Simulate cost over the 2 periods an order serves.

    At t+2: existing_age1 units (from previous inventory) + order_qty as age_0.
    At t+3: leftover age_0 from t+2 becomes age_1.

    Returns total cost over these 2 periods.
    """
    # --- Period t+2 ---
    oh0 = order_qty    # new order arrives as age_0
    oh1 = existing_age1  # carried forward from t+1

    # FIFO: sell age_1 first
    sell1 = min(oh1, d_t2)
    rem = d_t2 - sell1
    sell0 = min(oh0, rem)
    rem -= sell0

    shortage_t2 = max(0, rem)
    expired_t2 = oh1 - sell1  # unsold age_1 expires
    surviving_t2 = oh0 - sell0  # leftover age_0

    cost_t2 = (shortage_t2 * SHORTAGE_COST +
               expired_t2 * EXPIRY_COST +
               surviving_t2 * HOLDING_COST)

    # --- Period t+3 ---
    # surviving age_0 from t+2 becomes age_1
    # No new order from us in this calculation (that's a future decision)
    oh0_t3 = 0  # we don't control the next order
    oh1_t3 = surviving_t2

    sell1_t3 = min(oh1_t3, d_t3)
    rem_t3 = d_t3 - sell1_t3
    shortage_t3 = max(0, rem_t3)
    expired_t3 = oh1_t3 - sell1_t3

    cost_t3 = (shortage_t3 * SHORTAGE_COST +
               expired_t3 * EXPIRY_COST)

    return cost_t2 + cost_t3


def _project_inventory_to_t2(inv, mean_t, mean_t1):
    """Project existing inventory forward to estimate state at t+2."""
    oh0_t = inv.on_hand[0] + inv.pipeline[0]
    oh1_t = inv.on_hand[1]

    sell1_t = min(oh1_t, mean_t)
    rem_t = mean_t - sell1_t
    sell0_t = min(oh0_t, rem_t)
    oh0_after_t = oh0_t - sell0_t

    carry_to_t1 = max(0, oh0_after_t)

    oh0_t1 = inv.pipeline[1]
    oh1_t1 = carry_to_t1

    sell1_t1 = min(oh1_t1, mean_t1)
    rem_t1 = mean_t1 - sell1_t1
    sell0_t1 = min(oh0_t1, rem_t1)
    oh0_after_t1 = oh0_t1 - sell0_t1

    existing_age1_at_t2 = max(0, oh0_after_t1)
    return existing_age1_at_t2


def compute_recommended_order(inv, forecaster, test_rows, period):
    """
    Two-period stochastic DP: test each order qty against demand scenarios
    over the 2 periods the order will serve, pick the one with lowest
    expected cost.
    """
    n_periods = len(test_rows)

    if period + 2 >= n_periods:
        return 0

    def get_forecasts(p):
        if 0 <= p < n_periods:
            return forecaster.predict_mean(test_rows[p]), forecaster.predict_quantiles(test_rows[p])
        return 0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0}

    mean_t, _ = get_forecasts(period)
    mean_t1, _ = get_forecasts(period + 1)
    mean_t2, q_t2 = get_forecasts(period + 2)
    mean_t3, q_t3 = get_forecasts(period + 3) if period + 3 < n_periods else (0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0})

    # Project existing inventory to t+2
    existing_age1 = _project_inventory_to_t2(inv, mean_t, mean_t1)

    # Build demand scenarios for t+2 and t+3
    # Use quantile forecasts + zero scenario to capture full range
    demands_t2 = [0, max(0, mean_t2 * 0.3)]
    demands_t2 += [q_t2.get(q, mean_t2) for q in [0.5, 0.75, 0.9, 0.95]]
    demands_t2 = sorted(set(d for d in demands_t2))

    demands_t3 = [0, max(0, mean_t3 * 0.3)]
    demands_t3 += [q_t3.get(q, mean_t3) for q in [0.5, 0.75, 0.9, 0.95]]
    demands_t3 = sorted(set(d for d in demands_t3))

    # Weights: probability mass between quantile boundaries
    # Approximate: give more weight to central scenarios
    def scenario_weight(d, d_list, q_dict, mean_d):
        """Heuristic weight: higher near the mean, lower at extremes."""
        if mean_d <= 0:
            return 1.0
        dist = abs(d - mean_d) / max(mean_d, 1)
        return max(0.1, np.exp(-dist))

    w_t2 = [scenario_weight(d, demands_t2, q_t2, mean_t2) for d in demands_t2]
    w_t3 = [scenario_weight(d, demands_t3, q_t3, mean_t3) for d in demands_t3]

    total_w = sum(w2 * w3 for w2 in w_t2 for w3 in w_t3)

    best_order = 0
    best_cost = float('inf')

    for order_qty in range(0, 22):
        expected_cost = 0
        for i, d2 in enumerate(demands_t2):
            for j, d3 in enumerate(demands_t3):
                cost = _simulate_two_period_cost(existing_age1, order_qty, d2, d3)
                expected_cost += cost * w_t2[i] * w_t3[j]

        avg_cost = expected_cost / total_w
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_order = order_qty

    return best_order


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
    print("  PERISHABLE INVENTORY GAME - V3: Two-Period Stochastic DP")
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
        print(f"  >>> RECOMMENDED ORDER: {recommended} (2-period DP)")

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
