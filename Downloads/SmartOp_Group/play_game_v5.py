"""
Policy V5: Scenario-Tree Optimization with Weighted Quantiles.

Builds a 5×5 scenario tree from quantile forecasts over the order's
2-period lifetime. For each candidate order, simulates all 25 scenario
paths through the ACTUAL FIFO/expiry mechanics of PerishableInventory.

Unlike newsvendor approximations, this fully respects:
- FIFO selling (age_1 first)
- Expiry at end of period for age_1
- Holding cost on surviving age_0
- The 2-period lifecycle of each order

Usage:
    python play_game_v5.py           # interactive play
    python play_game_v5.py backtest  # run backtest
"""
import sys
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory

SHORTAGE_COST = 19
HOLDING_COST = 1
EXPIRY_COST = 9

# Quantile levels for scenario generation
SCENARIO_QUANTILES = [0.0, 0.50, 0.75, 0.90, 0.95]
# Approximate probability weights between quantile boundaries
# P(D <= q0)=~0.10, P(q0<D<=q50)=0.40, P(q50<D<=q75)=0.25, etc.
SCENARIO_WEIGHTS = [0.10, 0.40, 0.25, 0.15, 0.10]


def _build_demand_scenarios(q_dict, mean_d):
    """Build demand scenarios from quantile forecasts with probability weights."""
    scenarios = []
    # Scenario 0: very low demand (below median)
    scenarios.append(max(0, mean_d * 0.2))
    # Scenarios from quantile forecasts
    for q in [0.5, 0.75, 0.9, 0.95]:
        scenarios.append(max(0, q_dict.get(q, mean_d)))
    return scenarios


def _simulate_full_cost(inv_state, pipeline_state, order_qty, d_t, d_t1, d_t2, d_t3):
    """
    Full 4-period forward simulation through actual FIFO/expiry mechanics.

    Simulates from current period through t+3 to capture the complete
    lifecycle of the order placed now.
    """
    oh0 = inv_state[0]
    oh1 = inv_state[1]
    pipe = list(pipeline_state)

    total_cost = 0

    for t, demand in enumerate([d_t, d_t1, d_t2, d_t3]):
        # Arrivals
        if t == 0:
            arrived = pipe[0]
        elif t == 1:
            arrived = pipe[1]
        elif t == 2:
            arrived = order_qty  # our order arrives
        else:
            arrived = 0

        oh0 += arrived

        # FIFO: sell age_1 first
        sell1 = min(oh1, demand)
        rem = demand - sell1
        sell0 = min(oh0, rem)
        rem -= sell0

        shortage = max(0, rem)
        expired = oh1 - sell1
        holding_units = oh0 - sell0

        total_cost += (shortage * SHORTAGE_COST +
                       expired * EXPIRY_COST +
                       holding_units * HOLDING_COST)

        # Age transition
        oh1 = oh0 - sell0  # surviving age_0 becomes age_1
        oh0 = 0

    return total_cost


def compute_recommended_order(inv, forecaster, test_rows, period):
    """
    Scenario-tree optimization: enumerate 25 demand paths over 4 periods,
    simulate full FIFO/expiry for each, pick order minimizing expected cost.
    """
    n_periods = len(test_rows)

    if period + 2 >= n_periods:
        return 0

    def get_forecasts(p):
        if 0 <= p < n_periods:
            return forecaster.predict_mean(test_rows[p]), forecaster.predict_quantiles(test_rows[p])
        return 0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0}

    mean_t, q_t = get_forecasts(period)
    mean_t1, q_t1 = get_forecasts(period + 1)
    mean_t2, q_t2 = get_forecasts(period + 2)
    mean_t3, q_t3 = get_forecasts(period + 3) if period + 3 < n_periods else (0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0})

    # For t and t+1, use point forecast (we don't control those periods)
    d_t = mean_t
    d_t1 = mean_t1

    # Build scenario tree for t+2 and t+3 (the periods our order serves)
    scenarios_t2 = _build_demand_scenarios(q_t2, mean_t2)
    scenarios_t3 = _build_demand_scenarios(q_t3, mean_t3)

    inv_state = list(inv.on_hand)
    pipe_state = list(inv.pipeline)

    best_order = 0
    best_cost = float('inf')

    for order_qty in range(0, 22):
        expected_cost = 0
        total_weight = 0

        for i, d2 in enumerate(scenarios_t2):
            w2 = SCENARIO_WEIGHTS[i]
            for j, d3 in enumerate(scenarios_t3):
                w3 = SCENARIO_WEIGHTS[j]
                weight = w2 * w3

                cost = _simulate_full_cost(
                    inv_state, pipe_state, order_qty,
                    d_t, d_t1, d2, d3
                )
                expected_cost += cost * weight
                total_weight += weight

        avg_cost = expected_cost / total_weight
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
    print("  PERISHABLE INVENTORY GAME - V5: Scenario-Tree Optimization")
    print("  25 demand paths × 22 order candidates = 550 simulations/period")
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
        print(f"  >>> RECOMMENDED ORDER: {recommended} (scenario-tree)")

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
