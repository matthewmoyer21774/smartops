"""
Interactive inventory game CLI for live play.

Usage: python play_game.py
"""
import sys
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory


def compute_recommended_order(inv, forecaster, test_rows, period):
    """
    Perishable newsvendor policy with careful inventory projection.

    Key idea: order placed at t arrives at t+2.
    Those units serve demand at t+2 (as age_0) and t+3 (as age_1, then expire).
    We project forward through the FIFO/expiry mechanics to estimate what
    inventory will be alive when the order arrives, then target the critical
    fractile (0.95) for coverage.
    """
    n_periods = len(test_rows)

    # If order arrives after game ends, order 0
    if period + 2 >= n_periods:
        return 0

    # --- Forecasts for relevant periods ---
    def get_forecasts(p):
        if 0 <= p < n_periods:
            return forecaster.predict_mean(test_rows[p]), forecaster.predict_quantiles(test_rows[p])
        return 0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0}

    mean_t, q_t = get_forecasts(period)
    mean_t1, q_t1 = get_forecasts(period + 1)
    mean_t2, q_t2 = get_forecasts(period + 2)
    mean_t3, q_t3 = get_forecasts(period + 3) if period + 3 < n_periods else (0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0})

    # --- Simulate forward to estimate inventory at t+2 (without this order) ---
    # Use mean demand for consumption estimate (balanced between over/under-ordering)
    consume_t = mean_t
    consume_t1 = mean_t1

    # Period t: pipeline[0] arrives, demand occurs, age_1 expires, age transition
    oh0_t = inv.on_hand[0] + inv.pipeline[0]  # age_0 after arrival
    oh1_t = inv.on_hand[1]                      # age_1

    # Sell FIFO: age_1 first, then age_0
    sell1_t = min(oh1_t, consume_t)
    rem_t = consume_t - sell1_t
    sell0_t = min(oh0_t, rem_t)
    oh0_after_sell_t = oh0_t - sell0_t

    # End of t: age_1 remaining expires. age_0 becomes age_1.
    carry_to_t1_as_age1 = max(0, oh0_after_sell_t)

    # Period t+1: pipeline[1] arrives as age_0
    oh0_t1 = inv.pipeline[1]
    oh1_t1 = carry_to_t1_as_age1

    # Sell FIFO at t+1
    sell1_t1 = min(oh1_t1, consume_t1)
    rem_t1 = consume_t1 - sell1_t1
    sell0_t1 = min(oh0_t1, rem_t1)
    oh0_after_sell_t1 = oh0_t1 - sell0_t1

    # End of t+1: age_1 expires, age transition
    carry_to_t2_as_age1 = max(0, oh0_after_sell_t1)

    # Period t+2: the NEW ORDER arrives as age_0.
    # Existing inventory at t+2 = carry_to_t2_as_age1 (as age_1)
    existing_at_t2 = carry_to_t2_as_age1

    # --- Compute order quantity ---
    # The order arrives as age_0 at t+2. existing_at_t2 is age_1 (sold first via FIFO).
    # So effective coverage = existing_at_t2 + order.
    # Unsold order units at end of t+2 become age_1 at t+3 (second chance to sell).

    periods_remaining = n_periods - (period + 2)  # how many periods from arrival to end

    # Adjust target quantile based on remaining periods (taper at end-game)
    # With shelf_life=2, order at t serves t+2 and t+3. If t+3 is beyond the game,
    # any unsold units at t+2 expire at end of t+2 (or t+3 if it exists but is last).
    if periods_remaining <= 1:
        target_quantile = 0.50  # very conservative at the end
    elif periods_remaining <= 2:
        target_quantile = 0.75
    elif periods_remaining <= 3:
        target_quantile = 0.90
    else:
        target_quantile = 0.95

    target_t2 = q_t2[target_quantile]
    need = max(0, target_t2 - existing_at_t2)

    # Cap: max useful = demand over the periods the order can serve (t+2 and t+3)
    if period + 3 < n_periods:
        max_useful = q_t2[target_quantile] + q_t3[0.50]
    else:
        max_useful = q_t2[target_quantile]  # last period, no t+3
    need = min(need, max(0, max_useful - existing_at_t2))

    # Floor: at least cover the median at t+2
    floor = max(0, q_t2[0.5] - existing_at_t2)
    need = max(need, floor)

    return max(0, int(round(need)))


def run_backtest(forecaster, test_rows, demands):
    """Run a backtest with known demands to evaluate the policy."""
    inv = PerishableInventory()
    for period in range(len(demands)):
        order = compute_recommended_order(inv, forecaster, test_rows, period)
        inv.step(order, demands[period])
    inv.summary()
    return inv.total_cost


def main():
    print("=" * 60)
    print("  PERISHABLE INVENTORY GAME - SKU 2921141")
    print("=" * 60)

    # Load and train
    print("\nLoading data and training demand models...")
    df = load_and_prepare()
    forecaster = DemandForecaster()
    forecaster.fit(df)
    test_rows = forecaster.get_test_features(df)
    n_periods = len(test_rows)
    print(f"Ready! {n_periods} test periods loaded.\n")

    # Initialize inventory
    inv = PerishableInventory(on_hand=[4, 3], pipeline=[5, 0])

    for period in range(n_periods):
        row = test_rows[period]
        date_str = row["date"].strftime("%Y-%m-%d (%a)")
        promo = "PROMO" if row["PROMO_01"] else "     "
        price = row["PRC_2_norm"]

        # Current state (before this period's step)
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

        # Recommended order
        recommended = compute_recommended_order(inv, forecaster, test_rows, period)
        print(f"  >>> RECOMMENDED ORDER: {recommended}")

        # Get user input
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

        # Execute period
        result = inv.step(order, demand)

        # Display results
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

    # Final summary
    inv.summary()


if __name__ == "__main__":
    main()
