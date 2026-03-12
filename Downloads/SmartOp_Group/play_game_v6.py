"""
Policy V6: Adaptive Fractile with State Feedback.

Starts with the cost-optimal critical fractile (q0.776) but dynamically
adjusts based on current inventory state, pipeline, and upcoming demand
signals:

- High inventory → shift toward q0.50 (reduce waste)
- Low inventory → shift toward q0.90 (reduce shortage)
- Large pipeline → reduce order (supply incoming)
- Promo in 2 periods → boost order (demand spike)
- Near end of game → taper down

The key insight: a STATIC fractile ignores the current state. When you
already have plenty of stock, ordering at q0.776 wastes. When you're
empty, q0.776 risks shortage. This policy adapts.

Usage:
    python play_game_v6.py           # interactive play
    python play_game_v6.py backtest  # run backtest
"""
import sys
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory

SHORTAGE_COST = 19
HOLDING_COST = 1
EXPIRY_COST = 9
P_EXPIRE = 0.50
CO_EFFECTIVE = HOLDING_COST + P_EXPIRE * EXPIRY_COST
BASE_FRACTILE = SHORTAGE_COST / (SHORTAGE_COST + CO_EFFECTIVE)  # 0.776


def _interpolate_quantile(q_dict, target):
    """Linearly interpolate between available quantile predictions."""
    sorted_q = sorted(q_dict.items())
    for q, val in sorted_q:
        if abs(q - target) < 0.001:
            return val
    for i in range(len(sorted_q) - 1):
        q_lo, v_lo = sorted_q[i]
        q_hi, v_hi = sorted_q[i + 1]
        if q_lo <= target <= q_hi:
            frac = (target - q_lo) / (q_hi - q_lo)
            return v_lo + frac * (v_hi - v_lo)
    if target < sorted_q[0][0]:
        return sorted_q[0][1]
    return sorted_q[-1][1]


def compute_recommended_order(inv, forecaster, test_rows, period):
    """
    Adaptive fractile policy: adjust service level based on current state.
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

    # --- Adaptive fractile adjustment ---
    fractile = BASE_FRACTILE  # start at 0.776

    # 1. Inventory level feedback
    total_on_hand = inv.on_hand[0] + inv.on_hand[1]
    total_pipeline = inv.pipeline[0] + inv.pipeline[1]
    total_supply = total_on_hand + total_pipeline

    # Compare supply to expected demand over lead time + 1
    expected_demand_lt = mean_t + mean_t1 + mean_t2
    if expected_demand_lt > 0:
        supply_ratio = total_supply / expected_demand_lt
    else:
        supply_ratio = 2.0  # plenty

    if supply_ratio > 1.5:
        # Well stocked — reduce fractile to avoid waste
        fractile -= 0.10
    elif supply_ratio > 1.2:
        fractile -= 0.05
    elif supply_ratio < 0.5:
        # Dangerously low — increase fractile
        fractile += 0.10
    elif supply_ratio < 0.8:
        fractile += 0.05

    # 2. Expiry pressure: if age_1 is large relative to demand, be cautious
    if inv.on_hand[1] > mean_t * 0.8:
        # Lots about to expire — don't add more
        fractile -= 0.05

    # 3. Promo signal at t+2
    if period + 2 < n_periods and test_rows[period + 2].get("PROMO_01", 0):
        # Demand spike coming — boost
        fractile += 0.05

    # 4. End-of-game taper
    remaining = n_periods - (period + 2)
    if remaining <= 1:
        fractile = min(fractile, 0.50)
    elif remaining <= 2:
        fractile = min(fractile, 0.65)
    elif remaining <= 3:
        fractile = min(fractile, 0.70)

    # Clamp
    fractile = max(0.40, min(0.95, fractile))

    # --- Forward inventory projection (same as V1/V2) ---
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
    existing_at_t2 = max(0, oh0_after_t1)

    # --- Order at adaptive fractile ---
    target_demand = _interpolate_quantile(q_t2, fractile)
    need = max(0, target_demand - existing_at_t2)

    # Cap
    can_serve = min(2, n_periods - (period + 2))
    if can_serve >= 2 and period + 3 < n_periods:
        max_useful = target_demand + q_t3.get(0.5, mean_t3)
    else:
        max_useful = target_demand
    need = min(need, max(0, max_useful - existing_at_t2))

    # Floor
    floor = max(0, q_t2.get(0.5, mean_t2) - existing_at_t2)
    need = max(need, floor)

    return max(0, int(round(need)))


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
    print("  PERISHABLE INVENTORY GAME - V6: Adaptive Fractile")
    print(f"  Base fractile = {BASE_FRACTILE:.3f}, adapts to inventory state")
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
        print(f"  >>> RECOMMENDED ORDER: {recommended} (adaptive fractile)")

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
