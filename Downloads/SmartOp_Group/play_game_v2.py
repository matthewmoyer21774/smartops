"""
Policy V2: Cost-Optimal Critical Fractile Newsvendor.

Same forward inventory projection as V1, but targets the newsvendor
critical fractile that accounts for expiry cost:

    Cu = 19 (shortage cost)
    Co = 1 + p_expire * 9 (holding + expected expiry)

With p_expire = 0.50: Co = 5.5, CF = 19/24.5 = 0.776
V1 targets q0.95 which massively over-orders for perishables.

Usage:
    python play_game_v2.py           # interactive play
    python play_game_v2.py backtest  # run backtest with known demands
"""

import sys
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory

# Cost-optimal critical fractile
SHORTAGE_COST = 19
HOLDING_COST = 1
EXPIRY_COST = 9
P_EXPIRE = 0.45  # probability that excess inventory expires (shelf_life=2)
CO_EFFECTIVE = HOLDING_COST + P_EXPIRE * EXPIRY_COST  # 5.5
CRITICAL_FRACTILE = SHORTAGE_COST / (SHORTAGE_COST + CO_EFFECTIVE)  # 0.776


def compute_recommended_order(inv, forecaster, test_rows, period):
    """
    Cost-optimal newsvendor with forward inventory projection.

    Identical to V1's projection logic but targets q0.776 instead of q0.95.
    The critical fractile Cu/(Cu+Co) accounts for the fact that over-ordering
    perishables incurs expiry cost, not just holding cost.
    """
    n_periods = len(test_rows)

    if period + 2 >= n_periods:
        return 0

    # --- Forecasts for relevant periods ---
    def get_forecasts(p):
        if 0 <= p < n_periods:
            return forecaster.predict_mean(test_rows[p]), forecaster.predict_quantiles(
                test_rows[p]
            )
        return 0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0}

    mean_t, q_t = get_forecasts(period)
    mean_t1, q_t1 = get_forecasts(period + 1)
    mean_t2, q_t2 = get_forecasts(period + 2)
    mean_t3, q_t3 = (
        get_forecasts(period + 3)
        if period + 3 < n_periods
        else (0, {0.5: 0, 0.75: 0, 0.9: 0, 0.95: 0})
    )

    # --- Simulate forward to estimate inventory at t+2 ---
    consume_t = mean_t
    consume_t1 = mean_t1

    oh0_t = inv.on_hand[0]
    oh1_t = inv.on_hand[1]

    sell1_t = min(oh1_t, consume_t)
    rem_t = consume_t - sell1_t
    sell0_t = min(oh0_t, rem_t)
    oh0_after_sell_t = oh0_t - sell0_t

    carry_to_t1_as_age1 = max(0, oh0_after_sell_t)

    oh0_t1 = inv.pipeline[0]
    oh1_t1 = carry_to_t1_as_age1

    sell1_t1 = min(oh1_t1, consume_t1)
    rem_t1 = consume_t1 - sell1_t1
    sell0_t1 = min(oh0_t1, rem_t1)
    oh0_after_sell_t1 = oh0_t1 - sell0_t1

    carry_to_t2_as_age1 = max(0, oh0_after_sell_t1)
    existing_at_t2 = carry_to_t2_as_age1

    # --- Compute order quantity at cost-optimal fractile ---
    can_serve = min(2, n_periods - (period + 2))

    # Taper near end of game
    if can_serve <= 0:
        target_quantile = 0.50
    elif can_serve == 1:
        # Only one period to sell — fractile shifts down
        target_quantile = min(CRITICAL_FRACTILE, 0.65)
    elif n_periods - (period + 2) <= 2:
        target_quantile = CRITICAL_FRACTILE * 0.95  # slight taper
    else:
        target_quantile = CRITICAL_FRACTILE  # ~0.776

    # Interpolate between available quantiles to hit the target
    target_demand = _interpolate_quantile(q_t2, target_quantile)

    need = max(0, target_demand - existing_at_t2)

    # Cap: max useful = demand over the 2 periods the order can serve
    if period + 3 < n_periods:
        max_useful = target_demand + q_t3[0.5]
    else:
        max_useful = target_demand
    need = min(need, max(0, max_useful - existing_at_t2))

    # Floor: at least cover median at t+2
    floor = max(0, q_t2[0.5] - existing_at_t2)
    need = max(need, floor)

    return max(0, int(round(need)))


def _interpolate_quantile(q_dict, target):
    """Linearly interpolate between available quantile predictions."""
    sorted_q = sorted(q_dict.items())
    # If target matches an available quantile
    for q, val in sorted_q:
        if abs(q - target) < 0.001:
            return val
    # Interpolate
    for i in range(len(sorted_q) - 1):
        q_lo, v_lo = sorted_q[i]
        q_hi, v_hi = sorted_q[i + 1]
        if q_lo <= target <= q_hi:
            frac = (target - q_lo) / (q_hi - q_lo)
            return v_lo + frac * (v_hi - v_lo)
    # Extrapolate from nearest
    if target < sorted_q[0][0]:
        return sorted_q[0][1]
    return sorted_q[-1][1]


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
    print("  PERISHABLE INVENTORY GAME - V2: Cost-Optimal Fractile")
    print(
        f"  Critical Fractile = {CRITICAL_FRACTILE:.3f} "
        f"(Cu={SHORTAGE_COST}, Co={CO_EFFECTIVE:.1f})"
    )
    print("=" * 60)

    print("\nLoading data and training demand models...")
    df = load_and_prepare()
    forecaster = DemandForecaster()
    forecaster.fit(df)
    test_rows = forecaster.get_test_features(df)
    n_periods = len(test_rows)
    print(f"Ready! {n_periods} test periods loaded.\n")

    inv = PerishableInventory(on_hand=[4, 3], pipeline=[5])

    for period in range(n_periods):
        row = test_rows[period]
        print(row)
        date_str = row["date"].strftime("%Y-%m-%d (%a)")
        promo = "PROMO" if row["PROMO_01"] else "     "
        price = row["PRC_2_norm"]

        state = inv.get_state()
        q = forecaster.predict_quantiles(row)
        mean_d = forecaster.predict_mean(row)

        print(f"\n{'-'*60}")
        print(
            f"  PERIOD {period+1}/{n_periods}  |  {date_str}  |  {promo}  |  price={price:.2f}"
        )
        print(f"{'-'*60}")
        print(f"  On-hand: {state['on_hand']} (total={state['on_hand_total']})")
        print(f"  Pipeline: {state['pipeline']} (arrives next)")
        print(f"  Cumulative cost: {state['total_cost']:.0f}")
        print(
            f"\n  Demand forecast: mean={mean_d:.1f}, "
            f"q50={q[0.5]}, q75={q[0.75]}, q90={q[0.9]}, q95={q[0.95]}"
        )

        recommended = compute_recommended_order(inv, forecaster, test_rows, period)
        print(f"  >>> RECOMMENDED ORDER: {recommended} (CF={CRITICAL_FRACTILE:.3f})")

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
        print(
            f"    Arrived: {result['arrived']} | Ordered: {result['order']} | Demand: {result['demand']}"
        )
        print(
            f"    Sold: {result['sold']} (age1={result['sold_age1']}, age0={result['sold_age0']})"
        )
        if result["shortage"] > 0:
            print(
                f"    *** SHORTAGE: {result['shortage']} units (cost: {result['shortage_cost']})"
            )
        if result["expired"] > 0:
            print(
                f"    *** EXPIRED: {result['expired']} units (cost: {result['expiry_cost']})"
            )
        print(
            f"    Period cost: {result['period_cost']} "
            f"(hold={result['holding_cost']}, short={result['shortage_cost']}, exp={result['expiry_cost']})"
        )
        print(f"    Running total: {result['total_cost']:.0f}")

    inv.summary()


if __name__ == "__main__":
    main()
