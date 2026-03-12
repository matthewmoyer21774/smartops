"""
Policy V10: Aggressive Adaptive Fractile.

V6 was too gentle with ±0.05/0.10 adjustments. V10 uses hard state-based
switching: when inventory is high, slam the fractile to 0.50 or lower.
When low, push to 0.90. Also adds a hard cap on order quantity.

Usage:
    python play_game_v10.py
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
    Aggressive adaptive: hard state-based fractile switching + order cap.
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

    # --- Aggressive state-based fractile ---
    total_on_hand = inv.on_hand[0] + inv.on_hand[1]
    total_pipeline = inv.pipeline[0] + inv.pipeline[1]
    total_supply = total_on_hand + total_pipeline
    expected_demand_lt = mean_t + mean_t1 + mean_t2

    if expected_demand_lt > 0:
        supply_ratio = total_supply / expected_demand_lt
    else:
        supply_ratio = 3.0

    # Hard switching (not gentle adjustments)
    if supply_ratio > 1.5:
        fractile = 0.40  # heavily over-stocked → barely order
    elif supply_ratio > 1.2:
        fractile = 0.50  # well stocked → order conservatively
    elif supply_ratio < 0.4:
        fractile = 0.90  # emergency → order aggressively
    elif supply_ratio < 0.6:
        fractile = 0.85
    else:
        fractile = BASE_FRACTILE  # normal range → cost-optimal

    # Age_1 pressure: if lots about to expire, hard suppress
    if inv.on_hand[1] > mean_t:
        fractile = min(fractile, 0.50)

    # End-of-game taper
    remaining = n_periods - (period + 2)
    if remaining <= 1:
        fractile = min(fractile, 0.40)
    elif remaining <= 2:
        fractile = min(fractile, 0.50)
    elif remaining <= 3:
        fractile = min(fractile, 0.60)

    fractile = max(0.30, min(0.95, fractile))

    # --- Forward projection (same as V2) ---
    oh0_t = inv.on_hand[0] + inv.pipeline[0]
    oh1_t = inv.on_hand[1]
    sell1_t = min(oh1_t, mean_t)
    rem_t = mean_t - sell1_t
    sell0_t = min(oh0_t, rem_t)
    carry_to_t1 = max(0, oh0_t - sell0_t)

    oh0_t1 = inv.pipeline[1]
    oh1_t1 = carry_to_t1
    sell1_t1 = min(oh1_t1, mean_t1)
    rem_t1 = mean_t1 - sell1_t1
    sell0_t1 = min(oh0_t1, rem_t1)
    existing_at_t2 = max(0, oh0_t1 - sell0_t1)

    target_demand = _interpolate_quantile(q_t2, fractile)
    order = max(0, target_demand - existing_at_t2)

    # Hard cap: never order more than q75 + 1
    cap = q_t2.get(0.75, mean_t2 * 1.2) + 1
    order = min(order, cap)

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
    print("  PERISHABLE INVENTORY GAME - V10: Aggressive Adaptive")
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
        print(f"  Forecast: mean={mean_d:.1f}, q50={q[0.5]}, q75={q[0.75]}")
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
