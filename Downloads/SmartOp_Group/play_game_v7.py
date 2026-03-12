"""
Policy V7: Lean Newsvendor — No Forward Projection.

V2 simulates inventory forward through t and t+1 to estimate what exists
at t+2. That projection uses mean forecasts which can be wrong, potentially
causing over-ordering.

V7 simplifies: just subtract the total inventory position from the
cost-optimal quantile forecast. No FIFO simulation, no projection errors.

    order = max(0, q_CF(t+2) − IP)
    IP = on_hand[0] + on_hand[1] + pipeline[0] + pipeline[1]

Usage:
    python play_game_v7.py
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
CRITICAL_FRACTILE = SHORTAGE_COST / (SHORTAGE_COST + CO_EFFECTIVE)  # 0.776


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
    Lean newsvendor: q_CF(t+2) minus total inventory position.
    No forward projection, no FIFO simulation.
    """
    n_periods = len(test_rows)

    if period + 2 >= n_periods:
        return 0

    _, q_t2 = forecaster.predict_mean(test_rows[period + 2]), forecaster.predict_quantiles(test_rows[period + 2])

    # Taper near end
    remaining = n_periods - (period + 2)
    if remaining <= 1:
        target_q = 0.50
    elif remaining <= 2:
        target_q = 0.65
    else:
        target_q = CRITICAL_FRACTILE

    target_demand = _interpolate_quantile(q_t2, target_q)

    # Simple inventory position: everything we have + everything coming
    IP = inv.on_hand[0] + inv.on_hand[1] + inv.pipeline[0] + inv.pipeline[1]

    order = max(0, target_demand - IP)
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
    print("  PERISHABLE INVENTORY GAME - V7: Lean Newsvendor")
    print(f"  CF={CRITICAL_FRACTILE:.3f}, no forward projection")
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
        q = forecaster.predict_quantiles(row)
        mean_d = forecaster.predict_mean(row)
        print(f"\n{'-'*60}")
        print(f"  PERIOD {period+1}/{n_periods}  |  {date_str}  |  {promo}")
        print(f"  On-hand: {state['on_hand']} Pipeline: {state['pipeline']} Cost: {state['total_cost']:.0f}")
        print(f"  Forecast: mean={mean_d:.1f}, q50={q[0.5]}, q75={q[0.75]}, q90={q[0.9]}, q95={q[0.95]}")
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
