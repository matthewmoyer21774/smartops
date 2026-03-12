"""
Policy V9: Exponential Smoothing Reactive.

Ignores the LightGBM model entirely for ordering. Instead, tracks demand
with simple exponential smoothing on revealed demands and orders based
on that. Model-free, purely reactive.

If the LightGBM quantile forecasts are poorly calibrated, this avoids
model error entirely. Uses only realized demands + inventory state.

Usage:
    python play_game_v9.py
"""
import sys
import numpy as np
from demand_model import load_and_prepare, DemandForecaster, FEATURES
from inventory_engine import PerishableInventory

# Exponential smoothing parameter
ALPHA = 0.3  # responsiveness to recent demand
INITIAL_FORECAST = 4.0  # reasonable starting point based on historical mean

# Track state across periods
_ema_forecast = INITIAL_FORECAST
_demand_history = []


def compute_recommended_order(inv, forecaster, test_rows, period):
    """
    Exponential smoothing: forecast = α × last_demand + (1-α) × prev_forecast.
    Order = forecast minus net inventory position.
    """
    global _ema_forecast, _demand_history
    n_periods = len(test_rows)

    if period + 2 >= n_periods:
        return 0

    # Use EMA forecast (updated externally after each demand reveal)
    forecast = _ema_forecast

    # Inventory position
    IP = inv.on_hand[0] + inv.on_hand[1] + inv.pipeline[0] + inv.pipeline[1]

    # Expected demand consumed before order arrives (2 periods)
    demand_before = forecast * 2

    # Net position at arrival
    net = IP - demand_before

    # Order to cover forecast at arrival, with slight buffer
    # Use 1.1× forecast since we're accepting some shortage risk
    order = forecast * 1.1 - net

    # Hard cap: never order more than 2× forecast
    order = min(order, forecast * 2)

    # Taper near end
    remaining = n_periods - (period + 2)
    if remaining <= 1:
        order = min(order, forecast * 0.5)
    elif remaining <= 2:
        order = min(order, forecast * 0.7)

    return max(0, int(round(order)))


def update_ema(demand):
    """Update exponential moving average with new demand observation."""
    global _ema_forecast, _demand_history
    _demand_history.append(demand)
    _ema_forecast = ALPHA * demand + (1 - ALPHA) * _ema_forecast


def run_backtest(forecaster, test_rows, demands):
    global _ema_forecast, _demand_history
    _ema_forecast = INITIAL_FORECAST
    _demand_history = []

    original_rows = [dict(r) for r in test_rows]
    original_tail = list(forecaster.training_tail) if forecaster.training_tail else []
    forecaster.revealed_demands = []
    sim_rows = [dict(r) for r in original_rows]
    inv = PerishableInventory()
    for period in range(len(demands)):
        order = compute_recommended_order(inv, forecaster, sim_rows, period)
        inv.step(order, demands[period])
        update_ema(demands[period])
        forecaster.update_with_demand(sim_rows, period, demands[period])
    inv.summary()
    forecaster.training_tail = list(original_tail)
    forecaster.revealed_demands = []
    _ema_forecast = INITIAL_FORECAST
    _demand_history = []
    return inv.total_cost


def main():
    global _ema_forecast, _demand_history
    _ema_forecast = INITIAL_FORECAST
    _demand_history = []

    print("=" * 60)
    print("  PERISHABLE INVENTORY GAME - V9: Exponential Smoothing")
    print(f"  alpha={ALPHA}, initial forecast={INITIAL_FORECAST}")
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
        print(f"\n{'-'*60}")
        print(f"  PERIOD {period+1}/{n_periods}  |  {date_str}")
        print(f"  On-hand: {state['on_hand']} Pipeline: {state['pipeline']} Cost: {state['total_cost']:.0f}")
        print(f"  EMA forecast: {_ema_forecast:.1f}")
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
        update_ema(demand)
        forecaster.update_with_demand(test_rows, period, demand)
        print(f"  Sold: {result['sold']} Short: {result['shortage']} Expired: {result['expired']} Cost: {result['period_cost']} Total: {result['total_cost']:.0f}")
    inv.summary()


if __name__ == "__main__":
    main()
