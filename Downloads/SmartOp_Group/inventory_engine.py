"""
Perishable inventory simulation engine with FIFO selling and expiry.

State: on_hand = [age_0, age_1] where age_1 expires at end of period.
       pipeline = [arriving_next_period, arriving_in_2_periods]

Sequence per period (matching assignment):
  1. Start of period: pipeline[0] arrives (becomes age_0 on-hand).
     Observe state, decide order -> enters pipeline[1].
  2. During period: demand occurs, FIFO (sell age_1 first, then age_0).
  3. End of period: age_1 remaining expires (cost 9). Holding cost on all remaining.
     Age transition: age_0 -> age_1, age_0 reset to 0.
  4. Move to next period.
"""


class PerishableInventory:
    def __init__(self, on_hand=None, pipeline=None):
        # Starting position: [5, 4, 3]
        # 5 arriving next day, 4 = age_0 (delivered today), 3 = age_1 (delivered yesterday)
        self.on_hand = list(on_hand or [4, 3])  # [age_0, age_1]
        self.pipeline = list(pipeline or [5, 0])  # [arrives next period, arrives in 2]
        self.total_cost = 0.0
        self.period = 0
        self.history = []

        # Cost parameters
        self.holding_cost = 1
        self.shortage_cost = 19
        self.expiry_cost = 9

    def get_state(self):
        return {
            "period": self.period,
            "on_hand": list(self.on_hand),
            "on_hand_total": sum(self.on_hand),
            "pipeline": list(self.pipeline),
            "total_cost": self.total_cost,
        }

    def step(self, order_qty, actual_demand):
        """
        Execute one period. Returns dict with period details.

        Args:
            order_qty: units to order (arrives in 2 periods)
            actual_demand: revealed demand for this period
        """
        order_qty = max(0, int(round(order_qty)))
        actual_demand = max(0, int(round(actual_demand)))

        # --- 1. Start of period: receive shipment ---
        arrived = self.pipeline[0]
        # self.on_hand[0] += arrived  # arrived units are fresh (age_0)

        # Update pipeline: shift and add new order
        self.pipeline[0] = order_qty
        # self.pipeline[1] = order_qty

        # --- 2. During period: sell FIFO (oldest = age_1 first) ---
        remaining_demand = actual_demand
        sold_age1 = min(self.on_hand[1], remaining_demand)
        self.on_hand[1] -= sold_age1
        remaining_demand -= sold_age1

        sold_age0 = min(self.on_hand[0], remaining_demand)
        self.on_hand[0] -= sold_age0
        remaining_demand -= sold_age0

        shortage = remaining_demand
        total_sold = sold_age0 + sold_age1

        # --- 3. End of period: expiry and holding ---
        expired = self.on_hand[1]  # age_1 units that weren't sold expire

        # Assuming its the next day, update stock
        new_age1 = self.on_hand[0]
        self.on_hand[1] = new_age1
        self.on_hand[0] = arrived

        # Holding cost only on units that have NOT expired, i.e. the ones that
        # have been held over night
        holding_units = self.on_hand[1]  # only non-expired units

        # --- 4. Compute costs ---
        h_cost = holding_units * self.holding_cost
        s_cost = shortage * self.shortage_cost
        e_cost = expired * self.expiry_cost
        period_cost = h_cost + s_cost + e_cost
        self.total_cost += period_cost

        result = {
            "period": self.period,
            "arrived": arrived,
            "order": order_qty,
            "demand": actual_demand,
            "sold": total_sold,
            "sold_age0": sold_age0,
            "sold_age1": sold_age1,
            "shortage": shortage,
            "expired": expired,
            "holding_units": holding_units,
            "holding_cost": h_cost,
            "shortage_cost": s_cost,
            "expiry_cost": e_cost,
            "period_cost": period_cost,
            "total_cost": self.total_cost,
            "on_hand_after": list(self.on_hand),
            "pipeline_after": list(self.pipeline),
        }
        self.history.append(result)
        self.period += 1
        return result

    def reset(self, on_hand=None, pipeline=None):
        self.on_hand = list(on_hand or [4, 3])
        self.pipeline = list(pipeline or [5, 0])
        self.total_cost = 0.0
        self.period = 0
        self.history = []

    def summary(self):
        """Print cost summary."""
        total_h = sum(r["holding_cost"] for r in self.history)
        total_s = sum(r["shortage_cost"] for r in self.history)
        total_e = sum(r["expiry_cost"] for r in self.history)
        print(f"\n{'='*50}")
        print(f"GAME OVER - {len(self.history)} periods played")
        print(f"  Holding cost:  {total_h:>6.0f}")
        print(f"  Shortage cost: {total_s:>6.0f}")
        print(f"  Expiry cost:   {total_e:>6.0f}")
        print(f"  TOTAL COST:    {self.total_cost:>6.0f}")
        print(f"{'='*50}")


if __name__ == "__main__":
    # Quick test with made-up demands
    inv = PerishableInventory()
    print("Initial state:", inv.get_state())
    demands = [5, 4, 6, 3, 8, 5]
    for i, d in enumerate(demands):
        order = 5  # constant order for testing
        result = inv.step(order, d)
        print(
            f"Period {i}: demand={d}, order={order}, sold={result['sold']}, "
            f"shortage={result['shortage']}, expired={result['expired']}, "
            f"cost={result['period_cost']}, on_hand={result['on_hand_after']}, "
            f"pipeline={result['pipeline_after']}"
        )
    inv.summary()
