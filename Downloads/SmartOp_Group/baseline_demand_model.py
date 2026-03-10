import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_parquet("df_6_art_train_project.parquet")

# Focus on one SKU
art_id = 2921141
sku_df = df[df["art_id"] == art_id].copy()

# Drop missing values
sku_df = sku_df.dropna(subset=["date", "sales"])
sku_df["date"] = pd.to_datetime(sku_df["date"])
sku_df = sku_df.sort_values("date").set_index("date")

# Fill missing days (assume zero sales)
sku_df = sku_df.asfreq("D").fillna(0)

# ----------------------------
# Create Lag Features (from your PACF/ACF)
# ----------------------------
lags = [1, 2, 3, 4, 5, 6, 7, 14]
for lag in lags:
    sku_df[f"lag_{lag}"] = sku_df["sales"].shift(lag)

sku_df = sku_df.dropna(subset=[f"lag_{lag}" for lag in lags])

# ----------------------------
# Simulation Setup
# ----------------------------
# Costs
holding_cost = 1
shortage_cost = 19
expiry_cost = 9
lead_time = 2
shelf_life = 2

# Starting inventory on 2021-07-02
initial_inventory = [5, 4, 3]  # oldest to newest
inventory = initial_inventory.copy()
pipeline = [0] * lead_time  # pipeline for orders in transit

# Simulation period (example: 26 days)
sim_start = pd.Timestamp("2021-07-02")
sim_end = sim_start + pd.Timedelta(days=25)
sim_dates = pd.date_range(sim_start, sim_end, freq="D")
sim_sales = sku_df.loc[sim_dates, "sales"]

# Safety factor (optional)
safety_factor = 1.0  # no extra buffer for baseline

# ----------------------------
# Run Simulation
# ----------------------------
history = []

for t, date in enumerate(sim_dates):
    # Forecast demand for arrival in 2 days
    last_lags = sku_df.loc[
        date - pd.Timedelta(days=max(lags)), date - pd.Timedelta(days=1), f"lag_{lag}"
    ]
    forecast = sku_df.loc[date, [f"lag_{lag}" for lag in lags]].mean()

    # Decide order (arrives in 2 days)
    order_qty = max(int(safety_factor * forecast) - sum(inventory) - sum(pipeline), 0)
    pipeline.append(order_qty)

    # Sales occurs
    demand = sim_sales.loc[date]
    sold = 0
    remaining_demand = demand

    # FIFO sales
    for i in range(len(inventory)):
        sell_now = min(inventory[i], remaining_demand)
        inventory[i] -= sell_now
        remaining_demand -= sell_now
        sold += sell_now
        if remaining_demand <= 0:
            break

    # Shortage cost
    shortage = remaining_demand
    shortage_cost_today = shortage * shortage_cost

    # Expiry: items older than shelf life expire
    expired = 0
    if len(inventory) > shelf_life:
        expired = sum(inventory[:-shelf_life])
        inventory = inventory[-shelf_life:]  # keep only freshest units

    expiry_cost_today = expired * expiry_cost

    # Holding cost: inventory that remains
    holding_cost_today = sum(inventory) * holding_cost

    # Receive order if lead time passed
    received = pipeline.pop(0)
    inventory.append(received)

    # Record
    history.append(
        {
            "date": date,
            "demand": demand,
            "sold": sold,
            "shortage": shortage,
            "expired": expired,
            "order": order_qty,
            "inventory_end": sum(inventory),
            "holding_cost": holding_cost_today,
            "shortage_cost": shortage_cost_today,
            "expiry_cost": expiry_cost_today,
            "total_cost": holding_cost_today + shortage_cost_today + expiry_cost_today,
        }
    )

sim_df = pd.DataFrame(history).set_index("date")

# ----------------------------
# Summarize Costs
# ----------------------------
print("Total cost over simulation:", sim_df["total_cost"].sum())
sim_df[["demand", "order", "inventory_end"]].plot(title="Demand, Order, Inventory")
plt.show()
