import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ----------------------------
# Load data
# ----------------------------
df = pd.read_parquet("df_6_art_train_project.parquet")

print(df.head())
print(df.columns)

# ----------------------------
# Select a SKU
# ----------------------------
sku_id = 2921141  # change this

sku_df = df[df["art_id"] == sku_id].copy()

sku_df = sku_df.dropna(subset=["date", "sales"])
sku_df["sales_diff"] = sku_df["sales"].diff().dropna()

# Ensure time order
sku_df = sku_df.sort_values("date")
sku_df["date"] = pd.to_datetime(sku_df["date"])
sku_df = sku_df.set_index("date")
sales = sku_df["sales"]
sales_diff = sku_df["sales_diff"].dropna()

# ----------------------------
# Plot sales over time
# ----------------------------
plt.figure()
plt.plot(sku_df.index, sales)
plt.title(f"Sales Over Time - SKU {sku_id}")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

plt.figure()
plt.plot(sku_df["sales_diff"])
plt.title("First Difference of Sales")
plt.xlabel("Date")
plt.ylabel("Δ Sales")
plt.show()

# ----------------------------
# ACF plot
# ----------------------------
plot_acf(sales_diff, lags=40)
plt.title(f"ACF - SKU {sku_id}")
plt.show()

# ----------------------------
# PACF plot
# ----------------------------
plot_pacf(sales_diff, lags=40, method="ywm")
plt.title(f"PACF - SKU {sku_id}")
plt.show()
