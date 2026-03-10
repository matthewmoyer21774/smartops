"""
Demand Model V2: Cost-Aware Forecaster for SKU 2921141.

Instead of targeting arbitrary quantiles (q50/q75/q90/q95), V2 trains
LightGBM at the cost-optimal critical fractile derived from the actual
inventory cost structure:

    - Shortage (underage):  Cu = 19 per unit
    - Holding (overage):    Co_hold = 1 per unit
    - Expiry (perishable):  Co_exp = 9 per unit

For perishable goods with shelf_life=2, excess inventory has ~50% chance
of expiring. Effective overage cost = 1 + 0.5 * 9 = 5.5.
Optimal critical fractile = Cu / (Cu + Co) = 19 / 24.5 = 0.776.

Key insight: asymmetric inventory cost loss IS quantile regression at
the critical fractile. No custom loss function needed — we just train
at the right quantile.

Usage:
    python demand_model_v2.py          # run comparison vs V1
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

from demand_model import load_and_prepare, DemandForecaster, FEATURES, LGB_PARAMS
from inventory_engine import PerishableInventory

# ---------------------------------------------------------------------------
#  Cost parameters (must match inventory_engine.py)
# ---------------------------------------------------------------------------
HOLDING_COST = 1
SHORTAGE_COST = 19
EXPIRY_COST = 9

# Effective overage cost depends on expiry probability.
# We test multiple assumptions to find the best one.
EXPIRY_PROBS = {
    "optimistic (30%)": 0.30,   # Co = 1 + 0.3*9 = 3.7, CF = 0.837
    "moderate (50%)":   0.50,   # Co = 1 + 0.5*9 = 5.5, CF = 0.776
    "pessimistic (70%)": 0.70,  # Co = 1 + 0.7*9 = 7.3, CF = 0.722
}


def critical_fractile(p_expire):
    """Compute newsvendor critical fractile for perishable goods."""
    co = HOLDING_COST + p_expire * EXPIRY_COST
    return SHORTAGE_COST / (SHORTAGE_COST + co)


# ---------------------------------------------------------------------------
#  V2: Cost-Aware LightGBM Forecaster
# ---------------------------------------------------------------------------
class CostAwareForecaster:
    """
    LightGBM trained at the cost-optimal critical fractile.
    Also trains standard quantiles for probabilistic coverage.
    """

    def __init__(self, p_expire=0.50):
        self.p_expire = p_expire
        self.cf = critical_fractile(p_expire)
        self.co = HOLDING_COST + p_expire * EXPIRY_COST
        # Standard quantiles plus the cost-optimal fractile
        self.quantiles = (0.50, 0.75, self.cf, 0.90, 0.95)
        self.models = {}
        self.mean_model = None
        self.training_tail = []
        self.revealed_demands = []

    def fit(self, df, val_size=200, test_size=200):
        known = df.dropna(subset=["sales"]).copy()
        known = known.dropna(subset=FEATURES)

        n = len(known)
        train_end = n - val_size - test_size
        val_end = n - test_size

        train_data = known.iloc[:train_end]
        val_data = known.iloc[train_end:val_end]
        trainval = known.iloc[:val_end]

        X_train = train_data[FEATURES].values
        y_train = train_data["sales"].values
        X_val = val_data[FEATURES].values
        y_val = val_data["sales"].values
        X_trainval = trainval[FEATURES].values
        y_trainval = trainval["sales"].values

        print(f"  Split: Train={len(train_data)} | Val={len(val_data)} | Test={test_size}")
        print(f"  P(expire)={self.p_expire:.0%}, Co={self.co:.1f}, "
              f"Critical fractile={self.cf:.3f}")

        # --- Quantile models (including cost-optimal fractile) ---
        for q in self.quantiles:
            params_q = {**LGB_PARAMS, "objective": "quantile", "alpha": q}
            ds_train_q = lgb.Dataset(X_train, label=y_train)
            ds_val_q = lgb.Dataset(X_val, label=y_val, reference=ds_train_q)
            model_es = lgb.train(
                params_q, ds_train_q, valid_sets=[ds_val_q],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            )
            best_rounds = model_es.best_iteration
            ds_full_q = lgb.Dataset(X_trainval, label=y_trainval)
            self.models[q] = lgb.train(params_q, ds_full_q, num_boost_round=best_rounds)
            label = f"q{q:.3f} (COST-OPTIMAL)" if q == self.cf else f"q{q:.2f}"
            print(f"    {label}: {best_rounds} rounds")

        # --- Poisson mean model ---
        params_mean = {**LGB_PARAMS, "objective": "poisson"}
        ds_train_m = lgb.Dataset(X_train, label=y_train)
        ds_val_m = lgb.Dataset(X_val, label=y_val, reference=ds_train_m)
        model_es = lgb.train(
            params_mean, ds_train_m, valid_sets=[ds_val_m],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        ds_full_m = lgb.Dataset(X_trainval, label=y_trainval)
        self.mean_model = lgb.train(
            params_mean, ds_full_m, num_boost_round=model_es.best_iteration
        )
        print(f"    mean (Poisson): {model_es.best_iteration} rounds")

        return self

    def _feature_row(self, row):
        return np.array([[row[f] for f in FEATURES]])

    def predict_cost_optimal(self, row):
        """Predict at the cost-optimal critical fractile."""
        X = self._feature_row(row)
        return max(0, round(float(self.models[self.cf].predict(X)[0]), 1))

    def predict_quantiles(self, row):
        X = self._feature_row(row)
        return {q: max(0, round(float(m.predict(X)[0]), 1))
                for q, m in self.models.items()}

    def predict_mean(self, row):
        X = self._feature_row(row)
        return max(0, self.mean_model.predict(X)[0])

    def get_test_features(self, df):
        known = df.dropna(subset=["sales"])
        self.training_tail = known["sales"].values[-7:].tolist()
        self.revealed_demands = []
        test = df[df["sales"].isna()].copy()
        rows = []
        for _, r in test.iterrows():
            rows.append({f: r[f] for f in FEATURES} | {"date": r["date"]})
        return rows


# ---------------------------------------------------------------------------
#  Inventory cost evaluation
# ---------------------------------------------------------------------------
def inventory_cost_per_obs(y_true, y_pred, co):
    """Compute asymmetric inventory cost per observation."""
    residual = y_true - y_pred
    shortage = np.maximum(0, residual)
    excess = np.maximum(0, -residual)
    return SHORTAGE_COST * shortage + co * excess


# ---------------------------------------------------------------------------
#  Comparison
# ---------------------------------------------------------------------------
def compare_v1_v2(df, n_test=200):
    """
    V1 (standard pinball loss at fixed quantiles) vs
    V2 (cost-optimal critical fractile targeting actual inventory costs).
    """
    known = df.dropna(subset=["sales"]).copy()
    known = known.dropna(subset=FEATURES)
    test_data = known.iloc[-n_test:]
    y_test = test_data["sales"].values
    X_test = test_data[FEATURES].values

    test_rows = []
    for _, r in test_data.iterrows():
        test_rows.append({f: r[f] for f in FEATURES})

    print("=" * 65)
    print("  V1 (Standard Quantile Loss) vs V2 (Cost-Optimal Loss)")
    print("=" * 65)

    # --- V1 ---
    print("\n--- V1: LightGBM Standard Quantile Loss ---")
    v1 = DemandForecaster()
    v1.fit(df)

    # --- V2 at multiple expiry assumptions ---
    v2_models = {}
    for label, p_exp in EXPIRY_PROBS.items():
        print(f"\n--- V2: Cost-Optimal, {label} ---")
        v2 = CostAwareForecaster(p_expire=p_exp)
        v2.fit(df)
        v2_models[label] = v2

    n = len(y_test)

    # === Standard Metrics ===
    print(f"\n{'=' * 65}")
    print(f"  PINBALL LOSS (test={n})")
    print(f"{'=' * 65}")

    # V1 pinball
    print(f"\n  V1 (standard quantile targets):")
    for q in (0.50, 0.75, 0.90, 0.95):
        preds = v1.models[q].predict(X_test)
        err = y_test - preds
        pb = np.mean(np.where(err >= 0, q * err, (q - 1) * err))
        cov = np.mean(y_test <= preds)
        print(f"    q{q:.2f}: pinball={pb:.3f}, coverage={cov:.1%} (target {q:.0%})")

    # V2 pinball at their respective critical fractiles
    for label, v2 in v2_models.items():
        cf = v2.cf
        preds_cf = v2.models[cf].predict(X_test)
        err = y_test - preds_cf
        pb = np.mean(np.where(err >= 0, cf * err, (cf - 1) * err))
        cov = np.mean(y_test <= preds_cf)
        print(f"\n  V2 {label}:")
        print(f"    q{cf:.3f} (cost-optimal): pinball={pb:.3f}, coverage={cov:.1%} (target {cf:.0%})")

    # === THE KEY METRIC: Inventory Cost ===
    print(f"\n{'=' * 65}")
    print(f"  INVENTORY COST (the actual objective)")
    print(f"{'=' * 65}")
    print(f"\n  Using effective overage cost for 'moderate (50%)' scenario:")
    co_moderate = HOLDING_COST + 0.5 * EXPIRY_COST

    # V1 at different quantile targets
    print(f"\n  {'Forecast':<35} {'Avg Cost':>10} {'Total':>10}")
    print(f"  {'-' * 55}")

    for q in (0.50, 0.75, 0.90, 0.95):
        preds = v1.models[q].predict(X_test)
        costs = inventory_cost_per_obs(y_test, preds, co_moderate)
        print(f"  V1 q{q:.2f}{'':25} {np.mean(costs):>10.2f} {np.sum(costs):>10.0f}")

    v1_mean_preds = v1.mean_model.predict(X_test)
    costs_mean = inventory_cost_per_obs(y_test, v1_mean_preds, co_moderate)
    print(f"  V1 Poisson mean{'':19} {np.mean(costs_mean):>10.2f} {np.sum(costs_mean):>10.0f}")

    for label, v2 in v2_models.items():
        cf = v2.cf
        preds_cf = v2.models[cf].predict(X_test)
        costs_cf = inventory_cost_per_obs(y_test, preds_cf, co_moderate)
        print(f"  V2 q{cf:.3f} ({label[:12]}...){'':3} {np.mean(costs_cf):>10.2f} {np.sum(costs_cf):>10.0f}")

    # Find best
    print(f"\n  Winner analysis:")
    all_results = {}
    for q in (0.50, 0.75, 0.90, 0.95):
        preds = v1.models[q].predict(X_test)
        total = np.sum(inventory_cost_per_obs(y_test, preds, co_moderate))
        all_results[f"V1 q{q:.2f}"] = total
    all_results["V1 mean"] = np.sum(costs_mean)
    for label, v2 in v2_models.items():
        cf = v2.cf
        preds = v2.models[cf].predict(X_test)
        total = np.sum(inventory_cost_per_obs(y_test, preds, co_moderate))
        all_results[f"V2 CF={cf:.3f}"] = total

    sorted_results = sorted(all_results.items(), key=lambda x: x[1])
    for rank, (name, cost) in enumerate(sorted_results, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"    {rank}. {name:<25} {cost:>10.0f}{marker}")

    # === RMSE / MAE ===
    v2_mod = v2_models["moderate (50%)"]
    cf = v2_mod.cf
    mean_v1 = v1.mean_model.predict(X_test)
    mean_v2 = v2_mod.mean_model.predict(X_test)
    cost_opt_v2 = v2_mod.models[cf].predict(X_test)

    print(f"\n  {'Metric':<10} {'V1 Mean':>12} {'V2 Mean':>12} {'V2 CostOpt':>12}")
    print(f"  {'-' * 46}")
    print(f"  {'RMSE':<10} {np.sqrt(np.mean((y_test-mean_v1)**2)):>12.3f} "
          f"{np.sqrt(np.mean((y_test-mean_v2)**2)):>12.3f} "
          f"{np.sqrt(np.mean((y_test-cost_opt_v2)**2)):>12.3f}")
    print(f"  {'MAE':<10} {np.mean(np.abs(y_test-mean_v1)):>12.3f} "
          f"{np.mean(np.abs(y_test-mean_v2)):>12.3f} "
          f"{np.mean(np.abs(y_test-cost_opt_v2)):>12.3f}")

    # === 26-Period Forecasts ===
    print(f"\n{'=' * 65}")
    print(f"  26-PERIOD TEST FORECASTS")
    print(f"{'=' * 65}")

    v1_test = v1.get_test_features(df)
    v2_test = v2_mod.get_test_features(df)
    test_dates = df[df["sales"].isna()]["date"].values

    print(f"\n  {'Date':<12} {'V1 mean':>8} {'V1 q95':>8} {'V2 cost':>8} {'V2 q95':>8} {'Delta':>8}")
    print(f"  {'-' * 52}")
    for i in range(min(10, len(v1_test))):
        date_str = pd.Timestamp(test_dates[i]).strftime("%Y-%m-%d")
        m_v1 = v1.predict_mean(v1_test[i])
        q95_v1 = v1.predict_quantiles(v1_test[i])[0.95]
        c_v2 = v2_mod.predict_cost_optimal(v2_test[i])
        q95_v2 = v2_mod.predict_quantiles(v2_test[i])[0.95]
        delta = c_v2 - q95_v1
        print(f"  {date_str:<12} {m_v1:>8.1f} {q95_v1:>8.1f} {c_v2:>8.1f} {q95_v2:>8.1f} {delta:>+8.1f}")

    # Summary stats
    v2_cost_preds = [v2_mod.predict_cost_optimal(v2_test[i]) for i in range(len(v2_test))]
    v1_q95_preds = [v1.predict_quantiles(v1_test[i])[0.95] for i in range(len(v1_test))]
    v1_mean_preds_test = [v1.predict_mean(v1_test[i]) for i in range(len(v1_test))]

    print(f"\n  Forecast summary across 26 test periods:")
    print(f"    V1 mean:      avg={np.mean(v1_mean_preds_test):.1f}")
    print(f"    V1 q95:       avg={np.mean(v1_q95_preds):.1f}")
    print(f"    V2 cost-opt:  avg={np.mean(v2_cost_preds):.1f} (q{cf:.3f})")

    print(f"\n  Interpretation:")
    print(f"    V1 orders at q0.95 ({SHORTAGE_COST}/({SHORTAGE_COST}+{HOLDING_COST}) fractile)")
    print(f"    V2 orders at q{cf:.3f} ({SHORTAGE_COST}/({SHORTAGE_COST}+{v2_mod.co:.1f}) fractile)")
    print(f"    V2 accounts for expiry risk in the overage cost,")
    print(f"    resulting in LOWER order quantities that avoid wasteful expiry.")


if __name__ == "__main__":
    df = load_and_prepare()
    compare_v1_v2(df, n_test=200)
