"""
Demand forecasting for SKU 2921141 using LightGBM quantile regression.

Feature-rich model with lag/rolling sales features, cross-SKU signals,
interaction terms, and proper train/validation/test temporal splits.
Techniques inspired by Van Hevel (2025) master thesis on probabilistic
forecasting: CRPS evaluation, PACF-guided lag selection, z-score
normalization, and residual diagnostics.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


def load_and_prepare(parquet_path="df_6_art_train_project.parquet", art_id=2921141):
    """Load data, filter to target SKU, engineer rich feature set."""
    df_all = pd.read_parquet(parquet_path)

    # --- Cross-SKU feature (subgroup 109, excluding target) ---
    sub109 = df_all[(df_all["SUBGROUP"] == 109) & (df_all["art_id"] != art_id)].copy()
    if len(sub109) > 0:
        sub109_daily = sub109.groupby("date")["sales"].mean()
        sub109_roll3 = sub109_daily.rolling(3, min_periods=1).mean().rename("subgroup_sales_roll3")
    else:
        sub109_roll3 = pd.Series(dtype=float, name="subgroup_sales_roll3")

    # Filter to target SKU
    df = df_all[df_all["art_id"] == art_id].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Merge cross-SKU feature
    if len(sub109_roll3) > 0:
        df = df.merge(sub109_roll3, on="date", how="left")
        df["subgroup_sales_roll3"] = df["subgroup_sales_roll3"].fillna(0)
    else:
        df["subgroup_sales_roll3"] = 0

    # --- Calendar features ---
    df["dow"] = df["date"].dt.dayofweek
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_monday"] = (df["dow"] == 0).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    # --- Lag features (shift by 1 to avoid leakage) ---
    df["sales_lag1"] = df["sales"].shift(1)
    df["sales_lag2"] = df["sales"].shift(2)
    df["sales_lag3"] = df["sales"].shift(3)
    df["sales_lag7"] = df["sales"].shift(7)

    # --- Rolling features ---
    df["sales_roll3"] = df["sales"].shift(1).rolling(3, min_periods=1).mean()
    df["sales_roll7"] = df["sales"].shift(1).rolling(7, min_periods=1).mean()
    df["sales_roll3_std"] = df["sales"].shift(1).rolling(3, min_periods=1).std().fillna(0)
    df["sales_roll7_median"] = df["sales"].shift(1).rolling(7, min_periods=1).median()

    # --- Z-score normalization (Van Hevel thesis technique) ---
    roll7_std = df["sales"].shift(1).rolling(7, min_periods=2).std().fillna(1).replace(0, 1)
    df["sales_zscore"] = (df["sales_lag1"] - df["sales_roll7"]) / roll7_std

    # --- Interaction features ---
    df["sat_x_promo"] = df["is_saturday"] * df["PROMO_01"]
    df["fri_x_promo"] = df["is_friday"] * df["PROMO_01"]
    df["promo_x_depth"] = df["PROMO_01"] * df["PROMO_DEPTH"]
    df["price_x_promo"] = df["PRC_2_norm"] * df["PROMO_01"]

    # --- Price dynamics ---
    df["price_lag1"] = df["PRC_2_norm"].shift(1)
    df["price_change"] = df["PRC_2_norm"] - df["price_lag1"]
    df["price_roll7_mean"] = df["PRC_2_norm"].shift(1).rolling(7, min_periods=1).mean()

    # --- Holiday enhancement ---
    df["near_holiday"] = df[["OFFICIAL_HOLIDAY_01_f1", "OFFICIAL_HOLIDAY_01_l1"]].max(axis=1)

    # --- Derived ---
    promo_shifted = df["PROMO_01"].shift(1)
    promo_groups = (promo_shifted == 1).cumsum()
    no_promo_mask = promo_shifted != 1
    df["days_since_last_promo"] = no_promo_mask.groupby(promo_groups).cumsum().fillna(0)

    # Fill NaN lag features for early rows
    for col in ["sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7",
                "sales_roll3", "sales_roll7", "sales_roll3_std", "sales_roll7_median",
                "sales_zscore", "price_lag1", "price_change", "price_roll7_mean"]:
        df[col] = df[col].fillna(0)

    return df


FEATURES = [
    # Calendar
    "dow", "is_saturday", "is_friday", "is_monday",
    "week_of_year", "month",
    # Lag
    "sales_lag1", "sales_lag2", "sales_lag3", "sales_lag7",
    # Rolling
    "sales_roll3", "sales_roll7", "sales_roll3_std", "sales_roll7_median",
    "sales_zscore",
    # Promo / price
    "PROMO_01", "PROMO_DEPTH", "PRC_2_norm",
    "sat_x_promo", "fri_x_promo", "promo_x_depth", "price_x_promo",
    "price_lag1", "price_change", "price_roll7_mean",
    # Holiday
    "OFFICIAL_HOLIDAY_01_f1", "OFFICIAL_HOLIDAY_01_l1", "near_holiday",
    # Cross-SKU
    "subgroup_sales_roll3",
    # Derived
    "days_since_last_promo",
]

LGB_PARAMS = {
    "verbose": -1,
    "n_jobs": 1,
    "num_leaves": 31,
    "min_child_samples": 15,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
}


class DemandForecaster:
    def __init__(self, quantiles=(0.50, 0.75, 0.90, 0.95)):
        self.quantiles = quantiles
        self.models = {}
        self.mean_model = None
        self.training_tail = []
        self.revealed_demands = []

    def fit(self, df, val_size=200, test_size=200):
        """
        Train quantile + mean models with proper 3-way temporal split.

        Split: Train (all but val+test) | Val (early stopping) | Test (held out)
        After finding best rounds via early stopping on Val, retrain on Train+Val.
        """
        known = df.dropna(subset=["sales"]).copy()
        known = known.dropna(subset=FEATURES)

        n = len(known)
        train_end = n - val_size - test_size
        val_end = n - test_size

        train_data = known.iloc[:train_end]
        val_data = known.iloc[train_end:val_end]

        X_train = train_data[FEATURES].values
        y_train = train_data["sales"].values
        X_val = val_data[FEATURES].values
        y_val = val_data["sales"].values

        # Train+Val combined for final retraining
        trainval = known.iloc[:val_end]
        X_trainval = trainval[FEATURES].values
        y_trainval = trainval["sales"].values

        print(f"  Split: Train={len(train_data)} | Val={len(val_data)} | Test={test_size} held out")

        # --- Mean model (Poisson) ---
        ds_train = lgb.Dataset(X_train, label=y_train)
        ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
        params_mean = {**LGB_PARAMS, "objective": "poisson"}
        model_es = lgb.train(
            params_mean, ds_train, valid_sets=[ds_val],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        best_rounds_mean = model_es.best_iteration

        ds_full = lgb.Dataset(X_trainval, label=y_trainval)
        self.mean_model = lgb.train(params_mean, ds_full, num_boost_round=best_rounds_mean)
        print(f"  Mean model: {best_rounds_mean} rounds (early stopping)")

        # --- Quantile models ---
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
            print(f"  q{q:.2f}: {best_rounds} rounds")

        return self

    def _feature_row(self, row):
        return np.array([[row[f] for f in FEATURES]])

    def predict_quantiles(self, row):
        X = self._feature_row(row)
        result = {}
        for q, model in self.models.items():
            val = model.predict(X)[0]
            result[q] = max(0, round(val, 1))
        return result

    def predict_mean(self, row):
        X = self._feature_row(row)
        return max(0, self.mean_model.predict(X)[0])

    def get_test_features(self, df):
        """Return list of feature dicts for test periods, store training tail."""
        known = df.dropna(subset=["sales"])
        self.training_tail = known["sales"].values[-7:].tolist()
        self.revealed_demands = []

        test = df[df["sales"].isna()].copy()
        rows = []
        for _, r in test.iterrows():
            rows.append({f: r[f] for f in FEATURES} | {"date": r["date"]})
        return rows

    def update_with_demand(self, test_rows, period, revealed_demand):
        """
        After demand is revealed, update lag/rolling features for future periods.
        This makes the model adaptive during live play.
        """
        self.revealed_demands.append(revealed_demand)
        all_demands = list(self.training_tail) + self.revealed_demands

        for future_p in range(period + 1, len(test_rows)):
            offset = future_p - (period + 1)
            idx = len(all_demands) - 1  # index of latest revealed demand

            # Update lag features
            for lag, feat in [(1, "sales_lag1"), (2, "sales_lag2"),
                              (3, "sales_lag3"), (7, "sales_lag7")]:
                look_back = idx - offset - (lag - 1)
                if 0 <= look_back < len(all_demands):
                    test_rows[future_p][feat] = all_demands[look_back]

            # Update rolling features
            # sales_roll3: mean of last 3 demands
            start_3 = idx - offset - 2
            end_3 = idx - offset + 1
            if start_3 >= 0:
                vals = all_demands[max(0, start_3):end_3]
                if vals:
                    test_rows[future_p]["sales_roll3"] = np.mean(vals)

            # sales_roll7: mean of last 7 demands
            start_7 = idx - offset - 6
            end_7 = idx - offset + 1
            if start_7 >= 0:
                vals = all_demands[max(0, start_7):end_7]
                if vals:
                    test_rows[future_p]["sales_roll7"] = np.mean(vals)

            # sales_roll3_std
            if start_3 >= 0:
                vals = all_demands[max(0, start_3):end_3]
                if len(vals) >= 2:
                    test_rows[future_p]["sales_roll3_std"] = float(np.std(vals, ddof=1))

            # sales_roll7_median
            if start_7 >= 0:
                vals = all_demands[max(0, start_7):end_7]
                if vals:
                    test_rows[future_p]["sales_roll7_median"] = float(np.median(vals))

            # sales_zscore
            roll7_val = test_rows[future_p].get("sales_roll7", 0)
            lag1_val = test_rows[future_p].get("sales_lag1", 0)
            if start_7 >= 0:
                vals = all_demands[max(0, start_7):end_7]
                if len(vals) >= 2:
                    std_val = float(np.std(vals, ddof=1))
                    if std_val > 0:
                        test_rows[future_p]["sales_zscore"] = (lag1_val - roll7_val) / std_val
                    else:
                        test_rows[future_p]["sales_zscore"] = 0

    def feature_importance(self, top_n=15):
        """Print feature importance from the mean model."""
        if self.mean_model is None:
            return
        imp = self.mean_model.feature_importance(importance_type="gain")
        pairs = sorted(zip(FEATURES, imp), key=lambda x: -x[1])
        print(f"\nTop {top_n} features by gain:")
        for name, gain in pairs[:top_n]:
            print(f"  {name:30s} {gain:10.1f}")

    def validate(self, df, n_test=200):
        """
        Validate on held-out test set (last n_test rows of known sales).
        Uses the production models trained in fit().
        Reports pinball loss, coverage, CRPS, residual diagnostics, PACF.
        """
        known = df.dropna(subset=["sales"]).copy()
        known = known.dropna(subset=FEATURES)
        test = known.iloc[-n_test:]
        X_test = test[FEATURES].values
        y_test = test["sales"].values

        print(f"\nValidation on held-out test set ({n_test} rows):")

        # Collect quantile predictions for CRPS
        all_preds = {}
        for q in self.quantiles:
            preds = self.models[q].predict(X_test)
            errors = y_test - preds
            pinball = np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))
            coverage = np.mean(y_test <= preds)
            all_preds[q] = preds
            print(f"  q={q:.2f}: pinball={pinball:.3f}, coverage={coverage:.1%} (target {q:.0%})")

        # CRPS (quantile-based approximation)
        crps = self._compute_crps(y_test, all_preds)
        print(f"\n  CRPS: {crps:.3f}")

        # Residual diagnostics (from mean model)
        mean_preds = self.mean_model.predict(X_test)
        residuals = y_test - mean_preds
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        print(f"\n  Residual diagnostics (mean model):")
        print(f"    RMSE:     {rmse:.3f}")
        print(f"    MAE:      {mae:.3f}")
        print(f"    Mean:     {np.mean(residuals):.3f}")
        print(f"    Std:      {np.std(residuals):.3f}")
        print(f"    Skewness: {_skewness(residuals):.3f}")
        print(f"    Kurtosis: {_kurtosis(residuals):.3f}")

        # PACF analysis
        _print_pacf(y_test)

    def validate_old_vs_new(self, df, n_test=200):
        """Side-by-side comparison of old 8-feature model vs new 30-feature model."""
        known = df.dropna(subset=["sales"]).copy()
        known = known.dropna(subset=FEATURES)
        test = known.iloc[-n_test:]
        train = known.iloc[:-n_test]

        X_test = test[FEATURES].values
        y_test = test["sales"].values

        OLD_FEATURES = ["dow", "is_saturday", "is_friday", "PROMO_01",
                        "PROMO_DEPTH", "PRC_2_norm",
                        "OFFICIAL_HOLIDAY_01_f1", "OFFICIAL_HOLIDAY_01_l1"]

        X_test_old = test[OLD_FEATURES].values
        X_train_old = train[OLD_FEATURES].values
        y_train_old = train["sales"].values

        print(f"\n{'='*60}")
        print(f"  OLD vs NEW Model Comparison (test={n_test})")
        print(f"{'='*60}")
        print(f"{'Quantile':<10} {'Old Pinball':>12} {'New Pinball':>12} {'Improvement':>12}")
        print(f"{'-'*46}")

        for q in self.quantiles:
            # Old model
            ds_old = lgb.Dataset(X_train_old, label=y_train_old)
            model_old = lgb.train(
                {"objective": "quantile", "alpha": q, "verbose": -1, "n_jobs": 1,
                 "num_leaves": 16, "min_child_samples": 20, "learning_rate": 0.05},
                ds_old, num_boost_round=300,
            )
            preds_old = model_old.predict(X_test_old)
            errors_old = y_test - preds_old
            pinball_old = np.mean(np.where(errors_old >= 0, q * errors_old, (q - 1) * errors_old))

            # New model (production)
            preds_new = self.models[q].predict(X_test)
            errors_new = y_test - preds_new
            pinball_new = np.mean(np.where(errors_new >= 0, q * errors_new, (q - 1) * errors_new))

            improvement = (pinball_old - pinball_new) / pinball_old * 100
            print(f"  q{q:.2f}     {pinball_old:>10.3f}   {pinball_new:>10.3f}   {improvement:>+10.1f}%")

    def _compute_crps(self, y_true, quantile_preds):
        """CRPS approximation from quantile predictions."""
        quantiles = sorted(quantile_preds.keys())
        n = len(y_true)
        crps = 0
        for i in range(len(quantiles) - 1):
            q_lo = quantiles[i]
            q_hi = quantiles[i + 1]
            dq = q_hi - q_lo
            for j in range(n):
                y = y_true[j]
                f_lo = quantile_preds[q_lo][j]
                f_hi = quantile_preds[q_hi][j]
                midpoint = (f_lo + f_hi) / 2
                indicator = 1.0 if y <= midpoint else 0.0
                crps += dq * (indicator - (q_lo + q_hi) / 2) ** 2
        return crps / n


def _skewness(x):
    n = len(x)
    if n < 3:
        return 0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0
    return (n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3)


def _kurtosis(x):
    n = len(x)
    if n < 4:
        return 0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0
    k = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(((x - m) / s) ** 4)
    return k - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))


def _print_pacf(y, nlags=10):
    """Print PACF values using statsmodels if available."""
    try:
        from statsmodels.tsa.stattools import pacf
        pacf_vals = pacf(y, nlags=nlags, method="ols")
        print(f"\n  PACF analysis (partial autocorrelation):")
        threshold = 1.96 / np.sqrt(len(y))
        for i, val in enumerate(pacf_vals[1:], 1):
            sig = " *" if abs(val) > threshold else ""
            bar = "#" * int(abs(val) * 20)
            print(f"    Lag {i:2d}: {val:+.3f} {bar}{sig}")
    except ImportError:
        print("\n  (statsmodels not installed, skipping PACF analysis)")


if __name__ == "__main__":
    df = load_and_prepare()
    forecaster = DemandForecaster()
    print("Training models...")
    forecaster.fit(df)
    forecaster.feature_importance()
    print("\nValidation:")
    forecaster.validate(df)
    forecaster.validate_old_vs_new(df)
    print("\nTest period forecasts:")
    test_rows = forecaster.get_test_features(df)
    for row in test_rows[:5]:
        q = forecaster.predict_quantiles(row)
        m = forecaster.predict_mean(row)
        print(f"  {row['date'].strftime('%Y-%m-%d')} dow={int(row['dow'])} promo={int(row['PROMO_01'])} "
              f"-> mean={m:.1f}, q50={q[0.5]}, q75={q[0.75]}, q90={q[0.9]}, q95={q[0.95]}")
