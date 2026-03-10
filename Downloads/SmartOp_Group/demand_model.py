"""
Demand forecasting for SKU 2921141 using LightGBM quantile regression.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


def load_and_prepare(parquet_path="df_6_art_train_project.parquet", art_id=2921141):
    """Load data, filter to target SKU, engineer features."""
    df = pd.read_parquet(parquet_path)
    df = df[df["art_id"] == art_id].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["dow"] = df["date"].dt.dayofweek  # 0=Mon ... 6=Sun
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    return df


FEATURES = [
    "dow", "is_saturday", "is_friday",
    "PROMO_01", "PROMO_DEPTH", "PRC_2_norm",
    "OFFICIAL_HOLIDAY_01_f1", "OFFICIAL_HOLIDAY_01_l1",
]


class DemandForecaster:
    def __init__(self, quantiles=(0.50, 0.75, 0.90, 0.95)):
        self.quantiles = quantiles
        self.models = {}  # {quantile: lgb.Booster}
        self.mean_model = None

    def fit(self, df):
        """Train quantile models on rows with known sales."""
        train = df.dropna(subset=["sales"]).copy()
        X = train[FEATURES].values
        y = train["sales"].values

        # Train mean model (for reference)
        ds_mean = lgb.Dataset(X, label=y)
        self.mean_model = lgb.train(
            {"objective": "poisson", "verbose": -1, "n_jobs": 1,
             "num_leaves": 16, "min_child_samples": 20, "n_estimators": 300,
             "learning_rate": 0.05},
            ds_mean, num_boost_round=300,
        )

        # Train quantile models
        for q in self.quantiles:
            ds = lgb.Dataset(X, label=y)
            model = lgb.train(
                {"objective": "quantile", "alpha": q, "verbose": -1, "n_jobs": 1,
                 "num_leaves": 16, "min_child_samples": 20,
                 "learning_rate": 0.05},
                ds, num_boost_round=300,
            )
            self.models[q] = model

        return self

    def _feature_row(self, row):
        """Extract feature array from a dict or Series."""
        return np.array([[row[f] for f in FEATURES]])

    def predict_quantiles(self, row):
        """Return {quantile: value} for a given feature row."""
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
        """Return list of feature dicts for the 26 test periods."""
        test = df[df["sales"].isna()].copy()
        rows = []
        for _, r in test.iterrows():
            rows.append({f: r[f] for f in FEATURES} | {"date": r["date"]})
        return rows

    def validate(self, df, n_test=200):
        """Time-series validation on last n_test training rows."""
        known = df.dropna(subset=["sales"]).copy()
        train = known.iloc[:-n_test]
        test = known.iloc[-n_test:]

        # Temporarily train on subset
        X_train = train[FEATURES].values
        y_train = train["sales"].values
        X_test = test[FEATURES].values
        y_test = test["sales"].values

        print(f"Validation: train={len(train)}, test={len(test)}")
        for q in self.quantiles:
            ds = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(
                {"objective": "quantile", "alpha": q, "verbose": -1, "n_jobs": 1,
                 "num_leaves": 16, "min_child_samples": 20,
                 "learning_rate": 0.05},
                ds, num_boost_round=300,
            )
            preds = model.predict(X_test)
            # Pinball loss
            errors = y_test - preds
            pinball = np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))
            coverage = np.mean(y_test <= preds)
            print(f"  q={q:.2f}: pinball={pinball:.3f}, coverage={coverage:.1%} (target {q:.0%})")


if __name__ == "__main__":
    df = load_and_prepare()
    forecaster = DemandForecaster()
    print("Training models...")
    forecaster.fit(df)
    print("\nValidation:")
    forecaster.validate(df)
    print("\nTest period forecasts:")
    test_rows = forecaster.get_test_features(df)
    for row in test_rows[:5]:
        q = forecaster.predict_quantiles(row)
        m = forecaster.predict_mean(row)
        print(f"  {row['date'].strftime('%Y-%m-%d')} dow={int(row['dow'])} promo={int(row['PROMO_01'])} "
              f"-> mean={m:.1f}, q50={q[0.5]}, q75={q[0.75]}, q90={q[0.9]}, q95={q[0.95]}")
