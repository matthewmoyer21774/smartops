from data_loader import load_data
from feature_engineering import build_features
from quantile_model import QuantileModel
from FEATURES import FEATURES_FULL, FEATURES_SIMPLE
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)

from sklearn.linear_model import QuantileRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")


def build_models(quantiles):
    models = {}

    # AR-style linear quantile regression
    models["AR_Quantile"] = QuantileModel(
        base_model=QuantileRegressor(alpha=0, solver="highs"),
        quantiles=quantiles,
        model_type="sklearn",
    )

    # Random Forest wrapper
    models["RF_Quantile"] = QuantileModel(
        base_model=RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
        ),
        quantiles=quantiles,
        model_type="sklearn",
    )

    models["LightGBM_Quantile"] = QuantileModel(
        base_model=lgb.LGBMRegressor(
            objective="quantile",  # required for quantile regression
            n_estimators=500,  # slightly more for stability with low learning rate
            learning_rate=0.05,  # a bit higher to converge faster
            max_depth=4,  # prevent overfitting small data
            num_leaves=7,  # small tree to avoid "no split" warnings
            min_data_in_leaf=2,  # ensures each leaf has enough samples
            subsample=0.9,  # row sampling for regularization
            colsample_bytree=0.8,  # feature sampling
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            random_state=42,
            importance_type="gain",  # for feature importance
            verbose=-1,  # suppress info logs
        ),
        quantiles=quantiles,
        model_type="lightgbm",
    )

    return models


def split_data(df, val_size=200, test_size=200):
    n = len(df)
    train_end = n - val_size - test_size
    val_end = n - test_size

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    return df_train, df_val, df_test


def train_models(models, df_train, df_val, features, target="sales"):
    X_train = df_train[features].values
    y_train = df_train[target].values

    X_val = df_val[features].values
    y_val = df_val[target].values

    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train, X_val, y_val)
        print(f"Finished {name}\n")

    return models


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_models_rolling_with_metrics(
    models, df, features, target="sales", horizon=4, lag_features=None
):
    """
    Evaluate trained models over a rolling horizon using:
      - Pinball loss per quantile
      - MAE and RMSE for the median (0.5) prediction

    Args:
        models: dict of QuantileModel objects
        df: pd.DataFrame containing features + target (validation or test set)
        features: list of feature column names
        target: target column name
        horizon: number of steps to forecast
        lag_features: list of lag feature names (optional)

    Returns:
        results: dict with pinball loss per quantile
        median_metrics: dict with MAE and RMSE for median prediction
    """

    def pinball_loss(y_true, y_pred, q):
        e = y_true - y_pred
        return np.mean(np.maximum(q * e, (q - 1) * e))

    df_copy = df.copy()

    # Initialize results
    results = {name: {q: [] for q in model.quantiles} for name, model in models.items()}
    median_metrics = {name: {"MAE": [], "RMSE": []} for name in models.keys()}

    for step in range(horizon):
        X = df_copy[features].values
        y_true = df_copy[target].values

        for name, model in models.items():
            # Quantile predictions
            preds = model.predict(X)  # dict {quantile: array}
            for q, p in preds.items():
                results[name][q].append(pinball_loss(y_true, p, q))

            # Median point prediction
            y_median = model.predict_point(X, strategy="median")
            median_metrics[name]["MAE"].append(mean_absolute_error(y_true, y_median))
            median_metrics[name]["RMSE"].append(
                root_mean_squared_error(y_true, y_median)
            )

            # Update lag features if provided
            if lag_features is not None:
                for lag_col in lag_features:
                    if "lag" in lag_col:  # sales_lag1, sales_lag2, etc.
                        lag_num = int(lag_col.split("_")[-1].replace("lag", ""))
                        df_copy[lag_col] = df_copy[target].shift(lag_num)
                    elif (
                        "roll" in lag_col
                    ):  # rolling features, just shift by 1 for next step
                        df_copy[lag_col] = df_copy[target].shift(1)
                # Drop NaNs introduced by shifting
                df_copy = df_copy.dropna(subset=features + [target])

    return results, median_metrics


if __name__ == "__main__":

    # get the dataframe
    df = load_data("df_6_art_train_project.parquet", art_id=2921141)

    # Drop the NaNs
    df = df.dropna(subset=["sales"])

    # Feature engineering
    df = build_features(df)
    df = df.dropna(subset=FEATURES_FULL)

    # Split into training, validation and test data
    df_train, df_val, df_test = split_data(df, val_size=365, test_size=365 * 2)

    # Build the models
    models = build_models(quantiles=(0.5, 0.7, 0.8, 0.9, 0.95, 0.99))

    # Train the models
    trained_models = train_models(
        models=models,
        df_train=df_train,
        df_val=df_val,
        features=FEATURES_FULL,
        target="sales",
    )

    # Extract lag features from FEATURES_FULL
    lag_features = [
        f
        for f in FEATURES_FULL
        if f.startswith("sales_lag") or f.startswith("sales_roll")
    ]

    print("Lag features:", lag_features)
    horizon = 4

    pinball_results, median_results = evaluate_models_rolling_with_metrics(
        models=trained_models,
        df=df_val,
        features=FEATURES_FULL,
        target="sales",
        horizon=horizon,
        lag_features=lag_features,
    )

    # Print pinball losses per quantile
    for model_name, quantile_losses in pinball_results.items():
        print(f"\nPinball Loss - Model: {model_name}")
        for q, losses in quantile_losses.items():
            print(f"  Quantile {q}: {losses} (avg: {np.mean(losses):.4f})")

    # Print MAE and RMSE for median predictions
    for model_name, metrics in median_results.items():
        mae_avg = np.mean(metrics["MAE"])
        rmse_avg = np.mean(metrics["RMSE"])
        print(f"\nMedian Prediction - Model: {model_name}")
        print(f"  MAE: {mae_avg:.4f}")
        print(f"  RMSE: {rmse_avg:.4f}")

    import matplotlib.pyplot as plt
    import numpy as np

    # Horizon (number of steps)
    horizon = len(
        next(iter(pinball_results.values()))[0.99]
    )  # length of horizon from results
    steps = np.arange(1, horizon + 1)

    # Models
    model_names = list(pinball_results.keys())

    # --- Plot 1: Quantile 0.99 Pinball Loss ---
    plt.figure(figsize=(8, 5))
    for model_name in model_names:
        q99_losses = pinball_results[model_name][0.99]  # list of losses per step
        plt.plot(steps, q99_losses, marker="o", label=model_name)

    plt.xlabel("Forecast Horizon")
    plt.ylabel("Pinball Loss (q=0.99)")
    plt.title("Quantile Loss (0.99) over Horizon")
    plt.xticks(steps)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # --- Plot 2: RMSE of Median Prediction ---
    plt.figure(figsize=(8, 5))
    for model_name in model_names:
        rmse_losses = median_results[model_name]["RMSE"]
        plt.plot(steps, rmse_losses, marker="o", label=model_name)

    plt.xlabel("Forecast Horizon")
    plt.ylabel("RMSE (Median Prediction)")
    plt.title("RMSE of Median Prediction over Horizon")
    plt.xticks(steps)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
