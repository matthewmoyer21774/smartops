import numpy as np
import pandas as pd


def add_calendar_features(df):

    df["dow"] = df["date"].dt.dayofweek
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_monday"] = (df["dow"] == 0).astype(int)

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    return df


def add_lag_features(df, lags=(1, 2, 3, 7)):

    for lag in lags:
        df[f"sales_lag{lag}"] = df["sales"].shift(lag)

    return df


def add_rolling_features(df):

    df["sales_roll3"] = df["sales"].shift(1).rolling(3, min_periods=1).mean()
    df["sales_roll7"] = df["sales"].shift(1).rolling(7, min_periods=1).mean()

    df["sales_roll3_std"] = (
        df["sales"].shift(1).rolling(3, min_periods=1).std().fillna(0)
    )

    df["sales_roll7_median"] = df["sales"].shift(1).rolling(7, min_periods=1).median()

    return df


def add_price_features(df):

    df["price_lag1"] = df["PRC_2_norm"].shift(1)
    df["price_change"] = df["PRC_2_norm"] - df["price_lag1"]

    df["price_roll7_mean"] = df["PRC_2_norm"].shift(1).rolling(7, min_periods=1).mean()

    return df


def add_interactions(df):

    df["sat_x_promo"] = df["is_saturday"] * df["PROMO_01"]
    df["fri_x_promo"] = df["is_friday"] * df["PROMO_01"]

    df["promo_x_depth"] = df["PROMO_01"] * df["PROMO_DEPTH"]

    df["price_x_promo"] = df["PRC_2_norm"] * df["PROMO_01"]

    return df


def add_holiday_features(df):

    df["near_holiday"] = df[["OFFICIAL_HOLIDAY_01_f1", "OFFICIAL_HOLIDAY_01_l1"]].max(
        axis=1
    )

    return df


def finalize_features(df):

    lag_cols = [c for c in df.columns if "lag" in c or "roll" in c]

    df[lag_cols] = df[lag_cols].fillna(0)

    return df


def build_features(df):
    """
    Main feature engineering pipeline.
    """

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_price_features(df)
    df = add_interactions(df)
    df = add_holiday_features(df)
    df = finalize_features(df)

    return df
