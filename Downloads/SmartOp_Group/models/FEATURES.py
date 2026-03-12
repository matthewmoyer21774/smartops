FEATURES_FULL = [
    # calendar
    "dow",
    "is_saturday",
    "is_friday",
    "is_monday",
    "week_of_year",
    "month",
    # lag
    "sales_lag1",
    "sales_lag2",
    "sales_lag3",
    "sales_lag7",
    # rolling
    "sales_roll3",
    "sales_roll7",
    "sales_roll3_std",
    "sales_roll7_median",
    # promo / price
    "PROMO_01",
    "PROMO_DEPTH",
    "PRC_2_norm",
    "sat_x_promo",
    "fri_x_promo",
    "promo_x_depth",
    "price_x_promo",
    "price_lag1",
    "price_change",
    "price_roll7_mean",
    # holiday
    "OFFICIAL_HOLIDAY_01_f1",
    "OFFICIAL_HOLIDAY_01_l1",
    "near_holiday",
]

FEATURES_SIMPLE = ["dow", "sales_lag1", "sales_lag7", "PROMO_01"]
