# quantile_model_final.py
import copy
import numpy as np
import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")


class QuantileModel:
    """
    Generic probabilistic forecasting wrapper supporting quantiles.
    Works with sklearn-like models and LightGBM (sklearn API).
    Always uses validation set if provided.
    """

    def __init__(self, base_model, quantiles=(0.5, 0.7, 0.9), model_type="sklearn"):
        self.base_model = base_model
        self.quantiles = sorted(quantiles)
        self.model_type = model_type
        self.models = {}

    def _build_model(self, q):
        """Clone base model and set quantile/alpha if supported"""
        model = copy.deepcopy(self.base_model)

        if self.model_type == "lightgbm":
            model.set_params(objective="quantile", alpha=q)
        elif self.model_type == "sklearn":
            if hasattr(model, "quantile"):
                model.set_params(quantile=q)
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train one model per quantile, always using validation if provided."""
        if X_val is None or y_val is None:
            raise ValueError("Validation set (X_val, y_val) must be provided!")

        self.models = {}

        for q in self.quantiles:
            model = self._build_model(q)

            # LightGBM sklearn API: use eval_set + early stopping
            if self.model_type == "lightgbm":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="quantile",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(period=0),
                    ],
                )
            else:
                # For other sklearn models that support validation, add custom logic here
                model.fit(X_train, y_train)

            self.models[q] = model

    def predict(self, X):
        """Return dictionary of predictions for all quantiles"""
        preds = {}
        for q, model in self.models.items():
            p = model.predict(X)
            preds[q] = np.maximum(0, p)
        return preds

    def predict_point(self, X, strategy="median"):
        """
        Derive a single point forecast from quantiles
        strategy: 'median' or 'mean'
        """
        all_preds = np.array(list(self.predict(X).values()))
        if strategy == "median":
            return np.median(all_preds, axis=0)
        elif strategy == "mean":
            return np.mean(all_preds, axis=0)
        else:
            raise ValueError("strategy must be 'median' or 'mean'")
