import numpy as np
import pandas as pd


class DemandForecaster:

    def __init__(self, model, features, target_col="sales", quantile_model=True):
        """
        model: ForecastModel
        features: list of features to use
        """

        self.model = model
        self.features = features
        self.target_col = target_col
        self.quantile_model = quantile_model

        self.training_tail = []
        self.revealed_demands = []

    def _split_data(self, df, val_size=200, test_size=200):

        known = df.dropna(subset=[self.target_col])
        known = known.dropna(subset=self.features)

        n = len(known)

        train_end = n - val_size - test_size
        val_end = n - test_size

        train = known.iloc[:train_end]
        val = known.iloc[train_end:val_end]
        test = known.iloc[val_end:]

        return train, val, test

    def fit(self, df, val_size=200, test_size=200):

        train, val, _ = self._split_data(df, val_size, test_size)

        X_train = train[self.features].values
        y_train = train[self.target_col].values

        X_val = val[self.features].values
        y_val = val[self.target_col].values

        print(f"Training model using {len(self.features)} features")

        self.model.fit(X_train, y_train, X_val, y_val)

        return self

    def predict_row(self, row):

        X = np.array([[row[f] for f in self.features]])

        if self.quantile_model:

            q_preds = self.model.predict_quantiles(X)

            result = {q: round(v[0], 2) for q, v in q_preds.items()}

            result["mean"] = round(self.model.predict(X)[0], 2)

            return result

        else:

            return {"mean": round(self.model.predict(X)[0], 2)}

    def predict_dataframe(self, df):

        X = df[self.features].values

        if self.quantile_model:

            q_preds = self.model.predict_quantiles(X)

            result = pd.DataFrame()

            for q, vals in q_preds.items():
                result[f"q{q}"] = vals

            result["mean"] = self.model.predict(X)

            return result

        else:

            return pd.DataFrame({"mean": self.model.predict(X)})

    def get_test_rows(self, df):

        known = df.dropna(subset=[self.target_col])
        self.training_tail = known[self.target_col].values[-7:].tolist()
        self.revealed_demands = []

        test = df[df[self.target_col].isna()].copy()

        rows = []

        for _, r in test.iterrows():

            rows.append({f: r[f] for f in self.features} | {"date": r["date"]})

        return rows
