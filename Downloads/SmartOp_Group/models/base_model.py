from abc import ABC, abstractmethod


class ForecastModel(ABC):
    """
    Base interface for any forecasting model.
    """

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass
