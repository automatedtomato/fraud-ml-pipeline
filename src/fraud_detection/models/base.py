from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """
    Abstract base model for all ML models.
    All models inherited from this class should implement the following methods:
        - train
        - predict
        - predict_proba
        - save_model
        - load_model
    """

    def __init__(self, model_params: dict = None):
        """
        Recieve params from YAML
        """
        self.model = None
        self.model_params = model_params

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame, **kwargs) -> pd.Series:
        pass

    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame, **kwargs) -> pd.Series:
        pass

    @abstractmethod
    def save_model(self, path: str):
        pass

    @abstractmethod
    def load_model(self, path: str) -> Any:
        pass
