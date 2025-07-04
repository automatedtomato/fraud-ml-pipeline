from logging import getLogger
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from common.log_setting import setup_logger

from .base import BaseModel

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


class IsolationForestModel(BaseModel):
    """
    Time window ensemble model using multiple Isolation Forest instances.
    Each data chunk is used to train a separate Isolation Forest model.
    Predictions are an average of anomaly scores from all models.
    """

    def __init__(self, model_params: Dict[str, Any]):
        super().__init__(model_params)
        self.models: List[IsolationForest] = []
        self.scaler = MinMaxScaler()
        self._is_trained = False
        self.counter = 0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Fit a new IF model on the given data chunk and adds it to the ensemble.
        y_train is ignored (unsupervised learning).
        """

        model_for_chunk = IsolationForest(**self.model_params)
        model_for_chunk.fit(X_train)

        self.models.append(model_for_chunk)
        self._is_trained = True

        self.counter += 1
        if self.counter >= 10:
            logger.info(f"Fitted total of {len(self.models)} models...")
            self.counter = 0

    def predict_proba(self, X_test: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Returns a fraud score based on the anomaly score.
        The score is not a true probability but is scaled to the [0, 1] range,
        where 1 indicates the highest likelyhood of being an anomaly (fraud).
        """
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call `fit()` at least once.")

        # Create linear weights: [1, 2, 3, ..., N] where N is the number of models
        all_scores = np.zeros(len(X_test), dtype=np.float64)

        weights = np.arange(1, len(self.models) + 1)

        for i, model in enumerate(self.models):
            anomaly_scores = model.decision_function(X_test)
            fraud_scores = -anomaly_scores

            all_scores += fraud_scores * weights[i]

            self.counter += 1
            if self.counter >= 10:
                logger.info(f"Predicted with total of {len(self.models)} models...")
                self.counter = 0

        weighted_avg_scores = all_scores / np.sum(weights)

        return pd.Series(weighted_avg_scores, index=X_test.index)

    def predict(self, X_test: pd.DataFrame, **kwargs) -> pd.Series:
        # For an ensemble, a simple majority vote or average score threshold could work.
        # Let's use the average score and a threshold of 0 for now.
        avg_scores = self.predict_proba(X_test, **kwargs)
        # predict() should return binary 0 or 1, let's use a placeholder logic
        # -1 (anomaly) vs 1 (normal) is the sklearn standard. We will map to 1 vs 0.
        # This part might need more tuning based on the score distribution.
        return pd.Series(np.where(avg_scores > 0, 1, 0), index=X_test.index)

    def save_model(self, path: str):
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call `fit()` at least once.")

        logger.info(f"Saving Isolation Forest model and scaler to {path}...")
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load_model(self, path: str):
        logger.info(f"Loading Isolation Forest model and scaler from {path}...")
        saved_objects = joblib.load(path)
        self.model = saved_objects["model"]
        self.scaler = saved_objects["scaler"]
        self._is_trained = True
