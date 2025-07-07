from logging import getLogger
from typing import Any, Dict

import pandas as pd
import torch
from xgboost import XGBClassifier

from common.log_setting import setup_logger

from .base import BaseModel

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")

# ========== Device Setting ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")


class XGBoostModel(BaseModel):
    """
    XGBoost classifier class.
        - Runs based on YAML configuration
        - Capsules incremental training logic
        - Handles overfitting with early stopping
        - Handles imbalanced data with scale_pos_weight
    """

    def __init__(self, model_params: Dict[str, Any]):
        super().__init__(model_params)
        self.model = XGBClassifier(**self.model_params, device=device, n_jobs=-1)
        self._is_trained = False  # Flag to manage incremental training

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):

        booster = self.model.get_booster() if self._is_trained else None

        fit_params = kwargs.copy()

        self.model.fit(X_train, y_train, xgb_model=booster, **fit_params)
        self._is_trained = True

    def predict(self, X_test: pd.DataFrame, **kwargs):

        # TODO: implement prediction logic

        pass

    def predict_proba(self, X_test):
        if self.model is None or not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call `fit()` at least once.")

        proba = self.model.predict_proba(X_test)[:, 1]

        return proba

    def save_model(self, path):
        logger.info("Saving XGBoost model...")

        # TODO: implement saving logic

        logger.info("Model saved.")

        pass

    def load_model(self, path):
        logger.info("Loading XGBoost model...")

        # TODO: implement loading logic

        logger.info("Model loaded.")

        pass
