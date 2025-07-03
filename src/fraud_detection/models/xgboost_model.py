from logging import getLogger
from typing import Dict, Any
import pandas as pd
from xgboost import XGBClassifier

from common.log_setting import setup_logger

from .base import BaseModel

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


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
        self.model = XGBClassifier(**self.model_params)
        self._is_trained = False  # Flag to manage incremental training

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        logger.info("Training XGBoost model...")
        
        model_to_train = self.model if not self._is_trained else None
        
        fit_params = kwargs.copy()
        
        self.model.fit(
            X_train,
            y_train,
            xgb_model=model_to_train,
            **fit_params
        )
        self._is_trained = True

        logger.info("Training step completed.")


    def predict(self, X_test: pd.DataFrame, **kwargs):
        logger.info("Predicting with XGBoost model...")

        # TODO: implement prediction logic

        logger.info("Prediction completed.")

        pass

    def predict_proba(self, X_test):
        if self.model is None or not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call `fit()` at least once.")
        
        logger.info("Predicting probability with XGBoost model...")
        proba = self.model.predict_proba(X_test)[:, 1]
        logger.info("Prediction completed.")
        
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
