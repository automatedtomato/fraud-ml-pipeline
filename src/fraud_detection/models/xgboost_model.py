import pandas as pd
from xgboost import XGBClassifier
from .base import BaseModel

from logging import getLogger

from common.log_setting import setup_logger

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


class XGBoost(BaseModel):
    """
    XGBoost classifier class.
    Runs based on YAML configuration
    """
    
    def __init__(self, model_params: dict):
        super().__init__(model_params)
        self.model = XGBClassifier(**self.model_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        logger.info("Training XGBoost model...")
        
        # TODO: implement training logic
        
        logger.info("Training completed.")
        
        pass
    
    def predict(self, X_test: pd.DataFrame, **kwargs):
        logger.info("Predicting with XGBoost model...")
        
        # TODO: implement prediction logic
        
        logger.info("Prediction completed.")
        
        pass
    
    def predict_proba(self, X_test, **kwargs):
        logger.info("Predicting probability with XGBoost model...")
        
        # TODO: implement prediction logic
        
        logger.info("Prediction completed.")
        
        pass
    
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