from logging import getLogger
from typing import Any, Dict

import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from fraud_detection.evaluation.metrics import PR_AUC

from common.log_setting import setup_logger

from .base import BaseModel

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


class PyTorchModel(BaseModel):
    """
    TabNet model for fraud detection, using the pytorch-tabnet library.
    """

    def __init__(self, model_params: Dict[str, Any]):
        super().__init__(model_params)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TabNet run on device: {self.device}")
        self.optimizer_fn = None
        self.scheduler_fn = None

        # Instanciate TabNet
        # Pass optimizer and scheduler as parameters
        self.model = TabNetClassifier(device_name=self.device, **self.model_params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Fit TabNet model.
        It uses an `eval_set` for early stopping like XGBoost.
        """

        X_train_np = X_train.values
        y_train_np = y_train.values

        eval_set = kwargs.get("eval_set")
        eval_set_np = []
        
        if eval_set:
            X_val, y_val = eval_set[0]
            eval_set_np = [(X_val.values, y_val.values)]
            logger.info(f"Using validation set of size: {len(X_val)} for early stopping.")

        self.model.fit(
            X_train_np,
            y_train_np,
            eval_set=eval_set_np,
            patience=50,
            eval_metric=[PR_AUC],
            batch_size=1024 * 8,
            warm_start=True,  # Continue training from previous state
        )

    def predict_proba(self, X_test: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Predicts probabilities using TabNet model.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained yet. Call `fit()` at least once.")

        logger.info("Predicting with TabNet model...")
        X_test_np = X_test.values

        prods = self.model.predict_proba(X_test_np)[:, 1]

        pred_probas = pd.Series(prods, index=X_test.index)

        return pred_probas

    def predict(self, X_test, **kwargs):
        raise NotImplementedError("TabNet model does not have a predict() method.")

    def save_model(self, path: str):
        if self.model is None:
            raise RuntimeError("Cannot save a model that has not been fitted.")

        saved_path = self.model.save_model(path)
        logger.info(f"PyTorch TabNet model saved to {saved_path}")

    def load_model(self, path: str):
        logger.info(f"Loading PyTorch TabNet model from {path}.zip")
        self.model.load_model(f"{path}.zip")
