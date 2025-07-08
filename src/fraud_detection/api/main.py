from logging import getLogger
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException

from common.log_setting import setup_logger
from fraud_detection.core.config import load_config
from fraud_detection.models.pytorch_model import PyTorchModel

from .schemas.schemas import PredictionResponse, TransactionRequest

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


# ========== Base Setting ==========
# DEBUG CODE
file_path = Path(__file__)
# DEBUG CODE END
app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time fraud scoring",
    version="0.1.0",
)

model_assets: Dict[str, Any] = {"model": None, "model_version": None}

# Load feature list
try:
    data_config = load_config(root="data")
    FEATURES = data_config["common_features"]
    logger.info(f"Features loaded. Total {len(FEATURES)} features.")
except Exception as e:
    logger.error(f"Failed to load features: {e}")
    FEATURES = []


# ========== Endpoints ==========
@app.on_event("startup")
async def load_model():
    """
    Load the champion model into memory when the API starts up.
    This is much more efficient than loading the model every time a prediction is made.
    """

    logger.info("--- Loading model at startup... ---")

    # Instanciate the model class. We need the same params as during training for the architecture.
    model_params = (
        load_config(root="models").get("pytorch_tabnet_v1", {}).get("model_params", {})
    )
    model_assets["model"] = PyTorchModel(model_params=model_params)

    model_version = "pytorch_tabnet_v1"
    model_path = f"models/{model_version}"

    try:
        model_assets["model"].load_model(model_path)
        model_assets["model_version"] = model_version
        logger.info(f"Succeccfully loaded model: {model_version}")
    except FileNotFoundError:
        logger.error(
            f"Model not found: {model_path}. The API will not be able to make predictions."
        )
        model_assets["model"] = None


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {
        "status": "OK",
        "message": "API is running healthy.",
        "serving_from": str(file_path),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: TransactionRequest):
    """
    Accepts transaction data and returns a fraud score.
    """

    if model_assets["model"] is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. The service is not available."
        )
    if not FEATURES:
        raise HTTPException(
            status_code=500, detail="Features not loaded. The service is not available."
        )

    try:
        # 1. Convert the incoming pydantic request to a pandas DataFrame
        input_df = pd.DataFrame([transaction.model_dump()])

        # 2. Select and reorder columns to match the training feature set
        input_feature_df = input_df[FEATURES]

        # 3. Get the fraud score
        score = model_assets["model"].predict_proba(input_feature_df).iloc[0]

        # 4. Determine the risk level
        risk_level = "low"

        if score > 0.8:
            risk_level = "high"
        elif score > 0.5:
            risk_level = "medium"

        return PredictionResponse(
            fraud_score=score,
            model_version=model_assets["model_version"],
            risk_level=risk_level,
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {str(e)}."
        )
