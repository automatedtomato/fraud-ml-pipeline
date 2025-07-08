from fastapi import FastAPI

from logging import getLogger

import pandas as pd

from common.log_setting import setup_logger
from .schemas.schemas import TransactionRequest, PredictionResponse

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


# ========== FastAPI Setting ==========
app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time fraud scoring",
    version="0.1.0",
)

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "OK", "message": "API is running"}

@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: TransactionRequest):
    """
    Accepts transaction data and returns a fraud score.
    """
    # NOTE: For now, returns dummy response.
    dummy_score = 0.85
    risk_level = "low"
    if dummy_score > 0.8:
        risk_level = "high"
    elif dummy_score > 0.5:
        risk_level = "medium"
        
    return PredictionResponse(fraud_score=dummy_score, model_version="dummy", risk_level=risk_level)
    