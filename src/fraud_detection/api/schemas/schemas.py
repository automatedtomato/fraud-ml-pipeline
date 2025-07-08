from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """
    Pydantic model for the incoming transaction darta for prediction.
    It includes a subset of the features our model was trained on.
    """

    trans_ts: str = Field(
        ..., example="2023-07-08 15:00:00", description="Timestamp of the transaction"
    )
    amt: float = Field(..., example=125.50, description="Transaction amount")
    lat: float = Field(..., example=40.7128, description="Latitude of the transaction")
    long: float = Field(
        ..., example=-74.0060, description="Longitude of the transaction"
    )
    city_pop: int = Field(
        ..., example=8419000, description="Population of the transaction city"
    )
    age_at_tx: int = Field(
        ...,
        example=25,
        description="Age of the customer at the time of the transaction",
    )
    hour_of_day: int = Field(
        ..., example=14, description="Hour of the day the transaction occurred"
    )
    day_of_week: int = Field(
        ...,
        example=3,
        description="Day of the week the transaction occurred (0: Sunday)",
    )
    distance_from_home_km: float = Field(
        ...,
        example=5.2,
        description="Distance from the customer's home to the transaction location",
    )
    time_since_last_tx_sec: Optional[float] = Field(
        0.0, example=3600.0, description="Time since the last transaction in seconds"
    )
    tx_in_last_24h: int = Field(
        ..., example=3, description="Number of transactions in the last 24 hours"
    )
    tx_in_last_7d: int = Field(
        ..., example=5, description="Number of transactions in the last 7 days"
    )
    avg_amt_in_last_24h: float = Field(
        ...,
        example=100.0,
        description="Average amount of transactions in the last 24 hours",
    )
    avg_amt_in_last_7d: float = Field(
        ...,
        example=200.0,
        description="Average amount of transactions in the last 7 days",
    )
    amt_ratio_to_avg: float = Field(
        ...,
        example=1.1,
        description="Ratio of the transaction amount to the average amount in the last 7 days",
    )
    ratio_to_avg_amt_in_last_7d: float = Field(
        ...,
        example=0.5,
        description="Ratio of the transaction amount to the average amount as of the transaction date",
    )
    ratio_to_avg_amt_in_last_24h: float = Field(
        ...,
        example=0.5,
        description="Ratio of the transaction amount to the average amount in the last 24 hours",
    )
    avg_amt_historical: float = Field(
        ..., example=200.0, description="Average amount of all transactions"
    )
    category_tx_count: int = Field(
        ...,
        example=3,
        description="Number of transactions in in the same category as the current transaction",
    )
    tx_with_same_merch_in_last_1h: int = Field(
        ...,
        example=2,
        description="Number of transactions with the same merchant in the last 1 hour",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "trans_ts": "2023-07-08 15:00:00",
                "amt": 125.50,
                "lat": 40.7128,
                "long": -74.0060,
                "city_pop": 8419000,
                "age_at_tx": 25,
                "hour_of_day": 14,
                "day_of_week": 3,
                "distance_from_home_km": 5.2,
                "time_since_last_tx_sec": 3600.0,
                "tx_in_last_24h": 3,
                "tx_in_last_7d": 5,
                "avg_amt_in_last_24h": 100.0,
                "avg_amt_in_last_7d": 200.0,
                "amt_ratio_to_avg": 1.1,
                "ratio_to_avg_amt_in_last_7d": 0.5,
                "ratio_to_avg_amt_in_last_24h": 0.5,
                "avg_amt_historical": 200.0,
                "category_tx_count": 3,
                "tx_with_same_merch_in_last_1h": 2,
            }
        }


class PredictionResponse(BaseModel):
    """
    Pydantic model for the prediction response.
    """

    fraud_score: float = Field(
        ..., example=0.85, description="The predicted fraud score for the transaction"
    )
    model_version: str = Field(
        ...,
        example="pytorch_tabnet_v2",
        description="The version of the model used for prediction",
    )
    risk_level: str = Field(
        ...,
        example="high",
        description="The risk level of the transaction (categorical: low, medium, high)",
    )
