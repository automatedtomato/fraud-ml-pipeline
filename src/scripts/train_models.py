import gc
import importlib
import math
from logging import getLogger

import numpy as np
import pandas as pd

from common.log_setting import setup_logger
from fraud_detection.core.config import load_config
from fraud_detection.data.database import get_db_engine

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


# ========== Constants ==========
CHUNK_SIZE = 10000  # Chunk size for loading data.


# ========== Functions ==========
def import_class(class_path: str):
    """
    Dynamically import a class object from import path (string)

    e.g.
        import_class("fraud_detection.models.xgboost_model.XGBoostModel")
    """

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)

    return getattr(module, class_name)


def optimize_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize data types and reduce memory overhead
    """

    for col in df.columns:
        if df[col].dtype.kind in "if":  # 'i' integer, 'f' float
            if df[col].dtype.kind == "i" and df[col].isnull().sum() == 0:  # integer
                if (
                    df[col].min() >= np.iinfo(np.int8).min
                    and df[col].max() <= np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype("int8")
                elif (
                    df[col].min() >= np.iinfo(np.int16).min
                    and df[col].max() <= np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype("int16")
                elif (
                    df[col].min() >= np.iinfo(np.int32).min
                    and df[col].max() <= np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype("int32")
            elif df[col].dtype == "float64":  # float
                df[col].dtype == "float32"
        elif df[col].dtype == "object" and df[col].unique().size < 100:  # categorical
            df[col].dtype == "category"

    return df


def train(split_ratio: float = 0.8):
    """
    Load model config YAML and perform training
    """

    config = load_config(root="models")
    engine = get_db_engine()
    table_name = "feature_transactions"

    # Step 1: Train-val split
    total_rows = pd.read_sql(f"SELECT COUNT(*) FROM {table_name}", engine).iloc[0, 0]
    train_rows = int(total_rows * split_ratio)

    total_chunks = math.ceil(total_rows / CHUNK_SIZE)
    train_chunks = math.ceil(train_rows / CHUNK_SIZE)

    logger.info(
        f"Total chunks: {total_chunks}, Train chunks: {train_chunks}, Validation chunks: {total_chunks - train_chunks}"
    )
    logger.info(
        f"Total rows: {total_rows}, Train rows: {train_rows}, Validation rows: {total_rows - train_rows}"
    )

    # Step 2: Instantiate learning model
    models_to_train = {}
    for model_name, model_config in config.items():

        # Load`run: true` models
        if model_config.get("run", False):
            logger.info(f"Preparing model: '{model_name}'...")
            ModelClass = import_class(model_config["class_path"])
            models_to_train[model_name] = ModelClass(model_config["model_params"])
        else:
            logger.info(f"Skipping model: '{model_name}'...")
            continue

    if not models_to_train:
        logger.warning("No models to train. Exiting...")
        return

    # Step 3: Train phase
    train_query = f"SELECT * FROM {table_name} ORDER BY trans_ts LIMIT {train_rows}"
    train_iterator = pd.read_sql_query(
        train_query, engine, chunksize=CHUNK_SIZE, parse_dates=["trans_ts"]
    )

    logger.info("--- Starting Training Phase ---")

    for i, chunk_df in enumerate(train_iterator):
        logger.info(f"Training on chunk {i+1}/{train_chunks}")

        # 3.1. Optimize data types
        mem_before = chunk_df.memory_usage(deep=True).sum() / 1024**2
        chunk_df = optimize_dtype(chunk_df)
        mem_after = chunk_df.memory_usage(deep=True).sum() / 1024**2
        logger.info(
            f"Memory usage optimized: {mem_before:.2f} MB -> {mem_after:.2f} MB"
        )

        # 3.2. Train models
        for model_name, model in models_to_train.items():
            features = model_config["features"]
            target = model_config["target"]
            X_chunk = chunk_df[features].copy()
            y_chunk = chunk_df[target].copy()

            model.fit(X_chunk, y_chunk)

        del chunk_df, X_chunk, y_chunk
        gc.collect()  # explicit garbage collection

    logger.info("--- Training Phase Completed ---")

    # Step 4: Validateion phase
    val_query = f"SELECT * FROM {table_name} ORDER BY trans_ts OFFSET {train_rows}"
    val_iterator = pd.read_sql_query(
        val_query, engine, chunksize=CHUNK_SIZE, parse_dates=["trans_ts"]
    )

    logger.info("--- Starting Validation Phase ---")
    all_preds = []
    all_trues = []
    for i, chunk_df in enumerate(val_iterator):
        logger.info(f"Validating on chunk {i+1}/{total_chunks - train_chunks}")

        mem_before = chunk_df.memory_usage(deep=True).sum() / 1024**2
        chunk_df = optimize_dtype(chunk_df)
        mem_after = chunk_df.memory_usage(deep=True).sum() / 1024**2
        logger.info(
            f"Memory usage optimized: {mem_before:.2f} MB -> {mem_after:.2f} MB"
        )

        X_val_chunk = chunk_df[features].copy()
        y_val_chunk = chunk_df[target].copy()

        for model_name, model in models_to_train.items():
            preds = model.predict_proba(X_val_chunk)
            all_preds.extend(preds)
            all_trues.extend(y_val_chunk)

        del chunk_df, X_val_chunk, y_val_chunk
        gc.collect()  # explicit garbage collection

    # TODO: Evaluating logic
    print()

    logger.info("--- Validation Phase Completed ---")


if __name__ == "__main__":
    train()
