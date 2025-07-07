from logging import getLogger

import numpy as np
import pandas as pd
import math
import gc
from typing import Dict, Any

from common.log_setting import setup_logger
from fraud_detection.utils import _optimize_dtype

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


def train_models(
    models_to_train: Dict[str, Any],
    models_config: Dict[str, Any],
    data_config: Dict[str, Any],
    engine: Any,
) -> Dict[str, Any]:
    
    """
    Handles the training phase for all models based on its strategy.
    
    Args:
        models_to_train (Dict[str, Any]): Dictionary of instantiated model objects.
        models_config (Dict[str, Any]): Configuration for the models.
        data_config (Dict[str, Any]): Configuration for the data.
        engine (Any): Database engine.
        val_sample (int): Number of validation samples.
        
    Returns:
        Dict[str, Any]: Dictionary of trained model objects.
    """
    
    table_name = "feature_transactions"
    target = data_config["target_column"]
    features = data_config["common_features"]
    chunk_size = data_config.get("chunk_size", 5000)
    val_sample_size = data_config.get("val_sample", 10000)    
    
    total_rows = pd.read_sql(f"SELECT COUNT(*) FROM {table_name}", engine).iloc[0, 0]
    train_rows = int(total_rows * data_config.get("split_ratio", 0.8))
    train_query = f"""
        SELECT * FROM {table_name} ORDER BY trans_ts LIMIT {train_rows}
    """
    
    total_chunks = math.ceil(total_rows / chunk_size)
    train_chunks = math.ceil(train_rows / chunk_size)
    
    logger.info(
        f"Total chunks: {total_chunks}, Train chunks: {train_chunks}, Validation chunks: {total_chunks - train_chunks}"
    )
    logger.info(
        f"Total rows: {total_rows}, Train rows: {train_rows}, Validation rows: {total_rows - train_rows}"
    )
    
    # Step1: Prepare data
    logger.info("--- Preparing Data ---")
    val_sample_query = f"""
        SELECT * FROM {table_name} ORDER BY trans_ts LIMIT {val_sample_size}
    """
    val_sampl_df = pd.read_sql(
        val_sample_query, engine, parse_dates=["trans_ts"]
    )
    logger.info(f"Validation sample size for early stopping: {val_sample_size}")
    
    val_sample_df = _optimize_dtype(val_sampl_df)
    val_sample_df.fillna(0, inplace=True)
    
    # Step2: Train
    for model_name, model in models_to_train.items():
        model_config = models_config[model_name]
        strategy = model_config.get("training_strategy", "incremental")
        
        logger.info(f"--- Starting Training Phase | Model: '{model_name}' | Strategy: '{strategy}' ---")
        if strategy == 'incremental':
            train_iterator = pd.read_sql_query(
                train_query, engine, chunksize = chunk_size,  parse_dates = ["trans_ts"]
            )
            
            counter = 0
            
            x_val_sample = val_sample_df[features].copy()
            y_val_sample = val_sample_df[target].copy()
            
            fit_params = model_config.get("fit_params", {})
            fit_params["eval_set"] = [(x_val_sample, y_val_sample)]
            
            logger.info(f"Starting incremental training for '{model_name}...")
            for i, chunk_df in enumerate(train_iterator):
                chunk_df.fillna(0, inplace=True)
                chunk_df = _optimize_dtype(chunk_df)
                X_chunk = chunk_df[features].copy()
                y_chunk = chunk_df[target].copy()
                model.fit(X_chunk, y_chunk, **fit_params)
                
                counter += 1
                
                if counter >= 50:
                    logger.info(f"Training on chunk {i+1}/{train_chunks} completed")
                    counter = 0
                    
                del chunk_df, X_chunk, y_chunk
                gc.collect()  # explicit garbage collection
                
        elif strategy == 'batch':
            # For batch models that can't handle the full dataset,
            # we train on a large, random sample of the training data.
            batch_train_sample_size = data_config.get("batch_train_sample_size", 100000)
            
            logger.info(f"Loading {batch_train_sample_size} random samples from training data for batch training of '{model_name}'...")
            
            # SQL to get a random sample ONLY from the training portion of the data
            # This uses a WITH clause to first define the training set, then sample from it.
            batch_train_query = f"""
                WITH training_data AS ({train_query})
                SELECT * FROM training_data ORDER BY RANDOM() LIMIT {batch_train_sample_size}
            """
            train_df = pd.read_sql(batch_train_query, engine, parse_dates=["trans_ts"])
            
            logger.info(f"Loaded {len(train_df)} rows for batch training.")
            
            train_df.fillna(0, inplace=True)
            train_df = _optimize_dtype(train_df)
            
            # Use the pre-loaded val_sample_df for early stopping
            X_val_sample = val_sample_df[features]
            y_val_sample = val_sample_df[target]
            fit_params = model_config.get("fit_params", {})
            fit_params["eval_set"] = [(X_val_sample, y_val_sample)]

            model.fit(train_df[features], train_df[target], **fit_params)
            
            del train_df
            gc.collect()
        
    logger.info("--- Training Phase Completed ---")
    return models_to_train