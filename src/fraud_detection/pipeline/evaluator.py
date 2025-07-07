from logging import getLogger

import numpy as np
import math
import pandas as pd

from typing import Dict, Any

from common.log_setting import setup_logger
from fraud_detection.utils import _optimize_dtype
from fraud_detection.evaluation.metrics import precision_at_k, pr_auc_score

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


def evaluate_models(
    trained_models: Dict[str, Any],
    data_config: Dict[str, Any],
    engine: Any,
) -> Dict[str, Any]:
    """
    Handles the evaluation phase for all models based on its strategy.
    
    Args:
        trained_models (Dict[str, Any]): Dictionary of trained model objects.
        data_config (Dict[str, Any]): Configuration for the data.
        engine (Any): Database engine.
        
    Returns:
        Dict[str, Any]: Dictionary of evaluated model objects.
    """
    
    logger.info("--- Starting Validation Phase ---")
    target_name = data_config["target_column"]
    table_name = "feature_transactions"
    features = data_config["common_features"]
    
    chunk_size = data_config.get("chunk_size", 5000)

    total_rows = pd.read_sql(f"SELECT COUNT(*) FROM {table_name}", engine).iloc[0, 0]
    train_rows = int(total_rows * data_config.get("split_ratio", 0.8))
    total_chunks = math.ceil(total_rows / chunk_size)
    train_chunks = math.ceil(train_rows / chunk_size)
    
    val_query = f"""
        SELECT * FROM {table_name} ORDER BY trans_ts OFFSET {train_rows}
    """
    val_iterator = pd.read_sql(val_query, engine, chunksize=chunk_size, parse_dates=["trans_ts"])
    
    all_preds = {model_name: [] for model_name in trained_models.keys()}
    all_trues = []
    counter = 0
    
    for i, chunk_df in enumerate(val_iterator):
        chunk_df = _optimize_dtype(chunk_df)
        chunk_df.fillna(0, inplace=True)
        all_trues.extend(chunk_df[target_name])
        
        for model_name, model in trained_models.items():
            X_val_chunk = chunk_df[features].copy()

            preds = model.predict_proba(X_val_chunk)
            if hasattr(preds, 'to_numpy'): preds = preds.to_numpy()
            all_preds[model_name].extend(preds)
            
            counter += 1
            if counter >= 10:
                logger.info(
                    f"Validating on chunk {i+1}/{total_chunks - train_chunks} completed"
                )
                counter = 0
            
        del chunk_df, X_val_chunk
        
    logger.info("--- Final Evaluation Results ---")
    y_true = np.array(all_trues)
    results = {model_name: {} for model_name in all_preds.keys()}
    
    for model_name, y_pred_proba in all_preds.items():
        y_pred_proba = np.array(y_pred_proba)
        results[model_name] = {}
        results[model_name]["pr_auc"] = pr_auc_score(y_true, y_pred_proba)
        results[model_name]["p@1"] = precision_at_k(y_true, y_pred_proba, 1.0)
        results[model_name]["p@3"] = precision_at_k(y_true, y_pred_proba, 3.0)
        results[model_name]["p@5"] = precision_at_k(y_true, y_pred_proba, 5.0)
        results[model_name]["p@10"] = precision_at_k(y_true, y_pred_proba, 10.0)
        
        logger.info(f"--- Model: '{model_name}' ---")
        for metric, value in results[model_name].items():
            logger.info(f"  - {metric}: {value:.4f}")
            
    logger.info("--- Validateion Phase Completed ---")
            
    return results


    