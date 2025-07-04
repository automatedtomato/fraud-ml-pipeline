import gc
import importlib
import math
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from common.log_setting import setup_logger
from fraud_detection.core.config import load_config
from fraud_detection.data.database import get_db_engine

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


# ========== Constants ==========
CHUNK_SIZE = 5000  # Chunk size for loading data.


# ========== Functions ==========
def _import_class(class_path: str):
    """
    Dynamically import a class object from import path (string)

    e.g.
        import_class("fraud_detection.models.xgboost_model.XGBoostModel")
    """

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)

    return getattr(module, class_name)


def _optimize_dtype(df: pd.DataFrame) -> pd.DataFrame:
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


def _calc_scale_pos_weight(
    table_name: str, train_rows: int, target: str, engine: object
) -> float:
    logger.info("Calculating scale_pos_weight for training data...")

    count_query = f"""
        WITH training_data AS (
            SELECT {target} FROM {table_name} ORDER BY trans_ts LIMIT {train_rows}
        )
        SELECT {target}, COUNT(*) as count FROM training_data GROUP BY {target}
    """

    class_counts = pd.read_sql(count_query, engine).set_index(target)["count"]
    neg_count = class_counts.get(0, 0)
    pos_count = class_counts.get(1, 1)

    if pos_count == 0:
        logger.warinig(
            "No positive samples in training data. Setting scale_pos_weight to 1."
        )
        return 1

    scale_pos_weight = neg_count / pos_count
    logger.info(f"scale_pos_weight: {scale_pos_weight:.3f}")

    return scale_pos_weight


def train(split_ratio: float = 0.8):
    """
    Load model config YAML and perform training
    """

    models_config = load_config(root="models")
    data_config = load_config(root="data")
    engine = get_db_engine()
    table_name = "feature_transactions"
    target = data_config["target_column"]
    features = data_config["common_features"]

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

    train_query = f"SELECT * FROM {table_name} ORDER BY trans_ts LIMIT {train_rows}"
    train_iterator = pd.read_sql_query(
        train_query, engine, chunksize=CHUNK_SIZE, parse_dates=["trans_ts"]
    )

    # Step 1: Train-val split and calc scale_pos_weight
    logger.info("--- Preparing Data ---")
    val_sample_size = 10000
    val_sample_query = f"""
        SELECT * FROM {table_name} ORDER BY RANDOM() OFFSET {train_rows} LIMIT {val_sample_size}
    """

    logger.info(
        f"Loading validation sample ({val_sample_size} rows) for early stopping..."
    )

    val_sample_df = pd.read_sql(val_sample_query, engine, parse_dates=["trans_ts"])
    val_sample_df = _optimize_dtype(val_sample_df)

    # The rest of the val data will be loaded via iterator for final evaluation
    val_query = f"SELECT * FROM {table_name} ORDER BY trans_ts OFFSET {train_rows}"
    val_iterator = pd.read_sql(
        val_query, engine, chunksize=CHUNK_SIZE, parse_dates=["trans_ts"]
    )

    # Step 2: Instantiate learning model
    scale_pos_weight = _calc_scale_pos_weight(table_name, train_rows, target, engine)

    models_to_train = {}
    for model_name, model_config in models_config.items():

        # Load`run: true` models
        if model_config.get("run", False):
            logger.info(f"Preparing model: '{model_name}'...")
            ModelClass = _import_class(model_config["class_path"])

            params = model_config["model_params"]
            if "xgboost" in model_name.lower():
                params["scale_pos_weight"] = scale_pos_weight

            models_to_train[model_name] = ModelClass(model_params=params)
        else:
            logger.info(f"Skipping model: '{model_name}'...")
            continue

    if not models_to_train:
        logger.warning("No models to train. Exiting...")
        return

    # Step 3: Train phase
    logger.info("--- Starting Training Phase ---")

    counter = 0
    X_val_sample = val_sample_df[features].copy()
    y_val_sample = val_sample_df[target].copy()

    for i, chunk_df in enumerate(train_iterator):

        # 3.1. Optimize data types
        chunk_df = _optimize_dtype(chunk_df)

        # 3.2. Train models
        for model_name, model in models_to_train.items():
            model_config = models_config[model_name]

            if not model_config.get("run", False):
                continue

            # features = model_config["features"]
            # X_val_sample = val_sample_df[features].copy()
            # y_val_sample = val_sample_df[target].copy()

            fit_params = model_config.get("fit_params", {})

            # Set evaluation dataset
            fit_params["eval_set"] = [(X_val_sample, y_val_sample)]

            # Set train dataset
            X_chunk = chunk_df[features].copy()
            y_chunk = chunk_df[target].copy()

            model.fit(X_chunk, y_chunk, **fit_params)

            counter += 1
            if counter >= 50:
                logger.info(f"Training on chunk {i+1}/{train_chunks} completed")
                counter = 0

        del chunk_df, X_chunk, y_chunk
        gc.collect()  # explicit garbage collection

    logger.info("--- Training Phase Completed ---")

    # Step 4: Validateion phase
    logger.info("--- Starting Validation Phase ---")
    all_preds = {model_name: [] for model_name in models_to_train.keys()}
    all_trues = []
    counter = 0

    for i, chunk_df in enumerate(val_iterator):

        chunk_df = _optimize_dtype(chunk_df)
        y_val_chunk = chunk_df[target].copy()

        for model_name, model in models_to_train.items():

            X_val_chunk = chunk_df[features].copy()
            preds = model.predict_proba(X_val_chunk)
            all_preds[model_name].extend(preds)
            all_trues.extend(y_val_chunk)

            counter += 1
            if counter >= 10:
                logger.info(
                    f"Validating on chunk {i+1}/{total_chunks - train_chunks} completed"
                )
                counter = 0

        del chunk_df, X_val_chunk, y_val_chunk
        gc.collect()  # explicit garbage collection

    # --- Step 5: Evaluate final model ---
    logger.info("--- Final Evaluation Results ---")
    y_true = np.array(all_trues)

    for model_name, y_pred_proba in all_preds.items():
        y_pred_proba = np.array(y_pred_proba)

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        # Placeholder for Precision@K, which should be in evaluation module
        # p_at_5 = precision_at_k(y_true, y_pred_proba, k_percent=5.0)

        logger.info(f"Model: {model_name}")
        logger.info(f"  - PR AUC: {pr_auc:.4f}")
        # logger.info(f"  - Precision@5%: {p_at_5:.4f}") ...

    logger.info("--- Validation Phase Completed ---")


if __name__ == "__main__":
    train()
