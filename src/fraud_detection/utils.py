import importlib
from logging import getLogger

import numpy as np
import pandas as pd

from common.log_setting import setup_logger

logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


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


def _import_class(class_path: str):
    """
    Dynamically import a class object from string object

    e.g.
        import_class("fraud_detection.models.xgboost_model.XGBoostModel")
    """

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)

    return getattr(module, class_name)
