from logging import getLogger
import pandas as pd

from common.log_setting import setup_logger
from fraud_detection.core.config import load_config
from fraud_detection.data.database import get_db_engine
from fraud_detection.pipeline.trainer import train_models
from fraud_detection.pipeline.evaluator import evaluate_models
from fraud_detection.utils import _import_class, _calc_scale_pos_weight


# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


def run_pipeline():
    """
    Orchestrate the model training and evaluation pipeline.
    """
    
    logger.info("==== Starting Fraud Detection Pipeline ====")
    
    # Step1: Load Configurations
    models_config = load_config(root="models")
    data_config = load_config(root="data")
    engine = get_db_engine()
    
    # Step2: Prepare and Instantiate Models
    table_name = "feature_transactions"
    total_rows = pd.read_sql(f"SELECT COUNT(*) FROM {table_name}", engine).iloc[0, 0]
    train_rows = int(total_rows * data_config.get("split_ratio", 0.8))
    scale_pos_weight = _calc_scale_pos_weight(
        table_name, train_rows, data_config["target_column"], engine
    )
    
    models_to_train = {}
    for model_name, model_config in models_config.items():
        if model_config.get("run", False):
            ModelClass = _import_class(model_config["class_path"])
            params = model_config.get("model_params", {})
            
            if "optimizer_fn" in params and isinstance(params["optimizer_fn"], str):
                params["optimizer_fn"] = _import_class(params["optimizer_fn"])

            if "scheduler_fn" in params and isinstance(params["scheduler_fn"], str):
                params["scheduler_fn"] = _import_class(params["scheduler_fn"])

            for param_dict_key in ["optimizer_params", "scheduler_params"]:
                if param_dict_key in params:
                    for key, value in params[param_dict_key].items():
                        try:
                            params[param_dict_key][key] = float(value)
                        except (ValueError, TypeError):
                            # If conversion fails, it is genuinely string (like 'min'), so let's leave it as it is.
                            pass
                        
            if "xgboost" in model_name.lower():
                params["scale_pos_weight"] = scale_pos_weight

            models_to_train[model_name] = ModelClass(model_params=params)
        else:
            logger.info(f"Skipping model: '{model_name}'...")
            continue
    if not models_to_train:
        logger.warning("No models to train. Exiting...")
        return
    
    # Step3: Train Models
    trained_models = train_models(
        models_to_train=models_to_train,
        models_config=models_config,
        data_config=data_config,
        engine=engine,
    )
    
    # Step4: Evaluate Models
    evaluation_results = evaluate_models(
        trained_models=trained_models,
        data_config=data_config,
        engine=engine,
    )
    
    logger.info("==== Fraud Detection Pipeline Completed ====")
    print(f"Final Results: {evaluation_results}")
    
    
if __name__ == "__main__":
    run_pipeline()