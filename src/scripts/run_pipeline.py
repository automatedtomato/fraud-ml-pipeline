import json
from datetime import datetime
from logging import getLogger
import mlflow

import pandas as pd

from common.log_setting import setup_logger
from fraud_detection.core.config import load_config
from fraud_detection.data.database import get_db_engine
from fraud_detection.pipeline.evaluator import evaluate_models
from fraud_detection.pipeline.trainer import train_models
from fraud_detection.utils import _calc_scale_pos_weight, _import_class

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


def run_pipeline():
    """
    Orchestrate the model training and evaluation pipeline.
    """
    # Set up MLflow tracking
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Fraud Detection Pipeline")
    
    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"==== Starting Fraud Detection Pipeline (MLflow Run ID: {run_id}) ====")
        mlflow.log_param("run_timestamp", datetime.now().isoformat())

        # Step 1: Load Configurations
        models_config = load_config(root="models")
        data_config = load_config(root="data")
        engine = get_db_engine()

        # Step 2: Prepare and Instantiate Models
        table_name = "feature_transactions"
        total_rows = pd.read_sql(f"SELECT COUNT(*) FROM {table_name}", engine).iloc[0, 0]
        train_rows = int(total_rows * data_config.get("split_ratio", 0.8))
        scale_pos_weight = _calc_scale_pos_weight(
            table_name, train_rows, data_config["target_column"], engine
        )

        models_to_train = {}
        active_models = []
        for model_name, model_config in models_config.items():
            if model_config.get("run", False):
                active_models.append(model_name)
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
                                # If conversion fails, it is genuinely string (like 'min'),
                                # so let's leave it as it is.
                                pass

                if "xgboost" in model_name.lower():
                    params["scale_pos_weight"] = scale_pos_weight
                models_to_train[model_name] = ModelClass(model_params=params)
                
                flat_params = pd.json_normalize(params, sep='_').to_dict(orient='records')[0]
                mlflow.log_params({f"{model_name}_{k}": v for k, v in flat_params.items()})
            else:
                logger.info(f"Skipping model: '{model_name}'...")
                continue
        if not models_to_train:
            logger.warning("No models to train. Exiting...")
            return

        mlflow.log_param("active_models", ", ".join(active_models))

        # Step 3: Train Models
        trained_models = train_models(
            models_to_train=models_to_train,
            models_config=models_config,
            data_config=data_config,
            engine=engine,
        )

        # Step 4: Save Models
        logger.info("--- Saving Trained Models ---")
        for model_name, model in trained_models.items():
            save_path = f"models/{model_name}"
            model.save_model(save_path)
            logger.info(f"Model '{model_name}' saved to '{save_path}'")
        
        mlflow.log_artifacts("models", artifact_path="trained_models")
        logger.info("Logged trained models as artifacts to MLflow.")

        # Step 5: Evaluate Models
        evaluation_results = evaluate_models(
            trained_models=trained_models,
            data_config=data_config,
            engine=engine,
        )

        # Step 6: Save Evaluation Results
        logger.info("--- Saving Evaluation Results ---")
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"reports/evaluation_results_{time_stamp}.json"
        logger.info(f"Saving evaluation results to '{results_path}'")

        for model_name, metrics in evaluation_results.items():
            for metric_name, value in metrics.items():
                valid_metric_name = metric_name.replace("@", "at")
                mlflow.log_metric(f"{model_name}_{valid_metric_name}", value)

        for model_name, metrics in evaluation_results.items():
            for metric_name, value in metrics.items():
                if hasattr(value, "item"):
                    metrics[metric_name] = (
                        value.item()
                    )  # Convert numpy/torch float/int to python float/int
                    
        with open(results_path, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        logger.info(f"Saved evaluation results to '{results_path}'")
        
        mlflow.log_artifact(results_path, artifact_path="evaluation_results")
        logger.info("Logged evaluation results as artifacts to MLflow.")

        logger.info("==== Fraud Detection Pipeline Completed ====")
        print(f"Final Results: {evaluation_results}")


if __name__ == "__main__":
    run_pipeline()
