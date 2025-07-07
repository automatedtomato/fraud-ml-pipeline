# Fraud ML Pipeline

A machine learning pipeline implementation project specialized for fraud detection. This project aims to learn and practice MLOps workflows using PostgreSQL + MLflow + FastAPI.

## Project Overview

This project builds a practical, end-to-end fraud detection system using a large-scale (2.5M+ transactions) simulated credit card dataset. It covers the entire ML lifecycle, from raw data generation and advanced feature engineering to training multiple model types, comparative evaluation, and preparation for API deployment.

The core of the project is a flexible pipeline that trains and evaluates three fundamentally different models to tackle the fraud detection problem from various angles:
1.  **XGBoost**: A powerful, industry-standard gradient boosting model.
2.  **Isolation Forest**: An unsupervised anomaly detection model, implemented as a custom time-window ensemble to handle concept drift.
3.  **PyTorch (TabNet)**: A modern, attention-based deep learning model for tabular data.

### Key Features
- **Modular & Maintainable Architecture**: The codebase is refactored into distinct `trainer` and `evaluator` modules, making it easy to test, maintain, and extend.
- **Three-Model Comparison**: Evaluates Gradient Boosting, Unsupervised Anomaly Detection, and Deep Learning approaches side-by-side.
- **Advanced Feature Engineering**: Utilizes PostgreSQL window functions to create sophisticated time-series and user behavior features.
- **Memory-Efficient Pipeline**: Handles large datasets by using chunk-based processing (`incremental` training) and custom ensemble techniques.
- **Business-Oriented Evaluation**: Focuses on `Precision@K` as a key metric to measure the efficiency of fraud investigations.
- **MLOps Ready**: Integrated with Docker for reproducibility and MLflow for experiment tracking.

## Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy, psycopg2-binary
- **Database**: PostgreSQL
- **ML**: PyTorch (TabNet), XGBoost, Scikit-learn
- **Data Handling**: Pandas, NumPy
- **MLOps**: MLflow, Docker, Docker Compose
- **Development**: Poetry, Jupyter, Pytest

## Setup and Usage

### 1. Prerequisites
- Docker and Docker Compose
- Git

### 2. Initial Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/automatedtomato/fraud-ml-pipeline.git
    cd fraud-ml-pipeline
    ```
2.  **Create `.env` file:**
    Copy the example environment file. The default values are fine for local development.
    ```bash
    cp .env.example .env
    ```
3.  **Build and run containers:**
    This will start the development container, PostgreSQL, and MLflow.
    ```bash
    docker-compose up -d --build

    # or wiht GPU
    docker-compose -f docker-compose.gpu.yml up -d
    ```
    

### 3. Raw Data Generation
Run the following command to generate the initial raw transaction data into the database.
```bash
docker-compose exec dev poetry run python src/scripts/generate_and_load.py
```

### 4. Feature Generation
This script executes the entire feature engineering pipeline. It reads the raw data, calculates all features, and creates the final `feature_transactions` table.
```bash
docker-compose exec dev poetry run python src/scripts/create_features.py
```

### 5. Run Training & Evaluation Pipeline
This is the main script to train all models specified in `config/config.yml` and output the evaluation results.
```bash
docker-compose exec dev poetry run python src/scripts/run_pipeline.py
```

### 6. Verify Data (Optional)
You can verify that the feature table has been created with the following command:
```bash
docker-compose exec db psql -U user -d fraud_db -c "SELECT COUNT(*) FROM feature_transactions;"
```

### Development Environment
It is highly recommended to use the **VSCode Dev Containers** extension for a seamless development experience.
1.  Open the project folder in VSCode.
2.  When prompted, click "Reopen in Container".

## Development Progress

  - ✅ **Sprint 1: Infrastructure setup & EDA - COMPLETE**
  - ✅ **Sprint 2: Feature engineering & Pipeline - COMPLETE**
  * ✅ **Sprint 3: Machine learning model comparison - COMPLETE**
  -  Sprint 4: API development & integration

## License

MIT License - See LICENSE file for details

## Author

Hikaru Tomizawa (富澤晃)

-----

*This project is developed for learning and portfolio purposes.*