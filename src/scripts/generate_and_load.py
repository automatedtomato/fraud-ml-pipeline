import subprocess
from logging import getLogger
from pathlib import Path
from typing import List

import pandas as pd
from sqlalchemy import text

from common.log_setting import setup_logger
from fraud_detection.core.config import load_config, settings
from fraud_detection.data.database import get_db_engine
from fraud_detection.data.loader import load_data_to_postgres

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


# ========== Constants ==========
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_SCRIPT = (
    PROJECT_ROOT / "vendor" / "Sparkov_Data_Generation" / "datagen_mod.py"
)
OUTPUT = PROJECT_ROOT / "data" / "raw" / "generated"
TABLE_NAME = "transactions"


def find_transaction_files() -> List[Path]:
    """
    Search for generated transaction files and return
    """
    logger.info(f"Searching for generated CSV files in {OUTPUT}")

    all_csv_files = OUTPUT.glob("*.csv")

    transaction_files = [f for f in all_csv_files if f.name != "customers.csv"]

    if not transaction_files:
        raise FileNotFoundError(f"No CSV files found in {OUTPUT}")

    logger.info(f"Found {len(transaction_files)} CSV files.")
    return transaction_files


def run_datagen(n_cunstomers: int, start_date: str, end_date: str):
    """
    Run `datage.py from Sparkov_Data_Generation in a subprocess

    NOTE:
        Date must be formatted as "YYYY-MM-DD"
    """

    logger.info("Starting data generation...")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    generator_dir = GENERATOR_SCRIPT.parent

    script_name = GENERATOR_SCRIPT.name

    # Structure the command
    cmd = [
        "python",
        script_name,
        "-n",
        str(n_cunstomers),
        "-o",
        str(OUTPUT),
        start_date,
        end_date,
    ]

    logger.info(f"Execute command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=generator_dir,
        )
        logger.info("Data generation completed.")
        logger.debug(f"Generator stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error("Data generation failed")
        logger.error(f"Retrun code: {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise


def main():
    """
    Main function to generate and load data
    """

    generator_config = load_config(root="generator")

    n_customers = generator_config.get("num_customers", 1000)
    start_date = generator_config.get("start_date", "2020-01-01")
    end_date = generator_config.get("end_date", "2020-12-31")

    run_datagen(n_customers, start_date, end_date)

    transaction_files = find_transaction_files()

    try:
        master_columns = pd.read_csv(transaction_files[0], sep="|").columns.tolist()
        logger.info(f"Master columns set: {master_columns}")
    except IndexError:
        logger.error("No transaction files were generated. Exiting.")
        return

    for i, file_path in enumerate(transaction_files):
        logger.info(f"Processing file {i+1}/{len(transaction_files)}: {file_path.name}")

        df_chunk = pd.read_csv(file_path, sep="|", header=0, names=master_columns)

        mode = "replace" if i == 0 else "append"

        load_data_to_postgres(df_chunk, TABLE_NAME, mode)

    logger.info("Data loading completed.")

    engine = get_db_engine()
    with engine.connect() as conn:
        total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
    logger.info(f"Total rows in {TABLE_NAME}: {total_rows}")


if __name__ == "__main__":
    main()
