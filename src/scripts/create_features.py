from logging import getLogger
from pathlib import Path

from sqlalchemy import text
from fraud_detection.data.database import get_db_engine

from common.log_setting import setup_logger


# ========== Log Setting ===========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")

# ========== Constants ==========
SQL_FILE_PATH = Path(__file__).parents[1] / "fraud_detection/features/aql/create_feature_transactions.sql"

def create_feature_table():
    """
    Create table for new features from SQL file
    """
    
    table_name = "feature_transactions"
    engine = get_db_engine()
    
    try:
        logger.info(f"Reading SQL file: {SQL_FILE_PATH}")
        with open(SQL_FILE_PATH, "r") as f:
            base_query = f.read()
    except FileNotFoundError:
        logger.error(f"SQL file not found: {SQL_FILE_PATH}")
        raise
    
    final_query = text(f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} AS
        {base_query};
    """)
    
    logger.info(f"Creating new table '{table_name}'...(this may take a while)")

    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(final_query)
        logger.info(f"New table '{table_name}' created.")
        
        with engine.connect() as conn:
            row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
            column_names = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1")).keys()
            logger.info(f"New table '{table_name}' has {row_count} rows.")
            logger.info(f"Columns: {list(column_names)}")
            
    except Exception as e:
        logger.error(f"Failed to create new table '{table_name}': {e}")
        raise
    
if __name__ == "__main__":
    create_feature_table()