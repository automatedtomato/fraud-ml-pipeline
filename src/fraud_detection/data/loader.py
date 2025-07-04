import io
from logging import getLogger

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from common.log_setting import setup_logger

from .database import get_db_engine

# ========== Log Setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names so that they can be loaded into the Postgres database
    """
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def load_data_to_postgres(
    df: pd.DataFrame, table_name: str, if_exists: str = "replace"
):
    """
    Load data into Postgres database
    """
    logger.info(f"Preparing to load {len(df)} rows into Postgres table {table_name}")

    df_clean = clean_col_names(df)

    engine = get_db_engine()

    try:
        with engine.connect() as conn:
            if if_exists == "replace":
                df_clean.head(0).to_sql(
                    name=table_name, con=conn, index=False, if_exists="replace"
                )
                logger.info(f"Table {table_name} created.")

            # Hand off data in csv to a buffer on memory
            buffer = io.StringIO()
            df_clean.to_csv(buffer, index=False, header=False, sep="|")
            buffer.seek(0)

            # Insert data with psycopg2 COPY command for faster loading
            raw_conn = conn.connection
            with raw_conn.cursor() as cur:
                cur.copy_expert(
                    f"COPY {table_name} FROM STDIN WITH (FORMAT text, DELIMITER E'|')",
                    buffer,
                )
            raw_conn.commit()
        logger.info(f"Successfully loaded {len(df_clean)} rows into '{table_name}'.")

    except SQLAlchemyError as e:
        logger.error(f"Error occurred during DB operation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise
