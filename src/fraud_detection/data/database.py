from logging import getLogger

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from fraud_detection.core.config import settings
from common.log_setting import setup_logger


# ========== Log setting ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")

# ========== Database connection ==========
def get_db_engine() -> Engine:
    """
    Generate SQLAlchemy DB engine and return it.

    Returns:
        Engine: SQLAlchemy DB engine
    """
    
    try:
        engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping = True
            # Set true to test DB connection when getting connectuon from pool
            )
        return engine
    
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


if __name__ == "__main__":
    logger.debug("----- DB Connection Test -----")
    try:
        db_engine = get_db_engine()
        with db_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.scalar() == 1:
                logger.debug("DB connection success!")
            else:
                logger.error("DB connection failed! Did not recieved 1.")
    except OperationalError as e:
        logger.error("DB connection failed. Please check the following:")
        logger.error("\t- Is the database container running? (docker-compose ps)")
        logger.error("\t- Are the .env settings correct?")
        logger.error(f"  - Original error: {e}")
    except Exception as e:
        logger.eror(f"Unexpected error occurred: {e}")