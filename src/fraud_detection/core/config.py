"""
Load configuration from .env file and
check types of variables with pydantic
"""
import os
from logging import getLogger
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


from common.log_setting import setup_logger


# ========== Setup logger ==========
logger = getLogger(__name__)
logger = setup_logger(logger, level="DEBUG")


# ========== Settings ==========
class Settings(BaseSettings):
    """
    Load env file and check types.
    Inhereted from pydantic_settings.BaseSettings
    """
    
    # DB connection info
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    POSTGRES_PORT: int = 5432
    POSTGRES_SERVER: str = "db"

    # Define as a property and you can use it as an attribute
    # e.g. settings.DATABASE_URL    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    

settings = Settings()

if __name__ == "__main__":
    logger.debug("----- Setting Test -----")
    logger.debug(f"PostgreSQL User: {settings.POSTGRES_USER}")
    logger.debug(f"PostgreSQL URL: {settings.DATABASE_URL}")
    logger.debug("------------------------")