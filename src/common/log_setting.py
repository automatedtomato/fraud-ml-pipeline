import logging
from logging import Formatter, StreamHandler, handlers
from pathlib import Path

# ========== Constants ==========
FORMAT = "%(levelname)-8s %(asctime)s - [%(filename)s:%(lineno)d] %(message)s"


def setup_logger(logger: logging.Logger, level: str | None = None) -> logging.Logger:

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if level is None:
        level = "INFO"

    if level.upper() not in level_map:
        raise ValueError(f"Invalid log level: {level}")

    logger.setLevel(level_map[level.upper()])

    st_handler = StreamHandler()

    file_path = Path(__file__).parents[2] / "./logs/app.log"

    fl_handler = handlers.RotatingFileHandler(
        filename=Path(__file__).parents[2] / file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    st_handler.setFormatter(Formatter(FORMAT))
    fl_handler.setFormatter(Formatter(FORMAT))

    logger.addHandler(st_handler)
    logger.addHandler(fl_handler)

    return logger
