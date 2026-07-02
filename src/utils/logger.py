import sys
from pathlib import Path
from loguru import logger


def setup_logger(name: str = "skilllens", log_dir: str = "logs") -> "logger":
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        log_path / f"{name}_{{time:YYYY-MM-DD}}.log",
        rotation="500 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level="DEBUG",
    )
    return logger
