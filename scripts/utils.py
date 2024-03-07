import logging
import os
import sys
from datetime import datetime


def set_logger(name: str):
    log_dir = "logs"
    log_file_name = f'{name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, log_file_name)),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(sys.argv)
    return logger
