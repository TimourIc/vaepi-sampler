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


def number_of_model_params(model):
    """custom function to get number of params in a torch model"""

    model_params = 0
    for p in list(model.parameters()):
        n = 1
        for s in list(p.size()):
            n = n * s
        model_params += n
    return model_params
