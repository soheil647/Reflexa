# modules/utils/logger.py
import logging
import os
import sys
from config import LOGS_DIR  # ensure this is defined in your config.py


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Ensure the logs directory exists
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)

        # Create a file handler with UTF-8 encoding
        fh = logging.FileHandler(os.path.join(LOGS_DIR, "reflexa.log"), encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # Reconfigure stdout to use UTF-8 if possible
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding='utf-8')

        # Create a console handler (StreamHandler) using sys.stdout
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
