import logging
import os


def setup_logger(log_file_path='./logs/output_log.txt'):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
