import logging
import os
from src.utils.global_var import project_name

def setup_logger(project_name=project_name, log_filename='training_log.txt'):
    
    log_file_path = os.path.join('./output', project_name, './log', log_filename)
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
