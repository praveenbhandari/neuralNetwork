import os
import json

from pathlib import Path
from argparse import ArgumentParser

from src.utils.logger import setup_logger

logger=setup_logger()

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument('--config_path', type=str, help='Path to the input config file')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(), 'DEBUG_DATA'), help='Path to output directory where we want to store results.')
    
    args = parser.parse_args()
    logger.info(f"Parsed arguments: config_path={args.config_path}, output_dir={args.output_dir}")
    

    return args