import os
import json

from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.utils.global_var import project_name
logger=setup_logger()

def load_config(config_path):

    logger.debug(f'Loading configuration from {config_path}')
    
    if not Path(config_path).exists():
        logger.error(f'Config file not found at: {config_path}')
        raise FileNotFoundError(f'Could not find config at specified path - {config_path}')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        # logger.debug(f'Config loaded: {config}')
        logger.debug(f"Config loaded:\n{json.dumps(config, indent=4)}")
        
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = Path('./output') / project_name
    os.makedirs(output_dir, exist_ok=True)
    
    config_copy_path = output_dir / 'config.json'
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=4)
        logger.debug(f'Config saved at {config_copy_path}')


    return config

def validate_config(config):
    params = {}

    # Define necessary / optional keys
    necessary_keys = [
        "model_name", 
        "function",
        "num_epochs",
        "num_points",
        "tau_lr", 
        "bias_lr"
    ]

    optional_keys = [
        "train_val_test_split",
        "num_epochs_per_snapshot",
        "apply_norm",
        "random_seed",
        "overwrite_output_dir"
    ]

    # Add necessary keys
    for i in necessary_keys:
        if i not in config:
            logger.error(f'Necessary key {i} not present in input config.')
            
            raise Exception(f'ERROR - Necessary key {i} not present in input config.')
        params[i] = config[i]
        # logger.debug(f'Key {i} found with value: {config[i]}')
    
    
    # Define default values for optional keys
    default_vals = {
        "train_val_test_split": [80,20,10],
        "num_epochs_per_snapshot": 1,
        "apply_norm": True,
        "random_seed": None,
        "overwrite_output_dir": True
    }

    for i in optional_keys:
        if i not in config:
            params[i] = default_vals[i]
            logger.debug(f'Optional key {i} not found. Using default value: {default_vals[i]}')
        
        else:
            params[i] = config[i]
            # logger.debug(f'Optional key {i} found with value: {config[i]}')
    
    logger.debug('Configuration validated successfully.')
    
    return params
    