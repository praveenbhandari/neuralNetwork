"""
Utility script used in train.py

"""

import os
import json

from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from sklearn.model_selection import train_test_split as train_test_split_func

from models import KeonNetwork
from functions import sin_function
import pandas as pd

from src.utils.logger import setup_logger

logger=setup_logger()
# logger = logging.getLogger('train_logger')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument('--config_path', type=str, help='Path to the input config file')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(), 'DEBUG_DATA'), help='Path to output directory where we want to store results.')

    args = parser.parse_args()

    logger.debug(f"Arguments received: {args}")
    return args

def parse_config(config_path):
    # Check config exists
    if not Path(config_path).exists():
        logger.error(f'Could not find config at specified path - {config_path}')
        
        raise FileNotFoundError(f'Could not find config at specified path - {config_path}')
    
    with open(config_path, 'r') as f:
        config = json.load(f)

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
            logger.error(f'ERROR - Necessary key {i} not present in input config.')
            raise Exception(f'ERROR - Necessary key {i} not present in input config.')
        params[i] = config[i]
    
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
        else:
            params[i] = config[i]
    logger.debug(f"Parsed configuration: {params}")
    return params

def retrieve_model(config):
    """
    Retrieves a class to instantiate based off of a model name. For example if passed in 'KeonModel', 
    it will return the KeonModel class defined in models/KeonModel.py. The models available should be 
    first created in the models/ directory, added to that directory's __init__.py, and then added to the ModelImplementationDict below to 
    make them accessible to the training loop.

    args:
        - model_name (str): String of which model to retrieve
    
    Returns:
        - class: Class of the model we chose (needs to be instantiated later on)
    """
    from src.models import registry

    model_name = config['model']

    if model_name not in registry:
        raise Exception(f'ERROR - Model name that was passed in \'{model_name}\' is not supported. Supported models are - {registry.keys()}')
    
    logger.debug(f"Model {model_name} retrieved successfully.")
    return registry[model_name]
    
def retrieve_func(config):
    """
    Retrieves a function that we will use to generate data for our model to train on. The functions available should be 
    first created in the functions/ directory, added to that directory's __init__.py, and then added to the dictionary below
    make them accessible to the training loop.

    args:
        - function_name (str): String of which function to retrieve
    
    Returns:
        - class: Class of the function we chose (needs to be instantiated later on)
    """
    from src.functions import registry

    function_name = config['function']

    if function_name not in registry:
        logger.error(f'ERROR - Function name that was passed in \'{function_name}\' is not supported. Supported functions are - {registry.keys()}')
        raise Exception(f'ERROR - Function name that was passed in \'{function_name}\' is not supported. Supported functions are - {registry.keys()}')
    
    logger.debug(f"Function {function_name} retrieved successfully.")
    return registry[function_name]
    
# TODO: x_train, y_train, x_val, y_val, x_test, y_test = generate_data(original_func, params['num_points'], params['train_val_test_split'], params['apply_normalization'])
def generate_data(func, num_points, train_test_split, apply_norm):
    # Assert function is wrapped by NumPy, print a lot of mumbo jumbo if not

    logger.debug(f"Generating data with function {func}, num_points={num_points}, and split={train_test_split}")
    # TODO: FIGURE OUT LOWER/UPPER BOUNDS FOR THIS
    # TODO: FIGURE OUT HOW TO SUPPORT DIFFERENT SHAPES
    # Generate all X using Numpy
    X = np.random.rand(num_points)

    try:
        func = np.vectorize(func)
        # print(f'Successfully vectorized given function.')
        logger.debug(f'Successfully vectorized given function.')
    except:
        # print(f'Warning - could not vectorize given function.')
        logger.warning(f'Warning - could not vectorize given function.')
    
    y = func(X)

    # Split train+val from test
    x_train_val, x_test, y_train_val, y_test = train_test_split_func(X, y, test_size=(train_test_split[2] / 100))
    
    # Split train/val from eachother
    x_train, x_val, y_train, y_val = train_test_split_func(x_train_val, y_train_val, test_size=(train_test_split[1] / 100))

    # y_train = func(x_train)
    # y_val = func(x_val)
    # y_test = func(x_test)
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    # train_data = pd.DataFrame({'x_train': x_train, 'y_train': y_train})
    # val_data = pd.DataFrame({'x_val': x_val, 'y_val': y_val})
    # test_data = pd.DataFrame({'x_test': x_test, 'y_test': y_test})

    pd.DataFrame(x_train).to_csv(os.path.join(output_dir, 'x_train.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    pd.DataFrame(x_val).to_csv(os.path.join(output_dir, 'x_val.csv'), index=False)
    pd.DataFrame(y_val).to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    pd.DataFrame(x_test).to_csv(os.path.join(output_dir, 'x_test.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    

    #TODO : save this files in the folder and then move this files to project_name_folder

    # Save data as Numpy files
    np.save(os.path.join(output_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'x_val.npy'), x_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    logger.debug(f"Data generation completed. Sample data:")
    # print(x_train[0], y_train[0])
    # print(x_val[0], y_val[0])
    # print(x_test[0], y_test[0])

    
    return x_train, y_train, x_val, y_val, x_test, y_test
