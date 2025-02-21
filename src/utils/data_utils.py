import numpy as np
from sklearn.model_selection import train_test_split as train_test_split_func
from tqdm import tqdm
import os

from src.utils.logger import setup_logger
import pandas as pd
from src.utils.global_var import project_name
logger=setup_logger()
# logger=logging.getLogger()
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

def generate_data(func, num_points, train_test_split, apply_norm):
    # Assert function is wrapped by NumPy, print a lot of mumbo jumbo if not
    
    # TODO: FIGURE OUT LOWER/UPPER BOUNDS FOR THIS
    # TODO: FIGURE OUT HOW TO SUPPORT DIFFERENT SHAPES
    # Generate all X using Numpy
    X = np.random.rand(num_points)

    try:
        func = np.vectorize(func)
        print(f'Successfully vectorized given function.')
    except:
        # print(f'Warning - could not vectorize given function.')
        logger.warning(f'Warning - could not vectorize given function: {e}')

    
    y = func(X)

    # Split train+val from test
    x_train_val, x_test, y_train_val, y_test = train_test_split_func(X, y, test_size=(train_test_split[1] / 100))
    
    # Split train/val from eachother
    x_train, x_val, y_train, y_val = train_test_split_func(x_train_val, y_train_val, test_size=0.1) # SETTING VALIDATION AS 10% of train data

    # Adjust shapes for downstream model consumption
    x_train = x_train.reshape(-1, 1)  # shape (num_train, 1)
    y_train = x_train.reshape(-1, 1)  # shape (num_train, 1)
    x_val   = x_val.reshape(-1, 1)    # shape (num_val, 1)
    y_val   = x_val.reshape(-1, 1)    # shape (num_val, 1)
    x_test  = x_test.reshape(-1, 1)   # shape (num_test, 1)
    y_test  = x_test.reshape(-1, 1)   # shape (num_test, 1)
    
    # print(x_train[0], y_train[0])
    # print(x_val[0], y_val[0])
    # print(x_test[0], y_test[0])
    output_dir = './output/'+project_name+'/data'
    os.makedirs(output_dir, exist_ok=True)

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
    logger.debug(f'Sample x_train: {x_train[0]}, y_train: {y_train[0]}')
    logger.debug(f'Sample x_val: {x_val[0]}, y_val: {y_val[0]}')
    logger.debug(f'Sample x_test: {x_test[0]}, y_test: {y_test[0]}')

    
    return x_train, y_train, x_val, y_val, x_test, y_test
