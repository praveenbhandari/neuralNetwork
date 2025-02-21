import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import struct
import random
import math
import os
import json
import shutil
import sys
from pathlib import Path
import shutil
from argparse import ArgumentParser

from tabulate import tabulate
from array import array
from os.path import join
# from google.colab import files
# from google.colab import drive
from zipfile import ZipFile
import zipfile
# from google.colab import files
from io import StringIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from src.utils import (
    load_config,
    validate_config,
    get_model,
    get_optimizer,
    get_loss,
    get_function,
    generate_data,
    get_args
)
from tqdm import tqdm
from src.utils.logger import setup_logger

logger=setup_logger()
# def setup_logger(log_file_path):
    # logger = logging.getLogger()  # Get the root logger
    # logger.setLevel(logging.DEBUG)

    # # Create file handler for logging to a text file
    # file_handler = logging.FileHandler(log_file_path)
    # file_handler.setLevel(logging.DEBUG)

    # # Create console handler for logging to console
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)

    # # Create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # # Add handlers to the logger
    # logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    # return logger
def main(config_path,project_name):
    """
    Main function that will have training loop, used for validating our custom NN structure.

    Args:
        - config_path (str): Path to where our config for this run is stored
        - output_dir (str): Path to output directory where we store outputs for this run

    Returns:

    """
    # Loads / validates training config file that contains:
    # - Model Name (predefined in Python)
    # - Function definition
    # - Num epochs
    # - Number of training data points 
    # - Optimizer Params (Bias + Tau learning rates)
    # -------------
    # - Train/Validation/Test split (optional, default: (70/20/10))
    # - Number of Epochs per snapshot (optional, default: 1)
    # - Apply Normalization flag (optional, default:  True) 
    # - Random Seed (optional)
    # -------------
    # Load and make sure we have a valid config
    project_path = Path(project_name)
    # project_path.mkdir(parents=True, exist_ok=True)  # Ensure the project directory is created
    # output_dir = project_path / output_dir
    
    config = load_config(config_path)
    # TODO: validate_config(config)
    # os.mkdir(proje)
    # Create output directory
    if Path(project_path).exists():
        if config['overwrite_output_dir']:
            logger.info(f'NOTE - overwrite_output_dir is set to True (change this in config if desired), so deleting existing output directory...')
            print(f'NOTE - overwrite_output_dir is set to True (change this in config if desired), so deleting existing output directory...')
            shutil.rmtree(project_path)
        else:
            logger.info('ERROR - Output Directory exists')
            
            raise FileExistsError('ERROR - Output Directory exists')
    output_dir = Path(project_path)
    output_dir.mkdir(parents=True)


    # Dynamically retrieve class pointer to model
    model_cls = get_model(config)
    model = model_cls.build(1, 1)

    # Dynamically load function
    original_func = get_function(config)

    # Dynamically get loss function
    loss_fn = get_loss(config)

    # Dynamically grab optimizer and initialize it with our instantiated model
    optimizer = get_optimizer(config, model)

    # Generate training data and parse out some training config
    x_train, y_train, x_val, y_val, x_test, y_test = generate_data(original_func, config['num_points'], config['train_test_split'], config['apply_norm'])
    samples = len(x_train)
    batch_size = config['batch_size']

    # Create some variables to keep track of during training loop
    train_losses = []
    val_losses = []


    # Training Loop
    indices = np.arange(samples)
    # for this_epoch_index in range(config['num_epochs']):
    for this_epoch_index in tqdm(range(config['num_epochs']), desc=f"Epoch Progress",position=0):
        # Shuffle indicies for mini-batch gradient descent
        np.random.shuffle(indices)
        this_x_train = x_train[indices]
        this_y_train = y_train[indices]
        #TODO: look into random state to saving epoch order of data

        epoch_train_loss = 0.0
        num_batches = 0


        # Create and iterate over batches
        for i in range(0, samples, batch_size):
        # for i in tqdm(range(0, samples, batch_size), desc=f"Epoch {this_epoch_index+1}/{config['num_epochs']}",position=1):
            this_x_batch = this_x_train[i:i+batch_size]
            this_y_batch = this_y_train[i:i+batch_size]
            # TODO: LOOK HOW TO HANDLE LOSSES ACROSS DIFFERENT BATCHES (for tracking)

            # Get output of this model
            this_y_batch_pred = model.forward(this_x_batch) 

            # Get loss
            this_batch_loss = loss_fn(this_y_batch, this_y_batch_pred)
            # print(f'this batch loss -> {this_batch_loss}')

            # Accumulate for epoch-level train loss
            epoch_train_loss += this_batch_loss
            num_batches += 1

            # TODO: HANDLE VALIDATION LOSS (also figure out where this goes in the training loop)
            
            
            # Get gradients
            loss_grad = loss_fn.backward()
            grads = model.backward(loss_grad)

            # Update weights
            optimizer.step()
        
        # Calculate epoch loss
        epoch_train_loss /= num_batches
        train_losses.append(epoch_train_loss)

        # Validation loss
        y_val_pred = model.forward(x_val)
        val_loss = loss_fn(y_val, y_val_pred)
        val_losses.append(val_loss)

        # Store current loss outputs for debugging
        # print(f'EPOCH {this_epoch_index:03f} : loss - {this_loss} | val_loss - {this_val_loss}')
        logger.info(f"Epoch {this_epoch_index+1}/{config['num_epochs']} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Store a snapshot if specified by num_epochs_per_snapshot parameter
        if this_epoch_index % config['num_epochs_per_snapshot'] == 0:
            # Calculate test loss
            y_test_pred = model.forward(x_test)
            this_test_loss = loss_fn(y_test, y_test_pred)
            logger.info(f"Test Loss at Epoch {this_epoch_index+1}: {this_test_loss:.4f}")

            # Save snapshot
            # TODO: save_snapshot(output_dir, this_epoch_index, model, this_loss, this_val_loss, this_test_loss)
            pass

    # Save final model weights + other information to disk
    # TODO: final_test_loss = calculate_loss(model, x_test, y_test)
    # TODO: save_final_results(output_dir, config_path, model, losses, val_losses, final_test_loss)
    logger.info("Training completed!")
    logger.info(f"Final Train Losses: {train_losses[-1]}")
    logger.info(f"Final Validation Losses: {val_losses[-1]}")

    # print(train_losses)
    # print(val_losses)
    

# def old_main(base_path, custom_run=False):
#     print("Code has started execution.")

#     if not Path(base_path).exists():
#         Path(base_path).mkdir(exist_ok=True)

#     clear_folders(base_path)  # Clear previous data
#     create_new = True #by default always create new dataset when running the program for the 1st run


#     # IF CUSTOM RUN, WE ASK FOR INPUTS HERE
#     if custom_run:
#         debug_mode = input("Do you want to enter debug mode? (yes/no): ").strip().lower()
#         upload_mode = input("Do you want to upload existing run folder? (yes/no): ").strip().lower()
#         if upload_mode == 'yes':
#             uploaded = handle_file_upload()
            
#             #   # Step 2: Upload the compressed folder (.zip file)
#             #   print("Please upload a .zip file containing the folder.")
#             #   uploaded = files.upload()  # Prompts user to upload files

#             # Assuming only one file is uploaded, get its name
#             zip_filename = list(uploaded.keys())[0]

#             # Step 3: Extract the .zip to /content/DATA/
#             #   extract_path = '/content/DATA/'
#             extract_path = get_base_path()
#             os.makedirs(extract_path, exist_ok=True)  # Ensure the destination directory exists

#             with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
#             # Extract the entire folder (e.g., 'run_3') to '/content/DATA/run_3/'
#                 zip_ref.extractall(extract_path)
#                 print(f"Folder '{zip_filename}' has been extracted to '{extract_path}'.")

#                 # Get the list of folders inside the extracted directory
#                 extracted_folders = [f for f in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, f))]

#                 # Assuming only one folder exists (e.g., 'run_3'), get the folder name
#                 if len(extracted_folders) == 1:
#                     extracted_folder = extracted_folders[0]
#                     print(f"Detected extracted folder: {extracted_folder}")

#                     # Locate parameters.json file inside the extracted folder (e.g., 'run_3')
#                     parameters_file_path_in_run = os.path.join(extract_path, extracted_folder, 'parameters.json')
#                     # Check for x_train.npy and y_train.npy in the extracted folder
#                     x_train_file_path_in_run = os.path.join(extract_path, extracted_folder,'original_result', 'x_train.npy')
#                     y_train_file_path_in_run = os.path.join(extract_path, extracted_folder,'original_result', 'y_train.npy')


#                     # If parameters.json exists, copy it to /content/DATA/ directory
#                     if os.path.exists(parameters_file_path_in_run):
#                         shutil.copy(parameters_file_path_in_run, os.path.join(extract_path, 'parameters.json'))
#                         print(f"'parameters.json' has been copied to '{extract_path}'.")
#                     else:
#                         print("No 'parameters.json' file found in the extracted folder.")

#                     # If x_train.npy exists, copy it to the common folder
#                     if os.path.exists(x_train_file_path_in_run):
#                         shutil.copy(x_train_file_path_in_run, os.path.join(extract_path, 'x_train.npy'))
#                         print(f"'x_train.npy' has been copied to '{extract_path}'.")
#                     else:
#                         print("No 'x_train.npy' file found in the extracted folder.")

#                     # If y_train.npy exists, copy it to the common folder
#                     if os.path.exists(y_train_file_path_in_run):
#                         shutil.copy(y_train_file_path_in_run, os.path.join(extract_path))
#                         print(f"'y_train.npy' has been copied to '{extract_path}'.")
#                     else:
#                         print("No 'y_train.npy' file found in the extracted folder.")

#                 else:
#                     print("Error: More than one folder was extracted. Unable to locate the run folder.")

#     num_runs = int(input("Enter the number of runs: ").strip())
#     run_summary = []

#     for run_number in range(1, num_runs + 1):
#         print ('run_number: ', run_number)

#         # Create a folder for this run
#         run_folder = create_run_folder(base_path, run_number)
#         print("run_folder created")

#         # Capture the console output
#         # original_stdout = sys.stdout
#         # sys.stdout = StringIO()
        
#         # with RedirectStdout() as output:

#         try:
#             if upload_mode == "yes":
#                 num_training_inputs, lower_bound,max_lower_bound, upper_bound,min_upper_bound, func_expressions, num_layers, num_neurons, output_num_neurons, epochs, bias_learning_rate, tau_learning_rate, bias_values, tau_values, folder_path = get_user_inputs(False, run_number)
#                 x_train, y_train = generate_data(num_neurons[0], output_num_neurons, num_training_inputs, lower_bound, max_lower_bound, upper_bound, min_upper_bound, func_expressions, folder_path, False)
#             else:
#                 num_training_inputs, lower_bound,max_lower_bound, upper_bound,min_upper_bound, func_expressions, num_layers, num_neurons, output_num_neurons, epochs, bias_learning_rate, tau_learning_rate, bias_values, tau_values, folder_path = get_user_inputs(create_new if run_number == 1 else False, run_number)
#                 x_train, y_train,x_train_1 = generate_data(num_neurons[0], output_num_neurons, num_training_inputs, lower_bound, max_lower_bound, upper_bound, min_upper_bound, func_expressions, folder_path, create_new)

#             net = Network()
#             for i in range(num_layers):
#                 if i == num_layers - 1:  # last layer
#                     net.add(FCLayer(num_neurons[i], output_num_neurons, f'Layer {i+1}', bias=bias_values[i], tau=tau_values[i]))
#                 else:
#                     net.add(FCLayer(num_neurons[i], num_neurons[i + 1], f'Layer {i+1}', bias=bias_values[i], tau=tau_values[i]))
#                 net.add(ActivationLayer(ReLU))

#             net.use(mse)

#             print(f"Run number: {run_number}")
#             # Train the network

#             errors, final_bias_values, final_tau_values = net.fit(x_train, y_train, epochs, bias_learning_rate, tau_learning_rate, debug_mode)

#             # Check final error rate and save results in the appropriate folder
#             final_error = errors[-1]
#             print(f'Final Error: {final_error}')  # Debug statement to check the final error

#             rounded_final_error = round(final_error, 2)
#             if rounded_final_error == 0.00:
#                 rounded_final_error = 0
#             run_summary.append((run_number, rounded_final_error))

#             # Capture the console output to save
#             # output_text = sys.stdout.getvalue()
#             output_text = 'a temp fix'
#             # output_text = output.getvalue()

#             # Save results in the run folder
#             save_results(run_folder, run_number, x_train,x_train_1, y_train, bias_values, tau_values, errors, final_bias_values, final_tau_values, output_text, subfolder_name="original_result")

#         # except Exception as e:
#         #     print(f"Error during run {run_number}: {e}")
        
#         finally:
#             # Restore the original stdout
#             print("A temp finally line")
#             # sys.stdout = original_stdout

#         # Save the output to JSON and PDF
#         save_console_output(run_number, output_text, rounded_final_error, run_folder, None)

#     # Print summary
#     print("\nRun Summary:")
#     headers = ["Run", "Final Error"]
#     table = [[run_number, error] for run_number, error in run_summary]
#     print(tabulate(table, headers, tablefmt="grid"))

#     # Store former run summary before rerunning with updated bias
#     former_run_summary = run_summary.copy()

#     while True:
#         rerun_or_update = get_valid_input( "\nDo you want to rerun a run number or update the parameters? (rerun/update/exit): ",
#     validate_rerun_or_update, "Invalid input. Please enter 'rerun', 'update', or 'exit'." )
#         if rerun_or_update == "exit":
#             print("Exiting the program.")
#             break
#         else:
#           manage_rerun_update(rerun_or_update, base_path)
from src.utils.global_var import project_name
if __name__ == "__main__":
    # Call clear_seed at the beginning of the script
    # clear_seed()
    args = get_args()
    # log_file_path = os.path.join(project_name, 'training_log.txt')
    setup_logger(project_name)


    # SWITCH THESE ONCE DONE DEBUGGING
    # main(args.base_path, args.custom_run)
    # main(args.config_path, args.output_dir)
    main(config_path='/Users/praveenbhandari/Desktop/NN/NN/configs/example_config.json',
        #  output_dir='/Users/praveenbhandari/Desktop/NN/out', 
         project_name=project_name)
 