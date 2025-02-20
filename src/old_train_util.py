import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import struct
import random
import math
import os
import json
import shutil
import sys
import sympy as sp

from tabulate import tabulate
from array import array
from os.path import join
# from google.colab import files
# from google.colab import drive
import zipfile
from zipfile import ZipFile
from io import StringIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from core.network import Network
from core.fc_layer import FCLayer
from core.activation_layers import ActivationLayer
from core.neuron_util import ReLU, mse

# Function to parse args used in main()
def get_args():
    parser = ArgumentParser()
    
    parser.add_argument('--custom_run', action='store_true', help='Include this flag to ask for user input.')
    parser.add_argument('--base_path', type=str, default=os.path.join(os.getcwd(), 'DEBUG_DATA'), help='Include this flag to ask for user input.')

    args = parser.parse_args()

    return args

# Configuration for base paths 
def get_base_path():
    # Use platform detection if needed
    if 'google.colab' in sys.modules:
        return '/content/DATA/'  # Colab path
    else:
        return os.path.join(os.getcwd(), 'DEBUG_DATA')  # Local path
    
#The below function is a replacement for the direct code logic for uploading zip folders
def handle_file_upload():
    if 'google.colab' in sys.modules:
        from google.colab import files
        from google.colab import drive
        uploaded = files.upload()
        zip_filename = list(uploaded.keys())[0]
    else:
        zip_filename = input("Enter the path to your .zip file: ").strip()
        if not os.path.exists(zip_filename):
            raise FileNotFoundError("The specified file does not exist.")
    return zip_filename

    
def clear_seed():
    np.random.seed(None)
    random.seed(None)
    os.environ.pop('PYTHONHASHSEED', None)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def reset_seed():
    np.random.seed(None)
    random.seed(None)
    os.environ.pop('PYTHONHASHSEED', None)

# Context manager for selective redirection
class RedirectStdout:
    def __init__(self):
        self.original_stdout = sys.stdout
        self.string_io = StringIO()

    def __enter__(self):
        sys.stdout = self.string_io
        return self.string_io

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout

#This code handles the folders creation and saving results to review after the run

def create_run_folder(base_folder, run_number): # Create a folder for each run
    run_folder = os.path.join(base_folder, f"run_{run_number}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def save_results(folder_path, run_number, x_train,x_train_1, y_train, bias_values, tau_values, errors, final_bias_values, final_tau_values, output_text, subfolder_name=None):
    """
    Save the results under the specified folder path.
    If subfolder_name is None, files will be saved directly under the folder path.
    """
    # If subfolder_name is provided, create it under the folder path; otherwise, use the main folder path
    if subfolder_name:
        subfolder = os.path.join(folder_path, subfolder_name)
    else:
        subfolder = folder_path

    os.makedirs(subfolder, exist_ok=True)

    # Save x_train and y_train as .npy
    np.save(os.path.join(subfolder, "x_train.npy"), x_train)
    np.save(os.path.join(subfolder, "y_train.npy"), y_train)

     # Save x_train and y_train as CSV files
    pd.DataFrame(x_train).to_csv(os.path.join(subfolder, "x_train.csv"), index=False)
    pd.DataFrame(x_train_1).to_csv(os.path.join(subfolder, "x_train_1.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(subfolder, "y_train.csv"), index=False)

    # Save x_train and y_train as .json
    pd.DataFrame(x_train).to_json(os.path.join(subfolder, "x_train.json"))
    pd.DataFrame(y_train).to_json(os.path.join(subfolder, "y_train.json"))

    # Save bias_values and tau_values as .json
    with open(os.path.join(subfolder, "bias_values.json"), "w") as f:
        json.dump([b.tolist() for b in bias_values], f, indent=4)
    with open(os.path.join(subfolder, "tau_values.json"), "w") as f:
        json.dump([t.tolist() for t in tau_values], f, indent=4)

    # Save final bias_values and tau_values as .json
    with open(os.path.join(subfolder, "final_bias_values.json"), "w") as f:
        json.dump([b.tolist() for b in final_bias_values], f, indent=4)
    with open(os.path.join(subfolder, "final_tau_values.json"), "w") as f:
        json.dump([t.tolist() for t in final_tau_values], f, indent=4)

    # Save the errors array
    errors_file_path = os.path.join(subfolder, "errors.npy")
    np.save(errors_file_path, errors)
    with open(os.path.join(subfolder, "errors.json"), "w") as f:
        json.dump([b.tolist() for b in errors], f, indent=4)
    print(f"Errors saved to: {errors_file_path}")

    # Save the error plot
    plt.figure()
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Error')
    plt.title('Error Rate Across Epochs')
    plt.savefig(os.path.join(subfolder, "error_plot.png"))
    plt.close()

    # Save console output as .json
    output_file_json = os.path.join(subfolder, "output.json")
    with open(output_file_json, "w") as f:
        json.dump({"output": output_text}, f, indent=4)

    # Save console output as .pdf
    output_file_pdf = os.path.join(subfolder, "output.pdf")
    save_output_as_pdf(output_text, os.path.join(subfolder, "error_plot.png"), output_file_pdf, run_number)

    # Copy parameters.json file to the run's subfolder
    # original_parameters_path = "/content/DATA/parameters.json"
    # parameters_destination_path = os.path.join(folder_path, "parameters.json")
    # shutil.copyfile(original_parameters_path, parameters_destination_path)

    # Adjust + copy path for the parameters.json file
    if folder_path != subfolder:
        original_parameters_path = os.path.join(folder_path, "parameters.json")  # original params path
        parameters_destination_path = os.path.join(subfolder, "parameters.json")

        if os.path.exists(original_parameters_path):
            shutil.copyfile(original_parameters_path, parameters_destination_path)
            print(f"parameters.json copied to: {parameters_destination_path}")
        else:
            print("Error: parameters.json file not found.")

def save_output_as_pdf(output_text, plot_file, file_path, run_number):
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    # Add a big title with the run number
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, f"Run Number: {run_number}")

    # Add the console output text
    c.setFont("Helvetica", 10)
    x_margin = 50
    y_margin = height - 100
    line_height = 12

    lines = output_text.splitlines()
    y_position = y_margin
    for line in lines:
        if y_position < 50:  # start a new page if there's no more space
            c.showPage()
            y_position = height - 50
        c.drawString(x_margin, y_position, line)
        y_position -= line_height

    # Add the error plot to the PDF
    if os.path.exists(plot_file):
        c.showPage()  # Start a new page for the plot
        c.drawImage(ImageReader(plot_file), 100, 400, width=400, height=300)

    c.save()

def save_console_output(run_number, output_text, final_error, run_folder, is_bias_update=False):
    # Determine the subfolder based on whether it's a bias update or not
    subfolder_name = "bias_update_result" if is_bias_update else "original_result"
    run_folder_path = os.path.join(run_folder, subfolder_name)

    # Ensure the run_folder exists
    os.makedirs(run_folder_path, exist_ok=True)

    # Save console output as JSON
    output_file_json = os.path.join(run_folder_path, "output.json")
    with open(output_file_json, "w") as f:
        json.dump({"output": output_text}, f, indent=4)

    # Define the path to the existing error plot (if any)
    plot_file = os.path.join(run_folder_path, "error_plot.png")

    # Save console output as PDF
    output_file_pdf = os.path.join(run_folder_path, "output.pdf")
    save_output_as_pdf(output_text, plot_file, output_file_pdf, run_number)

#Loads the particular data for specific re-run

def load_run_data(run_folder, base_folder):
    parameters_file_path = os.path.join(base_folder, "parameters.json")
    with open(parameters_file_path, "r") as file:
        parameters = json.load(file)

    num_training_inputs = parameters['num_training_inputs']
    lower_bound = parameters['lower_bound']
    max_lower_bound = parameters['max_lower_bound']
    upper_bound = parameters['upper_bound']
    min_upper_bound = parameters['min_upper_bound']
    func_expressions = parameters['func_expressions']
    num_layers = parameters['num_layers']
    num_neurons = parameters['num_neurons']
    epochs = parameters['epochs']
    bias_learning_rate = parameters['bias_learning_rate']
    tau_learning_rate = parameters['tau_learning_rate']
    # tau_bouncing_mode = parameters['tau_bouncing_mode']
    output_num_neurons =  parameters['output_num_neurons']
    num_input_neurons =  parameters['num_input_neurons']

    # Load x_train and y_train
    x_train = np.load(os.path.join(run_folder, "x_train.npy"))
    y_train = np.load(os.path.join(run_folder, "y_train.npy"))

    # Load bias_values and tau_values
    with open(os.path.join(run_folder, "bias_values.json"), "r") as f:
        bias_values = [np.array(b) for b in json.load(f)]
    with open(os.path.join(run_folder, "tau_values.json"), "r") as f:
        tau_values = [np.array(t) for t in json.load(f)]

    return num_training_inputs, lower_bound, max_lower_bound, upper_bound, min_upper_bound, func_expressions, num_layers, num_neurons, num_input_neurons, output_num_neurons, epochs, bias_learning_rate, tau_learning_rate, bias_values, tau_values, x_train, y_train


def rerun_specific_run(run_number, base_folder):
    set_seed(42)  # Set the seed only for reruns

    # Define the folder where the run data is stored
    run_folder = os.path.join(base_folder, f"run_{run_number}", 'original_result')

    if not os.path.exists(run_folder):
        print(f"Run folder for run number {run_number} does not exist.")
        return

    print(f"Rerunning run number: {run_number} from folder: {run_folder}")

    # Load data from the run folder
    num_training_inputs, lower_bound, max_lower_bound, upper_bound,min_upper_bound, func_expressions, num_layers, num_neurons, num_input_neurons, output_num_neurons, epochs, bias_learning_rate, tau_learning_rate, bias_values, tau_values, x_train, y_train = load_run_data(run_folder, base_folder)

    net = Network()
    output_num_neurons = num_neurons[-1] #added by KW - present in the original code, not sure the imapct of this change
    for i in range(num_layers):
        if i == num_layers - 1:  # last layer
            net.add(FCLayer(num_neurons[i], output_num_neurons, f'Layer {i+1}', bias=bias_values[i], tau=tau_values[i]))
        else:
            net.add(FCLayer(num_neurons[i], num_neurons[i + 1], f'Layer {i+1}', bias=bias_values[i], tau=tau_values[i]))
        net.add(ActivationLayer(ReLU))

    net.use(mse)

    # No capturing of console output, just printing everything to the console - not saving the results
    errors, final_bias_values, final_tau_values = net.fit(x_train, y_train, epochs, bias_learning_rate, tau_learning_rate)

    final_error = errors[-1]
    print(f"Re-Run of run number {run_number} completed with final error: {final_error}")

    reset_seed()

# function to check if the last two epochs' error difference is constant
def is_error_stable(errors):
    # Set a tolerance level for floating-point comparisons
    tolerance = 1e-10

    # Ensure we have at least three error values to compare
    if len(errors) < 3:
        return False  # Not enough data to determine consistency

    # Calculate differences between consecutive error values
    print('error 1 :',errors[-1])
    print('error 2 :',errors[-2])
    print('error 3 :',errors[-3])
    diffs = np.diff(errors[-3:])

    # Case of stagnation: All differences are zero
    if np.all(np.abs(diffs) < tolerance):
      print('Error Status: Stagnant')
      return True

    return False

"""
    Update parameters for a specific run and rerun the process with the new values.

    This function fetches the original data for the specified run number, updates parameters on the provided
    `new_value` flags, and then reruns the neural network training process with these updated parameters.

    Args:
        run_number (int): The number of the run to update.
        base_folder (str): The base directory where run data is stored.
        new_value (list of tuples): A list containing tuples that represent whether a parameter should be updated
                                     (1) or not (0), along with the new value for the parameter if it is to be updated.
                                     Format: [(bias_update_flag, new_bias_learning_rate),
                                              (tau_update_flag, new_tau_learning_rate),
                                              (taubounce_update_flag),
                                              (epochs_update_flag, new_epochs),
                                              (dataset_update_flag, new_dataset)].

    Returns:
        tuple: A tuple containing the run number and the rounded final error after rerunning the training.

    Raises:
        FileNotFoundError: If the original run data folder does not exist.

    Example:
        run_number = 2
        base_folder = "/path/to/data"
        new_value = [(1, 0.01), (0, None), (1, 'yes'), (1, 100), (0, None)]
        result = update_rerun_parameter(run_number, base_folder, new_value)
    """
# def update_rerun_parameter(run_number, base_folder, param_to_update, new_value):
def update_rerun_parameter(run_number, base_folder, new_value):

    run_folder = os.path.join(base_folder, f"run_{run_number}", 'original_result')
    if os.path.exists(run_folder):
       # Load the original data and parameters
      num_training_inputs, lower_bound, max_lower_bound, upper_bound, min_upper_bound, func_expressions, num_layers, num_neurons, num_input_neurons, output_num_neurons, epochs, bias_learning_rate, tau_learning_rate, bias_values, tau_values, x_train, y_train = load_run_data(run_folder, base_folder)
      # Initialize debug_mode as 'no' by default
      debug_mode = 'no'

       # Create the folder structure for parameter updates
      # parameter_update_folder = os.path.join(base_folder, f"run_{run_number}", "parameter_update_result", f"{param_to_update}_result")
      parameter_update_folder = os.path.join(base_folder, f"run_{run_number}", "parameter_update_result")
      os.makedirs(parameter_update_folder, exist_ok=True)

      # Capture the console output
      original_stdout = sys.stdout
      sys.stdout = StringIO()

       # Update the specific parameter
      # if param_to_update == 'tau':
      #   tau_learning_rate = new_value
      # elif param_to_update == 'bias':
      #   bias_learning_rate = new_value
      # elif param_to_update == 'epochs':
      #   epochs = int(new_value)
      # elif param_to_update == 'taubounce':
      #   tau_bouncing_mode = new_value

      # Update the specific parameters based on the flags in new_value
      if new_value[0][0] == 1:  # Update bias learning rate
            bias_learning_rate = new_value[0][1]
      if new_value[1][0] == 1:  # Update tau learning rate
            tau_learning_rate = new_value[1][1]
      # if new_value[2][0] == 1:  # Update tau bouncing mode
      #       tau_bouncing_mode = new_value[2][1]
      if new_value[2][0] == 1:  # Update epochs
            epochs = int(new_value[2][1])
      if new_value[3][0] == 1:  # Update dataset
            # x_train, y_train = generate_data(num_neurons[0], "/content/DATA", True)
            x_train, y_train = generate_data(num_neurons[0], output_num_neurons, num_training_inputs, lower_bound, max_lower_bound, upper_bound, min_upper_bound, func_expressions, base_folder, True)
      if new_value[4][0] == 1:  # Enter the debug mode
            debug_mode = new_value[4][1]

      try:
        # Rerun the process for this specific run with the new parameters' value
        net = Network()
        for i in range(num_layers):
          if i == num_layers - 1:  # last layer
            net.add(FCLayer(num_neurons[i], output_num_neurons, f'Layer {i}', bias=bias_values[i], tau=tau_values[i]))
          else:
            net.add(FCLayer(num_neurons[i], num_neurons[i + 1], f'Layer {i}', bias=bias_values[i], tau=tau_values[i]))
          net.add(ActivationLayer(ReLU))

        net.use(mse)
        errors, final_bias_values, final_tau_values = net.fit(x_train, y_train, epochs, bias_learning_rate, tau_learning_rate, debug_mode)

        final_error = errors[-1]
        rounded_final_error = round(final_error, 2)

        # Save results directly in the update folder without the original_result subfolder
        save_results(parameter_update_folder, run_number, x_train,[], y_train, bias_values, tau_values, errors, final_bias_values, final_tau_values, sys.stdout.getvalue())

      finally:
                # Restore the original stdout
                sys.stdout = original_stdout

    else:
            print(f"Run folder for run number {run_number} does not exist.")

    # return new_run_summary
    return (run_number, rounded_final_error)

"""
    Manage the rerunning or updating of neural network training parameters based on user input.

    This function prompts the user to decide whether to rerun an existing training run or update the parameters
    for all runs. It collects new parameter values if necessary, calls the `update_rerun_parameter` function to
    perform the rerun or update, and displays a comparison of errors before and after the update.

    Args:
        rerun_or_update (str): A string indicating the operation to perform; either 'rerun' or 'update'.
        base_folder (str): The base directory where the run data is stored.

    Returns:
        None: This function does not return a value but prints the results and comparison tables to the console.

    Raises:
        ValueError: If the user inputs an invalid run number or if binary input for updates is not valid.

    Example:
        manage_rerun_update("update", "/path/to/data")
    """

def manage_rerun_update(rerun_or_update, base_folder):

    new_run_summary = []
    # Initialize new_value to have 6 elements as (0, None)
    new_value = [(0, None) for _ in range(6)]

    if rerun_or_update == "rerun":
      rerun_number = int(get_valid_input( f"Enter the run number to update the parameters for (between 1 to {num_runs}): ", lambda x: validate_run_number(x, num_runs),
        f"Please enter a run number." ))
    update_type = get_valid_input(
    "Do you want to update the bias learning rate, tau learning rate, epochs, change the dataset, or enter debug mode? Enter binary digits (e.g., 01001) 0 = no, 1 = yes: ",
    validate_binary_string, "Invalid input. Please enter a 5-digit binary string like '01001'." )

    # Parse the binary flags (e.g., '0110' -> bias: 0, tau: 1, taubounce: 1, epochs: 0)
    for i, flag in enumerate(update_type):
      if flag == '1':
        if i == 0:
            new_bias_learning_rate = get_valid_input("Enter new value for bias learning rate: ", is_valid_number, "Invalid input.")
            new_value[i] = (1, float(new_bias_learning_rate))  # Assign to the correct index
        elif i == 1:
            new_tau_learning_rate = get_valid_input("Enter new value for tau learning rate: ", is_valid_number, "Invalid input.")
            new_value[i] = (1, float(new_tau_learning_rate))  # Assign to the correct index
        # elif i == 2:
        #     new_tau_bouncing_mode = get_valid_input("Tau Bouncing Mode (yes/no): ", validate_yes_no, "Enter either 'yes' or 'no'")
        #     new_value[i] = (1, new_tau_bouncing_mode)  # Assign to the correct index
        elif i == 2:
            new_epochs = get_valid_input("Enter new value for epochs: ", is_valid_number, "Invalid input.")
            new_value[i] = (1, int(new_epochs))  # Assign to the correct index
        elif i == 3:
            new_dataset = get_valid_input("Do you want to generate a new dataset? (yes/no): ", validate_yes_no, "Enter either 'yes' or 'no'")
            new_value[i] = (1, new_dataset)  # Assign to the correct index
        elif i == 4:
            new_debug = get_valid_input("Do you want to enter debug mode (yes/no): ", validate_yes_no, "Enter either 'yes' or 'no'")
            new_value[i] = (1, new_debug)  # Assign to the correct index

    if rerun_or_update == "update":
      for run_number in range(1, num_runs + 1):
        # run_summary_tuple = update_rerun_parameter(run_number, base_folder, param_to_update, new_value)
        run_summary_tuple = update_rerun_parameter(run_number, base_folder, new_value)
        new_run_summary.append(run_summary_tuple)

      print("Update completed with updated parameters.")

      # Generate comparison table
      headers = ["Former Run Number", "Former Error", "New Error"]
      comparison_table = [[i+1, former_run_summary[i][1], new_run_summary[i][1]] for i in range(len(new_run_summary))]
      print(tabulate(comparison_table, headers, tablefmt="grid"))

    elif rerun_or_update == "rerun":
      # run_summary_tuple = update_rerun_parameter(rerun_number, base_folder, param_to_update, new_value)
      run_summary_tuple = update_rerun_parameter(rerun_number, base_folder, new_value)
      new_run_summary.append(run_summary_tuple)

      print(f"Rerun for run number {rerun_number} completed with updated {update_type}.")

      # Generate comparison table using former_run_summary
      headers = ["Run Number", "Former Error", "New Error"]
      comparison_table = [[rerun_number, former_run_summary[rerun_number - 1][1], new_run_summary[0][1]]]
      print(tabulate(comparison_table, headers, tablefmt="grid"))

#The clear_folders function is used to clear out a folder completely, removing all of its contents, and then recreating an empty folder with the same name

def clear_folders(base_folder):
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
    os.makedirs(base_folder, exist_ok=True)

#The zip_folder function compresses the contents of a folder into a ZIP file

def zip_folder(folder_path, zip_name):
    with ZipFile(zip_name, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

#Function to get user inputs - This function has two modes:
#it either loads existing parameters from a file or prompts the user to input new parameters
def get_input_with_default(prompt, default, data_type=str):
    # Temporarily restore stdout for user input prompts
        # original_stdout = sys.stdout
        # try:
        #     sys.stdout = original_stdout  # Restore stdout for user prompts
        user_input = input(f"{prompt} [{default}]: ")
        # finally:
        #     sys.stdout = original_stdout  # Ensure stdout is restored    
        return data_type(user_input) if user_input else default


def get_user_inputs(create_new, run_number=1):
    # folder_path = "/content/DATA"
    print("Inside get_user_inputs")
    folder_path = get_base_path()
    run_folder_path = os.path.join(folder_path, f"run_{run_number}")
    parameters_file_path = os.path.join(folder_path, "parameters.json")

    # Create the path to the parameters file within the specific run folder
    # parameters_file_path = os.path.join(run_folder_path, "parameters.json")

    if create_new and run_number == 1:
        print("Clearing existing files and creating new parameters...")
        clear_folders(folder_path)

    # if not create_new and os.path.exists(parameters_file_path):
    if not create_new:
      if os.path.exists(parameters_file_path):
            print(f"Loading parameters from {parameters_file_path} in common folder.")
      else:
        # If it doesn't exist in the common folder, check the run folder
        parameters_file_path = os.path.join(run_folder_path, "parameters.json")
        if os.path.exists(parameters_file_path):
                print(f"Loading parameters from {parameters_file_path} in run folder.")
        else:
                raise FileNotFoundError("Parameters file not found in either run or common folder.")

      with open(parameters_file_path, "r") as file:
          parameters = json.load(file)
      num_training_inputs = parameters['num_training_inputs']
      lower_bound = parameters['lower_bound']
      max_lower_bound = parameters['max_lower_bound']
      upper_bound = parameters['upper_bound']
      min_upper_bound = parameters['min_upper_bound']
      func_expressions = parameters['func_expressions']
      num_layers = parameters['num_layers']
      num_neurons = parameters['num_neurons']
      output_num_neurons =  parameters['output_num_neurons']
      epochs = parameters['epochs']
      bias_learning_rate = parameters['bias_learning_rate']
      tau_learning_rate = parameters['tau_learning_rate']
      bias_values = [np.array(bias) for bias in parameters['bias_values']]
      tau_values = [np.array(tau) for tau in parameters['tau_values']]

    #   folder_path = "/content/DATA"
      folder_path = get_base_path()

      # return num_layers, num_neurons, output_num_neurons, epochs, bias_learning_rate, tau_learning_rate, tau_bouncing_mode, bias_values, tau_values, folder_path  # Ensure initial values are returned
      return num_training_inputs, lower_bound, max_lower_bound, upper_bound,min_upper_bound, func_expressions, num_layers, num_neurons, output_num_neurons, epochs, bias_learning_rate, tau_learning_rate, bias_values, tau_values, folder_path  # Ensure initial values are returned

    else:
        print("No parameters file found or user chose to create new parameters. Asking user for inputs...")
        # Temporarily restore stdout for user input prompts
        # original_stdout = sys.stdout
        # try:
        #     sys.stdout = original_stdout  # Restore stdout for user prompts

        # Ask for training data details
        num_training_inputs = get_input_with_default("Enter the number of training inputs (total number of samples) in the dataset", 50, int)
        num_input_neurons = get_input_with_default("Enter the number of neurons for the input layer (>=1)", 2, int)
        lower_bound = get_input_with_default("Enter the lower bound of input range", 0.0, float)
        max_lower_bound = get_input_with_default("Enter the max lower bound for input", 0.45, float)
        upper_bound = get_input_with_default("Enter the upper bound of input range", 1.0, float)
        min_upper_bound = get_input_with_default("Enter the min upper bound for input", 0.55, float)
        output_num_neurons = get_input_with_default("Enter the number of neurons for the output layer (>=1)", 2, int)

        # Collect function expressions before running
        func_expressions = []
        for i in range(output_num_neurons):
            func_str = input(f"Enter y_train({i+1}) function using symbols x0, x1, x2,...(use '&' for AND, '|' for OR): ")
            func_expressions.append(func_str)

        num_layers = get_input_with_default("Enter the total number of layers (>=2): ", 3, int)
        num_layers -= 2   # Subtracting input and output layers
        hidden_num_neurons = [int(input(f"Enter the number of neurons for layer {i+1}: ")) for i in range(num_layers)]
        num_neurons = [num_input_neurons] + hidden_num_neurons
        epochs = int(input("Enter the number of epochs: "))
        bias_learning_rate = get_input_with_default("Enter the bias learning rate : ", 5.0, float)
        tau_learning_rate = get_input_with_default("Enter the tau learning rate : ", 3.0, float)
        bias_values = []
        tau_values = []
        num_layers= num_layers+1
        # values_choice = input("Enter '0' to manually input values for bias and tau, or '1' to generate random values: ")
        values_choice = 1

        if values_choice == '0':
        #Input for layers except the output layer
            for i in range(1, num_layers):
                current_layer_neurons = num_neurons[i]
                previous_layer_neurons = num_neurons[i - 1]

                # Input for each neuron in the current layer
                layer_biases = []
                layer_taus = []

                for j in range(current_layer_neurons):
                    for k in range(previous_layer_neurons):
                        layer_biases.append(int(input(f"Enter bias value for hidden layer {i}, neuron {k+1}: ")))
                        layer_taus.append(float(input(f"Enter tau value for hidden layer {i}, neuron {k+1}: ")))

                #Reshape and append for layers except the last layer
                layer_biases = np.array(layer_biases).reshape(num_neurons[i], num_neurons[i-1])
                layer_taus = np.array(layer_taus).reshape(num_neurons[i], num_neurons[i-1])
                bias_values.append(layer_biases)
                tau_values.append(layer_taus)

            #Input for the output layer separately
            layer_biases = []
            layer_taus = []
            previous_layer_neurons = num_neurons[-2]

            for j in range(num_neurons[-1]):
                for k in range(previous_layer_neurons):
                    layer_biases.append(int(input(f"Enter bias value for output layer, neuron {k+1}: ")))
                    layer_taus.append(float(input(f"Enter tau value for output layer, neuron {k+1}: ")))

            #Reshape and append for the output layer
            layer_biases = np.array(layer_biases).reshape(output_num_neurons, num_neurons[-1])
            layer_taus = np.array(layer_taus).reshape(output_num_neurons, num_neurons[-1])
            bias_values.append(layer_biases)
            tau_values.append(layer_taus)

        else:
        #randomly initialized bias values
            bias_values = [np.random.randint(0, 2, (num_neurons[i + 1], num_neurons[i])) for i in range(num_layers - 1)]
            #Reshape and append bias for the output layer
            bias_values.append(np.random.randint(0, 2, (output_num_neurons, num_neurons[-1])))

            #randomly initialized tau values
            tau_values = [np.round(np.random.rand(num_neurons[i + 1], num_neurons[i]) * 0.8+0.1, decimals=1) for i in range(num_layers - 1)]
            #Reshape and append tau for the output layer
            tau_values.append(np.round(np.random.rand(output_num_neurons, num_neurons[-1]) * 0.8+0.1, decimals=1))

        # finally:
        #     sys.stdout = original_stdout  # Ensure stdout is restored

        #prepare a dictionary of parameters, save these parameters to a JSON file, and return them
        parameters = {
            'num_training_inputs': num_training_inputs,
            'lower_bound': lower_bound,
            'max_lower_bound': max_lower_bound,
            'upper_bound': upper_bound,
            'min_upper_bound': min_upper_bound,
            'func_expressions': func_expressions,
            'num_layers': num_layers,
            'num_neurons' : num_neurons,
            'num_input_neurons' : num_input_neurons,
            'output_num_neurons': output_num_neurons,
            'epochs': epochs,
            'bias_learning_rate': bias_learning_rate,
            'tau_learning_rate': tau_learning_rate,
            'bias_values': [b.tolist() for b in bias_values],  # Ensure 2D structure
            'tau_values': [t.tolist() for t in tau_values]
        }
        with open(parameters_file_path, "w") as file:
            json.dump(parameters, file, indent=4)

        # return num_layers, num_neurons, output_num_neurons, epochs, bias_learning_rate, tau_learning_rate, tau_bouncing_mode, bias_values, tau_values, folder_path
        return num_training_inputs, lower_bound,max_lower_bound, upper_bound,min_upper_bound, func_expressions, num_layers, num_neurons, output_num_neurons, epochs, bias_learning_rate, tau_learning_rate, bias_values, tau_values, folder_path

    return None

# # Function to generate random inputs and compute outputs for AND, OR, XOR gates
# # def generate_data(num_input_neurons, output_num_neurons, folder_path, create_new):
# def generate_data(num_input_neurons, output_num_neurons, num_training_inputs, lower_bound, max_lower_bound, upper_bound, min_upper_bound, func_expressions, folder_path, create_new):
#   x_train_file_path = os.path.join(folder_path, "x_train.npy")
#   y_train_file_path = os.path.join(folder_path, "y_train.npy")
#   print ('value for create_new from generate_data: ', create_new)

#   # Function to check if the operation is logical (like AND/OR)
#   def is_logical_operation(func_str):
#     logical_ops = ['&', '|', 'and', 'or']  # Add any other logical operators you want to support
#     return any(op in func_str for op in logical_ops)

#   print("x_train_file_path: ", x_train_file_path)
#   print("y_train_file_path: ", y_train_file_path)
#   if not create_new and os.path.exists(x_train_file_path) and os.path.exists(y_train_file_path):
#     #if os.path.exists(x_train_file_path) and os.path.exists(y_train_file_path):
#         print("Loading data from files...")
#         x_train = np.load(x_train_file_path)
#         y_train = np.load(y_train_file_path)
#   else:

#         # Create X_train dataset with random values uniformly between the bounds
#         x_train = np.random.uniform(lower_bound, upper_bound, (num_training_inputs, num_input_neurons))
#         # Apply limiting conditions
#         for i in range(num_input_neurons):
#             mask = np.random.choice([0, 1], size=num_training_inputs)

#             # Set values in the range [lower_bound, max_lower_bound]
#             x_train[:, i][mask == 0] = np.random.uniform(lower_bound, max_lower_bound, size=(mask == 0).sum())

#             # Set values in the range [min_upper_bound, upper_bound]
#             x_train[:, i][mask == 1] = np.random.uniform(min_upper_bound, upper_bound, size=(mask == 1).sum())
#         # Normalize the X_train data between 0 and 1
#         x_train = np.round(x_train, 2)

#         x_train_1=x_train

#         # x_train_min = x_train.min(axis=0)
#         # x_train_max = x_train.max(axis=0)
#         # x_train = (x_train - x_train_min) / (x_train_max - x_train_min)
#         # normalization
#         x_train = (x_train - lower_bound) / (upper_bound - lower_bound)

#         x_train = np.round(x_train, 2)

#         # Create y_train by asking user to provide functions for each output
#         y_train = []
#         x_symbols = sp.symbols(f'x0:{num_input_neurons}')  # Creates symbols x0, x1, x2, ..., up to num_inputs

#         # for i in  range(output_num_neurons):
#         for func_str in  func_expressions:
#           func = sp.sympify(func_str)  # Convert string to sympy expression
#           func_callable = sp.lambdify(x_symbols, func, 'numpy')  # Convert to callable

#           if is_logical_operation(func_str):
#             # Apply thresholding if the function is a logical operation
#             binary_x_train = (x_train > 0.5).astype(int)  # Convert to binary values based on threshold

#             # func = sp.sympify(func_str)  # Convert string to sympy expression
#             # func_callable = sp.lambdify(x_symbols, func, 'numpy')  # Convert to callable

#             # Evaluate the function for each row
#             y_values = np.array([func_callable(*binary_x_train[row]) for row in range(num_training_inputs)])
#             # y_values = np.array([func_callable(*x_train[row]) for row in range(num_training_inputs)])

#             # Convert boolean results to binary (0 and 1)
#             y_values = y_values.astype(int)

#           else:
#             # For mathematical functions, apply directly on continuous values
#             # func = sp.sympify(func_str)  # Convert string to sympy expression
#             # func_callable = sp.lambdify(x_symbols, func, 'numpy')  # Convert to callable
#             y_values = np.array([func_callable(*x_train[row]) for row in range(num_training_inputs)])
#             # def evaluate_row(row):
#             #         input_map = {f'x{i}': row[i] for i in range(len(row))}
#             #         return float(func.subs(input_map).evalf())

#             #     # Evaluate the function for each row
#             # y_values = np.array([evaluate_row(x_train[row]) for row in range(num_training_inputs)])


#           # Append results to y_train
#           y_train.append(y_values)

#       # Convert y_train to numpy array and transpose to match the format (num_training_inputs, num_outputs)
#         y_train = np.array(y_train).T

#         # print("\ny_train:")
#         # print(y_train)

#         # Save x_train and y_train to the folder path
#         # np.save(x_train_file_path, x_train)
#         # np.save(y_train_file_path, y_train)

#         # dataset_file_path = os.path.join(folder_path, "dataset.csv")
#         #dataset = pd.DataFrame(np.hstack((x_train, y_train)), columns=[f'Input_{i+1}' for i in range(num_input_neurons)] + ['AND', 'OR', 'XOR'])
#         # dataset = pd.DataFrame(np.hstack((x_train, y_train)), columns=[f'Input_{i+1}' for i in range(num_input_neurons)] + [f'Output_{i+1}' for i in range(output_num_neurons)])

#         # dataset.to_csv(dataset_file_path, index=False)

#   return x_train, y_train,x_train_1

# Function to generate random inputs and compute outputs for AND, OR, XOR gates
# def generate_data(num_input_neurons, output_num_neurons, folder_path, create_new):
def generate_data(num_input_neurons, output_num_neurons, num_training_inputs, lower_bound, max_lower_bound, upper_bound, min_upper_bound, func_expressions, folder_path, create_new):
  x_train_file_path = os.path.join(folder_path, "x_train.npy")
  y_train_file_path = os.path.join(folder_path, "y_train.npy")
  print ('value for create_new from generate_data: ', create_new)

  # Function to check if the operation is logical (like AND/OR)
  def is_logical_operation(func_str):
    logical_ops = ['&', '|', 'and', 'or']  # Add any other logical operators you want to support
    return any(op in func_str for op in logical_ops)

  print("x_train_file_path: ", x_train_file_path)
  print("y_train_file_path: ", y_train_file_path)
  if not create_new and os.path.exists(x_train_file_path) and os.path.exists(y_train_file_path):
    #if os.path.exists(x_train_file_path) and os.path.exists(y_train_file_path):
        print("Loading data from files...")
        x_train = np.load(x_train_file_path)
        y_train = np.load(y_train_file_path)
  else:

        # Create X_train dataset with random values uniformly between the bounds
        x_train = np.random.uniform(lower_bound, upper_bound, (num_training_inputs, num_input_neurons))
        # Apply limiting conditions
        for i in range(num_input_neurons):
            mask = np.random.choice([0, 1], size=num_training_inputs)

            # Set values in the range [lower_bound, max_lower_bound]
            x_train[:, i][mask == 0] = np.random.uniform(lower_bound, max_lower_bound, size=(mask == 0).sum())

            # Set values in the range [min_upper_bound, upper_bound]
            x_train[:, i][mask == 1] = np.random.uniform(min_upper_bound, upper_bound, size=(mask == 1).sum())
        x_train_1=x_train
        x_train = np.round(x_train, 2)
        x_train = (x_train - lower_bound) / (upper_bound - lower_bound)
        # Create y_train by asking user to provide functions for each output
        y_train = []
        x_symbols = sp.symbols(f'x0:{num_input_neurons}')  # Creates symbols x0, x1, x2, ..., up to num_inputs

        # for i in  range(output_num_neurons):
        for func_str in  func_expressions:
          func = sp.sympify(func_str)  # Convert string to sympy expression
          func_callable = sp.lambdify(x_symbols, func, 'numpy')  # Convert to callable

          if is_logical_operation(func_str):
            # Apply thresholding if the function is a logical operation
            binary_x_train = (x_train > 0.5).astype(int)  # Convert to binary values based on threshold

            # func = sp.sympify(func_str)  # Convert string to sympy expression
            # func_callable = sp.lambdify(x_symbols, func, 'numpy')  # Convert to callable

            # Evaluate the function for each row
            y_values = np.array([func_callable(*binary_x_train[row]) for row in range(num_training_inputs)])
            # y_values = np.array([func_callable(*x_train[row]) for row in range(num_training_inputs)])

            # Convert boolean results to binary (0 and 1)
            y_values = y_values.astype(int)

          else:
            # For mathematical functions, apply directly on continuous values
            # func = sp.sympify(func_str)  # Convert string to sympy expression
            # func_callable = sp.lambdify(x_symbols, func, 'numpy')  # Convert to callable
            y_values = np.array([func_callable(*x_train[row]) for row in range(num_training_inputs)])

          # Append results to y_train
          y_train.append(y_values)

      # Convert y_train to numpy array and transpose to match the format (num_training_inputs, num_outputs)
        y_train = np.array(y_train).T

        # print("\ny_train:")
        # print(y_train)

        # Save x_train and y_train to the folder path
        # np.save(x_train_file_path, x_train)
        # np.save(y_train_file_path, y_train)

        # dataset_file_path = os.path.join(folder_path, "dataset.csv")
        #dataset = pd.DataFrame(np.hstack((x_train, y_train)), columns=[f'Input_{i+1}' for i in range(num_input_neurons)] + ['AND', 'OR', 'XOR'])
        # dataset = pd.DataFrame(np.hstack((x_train, y_train)), columns=[f'Input_{i+1}' for i in range(num_input_neurons)] + [f'Output_{i+1}' for i in range(output_num_neurons)])

        # dataset.to_csv(dataset_file_path, index=False)

  return x_train, y_train,x_train_1

def get_valid_input(prompt, validation_fn, error_message):
    """
    General function to get validated user input.

    Parameters:
    - prompt: The message to display when asking for input.
    - validation_fn: A function that takes the input and returns True if it's valid, False otherwise.
    - error_message: The message to display when the input is invalid.

    Returns:
    - The validated input.
    """
    while True:
        user_input = input(prompt).strip().lower()
        if validation_fn(user_input):
            return user_input
        else:
            print(error_message)

#  validation functions for different input types
def validate_run_number(input_value, max_runs):
    try:
        run_number = int(input_value)
        return 1 <= run_number <= max_runs
    except ValueError:
        return False

# def validate_update_type(input_value):
#     valid_types = {"bias", "tau", "taubounce", "epochs"}
#     return input_value.lower() in valid_types

def validate_yes_no(input_value):
    return input_value.lower() in {"yes", "no"}

def is_valid_number(value):
    """Check if the input value can be converted to a valid number (int or float)."""
    try:
        num_value = float(value.strip())  # Convert to float
        return True
    except ValueError:
        return False

def validate_binary_string(input_str):    #This is a replacement for the validate_update_type function defined above.
    """Check if the input string is a 5-digit binary string (i.e., contains only '0' or '1')."""
    return len(input_str) == 5 and all(char in '01' for char in input_str)

def validate_rerun_or_update(input_value):
    """Check if the input value is one of the valid options."""
    return input_value in {"rerun", "update", "exit"}