import numpy as np

from .base_layer import Layer
from .neuron_util import sigmoid, sigmoid_prime
from src.utils.logger import setup_logger
import tqdm
logger =setup_logger()
# inherit from base class Layer: Fully connected layer code
class FCLayer(Layer):

    def __init__(self, input_size, output_size, name, bias=None, tau=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.name = name  # Layer name
        self.bias = bias
        self.tau = tau

    # Set Z and constant based on the input_size
        if self.input_size >= 10:
            self.Z = 0.1
            self.constant = 10
        else :
            self.Z = round(1/self.input_size, 2)
            self.constant = self.input_size

    # returns output for a given input
    def forward_propagation(self, input_data):
        # logger.debug(f"Forward propagation for: {self.name} with input neurons: {self.input_size} and output neurons: {self.output_size}")
        
        # print(f"Forward propagation for : {self.name} with input neurons: {self.input_size} and output neurons: {self.output_size}")
        self.input = input_data
        # print(f"input for the layer : {self.input} with tau: {self.tau} and bias: {self.bias}")

        # Ensure input and tau are reshaped to 1D arrays for element-wise comparison
        input_flat = self.input.flatten()
        tau_flat = self.tau.flatten()

        # Ensure that tau and input have the same size before comparison
        min_size = min(input_flat.size, tau_flat.size)
        input_flat = input_flat[:min_size]
        tau_flat = tau_flat[:min_size]

        # Use np.isclose to find indices where input and tau are approximately the same within a tolerance
        identical_indices = np.where(np.isclose(input_flat, tau_flat, atol=0.01))

        if identical_indices[0].size > 0:
            # print(f"Identical input and tau detected at indices {identical_indices}. Updating input values.")
            logger.debug(f"Identical input and tau detected at indices {identical_indices}. Updating input values.")

            # Loop through the indices where input and tau are close
            for idx in identical_indices[0]:
            # for idx in tqdm(identical_indices[0], desc="Adjusting Inputs"):
              if (0.6 <= tau_flat[idx] <= 1) or (0.4 <= tau_flat[idx] <= 0.5):  #between 0.6 to 1 OR between 0.4 and 0.5
                # Subtract 0.1 if tau is greater than 0.8
                input_flat[idx] -= 0.1
                # print(f"Tau value at index {idx} is greater than 0.8. Subtracting 0.1 from input.")
                logger.debug(f"Tau value at index {idx} is greater than 0.8. Subtracting 0.1 from input.")
                
              else:
                # Add 0.1 if tau is 0.8 or less
                input_flat[idx] += 0.1
                logger.debug(f"Tau value at index {idx} is less than or equal to 0.8. Adding 0.1 to input.")

                # print(f"Tau value at index {idx} is less than or equal to 0.8. Adding 0.1 to input.")

            # Reshape input_flat back to original shape
            self.input = input_flat.reshape(self.input.shape)
            logger.debug(f"Updated input for the layer: {self.input} with tau: {self.tau} and bias: {self.bias}")

            # print(f"Updated input for the layer: {self.input} with tau: {self.tau} and bias: {self.bias}")

        #calculate z(i)
        self.z = self.Z * (np.sum(sigmoid((200 * self.bias - 100) * (self.input - self.tau)), axis = 1, keepdims=True) - self.input_size + self.constant)
        self.output =  self.z.T
        # print(f"output after Forward Propagation : {self.output}")
        logger.debug(f"Output after Forward Propagation: {self.output}")


        return self.output

    # computes thresholds - tau, tau_gradient and bias_gradient
    def backward_propagation(self, is_last_layer, layer_target_output, prev_tau_gradient, bias_learning_rate, tau_learning_rate, epochs,Change_bias,Change_tau):
        # print(f"Backward propagation begins for : {self.name}")
        logger.debug(f"Backward propagation begins for: {self.name}")

        bias_learning_rate_modified = bias_learning_rate
        tau_learning_rate_modified = tau_learning_rate
        #calculate the beta and theta for the previous layer
        beta = self.Z * sigmoid_prime((2 * self.bias - 1) * (self.input - self.tau)) * (2 * self.bias - 1)
        theta = self.Z * sigmoid_prime((2 * self.bias - 1) * (self.input - self.tau)) * 2 * (self.input - self.tau)

        print('beta from BP: ',  beta)
        print('theta from BP: ',  theta)

        if Change_tau == True:
          tau_learning_rate_modified = tau_learning_rate - 1

        if Change_bias == True:
          bias_learning_rate_modified = bias_learning_rate - 2

        print('Effective Bias Learning rate = ', bias_learning_rate_modified)
        print('Effective Tau Learning rate = ', tau_learning_rate_modified)

        #calculate gradients based on layer
        print(' Calculate tau and bias gradients:')
        if is_last_layer:

            tau_gradient = np.round((layer_target_output - self.output).T * beta, decimals=2)
            tau_gradient = np.where(tau_gradient < 0.1, tau_gradient * tau_learning_rate_modified, tau_gradient)

            bias_gradient = (self.output - layer_target_output).T * theta
            bias_gradient = np.where(np.absolute(bias_gradient) < 0.05, bias_gradient , bias_gradient * bias_learning_rate_modified)

        else:
            tau_gradient = np.round(np.dot(np.sum(prev_tau_gradient, axis=0), beta), decimals = 2)
            tau_gradient = np.where(tau_gradient < 0.1, tau_gradient * tau_learning_rate_modified, tau_gradient)

            bias_gradient = np.dot(np.sum(prev_tau_gradient, axis=0), -theta)
            bias_gradient = np.where(np.absolute(bias_gradient) < 0.05, bias_gradient , bias_gradient * bias_learning_rate_modified)
            bias_gradient = np.where(bias_gradient > 1, 1, bias_gradient)

        #tau_gradient = np.clip(tau_gradient, -1, 1)
        print('tau_gradient from BP: ',  tau_gradient)
        print('bias_gradient from BP: ',  bias_gradient)

        #store the value of tau_gradient to be passed on to the next layer in BP
        prev_tau_gradient = tau_gradient

        # update bias and tau for the current layer
        print(' Update tau and bias values based on gradients:')

        self.tau = np.round(self.tau - tau_gradient, decimals=2)
        self.tau = np.where(
            np.logical_or(self.tau > 0.9, self.tau < 0.1),
            np.where(self.tau > 0.9, 0.9 - (self.tau - 0.9), 0.1 + (0.1 - self.tau)),
            # Limit the change to 0.2 for values within the range
            np.clip(self.tau, self.tau - 0.2, self.tau + 0.2)
            )

        self.bias = np.round(self.bias - bias_gradient, decimals=0)
        self.bias = np.clip(self.bias, 0, 1)

        print('self.tau after update from BP: ', self.tau)
        print('self.bias after update from BP: ',  self.bias)
        print(f"Backward propagation ends for : {self.name}")

        return prev_tau_gradient