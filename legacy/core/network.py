import matplotlib.pyplot as plt
import numpy as np

from .fc_layer import FCLayer

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss):
        self.loss = loss

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samplesnp
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
              output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, bias_learning_rate, tau_learning_rate, debug_mode='no'):

        # sample dimension first
        prev_tau_gradient = None
        samples = len(x_train)
        errors = []  # List to store average errors for plotting

        Skip_tau_change = False

        # Track errors for the last 3 epochs to adjust bias learning rate
        last_3_errors = []
        lowest_error = float('inf')
        best_tau_values = []  # To track tau values corresponding to the lowest error
        best_bias_values = []  # To track tau values corresponding to the lowest error
        adjusted_bias_learning_rate = False
        initial_bias = bias_learning_rate  # Store the initial bias value

        # training loop
        for i in range(epochs):
            Change_tau = False
            Change_bias = False
            print(f"For epoch {i+1}, bias learning rate = {bias_learning_rate}")

            pi_va=np.pi/2

            print(f"For epoch {i+1}, skipping tau change = {Skip_tau_change}")
            err = 0  # Reset error for the current epoch
            print(f"For epoch {i+1}, Change_tau Prior to check= {Change_tau}, and Change_bias = {Change_bias}")
            if ((i % 2 != 0) or (i > 4 and not Skip_tau_change)):
                Change_tau = True
                Skip_tau_change = False
            if ((i % 2 == 0) or (i > 4)):
                Change_bias = True  # Check if the epoch is even
            print(f"For epoch {i+1}, Change_tau = {Change_tau}, and Change_bias = {Change_bias}")
            for j in range(samples):

                # forward propagation

                # updated val based on calculations
                output = x_train[j]*pi_va

                for layer in self.layers:
                    output = layer.forward_propagation(output)

                sample_err = self.loss(y_train[j], output)
                err += sample_err
                print(f'Epoch {i+1}, Sample {j+1}, Error: {sample_err}')


                #backward propagation
                print(f" *** Backward propagation Begins *** ")
                for layer in reversed(self.layers):
                    is_last_layer = (layer == self.layers[-2])  # Check if it's the last layerindex
                    prev_tau_gradient = layer.backward_propagation(is_last_layer, y_train[j], prev_tau_gradient, bias_learning_rate, tau_learning_rate, epochs,Change_bias,Change_tau)
                print(f" *** End of Backward propagation *** \n")

                # forward propagation - once again FP only for debug mode
                if debug_mode in ["yes", "y"]:
                  print(f" *** Forward propagation 2 Begins for the same input *** ")
                  for layer in self.layers:
                    output = layer.forward_propagation(x_train[j])

                  sample_err = self.loss(y_train[j], output)
                  err += sample_err
                  print(f'Epoch {i+1}, Sample {j+1}, Error: {sample_err}')
                  print(f" *** End of Forward propagation 2 *** \n")

            # calculate average error on all samples
            print(f" *** End of epoch: {i+1} ***")
            avg_err = err / samples
            errors.append(avg_err)
            print('Epoch %d/%d,   Average Error: %f \n' % (i+1, epochs, avg_err))

            # Track the lowest error and associated tau values
            if avg_err < lowest_error:
              lowest_error = avg_err
              for layer in self.layers:
                # if isinstance(layer, FCLayer):
                    best_bias_values.append(np.copy(layer.bias))
                    best_tau_values.append(np.copy(layer.tau))
                    print(f" Saving the tau value {best_tau_values} ")
                    print(f"Saving the bias value {best_bias_values}")

            # Check if the last 3 errors are constant or increasing
            last_3_errors.append(avg_err)
            if len(last_3_errors) > 3:
                last_3_errors.pop(0)  # Keep only the last 3 errors

              #  if len(last_3_errors) == 3 and last_3_errors[0] < last_3_errors[-1]:
              #   print(f"Error increasing/constant over last 3 epochs. Reverting to best tau and bias values observed so far.")
              #   for layer, best_tau, best_bias in zip(self.layers, best_tau_values, best_bias_values):
              #     # if isinstance(layer, FCLayer):
              #         layer.tau = np.copy(best_tau)
              #         layer.bias = np.copy(best_bias)
              #         print(f" Adjusted tau value {layer.tau} ")
              #         print(f"Adjusted bias value {layer.bias}")

                  # Error is increasing or constant, adjust learning rate
                Skip_tau_change = True
                if not adjusted_bias_learning_rate:
                    bias_learning_rate += 1
                    adjusted_bias_learning_rate = True
                    print(f"Adjusting bias learning rate to {bias_learning_rate} for Epoch {i+1}")

        # Save final bias and tau values
        final_bias_values = [layer.bias for layer in self.layers if isinstance(layer, FCLayer)]
        final_tau_values = [layer.tau for layer in self.layers if isinstance(layer, FCLayer)]

        # Plotting error rates across epochs
        plt.plot(range(1, epochs + 1), errors, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Average Error')
        plt.title('Error Rate Across Epochs')
        plt.show()

        return errors, final_bias_values, final_tau_values
