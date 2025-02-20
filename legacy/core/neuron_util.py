"""
This script will have helpful functions (mainly activation + loss functions) defined 
to be imported via class definitions in other files.
"""

import numpy as np

####### ACTIVATION FUNCTIONS ########
def ReLU(x):
    return np.maximum(0, x)

# Modified Sigmoid activation function used in FP
def sigmoid(x, alpha=1):
    x = x.astype(float)
    return 1 / (1 + np.exp(-alpha * x))

# Modified Derivative of Sigmoid activation function used in BP
def sigmoid_prime(x, alpha=1):
    sig_prime_1 = (1 - sigmoid(x, alpha))
    sig_prime_2 = sigmoid(x, alpha)
    return alpha * sigmoid(x, alpha) * (1 - sigmoid(x, alpha))

#Softmax Function for the last layer - used for classification tasks
def softmax(x, alpha=1):
    sig_prime_1 = (1 - sigmoid(x, alpha))
    sig_prime_2 = sigmoid(x, alpha)
    return alpha * sigmoid(x, alpha) * (1 - sigmoid(x, alpha))


####### LOSS FUNCTIONS ########
# loss function: Used to visualize the model learning - error after each epoch
def mse(y_true, y_pred):
    return np.mean(np.power(np.squeeze(y_true) - np.squeeze(y_pred), 2))