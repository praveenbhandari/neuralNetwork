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
    # return alpha * sigmoid(x, alpha) * (1 - sigmoid(x, alpha))
    return alpha * sig_prime_2*sig_prime_1

#Softmax Function for the last layer - used for classification tasks
def softmax(x, alpha=1):
    sig_prime_1 = (1 - sigmoid(x, alpha))
    sig_prime_2 = sigmoid(x, alpha)
    return alpha * sigmoid(x, alpha) * (1 - sigmoid(x, alpha))
    