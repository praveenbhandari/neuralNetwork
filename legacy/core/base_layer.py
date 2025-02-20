import abc
import numpy as np

class Layer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.input = None
        self.output = None
        self.bias = None
        self.tau = None
        self.z = None
        self.Z = None  # New attribute to store the value of Z
        self.constant = None  # New attribute to store the value of constant

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, bias_learning_rate, tau_learning_rate, epochs,Change_bias,Change_tau):
        raise NotImplementedError
