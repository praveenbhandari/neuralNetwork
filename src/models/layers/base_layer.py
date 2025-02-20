import numpy as np

# Base class
class BaseLayer:
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
    def backward_propagation(self, dL_dOut: np.ndarray):
        raise NotImplementedError
