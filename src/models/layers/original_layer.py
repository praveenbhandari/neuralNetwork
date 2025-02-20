import numpy as np

from .base_layer import BaseLayer
from .activations import sigmoid, sigmoid_prime
from src.utils.logger import setup_logger
import tqdm
logger =setup_logger()

# inherit from base class Layer: Fully connected layer code
class OriginalLayer(BaseLayer):

    def __init__(self, input_size, output_size, name=None, initial_tau=None):
        super(OriginalLayer, self).__init__()

        self.name = name  # Layer name
        self.input_size = input_size
        self.output_size = output_size
        self.tau_increment = 0.1

        if initial_tau is not None:
            self.tau = initial_tau  # shape: (input_size, output_size)
        else:
            # Random initialization
            self.tau = np.random.rand(input_size, output_size)
        ## NOTE: self.tau.size should be (number of inputs, number of neurons)


        # Outputs needed to store for back propogation
        self.output = None
        self.grad_tau = None
    
    def __repr__(self):
        return f"{self.name if self.name else self.__class__.__name__}|({self.input_size}->{self.output_size})"

    # returns output for a given input
    def forward(self, x: np.ndarray):
        # logger.debug(f"Forward propagation for layer: {self.name}")

        # Ensure x is at least 2D
        if x.ndim == 1:
            # Make it (1, input_size)
            x = x[np.newaxis, :]

        # Shape check
        batch_size, in_dim = x.shape
        if in_dim != self.input_size:
            logger.debug(f"Layer [{self.name}] Expected input size {self.input_size}, got {in_dim}")
            raise ValueError(f"Layer [{self.name}] Expected input size {self.input_size}, got {in_dim}")

        self.x = x

        # print(f"Forward propagation for : {self.name} with input neurons: {self.input_size} and output neurons: {self.output_size}")

        # Prepare output array
        out = np.zeros((batch_size, self.output_size), dtype=np.float32)

        # Assert we have the right shape of weights for the given input
        assert in_dim == self.tau.shape[0], ValueError(f'Shape of input + taus are mismatched. Input shape: {x.shape} | self.taus shape: {self.tau.shape}')

        # For each sample in the batch
        for b in range(batch_size):
            # For each output neuron
            for n in range(self.output_size):
                # 1) Count how many inputs are less than the threshold tau
                count_n = 0
                for k in range(self.input_size):
                    if self.tau[k, n] > x[b, k]:
                        count_n += 1

                # 2) Out_n = max(1 - tau_increment * count_n, 0)
                raw = 1.0 - self.tau_increment * count_n
                out[b, n] = raw if raw > 0 else 0
                # logger.debug(f"Output neuron {n}: Count = {count_n}, Raw output = {raw}")


        # Save output
        self.output = out 

        # Return shape (batch_size, output_size) or squeezed if batch_size=1
        return out if batch_size > 1 else out[0]

    def backward(self, dL_dOut: np.ndarray):
            """
            Compute gradients w.r.t tau:
            - dL_dOut: shape (batch_size, output_size) or (output_size,)

            We'll accumulate gradients in self.grad_tau (shape: (input_size, output_size)).

            Returns dL/dX if you want to pass gradient back to previous layer, 
            but here we might return None or a subgradient approach. 
            """
            # Make sure dL_dOut is 2D
            # logger.debug(f"Backward propagation for layer: {self.name}")

            if dL_dOut.ndim == 1:
                dL_dOut = dL_dOut[np.newaxis, :]

            batch_size, out_dim = dL_dOut.shape
            if out_dim != self.output_size:
                logger.debug(f"[{self.name}] Mismatch in output gradient size {out_dim} vs. {self.output_size}")

                raise ValueError(f"[{self.name}] Mismatch in output gradient size {out_dim} vs. {self.output_size}")

            # Reset/allocate grad_tau
            self.grad_tau = np.zeros_like(self.tau)

            # Also allocate gradient wrt X
            dL_dX = np.zeros_like(self.x)  # same shape as self.x => (batch_size, input_size)

            # Subgradient approach:
            # out_n = max(1 - tau_increment * count_n, 0)
            # If out_n > 0, dOut_n/dCount_n = -tau_increment
            # Then dCount_n/dTau[k,n] = 1 if tau[k,n] > x[k], else 0
            for b in range(batch_size):
                for n in range(self.output_size):
                    # If output was zero, no gradient flows
                    if self.output[b, n] <= 0:
                        continue

                    # chain rule => dL/dTau[k,n] = dL/dOut[n] * (-tau_increment) * I(tau[k,n] > x[k])
                    for k in range(self.input_size):
                        if self.tau[k, n] > self.x[b, k]:
                            # Grad wrt Tau
                            self.grad_tau[k, n] += dL_dOut[b, n] * (-self.tau_increment)

                            # Grad wrt X (straight-through subgradient)
                            dL_dX[b, k] += dL_dOut[b, n] * (self.tau_increment)
                            # logger.debug(f"Grad tau[{k},{n}] = {self.grad_tau[k, n]}")
                            # logger.debug(f"Grad X[{b},{k}] = {dL_dX[b, k]}")


            # If needed, you could compute gradient w.r.t inputs (dL/dX), 
            # but given your custom threshold logic, this is also a discrete step.
            # We'll return None for now (i.e. no upstream grad).
            return dL_dX
