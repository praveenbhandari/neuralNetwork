import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.resolve())
print(sys.path[-1])

import matplotlib.pyplot as plt
import numpy as np

from src.models.base_model import BaseModel
from src.models.layers import OriginalLayer

from src.utils.logger import setup_logger
import tqdm
logger =setup_logger()

class OriginalModel(BaseModel):
    def __init__(self):
        super(OriginalModel, self).__init__() # includes self.layers = []
    
    @classmethod
    def build(cls, input_size, output_size, num_hidden_layers=3, size_hidden_layers=10):
        """
        A factory method that constructs a KeonNetwork with a preset architecture.
        """
        model = cls()
        logger.info(f"Building model with input size: {input_size}, output size: {output_size}, "
                    f"{num_hidden_layers} hidden layers of size {size_hidden_layers}")
        
        # Input layer
        model.add(OriginalLayer(input_size, size_hidden_layers, name='Original_InputLayer'))
        logger.debug("Input layer added.")

        for _ in range(num_hidden_layers):
            model.add(OriginalLayer(size_hidden_layers, size_hidden_layers))
            logger.debug(f"Hidden layer {_+1} added.")

        # Output layer
        model.add(OriginalLayer(size_hidden_layers, output_size, name='Original_OutputLayer'))
        return model
    
    def parameters(self):
        return self.layers

    def add(self, layer):
        """
        Add a layer to the model.
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        Forward pass through all layers.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """
        Backward pass through all layers (in reverse).
        Note: This just computes gradients and stores them
              in the layers (no updates here).
        """
        for layer in reversed(self.layers):
        # for layer in tqdm.tqdm(reversed(self.layers), desc="Backward pass", leave=False):
            grad = layer.backward(grad)
            # logger.debug(f"Backward pass through layer: {layer.name}.")

        return grad
