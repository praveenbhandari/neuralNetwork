import numpy as np
from .base_loss import BaseLoss

class MSELoss(BaseLoss):
    """
    Implements Mean Squared Error loss manually (pure numpy).
    Demonstrates a forward (loss calculation) and backward (gradient) pass.
    """
    def __init__(self):
        super().__init__()
        self.diff = None  # We'll store the difference for use in backward()

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Forward pass: computes the MSE between predictions and targets.
        
        Args:
            predictions (np.ndarray): Model outputs, shape (N, ...)
            targets (np.ndarray): Ground truth, shape (N, ...)
        
        Returns:
            float: The mean squared error.
        """
        self.diff = predictions - targets
        return np.mean(self.diff ** 2)

    def backward(self) -> np.ndarray:
        """
        Backward pass: compute gradient of the MSE w.r.t. predictions.
        
        Returns:
            np.ndarray: Gradient of the MSE loss with respect to predictions,
                        same shape as `predictions`.
        """
        # MSE derivative: dL/dpred = 2/N * (pred - target)
        if self.diff is None:
            raise ValueError("Must call loss forward pass before backward().")

        n = self.diff.size  # or self.diff.shape[0], depends on the shape
        return (2.0 / n) * self.diff
