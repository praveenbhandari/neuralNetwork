import abc
import numpy as np

class BaseLoss(metaclass=abc.ABCMeta):
    """
    Abstract base class for custom loss functions.
    Defines the interface for forward (loss computation) and backward (gradient).
    """

    @abc.abstractmethod
    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Forward pass: compute the loss given predictions and targets.
        
        Args:
            predictions (np.ndarray): Model outputs.
            targets (np.ndarray): Ground truth values.
        
        Returns:
            float: The computed loss.
        """
        pass

    @abc.abstractmethod
    def backward(self) -> np.ndarray:
        """
        Backward pass: compute gradients with respect to predictions.
        
        Returns:
            np.ndarray: Gradient of the loss wrt. predictions (same shape as predictions).
        """
        pass
