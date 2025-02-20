import abc
import numpy as np

class BaseFunction(metaclass=abc.ABCMeta):
    """
    Abstract base class for custom functions that a model might be designed to fit.
    """

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at the given values of x.

        Args:
            x (np.ndarray): Input values (e.g. shape (N,) or (N,1))

        Returns:
            np.ndarray: Output values f(x), same shape as x or suitably broadcasted.
        """
        pass
