import numpy as np
from .base_function import BaseFunction
from src.utils.logger import setup_logger

logger=setup_logger()
class SinFunction(BaseFunction):
    """
    A sine wave function: f(x) = amplitude * sin(freq * x + phase).
    """

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0):
        """
        Args:
            amplitude (float): The vertical scaling (default=1.0).
            frequency (float): The frequency of the sine wave (default=1.0).
            phase (float): The horizontal shift (phase offset) (default=0.0).
        """
        super().__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        logger.debug(f"SinFunction initialized with amplitude={amplitude}, frequency={frequency}, phase={phase}")


    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the sine function at x.

        Args:
            x (np.ndarray): Input values (N,)

        Returns:
            np.ndarray: amplitude * sin(frequency * x + phase)
        """
        # logger.debug(f"Evaluating SinFunction with input x={x}")
        # result = self.amplitude * np.sin(self.frequency * x + self.phase)
        # logger.debug(f"SinFunction output: {result}")
        return self.amplitude * np.sin(self.frequency * x + self.phase)
        # return result
