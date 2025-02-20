import abc

class BaseModel(metaclass=abc.ABCMeta):
    """
    An abstract base class for custom ML models.
    
    Subclasses should implement the abstract methods to define
    the forward pass and any other required methods.
    """

    def __init__(self):
        self.layers = []

    def __repr__(self):
        
        return f"{self.__class__.__name__}(layers={self.layers})"

    @abc.abstractmethod
    def forward(self, x):
        """
        The forward pass of the model. Must be overridden by subclasses.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: The model's output.
        """
        pass

    @abc.abstractmethod
    def parameters(self):
        """
        Returns all of the learnable layers of a model. Must be overridden by subclasses.
        
        Returns:
            list[BaseLayer]: List of layers in a model
        """
        pass

    @classmethod
    @abc.abstractmethod
    def build(cls, input_size, hidden_size, output_size):
        """
        Factory method to build the model with a specific architecture.
        Must be implemented by subclasses.
        """
        pass

    def save(self, path: str):
        """
        Save the model state_dict to the specified path.

        Args:
            path (str): File path to save the model weights.
        """
        pass

    def load(self, path: str, map_location=None):
        """
        Load the model state_dict from the specified path.

        Args:
            path (str): File path to load the model weights.
            map_location: Device mapping if loading on a different device
                          than it was trained on.
        """
        pass

