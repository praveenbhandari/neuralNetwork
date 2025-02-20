class SGDOptimizer:
    """
    A very simple Stochastic Gradient Descent optimizer.
    """
    def __init__(self, model_layers, tau_lr=0.01):
        """
        tau_lr : learning rate for tau
        """
        self.layers = model_layers
        self.tau_lr = tau_lr

    def step(self):
        """
        Update parameters in each layer given the stored gradients.
        """
        for layer in self.layers:
            # Update taus
            if hasattr(layer, 'tau') and hasattr(layer, 'grad_tau'):
                layer.tau -= self.tau_lr * layer.grad_tau
