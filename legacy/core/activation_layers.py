from .base_layer import Layer

# inherit from base class Layer: Activation Layer code
class ActivationLayer(Layer):
    def __init__(self, activation_func):
        super().__init__()
        self.activation_func = activation_func

    # returns the activated input: y(i) = ReLU (Zi)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation_func(self.input)
        print('output for the layer after ReLu (y): ', self.output)
        return self.output

    # Not returning anything for the modified NN
    def backward_propagation(self, is_last_layer, layer_target_output, prev_tau_gradient, bias_learning_rate, tau_learning_rate, epochs,Change_bias,Change_tau):
        _ = self.activation_func(self.input)
        return prev_tau_gradient
