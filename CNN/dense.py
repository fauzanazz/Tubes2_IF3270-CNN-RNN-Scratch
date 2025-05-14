import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Xavier Initialization for weights
        # self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / (input_size + output_size))
        self.weights = np.zeros((output_size, input_size))
        self.bias = np.zeros((output_size, 1))  # Bias biasanya diinisialisasi dengan 0

    def forward(self, input):
        self.input = input
        output = np.dot(self.weights, self.input) + self.bias
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            print(f"Warning: NaN or Inf detected in forward pass.")
        return output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

