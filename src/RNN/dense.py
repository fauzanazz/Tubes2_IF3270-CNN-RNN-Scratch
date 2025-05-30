import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, weights, bias, activation="softmax"):
        super().__init__()
        self.weights = weights
        self.bias = bias
        self.activation = activation
        
    def forward(self, inputs):
        self.input = inputs
        
        # Standard dense layer computation: output = inputs @ weights + bias
        linear_output = np.dot(inputs, self.weights) + self.bias
        
        if self.activation == "softmax":
            shifted = linear_output - np.max(linear_output, axis=-1, keepdims=True)
            exp_vals = np.exp(shifted)
            self.output = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        elif self.activation == "relu":
            self.output = np.maximum(0, linear_output)
        elif self.activation == "tanh":
            self.output = np.tanh(linear_output)
        elif self.activation is None or self.activation == "linear":
            self.output = linear_output
            
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        return self.output
    
    def __str__(self):
        """String representation of the layer"""
        return f"Dense(in={self.weights.shape[0]}, out={self.weights.shape[1]}, activation={self.activation})"