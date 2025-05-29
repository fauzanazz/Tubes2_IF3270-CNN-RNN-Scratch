import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, weights, bias, activation="softmax"):
        """
        Initialize a fully-connected (dense) layer.
        
        Parameters:
        - weights: Pre-trained weights matrix with shape (input_features, output_features)
        - bias: Pre-trained bias vector with shape (output_features,)
        - activation: Activation function to use ('softmax', 'relu', 'tanh', or None)
        """
        super().__init__()
        self.weights = weights
        self.bias = bias
        self.activation = activation
        
    def forward(self, inputs):
        """
        Forward pass through the dense layer.
        
        Parameters:
        - inputs: Input tensor with shape (batch_size, input_features)
        
        Returns:
        - Output tensor with shape (batch_size, output_features)
        """
        # Linear transformation: y = x*W + b
        self.input = inputs
        linear_output = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        if self.activation == "softmax":
            # Softmax activation for classification
            # Subtract max for numerical stability
            exp_vals = np.exp(linear_output - np.max(linear_output, axis=1, keepdims=True))
            self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
        elif self.activation == "relu":
            # ReLU activation
            self.output = np.maximum(0, linear_output)
        
        elif self.activation == "tanh":
            # Tanh activation
            self.output = np.tanh(linear_output)
        
        elif self.activation is None:
            # Linear activation (no activation)
            self.output = linear_output
            
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
            
        return self.output
    
    def __str__(self):
        """String representation of the layer"""
        return f"Dense(in={self.weights.shape[0]}, out={self.weights.shape[1]}, activation={self.activation})"