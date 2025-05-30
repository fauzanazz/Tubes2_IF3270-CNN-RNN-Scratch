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
        # Add input normalization
        inputs = (inputs - np.mean(inputs)) / (np.std(inputs) + 1e-8)
        
        self.input = inputs
        MAX_VALUE = 1e9
        EPSILON = 1e-7
        
        # Add numerical stability to dot product
        linear_output = np.dot(inputs, self.weights)
        linear_output = np.clip(linear_output + self.bias, -MAX_VALUE, MAX_VALUE)
        
        if self.activation == "softmax":
            # Match Keras softmax implementation more closely
            shifted_input = linear_output - np.max(linear_output, axis=1, keepdims=True)
            exp_vals = np.exp(shifted_input)
            sum_vals = np.sum(exp_vals, axis=1, keepdims=True)
            self.output = np.divide(exp_vals, sum_vals + EPSILON, 
                                out=np.zeros_like(exp_vals), 
                                where=sum_vals != 0)
            
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