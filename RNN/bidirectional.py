import numpy as np

class Bidirectional:
    def __init__(self, forward_layer, backward_layer):
        self.forward_layer = forward_layer
        self.backward_layer = backward_layer
        self.input = None
    
    def forward(self, inputs):
        """
        Forward pass for Bidirectional RNN
        Args:
            inputs: Input array of shape (batch_size, timesteps, features)
        Returns:
            Concatenated output from forward and backward layers
        """
        # Add input normalization
        inputs = (inputs - np.mean(inputs)) / (np.std(inputs) + 1e-8)
        
        self.input = inputs
        
        forward_output = self.forward_layer.forward(inputs)
        
        # Ensure proper copy of reversed input
        reversed_input = np.flip(inputs, axis=1).copy()
        backward_output = self.backward_layer.forward(reversed_input)
        
        # Add numerical stability check
        if np.any(np.isnan(forward_output)) or np.any(np.isnan(backward_output)):
            raise ValueError("NaN values detected in Bidirectional output")

        # Ensure consistent scaling between forward and backward passes
        forward_output = forward_output / 2.0
        backward_output = backward_output / 2.0
        
        if getattr(self.forward_layer, 'return_sequences', False):
            backward_output = np.flip(backward_output, axis=1)
            return np.concatenate([forward_output, backward_output], axis=-1)
        else:
            return np.concatenate([forward_output, backward_output], axis=-1)