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
        self.input = inputs
        
        # Forward direction
        forward_output = self.forward_layer.forward(inputs)
        
        # Backward direction - reverse the time dimension (axis=1)
        reversed_input = np.flip(inputs, axis=1).copy()
        backward_output = self.backward_layer.forward(reversed_input)
        
        # If we're returning sequences, we need to flip the backward output back
        if getattr(self.forward_layer, 'return_sequences', False):
            backward_output = np.flip(backward_output, axis=1)
            # Concatenate along feature dimension
            return np.concatenate([forward_output, backward_output], axis=-1)
        else:
            # For single output, just concatenate the final states
            return np.concatenate([forward_output, backward_output], axis=-1)