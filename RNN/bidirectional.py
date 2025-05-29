from layer import Layer
import numpy as np

class Bidirectional(Layer):
    def __init__(self, rnn_layer, backward_layer=None):
        self.type = "bidirectional"
        self.forward_layer = rnn_layer
        
        # If backward_layer is not provided, create a copy of the forward layer
        if backward_layer is None:
            self.backward_layer = type(rnn_layer)(
                units=rnn_layer.units,
                return_sequences=rnn_layer.return_sequences,
                kernel=rnn_layer.kernel,
                recurrent_kernel=rnn_layer.recurrent_kernel,
                bias=rnn_layer.bias
            )
        else:
            self.backward_layer = backward_layer
        
        # The output units will be doubled because we concatenate forward and backward outputs
        self.units = self.forward_layer.units * 2
        self.return_sequences = self.forward_layer.return_sequences
    
    def forward(self, input):
        """
        Forward pass for bidirectional layer
        
        Parameters:
        - input: Numpy array of shape (batch_size, time_steps, features)
        
        Returns:
        - output: Numpy array 
                 If return_sequences=True: shape (batch_size, time_steps, units*2)
                 If return_sequences=False: shape (batch_size, units*2)
        """
        self.input = input
        
        # Forward direction - use the input as is
        forward_output = self.forward_layer.forward(input)
        
        # Backward direction - reverse the time dimension (axis=1)
        reversed_input = np.flip(input, axis=1).copy()
        backward_output = self.backward_layer.forward(reversed_input)
        
        # If we're returning sequences, we need to flip the backward output back
        if self.return_sequences:
            backward_output = np.flip(backward_output, axis=1).copy()
        
        # Concatenate along the last dimension
        output = np.concatenate([forward_output, backward_output], axis=-1)
        
        return output