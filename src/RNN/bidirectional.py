import numpy as np

class Bidirectional:
    def __init__(self, forward_layer, backward_layer):
        self.forward_layer = forward_layer
        self.backward_layer = backward_layer
        self.input = None
    
    def forward(self, inputs):
        self.input = inputs
        
        forward_output = self.forward_layer.forward(inputs)
        reversed_input = np.flip(inputs, axis=1)
        backward_output = self.backward_layer.forward(reversed_input)
        
        if getattr(self.forward_layer, 'return_sequences', False):
            backward_output = np.flip(backward_output, axis=1)
            output = np.concatenate([forward_output, backward_output], axis=-1)
        else:
            output = np.concatenate([forward_output, backward_output], axis=-1)
        return output