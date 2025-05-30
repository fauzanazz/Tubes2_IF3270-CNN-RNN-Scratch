from .layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self):
        self.type = "flatten"

    def forward(self, input):
        self.input_shape = input.shape  # (batch_size, height, width, channels)
        batch_size = input.shape[0]
        # Reshape ke (batch_size, height*width*channels)
        self.output = np.reshape(input, (batch_size, -1))
        return self.output
