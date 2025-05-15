import tensorflow as tf
from layer import Layer

class Flatten(Layer):
    def __init__(self):
        self.type = "flatten"

    def forward(self, input):
        self.input_shape = input.shape  # (batch_size, height, width, channels)
        batch_size = input.shape[0]
        # Reshape ke (batch_size, height*width*channels)
        self.output = tf.reshape(input, (batch_size, -1))
        return self.output
