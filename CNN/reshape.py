import tensorflow as tf
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input = input
        return tf.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return tf.reshape(output_gradient, self.input_shape)
