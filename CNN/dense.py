import tensorflow as tf
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size, activation=None, weights=None, biases=None):
        self.type = "dense"
        if weights is not None:
            self.weights = tf.Variable(weights, trainable=True)
        else:
            # Xavier initialization
            limit = tf.sqrt(6.0 / (input_size + output_size))
            self.weights = tf.Variable(
                tf.random.uniform((input_size, output_size), -limit, limit), trainable=True
            )

        if biases is not None:
            self.bias = tf.Variable(biases, trainable=True)
        else:
            self.bias = tf.Variable(tf.zeros((output_size,)), trainable=True)  # shape: (output_size,)

        self.activation = activation

    def forward(self, input):
        self.input = input  # shape: (batch_size, input_size)

        # Tambahkan bias secara broadcast: (batch_size, output_size)
        output = tf.matmul(input, self.weights) + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output
