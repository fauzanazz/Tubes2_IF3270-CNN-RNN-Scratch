import tensorflow as tf
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Inisialisasi Xavier (Glorot)
        limit = tf.sqrt(6.0 / (input_size + output_size))
        self.weights = tf.Variable(tf.random.uniform((input_size, output_size), -limit, limit), trainable=True)
        self.bias = tf.Variable(tf.zeros((1, output_size)), trainable=True)

    def forward(self, input):
        self.input = input  # Bentuk: (batch_size, input_size)
        output = tf.matmul(input, self.weights) + self.bias  # output: (batch_size, output_size)
        if tf.math.reduce_any(tf.math.is_nan(output)) or tf.math.reduce_any(tf.math.is_inf(output)):
            tf.print("Warning: NaN or Inf detected in forward pass.")
        return output

    def backward(self, output_gradient, learning_rate):
        # output_gradient: (batch_size, output_size)
        batch_size = tf.cast(tf.shape(self.input)[0], tf.float32)

        # Gradien parameter
        grad_weights = tf.matmul(tf.transpose(self.input), output_gradient) / batch_size
        grad_bias = tf.reduce_mean(output_gradient, axis=0, keepdims=True)

        # Gradien untuk input layer sebelumnya
        input_gradient = tf.matmul(output_gradient, tf.transpose(self.weights))

        # Update parameter
        self.weights.assign_sub(learning_rate * grad_weights)
        self.bias.assign_sub(learning_rate * grad_bias)

        return input_gradient
