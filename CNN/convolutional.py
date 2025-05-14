import tensorflow as tf
from layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        self.input_depth, self.input_height, self.input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth

        # weights: (depth, input_depth, kernel_size, kernel_size)
        limit = tf.sqrt(6.0 / (self.input_depth * kernel_size * kernel_size + depth))
        self.kernels = tf.Variable(tf.random.uniform(
            shape=(depth, self.input_depth, kernel_size, kernel_size), minval=-limit, maxval=limit
        ), trainable=True)
        self.biases = tf.Variable(tf.zeros((depth,)), trainable=True)

    def forward(self, input):
        # Input shape: (input_depth, height, width)
        self.input = input
        input_depth, input_height, input_width = input.shape

        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1

        output = tf.TensorArray(dtype=tf.float32, size=self.depth)

        for d in tf.range(self.depth):
            feature_map = tf.zeros((output_height, output_width), dtype=tf.float32)
            for c in tf.range(self.input_depth):
                kernel = self.kernels[d, c]
                channel = input[c]

                # lakukan konvolusi manual
                conv = []
                for i in range(output_height):
                    row = []
                    for j in range(output_width):
                        patch = channel[i:i+self.kernel_size, j:j+self.kernel_size]
                        row.append(tf.reduce_sum(patch * kernel))
                    conv.append(tf.stack(row))
                conv = tf.stack(conv)
                feature_map += conv
            feature_map += self.biases[d]
            output = output.write(d, feature_map)

        self.output = tf.transpose(output.stack(), perm=[0, 1, 2])  # (depth, h, w)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # output_gradient: (depth, out_h, out_w)
        input_gradient = tf.zeros_like(self.input)
        kernel_gradient = tf.zeros_like(self.kernels)
        bias_gradient = tf.reduce_sum(output_gradient, axis=[1, 2])  # sum over height & width

        for d in range(self.depth):
            for c in range(self.input_depth):
                for i in range(self.input_height - self.kernel_size + 1):
                    for j in range(self.input_width - self.kernel_size + 1):
                        patch = self.input[c, i:i+self.kernel_size, j:j+self.kernel_size]
                        kernel_gradient = kernel_gradient.numpy()
                        kernel_gradient[d, c] += output_gradient[d, i, j] * patch.numpy()
                        kernel_gradient = tf.convert_to_tensor(kernel_gradient)

                        input_gradient = input_gradient.numpy()
                        input_gradient[c, i:i+self.kernel_size, j:j+self.kernel_size] += output_gradient[d, i, j].numpy() * self.kernels[d, c].numpy()
                        input_gradient = tf.convert_to_tensor(input_gradient)

        self.kernels.assign_sub(learning_rate * kernel_gradient)
        self.biases.assign_sub(learning_rate * bias_gradient)

        return input_gradient
