import tensorflow as tf
from layer import Layer

class MaxPooling2D(Layer):
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input  # input: (C, H, W)
        c, h, w = input.shape
        h_out = h // self.pool_size
        w_out = w // self.pool_size

        output = tf.TensorArray(dtype=tf.float32, size=c)
        self.max_indices = {}

        for ch in range(c):
            feature_map = tf.zeros((h_out, w_out), dtype=tf.float32)
            for i in range(h_out):
                for j in range(w_out):
                    h_start = i * self.pool_size
                    w_start = j * self.pool_size
                    patch = input[ch, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                    max_val = tf.reduce_max(patch)
                    feature_map[i, j] = max_val

                    # Store the indices for backpropagation
                    max_pos = tf.unravel_index(tf.argmax(patch), patch.shape)
                    self.max_indices[(ch, i, j)] = (h_start + max_pos[0], w_start + max_pos[1])

            output = output.write(ch, feature_map)

        self.output = tf.transpose(output.stack(), perm=[0, 1, 2])  # (C, H_out, W_out)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = tf.zeros_like(self.input)

        for (ch, i, j), (h_idx, w_idx) in self.max_indices.items():
            input_gradient = input_gradient.numpy()
            input_gradient[ch, h_idx, w_idx] = output_gradient[ch, i, j]
            input_gradient = tf.convert_to_tensor(input_gradient)

        return input_gradient


class AvgPooling2D(Layer):
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input  # input: (C, H, W)
        c, h, w = input.shape
        h_out = h // self.pool_size
        w_out = w // self.pool_size

        output = tf.TensorArray(dtype=tf.float32, size=c)

        for ch in range(c):
            feature_map = tf.zeros((h_out, w_out), dtype=tf.float32)
            for i in range(h_out):
                for j in range(w_out):
                    h_start = i * self.pool_size
                    w_start = j * self.pool_size
                    patch = input[ch, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                    feature_map[i, j] = tf.reduce_mean(patch)

            output = output.write(ch, feature_map)

        self.output = tf.transpose(output.stack(), perm=[0, 1, 2])  # (C, H_out, W_out)
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = tf.zeros_like(self.input)
        c, h, w = self.input.shape

        for ch in range(c):
            for i in range(h // self.pool_size):
                for j in range(w // self.pool_size):
                    grad = output_gradient[ch, i, j] / (self.pool_size * self.pool_size)
                    h_start = i * self.pool_size
                    w_start = j * self.pool_size
                    for m in range(self.pool_size):
                        for n in range(self.pool_size):
                            input_gradient[ch, h_start + m, w_start + n] += grad

        return input_gradient
