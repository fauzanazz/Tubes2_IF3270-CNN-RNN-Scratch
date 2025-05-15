import tensorflow as tf
from layer import Layer

class Convolutional(Layer):
    def __init__(self, filters, kernel_size, activation=None, input_shape=None, kernels=None, biases=None):
        self.type = "convolution"
        self.filters = filters
        # kernel_size bisa tuple (kh, kw) atau int
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        self.activation = activation

        self.kernels = None
        self.biases = None

        if input_shape is not None:
            # input_shape NHWC: (height, width, channels)
            self.input_height, self.input_width, self.input_depth = input_shape
        else:
            self.input_depth = None  # Akan diinisialisasi saat forward

        # Jika bobot diberikan langsung saat konstruktor
        if kernels is not None and biases is not None:
            self.kernels = tf.Variable(kernels, trainable=True)
            self.biases = tf.Variable(biases, trainable=True)

    def forward(self, input):
        # input shape: (batch_size, height, width, channels)
        # print("Input shape:", input.shape)
        self.input = input
        batch_size, input_height, input_width, input_depth = input.shape

        # Initialize kernels dan biases jika belum ada
        if self.kernels is None:
            limit = tf.sqrt(6.0 / (input_depth * self.kernel_size[0] * self.kernel_size[1] + self.filters))
            # kernel shape: (kh, kw, input_depth, filters)
            self.kernels = tf.Variable(tf.random.uniform(
                shape=(self.kernel_size[0], self.kernel_size[1], input_depth, self.filters),
                minval=-limit, maxval=limit
            ), trainable=True)
            self.biases = tf.Variable(tf.zeros((self.filters,)), trainable=True)

        kh, kw = self.kernel_size
        output_height = input_height - kh + 1
        output_width = input_width - kw + 1
        # print("Output shape (H, W):", output_height, output_width)

        # Tempat simpan hasil konvolusi tiap batch
        batch_outputs = tf.TensorArray(dtype=tf.float32, size=batch_size)

        for b in tf.range(batch_size):
            input_sample = input[b]  # shape: (input_height, input_width, input_depth)
            feature_maps = tf.TensorArray(dtype=tf.float32, size=self.filters)

            for f in tf.range(self.filters):
                # buat feature map untuk filter f
                feature_map = tf.zeros((output_height, output_width), dtype=tf.float32)

                for i in range(output_height):
                    for j in range(output_width):
                        # ambil patch (kh, kw, c)
                        patch = input_sample[i:i+kh, j:j+kw, :]  # shape: (kh, kw, input_depth)
                        kernel = self.kernels[:, :, :, f]        # shape: (kh, kw, input_depth)
                        conv_value = tf.reduce_sum(patch * kernel)
                        feature_map = tf.tensor_scatter_nd_update(feature_map, [[i, j]], [conv_value])

                # tambahkan bias
                feature_map += self.biases[f]
                feature_maps = feature_maps.write(f, feature_map)

            sample_output = feature_maps.stack()  # shape: (filters, output_height, output_width)
            # transpose ke NHW format jadi H,W,C supaya konsisten (opsional)
            sample_output = tf.transpose(sample_output, perm=[1, 2, 0])  # (H, W, filters)

            if self.activation is not None:
                sample_output = self.activation(sample_output)

            batch_outputs = batch_outputs.write(b, sample_output)

        self.output = batch_outputs.stack()  # shape: (batch_size, H, W, filters)
        print("Output shape after conv:", self.output.shape)
        return self.output
