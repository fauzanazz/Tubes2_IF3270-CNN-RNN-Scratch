import numpy as np
from .layer import Layer

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
            self.input_depth = None  # init pas forward

        # klo bobot diberikan langsung saat konstruktor
        if kernels is not None and biases is not None:
            self.kernels = np.array(kernels, dtype=np.float32)
            self.biases = np.array(biases, dtype=np.float32)

    def forward(self, input):
        # input shape: (batch_size, height, width, channels)
        self.input = input
        batch_size, input_height, input_width, input_depth_from_input = input.shape
        # print(f"input size = {input.shape}")
        # Init kernels dan bias klo belum ada
        if self.kernels is None:

            # Cek init depth
            current_input_depth = self.input_depth if self.input_depth is not None else input_depth_from_input
            if current_input_depth is None:
                raise ValueError("Input depth tidak diketahui untuk inisialisasi kernel.")
            self.input_depth = current_input_depth


            limit_denominator = float(self.input_depth * self.kernel_size[0] * self.kernel_size[1] + self.filters)
            limit = np.sqrt(6.0 / limit_denominator)

            # kernel shape: (kh, kw, input_depth, filters)
            self.kernels = np.random.uniform(
                low=-limit, high=limit,
                size=(self.kernel_size[0], self.kernel_size[1], self.input_depth, self.filters)
            ).astype(np.float32)
            self.biases = np.zeros((self.filters,), dtype=np.float32)

        kh, kw = self.kernel_size
        output_height = input_height - kh + 1
        output_width = input_width - kw + 1


        self.output = np.zeros((batch_size, output_height, output_width, self.filters), dtype=np.float32)

        for b in range(batch_size):
            input_sample = input[b]  # shape: (input_height, input_width, input_depth)
            # tes_feature_maps = np.zeros((output_height, output_width, self.filters), dtype=np.float32)

            for f in range(self.filters):
                feature_map_f = np.zeros((output_height, output_width), dtype=np.float32)
                current_kernel = self.kernels[:, :, :, f]

                for i in range(output_height):
                    for j in range(output_width):
                        patch = input_sample[i:i+kh, j:j+kw, :]  # shape: (kh, kw, input_depth)
                        # if patch.shape != (3,3,3): print(f"patch shape = {patch.shape}")
                        conv_value = np.sum(patch * current_kernel)
                        feature_map_f[i, j] = conv_value

                feature_map_f += self.biases[f]
                self.output[b, :, :, f] = feature_map_f

            # tes_output = tes_feature_maps

            if self.activation is not None:
                self.output[b, :, :, :] = self.activation(self.output[b, :, :, :])

        return self.output
