import numpy as np
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
            self.kernels = np.array(kernels, dtype=np.float32)
            self.biases = np.array(biases, dtype=np.float32)

    def forward(self, input):
        # input shape: (batch_size, height, width, channels)
        self.input = input
        batch_size, input_height, input_width, input_depth_from_input = input.shape

        # Initialize kernels dan biases jika belum ada
        if self.kernels is None:
            # Gunakan input_depth yang ditentukan saat init atau dari input pertama
            current_input_depth = self.input_depth if self.input_depth is not None else input_depth_from_input
            if current_input_depth is None: # Jika self.input_depth juga None
                raise ValueError("Input depth tidak diketahui untuk inisialisasi kernel.")
            self.input_depth = current_input_depth # Pastikan self.input_depth di-set

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

        # Tempat simpan hasil konvolusi tiap batch
        # Inisialisasi output array dengan bentuk yang benar
        self.output = np.zeros((batch_size, output_height, output_width, self.filters), dtype=np.float32)

        for b in range(batch_size):
            input_sample = input[b]  # shape: (input_height, input_width, input_depth)
            # feature_maps_for_sample = np.zeros((output_height, output_width, self.filters), dtype=np.float32)

            for f in range(self.filters):
                # buat feature map untuk filter f
                feature_map_f = np.zeros((output_height, output_width), dtype=np.float32)
                current_kernel = self.kernels[:, :, :, f]

                for i in range(output_height):
                    for j in range(output_width):
                        # ambil patch (kh, kw, c)
                        patch = input_sample[i:i+kh, j:j+kw, :]  # shape: (kh, kw, input_depth)
                        conv_value = np.sum(patch * current_kernel)
                        feature_map_f[i, j] = conv_value

                # tambahkan bias
                feature_map_f += self.biases[f]
                self.output[b, :, :, f] = feature_map_f

            # Tidak perlu transpose lagi karena kita mengisi self.output[b, :, :, f]
            # sample_output = feature_maps_for_sample

            if self.activation is not None:
                # Terapkan aktivasi ke seluruh output sample saat ini
                # Pastikan aktivasi diterapkan setelah semua filter untuk batch item tersebut dihitung dan bias ditambahkan
                self.output[b, :, :, :] = self.activation(self.output[b, :, :, :])

        return self.output
