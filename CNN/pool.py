import numpy as np
from layer import Layer

class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2)):
        self.type = "maxpool"
        # pool_size bisa tuple (ph, pw) atau int
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

    def forward(self, input_tensor):
        # input shape: (batch_size, height, width, channels)
        self.input = input_tensor # Simpan input asli jika diperlukan di tempat lain

        batch_size, height, width, num_channels = input_tensor.shape
        ph, pw = self.pool_size

        # Hitung dimensi output
        out_h = height // ph
        out_w = width // pw
        
        # Inisialisasi output array
        output_array = np.zeros((batch_size, out_h, out_w, num_channels), dtype=input_tensor.dtype)

        for b_idx in range(batch_size):
            sample = input_tensor[b_idx]  # shape: (height, width, channels)
            for i_idx in range(out_h):
                for j_idx in range(out_w):
                    h_start = i_idx * ph
                    w_start = j_idx * pw
                    patch = sample[h_start:h_start+ph, w_start:w_start+pw, :]  # shape: (ph, pw, channels)
                    # Lakukan max pooling per channel
                    output_array[b_idx, i_idx, j_idx, :] = np.max(patch, axis=(0, 1))

        return output_array


class AvgPooling2D(Layer):
    def __init__(self, pool_size=(2, 2)):
        self.type = "avgpool"
        # pool_size bisa tuple (ph, pw) atau int
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

    def forward(self, input_tensor):
        # input shape: (batch_size, height, width, channels)
        self.input = input_tensor # Simpan input asli jika diperlukan di tempat lain

        batch_size, height, width, num_channels = input_tensor.shape
        ph, pw = self.pool_size

        # Hitung dimensi output
        out_h = height // ph
        out_w = width // pw
        
        # Inisialisasi output array
        output_array = np.zeros((batch_size, out_h, out_w, num_channels), dtype=input_tensor.dtype)

        for b_idx in range(batch_size):
            sample = input_tensor[b_idx]  # shape: (height, width, channels)
            for i_idx in range(out_h):
                for j_idx in range(out_w):
                    h_start = i_idx * ph
                    w_start = j_idx * pw
                    patch = sample[h_start:h_start+ph, w_start:w_start+pw, :]  # shape: (ph, pw, channels)
                    # Lakukan average pooling per channel
                    output_array[b_idx, i_idx, j_idx, :] = np.mean(patch, axis=(0, 1))

        return output_array
