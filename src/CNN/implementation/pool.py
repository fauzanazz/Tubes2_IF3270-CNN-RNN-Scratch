import numpy as np
from .layer import Layer

class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2)):
        self.type = "maxpool"
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

    def forward(self, input_tensor):
        # print(f"input maxpool shape: {input_tensor.shape}")
        # input shape: (batch_size, height, width, channels)
        self.input = input_tensor

        batch_size, height, width, num_channels = input_tensor.shape
        ph, pw = self.pool_size

        # dimensi output
        out_h = height // ph
        out_w = width // pw
        
        output_array = np.zeros((batch_size, out_h, out_w, num_channels), dtype=input_tensor.dtype)

        for b_idx in range(batch_size):
            sample = input_tensor[b_idx]  # shape: (height, width, channels)
            for i_idx in range(out_h):
                for j_idx in range(out_w):
                    h_start = i_idx * ph
                    w_start = j_idx * pw
                    patch = sample[h_start:h_start+ph, w_start:w_start+pw, :]  # shape: (ph, pw, channels)
                    output_array[b_idx, i_idx, j_idx, :] = np.max(patch, axis=(0, 1))
        # print(f"output maxpool shape: {output_array.shape}")
        return output_array


class AvgPooling2D(Layer):
    def __init__(self, pool_size=(2, 2)):
        self.type = "avgpool"
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

    def forward(self, input_tensor):
        # print(f"input avgpool shape: {input_tensor.shape}")
        # input shape: (batch_size, height, width, channels)
        self.input = input_tensor

        batch_size, height, width, num_channels = input_tensor.shape
        ph, pw = self.pool_size

        # dimensi output
        out_h = height // ph
        out_w = width // pw
        
        output_array = np.zeros((batch_size, out_h, out_w, num_channels), dtype=input_tensor.dtype)

        for b_idx in range(batch_size):
            sample = input_tensor[b_idx]  # shape: (height, width, channels)
            for i_idx in range(out_h):
                for j_idx in range(out_w):
                    h_start = i_idx * ph
                    w_start = j_idx * pw
                    patch = sample[h_start:h_start+ph, w_start:w_start+pw, :]  # shape: (ph, pw, channels)
                    output_array[b_idx, i_idx, j_idx, :] = np.mean(patch, axis=(0, 1))

        # print(f"output avgpool shape: {output_array.shape}")
        return output_array
