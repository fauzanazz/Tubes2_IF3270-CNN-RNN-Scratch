import tensorflow as tf
from layer import Layer

class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2)):
        self.type = "maxpool"
        self.pool_size = pool_size

    # def forward(self, input):
    #     # input shape: (batch_size, height, width, channels)
    #     self.input = input
    #     batch_size, height, width, channels = input.shape
    #     ph, pw = self.pool_size

    #     h_out = height // ph
    #     w_out = width // pw

    #     outputs = []

    #     for b in range(batch_size):
    #         sample = input[b]  # shape: (height, width, channels)
    #         pooled = []
    #         for i in range(h_out):
    #             row = []
    #             for j in range(w_out):
    #                 h_start = i * ph
    #                 w_start = j * pw
    #                 patch = sample[h_start:h_start+ph, w_start:w_start+pw, :]  # shape: (ph, pw, channels)
    #                 max_val = tf.reduce_max(patch, axis=[0, 1])  # shape: (channels,)
    #                 row.append(max_val)
    #             pooled.append(tf.stack(row))  # shape: (w_out, channels)
    #         outputs.append(tf.stack(pooled))  # shape: (h_out, w_out, channels)

    #     output_tensor = tf.stack(outputs)  # shape: (batch_size, h_out, w_out, channels)
    #     # print("MaxPooling2D Output shape:", output_tensor.shape)
    #     return output_tensor
    
    def forward(self, input):
        self.input = input
        # input shape: (batch_size, height, width, channels)
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]  # kernel size (batch, h, w, channels)
        strides = [1, self.pool_size[0], self.pool_size[1], 1]  # strides sesuai pool_size
        padding = 'VALID'  # atau 'SAME', sesuai kebutuhan

        output = tf.nn.max_pool2d(input, ksize=ksize, strides=strides, padding=padding)
        # print("MaxPooling2D Output shape:", output.shape)
        return output


class AvgPooling2D(Layer):
    def __init__(self, pool_size=(2, 2)):
        self.type = "avgpool"
        self.pool_size = pool_size

    def forward(self, input):
        # input shape: (batch_size, height, width, channels)
        self.input = input
        batch_size, height, width, channels = input.shape
        ph, pw = self.pool_size

        h_out = height // ph
        w_out = width // pw

        outputs = []

        for b in range(batch_size):
            sample = input[b]  # shape: (height, width, channels)
            pooled = []
            for i in range(h_out):
                row = []
                for j in range(w_out):
                    h_start = i * ph
                    w_start = j * pw
                    patch = sample[h_start:h_start+ph, w_start:w_start+pw, :]  # shape: (ph, pw, channels)
                    mean_val = tf.reduce_mean(patch, axis=[0, 1])  # shape: (channels,)
                    row.append(mean_val)
                pooled.append(tf.stack(row))  # shape: (w_out, channels)
            outputs.append(tf.stack(pooled))  # shape: (h_out, w_out, channels)

        output_tensor = tf.stack(outputs)  # shape: (batch_size, h_out, w_out, channels)
        # print("AvgPooling2D Output shape:", output_tensor.shape)
        return output_tensor
