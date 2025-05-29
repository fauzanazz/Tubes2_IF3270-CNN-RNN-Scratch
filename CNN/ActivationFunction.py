import numpy as np

class ActivationFunction:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        # Stabilisasi numerik dengan mengurangi nilai maksimum dari x
        # sebelum eksponensiasi untuk menghindari overflow.
        # axis=-1 mengasumsikan x adalah (batch_size, num_features)
        # atau (num_features,) dan softmax dihitung pada dimensi terakhir.
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)

    @staticmethod
    def prelu(x, alpha):
        # Dalam NumPy, alpha bisa berupa skalar atau array yang dapat di-broadcast
        # ke bentuk x. Jika alpha adalah parameter per channel, pastikan
        # bentuknya sesuai untuk broadcasting (misalnya, (1, num_channels)
        # jika x adalah (batch_size, num_channels)).
        # Untuk kasus umum, np.maximum akan menangani broadcasting jika alpha
        # adalah skalar atau array yang kompatibel.
        return np.maximum(x, alpha * x)
