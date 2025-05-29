import numpy as np
from numba import njit

class ActivationFunction:
    @staticmethod
    @njit(cache=True, fastmath=True)
    def linear(x):
        return x

    @staticmethod
    @njit(cache=True, fastmath=True)
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    @staticmethod
    @njit(cache=True, fastmath=True)
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softmax(x):
        x = np.clip(x, -500, 500)
        if x.ndim == 1:
            x_shifted = x - np.max(x)
            exp_x = np.exp(x_shifted)
            return exp_x / (np.sum(exp_x) + 1e-10)
        else:
            x_shifted = x - np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def leaky_relu(x, alpha=0.01):
        return np.maximum(x, alpha * x)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def prelu(x, alpha):
        return np.maximum(x, alpha * x)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def derivative_linear(x):
        return np.ones_like(x)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def derivative_relu(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def derivative_sigmoid(x):
        sig = ActivationFunction.sigmoid(x)
        return sig * (1 - sig)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def derivative_tanh(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def derivative_softmax(x):
        s = np.exp(x - np.max(x, axis=-1, keepdims=True))
        s = s / np.sum(s, axis=-1, keepdims=True)
        
        if len(x.shape) == 1:
            n = x.shape[0]
            jacobian = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        jacobian[i, j] = s[i] * (1 - s[i])
                    else:
                        jacobian[i, j] = -s[i] * s[j]
            return jacobian
        else:
            batch_size, n = x.shape
            jacobian = np.zeros((batch_size, n, n))
            
            for b in range(batch_size):
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            jacobian[b, i, j] = s[b, i] * (1 - s[b, i])
                        else:
                            jacobian[b, i, j] = -s[b, i] * s[b, j]
            return jacobian

    @staticmethod
    @njit(cache=True, fastmath=True)
    def derivative_leaky_relu(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def derivative_prelu(x, alpha):
        return np.where(x > 0, 1.0, alpha)