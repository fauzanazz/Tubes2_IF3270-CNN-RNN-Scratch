from numba import njit
import numpy as np


class LossFunction:
    @staticmethod
    @njit(cache=True, fastmath=True)
    def mean_squared_error(y_pred, y_true, epsilon=1e-7):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def mean_squared_error_derivative(y_pred, y_true, epsilon=1e-7):
        return 2 * (y_pred - y_true)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def binary_cross_entropy(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    @njit(cache=True, fastmath=True)
    def binary_cross_entropy_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -((y_true / y_pred) - ((1 - y_true) / (1 - y_pred)))

    @staticmethod
    @njit(cache=True, fastmath=True)
    def categorical_cross_entropy(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        if y_pred.ndim == 1:
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    @njit(cache=True, fastmath=True)
    def categorical_cross_entropy_derivative(y_pred, y_true, epsilon=1e-7):
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return y_pred - y_true