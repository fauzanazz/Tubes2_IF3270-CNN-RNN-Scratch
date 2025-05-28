from enums import InitializerType
import numpy as np

class Initializer:
    """
    Initializer Class using NumPy operations.
    """
    @staticmethod
    def init_weights(weight_init: InitializerType, param_1, param_2, input_size, num_neurons):
        if weight_init == InitializerType.ZERO:
            return np.zeros((num_neurons, input_size))
        elif weight_init == InitializerType.RANDOM_DIST_UNIFORM:
            return (np.random.uniform(param_1, param_2, (num_neurons, input_size))) * np.sqrt(2.0 / input_size)
        elif weight_init == InitializerType.RANDOM_DIST_NORMAL:
            return (np.random.normal(param_1, param_2, (num_neurons, input_size))) * np.sqrt(2.0 / input_size)
        elif weight_init == InitializerType.XAVIER:
            limit = np.sqrt(6 / (input_size + num_neurons))
            return np.random.uniform(-limit, limit, (num_neurons, input_size))
        elif weight_init == InitializerType.HE:
            std_dev = np.sqrt(2.0 / input_size)
            return np.random.normal(0, std_dev, (num_neurons, input_size))

    @staticmethod
    def init_bias(bias_init: InitializerType, param_1, param_2, num_neurons):
        if bias_init == InitializerType.ZERO:
            return np.zeros(num_neurons)
        elif bias_init == InitializerType.RANDOM_DIST_UNIFORM:
            return np.random.uniform(param_1, param_2, num_neurons)
        elif bias_init == InitializerType.RANDOM_DIST_NORMAL:
            return np.random.normal(param_1, param_2, num_neurons)
        elif bias_init == InitializerType.XAVIER:
            return np.zeros(num_neurons)
        elif bias_init == InitializerType.HE:
            return np.zeros(num_neurons)