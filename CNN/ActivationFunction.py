import tensorflow as tf

class ActivationFunction:
    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def softmax(x):
        return tf.nn.softmax(x)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return tf.nn.leaky_relu(x, alpha=alpha)

    @staticmethod
    def prelu(x, alpha):
        # tf.nn.prelu butuh alpha parameter tensor yang sama shape dengan x channels
        # Jadi alpha harus tensor dengan shape cocok (biasanya per channel)
        return tf.maximum(x, alpha * x)
