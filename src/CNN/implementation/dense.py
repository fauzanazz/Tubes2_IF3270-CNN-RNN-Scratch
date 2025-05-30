import numpy as np
from .layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size, activation=None, weights=None, biases=None):
        self.type = "dense"
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32)
        else:
            # xavier init
            limit = np.sqrt(6.0 / (input_size + output_size))
            self.weights = np.random.uniform(
                low=-limit, high=limit,
                size=(input_size, output_size)
            ).astype(np.float32)

        if biases is not None:
            self.bias = np.array(biases, dtype=np.float32)
        else:
            self.bias = np.zeros((output_size,), dtype=np.float32)  # shape: (output_size,)

        self.activation = activation

    def forward(self, input):
        self.input = input  # shape: (batch_size, input_size)

        z = np.dot(input, self.weights)

        output = z + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output
