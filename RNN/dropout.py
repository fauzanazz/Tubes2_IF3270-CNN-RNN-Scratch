from layer import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, rate):
        self.type = "dropout"
        self.rate = rate
        self.mask = None
    
    def forward(self, input, training=False):
        """
        Forward pass for dropout layer
        
        Parameters:
        - input: Numpy array
        - training: Boolean indicating if in training mode
        
        Returns:
        - output: Numpy array with same shape as input
        """
        self.input = input
        
        if training:
            # Generate a mask with probability (1-rate) for keeping units
            self.mask = np.random.binomial(1, 1-self.rate, size=input.shape) / (1-self.rate)
            return input * self.mask
        else:
            # During inference, we don't apply dropout
            return input
    
    # During training with Keras, we would use this version, but for inference with our from-scratch
    # implementation, we just call forward with training=False
    def forward_inference(self, input):
        """Forward pass for inference (no dropout)"""
        return self.forward(input, training=False)