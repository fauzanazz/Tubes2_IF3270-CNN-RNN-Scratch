from layer import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, rate):
        self.type = "dropout"
        self.rate = rate
        self.mask = None
    
    def forward(self, input, training=False):
        self.input = input
        
        if training and self.rate > 0:
            keep_prob = 1.0 - self.rate
            self.mask = np.random.binomial(1, keep_prob, size=input.shape) / keep_prob
            return input * self.mask
        else:
            return input
    
    def forward_inference(self, input):
        return self.forward(input, training=False)