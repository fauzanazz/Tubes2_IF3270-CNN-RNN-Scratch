import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """Base forward method to be implemented by subclasses"""
        pass
    
    def __str__(self):
        """String representation of the layer"""
        return self.__class__.__name__

