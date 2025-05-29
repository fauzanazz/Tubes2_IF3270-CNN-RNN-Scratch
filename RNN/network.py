import numpy as np

class Network:
    def __init__(self, layers):
        """
        Initialize the neural network with a list of layers
        
        Parameters:
        - layers: List of Layer objects that will be used in the network
        """
        self.layers = layers
    
    def forward(self, inputs):
        """
        Forward pass through the network
        
        Parameters:
        - inputs: Numpy array of inputs
        
        Returns:
        - Output of the network after forward pass
        """
        outputs = inputs
        
        # Pass through each layer in sequence
        for layer in self.layers:
            # Handle special case for dropout layers
            if hasattr(layer, "type") and layer.type == "dropout":
                outputs = layer.forward(outputs, training=False)
            else:
                outputs = layer.forward(outputs)
        
        return outputs

    def predict(self, inputs):
        """
        Make predictions using the network
        
        Parameters:
        - inputs: Numpy array of inputs
        
        Returns:
        - Predictions as numpy array
        """
        return self.forward(inputs)


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    yt = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    return yt